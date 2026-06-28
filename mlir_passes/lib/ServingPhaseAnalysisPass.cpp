#include "FusionPasses.h"
#include "QuantizationUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_SERVINGPHASEANALYSIS
#include "FusionPasses.h.inc"

// Default cost constants (formula-calibration estimates).
// Used when the target profile does not supply target.prefill_ms_per_token /
// target.decode_ms_per_token / target.pd_bandwidth_mb_per_ms module attrs.
// Truth boundary: estimated; not measured latency for any specific model or hardware.
static constexpr double kDefaultPrefillMsPerToken  = 0.08;
static constexpr double kDefaultDecodeMsPerToken   = 0.12;
static constexpr double kDefaultPdBandwidthMbPerMs = 24.0;
static constexpr double kQueueMsPerPromptToken = 0.02;
static constexpr double kPdSplitQueueFactor    = 0.4;
static constexpr double kPdCoordinationMs      = 2.0;

static constexpr int64_t kDefaultNumLayers    = 12;
static constexpr int64_t kDefaultHiddenSize   = 768;
static constexpr int64_t kDefaultPromptTokens = 128;
static constexpr int64_t kDefaultOutputTokens = 64;

struct ServingCostBreakdown {
  double compute_ms;
  double queue_ms;
  double kv_transfer_ms;
  double total_ms;
};

static ServingCostBreakdown computeColocatedCost(int64_t promptTokens,
                                                  int64_t outputTokens,
                                                  double prefillMsPerToken,
                                                  double decodeMsPerToken) {
  double compute = prefillMsPerToken * static_cast<double>(promptTokens)
                 + decodeMsPerToken  * static_cast<double>(outputTokens);
  double queue   = kQueueMsPerPromptToken * static_cast<double>(promptTokens);
  return {compute, queue, 0.0, compute + queue};
}

static ServingCostBreakdown computePdSplitCost(int64_t promptTokens,
                                               int64_t outputTokens,
                                               int64_t numLayers,
                                               int64_t hiddenSize,
                                               double prefillMsPerToken,
                                               double decodeMsPerToken,
                                               double pdBandwidthMbPerMs,
                                               double dtypeBytes) {
  double compute = prefillMsPerToken * static_cast<double>(promptTokens)
                 + decodeMsPerToken  * static_cast<double>(outputTokens);
  double queue   = kQueueMsPerPromptToken * static_cast<double>(promptTokens)
                 * kPdSplitQueueFactor + kPdCoordinationMs;
  double kvMb    = static_cast<double>(numLayers) * 2.0
                 * static_cast<double>(hiddenSize)
                 * dtypeBytes
                 * static_cast<double>(promptTokens + outputTokens)
                 / (1024.0 * 1024.0);
  double kvXfer  = kvMb / pdBandwidthMbPerMs;
  return {compute, queue, kvXfer, compute + queue + kvXfer};
}

struct ServingPhaseAnalysisPass
    : impl::ServingPhaseAnalysisBase<ServingPhaseAnalysisPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    // Read model attributes from the enclosing module.
    int64_t numLayers  = kDefaultNumLayers;
    int64_t hiddenSize = kDefaultHiddenSize;
    Operation *parent  = funcOp->getParentOp();
    if (parent) {
      if (auto a = parent->getAttrOfType<IntegerAttr>("llm.num_layers"))
        numLayers = a.getInt();
      if (auto a = parent->getAttrOfType<IntegerAttr>("llm.hidden_size"))
        hiddenSize = a.getInt();
    }

    // Read quantization plan dtype from QuantizationPlanningPass output.
    // Falls back to fp16 / 2.0 when quantization.plan_dtype is absent.
    std::string effectiveDtype = getPlanDtype(parent).str();
    double dtypeBytes = dtypeBytesFromPlan(parent);

    // Read target-profile cost constants. Fall back to defaults when absent.
    double prefillMsPerToken  = kDefaultPrefillMsPerToken;
    double decodeMsPerToken   = kDefaultDecodeMsPerToken;
    double pdBandwidthMbPerMs = kDefaultPdBandwidthMbPerMs;
    bool hasTargetCostAttrs   = false;
    if (parent) {
      if (auto a = parent->getAttrOfType<FloatAttr>("target.prefill_ms_per_token")) {
        prefillMsPerToken  = a.getValueAsDouble();
        hasTargetCostAttrs = true;
      }
      if (auto a = parent->getAttrOfType<FloatAttr>("target.decode_ms_per_token")) {
        decodeMsPerToken   = a.getValueAsDouble();
        hasTargetCostAttrs = true;
      }
      if (auto a = parent->getAttrOfType<FloatAttr>("target.pd_bandwidth_mb_per_ms")) {
        pdBandwidthMbPerMs = a.getValueAsDouble();
        hasTargetCostAttrs = true;
      }
    }

    // Walk nested ops for llm.attention_prefill or llm.attention_decode.
    int64_t promptTokens = kDefaultPromptTokens;
    int64_t outputTokens = kDefaultOutputTokens;
    bool hasServingOp = false;

    funcOp.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      if (opName != "llm.attention_prefill" &&
          opName != "llm.attention_decode")
        return;
      hasServingOp = true;
      if (auto pt = op->getAttrOfType<IntegerAttr>("serving.prompt_tokens"))
        promptTokens = pt.getInt();
      if (auto ot = op->getAttrOfType<IntegerAttr>("serving.output_tokens"))
        outputTokens = ot.getInt();
    });

    if (!hasServingOp)
      return;

    ServingCostBreakdown col = computeColocatedCost(promptTokens, outputTokens,
                                                     prefillMsPerToken,
                                                     decodeMsPerToken);
    ServingCostBreakdown pd  = computePdSplitCost(promptTokens, outputTokens,
                                                   numLayers, hiddenSize,
                                                   prefillMsPerToken,
                                                   decodeMsPerToken,
                                                   pdBandwidthMbPerMs,
                                                   dtypeBytes);

    double marginMs  = std::abs(pd.total_ms - col.total_ms);
    double minTotal  = std::min(col.total_ms, pd.total_ms);
    double marginPct = marginMs / minTotal * 100.0;
    double diffFrac  = marginMs / minTotal;

    StringRef policy = col.total_ms <= pd.total_ms ? "colocated" : "pd_split";

    StringRef confidence;
    if (diffFrac < 0.05)
      confidence = "low";
    else if (diffFrac < 0.15)
      confidence = "medium";
    else
      confidence = "high";

    // cost_source tracks whether costs came from the target profile or defaults.
    StringRef costSource = hasTargetCostAttrs
        ? StringRef("target_profile_formula_estimate")
        : StringRef("formula_synthetic");
    StringRef costCalibration = hasTargetCostAttrs
        ? StringRef("target_profile")
        : StringRef("default_constants");

    Type f64 = Float64Type::get(context);
    funcOp->setAttr("serving.policy",
                    StringAttr::get(context, policy));
    funcOp->setAttr("serving.colocated_total_ms",
                    FloatAttr::get(f64, col.total_ms));
    funcOp->setAttr("serving.pd_split_total_ms",
                    FloatAttr::get(f64, pd.total_ms));
    funcOp->setAttr("serving.decision_margin_ms",
                    FloatAttr::get(f64, marginMs));
    funcOp->setAttr("serving.decision_margin_pct",
                    FloatAttr::get(f64, marginPct));
    funcOp->setAttr("serving.confidence",
                    StringAttr::get(context, confidence));
    funcOp->setAttr("serving.cost_calibration",
                    StringAttr::get(context, costCalibration));
    funcOp->setAttr("serving.cost_source",
                    StringAttr::get(context, costSource));
    funcOp->setAttr("serving.truth_boundary",
                    StringAttr::get(context,
                                    "estimated_cost_not_measured_latency"));

    // Propagate the quantization dtype decision to the function level so that
    // downstream serving passes and plan readers can see which dtype was used
    // in cost calculations without re-reading the module attr.
    funcOp->setAttr("quantization.effective_dtype",
                    StringAttr::get(context, effectiveDtype));
    funcOp->setAttr("quantization.dtype_bytes",
                    FloatAttr::get(f64, dtypeBytes));
    funcOp->setAttr("quantization.truth_boundary",
                    StringAttr::get(context,
                        "precision_selection_from_target_profile_not_calibrated"));
  }
};

} // namespace

std::unique_ptr<Pass> createServingPhaseAnalysisPass() {
  return std::make_unique<ServingPhaseAnalysisPass>();
}

} // namespace mlir::hir
