// ServingCostModelPass — upgraded CandidateEvaluationPass.
//
// Applies ServingStaticCostModel_v1 to each compiler.candidates candidate dict
// and emits two layers of output per evaluated candidate:
//
//   Layer 1 (V0 backward-compat): evaluation.penalty_score, evaluation.status,
//     evaluation.reason, evaluation.truth_boundary.
//     Values and semantics are unchanged from CandidateEvaluationPass so that
//     PlanSelectionPass ranking and FileCheck tests remain green.
//
//   Layer 2 (V1 structured evidence): evaluation.cost.{compute,memory,dequant,
//     requant,layout_transform,cast,backend_switch,launch_overhead,kv_cache,
//     transfer,unsupported,total,model_id,truth_boundary}.
//     total == exact sum of all component fields (verifiable by the builder).
//
// Pass registration keeps the name "candidate-evaluation" so that the MLIR
// pipeline string, FileCheck tests, and ServingPipeline.cpp are unaffected.
// The C++ factory is createServingCostModelPass() (new) with a backward-compat
// alias createCandidateEvaluationPass() declared in FusionPasses.h.

#include "serving/ServingCostModel.h"
#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CANDIDATEEVALUATION
#include "FusionPasses.h.inc"

// ---------------------------------------------------------------------------
// V0 backward-compat penalty constants (kept for evaluation.penalty_score).
// These values are preserved to avoid changing PlanSelectionPass ranking
// and to keep FileCheck tests in candidate_evaluation.mlir green.
// ---------------------------------------------------------------------------
static constexpr int64_t kV0PenaltyDirectLower     = 0;
static constexpr int64_t kV0PenaltyDecomposition   = 5;
static constexpr int64_t kV0PenaltyReprConvBase    = 3;
static constexpr int64_t kV0PenaltyBoundaryDequant = 3;
static constexpr int64_t kV0PenaltyBoundaryCast    = 2;
static constexpr int64_t kV0PenaltyBoundaryLayout  = 2;
static constexpr int64_t kV0PenaltyLayoutConv      = 4;
static constexpr int64_t kV0PenaltyCastConv        = 2;
static constexpr int64_t kV0PenaltyBackendFallback = 20;
static constexpr int64_t kV0PenaltyUnsupported     = 100;
static constexpr int64_t kV0PenaltyUnknownCost     = 10;

// V0 truth_boundary (kept for evaluation.truth_boundary to minimize diff
// against tests that may check this field).
static constexpr StringLiteral kV0Truth =
    "candidate_evaluation_static_penalty_not_measured_latency";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string readStr(DictionaryAttr dict, StringRef key) {
  if (!dict) return {};
  if (auto a = dict.get(key))
    if (auto s = dyn_cast<StringAttr>(a)) return s.getValue().str();
  return {};
}

static std::vector<std::string> readStrs(DictionaryAttr dict, StringRef key) {
  std::vector<std::string> out;
  if (!dict) return out;
  if (auto a = dict.get(key))
    if (auto arr = dyn_cast<ArrayAttr>(a))
      for (auto e : arr)
        if (auto s = dyn_cast<StringAttr>(e)) out.push_back(s.getValue().str());
  return out;
}

// ---------------------------------------------------------------------------
// V0 penalty evaluation — preserved for evaluation.penalty_score backward compat.
// Returns {penalty_score, status, reason}.
// ---------------------------------------------------------------------------
struct V0Eval {
  int64_t penaltyScore;
  std::string status;
  std::string reason;
};

static V0Eval evaluatePenaltyV0(const std::string& candType,
                                  const std::vector<std::string>& boundaryOps) {
  int64_t directLower = 0, decomposition = 0, reprConv = 0;
  int64_t layoutConv = 0, castConv = 0, backendFallback = 0;
  int64_t unsupported = 0, unknownCost = 0;
  std::string status = "evaluated";
  std::string reason;

  if (candType == "direct_lower") {
    directLower = kV0PenaltyDirectLower;
    reason = "exact_kernel_zero_penalty";
  } else if (candType == "algebraic_decomposition") {
    decomposition = kV0PenaltyDecomposition;
    reason = "decomposition_base_penalty";
  } else if (candType == "representation_conversion") {
    reprConv = kV0PenaltyReprConvBase;
    reason = "representation_conversion_base_penalty";
    for (const auto& b : boundaryOps) {
      if (b == "dequant_weight" || b == "dequant") {
        reprConv += kV0PenaltyBoundaryDequant;
        reason += "+dequant_boundary";
      } else if (b == "cast") {
        reprConv += kV0PenaltyBoundaryCast;
        reason += "+cast_boundary";
      } else if (b == "layout_transform") {
        reprConv += kV0PenaltyBoundaryLayout;
        reason += "+layout_boundary";
      } else {
        unknownCost += kV0PenaltyUnknownCost;
        status = "partially_evaluated";
        reason += "+unknown_boundary";
      }
    }
  } else if (candType == "layout_conversion") {
    layoutConv = kV0PenaltyLayoutConv;
    reason = "layout_conversion_penalty";
  } else if (candType == "cast_conversion") {
    castConv = kV0PenaltyCastConv;
    reason = "cast_conversion_penalty";
  } else if (candType == "backend_fallback") {
    backendFallback = kV0PenaltyBackendFallback;
    reason = "backend_fallback_high_penalty";
  } else if (candType == "unsupported") {
    unsupported = kV0PenaltyUnsupported;
    status = "rejected";
    reason = "no_viable_lowering_path";
  } else {
    unknownCost = kV0PenaltyUnknownCost;
    status = "partially_evaluated";
    reason = "unknown_candidate_type";
  }

  int64_t total = directLower + decomposition + reprConv + layoutConv
                + castConv + backendFallback + unsupported + unknownCost;
  return {total, status, reason};
}

// ---------------------------------------------------------------------------
// Evaluate one candidate dict; return augmented DictionaryAttr with
// evaluation.penalty_score (V0), evaluation.status, evaluation.reason,
// evaluation.truth_boundary, and evaluation.cost.* (V1).
// ---------------------------------------------------------------------------
static DictionaryAttr evaluateCandidate(MLIRContext* ctx,
                                         DictionaryAttr dict,
                                         const ServingCostModel& model) {
  std::string candType    = readStr(dict, "candidate_type");
  std::string fbBackend   = readStr(dict, "fallback_backend");
  auto boundaryOps        = readStrs(dict, "required_boundary_ops");

  // V0 penalty (preserved for backward compat and PlanSelectionPass ranking).
  V0Eval v0 = evaluatePenaltyV0(candType, boundaryOps);

  // V1 structured cost (new evidence; total may differ from V0 penalty_score).
  DecisionCost v1 = model.compute(candType, boundaryOps, fbBackend);

  auto I64 = [&](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(ctx, 64), v);
  };
  auto S = [&](StringRef s) -> Attribute { return StringAttr::get(ctx, s); };

  // Copy original candidate fields and append evaluation.* fields.
  SmallVector<NamedAttribute> augmented(dict.begin(), dict.end());

  // V0 backward-compat attrs.
  augmented.push_back({StringAttr::get(ctx, "evaluation.penalty_score"), I64(v0.penaltyScore)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.reason"),        S(v0.reason)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.status"),        S(v0.status)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.truth_boundary"),S(kV0Truth)});

  // V1 structured cost attrs.
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.backend_switch"),  I64(v1.backend_switch_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.cast"),            I64(v1.cast_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.compute"),         I64(v1.compute_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.dequant"),         I64(v1.dequant_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.kv_cache"),        I64(v1.kv_cache_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.launch_overhead"), I64(v1.launch_overhead_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.layout_transform"),I64(v1.layout_transform_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.memory"),          I64(v1.memory_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.model_id"),        S(v1.cost_model_id)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.requant"),         I64(v1.requant_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.total"),           I64(v1.total_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.transfer"),        I64(v1.transfer_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.truth_boundary"),  S(v1.truth_boundary)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.unsupported"),     I64(v1.unsupported_penalty)});

  return DictionaryAttr::get(ctx, augmented);
}

// ---------------------------------------------------------------------------
// Pass struct — renamed to ServingCostModelPass.
// Still uses CandidateEvaluationBase to keep MLIR registration name
// "candidate-evaluation" and avoid breaking FileCheck tests / pipeline strings.
// ---------------------------------------------------------------------------
struct ServingCostModelPass
    : impl::CandidateEvaluationBase<ServingCostModelPass> {
  explicit ServingCostModelPass(StaticCostWeights w = StaticCostWeights{})
      : weights_(std::move(w)) {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* ctx = funcOp.getContext();
    ServingCostModel model(weights_);

    if (funcOp.getBody().empty()) return;
    for (Operation& op : funcOp.getBody().front().without_terminator()) {
      auto candArr = op.getAttrOfType<ArrayAttr>("compiler.candidates");
      if (!candArr) continue;

      SmallVector<Attribute> evaluated;
      evaluated.reserve(candArr.size());
      for (auto elem : candArr) {
        if (auto dict = dyn_cast<DictionaryAttr>(elem))
          evaluated.push_back(evaluateCandidate(ctx, dict, model));
        else
          evaluated.push_back(elem);
      }

      op.setAttr("compiler.evaluated_candidates",
                 ArrayAttr::get(ctx, evaluated));
      op.setAttr("compiler.evaluated_candidates.truth_boundary",
                 StringAttr::get(ctx, kV0Truth));
    }
  }

private:
  StaticCostWeights weights_;
};

} // namespace
} // namespace mlir::hir

namespace mlir::hir {

std::unique_ptr<mlir::Pass>
createServingCostModelPass(const StaticCostWeights& weights) {
  return std::make_unique<ServingCostModelPass>(weights);
}

// createCandidateEvaluationPass() is declared inline in FusionPasses.h
// and calls createServingCostModelPass() — no definition needed here.

} // namespace mlir::hir
