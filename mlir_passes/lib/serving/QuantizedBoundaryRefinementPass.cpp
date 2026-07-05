#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_QUANTIZEDBOUNDARYREFINEMENT
#include "FusionPasses.h.inc"

static constexpr StringLiteral kTruth =
    "quantized_boundary_refinement_static_not_materialized";

static bool inList(const std::vector<std::string> &list, const std::string &val) {
  return std::find(list.begin(), list.end(), val) != list.end();
}

static std::vector<std::string> readStringArr(mlir::Operation *op, llvm::StringRef attr) {
  std::vector<std::string> out;
  if (auto a = op->getAttrOfType<mlir::ArrayAttr>(attr))
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        out.push_back(s.getValue().str());
  return out;
}

struct QuantizedBoundaryRefinementPass
    : impl::QuantizedBoundaryRefinementBase<QuantizedBoundaryRefinementPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *ctx = funcOp.getContext();
    mlir::Operation *module = funcOp->getParentOp();

    if (funcOp.getBody().empty()) return;
    mlir::Block &entry = funcOp.getBody().front();

    // Function-level backend fallback when per-op lowering.target_backend is absent.
    std::string funcBackend;
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("representation.source_backend"))
      funcBackend = a.getValue().str();

    for (mlir::Operation &op : entry.without_terminator()) {
      std::string strategy;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("quant.strategy"))
        strategy = a.getValue().str();

      std::string weightDtype;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("quant.weight_dtype"))
        weightDtype = a.getValue().str();

      std::string loweringDecision;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("lowering.decision"))
        loweringDecision = a.getValue().str();

      // Effective backend: prefer lowering.target_backend, then kernel.fallback_backend,
      // then the function-level backend.
      std::string targetBackend;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("lowering.target_backend"))
        targetBackend = a.getValue().str();
      if (targetBackend.empty())
        if (auto a = op.getAttrOfType<mlir::StringAttr>("kernel.fallback_backend"))
          targetBackend = a.getValue().str();
      if (targetBackend.empty())
        targetBackend = funcBackend;

      bool weightDequantRequired = false;
      bool weightDtypeMismatch   = false;
      bool fallbackSupports      = true;
      std::string weightDequantReason;

      bool isWeightOnly = (strategy == "weight_only_int8");

      if (!isWeightOnly) {
        // Non-weight-quantized op: no dequant boundary needed.
        weightDequantReason = "not_weight_quantized";

      } else if (loweringDecision.empty()) {
        // LoweringDecisionPlanningPass has not run (standalone pipeline).
        // Optimistic default: do not assert a requirement we cannot verify.
        weightDequantReason = "target_backend_unknown";

      } else if (loweringDecision == "direct_lower") {
        // Exact kernel match: KernelAvailabilityPlanningPass already verified
        // the matched kernel supports the quant mode.
        weightDequantReason =
            "kernel_exact_match_backend_supports_weight_quantization";

      } else {
        // fallback_backend, rewrite_then_lower, dequant_then_lower, unsupported.
        // Check effective backend's declared quant capabilities.
        if (targetBackend.empty() || !module) {
          weightDequantReason = "target_backend_unknown";
        } else {
          auto quantModes = readStringArr(module,
              "target.backend_capabilities." + targetBackend + ".supported_quant_modes");
          // Consistent with QuantizationStrategyPlanningPass: empty list means
          // the backend has no declared quant support (not "any mode").
          bool supportsWeightOnly = inList(quantModes, "weight_only")
                                 || inList(quantModes, "static_int8");
          if (!supportsWeightOnly) {
            weightDequantRequired = true;
            weightDtypeMismatch   = true;
            fallbackSupports      = false;
            weightDequantReason   = "fallback_backend_lacks_quantized_weight_support";
          } else {
            weightDequantReason = "backend_supports_weight_quantization";
          }
        }
      }

      auto S = [&](llvm::StringRef s) { return mlir::StringAttr::get(ctx, s); };
      auto B = [&](bool b)            { return mlir::BoolAttr::get(ctx, b); };

      op.setAttr("boundary.fallback_backend_supports_weight_dtype", B(fallbackSupports));
      op.setAttr("boundary.quantized_weight_dtype",                 S(weightDtype));
      op.setAttr("boundary.weight_dequant_reason",                  S(weightDequantReason));
      op.setAttr("boundary.weight_dequant_required",                B(weightDequantRequired));
      op.setAttr("boundary.weight_dtype_mismatch",                  B(weightDtypeMismatch));
      op.setAttr("boundary.weight_truth_boundary",                  S(kTruth));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createQuantizedBoundaryRefinementPass() {
  return std::make_unique<QuantizedBoundaryRefinementPass>();
}

} // namespace mlir::hir
