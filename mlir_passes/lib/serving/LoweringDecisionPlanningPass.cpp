#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#define GEN_PASS_DEF_LOWERINGDECISIONPLANNING
#include "FusionPasses.h.inc"

namespace mlir::hir {
namespace {

static constexpr StringLiteral kTruth =
    "lowering_decision_static_not_backend_codegen_verified";

static void annotateOp(mlir::Operation &op,
                       llvm::StringRef decision,
                       llvm::StringRef reason,
                       llvm::StringRef requiredRewrite,
                       bool requiresCast,
                       bool requiresDequant,
                       bool requiresLayoutTransform,
                       llvm::StringRef targetBackend,
                       llvm::StringRef targetKernel,
                       llvm::StringRef targetKernelLibrary,
                       mlir::MLIRContext *ctx) {
  auto S = [&](llvm::StringRef s) { return mlir::StringAttr::get(ctx, s); };
  auto B = [&](bool b)            { return mlir::BoolAttr::get(ctx, b); };
  op.setAttr("lowering.decision",                S(decision));
  op.setAttr("lowering.reason",                  S(reason));
  op.setAttr("lowering.required_rewrite",        S(requiredRewrite));
  op.setAttr("lowering.requires_cast",           B(requiresCast));
  op.setAttr("lowering.requires_dequant",        B(requiresDequant));
  op.setAttr("lowering.requires_layout_transform", B(requiresLayoutTransform));
  op.setAttr("lowering.target_backend",          S(targetBackend));
  op.setAttr("lowering.target_kernel",           S(targetKernel));
  op.setAttr("lowering.target_kernel_library",   S(targetKernelLibrary));
  op.setAttr("lowering.truth_boundary",          S(kTruth));
}

struct LoweringDecisionPlanningPass
    : ::impl::LoweringDecisionPlanningBase<LoweringDecisionPlanningPass> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext *ctx  = &getContext();

    if (func.getBody().empty()) return;
    mlir::Block &entry = func.getBody().front();

    for (mlir::Operation &op : entry.without_terminator()) {
      // Skip ops that KernelAvailabilityPlanningPass did not annotate.
      auto statusAttr = op.getAttrOfType<mlir::StringAttr>("kernel.lowering_status");
      if (!statusAttr) continue;

      llvm::StringRef status = statusAttr.getValue();

      // Helper lambdas for reading op attrs.
      auto str = [&](llvm::StringRef k) -> std::string {
        if (auto a = op.getAttrOfType<mlir::StringAttr>(k)) return a.getValue().str();
        return {};
      };
      auto boolean = [&](llvm::StringRef k) -> bool {
        if (auto a = op.getAttrOfType<mlir::BoolAttr>(k)) return a.getValue();
        return false;
      };

      std::string kernelBackend  = str("kernel.backend");
      std::string kernelLibrary  = str("kernel.library");
      std::string kernelName     = str("kernel.name");
      std::string fallbackBackend = str("kernel.fallback_backend");
      std::string requiredRewrite = str("kernel.required_rewrite");

      bool dequantRequired       = boolean("boundary.dequant_required");
      bool castRequired          = boolean("boundary.cast_required");
      bool layoutTransformReq    = boolean("boundary.layout_transform_required");

      // Rule 1: exact kernel available → direct_lower.
      if (status == "lowerable") {
        annotateOp(op,
                   "direct_lower",
                   "kernel_available_for_op_dtype_layout",
                   "",
                   castRequired, dequantRequired, layoutTransformReq,
                   kernelBackend, kernelName, kernelLibrary,
                   ctx);
        continue;
      }

      // Rule 2: rewrite pattern available → rewrite_then_lower.
      if (status == "rewrite_candidate") {
        annotateOp(op,
                   "rewrite_then_lower",
                   "rewrite_then_lower_via_" + requiredRewrite,
                   requiredRewrite,
                   castRequired, dequantRequired, layoutTransformReq,
                   kernelBackend, kernelName, kernelLibrary,
                   ctx);
        continue;
      }

      // Rules 3 & 4: fallback_required.
      if (status == "fallback_required") {
        // Rule 3: dequant boundary required → dequant_then_lower (takes priority).
        if (dequantRequired) {
          std::string target = fallbackBackend.empty() ? kernelBackend : fallbackBackend;
          annotateOp(op,
                     "dequant_then_lower",
                     "dequant_required_before_fallback_to_" + target,
                     "",
                     castRequired, true, layoutTransformReq,
                     target, "", "",
                     ctx);
          continue;
        }
        // Rule 4: fallback backend declared.
        if (!fallbackBackend.empty()) {
          annotateOp(op,
                     "fallback_backend",
                     "fallback_to_" + fallbackBackend,
                     "",
                     castRequired, dequantRequired, layoutTransformReq,
                     fallbackBackend, "", "",
                     ctx);
          continue;
        }
        // fallback_required but no dequant and no fallback_backend → unsupported.
        annotateOp(op,
                   "unsupported",
                   "fallback_required_but_no_path",
                   "",
                   castRequired, dequantRequired, layoutTransformReq,
                   kernelBackend, "", "",
                   ctx);
        continue;
      }

      // Rule 5: unsupported (or unexpected status value).
      annotateOp(op,
                 "unsupported",
                 "no_kernel_no_rewrite_no_fallback",
                 "",
                 castRequired, dequantRequired, layoutTransformReq,
                 kernelBackend, "", "",
                 ctx);
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createLoweringDecisionPlanningPass() {
  return std::make_unique<LoweringDecisionPlanningPass>();
}

} // namespace mlir::hir
