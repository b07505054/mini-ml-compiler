#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::hir {

void registerServingOptimizationPipeline() {
  // Single-pass wrapper so kv-layout-planning can be used standalone via
  // --pass-pipeline='builtin.module(kv-layout-planning)', mirroring the
  // serving-phase-analysis wrapper already in registerFusionPasses().
  static PassPipelineRegistration<> kvLayoutPipeline(
      "kv-layout-planning",
      "Annotate serving functions with KV cache layout policy and byte estimate",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createKVLayoutPlanningPass());
      });

  // Two-pass serving pipeline: phase/cost/policy analysis then KV planning.
  static PassPipelineRegistration<> servingOptPipeline(
      "serving-optimization-pipeline",
      "HIR serving analysis pipeline: phase/cost/policy then KV layout planning",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createServingPhaseAnalysisPass());
        pm.addNestedPass<func::FuncOp>(createKVLayoutPlanningPass());
      });
}

} // namespace mlir::hir
