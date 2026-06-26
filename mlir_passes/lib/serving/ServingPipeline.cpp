#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::hir {

void registerServingOptimizationPipeline() {
  // Standalone wrappers mirror the serving-phase-analysis pattern in
  // registerFusionPasses(), allowing each pass to be used independently.
  static PassPipelineRegistration<> kvLayoutPipeline(
      "kv-layout-planning",
      "Annotate serving functions with KV cache layout policy and byte estimate",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createKVLayoutPlanningPass());
      });

  static PassPipelineRegistration<> replayEligibilityPipeline(
      "replay-eligibility",
      "Annotate serving functions with CUDA graph replay eligibility metadata",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createReplayEligibilityPass());
      });

  // Three-pass serving pipeline.
  static PassPipelineRegistration<> servingOptPipeline(
      "serving-optimization-pipeline",
      "HIR serving pipeline: phase/cost, KV layout, replay eligibility",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createServingPhaseAnalysisPass());
        pm.addNestedPass<func::FuncOp>(createKVLayoutPlanningPass());
        pm.addNestedPass<func::FuncOp>(createReplayEligibilityPass());
      });
}

} // namespace mlir::hir
