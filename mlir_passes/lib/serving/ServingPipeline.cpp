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

  static PassPipelineRegistration<> executionProviderPipeline(
      "execution-provider-planning",
      "Produce an execution provider plan for serving functions",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createExecutionProviderPlanningPass());
      });

  static PassPipelineRegistration<> representationPipeline(
      "representation-planning-pipeline",
      "Annotate serving functions with effective dtype and layout from backend capabilities",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createRepresentationPlanningPass());
      });

  static PassPipelineRegistration<> layoutPipeline(
      "layout-planning-pipeline",
      "Annotate ops with layout assignment from backend capabilities",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createLayoutPlanningPass());
      });

  static PassPipelineRegistration<> boundaryPipeline(
      "boundary-planning-pipeline",
      "Annotate ops with boundary materialization requirements from backend capabilities",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createBoundaryPlanningPass());
      });

  static PassPipelineRegistration<> quantStrategyPipeline(
      "quantization-strategy-planning-pipeline",
      "Per-op quantization strategy annotation for the HIR serving pipeline",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createQuantizationStrategyPlanningPass());
      });

  static PassPipelineRegistration<> kernelAvailPipeline(
      "kernel-availability-planning-pipeline",
      "Per-op kernel availability annotation from KernelLibraryCapability schema",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createKernelAvailabilityPlanningPass());
      });

  static PassPipelineRegistration<> loweringDecisionPipeline(
      "lowering-decision-planning-pipeline",
      "Per-op final lowering decision from kernel.* and boundary.* attrs",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createLoweringDecisionPlanningPass());
      });

  static PassPipelineRegistration<> candidatePipeline(
      "candidate-generation-pipeline",
      "Generate execution candidates from expanded BackendCapability schema",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCandidateGenerationPass());
      });

  // Eleven-pass serving pipeline (mlir-opt standalone; compile-for-target uses its own PM).
  static PassPipelineRegistration<> servingOptPipeline(
      "serving-optimization-pipeline",
      "HIR serving pipeline: phase/cost, KV layout, replay eligibility, execution provider, representation, layout, boundary, quant strategy, kernel availability, lowering decision, candidates",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createServingPhaseAnalysisPass());
        pm.addNestedPass<func::FuncOp>(createKVLayoutPlanningPass());
        pm.addNestedPass<func::FuncOp>(createReplayEligibilityPass());
        pm.addNestedPass<func::FuncOp>(createExecutionProviderPlanningPass());
        pm.addNestedPass<func::FuncOp>(createRepresentationPlanningPass());
        pm.addNestedPass<func::FuncOp>(createLayoutPlanningPass());
        pm.addNestedPass<func::FuncOp>(createBoundaryPlanningPass());
        pm.addNestedPass<func::FuncOp>(createQuantizationStrategyPlanningPass());
        pm.addNestedPass<func::FuncOp>(createKernelAvailabilityPlanningPass());
        pm.addNestedPass<func::FuncOp>(createLoweringDecisionPlanningPass());
        pm.addNestedPass<func::FuncOp>(createCandidateGenerationPass());
      });
}

} // namespace mlir::hir
