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

  static PassPipelineRegistration<> tilePlanningPipeline(
      "tile-planning-pipeline",
      "Static local-memory tile feasibility planning for matmul-like ops",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createTilePlanningPass());
      });

  static PassPipelineRegistration<> kernelSelectionPipeline(
      "kernel-selection-pipeline",
      "Select a concrete runtime kernel descriptor per op (kernel_selection_contract_v1)",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createKernelSelectionPass());
      });

  static PassPipelineRegistration<> quantCoDesignPipeline(
      "quant-codesign-pipeline",
      "Quantization co-design evidence per op (quantization_codesign_contract_v1); inert without quant.codesign.policy",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createQuantizationCoDesignPass());
      });

  static PassPipelineRegistration<> boundaryMaterializationPipeline(
      "boundary-materialization-pipeline",
      "Materialize planned cast boundary ops into IR after plan selection",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createBoundaryMaterializationPass());
      });

  static PassPipelineRegistration<> weightClassPipeline(
      "weight-classification-planning-pipeline",
      "Classify weight operands as constant or runtime before quantization",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createWeightClassificationPlanningPass());
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

  static PassPipelineRegistration<> quantizedBoundaryRefinementPipeline(
      "quantized-boundary-refinement-pipeline",
      "Refine per-op weight dequant boundary requirements after lowering decisions are known",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createQuantizedBoundaryRefinementPass());
      });

  static PassPipelineRegistration<> alternativeLoweringPipeline(
      "alternative-lowering-planning-pipeline",
      "Generate alternative legal lowering candidates when exact kernel lowering is unavailable",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createAlternativeLoweringPlanningPass());
      });

  static PassPipelineRegistration<> candidatePipeline(
      "candidate-generation-pipeline",
      "Generate execution candidates from expanded BackendCapability schema",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCandidateGenerationPass());
      });

  static PassPipelineRegistration<> candidateEvalPipeline(
      "candidate-evaluation-pipeline",
      "Evaluate compiler.candidates with static relative penalty scores",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCandidateEvaluationPass());
      });

  static PassPipelineRegistration<> planSelectionPipeline(
      "plan-selection-pipeline",
      "Select the best evaluated candidate per op using static penalty scores",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createPlanSelectionPass());
      });

  // Sixteen-pass serving pipeline (mlir-opt standalone; compile-for-target uses its own PM).
  static PassPipelineRegistration<> servingOptPipeline(
      "serving-optimization-pipeline",
      "HIR serving pipeline: phase/cost, KV layout, replay eligibility, execution provider, representation, layout, boundary, weight classification, quant strategy, kernel availability, lowering decision, quantized boundary refinement, alternative lowering, candidate generation, candidate evaluation, plan selection",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createAttentionCandidateGenerationPass());
        pm.addNestedPass<func::FuncOp>(createAttentionLegalityPass());
        pm.addNestedPass<func::FuncOp>(createAttentionSelectionLoweringPass());
        pm.addNestedPass<func::FuncOp>(createServingPhaseAnalysisPass());
        pm.addNestedPass<func::FuncOp>(createKVLayoutPlanningPass());
        pm.addNestedPass<func::FuncOp>(createReplayEligibilityPass());
        pm.addNestedPass<func::FuncOp>(createExecutionProviderPlanningPass());
        pm.addNestedPass<func::FuncOp>(createRepresentationPlanningPass());
        pm.addNestedPass<func::FuncOp>(createLayoutPlanningPass());
        pm.addNestedPass<func::FuncOp>(createBoundaryPlanningPass());
        pm.addNestedPass<func::FuncOp>(createWeightClassificationPlanningPass());
        pm.addNestedPass<func::FuncOp>(createQuantizationStrategyPlanningPass());
        pm.addNestedPass<func::FuncOp>(createKernelAvailabilityPlanningPass());
        pm.addNestedPass<func::FuncOp>(createLoweringDecisionPlanningPass());
        pm.addNestedPass<func::FuncOp>(createQuantizedBoundaryRefinementPass());
        pm.addNestedPass<func::FuncOp>(createAlternativeLoweringPlanningPass());
        pm.addNestedPass<func::FuncOp>(createCandidateGenerationPass());
        pm.addNestedPass<func::FuncOp>(createCandidateEvaluationPass());
        pm.addNestedPass<func::FuncOp>(createPlanSelectionPass());
      });
}

} // namespace mlir::hir
