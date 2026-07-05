#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::hir {

#define GEN_PASS_DECL
#include "FusionPasses.h.inc"

void registerFusionPasses();
void registerServingOptimizationPipeline();

std::unique_ptr<::mlir::Pass> createServingPhaseAnalysisPass();
std::unique_ptr<::mlir::Pass> createKVLayoutPlanningPass();
std::unique_ptr<::mlir::Pass> createReplayEligibilityPass();
std::unique_ptr<::mlir::Pass> createExecutionProviderPlanningPass();
std::unique_ptr<::mlir::Pass> createRepresentationPlanningPass();
std::unique_ptr<::mlir::Pass> createLayoutPlanningPass();
std::unique_ptr<::mlir::Pass> createBoundaryPlanningPass();
std::unique_ptr<::mlir::Pass> createWeightClassificationPlanningPass();
std::unique_ptr<::mlir::Pass> createQuantizationStrategyPlanningPass();
std::unique_ptr<::mlir::Pass> createKernelAvailabilityPlanningPass();
std::unique_ptr<::mlir::Pass> createLoweringDecisionPlanningPass();
std::unique_ptr<::mlir::Pass> createQuantizedBoundaryRefinementPass();
std::unique_ptr<::mlir::Pass> createAlternativeLoweringPlanningPass();
std::unique_ptr<::mlir::Pass> createCandidateGenerationPass();
std::unique_ptr<::mlir::Pass> createCandidateEvaluationPass();
std::unique_ptr<::mlir::Pass> createPlanSelectionPass();
std::unique_ptr<::mlir::Pass> createLLMFrontendNormalizationPass();
std::unique_ptr<::mlir::Pass> createQuantizationPlanningPass();
std::unique_ptr<::mlir::Pass> createHIRQuantCanonicalizationPass();
std::unique_ptr<::mlir::Pass> createHIRQuantPropagationPass();
std::unique_ptr<::mlir::Pass> createHIRINT8OperatorSelectionPass();
std::unique_ptr<::mlir::Pass> createCVFrontendNormalizationPass();
std::unique_ptr<::mlir::Pass> createCVShapeInferencePass();
std::unique_ptr<::mlir::Pass> createCVMemoryPlanningPass();
std::unique_ptr<::mlir::Pass> createCVExecutionDomainPlanningPass();

} // namespace mlir::hir
