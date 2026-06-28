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
std::unique_ptr<::mlir::Pass> createLLMFrontendNormalizationPass();
std::unique_ptr<::mlir::Pass> createQuantizationPlanningPass();
std::unique_ptr<::mlir::Pass> createCVFrontendNormalizationPass();
std::unique_ptr<::mlir::Pass> createCVShapeInferencePass();
std::unique_ptr<::mlir::Pass> createCVMemoryPlanningPass();
std::unique_ptr<::mlir::Pass> createCVExecutionDomainPlanningPass();

} // namespace mlir::hir