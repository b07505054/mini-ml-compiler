#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::hir {

#define GEN_PASS_DECL
#include "FusionPasses.h.inc"

void registerFusionPasses();
void registerServingOptimizationPipeline();

std::unique_ptr<::mlir::Pass> createServingPhaseAnalysisPass();
std::unique_ptr<::mlir::Pass> createKVLayoutPlanningPass();

} // namespace mlir::hir