#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::hir {

#define GEN_PASS_DECL
#include "FusionPasses.h.inc"

void registerFusionPasses();

std::unique_ptr<::mlir::Pass> createServingPhaseAnalysisPass();

} // namespace mlir::hir