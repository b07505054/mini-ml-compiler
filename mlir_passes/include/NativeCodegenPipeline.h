#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::hir {

void buildHIRStructuredLoweringPipeline(OpPassManager &pm);
void buildBufferizedLLVMCodegenPipeline(OpPassManager &pm);
void buildNativeCodegenPipeline(OpPassManager &pm);
void registerNativeCodegenPipelines();

} // namespace mlir::hir
