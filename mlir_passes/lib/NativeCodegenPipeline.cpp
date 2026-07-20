#include "NativeCodegenPipeline.h"

#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::hir {
namespace {

void appendPipeline(OpPassManager &pm, StringRef pipeline) {
  if (failed(parsePassPipeline(pipeline, pm)))
    llvm::report_fatal_error("failed to construct native codegen pipeline");
}

} // namespace

void buildHIRStructuredLoweringPipeline(OpPassManager &pm) {
  appendPipeline(pm,
                 "quantization-planning,matmul-bias-relu-fusion,"
                 "hir-fusion-lowering,hir-matmul-bias-relu-to-linalg");
}

void buildBufferizedLLVMCodegenPipeline(OpPassManager &pm) {
  appendPipeline(pm,
                 "one-shot-bufferize{bufferize-function-boundaries},"
                 "buffer-deallocation-pipeline,convert-linalg-to-loops,"
                 "convert-scf-to-cf,convert-index-to-llvm,"
                 "convert-math-to-llvm,convert-arith-to-llvm,"
                 "expand-strided-metadata,"
                 "finalize-memref-to-llvm,convert-func-to-llvm,"
                 "convert-cf-to-llvm,reconcile-unrealized-casts");
}

void buildNativeCodegenPipeline(OpPassManager &pm) {
  buildHIRStructuredLoweringPipeline(pm);
  buildBufferizedLLVMCodegenPipeline(pm);
}

void registerNativeCodegenPipelines() {
  static PassPipelineRegistration<> structured(
      "hir-structured-lowering",
      "Planning-guided MatMul+Bias+ReLU lowering to structured Linalg",
      buildHIRStructuredLoweringPipeline);
  static PassPipelineRegistration<> llvmCodegen(
      "bufferized-llvm-codegen",
      "Canonical scalar structured Linalg to LLVM-dialect lowering",
      buildBufferizedLLVMCodegenPipeline);
  static PassPipelineRegistration<> native(
      "hir-native-codegen",
      "Canonical planning-guided scalar HIR/Linalg to LLVM dialect",
      buildNativeCodegenPipeline);
}

} // namespace mlir::hir
