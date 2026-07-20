#include "FusionPasses.h"
#include "CV/IR/CVDialect.h"
#include "HIR/IR/HIRDialect.h"
#include "NativeCodegenPipeline.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MatMulBiasReluFusionPass",
          LLVM_VERSION_STRING,
          []() {
            mlir::hir::registerFusionPasses();
            mlir::hir::registerNativeCodegenPipelines();
          }};
}

extern "C" ::mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HIRDialect", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<mlir::cv::CVDialect, mlir::hir::HIRDialect>();
          }};
}
