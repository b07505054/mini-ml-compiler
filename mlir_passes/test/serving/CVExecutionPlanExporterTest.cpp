// CTest unit test for CVExecutionPlanExporter.

#include "CV/IR/CVDialect.h"
#include "serving/CVExecutionPlan.h"
#include "serving/CVExecutionPlanBuilder.h"
#include "serving/CVExecutionPlanExporter.h"
#include "FusionPasses.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <cassert>
#include <cstdio>
#include <string>

static const char kCVModule[] = R"mlir(
module {
  func.func @yoloseg_tiny(%x: tensor<1x1x2x2xf16>)
      -> tensor<1x4x2x2xf16> {
    %c = "cv.conv2d"(%x) {
      cv.source_op = "Conv"
    } : (tensor<1x1x2x2xf16>) -> tensor<1x4x2x2xf16>
    %s = "cv.silu"(%c) {
      cv.source_op = "Mul"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %d = "cv.detect_head"(%s) {
      cv.source_op = "Detect"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %p = "cv.prototype_head"(%d) {
      cv.source_op = "Proto"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %p : tensor<1x4x2x2xf16>
  }
}
)mlir";

static mlir::OwningOpRef<mlir::ModuleOp>
runCVPipeline(const char *src, mlir::MLIRContext &ctx) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
  assert(module && "MLIR parse failed");

  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVFrontendNormalizationPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVShapeInferencePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVMemoryPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVExecutionDomainPlanningPass());
  assert(pm.run(module.get()).succeeded() && "PassManager run failed");
  return module;
}

static bool contains(const std::string &text, const char *needle) {
  return text.find(needle) != std::string::npos;
}

int main() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect, mlir::cv::CVDialect>();

  auto module = runCVPipeline(kCVModule, ctx);
  mlir::hir::CVExecutionPlan plan =
      mlir::hir::CVExecutionPlanBuilder::build(module.get());

  llvm::SmallString<256> outPath;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("cv_execution_plan_export", "json",
                                         outPath);
  assert(!ec && "failed to create temporary output file");

  llvm::Error err =
      mlir::hir::CVExecutionPlanExporter::exportToFile(plan, outPath);
  assert(!err && "CVExecutionPlanExporter::exportToFile failed");

  auto file = llvm::MemoryBuffer::getFile(outPath);
  assert(file && "failed to read exported JSON");
  std::string json = file.get()->getBuffer().str();

  assert(contains(json, "cv_execution_plan"));
  assert(contains(json, "model_name"));
  assert(contains(json, "yoloseg"));
  assert(contains(json, "target_profile_id"));
  assert(contains(json, "portable_cv_pseudo_target"));
  assert(contains(json, "generated_at"));
  assert(contains(json, "graph_plans"));
  assert(contains(json, "execution_domain"));
  assert(contains(json, "accelerated"));
  assert(contains(json, "host"));
  assert(contains(json, "buffer_offset"));

  assert(!contains(json, "metal"));
  assert(!contains(json, "cpu"));
  assert(!contains(json, "cuda"));
  assert(!contains(json, "coreml"));
  assert(!contains(json, "ane"));

  llvm::sys::fs::remove(outPath);

  std::puts("CVExecutionPlanExporterTest: PASS");
  return 0;
}
