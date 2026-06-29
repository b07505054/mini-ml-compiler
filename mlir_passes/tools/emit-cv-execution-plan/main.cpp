// emit-cv-execution-plan: build and export a CVExecutionPlan from CV MLIR.

#include "CV/IR/CVDialect.h"
#include "FusionPasses.h"
#include "serving/CVExecutionPlan.h"
#include "serving/CVExecutionPlanBuilder.h"
#include "serving/CVExecutionPlanExporter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::opt<std::string> InputPath(
    "input", llvm::cl::desc("Path to input CV MLIR file"), llvm::cl::Required);

static llvm::cl::opt<std::string> OutputPath(
    "output", llvm::cl::desc("Output path for CV execution plan JSON"),
    llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "emit-cv-execution-plan: export a CV execution plan\n");

  // Load the registered CV dialect in the tool process. The official raw
  // pseudo-CV artifact still contains legacy op arities that are intentionally
  // not migrated in this commit, so parsing remains in the compatibility
  // context below until the fixture migration commit.
  mlir::MLIRContext registeredDialectCtx;
  registeredDialectCtx.loadDialect<mlir::func::FuncDialect,
                                   mlir::cv::CVDialect>();

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(InputPath, &ctx);
  if (!module) {
    llvm::errs() << "error: failed to parse CV MLIR file: " << InputPath
                 << "\n";
    return 1;
  }

  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVFrontendNormalizationPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVShapeInferencePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVMemoryPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVExecutionDomainPlanningPass());

  if (pm.run(module.get()).failed()) {
    llvm::errs() << "error: CV pass pipeline failed\n";
    return 1;
  }

  mlir::hir::CVExecutionPlan plan =
      mlir::hir::CVExecutionPlanBuilder::build(module.get());

  if (auto err =
          mlir::hir::CVExecutionPlanExporter::exportToFile(plan, OutputPath)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  llvm::outs() << "emit-cv-execution-plan:\n";
  llvm::outs() << "  output:            " << OutputPath << "\n";
  llvm::outs() << "  graph_count:       " << plan.graph_plans.size() << "\n";
  llvm::outs() << "  model_name:        " << plan.model_name << "\n";
  llvm::outs() << "  target_profile_id: " << plan.target_profile_id << "\n";
  return 0;
}
