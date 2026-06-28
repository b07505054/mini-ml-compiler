// CTest unit test for CVExecutionPlanBuilder.
// Pure C++. No JSON. No exporter. No plugin loading.

#include "serving/CVExecutionPlan.h"
#include "serving/CVExecutionPlanBuilder.h"
#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <cassert>
#include <cstdio>

static const char kFullPipelineModule[] = R"mlir(
module {
  func.func @yoloseg_tiny(%x: tensor<1x1x2x2xf16>)
      -> tensor<1x4x2x2xf16> {
    %c = "cv.conv2d"(%x) {
      cv.source_op = "Conv"
    } : (tensor<1x1x2x2xf16>) -> tensor<1x4x2x2xf16>
    %b = "cv.batch_norm"(%c) {
      cv.source_op = "BatchNormalization"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %s = "cv.silu"(%b) {
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

  func.func @no_cv(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    return %x : tensor<1x4x2x2xf16>
  }
}
)mlir";

static const char kDomainOnlyModule[] = R"mlir(
module {
  func.func @domain_only(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    %c = "cv.conv2d"(%x) {
      cv.source_op = "Conv"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %d = "cv.detect_head"(%c) {
      cv.source_op = "Detect"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %d : tensor<1x4x2x2xf16>
  }
}
)mlir";

static const char kTargetProfileModule[] = R"mlir(
module attributes {
  cv.target.profile_id = "test_cv_target"
} {
  func.func @target_profile(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    %c = "cv.conv2d"(%x) {
      cv.source_op = "Conv"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %c : tensor<1x4x2x2xf16>
  }
}
)mlir";

static mlir::OwningOpRef<mlir::ModuleOp>
parseModule(const char *src, mlir::MLIRContext &ctx) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
  assert(module && "MLIR parse failed");
  return module;
}

static mlir::OwningOpRef<mlir::ModuleOp>
runFullCVPipeline(const char *src, mlir::MLIRContext &ctx) {
  auto module = parseModule(src, ctx);
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

static mlir::OwningOpRef<mlir::ModuleOp>
runDomainOnlyPipeline(const char *src, mlir::MLIRContext &ctx) {
  auto module = parseModule(src, ctx);
  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVExecutionDomainPlanningPass());
  assert(pm.run(module.get()).succeeded() && "PassManager run failed");
  return module;
}

int main() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  auto fullModule = runFullCVPipeline(kFullPipelineModule, ctx);
  mlir::hir::CVExecutionPlan plan =
      mlir::hir::CVExecutionPlanBuilder::build(fullModule.get());

  assert(plan.model_name == "yoloseg" && "model_name mismatch");
  assert(plan.target_profile_id == "portable_cv_pseudo_target" &&
         "default target_profile_id mismatch");
  assert(plan.graph_plans.size() == 1 && "expected one CV graph plan");
  const mlir::hir::CVGraphPlan &graph = plan.graph_plans[0];
  assert(graph.function_name == "yoloseg_tiny" && "function name mismatch");
  assert(graph.steps.size() == 5 && "expected five CV steps");

  assert(graph.execution_domain_plan.status == "completed" &&
         "execution-domain status mismatch");
  assert(graph.execution_domain_plan.accelerated_ops == 3 &&
         "accelerated op count mismatch");
  assert(graph.execution_domain_plan.host_ops == 2 &&
         "host op count mismatch");
  assert(graph.execution_domain_plan.fallback_ops == 0 &&
         "fallback op count mismatch");
  assert(graph.execution_domain_plan.planned_ops == 5 &&
         "domain planned op count mismatch");

  assert(graph.memory_plan.status == "completed" &&
         "memory-plan status mismatch");
  assert(graph.memory_plan.buffer_count == 2 && "buffer count mismatch");
  assert(graph.memory_plan.peak_memory_bytes == 64 &&
         "peak memory mismatch");
  assert(graph.memory_plan.planned_ops == 5 &&
         "memory planned op count mismatch");
  assert(graph.memory_plan.reused_buffer_count == 3 &&
         "reuse count mismatch");
  assert(graph.memory_plan.total_allocated_bytes == 64 &&
         "allocated bytes mismatch");

  assert(graph.truth_boundaries.frontend ==
             "raw_pseudo_cv_mlir_not_full_onnx_importer" &&
         "frontend truth boundary mismatch");
  assert(graph.truth_boundaries.shape_inference ==
             "static_ranked_tensor_shape_metadata_not_runtime_allocation" &&
         "shape truth boundary mismatch");
  assert(graph.truth_boundaries.memory_plan ==
             "static_compiler_memory_plan_not_runtime_allocation" &&
         "memory truth boundary mismatch");
  assert(graph.truth_boundaries.execution_domain ==
             "static_execution_domain_classification_not_target_mapping" &&
         "domain truth boundary mismatch");

  assert(graph.steps[0].op_name == "cv.conv2d" && "op name mismatch");
  assert(graph.steps[0].source_op == "Conv" && "source op mismatch");
  assert(graph.steps[0].execution_domain == "accelerated" &&
         "conv domain mismatch");
  assert(graph.steps[0].execution_domain_reason ==
             "accelerated_tensor_policy" &&
         "conv domain reason mismatch");
  assert(graph.steps[0].buffer_id == 0 && "conv buffer id mismatch");
  assert(graph.steps[0].buffer_offset == 0 && "conv buffer offset mismatch");
  assert(graph.steps[0].lifetime_begin == 0 && "conv lifetime begin mismatch");
  assert(graph.steps[0].lifetime_end == 1 && "conv lifetime end mismatch");
  assert(graph.steps[0].bytes_estimate == 32 && "conv bytes mismatch");
  assert(graph.steps[0].num_elements == 16 && "conv elements mismatch");
  assert(graph.steps[0].memory_plan_reason ==
             "linear_lifetime_reuse_from_static_shape_metadata" &&
         "memory plan reason mismatch");

  assert(graph.steps[2].buffer_id == 0 && "silu should reuse buffer 0");
  assert(graph.steps[3].execution_domain == "host" &&
         "detect_head domain mismatch");
  assert(graph.steps[4].execution_domain == "host" &&
         "prototype_head domain mismatch");
  std::puts("  [PASS] full CV pipeline plan");

  auto domainOnlyModule = runDomainOnlyPipeline(kDomainOnlyModule, ctx);
  mlir::hir::CVExecutionPlan domainOnlyPlan =
      mlir::hir::CVExecutionPlanBuilder::build(domainOnlyModule.get());

  assert(domainOnlyPlan.graph_plans.size() == 1 &&
         "expected one domain-only graph plan");
  assert(domainOnlyPlan.model_name == "unknown" &&
         "missing frontend model_name should be unknown");
  const mlir::hir::CVGraphPlan &domainGraph =
      domainOnlyPlan.graph_plans[0];
  assert(domainGraph.steps.size() == 2 &&
         "expected two domain-only CV steps");
  assert(domainGraph.steps[0].execution_domain == "accelerated" &&
         "domain-only conv domain mismatch");
  assert(domainGraph.steps[1].execution_domain == "host" &&
         "domain-only detect domain mismatch");
  assert(domainGraph.steps[0].bytes_estimate == -1 &&
         "missing bytes_estimate should be -1");
  assert(domainGraph.steps[0].buffer_id == -1 &&
         "missing buffer_id should be -1");
  assert(domainGraph.steps[0].memory_plan_reason == "absent" &&
         "missing string attrs should be absent");
  assert(domainGraph.memory_plan.status == "absent" &&
         "missing memory summary should be absent");
  assert(domainGraph.memory_plan.peak_memory_bytes == -1 &&
         "missing memory summary ints should be -1");
  std::puts("  [PASS] domain-only missing-attr defaults");

  auto targetProfileModule = runDomainOnlyPipeline(kTargetProfileModule, ctx);
  mlir::hir::CVExecutionPlan targetProfilePlan =
      mlir::hir::CVExecutionPlanBuilder::build(targetProfileModule.get());

  assert(targetProfilePlan.target_profile_id == "test_cv_target" &&
         "module target_profile_id mismatch");
  std::puts("  [PASS] target profile metadata");

  std::puts("CVExecutionPlanBuilderTest: PASS");
  return 0;
}
