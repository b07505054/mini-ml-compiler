// CTest unit test for ServingExecutionPlanBuilder.
// Pure C++. No GoogleTest. No Python. No JSON. No plugin loading.
// Compiles ServingPhaseAnalysisPass, KVLayoutPlanningPass,
// ReplayEligibilityPass, and ServingExecutionPlanBuilder directly.

#include "serving/ServingExecutionPlan.h"
#include "serving/ServingExecutionPlanBuilder.h"
#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <cassert>
#include <cstdio>

// MLIR module with two serving functions:
//
//   @prefill  -- dynamic tensor shape, kv_cache.role="producer"
//     Expected: serving.policy="colocated", kv.layout=Paged, replay.eligible=false
//
//   @decode   -- static tensor shape, kv_cache.role="consumer"
//     Expected: kv.layout=Contiguous, replay.eligible=true, bucket="decode_static"
//
// Module attrs: num_layers=12, hidden_size=768, prompt_tokens=128, output_tokens=64
//   kv_mb = 12 * 2 * 768 * 2 * 192 / 1MB ≈ 6.75
static const char kTestModule[] = R"mlir(
module attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64
} {
  func.func @prefill(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }

  func.func @decode(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.phase = "decode",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}
)mlir";

int main() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(kTestModule, &ctx);
  assert(module && "MLIR parse failed");

  mlir::PassManager pm(&ctx);
  // addNestedPass constructs passes directly — no plugin loading needed.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createServingPhaseAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createKVLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createReplayEligibilityPass());
  assert(pm.run(module.get()).succeeded() && "PassManager run failed");

  mlir::hir::ServingExecutionPlan plan =
      mlir::hir::ServingExecutionPlanBuilder::build(module.get());

  // Module-level attrs.
  assert(plan.model_name == "tiny-gpt" && "model_name mismatch");
  assert(plan.num_layers  == 12        && "num_layers mismatch");
  assert(plan.hidden_size == 768       && "hidden_size mismatch");

  // Both functions are annotated (both have serving.policy after the pass).
  assert(plan.function_plans.size() == 2 && "expected 2 function plans");

  // -----------------------------------------------------------------------
  // @prefill assertions
  // -----------------------------------------------------------------------
  const mlir::hir::FunctionExecutionPlan &fp_prefill = plan.function_plans[0];
  assert(fp_prefill.function_name == "prefill"        && "function_name mismatch");
  assert(fp_prefill.serving_phase == mlir::hir::ServingPhase::Prefill
         && "serving_phase mismatch");
  assert(fp_prefill.execution_mode == mlir::hir::ExecutionMode::Colocated
         && "execution_mode mismatch");
  assert(fp_prefill.cost_summary.confidence == mlir::hir::Confidence::Low
         && "confidence mismatch");
  assert(fp_prefill.provenance.cost_source    == "formula_synthetic"
         && "cost_source mismatch");
  assert(fp_prefill.provenance.truth_boundary == "estimated_cost_not_measured_latency"
         && "truth_boundary mismatch");

  // KV: producer => paged layout.
  assert(fp_prefill.kv_plan.layout == mlir::hir::KVLayout::Paged
         && "kv_layout should be Paged for kv_cache.role=producer");
  assert(fp_prefill.kv_plan.kv_byte_estimate_mb > 0.0
         && "kv_byte_estimate_mb should be positive");

  // Replay: prefill is not eligible.
  assert(fp_prefill.replay_plan.replay_eligible == false
         && "prefill should not be replay-eligible");
  assert(fp_prefill.replay_plan.cuda_graph_bucket.empty()
         && "prefill cuda_graph_bucket should be empty");

  // All 3 passes contributed attrs.
  assert(fp_prefill.source_passes.size() == 3
         && fp_prefill.source_passes[0] == "serving-phase-analysis"
         && fp_prefill.source_passes[1] == "kv-layout-planning"
         && fp_prefill.source_passes[2] == "replay-eligibility"
         && "source_passes mismatch");

  // -----------------------------------------------------------------------
  // @decode assertions
  // -----------------------------------------------------------------------
  const mlir::hir::FunctionExecutionPlan &fp_decode = plan.function_plans[1];
  assert(fp_decode.function_name == "decode" && "function_name mismatch");
  assert(fp_decode.serving_phase == mlir::hir::ServingPhase::Decode
         && "serving_phase mismatch");

  // KV: consumer => contiguous layout.
  assert(fp_decode.kv_plan.layout == mlir::hir::KVLayout::Contiguous
         && "kv_layout should be Contiguous for kv_cache.role=consumer");
  assert(fp_decode.kv_plan.kv_byte_estimate_mb > 0.0
         && "kv_byte_estimate_mb should be positive");

  // Replay: static decode is eligible.
  assert(fp_decode.replay_plan.replay_eligible == true
         && "static decode should be replay-eligible");
  assert(fp_decode.replay_plan.cuda_graph_bucket == "decode_static"
         && "cuda_graph_bucket mismatch");

  std::puts("ServingExecutionPlanBuilderTest: PASS");
  return 0;
}
