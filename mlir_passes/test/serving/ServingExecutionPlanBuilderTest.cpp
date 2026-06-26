// CTest unit test for ServingExecutionPlanBuilder.
// Pure C++. No GoogleTest. No Python. No JSON. No plugin loading.
// Compiles ServingPhaseAnalysisPass, KVLayoutPlanningPass, and
// ServingExecutionPlanBuilder directly.

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

// MLIR module with:
//   - module-level llm.model, llm.num_layers=12, llm.hidden_size=768
//   - one func.func @prefill containing llm.attention_prefill (kv_cache.role="producer")
//   - serving.prompt_tokens=128, serving.output_tokens=64
// Expected after both serving passes:
//   serving: policy="colocated", confidence="low"  (see serving_phase_analysis.mlir)
//   kv:      layout=Paged, byte_estimate_mb ≈ 6.75
//            (12 * 2 * 768 * 2 * 192 / 1MB)
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
  assert(pm.run(module.get()).succeeded() && "PassManager run failed");

  mlir::hir::ServingExecutionPlan plan =
      mlir::hir::ServingExecutionPlanBuilder::build(module.get());

  // Module-level attrs
  assert(plan.model_name == "tiny-gpt"  && "model_name mismatch");
  assert(plan.num_layers == 12          && "num_layers mismatch");
  assert(plan.hidden_size == 768        && "hidden_size mismatch");

  // One function annotated
  assert(plan.function_plans.size() == 1 && "expected 1 function plan");

  const mlir::hir::FunctionExecutionPlan &fp = plan.function_plans[0];
  assert(fp.function_name == "prefill" && "function_name mismatch");
  assert(fp.serving_phase == mlir::hir::ServingPhase::Prefill
         && "serving_phase mismatch");
  assert(fp.execution_mode == mlir::hir::ExecutionMode::Colocated
         && "execution_mode mismatch");
  assert(fp.cost_summary.confidence == mlir::hir::Confidence::Low
         && "confidence mismatch");
  assert(fp.provenance.cost_source == "formula_synthetic"
         && "cost_source mismatch");
  assert(fp.provenance.truth_boundary == "estimated_cost_not_measured_latency"
         && "truth_boundary mismatch");

  // KV plan from KVLayoutPlanningPass: producer role => paged layout.
  assert(fp.kv_plan.layout == mlir::hir::KVLayout::Paged
         && "kv_layout should be Paged for kv_cache.role=producer");
  assert(fp.kv_plan.kv_byte_estimate_mb > 0.0
         && "kv_byte_estimate_mb should be positive");

  // replay_plan remains at defaults until ReplayEligibilityPass is implemented.
  assert(fp.replay_plan.replay_eligible == false
         && "replay_eligible should be false before ReplayEligibilityPass");

  // Builder records one source_pass entry per contributing pass.
  assert(fp.source_passes.size() == 2
         && fp.source_passes[0] == "serving-phase-analysis"
         && fp.source_passes[1] == "kv-layout-planning"
         && "source_passes mismatch");

  std::puts("ServingExecutionPlanBuilderTest: PASS");
  return 0;
}
