// CTest unit test for ServingExecutionPlanBuilder.
// Pure C++. No GoogleTest. No Python. No JSON. No plugin loading.
// Compiles ServingPhaseAnalysisPass, KVLayoutPlanningPass,
// ReplayEligibilityPass, ExecutionProviderPlanningPass, and
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

// ---------------------------------------------------------------------------
// Module 1: unconstrained (no target.* attrs).
//
//   @prefill  -- dynamic tensor shape, producer role
//     kv.layout=Paged, replay.eligible=false
//     EP: constraint_conflict (paged KV, cpu not paged-compatible)
//
//   @decode   -- static tensor shape, consumer role
//     kv.layout=Contiguous, replay.eligible=true, bucket="decode_static"
//     EP: default_policy, primary="cpu"
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Module 2: constrained (preferred_backend = "coreml", static decode).
//
//   @decode_constrained  -- static, consumer role -> contiguous KV
//     EP: target_preferred, primary="coreml", fallback=["metal","cpu"]
//         required_precision="fp16", requires_replay=true
// ---------------------------------------------------------------------------
static const char kConstrainedModule[] = R"mlir(
module attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.preferred_backend = "coreml",
  target.allowed_backends = ["coreml", "metal", "cpu"],
  target.supported_precisions = ["fp16", "int8"]
} {
  func.func @decode_constrained(%token: tensor<1xi32>) -> tensor<1x768xf16> {
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

// ---------------------------------------------------------------------------
// Module 3: Apple-style target with paged_kv_compatible_backends=["metal"].
//
//   @prefill_apple  -- dynamic, producer role -> paged KV
//     paged_kv_compatible_backends=["metal"]: coreml/cpu filtered out.
//     preferred "coreml" was in original allowed but removed by KV filter.
//     EP: kv_layout_constraint, primary="metal", fallback=[]
// ---------------------------------------------------------------------------
static const char kApplePagedKVModule[] = R"mlir(
module attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.preferred_backend = "coreml",
  target.allowed_backends = ["coreml", "metal", "cpu"],
  target.supported_precisions = ["fp16"],
  target.paged_kv_compatible_backends = ["metal"]
} {
  func.func @prefill_apple(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
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

static mlir::OwningOpRef<mlir::ModuleOp>
runFourPassPipeline(const char *src, mlir::MLIRContext &ctx) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
  assert(module && "MLIR parse failed");
  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createServingPhaseAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createKVLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createReplayEligibilityPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createExecutionProviderPlanningPass());
  assert(pm.run(module.get()).succeeded() && "PassManager run failed");
  return module;
}

int main() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  auto module = runFourPassPipeline(kTestModule, ctx);

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

  // All 4 passes contributed attrs.
  assert(fp_prefill.source_passes.size() == 4
         && fp_prefill.source_passes[0] == "serving-phase-analysis"
         && fp_prefill.source_passes[1] == "kv-layout-planning"
         && fp_prefill.source_passes[2] == "replay-eligibility"
         && fp_prefill.source_passes[3] == "execution-provider-planning"
         && "source_passes mismatch");

  // EP: paged KV, no allowed_backends -> constraint_conflict, cpu fallback.
  assert(fp_prefill.backend_execution_plan.primary_backend == "cpu"
         && "EP primary should be cpu for constraint_conflict");
  assert(fp_prefill.backend_execution_plan.decision_source == "constraint_conflict"
         && "EP decision_source mismatch");
  assert(fp_prefill.backend_execution_plan.required_precision == "fp16"
         && "EP required_precision mismatch");
  assert(fp_prefill.backend_execution_plan.required_kv_layout == "paged"
         && "EP required_kv_layout mismatch");
  assert(fp_prefill.backend_execution_plan.requires_replay == false
         && "EP requires_replay should be false for prefill");
  assert(fp_prefill.backend_execution_plan.fallback_chain.empty()
         && "EP fallback_chain should be empty for constraint_conflict");

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

  // EP: contiguous KV, no allowed_backends -> default_policy, cpu.
  assert(fp_decode.backend_execution_plan.primary_backend == "cpu"
         && "EP primary should be cpu for default_policy");
  assert(fp_decode.backend_execution_plan.decision_source == "default_policy"
         && "EP decision_source mismatch");
  assert(fp_decode.backend_execution_plan.required_kv_layout == "contiguous"
         && "EP required_kv_layout mismatch");
  assert(fp_decode.backend_execution_plan.requires_replay == true
         && "EP requires_replay should be true for eligible static decode");

  // -----------------------------------------------------------------------
  // Module 2: constrained (target_preferred -> coreml, static decode).
  // -----------------------------------------------------------------------
  std::puts("  [PASS] unconstrained module");

  auto module2 = runFourPassPipeline(kConstrainedModule, ctx);
  mlir::hir::ServingExecutionPlan plan2 =
      mlir::hir::ServingExecutionPlanBuilder::build(module2.get());

  assert(plan2.function_plans.size() == 1 && "expected 1 function plan");

  const mlir::hir::FunctionExecutionPlan &fp_c = plan2.function_plans[0];
  assert(fp_c.function_name == "decode_constrained" && "function_name mismatch");

  // EP: target_preferred -> coreml; fallback = [metal, cpu].
  assert(fp_c.backend_execution_plan.primary_backend == "coreml"
         && "EP primary should be coreml (target_preferred)");
  assert(fp_c.backend_execution_plan.decision_source == "target_preferred"
         && "EP decision_source mismatch");
  assert(fp_c.backend_execution_plan.required_precision == "fp16"
         && "EP required_precision should be first of supported_precisions");
  assert(fp_c.backend_execution_plan.required_kv_layout == "contiguous"
         && "EP required_kv_layout mismatch");
  assert(fp_c.backend_execution_plan.requires_replay == true
         && "EP requires_replay should be true for eligible static decode");
  assert(fp_c.backend_execution_plan.fallback_chain.size() == 2
         && fp_c.backend_execution_plan.fallback_chain[0] == "metal"
         && fp_c.backend_execution_plan.fallback_chain[1] == "cpu"
         && "EP fallback_chain mismatch");

  // source_passes includes execution-provider-planning.
  bool hasEP = false;
  for (const auto &p : fp_c.source_passes)
    if (p == "execution-provider-planning") { hasEP = true; break; }
  assert(hasEP && "source_passes should include execution-provider-planning");

  std::puts("  [PASS] constrained module (target_preferred)");

  // -----------------------------------------------------------------------
  // Module 3: Apple target with paged_kv_compatible_backends=["metal"].
  // Prefill (paged KV) with coreml preferred but metal paged-KV-capable.
  // Expected: kv_layout_constraint, primary=metal.
  // This validates that paged-KV compatibility is profile-driven, not
  // hardcoded as cuda/vllm in the compiler.
  // -----------------------------------------------------------------------
  auto module3 = runFourPassPipeline(kApplePagedKVModule, ctx);
  mlir::hir::ServingExecutionPlan plan3 =
      mlir::hir::ServingExecutionPlanBuilder::build(module3.get());

  assert(plan3.function_plans.size() == 1 && "expected 1 function plan");

  const mlir::hir::FunctionExecutionPlan &fp_apple = plan3.function_plans[0];
  assert(fp_apple.function_name == "prefill_apple" && "function_name mismatch");
  assert(fp_apple.serving_phase == mlir::hir::ServingPhase::Prefill
         && "serving_phase should be Prefill");

  // KV: producer -> paged.
  assert(fp_apple.kv_plan.layout == mlir::hir::KVLayout::Paged
         && "kv_layout should be Paged for producer role");

  // EP: paged KV filter keeps only "metal" (from paged_kv_compatible_backends).
  // preferred "coreml" was in allowed but removed by filter -> kv_layout_constraint.
  assert(fp_apple.backend_execution_plan.primary_backend == "metal"
         && "EP primary should be metal (only paged-KV-compatible backend)");
  assert(fp_apple.backend_execution_plan.decision_source == "kv_layout_constraint"
         && "EP decision_source should be kv_layout_constraint");
  assert(fp_apple.backend_execution_plan.fallback_chain.empty()
         && "EP fallback_chain should be empty (metal is the only paged-KV backend)");
  assert(fp_apple.backend_execution_plan.required_kv_layout == "paged"
         && "EP required_kv_layout should be paged");
  assert(fp_apple.backend_execution_plan.required_precision == "fp16"
         && "EP required_precision should be fp16");
  assert(fp_apple.backend_execution_plan.requires_replay == false
         && "EP requires_replay should be false for prefill");

  std::puts("  [PASS] Apple paged-KV target (kv_layout_constraint/metal)");
  std::puts("ServingExecutionPlanBuilderTest: PASS");
  return 0;
}
