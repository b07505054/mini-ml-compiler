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

// ---------------------------------------------------------------------------
// Module 4: target profile cost attrs → cost_source = "target_profile_formula_estimate".
//
// Exercises the target-driven cost path in ServingPhaseAnalysisPass.
// Uses values that differ from the defaults to confirm they are applied.
// ---------------------------------------------------------------------------
static const char kProfileCostModule[] = R"mlir(
module attributes {
  llm.model = "profile-cost-model",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.prefill_ms_per_token = 1.000000e-01 : f64,
  target.decode_ms_per_token = 1.500000e-01 : f64,
  target.pd_bandwidth_mb_per_ms = 2.000000e+01 : f64
} {
  func.func @prefill_profile(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
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

// ---------------------------------------------------------------------------
// Module 5: pre-annotated lowering decisions covering all 5 decision values.
//
// Does NOT run any pipeline passes. serving.policy pre-annotated so the
// builder includes the function. lowering.* and kernel.* attrs pre-annotated
// to cover all 5 LoweringDecisionPlanningPass outcomes:
//   op_0  direct_lower
//   op_1  rewrite_then_lower
//   op_2  dequant_then_lower   (unreachable through compile-for-target due to
//                               weight_only_int8 fp16-activation invariant)
//   op_3  fallback_backend
//   op_4  unsupported
// ---------------------------------------------------------------------------
static const char kLoweringDecisionModule[] = R"mlir(
module attributes {
  llm.model = "lowering-decision-builder-test",
  llm.num_layers = 1 : i64,
  llm.hidden_size = 64 : i64
} {
  func.func @prefill(%x: tensor<1x64xf16>) -> tensor<1x64xf16>
    attributes {
      serving.policy = "colocated",
      serving.phase  = "prefill"
    } {
    %a = "hir.matmul"(%x, %x) {
      kernel.backend         = "cpu",
      kernel.exists          = true,
      kernel.fallback_backend = "cpu_simd",
      kernel.library         = "cpu_blas",
      kernel.lowering_status = "lowerable",
      kernel.name            = "cpu_matmul_fp16",
      lowering.decision      = "direct_lower",
      lowering.reason        = "kernel_available_for_op_dtype_layout",
      lowering.required_rewrite = "",
      lowering.requires_cast = false,
      lowering.requires_dequant = false,
      lowering.requires_layout_transform = false,
      lowering.target_backend = "cpu",
      lowering.target_kernel  = "cpu_matmul_fp16",
      lowering.target_kernel_library = "cpu_blas",
      lowering.truth_boundary = "lowering_decision_export_static_not_backend_codegen_verified"
    } : (tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
    %b = "hir.conv2d"(%a, %a) {
      kernel.backend         = "cpu",
      kernel.exists          = false,
      kernel.fallback_backend = "",
      kernel.library         = "cpu_blas",
      kernel.lowering_status = "rewrite_candidate",
      kernel.name            = "cpu_conv2d_fp32",
      lowering.decision      = "rewrite_then_lower",
      lowering.reason        = "rewrite_then_lower_via_fp16_to_fp32_cast",
      lowering.required_rewrite = "fp16_to_fp32_cast",
      lowering.requires_cast = false,
      lowering.requires_dequant = false,
      lowering.requires_layout_transform = false,
      lowering.target_backend = "cpu",
      lowering.target_kernel  = "cpu_conv2d_fp32",
      lowering.target_kernel_library = "cpu_blas",
      lowering.truth_boundary = "lowering_decision_export_static_not_backend_codegen_verified"
    } : (tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
    %c = "hir.softmax"(%b) {
      kernel.backend         = "cpu",
      kernel.exists          = false,
      kernel.fallback_backend = "cpu_ref",
      kernel.library         = "",
      kernel.lowering_status = "fallback_required",
      kernel.name            = "",
      lowering.decision      = "dequant_then_lower",
      lowering.reason        = "dequant_required_before_fallback_to_cpu_ref",
      lowering.required_rewrite = "",
      lowering.requires_cast = false,
      lowering.requires_dequant = true,
      lowering.requires_layout_transform = false,
      lowering.target_backend = "cpu_ref",
      lowering.target_kernel  = "",
      lowering.target_kernel_library = "",
      lowering.truth_boundary = "lowering_decision_export_static_not_backend_codegen_verified"
    } : (tensor<1x64xf16>) -> tensor<1x64xf16>
    %d = "hir.gelu"(%c) {
      kernel.backend         = "cpu",
      kernel.exists          = false,
      kernel.fallback_backend = "cpu_ref",
      kernel.library         = "",
      kernel.lowering_status = "fallback_required",
      kernel.name            = "",
      lowering.decision      = "fallback_backend",
      lowering.reason        = "fallback_to_cpu_ref",
      lowering.required_rewrite = "",
      lowering.requires_cast = false,
      lowering.requires_dequant = false,
      lowering.requires_layout_transform = false,
      lowering.target_backend = "cpu_ref",
      lowering.target_kernel  = "",
      lowering.target_kernel_library = "",
      lowering.truth_boundary = "lowering_decision_export_static_not_backend_codegen_verified"
    } : (tensor<1x64xf16>) -> tensor<1x64xf16>
    %e = "hir.rms_norm"(%d) {
      kernel.backend         = "cpu",
      kernel.exists          = false,
      kernel.fallback_backend = "",
      kernel.library         = "",
      kernel.lowering_status = "unsupported",
      kernel.name            = "",
      lowering.decision      = "unsupported",
      lowering.reason        = "no_kernel_no_rewrite_no_fallback",
      lowering.required_rewrite = "",
      lowering.requires_cast = false,
      lowering.requires_dequant = false,
      lowering.requires_layout_transform = false,
      lowering.target_backend = "cpu",
      lowering.target_kernel  = "",
      lowering.target_kernel_library = "",
      lowering.truth_boundary = "lowering_decision_export_static_not_backend_codegen_verified"
    } : (tensor<1x64xf16>) -> tensor<1x64xf16>
    return %e : tensor<1x64xf16>
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

  // Quantization plan: no quantization-planning in pipeline; ServingPhaseAnalysisPass
  // emits quantization.effective_dtype="fp16" and quantization.dtype_bytes=2.0 as
  // defaults when quantization.plan_dtype is absent on the module.
  assert(fp_prefill.quantization_plan.plan_dtype == "fp16"
         && "quantization plan_dtype should default to fp16");
  assert(fp_prefill.quantization_plan.plan_source == "default_dtype"
         && "quantization plan_source should be default_dtype (no quantization-planning ran)");
  assert(fp_prefill.quantization_plan.dtype_bytes == 2.0
         && "quantization dtype_bytes should default to 2.0 (fp16)");
  assert(!fp_prefill.quantization_plan.truth_boundary.empty()
         && "quantization truth_boundary should be non-empty");

  // All 4 serving passes contributed attrs; quantization-planning was not in the
  // pipeline so source_passes should still be exactly 4 entries.
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

  // -----------------------------------------------------------------------
  // Module 4: target profile cost attrs drive cost_source.
  // cost_source must be "target_profile_formula_estimate" (not
  // "formula_synthetic") when target.prefill_ms_per_token etc. are present.
  // colocated_total_ms must reflect profile values (0.10/0.15 ≠ 0.08/0.12).
  // -----------------------------------------------------------------------
  auto module4 = runFourPassPipeline(kProfileCostModule, ctx);
  mlir::hir::ServingExecutionPlan plan4 =
      mlir::hir::ServingExecutionPlanBuilder::build(module4.get());

  assert(plan4.function_plans.size() == 1 && "expected 1 function plan");

  const mlir::hir::FunctionExecutionPlan &fp4 = plan4.function_plans[0];
  assert(fp4.function_name == "prefill_profile" && "function_name mismatch");
  assert(fp4.serving_phase == mlir::hir::ServingPhase::Prefill
         && "serving_phase should be Prefill");

  // cost_source must reflect the profile-driven path.
  assert(fp4.provenance.cost_source == "target_profile_formula_estimate"
         && "cost_source should be target_profile_formula_estimate");

  // colocated_total_ms = 0.10*128 + 0.15*64 + 0.02*128 = 24.96
  // Must differ from the default-constants result (20.48).
  assert(fp4.cost_summary.colocated_total_ms > 24.0 &&
         fp4.cost_summary.colocated_total_ms < 26.0
         && "colocated_total_ms should reflect profile cost values (~24.96)");

  // truth_boundary unchanged regardless of cost source.
  assert(fp4.provenance.truth_boundary == "estimated_cost_not_measured_latency"
         && "truth_boundary mismatch");

  std::puts("  [PASS] profile-driven cost model (target_profile_formula_estimate)");

  // -----------------------------------------------------------------------
  // Module 5: per_op_lowering_decisions — all 5 decision values.
  // No passes run; lowering.* and kernel.* attrs are pre-annotated.
  // Covers dequant_then_lower which is unreachable through compile-for-target.
  // -----------------------------------------------------------------------
  auto module5 =
      mlir::parseSourceString<mlir::ModuleOp>(kLoweringDecisionModule, &ctx);
  assert(module5 && "MLIR parse failed for lowering-decision module");

  mlir::hir::ServingExecutionPlan plan5 =
      mlir::hir::ServingExecutionPlanBuilder::build(module5.get());

  assert(plan5.function_plans.size() == 1 && "expected 1 function plan");
  const mlir::hir::FunctionExecutionPlan &fp5 = plan5.function_plans[0];

  assert(fp5.function_name == "prefill" && "function_name mismatch");
  assert(fp5.per_op_lowering_decisions.size() == 5
         && "expected 5 per_op_lowering_decisions");

  const auto &ld = fp5.per_op_lowering_decisions;

  // op_0: direct_lower (hir.matmul)
  assert(ld[0].op_type           == "hir.matmul"      && "op_0 op_type mismatch");
  assert(ld[0].lowering_decision == "direct_lower"    && "op_0 decision mismatch");
  assert(ld[0].kernel_exists     == true              && "op_0 kernel_exists mismatch");
  assert(ld[0].target_backend    == "cpu"             && "op_0 target_backend mismatch");
  assert(ld[0].target_kernel     == "cpu_matmul_fp16" && "op_0 target_kernel mismatch");
  assert(ld[0].requires_dequant  == false             && "op_0 requires_dequant mismatch");

  // op_1: rewrite_then_lower (hir.conv2d)
  assert(ld[1].op_type           == "hir.conv2d"         && "op_1 op_type mismatch");
  assert(ld[1].lowering_decision == "rewrite_then_lower" && "op_1 decision mismatch");
  assert(ld[1].required_rewrite  == "fp16_to_fp32_cast"  && "op_1 required_rewrite mismatch");
  assert(ld[1].kernel_exists     == false                && "op_1 kernel_exists mismatch");

  // op_2: dequant_then_lower (hir.softmax)
  assert(ld[2].op_type           == "hir.softmax"         && "op_2 op_type mismatch");
  assert(ld[2].lowering_decision == "dequant_then_lower"  && "op_2 decision mismatch");
  assert(ld[2].requires_dequant  == true                  && "op_2 requires_dequant mismatch");
  assert(ld[2].target_backend    == "cpu_ref"             && "op_2 target_backend mismatch");

  // op_3: fallback_backend (hir.gelu)
  assert(ld[3].op_type           == "hir.gelu"         && "op_3 op_type mismatch");
  assert(ld[3].lowering_decision == "fallback_backend" && "op_3 decision mismatch");
  assert(ld[3].requires_dequant  == false              && "op_3 requires_dequant mismatch");
  assert(ld[3].target_backend    == "cpu_ref"          && "op_3 target_backend mismatch");

  // op_4: unsupported (hir.rms_norm)
  assert(ld[4].op_type           == "hir.rms_norm" && "op_4 op_type mismatch");
  assert(ld[4].lowering_decision == "unsupported"  && "op_4 decision mismatch");
  assert(ld[4].target_kernel.empty()               && "op_4 target_kernel should be empty");

  // Truth boundary propagated to all entries.
  for (const auto &entry : ld)
    assert(entry.truth_boundary ==
               "lowering_decision_export_static_not_backend_codegen_verified"
           && "truth_boundary mismatch");

  // Pass tracked in source_passes.
  bool hasLoweringPass = false;
  for (const auto &p : fp5.source_passes)
    if (p == "lowering-decision-planning") { hasLoweringPass = true; break; }
  assert(hasLoweringPass && "source_passes should include lowering-decision-planning");

  std::puts("  [PASS] per_op_lowering_decisions (all 5 decision values incl. dequant_then_lower)");
  std::puts("ServingExecutionPlanBuilderTest: PASS");
  return 0;
}
