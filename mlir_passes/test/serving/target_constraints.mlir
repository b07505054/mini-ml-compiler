// FileCheck test for target.* module attribute constraints consumed by
// KVLayoutPlanningPass and ReplayEligibilityPass.
//
// These attrs are produced by a TargetLowering step (outside compiler core)
// that converts a TargetDeviceProfile into MLIR module attributes.
// The passes read them as optional constraints; absent means unconstrained.
//
// Run with the HIR plugin loaded:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(serving-optimization-pipeline)' \
//   | FileCheck %s
//
// Target constraints applied:
//   target.profile_id          = "apple-a17pro-mobile"
//   target.memory_budget_mb    = 5.0  (threshold = budget * 0.4 = 2.0 MB)
//   target.static_shape_support = false
//
// KV estimate for these functions:
//   num_layers=12, hidden_size=768, seq_tokens=128+64=192, dtype=fp16 (2B)
//   kv_mb = 12 * 2 * 768 * 2 * 192 / 1048576 ≈ 6.75 MB
//   6.75 > 2.0  →  memory_budget_constrained  →  kv.layout = "contiguous"
//
// @prefill: producer role would normally yield kv.layout="paged".
//   Budget constraint overrides to "contiguous" + kv.layout_reason.
//   Prefill is already replay-ineligible; target override also applied.
//
// @decode_static: static-shape decode would normally yield replay.eligible=true.
//   target.static_shape_support=false overrides to eligible=false + override_reason.

// -- @prefill ----------------------------------------------------------------
// Budget constraint forces contiguous despite producer (paged) role.
// No target.allowed_backends -> EP defaults to cpu, decision_source=default_policy.
// kv.layout="contiguous" (budget override) -> KV filter not applied.
// CHECK-LABEL: func.func @prefill
// CHECK-SAME:  execution_provider.decision_source = "default_policy"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cpu"
// CHECK-SAME:  execution_provider.required_kv_layout = "contiguous"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = false
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"
// CHECK-SAME:  kv.byte_estimate_mb
// CHECK-SAME:  kv.layout = "contiguous"
// CHECK-SAME:  kv.layout_reason = "memory_budget_constrained"
// CHECK-SAME:  kv.truth_boundary = "static_formula_estimate_not_measured_memory"
// CHECK-SAME:  replay.cuda_graph_bucket = ""
// CHECK-SAME:  replay.eligible = false
// CHECK-SAME:  replay.override_reason = "target_static_shape_unsupported"
// CHECK-SAME:  replay.truth_boundary = "static_shape_replay_eligibility_not_cuda_graph_capture"

// -- @decode_static ----------------------------------------------------------
// Static decode would be replay-eligible without the target constraint.
// target.static_shape_support=false overrides to ineligible.
// CHECK-LABEL: func.func @decode_static
// CHECK-SAME:  execution_provider.decision_source = "default_policy"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cpu"
// CHECK-SAME:  execution_provider.required_kv_layout = "contiguous"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = false
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"
// CHECK-SAME:  kv.layout = "contiguous"
// CHECK-SAME:  kv.layout_reason = "memory_budget_constrained"
// CHECK-SAME:  kv.truth_boundary = "static_formula_estimate_not_measured_memory"
// CHECK-SAME:  replay.cuda_graph_bucket = ""
// CHECK-SAME:  replay.eligible = false
// CHECK-SAME:  replay.override_reason = "target_static_shape_unsupported"
// CHECK-SAME:  replay.truth_boundary = "static_shape_replay_eligibility_not_cuda_graph_capture"

module attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.memory_budget_mb = 5.000000e+00 : f64,
  target.profile_id = "apple-a17pro-mobile",
  target.static_shape_support = false
} {
  // Producer with dynamic sequence length.
  func.func @prefill(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }

  // Consumer with static shapes: would be replay-eligible without target constraint.
  func.func @decode_static(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.phase = "decode",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}
