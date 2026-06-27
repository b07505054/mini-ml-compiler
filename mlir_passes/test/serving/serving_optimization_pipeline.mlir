// FileCheck test for the serving-optimization-pipeline named pipeline.
//
// Run with the HIR plugin loaded:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(serving-optimization-pipeline)' \
//   | FileCheck %s
//
// The named pipeline composes:
//   1. serving-phase-analysis         -> serving.* attrs
//   2. kv-layout-planning             -> kv.* attrs
//   3. replay-eligibility             -> replay.* attrs
//   4. execution-provider-planning    -> execution_provider.* attrs
//
// Attribute order on func.func output (MLIR dict, alphabetical):
//   execution_provider.decision_source, execution_provider.fallback_chain,
//   execution_provider.primary, execution_provider.required_kv_layout,
//   execution_provider.required_precision, execution_provider.requires_replay,
//   execution_provider.truth_boundary,
//   kv.byte_estimate_mb, kv.layout, kv.truth_boundary,
//   replay.cuda_graph_bucket, replay.eligible, replay.truth_boundary,
//   serving.colocated_total_ms, serving.confidence, serving.cost_source,
//   serving.decision_margin_ms, serving.decision_margin_pct,
//   serving.pd_split_total_ms, serving.policy, serving.truth_boundary
//
// No target.* module attrs -> default_policy for decode, constraint_conflict
// for prefill (paged KV default candidate "cpu" is not paged-compatible).

// Prefill: colocated policy, paged KV layout, not replay-eligible.
// Execution provider: constraint_conflict (cpu not paged-KV-compatible).
// CHECK-LABEL: func.func @prefill
// CHECK-SAME:  execution_provider.decision_source = "constraint_conflict"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cpu"
// CHECK-SAME:  execution_provider.required_kv_layout = "paged"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = false
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"
// CHECK-SAME:  kv.byte_estimate_mb
// CHECK-SAME:  kv.layout = "paged"
// CHECK-SAME:  kv.truth_boundary = "static_formula_estimate_not_measured_memory"
// CHECK-SAME:  replay.cuda_graph_bucket = ""
// CHECK-SAME:  replay.eligible = false
// CHECK-SAME:  replay.truth_boundary = "static_shape_replay_eligibility_not_cuda_graph_capture"
// CHECK-SAME:  serving.confidence = "low"
// CHECK-SAME:  serving.cost_source = "formula_synthetic"
// CHECK-SAME:  serving.policy = "colocated"
// CHECK-SAME:  serving.truth_boundary = "estimated_cost_not_measured_latency"

// Static decode: contiguous KV layout, replay-eligible.
// Execution provider: default_policy (no target constraints, contiguous KV).
// CHECK-LABEL: func.func @decode
// CHECK-SAME:  execution_provider.decision_source = "default_policy"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cpu"
// CHECK-SAME:  execution_provider.required_kv_layout = "contiguous"
// CHECK-SAME:  execution_provider.requires_replay = true
// CHECK-SAME:  kv.layout = "contiguous"
// CHECK-SAME:  replay.cuda_graph_bucket = "decode_static"
// CHECK-SAME:  replay.eligible = true

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
