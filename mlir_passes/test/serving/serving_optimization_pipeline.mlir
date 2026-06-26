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
//   1. serving-phase-analysis  -> serving.* attrs (policy, costs, confidence, provenance)
//   2. kv-layout-planning      -> kv.* attrs (layout, byte_estimate_mb, truth_boundary)
//
// Attribute order on func.func output (MLIR dict, alphabetical):
//   kv.byte_estimate_mb, kv.layout, kv.truth_boundary,
//   serving.colocated_total_ms, serving.confidence, serving.cost_source,
//   serving.decision_margin_ms, serving.decision_margin_pct,
//   serving.pd_split_total_ms, serving.policy, serving.truth_boundary

// CHECK-LABEL: func.func @prefill
// CHECK-SAME:  kv.byte_estimate_mb
// CHECK-SAME:  kv.layout = "paged"
// CHECK-SAME:  kv.truth_boundary = "static_formula_estimate_not_measured_memory"
// CHECK-SAME:  serving.confidence = "low"
// CHECK-SAME:  serving.cost_source = "formula_synthetic"
// CHECK-SAME:  serving.policy = "colocated"
// CHECK-SAME:  serving.truth_boundary = "estimated_cost_not_measured_latency"

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
