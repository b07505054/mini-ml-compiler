// FileCheck test for KVLayoutPlanningPass.
//
// Run with the HIR plugin loaded:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(kv-layout-planning)' \
//   | FileCheck %s
//
// KV byte estimate formula (fp16, 2 bytes/element):
//   kv_mb = num_layers * 2 * hidden_size * 2 * (prompt + output) / (1024 * 1024)
// With num_layers=12, hidden_size=768, prompt=128, output=64:
//   kv_mb = 12 * 2 * 768 * 2 * 192 / 1048576 ≈ 6.75

// Prefill function has kv_cache.role="producer" -> paged layout.
// CHECK-LABEL: func.func @prefill
// CHECK-SAME:  kv.byte_estimate_mb
// CHECK-SAME:  kv.layout = "paged"
// CHECK-SAME:  kv.truth_boundary = "static_formula_estimate_not_measured_memory"

// Decode function has kv_cache.role="consumer" -> contiguous layout.
// CHECK-LABEL: func.func @decode
// CHECK-SAME:  kv.byte_estimate_mb
// CHECK-SAME:  kv.layout = "contiguous"
// CHECK-SAME:  kv.truth_boundary = "static_formula_estimate_not_measured_memory"

module attributes {
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64
} {
  func.func @prefill(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }

  func.func @decode(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}
