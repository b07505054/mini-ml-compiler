// FileCheck test for llm-frontend-normalization pass.
//
// Run via tools/run_mlir_pass_tests.sh, or directly:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(llm-frontend-normalization)' \
//   | FileCheck %s
//
// Verifies Path B IR contract:
//   raw pseudo-LLM attention graph  ->  canonical llm.attention_prefill/decode
//
// The canonical output is structurally identical to qwen-to-serving-mlir output
// so the serving-optimization-pipeline cannot distinguish Path A from Path B.

// ---------------------------------------------------------------------------
// Prefill function: has llm.kv_cache_write -> llm.attention_prefill expected
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @raw_qwen_prefill_graph
// Raw ops must be completely absent from the normalized body.
// CHECK-NOT:   "llm.q_proj"
// CHECK-NOT:   "llm.k_proj"
// CHECK-NOT:   "llm.v_proj"
// CHECK-NOT:   "llm.attention_scores"
// CHECK-NOT:   "llm.kv_cache_write"
// Canonical prefill op must be present with required attrs.
// Attrs are printed alphabetically by MLIR; CHECK-SAME must match in that order.
// CHECK:       "llm.attention_prefill"
// CHECK-SAME:  frontend.source = "llm_graph_pattern"
// CHECK-SAME:  frontend.truth_boundary = "attention_pattern_simplified_not_full_transformer_lowering"
// CHECK-SAME:  kv_cache.role = "producer"
// CHECK-SAME:  serving.phase = "prefill"
// Return op must preserve the original result type.
// CHECK:       return
// CHECK-SAME:  tensor<?x896xf16>

// ---------------------------------------------------------------------------
// Decode function: has llm.kv_cache_read -> llm.attention_decode expected
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @raw_qwen_decode_graph
// Raw ops must be completely absent.
// CHECK-NOT:   "llm.q_proj"
// CHECK-NOT:   "llm.k_proj"
// CHECK-NOT:   "llm.v_proj"
// CHECK-NOT:   "llm.attention_scores"
// CHECK-NOT:   "llm.kv_cache_read"
// Canonical decode op must be present.
// CHECK:       "llm.attention_decode"
// CHECK-SAME:  frontend.source = "llm_graph_pattern"
// CHECK-SAME:  frontend.truth_boundary = "attention_pattern_simplified_not_full_transformer_lowering"
// CHECK-SAME:  kv_cache.role = "consumer"
// CHECK-SAME:  serving.phase = "decode"
// Return op must preserve the original result type.
// CHECK:       return
// CHECK-SAME:  tensor<1x896xf16>

module attributes {
  llm.model = "qwen2.5-0.5b",
  llm.num_layers = 24 : i64,
  llm.hidden_size = 896 : i64,
  llm.num_attention_heads = 14 : i64,
  llm.num_key_value_heads = 2 : i64,
  frontend.truth_boundary = "frontend_pattern_recognition_on_simplified_llm_graph_not_full_stablehlo_import"
} {

  func.func @raw_qwen_prefill_graph(%hidden: tensor<?x896xf16>) -> tensor<?x896xf16> {
    %q = "llm.q_proj"(%hidden) : (tensor<?x896xf16>) -> tensor<?x896xf16>
    %k = "llm.k_proj"(%hidden) : (tensor<?x896xf16>) -> tensor<?x128xf16>
    %v = "llm.v_proj"(%hidden) : (tensor<?x896xf16>) -> tensor<?x128xf16>
    %scores = "llm.attention_scores"(%q, %k)
        : (tensor<?x896xf16>, tensor<?x128xf16>) -> tensor<?x?xf16>
    %probs = "llm.softmax"(%scores) : (tensor<?x?xf16>) -> tensor<?x?xf16>
    %attn = "llm.attention_output"(%probs, %v)
        : (tensor<?x?xf16>, tensor<?x128xf16>) -> tensor<?x896xf16>
    "llm.kv_cache_write"(%k, %v) : (tensor<?x128xf16>, tensor<?x128xf16>) -> ()
    return %attn : tensor<?x896xf16>
  }

  func.func @raw_qwen_decode_graph(%hidden: tensor<1x896xf16>) -> tensor<1x896xf16> {
    %q = "llm.q_proj"(%hidden) : (tensor<1x896xf16>) -> tensor<1x896xf16>
    %k = "llm.k_proj"(%hidden) : (tensor<1x896xf16>) -> tensor<1x128xf16>
    %v = "llm.v_proj"(%hidden) : (tensor<1x896xf16>) -> tensor<1x128xf16>
    %ck, %cv = "llm.kv_cache_read"(%q)
        : (tensor<1x896xf16>) -> (tensor<?x128xf16>, tensor<?x128xf16>)
    %scores = "llm.attention_scores"(%q, %ck)
        : (tensor<1x896xf16>, tensor<?x128xf16>) -> tensor<1x?xf16>
    %probs = "llm.softmax"(%scores) : (tensor<1x?xf16>) -> tensor<1x?xf16>
    %attn = "llm.attention_output"(%probs, %cv)
        : (tensor<1x?xf16>, tensor<?x128xf16>) -> tensor<1x896xf16>
    return %attn : tensor<1x896xf16>
  }
}
