// Raw Qwen 0.5B attention graph for frontend normalization (Path B).
//
// Truth boundary: frontend_pattern_recognition_on_simplified_llm_graph_not_full_stablehlo_import
//
// These are simplified pseudo-LLM ops that demonstrate graph-level pattern
// recognition. This is NOT a full StableHLO/Torch/ONNX graph import.
// No weights are imported; operator lowering is not performed; shapes are
// not propagated across projections.
//
// Pipeline:
//   mlir-opt --load-pass-plugin=<plugin> \
//     --pass-pipeline='builtin.module(llm-frontend-normalization)'
//   -> canonical llm.attention_prefill / llm.attention_decode
//   -> compile-for-target --profile apple_a17pro_mobile.json
//   -> serving_execution_plan JSON
//
// See: tools/run_mlir_pass_tests.sh (llm-frontend-normalization test)
//      artifacts/apple_demo/raw_qwen_frontend_serving_execution_plan_iphone.json

module attributes {
  llm.model = "qwen2.5-0.5b",
  llm.num_layers = 24 : i64,
  llm.hidden_size = 896 : i64,
  llm.num_attention_heads = 14 : i64,
  llm.num_key_value_heads = 2 : i64,
  llm.intermediate_size = 4864 : i64,
  llm.vocab_size = 151936 : i64,
  frontend.truth_boundary = "frontend_pattern_recognition_on_simplified_llm_graph_not_full_stablehlo_import"
} {

  // Prefill phase: q/k/v projection + attention + kv_cache_write signals producer.
  // head_dim = 64 (896 / 14); kv_proj_dim = 128 (2 * 64, GQA 14q:2kv).
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

  // Decode phase: same projection pattern + kv_cache_read signals consumer.
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
