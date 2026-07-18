// Real local Qwen2.5-0.5B attention contract: GQA 14 query heads / 2 KV heads.
module {
  func.func @qwen_prefill(
      %q: tensor<1x14x11x64xf32>,
      %k: tensor<1x2x11x64xf32>,
      %v: tensor<1x2x11x64xf32>) -> tensor<1x14x11x64xf32> {
    %out = hir.attention %q, %k, %v {
      alignment_bytes = 4 : i64,
      batch = 1 : i64,
      causal = true,
      context_length = 11 : i64,
      dtype = "fp32",
      head_dim = 64 : i64,
      input_layout = "bhsd_contiguous",
      num_kv_heads = 2 : i64,
      num_query_heads = 14 : i64,
      output_layout = "bhsd_contiguous",
      phase = "prefill",
      query_length = 11 : i64,
      truth_boundary = "real_qwen_gqa_execution_contract",
      workspace_bytes = 0 : i64
    } : (tensor<1x14x11x64xf32>, tensor<1x2x11x64xf32>,
         tensor<1x2x11x64xf32>) -> tensor<1x14x11x64xf32>
    return %out : tensor<1x14x11x64xf32>
  }
}
