module attributes {
  attention.cpu.artifact_ref = "native/libattention_fp32.so",
  attention.cpu.artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  attention.cpu.artifact_version = "hir.cpu_attention.v1",
  attention.cpu.kv_capacity_tokens = 512 : i64,
  llm.num_attention_heads = 2 : i64,
  llm.num_key_value_heads = 2 : i64,
  llm.num_layers = 1 : i64,
  llm.hidden_size = 16 : i64
} {
  func.func @attention_prefill(%x: tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32> {
    %q = "llm.q_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %k = "llm.k_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %v = "llm.v_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %s = "llm.attention_scores"(%q, %k) {attention.causal = true, attention.key_transposed = true, attention.scale = 0.3535533905932738 : f64} : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x4xf32>
    %p = "llm.softmax"(%s) {attention.softmax_axis = -1 : i64} : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %o = "llm.attention_output"(%p, %v) : (tensor<1x2x4x4xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    "llm.kv_cache_write"(%k, %v) : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> ()
    return %o : tensor<1x2x4x8xf32>
  }
  func.func @attention_decode(%qinput: tensor<1x2x1x8xf32>, %token: tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32> {
    %q = "llm.q_proj"(%qinput) : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    %k = "llm.k_proj"(%token) : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    %v = "llm.v_proj"(%token) : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
    %ck, %cv = "llm.kv_cache_read"(%k, %v) : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> (tensor<1x2x5x8xf32>, tensor<1x2x5x8xf32>)
    %s = "llm.attention_scores"(%q, %ck) {attention.causal = true, attention.key_transposed = true, attention.scale = 0.3535533905932738 : f64} : (tensor<1x2x1x8xf32>, tensor<1x2x5x8xf32>) -> tensor<1x2x1x5xf32>
    %p = "llm.softmax"(%s) {attention.softmax_axis = -1 : i64} : (tensor<1x2x1x5xf32>) -> tensor<1x2x1x5xf32>
    %o = "llm.attention_output"(%p, %cv) : (tensor<1x2x1x5xf32>, tensor<1x2x5x8xf32>) -> tensor<1x2x1x8xf32>
    return %o : tensor<1x2x1x8xf32>
  }
}
