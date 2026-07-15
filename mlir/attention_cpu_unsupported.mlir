// Each graph deliberately violates the narrow simplified-attention contract.
module {
  func.func @incomplete_qkv(%x: tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32> {
    %q = "llm.q_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    return %q : tensor<1x2x4x8xf32>
  }
  func.func @missing_softmax(%x: tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32> {
    %q = "llm.q_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %k = "llm.k_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %v = "llm.v_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %s = "llm.attention_scores"(%q, %k) {attention.causal = true, attention.key_transposed = true, attention.scale = 0.3535533905932738 : f64} : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x4xf32>
    %o = "llm.attention_output"(%s, %v) : (tensor<1x2x4x4xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    return %o : tensor<1x2x4x8xf32>
  }
  func.func @missing_causal(%x: tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32> {
    %q = "llm.q_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %k = "llm.k_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %v = "llm.v_proj"(%x) : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %s = "llm.attention_scores"(%q, %k) {attention.key_transposed = true, attention.scale = 0.3535533905932738 : f64} : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x4xf32>
    %p = "llm.softmax"(%s) {attention.softmax_axis = -1 : i64} : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %o = "llm.attention_output"(%p, %v) : (tensor<1x2x4x4xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
    return %o : tensor<1x2x4x8xf32>
  }
  func.func @unsupported_f16(%x: tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf16> {
    %q = "llm.q_proj"(%x) : (tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf16>
    %k = "llm.k_proj"(%x) : (tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf16>
    %v = "llm.v_proj"(%x) : (tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf16>
    %s = "llm.attention_scores"(%q, %k) {attention.causal = true, attention.key_transposed = true, attention.scale = 0.3535533905932738 : f64} : (tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> tensor<1x2x4x4xf16>
    %p = "llm.softmax"(%s) {attention.softmax_axis = -1 : i64} : (tensor<1x2x4x4xf16>) -> tensor<1x2x4x4xf16>
    %o = "llm.attention_output"(%p, %v) : (tensor<1x2x4x4xf16>, tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf16>
    return %o : tensor<1x2x4x8xf16>
  }
  func.func @unsupported_gqa(%q0: tensor<1x4x1x8xf32>, %kv: tensor<1x2x7x8xf32>) -> tensor<1x4x1x8xf32> {
    %q = "llm.q_proj"(%q0) : (tensor<1x4x1x8xf32>) -> tensor<1x4x1x8xf32>
    %k = "llm.k_proj"(%kv) : (tensor<1x2x7x8xf32>) -> tensor<1x2x7x8xf32>
    %v = "llm.v_proj"(%kv) : (tensor<1x2x7x8xf32>) -> tensor<1x2x7x8xf32>
    %s = "llm.attention_scores"(%q, %k) {attention.causal = true, attention.key_transposed = true, attention.scale = 0.3535533905932738 : f64} : (tensor<1x4x1x8xf32>, tensor<1x2x7x8xf32>) -> tensor<1x4x1x7xf32>
    %p = "llm.softmax"(%s) {attention.softmax_axis = -1 : i64} : (tensor<1x4x1x7xf32>) -> tensor<1x4x1x7xf32>
    %o = "llm.attention_output"(%p, %v) : (tensor<1x4x1x7xf32>, tensor<1x2x7x8xf32>) -> tensor<1x4x1x8xf32>
    return %o : tensor<1x4x1x8xf32>
  }
  func.func @dynamic_head_dim(%x: tensor<1x2x4x?xf32>) -> tensor<1x2x4x?xf32> {
    %q = "llm.q_proj"(%x) : (tensor<1x2x4x?xf32>) -> tensor<1x2x4x?xf32>
    %k = "llm.k_proj"(%x) : (tensor<1x2x4x?xf32>) -> tensor<1x2x4x?xf32>
    %v = "llm.v_proj"(%x) : (tensor<1x2x4x?xf32>) -> tensor<1x2x4x?xf32>
    %s = "llm.attention_scores"(%q, %k) {attention.causal = true, attention.key_transposed = true, attention.scale = 1.0 : f64} : (tensor<1x2x4x?xf32>, tensor<1x2x4x?xf32>) -> tensor<1x2x4x4xf32>
    %p = "llm.softmax"(%s) {attention.softmax_axis = -1 : i64} : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %o = "llm.attention_output"(%p, %v) : (tensor<1x2x4x4xf32>, tensor<1x2x4x?xf32>) -> tensor<1x2x4x?xf32>
    return %o : tensor<1x2x4x?xf32>
  }
}
