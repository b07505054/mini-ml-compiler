module attributes {
  attention.cpu.artifact_ref = "native/libattention_fp32.so",
  attention.cpu.artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  attention.cpu.artifact_version = "hir.cpu_attention.v1",
  attention.cpu.kv_capacity_tokens = 64 : i64,
  attention.cpu.requested_kv_layout = "paged",
  attention.cpu.page_tokens = 8 : i64,
  attention.cpu.num_physical_pages = 8 : i64
} {
  func.func @paged_prefill(%x: tensor<1x2x7x8xf32>) -> tensor<1x2x7x8xf32> {
    %q = "llm.q_proj"(%x) : (tensor<1x2x7x8xf32>) -> tensor<1x2x7x8xf32>
    %k = "llm.k_proj"(%x) : (tensor<1x2x7x8xf32>) -> tensor<1x2x7x8xf32>
    %v = "llm.v_proj"(%x) : (tensor<1x2x7x8xf32>) -> tensor<1x2x7x8xf32>
    %s = "llm.attention_scores"(%q,%k) {attention.causal=true,attention.key_transposed=true,attention.scale=0.3535533905932738:f64} : (tensor<1x2x7x8xf32>,tensor<1x2x7x8xf32>)->tensor<1x2x7x7xf32>
    %p = "llm.softmax"(%s) {attention.softmax_axis=-1:i64} : (tensor<1x2x7x7xf32>)->tensor<1x2x7x7xf32>
    %o = "llm.attention_output"(%p,%v) : (tensor<1x2x7x7xf32>,tensor<1x2x7x8xf32>)->tensor<1x2x7x8xf32>
    "llm.kv_cache_write"(%k,%v) : (tensor<1x2x7x8xf32>,tensor<1x2x7x8xf32>)->()
    return %o : tensor<1x2x7x8xf32>
  }
  func.func @paged_decode(%q0:tensor<1x2x1x8xf32>,%token:tensor<1x2x1x8xf32>)->tensor<1x2x1x8xf32>{
    %q="llm.q_proj"(%q0):(tensor<1x2x1x8xf32>)->tensor<1x2x1x8xf32>
    %k="llm.k_proj"(%token):(tensor<1x2x1x8xf32>)->tensor<1x2x1x8xf32>
    %v="llm.v_proj"(%token):(tensor<1x2x1x8xf32>)->tensor<1x2x1x8xf32>
    %ck,%cv="llm.kv_cache_read"(%k,%v):(tensor<1x2x1x8xf32>,tensor<1x2x1x8xf32>)->(tensor<1x2x8x8xf32>,tensor<1x2x8x8xf32>)
    %s="llm.attention_scores"(%q,%ck){attention.causal=true,attention.key_transposed=true,attention.scale=0.3535533905932738:f64}:(tensor<1x2x1x8xf32>,tensor<1x2x8x8xf32>)->tensor<1x2x1x8xf32>
    %p="llm.softmax"(%s){attention.softmax_axis=-1:i64}:(tensor<1x2x1x8xf32>)->tensor<1x2x1x8xf32>
    %o="llm.attention_output"(%p,%cv):(tensor<1x2x1x8xf32>,tensor<1x2x8x8xf32>)->tensor<1x2x1x8xf32>
    return %o:tensor<1x2x1x8xf32>
  }
}
