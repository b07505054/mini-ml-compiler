// ANALYTICAL: native.cost_model.mode = "analytical"
// ANALYTICAL: native.cost_model.version = "static_cost_model_v2"
// GBDT: native.cost_model.mode = "gbdt"
// GBDT: native.cost_model.version = "candidate_latency_gbdt_v1"
// HYBRID: candidate_id = "fused_tiled_vector_full_tiles"
// HYBRID-SAME: requires_full_k_tile = true
// HYBRID-SAME: requires_full_m_tile = true
// HYBRID-SAME: requires_full_n_tile = true
// HYBRID-SAME: tiling_kind = "tiled"
// HYBRID-SAME: vectorization_kind = "tiled_vector"
// HYBRID-SAME: vectorized_dimension = "multiple"
// HYBRID-SAME: zero_fill_bytes = 0
// HYBRID: native.cost_model.mode = "hybrid_
// HYBRID-SAME: native.cost_model.version = "candidate_latency_gbdt_v1"

module attributes {
  target.profile_id = "raspberry-pi5-cortex-a76-cpu",
  target.native_cost_v2.supports_vector = true,
  target.native_cost_v2.effective_scalar_ops_per_ns = 0.52 : f64,
  target.native_cost_v2.effective_vector_ops_per_ns = 14.5 : f64,
  target.native_cost_v2.effective_bandwidth_bytes_per_ns = 12.0 : f64
} {
  func.func @aligned(
      %lhs: tensor<16x32xf32>, %rhs: tensor<32x16xf32>,
      %bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %empty = tensor.empty() : tensor<16x16xf32>
    %mm = linalg.matmul
      ins(%lhs, %rhs : tensor<16x32xf32>, tensor<32x16xf32>)
      outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %add = linalg.map { arith.addf }
      ins(%mm, %bias : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>)
    %zero = arith.constant 0.0 : f32
    %relu = linalg.map
      ins(%add : tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>)
      (%x: f32) {
        %y = arith.maximumf %x, %zero : f32
        linalg.yield %y : f32
      }
    return %relu : tensor<16x16xf32>
  }
}
