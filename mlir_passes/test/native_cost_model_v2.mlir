// CHECK: linalg.matmul {fusion.candidate = "matmul_bias_relu"
// CHECK-SAME: native.cost_model.candidates = [
// CHECK-SAME: candidate_id = "unfused_scalar"
// CHECK-SAME: intermediate_read_count = 2
// CHECK-SAME: candidate_id = "unfused_whole_shape_vector"
// CHECK-SAME: schedule_mode = "whole_shape_vector"
// CHECK-SAME: candidate_id = "fused_scalar"
// CHECK-SAME: intermediate_read_count = 0
// CHECK-SAME: candidate_id = "fused_whole_shape_vector"
// CHECK-SAME: interaction_correction_ns
// CHECK-SAME: selected = true
// CHECK-SAME: total_ns
// CHECK-SAME: candidate_id = "fused_tiled_vector"
// CHECK-SAME: edge_tile_count = 0
// CHECK-SAME: estimated_code_size_bytes = 4096
// CHECK-SAME: native.cost_model.selected_candidate = "fused_whole_shape_vector"
// CHECK-SAME: native.cost_model.version = "static_cost_model_v2"

module attributes {
  llm.dtype = "fp32",
  target.supported_precisions = ["fp32"],
  target.native_cost_v2.supports_vector = true,
  target.native_cost_v2.effective_scalar_ops_per_ns = 0.52 : f64,
  target.native_cost_v2.effective_vector_ops_per_ns = 14.5 : f64,
  target.native_cost_v2.vector_utilization = 0.82 : f64,
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
