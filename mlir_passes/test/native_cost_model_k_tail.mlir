// RUN: mlir-opt %s --pass-pipeline='builtin.module(quantization-planning,matmul-bias-relu-fusion)' | FileCheck %s

// CHECK: linalg.matmul
// CHECK-SAME: candidate_kind = "whole_shape_vector_no_padding"
// CHECK-SAME: candidate_kind = "tiled_vector_materialized_tail"
// CHECK-SAME: candidate_kind = "tiled_vector_direct_cleanup"
// CHECK-SAME: k_tail_strategy = "direct_scalar_cleanup"
// CHECK-SAME: rejection_reason = "direct_scalar_k_cleanup_not_yet_production_lowered"
// CHECK-SAME: candidate_kind = "tiled_vector_direct_cleanup"
// CHECK-SAME: direct_tail_ns
// CHECK-SAME: k_remainder = 7
// CHECK-SAME: k_tail_strategy = "direct_vector_cleanup"
// CHECK-SAME: temporary_allocated_bytes = 0
// CHECK-SAME: zero_fill_bytes = 0
// CHECK-SAME: candidate_kind = "tiled_vector_specialized_tail"
// CHECK-SAME: rejection_reason = "specialized_k_tail_microkernel_not_registered"
// CHECK-SAME: candidate_kind = "whole_shape_vector_materialized_padding"
// CHECK-SAME: padding_policy = "whole_shape_materialized"
// CHECK-SAME: native.cost_model.runner_up
// CHECK-SAME: native.cost_model.uncertainty_margin = 1.500000e-01

module attributes {
  llm.dtype = "fp32",
  target.profile_id = "raspberry-pi5-cortex-a76-cpu",
  target.supported_precisions = ["fp32"]
} {
  func.func @k_tail(
      %lhs: tensor<8x15xf32>, %rhs: tensor<15x8xf32>,
      %bias: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf32>
    %mm = linalg.matmul
      ins(%lhs, %rhs : tensor<8x15xf32>, tensor<15x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
    %add = linalg.map { arith.addf }
      ins(%mm, %bias : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>)
    %zero = arith.constant 0.0 : f32
    %relu = linalg.map
      ins(%add : tensor<8x8xf32>) outs(%empty : tensor<8x8xf32>)
      (%value: f32) {
        %clamped = arith.maximumf %value, %zero : f32
        linalg.yield %clamped : f32
      }
    return %relu : tensor<8x8xf32>
  }
}
