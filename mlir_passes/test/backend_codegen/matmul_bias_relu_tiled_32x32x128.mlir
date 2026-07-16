// Input HIR fixture for the tiled AArch64 native-codegen variant.
// Identical in op structure to the other matmul_bias_relu_tiled_<shape>
// fixtures. Added for Stage 17 (schedule-unroll boundary search) as the
// "Category B -- larger K-loop trip count" stress domain: M=N=32, K=128
// gives tile-8x8x8 a K-loop trip count of 16 (32/8=16 at
// schedule-unroll-k=1, 4 at schedule-unroll-k=4 -- a genuine PARTIAL
// unroll, not a full collapse), distinct from every shape tested in
// Stages 12-16 (max K trip count tested there was 8, at cube64).
//
// Shape: M=32, N=32, K=128.

func.func @matmul_bias_relu_tiled_32x32x128(
    %lhs: tensor<32x128xf32>,
    %rhs: tensor<128x32xf32>,
    %bias: tensor<32x32xf32>) -> tensor<32x32xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
