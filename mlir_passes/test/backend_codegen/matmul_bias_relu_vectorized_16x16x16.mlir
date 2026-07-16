// Input HIR fixture for the vectorized AArch64 native-codegen variant.
// See matmul_bias_relu_vectorized_8x8x8.mlir for the full explanation.
//
// Shape: M=16, N=16, K=16.

func.func @matmul_bias_relu_vectorized_16x16x16(
    %lhs: tensor<16x16xf32>,
    %rhs: tensor<16x16xf32>,
    %bias: tensor<16x16xf32>) -> tensor<16x16xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
