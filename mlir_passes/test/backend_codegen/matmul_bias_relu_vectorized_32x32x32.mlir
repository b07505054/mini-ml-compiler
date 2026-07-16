// Input HIR fixture for the vectorized AArch64 native-codegen variant.
// See matmul_bias_relu_vectorized_8x8x8.mlir for the full explanation.
//
// Shape: M=32, N=32, K=32.

func.func @matmul_bias_relu_vectorized_32x32x32(
    %lhs: tensor<32x32xf32>,
    %rhs: tensor<32x32xf32>,
    %bias: tensor<32x32xf32>) -> tensor<32x32xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
