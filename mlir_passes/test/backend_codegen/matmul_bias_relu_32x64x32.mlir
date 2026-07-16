// Input HIR fixture for the generic AArch64 native-codegen variant.
// See mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant generic.
//
// Shape: M=32, N=64, K=32.

func.func @matmul_bias_relu_32x64x32(
    %lhs: tensor<32x32xf32>,
    %rhs: tensor<32x64xf32>,
    %bias: tensor<32x64xf32>) -> tensor<32x64xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}
