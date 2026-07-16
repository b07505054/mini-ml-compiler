// Input HIR fixture for the tiled AArch64 native-codegen variant.
// Identical in op structure to matmul_bias_relu_64x64x64.mlir; the function is
// named distinctly (matmul_bias_relu_tiled_<shape>) so the generic,
// fully-unrolled-vectorized, and tiled-vectorized objects can be linked into
// the same benchmark harness without symbol collisions. See
// mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant tiled.
//
// Shape: M=64, N=64, K=64.

func.func @matmul_bias_relu_tiled_64x64x64(
    %lhs: tensor<64x64xf32>,
    %rhs: tensor<64x64xf32>,
    %bias: tensor<64x64xf32>) -> tensor<64x64xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
