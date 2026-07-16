// Input HIR fixture for the tiled AArch64 native-codegen variant.
// Identical in op structure to matmul_bias_relu_32x32x32.mlir; the function is
// named distinctly (matmul_bias_relu_tiled_<shape>) so the generic,
// fully-unrolled-vectorized, and tiled-vectorized objects can be linked into
// the same benchmark harness without symbol collisions. See
// mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant tiled.
//
// Shape: M=32, N=32, K=32.

func.func @matmul_bias_relu_tiled_32x32x32_tm8_tn8_tk8(
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
