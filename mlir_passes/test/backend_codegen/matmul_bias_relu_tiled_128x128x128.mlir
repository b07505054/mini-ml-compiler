// Input HIR fixture for the tiled AArch64 native-codegen variant.
// Identical in op structure to the other matmul_bias_relu_tiled_<shape>
// fixtures; the function is named distinctly (matmul_bias_relu_tiled_<shape>)
// so objects for different shapes can be linked into the same benchmark
// harness without symbol collisions. See
// mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant
// tiled-scheduled.
//
// Added for Stage 13 (Raspberry Pi correctness/benchmark validation of the
// machine-scheduling slice) as the "one larger supported shape" required by
// the task brief -- M=N=K=128 gives tile-8x8x8 a K-loop trip count of 16
// (32/8=16 at schedule-unroll-k=1, 8 at schedule-unroll-k=2), well clear of
// any trip-count-one collapse, exercising the primary schedule transform at
// a size distinct from the 32x32x32/64x64x64 candidates already used in
// Stage 12.
//
// Shape: M=128, N=128, K=128.

func.func @matmul_bias_relu_tiled_128x128x128(
    %lhs: tensor<128x128xf32>,
    %rhs: tensor<128x128xf32>,
    %bias: tensor<128x128xf32>) -> tensor<128x128xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
