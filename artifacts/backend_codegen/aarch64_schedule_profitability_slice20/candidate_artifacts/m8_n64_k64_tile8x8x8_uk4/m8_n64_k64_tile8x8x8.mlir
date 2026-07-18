func.func @matmul_bias_relu_tiled_8x64x64(%lhs: tensor<8x64xf32>, %rhs: tensor<64x64xf32>, %bias: tensor<8x64xf32>) -> tensor<8x64xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {fusion.candidate = "matmul_bias_relu", kernel.selection = "native_cpu", lowering.source = "linalg.matmul_add_relu"} : (tensor<8x64xf32>, tensor<64x64xf32>, tensor<8x64xf32>) -> tensor<8x64xf32>
  return %0 : tensor<8x64xf32>
}
