func.func @matmul_bias_relu_tiled_8x8x32(%lhs: tensor<8x32xf32>, %rhs: tensor<32x8xf32>, %bias: tensor<8x8xf32>) -> tensor<8x8xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {fusion.candidate = "matmul_bias_relu", kernel.selection = "native_cpu", lowering.source = "linalg.matmul_add_relu"} : (tensor<8x32xf32>, tensor<32x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
