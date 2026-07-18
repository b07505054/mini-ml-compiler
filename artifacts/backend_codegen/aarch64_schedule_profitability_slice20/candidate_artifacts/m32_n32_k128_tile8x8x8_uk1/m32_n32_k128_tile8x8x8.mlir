func.func @matmul_bias_relu_tiled_32x32x128(%lhs: tensor<32x128xf32>, %rhs: tensor<128x32xf32>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {fusion.candidate = "matmul_bias_relu", kernel.selection = "native_cpu", lowering.source = "linalg.matmul_add_relu"} : (tensor<32x128xf32>, tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
