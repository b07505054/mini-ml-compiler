func.func @matmul_bias_relu_direct_8x8x15(
    %lhs: tensor<8x15xf32>, %rhs: tensor<15x8xf32>,
    %bias: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {llvm.emit_c_interface} {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu",
    target.padding = "none"
  } : (tensor<8x15xf32>, tensor<15x8xf32>, tensor<8x8xf32>)
      -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
