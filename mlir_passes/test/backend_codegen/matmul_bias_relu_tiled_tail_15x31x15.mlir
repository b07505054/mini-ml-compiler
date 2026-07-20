// Original-shape fused HIR input for the bounded tiled tail schedule.
func.func @matmul_bias_relu_tiled_tail_15x31x15(
    %lhs: tensor<15x31xf32>, %rhs: tensor<31x15xf32>,
    %bias: tensor<15x15xf32>) -> tensor<15x15xf32>
    attributes {llvm.emit_c_interface} {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu",
    target.padding = "none"
  } : (tensor<15x31xf32>, tensor<31x15xf32>, tensor<15x15xf32>)
      -> tensor<15x15xf32>
  return %0 : tensor<15x15xf32>
}
