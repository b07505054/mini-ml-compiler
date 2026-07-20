// Padded fused-vector fixture. The compute executes at 16x32x16 and the
// returned tensor is cropped to the original 15x15 result.

func.func @matmul_bias_relu_vectorized_15x31x15(
    %lhs: tensor<15x31xf32>,
    %rhs: tensor<31x15xf32>,
    %bias: tensor<15x15xf32>) -> tensor<15x15xf32>
    attributes {llvm.emit_c_interface} {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu",
    target.original_m = 15 : i64,
    target.original_n = 15 : i64,
    target.original_k = 31 : i64,
    target.padded_m = 16 : i64,
    target.padded_n = 16 : i64,
    target.padded_k = 32 : i64,
    target.padding = "pad_to_tile_with_crop",
    target.valid_region = "original_m_n"
  } : (tensor<15x31xf32>, tensor<31x15xf32>, tensor<15x15xf32>)
      -> tensor<15x15xf32>
  return %0 : tensor<15x15xf32>
}
