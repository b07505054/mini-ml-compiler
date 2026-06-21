// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_padded_target(
  %lhs: tensor<127x255xf32>,
  %rhs: tensor<255x129xf32>,
  %bias: tensor<1x129xf32>
) -> tensor<127x129xf32> {
  // expected-error @+1 {{requires padded dimensions to be multiples of target tile shape}}
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "runtime_profile",
    lowering.source = "linalg.matmul_add_relu",
    target.alignment = 128 : i32,
    target.collective = "none",
    target.memory_hierarchy = "global_sram_register",
    target.model = "sparsecore_like_v1",
    target.original_k = 255 : i64,
    target.original_m = 127 : i64,
    target.original_n = 129 : i64,
    target.pad_k = 1 : i64,
    target.pad_m = 0 : i64,
    target.pad_n = 15 : i64,
    target.padded_k = 256 : i64,
    target.padded_m = 127 : i64,
    target.padded_n = 144 : i64,
    target.padding = "pad_to_tile_with_crop",
    target.padding_compute_overhead_ratio = 1.121644 : f64,
    target.padding_output_overhead_ratio = 1.124122 : f64,
    target.sparse_layout = "dense_or_2_4",
    target.sram_kb = 256 : i32,
    target.tile_k = 32 : i32,
    target.tile_m = 16 : i32,
    target.tile_n = 16 : i32,
    target.valid_region = "original_m_n",
    target.vector_bytes = 128 : i32
  } : (tensor<127x255xf32>, tensor<255x129xf32>, tensor<1x129xf32>) -> tensor<127x129xf32>
  return %0 : tensor<127x129xf32>
}
