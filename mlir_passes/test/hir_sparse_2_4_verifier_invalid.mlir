// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_sparse_metadata(
  %lhs: tensor<16x32xf32>,
  %rhs: tensor<32x16xf32>,
  %bias: tensor<16x16xf32>
) -> tensor<16x16xf32> {
  // expected-error @+1 {{requires string attribute 'target.sparse_axis'}}
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "sparse_2_4_profile",
    lowering.source = "linalg.matmul_add_relu",
    sparse.candidate = "2_4",
    sparse.legal = true,
    target.alignment = 128 : i32,
    target.collective = "none",
    target.memory_hierarchy = "global_sram_register",
    target.model = "sparsecore_like_v1",
    target.padding = "none",
    target.sparse_layout = "structured_2_4",
    target.sram_kb = 256 : i32,
    target.tile_k = 32 : i32,
    target.tile_m = 16 : i32,
    target.tile_n = 16 : i32,
    target.vector_bytes = 128 : i32
  } : (tensor<16x32xf32>, tensor<32x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
