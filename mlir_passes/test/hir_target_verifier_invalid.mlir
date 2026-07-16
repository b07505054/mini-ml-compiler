// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_target(
  %lhs: tensor<16x128xf32>,
  %rhs: tensor<128x64xf32>,
  %bias: tensor<1x64xf32>
) -> tensor<16x64xf32> {
  // expected-error @+1 {{requires target tile shape 16x16x32}}
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "runtime_profile",
    lowering.source = "linalg.matmul_add_relu",
    target.alignment = 128 : i32,
    target.collective = "none",
    target.memory_hierarchy = "global_sram_register",
    target.model = "portable_accelerator_v1",
    target.sram_kb = 256 : i32,
    target.tile_k = 16 : i32,
    target.tile_m = 16 : i32,
    target.tile_n = 16 : i32,
    target.vector_bytes = 128 : i32
  } : (tensor<16x128xf32>, tensor<128x64xf32>, tensor<1x64xf32>) -> tensor<16x64xf32>
  return %0 : tensor<16x64xf32>
}
