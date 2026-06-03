// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_bias(
  %lhs: tensor<16x128xf32>,
  %rhs: tensor<128x64xf32>,
  %bias: tensor<2x64xf32>
) -> tensor<16x64xf32> {
  // expected-error @+1 {{expects bias M dimension to be 1 or match result M dimension}}
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "runtime_profile",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<16x128xf32>, tensor<128x64xf32>, tensor<2x64xf32>) -> tensor<16x64xf32>
  return %0 : tensor<16x64xf32>
}
