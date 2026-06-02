// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_layout(
  %lhs: tensor<128x128xi8>,
  %rhs: tensor<128x128xi8>,
  %bias: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  // expected-error @+1 {{requires 128-byte activation alignment}}
  %0 = hir.fused_qmatmul_bias_relu %lhs, %rhs, %bias {
    alignment = 64 : i32,
    fusion.candidate = "qmatmul_bias_relu",
    input_layout = "NHWC",
    lhs_scale = 1.000000e-02 : f32,
    lhs_zero_point = 0 : i32,
    quantization.mode = "per_channel",
    quantized_dtype = "i8",
    rhs_scale = 1.000000e-02 : f32,
    rhs_zero_point = 0 : i32,
    weight_layout = "blocked_kc"
  } : (tensor<128x128xi8>, tensor<128x128xi8>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
