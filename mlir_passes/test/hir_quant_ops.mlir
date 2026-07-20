// RUN: mlir-opt %s --load-dialect-plugin=%plugin | FileCheck %s

func.func @quant_path(
  %x: tensor<128x128xf32>,
  %w: tensor<128x128xi8>,
  %bias: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  // CHECK: hir.quantize
  %xq = hir.quantize %x {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 1.000000e-02 : f32,
    zero_point = 0 : i32,
    clamp_min = -127 : i32,
    clamp_max = 127 : i32
  } : (tensor<128x128xf32>) -> tensor<128x128xi8>

  // CHECK: hir.qmatmul
  %mm = hir.qmatmul %xq, %w {
    lhs_scale = 1.000000e-02 : f32,
    lhs_zero_point = 0 : i32,
    quantized_dtype = "i8",
    rhs_scale = 1.000000e-02 : f32,
    rhs_zero_point = 0 : i32,
    packed_layout = "packed_b_transposed_nxk",
    accumulator_dtype = "int32",
    output_dtype = "fp32"
  } : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  // CHECK: hir.fused_qmatmul_bias_relu
  %fused = hir.fused_qmatmul_bias_relu %xq, %w, %bias {
    alignment = 128 : i32,
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

  // CHECK: hir.dequantize
  %dq = hir.dequantize %mm {
    quantized_dtype = "i8",
    scale = 1.000000e-02 : f32,
    zero_point = 0 : i32
  } : (tensor<128x128xi32>) -> tensor<128x128xf32>

  return %fused : tensor<128x128xf32>
}
