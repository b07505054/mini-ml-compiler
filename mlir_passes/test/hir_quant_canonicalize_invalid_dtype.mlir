// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_quantized_dtype(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{requires 'quantized_dtype' = "i8"}}
  %q = hir.quantize %x {
    quantization.mode = "per_tensor",
    quantized_dtype = "u8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32
  } : (tensor<4x4xf32>) -> tensor<4x4xi8>
  %dq = hir.dequantize %q {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32
  } : (tensor<4x4xi8>) -> tensor<4x4xf32>
  return %dq : tensor<4x4xf32>
}
