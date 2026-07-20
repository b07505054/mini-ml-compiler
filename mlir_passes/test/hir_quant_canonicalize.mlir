// RUN: mlir-opt %s --split-input-file --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-quant-canonicalize)' | FileCheck %s

// CHECK-LABEL: func.func @remove_dequantize_quantize
// CHECK-SAME: (%[[X:.*]]: tensor<4x4xf32>)
func.func @remove_dequantize_quantize(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NOT: hir.quantize
  // CHECK-NOT: hir.dequantize
  // CHECK: return %[[X]] : tensor<4x4xf32>
  %q = hir.quantize %x {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32,
    clamp_min = -127 : i32,
    clamp_max = 127 : i32
  } : (tensor<4x4xf32>) -> tensor<4x4xi8>
  %dq = hir.dequantize %q {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32
  } : (tensor<4x4xi8>) -> tensor<4x4xf32>
  return %dq : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @remove_quantize_dequantize
// CHECK-SAME: (%[[X:.*]]: tensor<4x4xi8>)
func.func @remove_quantize_dequantize(%x: tensor<4x4xi8>) -> tensor<4x4xi8> {
  // CHECK-NOT: hir.dequantize
  // CHECK-NOT: hir.quantize
  // CHECK: return %[[X]] : tensor<4x4xi8>
  %dq = hir.dequantize %x {
    quantization.mode = "per_channel",
    quantized_dtype = "i8",
    scale = 1.000000e-02 : f32,
    zero_point = 0 : i32
  } : (tensor<4x4xi8>) -> tensor<4x4xf32>
  %q = hir.quantize %dq {
    quantization.mode = "per_channel",
    quantized_dtype = "i8",
    scale = 1.000000e-02 : f32,
    zero_point = 0 : i32,
    clamp_min = -127 : i32,
    clamp_max = 127 : i32
  } : (tensor<4x4xf32>) -> tensor<4x4xi8>
  return %q : tensor<4x4xi8>
}

// -----

// CHECK-LABEL: func.func @keep_mismatched_scale
func.func @keep_mismatched_scale(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: hir.quantize
  // CHECK: hir.dequantize
  %q = hir.quantize %x {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32,
    clamp_min = -127 : i32,
    clamp_max = 127 : i32
  } : (tensor<4x4xf32>) -> tensor<4x4xi8>
  %dq = hir.dequantize %q {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 5.000000e-01 : f32,
    zero_point = 3 : i32
  } : (tensor<4x4xi8>) -> tensor<4x4xf32>
  return %dq : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @keep_mismatched_zero_point
func.func @keep_mismatched_zero_point(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: hir.quantize
  // CHECK: hir.dequantize
  %q = hir.quantize %x {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32,
    clamp_min = -127 : i32,
    clamp_max = 127 : i32
  } : (tensor<4x4xf32>) -> tensor<4x4xi8>
  %dq = hir.dequantize %q {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 4 : i32
  } : (tensor<4x4xi8>) -> tensor<4x4xf32>
  return %dq : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @keep_mismatched_mode
func.func @keep_mismatched_mode(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: hir.quantize
  // CHECK: hir.dequantize
  %q = hir.quantize %x {
    quantization.mode = "per_tensor",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32,
    clamp_min = -127 : i32,
    clamp_max = 127 : i32
  } : (tensor<4x4xf32>) -> tensor<4x4xi8>
  %dq = hir.dequantize %q {
    quantization.mode = "per_channel",
    quantized_dtype = "i8",
    scale = 2.500000e-01 : f32,
    zero_point = 3 : i32
  } : (tensor<4x4xi8>) -> tensor<4x4xf32>
  return %dq : tensor<4x4xf32>
}
