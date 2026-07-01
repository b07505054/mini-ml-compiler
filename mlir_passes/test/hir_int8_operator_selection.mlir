// RUN: mlir-opt %s --split-input-file --allow-unregistered-dialect --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-quant-propagate,hir-int8-operator-selection)' | FileCheck %s

// CHECK-LABEL: func.func @int8_selected_when_legal
func.func @int8_selected_when_legal(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>
) -> tensor<128x128xf32> attributes {target.backend = "cpu"} {
  %empty = tensor.empty() : tensor<128x128xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: quant.selection = "int8"
  // CHECK-SAME: quant.selection_reason = "capability_table_int8"
  %mm = linalg.matmul {
      input_layout = "NHWC",
      profile.quantized_path = "faster",
      quantization.candidate = "int8",
      weight_layout = "blocked_kc"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %mm : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @fallback_when_backend_lacks_int8
func.func @fallback_when_backend_lacks_int8(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>
) -> tensor<128x128xf32> attributes {target.backend = "metal"} {
  %empty = tensor.empty() : tensor<128x128xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: quant.selection = "fallback"
  // CHECK-SAME: quant.selection_reason = "backend_lacks_int8"
  %mm = linalg.matmul {
      input_layout = "NHWC",
      profile.quantized_path = "faster",
      quantization.candidate = "int8",
      weight_layout = "blocked_kc"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %mm : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @fallback_when_layout_illegal
func.func @fallback_when_layout_illegal(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>
) -> tensor<128x128xf32> attributes {target.backend = "cpu"} {
  %empty = tensor.empty() : tensor<128x128xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: quant.selection = "fallback"
  // CHECK-SAME: quant.selection_reason = "illegal_layout"
  %mm = linalg.matmul {
      input_layout = "NCHW",
      profile.quantized_path = "faster",
      quantization.candidate = "int8",
      weight_layout = "blocked_kc"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %mm : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @fallback_when_profile_slower
func.func @fallback_when_profile_slower(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>
) -> tensor<128x128xf32> attributes {target.backend = "cpu"} {
  %empty = tensor.empty() : tensor<128x128xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: quant.selection = "fallback"
  // CHECK-SAME: quant.selection_reason = "profile_not_faster"
  %mm = linalg.matmul {
      input_layout = "NHWC",
      profile.quantized_path = "slower",
      quantization.candidate = "int8",
      weight_layout = "blocked_kc"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %mm : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @fallback_when_shape_not_aligned
func.func @fallback_when_shape_not_aligned(
  %A: tensor<127x128xf32>,
  %B: tensor<128x127xf32>
) -> tensor<127x127xf32> attributes {target.backend = "cpu"} {
  %empty = tensor.empty() : tensor<127x127xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: quant.selection = "fallback"
  // CHECK-SAME: quant.selection_reason = "illegal_shape"
  %mm = linalg.matmul {
      input_layout = "NHWC",
      profile.quantized_path = "faster",
      quantization.candidate = "int8",
      weight_layout = "blocked_kc"
    }
    ins(%A, %B : tensor<127x128xf32>, tensor<128x127xf32>)
    outs(%empty : tensor<127x127xf32>) -> tensor<127x127xf32>
  return %mm : tensor<127x127xf32>
}

// -----

// CHECK-LABEL: func.func @relu_and_reshape_capability
func.func @relu_and_reshape_capability(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>
) -> tensor<1x128x128xf32> attributes {target.backend = "cpu"} {
  %empty = tensor.empty() : tensor<128x128xf32>
  %mm = linalg.matmul {
      input_layout = "NHWC",
      quantization.candidate = "int8",
      weight_layout = "blocked_kc"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %zero = arith.constant 0.0 : f32
  // CHECK: linalg.map
  // CHECK-SAME: quant.op = "relu"
  // CHECK-SAME: quant.selection = "int8"
  %relu = linalg.map
      ins(%mm : tensor<128x128xf32>)
      outs(%empty : tensor<128x128xf32>)
      (%x: f32, %out: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }
  // CHECK: tensor.expand_shape
  // CHECK-SAME: quant.op = "reshape"
  // CHECK-SAME: quant.selection = "int8"
  %reshaped = tensor.expand_shape %relu [[0, 1], [2]]
    output_shape [1, 128, 128] : tensor<128x128xf32> into tensor<1x128x128xf32>
  return %reshaped : tensor<1x128x128xf32>
}
