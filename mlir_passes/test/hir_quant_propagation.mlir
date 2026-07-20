// RUN: mlir-opt %s --split-input-file --allow-unregistered-dialect --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-quant-propagate)' | FileCheck %s

// CHECK-LABEL: func.func @matmul_relu_matmul_one_island
func.func @matmul_relu_matmul_one_island(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>,
  %C: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %empty = tensor.empty() : tensor<128x128xf32>

  // CHECK: linalg.matmul
  // CHECK-SAME: quant.island = "int8_island_0"
  %mm0 = linalg.matmul {
      quantization.candidate = "int8"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

  %zero = arith.constant 0.0 : f32
  // CHECK: linalg.map
  // CHECK-SAME: quant.island = "int8_island_0"
  // CHECK-SAME: quant.propagation = "relu"
  %relu = linalg.map
      ins(%mm0 : tensor<128x128xf32>)
      outs(%empty : tensor<128x128xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  // CHECK: linalg.matmul
  // CHECK-SAME: quant.island = "int8_island_0"
  %mm1 = linalg.matmul {
      quantization.candidate = "int8"
    }
    ins(%relu, %C : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

  return %mm1 : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @reshape_stays_inside_island
func.func @reshape_stays_inside_island(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>
) -> tensor<1x128x128xf32> {
  %empty = tensor.empty() : tensor<128x128xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: quant.island = "int8_island_0"
  %mm = linalg.matmul {
      quantization.candidate = "int8"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: tensor.expand_shape
  // CHECK-SAME: quant.island = "int8_island_0"
  // CHECK-SAME: quant.propagation = "reshape"
  %reshaped = tensor.expand_shape %mm [[0, 1], [2]]
    output_shape [1, 128, 128] : tensor<128x128xf32> into tensor<1x128x128xf32>
  return %reshaped : tensor<1x128x128xf32>
}

// -----

// CHECK-LABEL: func.func @unsupported_add_breaks_island
func.func @unsupported_add_breaks_island(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>,
  %C: tensor<128x128xf32>,
  %D: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %empty = tensor.empty() : tensor<128x128xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: quant.island = "int8_island_0"
  %mm0 = linalg.matmul {
      quantization.candidate = "int8"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: linalg.map
  // CHECK-NOT: quant.island
  %add = linalg.map
      ins(%mm0, %C : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%empty : tensor<128x128xf32>)
      (%x: f32, %y: f32) {
    %z = arith.addf %x, %y : f32
    linalg.yield %z : f32
  }

  // CHECK: linalg.matmul
  // CHECK-SAME: quant.island = "int8_island_1"
  %mm1 = linalg.matmul {
      quantization.candidate = "int8"
    }
    ins(%add, %D : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %mm1 : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @add_relu_order_not_rewritten
func.func @add_relu_order_not_rewritten(
  %x: tensor<4x4xf32>,
  %bias: tensor<4x4xf32>
) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  // CHECK: arith.addf
  // CHECK: arith.maximumf
  %add = linalg.map
      ins(%x, %bias : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%empty : tensor<4x4xf32>)
      (%a: f32, %b: f32) {
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  }
  %zero = arith.constant 0.0 : f32
  %relu = linalg.map
      ins(%add : tensor<4x4xf32>)
      outs(%empty : tensor<4x4xf32>)
      (%a: f32) {
    %out = arith.maximumf %a, %zero : f32
    linalg.yield %out : f32
  }
  return %relu : tensor<4x4xf32>
}
