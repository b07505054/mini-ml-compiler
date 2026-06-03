// RUN: mlir-opt %s --load-pass-plugin=%plugin --matmul-bias-relu-fusion | FileCheck %s

func.func @main(
  %A: tensor<16x128xf32>,
  %B: tensor<128x64xf32>,
  %bias: tensor<16x64xf32>
) -> tensor<16x64xf32> {
  %empty = tensor.empty() : tensor<16x64xf32>

  %mm = linalg.matmul
    ins(%A, %B : tensor<16x128xf32>, tensor<128x64xf32>)
    outs(%empty : tensor<16x64xf32>) -> tensor<16x64xf32>

  // CHECK: linalg.matmul
  // CHECK-SAME: fusion.candidate = "matmul_bias_relu"
  // CHECK-SAME: fusion.group = "matmul_bias_relu_0"
  // CHECK-SAME: fusion.role = "producer"
  // CHECK: linalg.map
  // CHECK-SAME: fusion.group = "matmul_bias_relu_0"
  // CHECK-SAME: fusion.role = "bias_add"
  // CHECK: linalg.map
  // CHECK-SAME: fusion.group = "matmul_bias_relu_0"
  // CHECK-SAME: fusion.role = "activation"

  %add = linalg.map
      ins(%mm, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
      (%x: f32, %b: f32, %out: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32

  %relu = linalg.map
      ins(%add : tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
      (%x: f32, %out: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<16x64xf32>
}
