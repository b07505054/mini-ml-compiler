// RUN: mlir-opt %s --load-pass-plugin=%plugin --matmul-bias-relu-fusion | FileCheck %s

func.func @main(
  %A: tensor<1x128xf32>,
  %B: tensor<128x64xf32>,
  %bias: tensor<1x64xf32>
) -> tensor<1x64xf32> {
  %empty = tensor.empty() : tensor<1x64xf32>

  %mm = linalg.matmul
    ins(%A, %B : tensor<1x128xf32>, tensor<128x64xf32>)
    outs(%empty : tensor<1x64xf32>) -> tensor<1x64xf32>

  // CHECK: linalg.matmul
  // CHECK-SAME: fusion.candidate = "matmul_bias_relu"

  %add = linalg.map
      ins(%mm, %bias : tensor<1x64xf32>, tensor<1x64xf32>)
      outs(%empty : tensor<1x64xf32>)
      (%x: f32, %b: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32

  %relu = linalg.map
      ins(%add : tensor<1x64xf32>)
      outs(%empty : tensor<1x64xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<1x64xf32>
}