// RUN: mlir-opt %s --load-pass-plugin=%plugin --pass-pipeline='builtin.module(matmul-bias-relu-fusion)' | FileCheck %s

func.func @main(
  %A: tensor<?x128xf32>,
  %B: tensor<128x64xf32>,
  %bias: tensor<?x64xf32>
) -> tensor<?x64xf32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %A, %c0 : tensor<?x128xf32>
  %empty = tensor.empty(%m) : tensor<?x64xf32>

  %mm = linalg.matmul
    ins(%A, %B : tensor<?x128xf32>, tensor<128x64xf32>)
    outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>

  %add = linalg.map
      ins(%mm, %bias : tensor<?x64xf32>, tensor<?x64xf32>)
      outs(%empty : tensor<?x64xf32>)
      (%x: f32, %b: f32, %out: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32
  %relu = linalg.map
      ins(%add : tensor<?x64xf32>)
      outs(%empty : tensor<?x64xf32>)
      (%x: f32, %out: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<?x64xf32>
}

// CHECK: linalg.matmul
// CHECK-NOT: fusion.candidate
