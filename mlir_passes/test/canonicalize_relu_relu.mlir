// RUN: mlir-opt %s --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize)' | FileCheck %s

func.func @main(%input: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %empty = tensor.empty() : tensor<1x64xf32>
  %zero = arith.constant 0.0 : f32

  %relu0 = linalg.map
      ins(%input : tensor<1x64xf32>)
      outs(%empty : tensor<1x64xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  %relu1 = linalg.map
      ins(%relu0 : tensor<1x64xf32>)
      outs(%empty : tensor<1x64xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu1 : tensor<1x64xf32>
}

// CHECK-LABEL: func.func @main
// CHECK: linalg.map
// CHECK-NOT: linalg.map
// CHECK: return
