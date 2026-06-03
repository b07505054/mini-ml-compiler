// RUN: mlir-opt %s --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize)' | FileCheck %s

func.func @main(%input: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %empty = tensor.empty() : tensor<1x64xf32>
  %zero = arith.constant 0.0 : f32

  %add_zero = linalg.map
      ins(%input : tensor<1x64xf32>)
      outs(%empty : tensor<1x64xf32>)
      (%x: f32, %out: f32) {
    %y = arith.addf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %add_zero : tensor<1x64xf32>
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: %[[INPUT:.*]]: tensor<1x64xf32>
// CHECK-NOT: arith.addf
// CHECK-NOT: linalg.map
// CHECK: return %[[INPUT]] : tensor<1x64xf32>
