// RUN: mlir-opt %s --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-matmul-bias-relu-to-linalg,one-shot-bufferize{bufferize-function-boundaries})' | FileCheck %s

func.func @matmul_bias_relu(
    %lhs: tensor<2x4xf32>,
    %rhs: tensor<4x3xf32>,
    %bias: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK-NOT: hir.fused_matmul_bias_relu
  // CHECK: memref.alloc
  // CHECK: linalg.matmul
  // CHECK: linalg.generic
  // CHECK: memref.cast
  // CHECK: return
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<2x4xf32>, tensor<4x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
