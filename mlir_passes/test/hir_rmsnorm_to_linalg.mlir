// RUN: mlir-opt %s --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-rmsnorm-to-linalg)' | FileCheck %s

func.func @rmsnorm(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK-NOT: hir.fused_rmsnorm
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK: arith.mulf
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK: math.rsqrt
  %0 = hir.fused_rmsnorm %x {
    fusion.candidate = "rmsnorm",
    kernel.selection = "native_cpu",
    lowering.source = "llm.rmsnorm"
  } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

