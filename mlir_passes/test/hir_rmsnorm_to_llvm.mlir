// RUN: mlir-opt %s --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-rmsnorm-to-linalg,one-shot-bufferize{bufferize-function-boundaries},convert-linalg-to-loops,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' | FileCheck %s

func.func @rmsnorm(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK-NOT: hir.fused_rmsnorm
  // CHECK-NOT: linalg.generic
  // CHECK: llvm.func @rmsnorm
  // CHECK: llvm.intr.sqrt
  // CHECK: llvm.return
  %0 = hir.fused_rmsnorm %x {
    fusion.candidate = "rmsnorm",
    kernel.selection = "native_cpu",
    lowering.source = "llm.rmsnorm"
  } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

