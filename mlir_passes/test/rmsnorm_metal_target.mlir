// RUN: mlir-opt %s --allow-unregistered-dialect --load-pass-plugin=%plugin --pass-pipeline='builtin.module(rmsnorm-kernel-selection,hir-fusion-lowering,hir-verify-fused-ops)' | FileCheck %s

func.func @main(%x: tensor<16x4096xf32>) -> tensor<16x4096xf32> {
  // CHECK-NOT: "llm.rmsnorm"
  // CHECK: hir.fused_rmsnorm
  // CHECK-SAME: fusion.candidate = "rmsnorm"
  // CHECK-SAME: kernel.selection = "runtime_profile"
  // CHECK-SAME: (tensor<16x4096xf32>) -> tensor<16x4096xf32>
  %0 = "llm.rmsnorm"(%x) : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
  return %0 : tensor<16x4096xf32>
}
