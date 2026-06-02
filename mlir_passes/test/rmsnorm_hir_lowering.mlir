// RUN: mlir-opt %s --allow-unregistered-dialect --load-pass-plugin=%plugin --pass-pipeline='builtin.module(rmsnorm-kernel-selection,hir-fusion-lowering)' | FileCheck %s

func.func @main(%x: tensor<16x768xf16>) -> tensor<16x768xf16> {
  // CHECK-NOT: "llm.rmsnorm"
  // CHECK: "hir.fused_rmsnorm"
  // CHECK-SAME: fusion.candidate = "rmsnorm"
  // CHECK-SAME: kernel.selection = "runtime_profile"
  // CHECK-SAME: lowering.source = "llm.rmsnorm"
  // CHECK-SAME: (tensor<16x768xf16>) -> tensor<16x768xf16>
  %0 = "llm.rmsnorm"(%x) : (tensor<16x768xf16>) -> tensor<16x768xf16>
  return %0 : tensor<16x768xf16>
}
