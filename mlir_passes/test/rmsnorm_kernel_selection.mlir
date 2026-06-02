// RUN: mlir-opt %s --allow-unregistered-dialect --load-pass-plugin=%plugin --rmsnorm-kernel-selection | FileCheck %s

func.func @main(%x: tensor<16x768xf16>) -> tensor<16x768xf16> {
  // CHECK: "llm.rmsnorm"
  // CHECK-SAME: fusion.candidate = "rmsnorm"
  // CHECK-SAME: fusion.group = "rmsnorm_0"
  // CHECK-SAME: fusion.role = "normalization"
  // CHECK-SAME: kernel.selection = "runtime_profile"
  // CHECK-SAME: lowering.hir_op = "hir.fused_rmsnorm"
  %0 = "llm.rmsnorm"(%x) : (tensor<16x768xf16>) -> tensor<16x768xf16>
  return %0 : tensor<16x768xf16>
}
