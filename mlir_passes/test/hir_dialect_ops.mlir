// RUN: mlir-opt %s --load-dialect-plugin=%plugin | FileCheck %s

func.func @matmul(
  %A: tensor<1x128xf32>,
  %B: tensor<128x64xf32>,
  %bias: tensor<1x64xf32>
) -> tensor<1x64xf32> {
  // CHECK: hir.fused_matmul_bias_relu
  // CHECK-SAME: fusion.candidate = "matmul_bias_relu"
  // CHECK-SAME: kernel.selection = "runtime_profile"
  // CHECK-SAME: lowering.source = "linalg.matmul_add_relu"
  %0 = hir.fused_matmul_bias_relu %A, %B, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "runtime_profile",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<1x128xf32>, tensor<128x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
  return %0 : tensor<1x64xf32>
}

func.func @rmsnorm(%x: tensor<16x768xf16>) -> tensor<16x768xf16> {
  // CHECK: hir.fused_rmsnorm
  // CHECK-SAME: fusion.candidate = "rmsnorm"
  // CHECK-SAME: kernel.selection = "runtime_profile"
  // CHECK-SAME: lowering.source = "llm.rmsnorm"
  %0 = hir.fused_rmsnorm %x {
    fusion.candidate = "rmsnorm",
    kernel.selection = "runtime_profile",
    lowering.source = "llm.rmsnorm"
  } : (tensor<16x768xf16>) -> tensor<16x768xf16>
  return %0 : tensor<16x768xf16>
}
