// RUN: mlir-opt %s --allow-unregistered-dialect --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering)' | FileCheck %s

func.func @main(
  %A: tensor<16x128xf32>,
  %B: tensor<128x64xf32>,
  %bias: tensor<16x64xf32>
) -> tensor<16x64xf32> {
  %empty = tensor.empty() : tensor<16x64xf32>

  // CHECK-NOT: linalg.matmul
  // CHECK: hir.fused_matmul_bias_relu
  // CHECK-SAME: fusion.candidate = "matmul_bias_relu"
  // CHECK-SAME: kernel.selection = "runtime_profile"
  // CHECK-SAME: lowering.source = "linalg.matmul_add_relu"
  // CHECK-SAME: target.model = "portable_accelerator_v1"
  // CHECK-SAME: target.tile_k = 32
  // CHECK-SAME: target.tile_m = 16
  // CHECK-SAME: target.tile_n = 16
  // CHECK-SAME: (tensor<16x128xf32>, tensor<128x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
  %mm = linalg.matmul
    ins(%A, %B : tensor<16x128xf32>, tensor<128x64xf32>)
    outs(%empty : tensor<16x64xf32>) -> tensor<16x64xf32>

  %add = linalg.map
      ins(%mm, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
      (%x: f32, %b: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32

  %relu = linalg.map
      ins(%add : tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<16x64xf32>
}
