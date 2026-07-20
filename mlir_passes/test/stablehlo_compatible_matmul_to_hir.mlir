// RUN: mlir-opt %s --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)' | FileCheck %s

// StableHLO-compatible frontend shape:
//   stablehlo.dot_general + stablehlo.add + stablehlo.maximum
// becomes the equivalent Linalg/Arith pattern before HIR fusion.
func.func @stablehlo_compatible_matmul_bias_relu(
    %lhs: tensor<16x128xf32>,
    %rhs: tensor<128x64xf32>,
    %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
  // CHECK-NOT: linalg.matmul
  // CHECK: hir.fused_matmul_bias_relu
  // CHECK-SAME: fusion.candidate = "matmul_bias_relu"
  // CHECK-SAME: lowering.source = "linalg.matmul_add_relu"
  %empty = tensor.empty() : tensor<16x64xf32>
  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<16x128xf32>, tensor<128x64xf32>)
      outs(%empty : tensor<16x64xf32>) -> tensor<16x64xf32>
  %add = linalg.map
      ins(%matmul, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
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
