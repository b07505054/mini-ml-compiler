// RUN: mlir-opt %s --allow-unregistered-dialect --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering)' | FileCheck %s

func.func @main(
  %A: tensor<127x255xf32>,
  %B: tensor<255x129xf32>,
  %bias: tensor<127x129xf32>
) -> tensor<127x129xf32> {
  %empty = tensor.empty() : tensor<127x129xf32>

  // CHECK-NOT: linalg.matmul
  // CHECK: hir.fused_matmul_bias_relu
  // CHECK-SAME: target.original_k = 255
  // CHECK-SAME: target.original_m = 127
  // CHECK-SAME: target.original_n = 129
  // CHECK-SAME: target.padded_k = 256
  // CHECK-SAME: target.padded_m = 128
  // CHECK-SAME: target.padded_n = 144
  // CHECK-SAME: target.padding = "pad_to_tile_with_crop"
  // CHECK-SAME: target.valid_region = "original_m_n"
  // CHECK-SAME: (tensor<127x255xf32>, tensor<255x129xf32>, tensor<127x129xf32>) -> tensor<127x129xf32>
  %mm = linalg.matmul
    ins(%A, %B : tensor<127x255xf32>, tensor<255x129xf32>)
    outs(%empty : tensor<127x129xf32>) -> tensor<127x129xf32>

  %add = linalg.map
      ins(%mm, %bias : tensor<127x129xf32>, tensor<127x129xf32>)
      outs(%empty : tensor<127x129xf32>)
      (%x: f32, %b: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32

  %relu = linalg.map
      ins(%add : tensor<127x129xf32>)
      outs(%empty : tensor<127x129xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<127x129xf32>
}
