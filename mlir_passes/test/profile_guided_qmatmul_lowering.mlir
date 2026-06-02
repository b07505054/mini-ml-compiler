// RUN: mlir-opt %s --allow-unregistered-dialect --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)' | FileCheck %s

func.func @main(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>,
  %bias: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %empty = tensor.empty() : tensor<128x128xf32>

  // CHECK-NOT: linalg.matmul
  // CHECK: hir.fused_qmatmul_bias_relu
  // CHECK-SAME: alignment = 128
  // CHECK-SAME: fusion.candidate = "qmatmul_bias_relu"
  // CHECK-SAME: input_layout = "NHWC"
  // CHECK-SAME: quantization.mode = "per_channel"
  // CHECK-SAME: quantized_dtype = "i8"
  // CHECK-SAME: weight_layout = "blocked_kc"
  %mm = linalg.matmul {
      profile.quantized_path = "faster",
      quantization.candidate = "int8"
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

  %add = linalg.map
      ins(%mm, %bias : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%empty : tensor<128x128xf32>)
      (%x: f32, %b: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32
  %relu = linalg.map
      ins(%add : tensor<128x128xf32>)
      outs(%empty : tensor<128x128xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<128x128xf32>
}
