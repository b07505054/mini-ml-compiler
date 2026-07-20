// RUN: mlir-opt %s --allow-unregistered-dialect --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)' | FileCheck %s

// quantization.candidate/profile.quantized_path are only a planning-time
// hint that an int8 kernel would be profitable; they do not themselves
// quantize the operands. Since %A/%B here are still tensor<128x128xf32>
// (no hir.quantize has run), hir-fusion-lowering must NOT honor the hint --
// doing so would construct hir.fused_qmatmul_bias_relu with f32 operands,
// which its own verifier rejects (expects i8 lhs/rhs). The correct,
// verifier-safe behavior is to fall back to the fp32 fused op.
func.func @main(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>,
  %bias: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %empty = tensor.empty() : tensor<128x128xf32>

  // CHECK-NOT: linalg.matmul
  // CHECK: hir.fused_matmul_bias_relu
  // CHECK-SAME: fusion.candidate = "matmul_bias_relu"
  // CHECK-SAME: lowering.source = "linalg.matmul_add_relu"
  // CHECK-NOT: hir.fused_qmatmul_bias_relu
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
