// RUN: mlir-opt %s --load-pass-plugin=%plugin --pass-pipeline='builtin.module(quantization-planning,matmul-bias-relu-fusion)' | FileCheck %s

module attributes {
  llm.dtype = "fp32",
  target.supported_precisions = ["fp32"]
} {
func.func @main(
  %A: tensor<17x65xf32>,
  %B: tensor<65x33xf32>,
  %bias: tensor<17x33xf32>
) -> tensor<17x33xf32> {
  %empty = tensor.empty() : tensor<17x33xf32>

  %mm = linalg.matmul
    ins(%A, %B : tensor<17x65xf32>, tensor<65x33xf32>)
    outs(%empty : tensor<17x33xf32>) -> tensor<17x33xf32>

  %add = linalg.map
      ins(%mm, %bias : tensor<17x33xf32>, tensor<17x33xf32>)
      outs(%empty : tensor<17x33xf32>)
      (%x: f32, %b: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32
  %relu = linalg.map
      ins(%add : tensor<17x33xf32>)
      outs(%empty : tensor<17x33xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<17x33xf32>
}
}

// CHECK: linalg.matmul
// CHECK-SAME: candidate_id = "fused_scalar_baseline"
// CHECK-SAME: requires_padding = true
// CHECK-SAME: native.cost_model.selected_candidate = "unfused_scalar_baseline"
// CHECK-NOT: fusion.candidate
