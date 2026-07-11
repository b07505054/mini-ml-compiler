// RUN: mlir-opt %s --load-pass-plugin=%plugin --matmul-bias-relu-fusion | FileCheck %s
//
// Benchmark-workload variant of matmul_bias_relu.mlir: M=N=K=128, matching
// the shape measured by apps/run_mlir_fused_kernel_benchmark.cpp so the
// profile-guided kernel selection has an exact-shape profile match. The bias
// operand is materialized to the result shape [M,N], mirroring the base test;
// the runtime kernel contract for the fused op consumes a rank-1 bias [N].

func.func @main(
  %A: tensor<128x128xf32>,
  %B: tensor<128x128xf32>,
  %bias: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %empty = tensor.empty() : tensor<128x128xf32>

  %mm = linalg.matmul
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: linalg.matmul
  // CHECK-SAME: fusion.candidate = "matmul_bias_relu"
  // CHECK-SAME: fusion.group = "matmul_bias_relu_0"
  // CHECK-SAME: fusion.role = "producer"
  // CHECK: linalg.map
  // CHECK-SAME: fusion.group = "matmul_bias_relu_0"
  // CHECK-SAME: fusion.role = "bias_add"
  // CHECK: linalg.map
  // CHECK-SAME: fusion.group = "matmul_bias_relu_0"
  // CHECK-SAME: fusion.role = "activation"

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
