// RUN: mlir-opt %s --allow-unregistered-dialect --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)' | FileCheck %s

func.func @main(
  %A: tensor<16x32xf32>,
  %bias: tensor<16x16xf32>
) -> tensor<16x16xf32> {
  %B = arith.constant dense<0.0> : tensor<32x16xf32>
  %empty = tensor.empty() : tensor<16x16xf32>

  // CHECK-NOT: linalg.matmul
  // CHECK: hir.fused_matmul_bias_relu
  // CHECK-SAME: kernel.selection = "sparse_2_4_profile"
  // CHECK-SAME: sparse.candidate = "2_4"
  // CHECK-SAME: sparse.legal = true
  // CHECK-SAME: target.sparse_axis = "rhs_k"
  // CHECK-SAME: target.sparse_group_size = 4
  // CHECK-SAME: target.sparse_layout = "structured_2_4"
  // CHECK-SAME: target.sparse_max_nonzero = 2
  %mm = linalg.matmul {
      profile.sparse_2_4_path = "faster",
      sparse.candidate = "2_4"
    }
    ins(%A, %B : tensor<16x32xf32>, tensor<32x16xf32>)
    outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>

  %add = linalg.map
      ins(%mm, %bias : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>)
      (%x: f32, %b: f32, %out: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }

  %zero = arith.constant 0.0 : f32
  %relu = linalg.map
      ins(%add : tensor<16x16xf32>)
      outs(%empty : tensor<16x16xf32>)
      (%x: f32, %out: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }

  return %relu : tensor<16x16xf32>
}
