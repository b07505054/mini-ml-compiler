// RUN: mlir-opt %s --load-dialect-plugin=%plugin --load-pass-plugin=%plugin --pass-pipeline='builtin.module(stablehlo-compatible-rmsnorm-import)' | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// StableHLO-compatible frontend shape:
//   multiply -> reduce -> divide -> rsqrt -> multiply
// represented after decomposition in standard Linalg/Arith/Math form.
func.func @stablehlo_compatible_rmsnorm(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: hir.fused_rmsnorm
  // CHECK-SAME: frontend.source = "stablehlo_compatible_rmsnorm_decomposition"
  // CHECK-SAME: lowering.source = "llm.rmsnorm"
  // CHECK-NOT: math.rsqrt
  %zero = arith.constant 0.0 : f32
  %eps = arith.constant 0.000001 : f32
  %hidden = arith.constant 4.0 : f32
  %row_empty = tensor.empty() : tensor<2xf32>
  %row_init = linalg.fill ins(%zero : f32)
      outs(%row_empty : tensor<2xf32>) -> tensor<2xf32>
  %sum = linalg.generic {
      indexing_maps = [#map, #row],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<2x4xf32>)
      outs(%row_init : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sq = arith.mulf %in, %in : f32
      %next = arith.addf %out, %sq : f32
      linalg.yield %next : f32
  } -> tensor<2xf32>
  %out_empty = tensor.empty() : tensor<2x4xf32>
  %normalized = linalg.generic {
      indexing_maps = [#map, #row, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%x, %sum : tensor<2x4xf32>, tensor<2xf32>)
      outs(%out_empty : tensor<2x4xf32>) {
    ^bb0(%in: f32, %row_sum: f32, %out: f32):
      %mean = arith.divf %row_sum, %hidden : f32
      %var = arith.addf %mean, %eps : f32
      %inv = math.rsqrt %var : f32
      %y = arith.mulf %in, %inv : f32
      linalg.yield %y : f32
  } -> tensor<2x4xf32>
  return %normalized : tensor<2x4xf32>
}

