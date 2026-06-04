// StableHLO textual subset fixture consumed by tools/import_stablehlo_subset.py.
// This file intentionally keeps stablehlo.* op names even when stablehlo-opt is
// not installed, so the repo can test the frontend boundary without vendoring
// OpenXLA tooling.

module {
  func.func @stablehlo_rmsnorm(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %mul0 = "stablehlo.multiply"(%x, %x) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %sum = "stablehlo.reduce"(%mul0) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<2x4xf32>) -> tensor<2xf32>
    %mean = "stablehlo.divide"(%sum) : (tensor<2xf32>) -> tensor<2xf32>
    %inv = "stablehlo.rsqrt"(%mean) : (tensor<2xf32>) -> tensor<2xf32>
    %out = "stablehlo.multiply"(%x, %inv) : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}

