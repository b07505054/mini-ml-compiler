// StableHLO textual subset fixture consumed by tools/import_stablehlo_subset.py.

module {
  func.func @stablehlo_matmul_bias_relu(
      %lhs: tensor<16x128xf32>,
      %rhs: tensor<128x64xf32>,
      %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %dot = "stablehlo.dot_general"(%lhs, %rhs) : (tensor<16x128xf32>, tensor<128x64xf32>) -> tensor<16x64xf32>
    %add = "stablehlo.add"(%dot, %bias) : (tensor<16x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %zero = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %out = "stablehlo.maximum"(%add, %zero) : (tensor<16x64xf32>, tensor<f32>) -> tensor<16x64xf32>
    return %out : tensor<16x64xf32>
  }
}

