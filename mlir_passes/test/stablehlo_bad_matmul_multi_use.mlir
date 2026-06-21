module {
  func.func @bad_matmul_multi_use(
      %lhs: tensor<16x128xf32>,
      %rhs: tensor<128x64xf32>,
      %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %dot = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<16x128xf32>, tensor<128x64xf32>) -> tensor<16x64xf32>
    %add = stablehlo.add %dot, %bias : tensor<16x64xf32>
    %extra = stablehlo.add %dot, %bias : tensor<16x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x64xf32>
    %out = stablehlo.maximum %add, %zero : tensor<16x64xf32>
    return %out : tensor<16x64xf32>
  }
}
