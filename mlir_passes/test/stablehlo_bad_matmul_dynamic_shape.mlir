module {
  func.func @bad_matmul_dynamic_shape(
      %lhs: tensor<?x128xf32>,
      %rhs: tensor<128x64xf32>,
      %bias: tensor<?x64xf32>) -> tensor<?x64xf32> {
    %dot = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<?x128xf32>, tensor<128x64xf32>) -> tensor<?x64xf32>
    %add = stablehlo.add %dot, %bias : tensor<?x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<?x64xf32>
    %out = stablehlo.maximum %add, %zero : tensor<?x64xf32>
    return %out : tensor<?x64xf32>
  }
}
