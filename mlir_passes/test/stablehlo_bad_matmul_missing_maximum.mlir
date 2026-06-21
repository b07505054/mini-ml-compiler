module {
  func.func @bad_matmul_missing_maximum(
      %lhs: tensor<16x128xf32>,
      %rhs: tensor<128x64xf32>,
      %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %dot = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<16x128xf32>, tensor<128x64xf32>) -> tensor<16x64xf32>
    %add = stablehlo.add %dot, %bias : tensor<16x64xf32>
    return %add : tensor<16x64xf32>
  }
}
