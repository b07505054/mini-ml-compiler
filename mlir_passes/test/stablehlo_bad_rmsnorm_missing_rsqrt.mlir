module {
  func.func @bad_rmsnorm_missing_rsqrt(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %mul0 = stablehlo.multiply %x, %x : tensor<2x4xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%mul0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean = stablehlo.divide %sum, %cst : tensor<2xf32>
    %out = stablehlo.multiply %x, %mean : tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
