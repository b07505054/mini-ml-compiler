module {
  func.func @bad_rmsnorm_f16(%x: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %mul0 = stablehlo.multiply %x, %x : tensor<2x4xf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %sum = stablehlo.reduce(%mul0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf16>, tensor<f16>) -> tensor<2xf16>
    %mean = stablehlo.divide %sum, %cst : tensor<2xf16>
    %inv = stablehlo.rsqrt %mean : tensor<2xf16>
    %out = stablehlo.multiply %x, %inv : tensor<2x4xf16>
    return %out : tensor<2x4xf16>
  }
}
