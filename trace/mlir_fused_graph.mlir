module {
  func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<1x64xf32>) -> tensor<1x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = "hir.fused_matmul_bias_relu"(%arg0, %arg1, %arg2) {fusion.candidate = "matmul_bias_relu", fusion.group = "matmul_bias_relu_0", kernel.selection = "runtime_profile", lowering.source = "linalg.matmul_add_relu"} : (tensor<1x128xf32>, tensor<128x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
    return %1 : tensor<1x64xf32>
  }
}

