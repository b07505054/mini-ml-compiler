module {
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = hir.fused_qmatmul_bias_relu %arg0, %arg1, %arg2 {alignment = 128 : i32, fusion.candidate = "qmatmul_bias_relu", fusion.group = "matmul_bias_relu_0", input_layout = "NHWC", kernel.selection = "runtime_profile", lhs_scale = 0.00999999977 : f32, lhs_zero_point = 0 : i32, lowering.source = "profile_guided_int8_matmul_add_relu", quantization.mode = "per_channel", quantized_dtype = "i8", rhs_scale = 0.00999999977 : f32, rhs_zero_point = 0 : i32, weight_layout = "blocked_kc"} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}

