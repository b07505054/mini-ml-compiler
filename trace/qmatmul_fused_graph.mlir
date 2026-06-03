module {
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = hir.fused_qmatmul_bias_relu %arg0, %arg1, %arg2 {alignment = 128 : i32, fusion.candidate = "qmatmul_bias_relu", fusion.group = "matmul_bias_relu_0", input_layout = "NHWC", kernel.selection = "runtime_profile", lhs_scale = 0.00999999977 : f32, lhs_zero_point = 0 : i32, lowering.source = "profile_guided_int8_matmul_add_relu", quantization.mode = "per_channel", quantized_dtype = "i8", rhs_scale = 0.00999999977 : f32, rhs_zero_point = 0 : i32, target.alignment = 128 : i32, target.collective = "none", target.memory_hierarchy = "global_sram_register", target.model = "sparsecore_like_v1", target.sparse_layout = "dense_or_2_4", target.sram_kb = 256 : i32, target.tile_k = 32 : i32, target.tile_m = 16 : i32, target.tile_n = 16 : i32, target.vector_bytes = 128 : i32, weight_layout = "blocked_kc"} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}

