module {
  func.func @main(%arg0: tensor<16x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x64xf32>
    %1 = hir.fused_matmul_bias_relu %arg0, %arg1, %arg2 {fusion.candidate = "matmul_bias_relu", fusion.group = "matmul_bias_relu_0", kernel.selection = "runtime_profile", lowering.source = "linalg.matmul_add_relu", target.alignment = 128 : i32, target.collective = "none", target.memory_hierarchy = "global_sram_register", target.model = "sparsecore_like_v1", target.sparse_layout = "dense_or_2_4", target.sram_kb = 256 : i32, target.tile_k = 32 : i32, target.tile_m = 16 : i32, target.tile_n = 16 : i32, target.vector_bytes = 128 : i32} : (tensor<16x128xf32>, tensor<128x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    return %1 : tensor<16x64xf32>
  }
}

