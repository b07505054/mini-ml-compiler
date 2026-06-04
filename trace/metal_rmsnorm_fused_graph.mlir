module {
  func.func @main(%arg0: tensor<16x4096xf32>) -> tensor<16x4096xf32> {
    %0 = hir.fused_rmsnorm %arg0 {fusion.candidate = "rmsnorm", fusion.group = "rmsnorm_0", kernel.selection = "runtime_profile", lowering.source = "llm.rmsnorm"} : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
    return %0 : tensor<16x4096xf32>
  }
}

