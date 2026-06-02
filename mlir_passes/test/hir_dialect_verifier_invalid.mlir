// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_rmsnorm(%x: tensor<16x768xf16>) -> tensor<16x768xf16> {
  // expected-error @+1 {{requires 'fusion.candidate' = "rmsnorm"}}
  %0 = hir.fused_rmsnorm %x {
    fusion.candidate = "not_rmsnorm",
    kernel.selection = "runtime_profile",
    lowering.source = "llm.rmsnorm"
  } : (tensor<16x768xf16>) -> tensor<16x768xf16>
  return %0 : tensor<16x768xf16>
}
