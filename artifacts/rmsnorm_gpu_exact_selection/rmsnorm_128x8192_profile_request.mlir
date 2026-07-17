// Profile-selection bridge input only. The current hir.fused_rmsnorm op is
// unweighted; weighted_rmsnorm identity is carried and validated by the exact
// GPU candidate contract. This file is not evidence of typed HIR equivalence.
func.func @main(%x: tensor<128x8192xf32>) -> tensor<128x8192xf32> {
  %0 = hir.fused_rmsnorm %x {fusion.candidate = "rmsnorm"} : (tensor<128x8192xf32>) -> tensor<128x8192xf32>
  return %0 : tensor<128x8192xf32>
}
