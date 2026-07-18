// Project-owned attributes because installed LLVM 21 has legacy mesh, not
// current upstream shard. This is not a new dialect named mesh.
module attributes {
  hir.sharding.mesh = {name = "cpu_mesh", axis = "cpu", size = 8 : i64},
  hir.sharding.rank_mapping = "rank i -> pinned logical CPU i"
} {
  func.func @linear_bias_relu(
      %x: tensor<11x13xf32>, %w: tensor<13x17xf32>,
      %b: tensor<17xf32>) -> tensor<11x17xf32>
      attributes {
        hir.sharding.strategy = "split_m",
        hir.sharding.tensor_dimension = 0 : i64,
        hir.sharding.uneven_policy = "balanced_remainder",
        hir.sharding.provenance = "compiler_inferred",
        hir.sharding.collective = "none_direct_disjoint_row_assembly"
      } {
    %0 = "hir.matmul"(%x, %w) : (tensor<11x13xf32>, tensor<13x17xf32>) -> tensor<11x17xf32>
    %1 = "hir.bias_add"(%0, %b) : (tensor<11x17xf32>, tensor<17xf32>) -> tensor<11x17xf32>
    %2 = "hir.relu"(%1) : (tensor<11x17xf32>) -> tensor<11x17xf32>
    return %2 : tensor<11x17xf32>
  }
}
