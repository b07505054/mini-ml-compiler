// Architecture C input: existing LLM linear subgraph, no sharding selected.
module {
  func.func @linear_bias_relu(
      %x: tensor<11x13xf32>, %w: tensor<13x17xf32>,
      %b: tensor<17xf32>) -> tensor<11x17xf32> {
    %0 = "hir.matmul"(%x, %w) : (tensor<11x13xf32>, tensor<13x17xf32>) -> tensor<11x17xf32>
    %1 = "hir.bias_add"(%0, %b) : (tensor<11x17xf32>, tensor<17xf32>) -> tensor<11x17xf32>
    %2 = "hir.relu"(%1) : (tensor<11x17xf32>) -> tensor<11x17xf32>
    return %2 : tensor<11x17xf32>
  }
}
