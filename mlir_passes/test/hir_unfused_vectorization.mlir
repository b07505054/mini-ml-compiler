// CHECK-NOT: hir.fused_matmul_bias_relu
// CHECK: vector.contract
// CHECK: linalg.map
// CHECK: linalg.map
// CHECK-NOT: hir.fused_matmul_bias_relu

module attributes {
  llm.dtype = "fp16",
  target.supported_precisions = ["fp16", "fp32"]
} {
  func.func @unfused_vector(
      %lhs: tensor<16x32xf32>, %rhs: tensor<32x16xf32>,
      %bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %empty = tensor.empty() : tensor<16x16xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %mm = linalg.matmul
      ins(%lhs, %rhs : tensor<16x32xf32>, tensor<32x16xf32>)
      outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %addempty = tensor.empty() : tensor<16x16xf32>
    %add = linalg.map { arith.addf }
      ins(%mm, %bias : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%addempty : tensor<16x16xf32>)
    %reluempty = tensor.empty() : tensor<16x16xf32>
    %relu = linalg.map
      ins(%add : tensor<16x16xf32>)
      outs(%reluempty : tensor<16x16xf32>)
      (%x: f32) {
        %y = arith.maximumf %x, %zero : f32
        linalg.yield %y : f32
      }
    return %relu : tensor<16x16xf32>
  }
}
