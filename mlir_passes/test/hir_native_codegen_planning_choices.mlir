// CHECK: quantization.plan_dtype = "fp32"
// CHECK: llvm.func @free
// CHECK: llvm.func @planned_fused
// CHECK: llvm.call @malloc
// CHECK: llvm.call @malloc
// CHECK: llvm.call @free
// CHECK: llvm.return

module attributes {
  llm.dtype = "fp32",
  target.supported_precisions = ["fp32", "fp16"]
} {
  func.func @planned_fused(
      %lhs: tensor<16x128xf32>, %rhs: tensor<128x64xf32>,
      %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %empty = tensor.empty() : tensor<16x64xf32>
    %mm = linalg.matmul
      ins(%lhs, %rhs : tensor<16x128xf32>, tensor<128x64xf32>)
      outs(%empty : tensor<16x64xf32>) -> tensor<16x64xf32>
    %add = linalg.map { arith.addf }
      ins(%mm, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
    %zero = arith.constant 0.0 : f32
    %relu = linalg.map
      ins(%add : tensor<16x64xf32>) outs(%empty : tensor<16x64xf32>)
      (%x: f32) {
        %y = arith.maximumf %x, %zero : f32
        linalg.yield %y : f32
      }
    return %relu : tensor<16x64xf32>
  }
}

// -----

// CHECK: quantization.plan_dtype = "fp16"
// CHECK-NOT: llvm.func @free
// CHECK: llvm.func @planned_unfused
// CHECK: llvm.call @malloc
// CHECK-NOT: llvm.call @malloc
// CHECK-NOT: llvm.call @free
// CHECK: llvm.return

module attributes {
  llm.dtype = "fp16",
  target.supported_precisions = ["fp16", "fp32"]
} {
  func.func @planned_unfused(
      %lhs: tensor<16x128xf32>, %rhs: tensor<128x64xf32>,
      %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %empty = tensor.empty() : tensor<16x64xf32>
    %mm = linalg.matmul
      ins(%lhs, %rhs : tensor<16x128xf32>, tensor<128x64xf32>)
      outs(%empty : tensor<16x64xf32>) -> tensor<16x64xf32>
    %add = linalg.map { arith.addf }
      ins(%mm, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
    %zero = arith.constant 0.0 : f32
    %relu = linalg.map
      ins(%add : tensor<16x64xf32>) outs(%empty : tensor<16x64xf32>)
      (%x: f32) {
        %y = arith.maximumf %x, %zero : f32
        linalg.yield %y : f32
      }
    return %relu : tensor<16x64xf32>
  }
}
