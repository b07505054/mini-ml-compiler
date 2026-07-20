// CHECK: module attributes
// CHECK-SAME: quantization.plan_dtype = "fp32"
// CHECK: llvm.func @free
// CHECK: llvm.func @malloc
// CHECK: llvm.func @planned
// CHECK: %[[TEMP:.*]] = llvm.call @malloc
// CHECK: %[[RESULT:.*]] = llvm.call @malloc
// CHECK: %[[TEMP_BASE:.*]] = llvm.extractvalue {{.*}}[0]
// CHECK: llvm.call @free(%[[TEMP_BASE]])
// CHECK-NOT: llvm.call @free(%[[RESULT]])
// CHECK: llvm.return
// CHECK-NOT: hir.
// CHECK-NOT: linalg.

module attributes {
  llm.dtype = "fp32",
  target.supported_precisions = ["fp32"]
} {
  func.func @planned(
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
