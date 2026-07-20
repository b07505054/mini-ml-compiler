// CHECK: quantization.plan_dtype = "fp32"
// CHECK: llvm.func @padded
// CHECK: llvm.call @malloc
// CHECK: llvm.call @free
// CHECK: llvm.return
// CHECK-NOT: tensor.
// CHECK-NOT: memref.
// CHECK-NOT: linalg.
// CHECK-NOT: hir.

module attributes {
  llm.dtype = "fp32",
  target.supported_precisions = ["fp32", "fp16"]
} {
  func.func @padded(
      %lhs: tensor<15x31xf32>, %rhs: tensor<31x15xf32>,
      %bias: tensor<15x15xf32>) -> tensor<15x15xf32>
      attributes {llvm.emit_c_interface} {
    %empty = tensor.empty() : tensor<15x15xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<15x15xf32>) -> tensor<15x15xf32>
    %mm = linalg.matmul
      ins(%lhs, %rhs : tensor<15x31xf32>, tensor<31x15xf32>)
      outs(%init : tensor<15x15xf32>) -> tensor<15x15xf32>
    %addempty = tensor.empty() : tensor<15x15xf32>
    %add = linalg.map { arith.addf }
      ins(%mm, %bias : tensor<15x15xf32>, tensor<15x15xf32>)
      outs(%addempty : tensor<15x15xf32>)
    %reluempty = tensor.empty() : tensor<15x15xf32>
    %relu = linalg.map
      ins(%add : tensor<15x15xf32>)
      outs(%reluempty : tensor<15x15xf32>)
      (%x: f32) {
        %y = arith.maximumf %x, %zero : f32
        linalg.yield %y : f32
      }
    return %relu : tensor<15x15xf32>
  }
}
