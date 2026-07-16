// Input HIR fixture for the AArch64 native-codegen vertical slice.
//
// This uses the same hir.fused_matmul_bias_relu op, in the same
// already-fused/annotated form, as mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir.
// The only addition is `llvm.emit_c_interface` on the function, which makes
// convert-func-to-llvm also emit a `_mlir_ciface_<fn>` wrapper taking
// pointers to StridedMemRefType-shaped descriptors, so this kernel can be
// called directly from C++ without guessing the raw unpacked-descriptor ABI.
//
// Shape: M=8, N=8, K=8.

func.func @matmul_bias_relu_8x8x8(
    %lhs: tensor<8x8xf32>,
    %rhs: tensor<8x8xf32>,
    %bias: tensor<8x8xf32>) -> tensor<8x8xf32> attributes { llvm.emit_c_interface } {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
