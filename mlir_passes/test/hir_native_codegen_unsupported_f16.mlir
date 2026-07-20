// expected-error @+4 {{hir-native-codegen cannot lower 'hir.fused_matmul_bias_relu': element type 'f16' is unsupported by HIRMatMulBiasReluToLinalg lowering; supported element type: f32; shape: 'tensor<2x3xf16>'}}
func.func @unsupported(
    %lhs: tensor<2x4xf16>, %rhs: tensor<4x3xf16>,
    %bias: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %result = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<2x4xf16>, tensor<4x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}
