// RUN: mlir-opt %s --load-dialect-plugin=%plugin --verify-diagnostics

func.func @bad_quantized_matmul(
  %lhs: tensor<128x128xi8>,
  %rhs: tensor<128x128xi8>
) -> tensor<128x128xi32> {
  // expected-error @+1 {{requires string attribute 'quantized_dtype'}}
  %0 = hir.qmatmul %lhs, %rhs {
    lhs_scale = 1.000000e-02 : f32,
    lhs_zero_point = 0 : i32,
    rhs_scale = 1.000000e-02 : f32,
    rhs_zero_point = 0 : i32
  } : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}
