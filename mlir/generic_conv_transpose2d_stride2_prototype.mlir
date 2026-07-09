// Prototype for the selected GenericGraphIR nn.conv_transpose2d subset:
// NCHW FP32, group 1, 2x2 kernel/stride, unit dilation, and zero padding.

// CHECK-DAG: #[[INPUT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2 floordiv 2, d3 floordiv 2)>
// CHECK-DAG: #[[WEIGHT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d1, d2 mod 2, d3 mod 2)>
// CHECK-LABEL: func.func @generic_conv_transpose2d_stride2
// CHECK: %[[BIAS_INIT:.*]] = linalg.generic
// CHECK: linalg.yield %{{.*}} : f32
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[INPUT_MAP]], #[[WEIGHT_MAP]],
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: return %[[RESULT]]

module {
  func.func @generic_conv_transpose2d_stride2(
      %input: tensor<1x64x80x80xf32>,
      %weight: tensor<64x64x2x2xf32>,
      %bias: tensor<64xf32>) -> tensor<1x64x160x160xf32> {
    %empty = tensor.empty() : tensor<1x64x160x160xf32>
    %bias_init = linalg.generic {
      indexing_maps = [
        affine_map<(n, oc, oh, ow) -> (oc)>,
        affine_map<(n, oc, oh, ow) -> (n, oc, oh, ow)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%bias : tensor<64xf32>)
      outs(%empty : tensor<1x64x160x160xf32>) {
    ^bb0(%bias_value: f32, %unused: f32):
      linalg.yield %bias_value : f32
    } -> tensor<1x64x160x160xf32>

    %result = linalg.generic {
      indexing_maps = [
        affine_map<(n, oc, oh, ow, ic) ->
          (n, ic, oh floordiv 2, ow floordiv 2)>,
        affine_map<(n, oc, oh, ow, ic) ->
          (ic, oc, oh mod 2, ow mod 2)>,
        affine_map<(n, oc, oh, ow, ic) -> (n, oc, oh, ow)>
      ],
      iterator_types = [
        "parallel", "parallel", "parallel", "parallel", "reduction"
      ]
    } ins(
        %input, %weight
        : tensor<1x64x80x80xf32>, tensor<64x64x2x2xf32>)
      outs(%bias_init : tensor<1x64x160x160xf32>) {
    ^bb0(%input_value: f32, %weight_value: f32, %acc: f32):
      %product = arith.mulf %input_value, %weight_value : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<1x64x160x160xf32>

    return %result : tensor<1x64x160x160xf32>
  }
}
