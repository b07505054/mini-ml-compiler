// Prototype for the selected GenericGraphIR nn.resize lowering subset:
// NCHW, nearest, asymmetric, floor, static 2x spatial scale.

// CHECK-LABEL: func.func @generic_resize_nearest_2x
// CHECK: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: %[[RESULT:.*]] = tensor.generate
// CHECK: %[[IH:.*]] = arith.divui %{{.*}}, %[[TWO]] : index
// CHECK: %[[IW:.*]] = arith.divui %{{.*}}, %[[TWO]] : index
// CHECK: %[[VALUE:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}, %[[IH]], %[[IW]]]
// CHECK: tensor.yield %[[VALUE]] : f32
// CHECK: return %[[RESULT]]

module {
  func.func @generic_resize_nearest_2x(
      %input: tensor<1x256x20x20xf32>) -> tensor<1x256x40x40xf32> {
    %two = arith.constant 2 : index
    %result = tensor.generate {
    ^bb0(%n: index, %c: index, %oh: index, %ow: index):
      %ih = arith.divui %oh, %two : index
      %iw = arith.divui %ow, %two : index
      %value = tensor.extract %input[%n, %c, %ih, %iw]
          : tensor<1x256x20x20xf32>
      tensor.yield %value : f32
    } : tensor<1x256x40x40xf32>
    return %result : tensor<1x256x40x40xf32>
  }
}
