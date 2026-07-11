// FileCheck test for minimum real-CV planning attrs over upstream MLIR.
//
// RUN: mlir-opt %s --load-pass-plugin=%plugin --load-dialect-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(cv-semantic-annotation,cv-execution-plan-attrs)' | FileCheck %s

// CHECK-LABEL: func.func @semantic_yoloseg_like
// CHECK-SAME: cv.execution_plan.status = "completed"
// CHECK-SAME: cv.memory.estimated_total_tensor_bytes
// CHECK-SAME: execution_provider.primary = "cpu"
// CHECK-SAME: representation.effective_dtype = "f32"
// CHECK-SAME: serving.policy = "cv_full_graph"
// CHECK: quant.strategy = "none"
// CHECK: layout.effective_layout = "nchw"
// CHECK: kernel.exists = false
// CHECK: kernel.lowering_status = "unsupported"

module attributes {
  target.allowed_backends = ["cpu"],
  target.preferred_backend = "cpu",
  target.profile_id = "unit-test-cpu"
} {
  func.func @semantic_yoloseg_like(
      %boxes: tensor<1x4x8400xf32>,
      %classes: tensor<1x80x8400xf32>,
      %masks: tensor<1x32x8400xf32>,
      %proto: tensor<1x32x160x160xf32>)
      -> (tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>) {
    %proto_empty = tensor.empty() : tensor<1x32x160x160xf32>
    %proto_out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%proto : tensor<1x32x160x160xf32>)
      outs(%proto_empty : tensor<1x32x160x160xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x160x160xf32>

    %det_empty = tensor.empty() : tensor<1x116x8400xf32>
    %det0 = tensor.insert_slice %boxes into %det_empty[0, 0, 0] [1, 4, 8400] [1, 1, 1]
      : tensor<1x4x8400xf32> into tensor<1x116x8400xf32>
    %det1 = tensor.insert_slice %classes into %det0[0, 4, 0] [1, 80, 8400] [1, 1, 1]
      : tensor<1x80x8400xf32> into tensor<1x116x8400xf32>
    %det2 = tensor.insert_slice %masks into %det1[0, 84, 0] [1, 32, 8400] [1, 1, 1]
      : tensor<1x32x8400xf32> into tensor<1x116x8400xf32>
    return %det2, %proto_out : tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>
  }
}
