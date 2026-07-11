// FileCheck test for cv-semantic-annotation on upstream MLIR.
//
// RUN: mlir-opt %s --load-pass-plugin=%plugin --load-dialect-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(cv-semantic-annotation)' | FileCheck %s

// CHECK-LABEL: func.func @semantic_yoloseg_like
// CHECK-SAME: cv.model_family = "yoloseg"
// CHECK-SAME: cv.semantic_annotation.source_name_dependency = "none"
// CHECK: cv.output_role = "segmentation_prototype"
// CHECK: cv.semantic_role = "segmentation_prototype"
// CHECK: cv.recognition_confidence = "high"
// CHECK: cv.output_role = "detection"
// CHECK: cv.region_id = "cv.region.detection_head"
// CHECK: cv.semantic_role = "detection_output"

module {
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
