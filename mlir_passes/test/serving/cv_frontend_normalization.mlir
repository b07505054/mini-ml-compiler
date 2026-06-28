// FileCheck test for cv-frontend-normalization pass.
//
// Run via tools/run_mlir_pass_tests.sh, or directly:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(cv-frontend-normalization)' \
//   | FileCheck %s
//
// Two functions in a single module:
//   @yoloseg_inference — full pattern (conv2d + detect_head + prototype_head)
//                        should receive all cv.frontend.* attrs.
//   @backbone_only     — no heads present; gate fails; no attrs emitted.
//
// MLIR prints func.func attributes alphabetically on the function definition
// line; CHECK-SAME directives must follow the same alphabetical order.

// ---------------------------------------------------------------------------
// Case 1: Full YOLOSeg pattern — expects all cv.frontend.* attrs.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @yoloseg_inference
// CHECK-SAME:  cv.frontend.batch_norm_count = 1 : i64
// CHECK-SAME:  cv.frontend.concat_count = 1 : i64
// CHECK-SAME:  cv.frontend.conv2d_count = 5 : i64
// CHECK-SAME:  cv.frontend.detect_head_count = 1 : i64
// CHECK-SAME:  cv.frontend.kind = "raw_pseudo_cv_mlir"
// CHECK-SAME:  cv.frontend.model_family = "yoloseg"
// CHECK-SAME:  cv.frontend.normalization_status = "detected_not_rewritten"
// CHECK-SAME:  cv.frontend.prototype_head_count = 1 : i64
// CHECK-SAME:  cv.frontend.silu_count = 4 : i64
// CHECK-SAME:  cv.frontend.truth_boundary = "raw_pseudo_cv_mlir_not_full_onnx_importer"
// CHECK-SAME:  cv.frontend.upsample_count = 1 : i64
// Body ops must be preserved (detect-and-annotate, no rewrite).
// CHECK:       "cv.conv2d"
// CHECK:       "cv.detect_head"
// CHECK:       "cv.prototype_head"

// ---------------------------------------------------------------------------
// Case 2: Backbone-only function — gate fails; no cv.frontend.* attrs emitted.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @backbone_only
// CHECK-NOT:   cv.frontend.kind

module attributes {
  cv.model = "yoloseg-nano-fp16",
  cv.truth_boundary = "raw_pseudo_cv_mlir_not_full_onnx_importer"
} {

  // Full YOLOSeg-like pattern: detect gate passes.
  func.func @yoloseg_inference(%input: tensor<1x3x640x640xf16>)
      -> (tensor<1x84x8400xf16>, tensor<1x32x160x160xf16>) {

    %stem_conv = "cv.conv2d"(%input) {
      cv.source_op = "Conv", cv.in_channels = 3 : i64, cv.out_channels = 16 : i64
    } : (tensor<1x3x640x640xf16>) -> tensor<1x16x320x320xf16>

    %stem_bn = "cv.batch_norm"(%stem_conv) {
      cv.source_op = "BatchNormalization"
    } : (tensor<1x16x320x320xf16>) -> tensor<1x16x320x320xf16>

    %stem_act = "cv.silu"(%stem_bn) {
      cv.source_op = "Mul"
    } : (tensor<1x16x320x320xf16>) -> tensor<1x16x320x320xf16>

    %p3_conv = "cv.conv2d"(%stem_act) {
      cv.source_op = "Conv", cv.in_channels = 16 : i64, cv.out_channels = 32 : i64
    } : (tensor<1x16x320x320xf16>) -> tensor<1x32x160x160xf16>

    %p3_act = "cv.silu"(%p3_conv) {
      cv.source_op = "Mul"
    } : (tensor<1x32x160x160xf16>) -> tensor<1x32x160x160xf16>

    %p4_conv = "cv.conv2d"(%p3_act) {
      cv.source_op = "Conv", cv.in_channels = 32 : i64, cv.out_channels = 64 : i64
    } : (tensor<1x32x160x160xf16>) -> tensor<1x64x80x80xf16>

    %p4_act = "cv.silu"(%p4_conv) {
      cv.source_op = "Mul"
    } : (tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16>

    %p5_conv = "cv.conv2d"(%p4_act) {
      cv.source_op = "Conv", cv.in_channels = 64 : i64, cv.out_channels = 128 : i64
    } : (tensor<1x64x80x80xf16>) -> tensor<1x128x40x40xf16>

    %p5_act = "cv.silu"(%p5_conv) {
      cv.source_op = "Mul"
    } : (tensor<1x128x40x40xf16>) -> tensor<1x128x40x40xf16>

    %p5_up = "cv.upsample"(%p5_act) {
      cv.source_op = "Resize", cv.mode = "nearest"
    } : (tensor<1x128x40x40xf16>) -> tensor<1x128x80x80xf16>

    %neck_cat = "cv.concat"(%p5_up, %p4_act) {
      cv.source_op = "Concat", cv.axis = 1 : i64
    } : (tensor<1x128x80x80xf16>, tensor<1x64x80x80xf16>) -> tensor<1x192x80x80xf16>

    %neck_conv = "cv.conv2d"(%neck_cat) {
      cv.source_op = "Conv", cv.in_channels = 192 : i64, cv.out_channels = 64 : i64
    } : (tensor<1x192x80x80xf16>) -> tensor<1x64x80x80xf16>

    %detect_out = "cv.detect_head"(%neck_conv, %p4_act, %p5_act) {
      cv.num_classes = 80 : i64, cv.output_anchors = 8400 : i64
    } : (tensor<1x64x80x80xf16>, tensor<1x64x80x80xf16>, tensor<1x128x40x40xf16>)
        -> tensor<1x84x8400xf16>

    %proto_out = "cv.prototype_head"(%neck_conv) {
      cv.num_masks = 32 : i64
    } : (tensor<1x64x80x80xf16>) -> tensor<1x32x160x160xf16>

    return %detect_out, %proto_out
        : tensor<1x84x8400xf16>, tensor<1x32x160x160xf16>
  }

  // Backbone-only: no detect_head, no prototype_head — gate fails.
  func.func @backbone_only(%input: tensor<1x3x640x640xf16>)
      -> tensor<1x32x160x160xf16> {

    %c0 = "cv.conv2d"(%input) {
      cv.source_op = "Conv", cv.in_channels = 3 : i64, cv.out_channels = 16 : i64
    } : (tensor<1x3x640x640xf16>) -> tensor<1x16x320x320xf16>

    %a0 = "cv.silu"(%c0) {
      cv.source_op = "Mul"
    } : (tensor<1x16x320x320xf16>) -> tensor<1x16x320x320xf16>

    %c1 = "cv.conv2d"(%a0) {
      cv.source_op = "Conv", cv.in_channels = 16 : i64, cv.out_channels = 32 : i64
    } : (tensor<1x16x320x320xf16>) -> tensor<1x32x160x160xf16>

    return %c1 : tensor<1x32x160x160xf16>
  }

}
