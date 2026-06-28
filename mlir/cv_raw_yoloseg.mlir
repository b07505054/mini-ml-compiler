// cv_raw_yoloseg.mlir — Simplified YOLOSeg-nano pseudo-CV raw input.
//
// Truth boundary: raw_pseudo_cv_mlir_not_full_onnx_importer
//
// This is a hand-written pseudo-CV IR capturing the structural pattern of a
// YOLOv8n-seg backbone → PANet neck → detection head + prototype head.
// Op names use the "cv.*" namespace; --allow-unregistered-dialect is required.
//
// What this represents (simplified, not verified against a real ONNX export):
//   - 5 cv.conv2d ops: stem, P3, P4, P5 backbone blocks, plus neck channel-
//     reduction conv after concat.
//   - 1 cv.batch_norm: stem normalization (kept separate from SiLU for clarity).
//   - 4 cv.silu: one activation per backbone conv block (stem, P3, P4, P5).
//   - 1 cv.upsample: P5 features 40×40 → 80×80 for neck fusion.
//   - 1 cv.concat: fuse P5_up (128ch) and P4 (64ch) → 192ch.
//   - 1 cv.detect_head: multi-scale bbox + class prediction → 1×84×8400.
//   - 1 cv.prototype_head: mask prototype generation → 1×32×160×160.
//
// What this does NOT represent:
//   - C2f / CSP block internals (residual skip connections, pooling).
//   - Full multi-scale FPN (only P4/P5 neck shown; P3 omitted for brevity).
//   - Detect head post-processing (NMS, decode) — belongs in runtime.
//   - Any weights, scales, or calibration data.
//   - CoreML, ANE, or Metal execution behavior.
//
// Pipeline:
//   mlir-opt --allow-unregistered-dialect \
//     --load-pass-plugin=<plugin> \
//     --pass-pipeline='builtin.module(cv-frontend-normalization)' \
//     mlir/cv_raw_yoloseg.mlir
//
// See: mlir_passes/test/serving/cv_frontend_normalization.mlir

module attributes {
  cv.model = "yoloseg-nano-fp16",
  cv.truth_boundary = "raw_pseudo_cv_mlir_not_full_onnx_importer"
} {

  func.func @yoloseg_inference(%input: tensor<1x3x640x640xf16>)
      -> (tensor<1x84x8400xf16>, tensor<1x32x160x160xf16>) {

    // ---- Stem block: Conv → BN → SiLU, stride 2 → 320×320 ----
    %stem_conv = "cv.conv2d"(%input) {
      cv.source_op = "Conv",
      cv.onnx_node_name = "/model/0/conv/Conv",
      cv.in_channels = 3 : i64,
      cv.out_channels = 16 : i64,
      cv.stride_h = 2 : i64,
      cv.stride_w = 2 : i64
    } : (tensor<1x3x640x640xf16>) -> tensor<1x16x320x320xf16>

    %stem_bn = "cv.batch_norm"(%stem_conv) {
      cv.source_op = "BatchNormalization",
      cv.onnx_node_name = "/model/0/bn/BatchNormalization"
    } : (tensor<1x16x320x320xf16>) -> tensor<1x16x320x320xf16>

    %stem_act = "cv.silu"(%stem_bn) {
      cv.source_op = "Mul",
      cv.onnx_node_name = "/model/0/act/Mul"
    } : (tensor<1x16x320x320xf16>) -> tensor<1x16x320x320xf16>

    // ---- P3 block: Conv → SiLU, stride 2 → 160×160 ----
    %p3_conv = "cv.conv2d"(%stem_act) {
      cv.source_op = "Conv",
      cv.onnx_node_name = "/model/2/conv/Conv",
      cv.in_channels = 16 : i64,
      cv.out_channels = 32 : i64,
      cv.stride_h = 2 : i64,
      cv.stride_w = 2 : i64
    } : (tensor<1x16x320x320xf16>) -> tensor<1x32x160x160xf16>

    %p3_act = "cv.silu"(%p3_conv) {
      cv.source_op = "Mul",
      cv.onnx_node_name = "/model/2/act/Mul"
    } : (tensor<1x32x160x160xf16>) -> tensor<1x32x160x160xf16>

    // ---- P4 block: Conv → SiLU, stride 2 → 80×80 ----
    %p4_conv = "cv.conv2d"(%p3_act) {
      cv.source_op = "Conv",
      cv.onnx_node_name = "/model/4/conv/Conv",
      cv.in_channels = 32 : i64,
      cv.out_channels = 64 : i64,
      cv.stride_h = 2 : i64,
      cv.stride_w = 2 : i64
    } : (tensor<1x32x160x160xf16>) -> tensor<1x64x80x80xf16>

    %p4_act = "cv.silu"(%p4_conv) {
      cv.source_op = "Mul",
      cv.onnx_node_name = "/model/4/act/Mul"
    } : (tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16>

    // ---- P5 block: Conv → SiLU, stride 2 → 40×40 ----
    %p5_conv = "cv.conv2d"(%p4_act) {
      cv.source_op = "Conv",
      cv.onnx_node_name = "/model/6/conv/Conv",
      cv.in_channels = 64 : i64,
      cv.out_channels = 128 : i64,
      cv.stride_h = 2 : i64,
      cv.stride_w = 2 : i64
    } : (tensor<1x64x80x80xf16>) -> tensor<1x128x40x40xf16>

    %p5_act = "cv.silu"(%p5_conv) {
      cv.source_op = "Mul",
      cv.onnx_node_name = "/model/6/act/Mul"
    } : (tensor<1x128x40x40xf16>) -> tensor<1x128x40x40xf16>

    // ---- Neck: upsample P5 → 80×80, concat with P4, reduce channels ----
    %p5_up = "cv.upsample"(%p5_act) {
      cv.source_op = "Resize",
      cv.onnx_node_name = "/model/10/Resize",
      cv.scale_h = 2 : i64,
      cv.scale_w = 2 : i64,
      cv.mode = "nearest"
    } : (tensor<1x128x40x40xf16>) -> tensor<1x128x80x80xf16>

    %neck_cat = "cv.concat"(%p5_up, %p4_act) {
      cv.source_op = "Concat",
      cv.onnx_node_name = "/model/11/Concat",
      cv.axis = 1 : i64
    } : (tensor<1x128x80x80xf16>, tensor<1x64x80x80xf16>) -> tensor<1x192x80x80xf16>

    %neck_conv = "cv.conv2d"(%neck_cat) {
      cv.source_op = "Conv",
      cv.onnx_node_name = "/model/12/conv/Conv",
      cv.in_channels = 192 : i64,
      cv.out_channels = 64 : i64,
      cv.stride_h = 1 : i64,
      cv.stride_w = 1 : i64
    } : (tensor<1x192x80x80xf16>) -> tensor<1x64x80x80xf16>

    // ---- Detection head: multi-scale bbox + class prediction ----
    // Inputs: neck_conv (P3-equiv at 80×80), p4_act (80×80), p5_act (40×40).
    // Output: 84 = 4 coords + 80 COCO classes, 8400 anchors (80²+40²+20²).
    %detect_out = "cv.detect_head"(%neck_conv, %p4_act, %p5_act) {
      cv.source_op = "Detect",
      cv.onnx_node_name = "/model/22/Detect",
      cv.num_classes = 80 : i64,
      cv.num_coords = 4 : i64,
      cv.output_anchors = 8400 : i64
    } : (tensor<1x64x80x80xf16>, tensor<1x64x80x80xf16>, tensor<1x128x40x40xf16>)
        -> tensor<1x84x8400xf16>

    // ---- Segmentation prototype head ----
    // Generates 32 prototype masks at 160×160 (4× spatial upscale from 40×40).
    %proto_out = "cv.prototype_head"(%neck_conv) {
      cv.source_op = "Proto",
      cv.onnx_node_name = "/model/22/Proto",
      cv.num_masks = 32 : i64
    } : (tensor<1x64x80x80xf16>) -> tensor<1x32x160x160xf16>

    return %detect_out, %proto_out
        : tensor<1x84x8400xf16>, tensor<1x32x160x160xf16>
  }

}
