// FileCheck test for CVShapeInferencePass.
//
// Run via tools/run_mlir_pass_tests.sh, or directly:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(cv-shape-inference)' \
//   | FileCheck %s
//
// Three functions in one module:
//   @tiny_cv    — two static cv.* ops; both annotated, total bytes = 64.
//   @has_skip   — one static + one dynamic-dim cv.* op; one annotated, one skipped.
//   @no_cv      — no cv.* ops; gate fails; no cv.shape_inference.* attrs set.
//
// Function attrs are printed alphabetically on the func.func definition line;
// CHECK-SAME directives follow the same order.
// Per-op attrs use plain directives (not CHECK-SAME) to tolerate line wrapping.
//
// Shape arithmetic for @tiny_cv:
//   tensor<1x4x2x2xf16>: elements = 1*4*2*2 = 16, bytes = 16*2 = 32
//   Two ops: total = 64 bytes.
//
// Shape arithmetic for @has_skip:
//   cv.silu output tensor<1x4x2x2xf16>: 16 elements, 32 bytes  -> annotated
//   cv.pooling output tensor<1x4x?x?xf16>: dynamic dims         -> skipped

// ---------------------------------------------------------------------------
// Case 1: @tiny_cv — two static ops; full function annotation expected.
// Function attrs alphabetical order:
//   cv.shape_inference.annotated_ops         (a)
//   cv.shape_inference.skipped_ops           (sk)
//   cv.shape_inference.status                (st)
//   cv.shape_inference.total_bytes_estimate  (to)
//   cv.shape_inference.truth_boundary        (tr)
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @tiny_cv
// CHECK-SAME:  cv.shape_inference.annotated_ops = 2 : i64
// CHECK-SAME:  cv.shape_inference.skipped_ops = 0 : i64
// CHECK-SAME:  cv.shape_inference.status = "completed"
// CHECK-SAME:  cv.shape_inference.total_bytes_estimate = 64 : i64
// CHECK-SAME:  cv.shape_inference.truth_boundary = "static_ranked_tensor_shape_metadata_not_runtime_allocation"
// Per-op checks for cv.conv2d (elements=16, bytes=32):
// CHECK: "cv.conv2d"
// CHECK: cv.bytes_estimate = 32 : i64
// CHECK: cv.num_elements = 16 : i64
// Per-op checks for cv.silu (elements=16, bytes=32):
// CHECK: "cv.silu"
// CHECK: cv.bytes_estimate = 32 : i64
// CHECK: cv.num_elements = 16 : i64

// ---------------------------------------------------------------------------
// Case 2: @has_skip — one annotated op, one dynamic-dim op (skipped).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @has_skip
// CHECK-SAME:  cv.shape_inference.annotated_ops = 1 : i64
// CHECK-SAME:  cv.shape_inference.skipped_ops = 1 : i64
// CHECK-SAME:  cv.shape_inference.status = "completed"
// CHECK-SAME:  cv.shape_inference.total_bytes_estimate = 32 : i64
// CHECK-SAME:  cv.shape_inference.truth_boundary = "static_ranked_tensor_shape_metadata_not_runtime_allocation"

// ---------------------------------------------------------------------------
// Case 3: @no_cv — no cv.* ops; gate fails; no shape_inference attrs.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_cv
// CHECK-NOT:   cv.shape_inference.status

module {

  // Case 1: two static cv.* ops.
  func.func @tiny_cv(%x: tensor<1x1x2x2xf16>)
      -> (tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>) {
    %c = "cv.conv2d"(%x) {
      cv.source_op = "Conv"
    } : (tensor<1x1x2x2xf16>) -> tensor<1x4x2x2xf16>
    %s = "cv.silu"(%c) {
      cv.source_op = "Mul"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %c, %s : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>
  }

  // Case 2: one static cv.* op (annotated), one dynamic-dim cv.* op (skipped).
  func.func @has_skip(%x: tensor<1x4x2x2xf16>)
      -> (tensor<1x4x2x2xf16>, tensor<1x4x?x?xf16>) {
    %s = "cv.silu"(%x) {
      cv.source_op = "Mul"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %p = "cv.pooling"(%x) {
      cv.source_op = "Pool"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x?x?xf16>
    return %s, %p : tensor<1x4x2x2xf16>, tensor<1x4x?x?xf16>
  }

  // Case 3: no cv.* ops; pass gate fails; no cv.shape_inference.* attrs set.
  func.func @no_cv(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    return %x : tensor<1x4x2x2xf16>
  }

}
