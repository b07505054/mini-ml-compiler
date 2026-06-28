// FileCheck test for CVMemoryPlanningPass.
//
// Run via tools/run_mlir_pass_tests.sh, or directly:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(cv-memory-planning)' \
//   | FileCheck %s
//
// cv.bytes_estimate attrs are hand-authored so this test is independent of
// CVShapeInferencePass.  All plannable ops use tensor<1x4x2x2xf16>
// (16 elements * 2 bytes = 32 bytes each).
//
// Three functions:
//   @chain       -- linear A->B->C: C reuses A's buffer (first-fit).
//   @branch      -- fan-out A->{B,C}: A lifetime extended to last consumer.
//   @missing_bytes -- one planned op, one skipped (no cv.bytes_estimate).
//
// Attr alphabetical ordering on func.func (all prefixed cv.memory_plan.):
//   buffer_count < peak_memory_bytes < planned_ops < reused_buffer_count
//   < skipped_ops < status < total_allocated_bytes < truth_boundary
//
// Per-op attrs printed alphabetically; CHECK directives follow that order.

// ---------------------------------------------------------------------------
// Case 1: @chain -- linear chain A->B->C with first-fit reuse.
//
// Lifetimes:
//   %a: begin=0 end=1  (%b consumes it)
//   %b: begin=1 end=2  (%c consumes it)
//   %c: begin=2 end=2  (no plannable consumer)
//
// Allocation:
//   %a -> B0 offset=0  freed at t=1
//   %b -> B1 offset=32 freed at t=2   (B0 not free: 1 < 1 is false)
//   %c -> B0 reused    offset=0        (B0 free: 1 < 2; first-fit wins)
//
// Expected: buffer_count=2, reused_buffer_count=1, total_allocated=64, peak=64
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @chain
// CHECK-SAME:  cv.memory_plan.buffer_count = 2 : i64
// CHECK-SAME:  cv.memory_plan.peak_memory_bytes = 64 : i64
// CHECK-SAME:  cv.memory_plan.planned_ops = 3 : i64
// CHECK-SAME:  cv.memory_plan.reused_buffer_count = 1 : i64
// CHECK-SAME:  cv.memory_plan.skipped_ops = 0 : i64
// CHECK-SAME:  cv.memory_plan.status = "completed"
// CHECK-SAME:  cv.memory_plan.total_allocated_bytes = 64 : i64
// CHECK-SAME:  cv.memory_plan.truth_boundary = "static_compiler_memory_plan_not_runtime_allocation"
// Per-op: %a (index 0, B0, not reused).
// CHECK: "cv.conv2d"
// CHECK: cv.buffer_id = 0 : i64
// CHECK: cv.buffer_offset = 0 : i64
// CHECK: cv.lifetime_begin = 0 : i64
// CHECK: cv.lifetime_end = 1 : i64
// CHECK: cv.reuse_group = 0 : i64
// Per-op: %b (index 1, B1, not reused).
// CHECK: "cv.silu"
// CHECK: cv.buffer_id = 1 : i64
// CHECK: cv.buffer_offset = 32 : i64
// CHECK: cv.lifetime_begin = 1 : i64
// CHECK: cv.lifetime_end = 2 : i64
// CHECK: cv.reuse_group = 1 : i64
// Per-op: %c (index 2, reuses B0).
// CHECK: "cv.upsample"
// CHECK: cv.buffer_id = 0 : i64
// CHECK: cv.buffer_offset = 0 : i64
// CHECK: cv.lifetime_begin = 2 : i64
// CHECK: cv.lifetime_end = 2 : i64
// CHECK: cv.reuse_group = 0 : i64

// ---------------------------------------------------------------------------
// Case 2: @branch -- fan-out: %a feeds both %b and %c.
//
// Lifetimes:
//   %a: begin=0 end=2  (consumers: %b at 1, %c at 2 -> max=2)
//   %b: begin=1 end=1  (no plannable consumer)
//   %c: begin=2 end=2  (no plannable consumer)
//
// Allocation:
//   %a -> B0 offset=0  freed at t=2
//   %b -> B1 offset=32 freed at t=1   (B0 not free: 2 < 1 is false)
//   %c -> B1 reused    offset=32       (B0 not free: 2<2 false; B1 free: 1<2)
//
// Key assertion: %a.lifetime_end=2; %c reuses %b's buffer (buffer_id=1).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @branch
// CHECK-SAME:  cv.memory_plan.buffer_count = 2 : i64
// CHECK-SAME:  cv.memory_plan.peak_memory_bytes = 64 : i64
// CHECK-SAME:  cv.memory_plan.planned_ops = 3 : i64
// CHECK-SAME:  cv.memory_plan.reused_buffer_count = 1 : i64
// CHECK-SAME:  cv.memory_plan.total_allocated_bytes = 64 : i64
// Per-op: %a (index 0, B0, lifetime extended to 2 by fan-out consumer %c).
// CHECK: "cv.conv2d"
// CHECK: cv.buffer_id = 0 : i64
// CHECK: cv.buffer_offset = 0 : i64
// CHECK: cv.lifetime_begin = 0 : i64
// CHECK: cv.lifetime_end = 2 : i64
// Per-op: %b (index 1, B1, freed at t=1).
// CHECK: "cv.silu"
// CHECK: cv.buffer_id = 1 : i64
// CHECK: cv.buffer_offset = 32 : i64
// CHECK: cv.lifetime_begin = 1 : i64
// CHECK: cv.lifetime_end = 1 : i64
// Per-op: %c (index 2, reuses B1 from %b; reuse_group=1 proves sharing).
// CHECK: "cv.upsample"
// CHECK: cv.buffer_id = 1 : i64
// CHECK: cv.buffer_offset = 32 : i64
// CHECK: cv.lifetime_begin = 2 : i64
// CHECK: cv.lifetime_end = 2 : i64
// CHECK: cv.reuse_group = 1 : i64

// ---------------------------------------------------------------------------
// Case 3: @missing_bytes -- one plannable op, one skipped (no bytes_estimate).
//
// %a has cv.bytes_estimate -> planned.
// %b lacks cv.bytes_estimate -> skipped.
// Expected: planned_ops=1, skipped_ops=1, buffer_count=1, total=32.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @missing_bytes
// CHECK-SAME:  cv.memory_plan.buffer_count = 1 : i64
// CHECK-SAME:  cv.memory_plan.planned_ops = 1 : i64
// CHECK-SAME:  cv.memory_plan.skipped_ops = 1 : i64
// CHECK-SAME:  cv.memory_plan.status = "completed"
// CHECK-SAME:  cv.memory_plan.total_allocated_bytes = 32 : i64
// CHECK: "cv.conv2d"
// CHECK: cv.buffer_id = 0 : i64
// CHECK: cv.lifetime_begin = 0 : i64
// CHECK: cv.lifetime_end = 0 : i64

module {

  // Case 1: linear chain A -> B -> C; C reuses A's buffer.
  func.func @chain(%x: tensor<1x4x2x2xf16>)
      -> (tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>) {
    %a = "cv.conv2d"(%x) {
      cv.bytes_estimate = 32 : i64,
      cv.source_op = "Conv"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %b = "cv.silu"(%a) {
      cv.bytes_estimate = 32 : i64,
      cv.source_op = "Mul"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %c = "cv.upsample"(%b) {
      cv.bytes_estimate = 32 : i64,
      cv.source_op = "Resize"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %a, %b, %c
        : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>
  }

  // Case 2: fan-out A -> {B, C}; A's lifetime extends to last consumer C.
  func.func @branch(%x: tensor<1x4x2x2xf16>)
      -> (tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>) {
    %a = "cv.conv2d"(%x) {
      cv.bytes_estimate = 32 : i64,
      cv.source_op = "Conv"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %b = "cv.silu"(%a) {
      cv.bytes_estimate = 32 : i64,
      cv.source_op = "Mul"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %c = "cv.upsample"(%a) {
      cv.bytes_estimate = 32 : i64,
      cv.source_op = "Resize"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %a, %b, %c
        : tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>
  }

  // Case 3: %a has bytes_estimate (planned); %b lacks it (skipped).
  func.func @missing_bytes(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    %a = "cv.conv2d"(%x) {
      cv.bytes_estimate = 32 : i64,
      cv.source_op = "Conv"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %b = "cv.silu"(%a) {
      cv.source_op = "Mul"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %b : tensor<1x4x2x2xf16>
  }

}
