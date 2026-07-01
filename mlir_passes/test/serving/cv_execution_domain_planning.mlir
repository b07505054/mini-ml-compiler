// FileCheck test for CVExecutionDomainPlanningPass.
//
// Run via tools/run_mlir_pass_tests.sh, or directly:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(cv-execution-domain-planning)' \
//   | FileCheck %s
//
// This pass classifies each cv.* op into a portable execution domain using
// op name only. It does not assign Metal, CPU, or any concrete backend.
// Two functions in one module; no --split-input-file required.
//
// Per-op attr alphabetical order:
//   cv.execution_domain < cv.execution_domain.truth_boundary
//   < cv.execution_domain_reason
//   (reasoning: common prefix "cv.execution_domain"; '\0' < '.' < '_')
//
// Function attr alphabetical order (all cv.execution_domain_plan.*):
//   accelerated_ops < fallback_ops < host_ops < planned_ops
//   < status < truth_boundary

// ---------------------------------------------------------------------------
// Case 1: @default_cv_graph -- exercises all three domain classifications.
//
// Ops and expected domains:
//   cv.conv2d        -> accelerated  (accelerated_tensor_policy)
//   cv.silu          -> accelerated  (accelerated_tensor_policy)
//   cv.upsample      -> accelerated  (accelerated_tensor_policy)
//   cv.concat        -> accelerated  (accelerated_tensor_policy)
//   cv.detect_head   -> host         (host_postprocess_policy)
//   cv.prototype_head -> host        (host_postprocess_policy)
//   cv.custom_op     -> fallback     (fallback_unknown_cv_op)
//
// Expected: accelerated_ops=4, host_ops=2, fallback_ops=1, planned_ops=7
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @default_cv_graph
// CHECK-SAME:  cv.execution_domain_plan.accelerated_ops = 4 : i64
// CHECK-SAME:  cv.execution_domain_plan.fallback_ops = 1 : i64
// CHECK-SAME:  cv.execution_domain_plan.host_ops = 2 : i64
// CHECK-SAME:  cv.execution_domain_plan.planned_ops = 7 : i64
// CHECK-SAME:  cv.execution_domain_plan.status = "completed"
// CHECK-SAME:  cv.execution_domain_plan.truth_boundary = "static_execution_domain_classification_not_target_mapping"
// Accelerated op: cv.conv2d -> domain="accelerated", reason="accelerated_tensor_policy"
// CHECK: cv.conv2d
// CHECK: cv.execution_domain = "accelerated"
// CHECK: cv.execution_domain.truth_boundary = "static_execution_domain_classification_not_target_mapping"
// CHECK: cv.execution_domain_reason = "accelerated_tensor_policy"
// Host op: cv.detect_head -> domain="host", reason="host_postprocess_policy"
// CHECK: cv.detect_head
// CHECK: cv.execution_domain = "host"
// CHECK: cv.execution_domain_reason = "host_postprocess_policy"
// Fallback op: cv.custom_op -> domain="fallback", reason="fallback_unknown_cv_op"
// CHECK: cv.custom_op
// CHECK: cv.execution_domain = "fallback"
// CHECK: cv.execution_domain_reason = "fallback_unknown_cv_op"

// ---------------------------------------------------------------------------
// Case 2: @no_cv -- gate fails; no function attrs emitted.
// This function must be LAST so CHECK-NOT scans to end of output.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_cv
// CHECK-NOT:   cv.execution_domain_plan.status

module {

  // Case 1: 7 cv.* ops covering all three domain classifications.
  func.func @default_cv_graph(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    %a = "cv.conv2d"(%x) {
      cv.source_op = "Conv"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %b = "cv.silu"(%a) {
      cv.source_op = "Silu"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %c = "cv.upsample"(%b) {
      cv.source_op = "Resize"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %d = "cv.concat"(%c) {
      cv.source_op = "Concat"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %e = "cv.detect_head"(%d) {
      cv.source_op = "Det"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %f = "cv.prototype_head"(%e) {
      cv.source_op = "Proto"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    %g = "cv.custom_op"(%f) {
      cv.source_op = "Custom"
    } : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
    return %g : tensor<1x4x2x2xf16>
  }

  // Case 2: no cv.* ops; gate fails; no function attrs set.
  func.func @no_cv(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
    return %x : tensor<1x4x2x2xf16>
  }

}
