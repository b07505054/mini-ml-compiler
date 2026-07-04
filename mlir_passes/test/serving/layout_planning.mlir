// FileCheck test for LayoutPlanningPass.
//
// Run with the HIR plugin loaded (split-input-file allows multiple modules):
//   mlir-opt %s --split-input-file --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(layout-planning-pipeline)' \
//   | FileCheck %s
//
// Behaviors under test:
//   1. Main path: NCHW input → conv → relu (agnostic) → conv, backend prefers NHWC.
//      First conv sees transform boundary; relu and second conv propagate without transforms.
//   2. Already-at-preferred layout: NHWC input, NHWC backend — no transforms anywhere.
//   3. No representation.preferred_activation_layout: func is skipped entirely.
//
// LayoutPlanningPass reads:
//   - representation.preferred_activation_layout from func.func (target layout)
//   - representation.source_backend from func.func
//   - layout.initial_layout from func.func (default "NCHW" if absent)
//   - target.backend_capabilities.{backend}.layout_agnostic_ops from module
//
// Per-op output attrs (alphabetical, all on the op's line):
//   layout.effective_layout       -- layout of op output
//   layout.layout_source          -- "backend_preference" or "layout_agnostic_propagation"
//   layout.required_input_layout  -- layout op requires its activation input to be in
//   layout.transform_required     -- bool; true only at the first layout-changing boundary
//   layout.truth_boundary         -- "layout_planning_static_cost_model_not_measured_kernel_performance"
//
// Truth boundary:
//   layout_planning_static_cost_model_not_measured_kernel_performance

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 1: NCHW → conv → relu → conv, backend prefers NHWC, relu is agnostic.
//
// Expected layout propagation:
//   conv1: required_input = NCHW, effective = NHWC, transform = true
//   relu:  required_input = NHWC, effective = NHWC, transform = false  (agnostic)
//   conv2: required_input = NHWC, effective = NHWC, transform = false
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @conv_relu_conv

// First conv: input is NCHW, backend prefers NHWC — layout boundary, transform required.
// CHECK:      "compute.conv"
// CHECK-SAME: layout.effective_layout = "NHWC"
// CHECK-SAME: layout.layout_source = "backend_preference"
// CHECK-SAME: layout.required_input_layout = "NCHW"
// CHECK-SAME: layout.transform_required = true
// CHECK-SAME: layout.truth_boundary = "layout_planning_static_cost_model_not_measured_kernel_performance"

// ReLU is layout-agnostic: propagates NHWC from previous op, no transform.
// CHECK:      "compute.relu"
// CHECK-SAME: layout.effective_layout = "NHWC"
// CHECK-SAME: layout.layout_source = "layout_agnostic_propagation"
// CHECK-SAME: layout.required_input_layout = "NHWC"
// CHECK-SAME: layout.transform_required = false

// Second conv: input is already NHWC — no transform needed.
// CHECK:      "compute.conv"
// CHECK-SAME: layout.effective_layout = "NHWC"
// CHECK-SAME: layout.layout_source = "backend_preference"
// CHECK-SAME: layout.required_input_layout = "NHWC"
// CHECK-SAME: layout.transform_required = false

module attributes {
  target.backend_capability_names = ["cuda"],
  target.backend_capabilities.cuda.layout_agnostic_ops = ["relu", "add", "reshape"]
} {
  func.func @conv_relu_conv(%arg0: tensor<?x3x64x64xf32>)
      -> tensor<?x256x64x64xf32> attributes {
    layout.initial_layout = "NCHW",
    representation.preferred_activation_layout = "NHWC",
    representation.source_backend = "cuda"
  } {
    %0 = "compute.conv"(%arg0) : (tensor<?x3x64x64xf32>) -> tensor<?x256x64x64xf32>
    %1 = "compute.relu"(%0) : (tensor<?x256x64x64xf32>) -> tensor<?x256x64x64xf32>
    %2 = "compute.conv"(%1) : (tensor<?x256x64x64xf32>) -> tensor<?x256x64x64xf32>
    return %2 : tensor<?x256x64x64xf32>
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 2: Already at preferred layout — NHWC input, NHWC backend.
// No transforms should appear on any op.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @already_nhwc

// CHECK:      "compute.conv"
// CHECK-SAME: layout.effective_layout = "NHWC"
// CHECK-SAME: layout.required_input_layout = "NHWC"
// CHECK-SAME: layout.transform_required = false

// CHECK:      "compute.relu"
// CHECK-SAME: layout.effective_layout = "NHWC"
// CHECK-SAME: layout.required_input_layout = "NHWC"
// CHECK-SAME: layout.transform_required = false

// -----

module attributes {
  target.backend_capability_names = ["cuda"],
  target.backend_capabilities.cuda.layout_agnostic_ops = ["relu", "add", "reshape"]
} {
  func.func @already_nhwc(%arg0: tensor<?x64x64x3xf32>)
      -> tensor<?x64x64x256xf32> attributes {
    layout.initial_layout = "NHWC",
    representation.preferred_activation_layout = "NHWC",
    representation.source_backend = "cuda"
  } {
    %0 = "compute.conv"(%arg0) : (tensor<?x64x64x3xf32>) -> tensor<?x64x64x256xf32>
    %1 = "compute.relu"(%0) : (tensor<?x64x64x256xf32>) -> tensor<?x64x64x256xf32>
    return %1 : tensor<?x64x64x256xf32>
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 3: No representation.preferred_activation_layout — func is skipped.
// No layout.* attrs should appear on any op.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @no_representation
// CHECK-NOT:   layout.effective_layout
// CHECK-NOT:   layout.transform_required

// -----

module {
  func.func @no_representation(%arg0: tensor<?x3x64x64xf32>)
      -> tensor<?x256x64x64xf32> {
    %0 = "compute.conv"(%arg0) : (tensor<?x3x64x64xf32>) -> tensor<?x256x64x64xf32>
    return %0 : tensor<?x256x64x64xf32>
  }
}
