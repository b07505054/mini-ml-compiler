// FileCheck test for BoundaryPlanningPass.
//
// Run with the HIR plugin loaded (split-input-file allows multiple modules):
//   mlir-opt %s --split-input-file --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(boundary-planning-pipeline)' \
//   | FileCheck %s
//
// Behaviors under test:
//   1. Cast required: op result is f32, plan dtype is fp16, supports_cast=true.
//   2. Layout transform required: layout.transform_required pre-set to true, supports_layout_transform=true.
//   3. No boundary: dtypes match, no layout transform — all _required flags false.
//   4. Dequant required: plan dtype is int8, op result is f32, supports_dequant_boundary=true.
//   5. Unsupported layout transform: layout.transform_required=true but supports_layout_transform=false.
//
// BoundaryPlanningPass reads:
//   - representation.effective_dtype from func.func (skips func if absent)
//   - representation.source_backend from func.func
//   - layout.transform_required per-op (pre-set by LayoutPlanningPass or test input)
//   - target.backend_capabilities.{backend}.{supports_cast, supports_dequant_boundary,
//     supports_layout_transform} from module
//
// Per-op output attrs (alphabetical order):
//   boundary.cast_required                -- true if float precision cast needed and supported
//   boundary.dequant_required             -- true if int→float dequant needed and supported
//   boundary.layout_transform_required    -- true if layout transform needed and supported
//   boundary.materialization_required     -- true if any boundary can be materialized
//   boundary.reason                       -- comma-separated decision summary
//   boundary.truth_boundary               -- "boundary_planning_only_no_ir_materialization"
//   boundary.unsupported_reason           -- (optional) comma-separated unsupported needs
//
// Truth boundary:
//   boundary_planning_only_no_ir_materialization

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Module 1 (cuda backend: supports_cast=true, supports_layout_transform=true,
//           supports_dequant_boundary=false)
//
// Case 1: @cast_required — f32 op result, fp16 plan, supports_cast=true.
//   Expected: cast_required=true, materialization_required=true, reason="cast"
//
// Case 2: @layout_transform_required — fp32 plan, layout.transform_required=true preset,
//   supports_layout_transform=true.
//   Expected: layout_transform_required=true, materialization_required=true
//
// Case 3: @no_boundary — fp32 plan, f32 op, no layout transform.
//   Expected: all _required=false, reason="no_boundaries"
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @cast_required

// CHECK:      "compute.matmul"
// CHECK-SAME: boundary.cast_required = true
// CHECK-SAME: boundary.dequant_required = false
// CHECK-SAME: boundary.layout_transform_required = false
// CHECK-SAME: boundary.materialization_required = true
// CHECK-SAME: boundary.reason = "cast"
// CHECK-SAME: boundary.truth_boundary = "boundary_planning_only_no_ir_materialization"

// CHECK-LABEL: func.func @layout_transform_required

// CHECK:      "compute.conv"
// CHECK-SAME: boundary.cast_required = false
// CHECK-SAME: boundary.dequant_required = false
// CHECK-SAME: boundary.layout_transform_required = true
// CHECK-SAME: boundary.materialization_required = true
// CHECK-SAME: boundary.reason = "layout_transform"
// CHECK-SAME: boundary.truth_boundary = "boundary_planning_only_no_ir_materialization"

// CHECK-LABEL: func.func @no_boundary

// CHECK:      "compute.relu"
// CHECK-SAME: boundary.cast_required = false
// CHECK-SAME: boundary.dequant_required = false
// CHECK-SAME: boundary.layout_transform_required = false
// CHECK-SAME: boundary.materialization_required = false
// CHECK-SAME: boundary.reason = "no_boundaries"
// CHECK-SAME: boundary.truth_boundary = "boundary_planning_only_no_ir_materialization"

module attributes {
  target.backend_capability_names = ["cuda"],
  target.backend_capabilities.cuda.supports_cast = true,
  target.backend_capabilities.cuda.supports_dequant_boundary = false,
  target.backend_capabilities.cuda.supports_layout_transform = true
} {
  // Case 1: f32 op, fp16 plan — cast boundary needed and supported.
  func.func @cast_required(%arg0: tensor<?x64x64xf32>)
      -> tensor<?x64x64xf32> attributes {
    representation.effective_dtype = "fp16",
    representation.source_backend = "cuda"
  } {
    %0 = "compute.matmul"(%arg0) : (tensor<?x64x64xf32>) -> tensor<?x64x64xf32>
    return %0 : tensor<?x64x64xf32>
  }

  // Case 2: layout.transform_required preset from a previous LayoutPlanningPass run.
  func.func @layout_transform_required(%arg0: tensor<?x3x64x64xf32>)
      -> tensor<?x3x64x64xf32> attributes {
    representation.effective_dtype = "fp32",
    representation.source_backend = "cuda"
  } {
    %0 = "compute.conv"(%arg0) {layout.transform_required = true}
         : (tensor<?x3x64x64xf32>) -> tensor<?x3x64x64xf32>
    return %0 : tensor<?x3x64x64xf32>
  }

  // Case 3: dtypes match, no layout transform — no boundaries.
  func.func @no_boundary(%arg0: tensor<?x64x64xf32>)
      -> tensor<?x64x64xf32> attributes {
    representation.effective_dtype = "fp32",
    representation.source_backend = "cuda"
  } {
    %0 = "compute.relu"(%arg0) : (tensor<?x64x64xf32>) -> tensor<?x64x64xf32>
    return %0 : tensor<?x64x64xf32>
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Module 2 (npu backend: supports_dequant_boundary=true, others false)
//
// Case 4: @dequant_required — int8 plan, op produces f32, supports_dequant_boundary=true.
//   needsCast: isFloat(i8) = false → false.
//   needsDequant: isQuantized(i8) && isFloat(f32) → true.
//   Expected: dequant_required=true, materialization_required=true, reason="dequant"
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @dequant_required

// CHECK:      "compute.dequant_op"
// CHECK-SAME: boundary.cast_required = false
// CHECK-SAME: boundary.dequant_required = true
// CHECK-SAME: boundary.layout_transform_required = false
// CHECK-SAME: boundary.materialization_required = true
// CHECK-SAME: boundary.reason = "dequant"
// CHECK-SAME: boundary.truth_boundary = "boundary_planning_only_no_ir_materialization"

// -----

module attributes {
  target.backend_capability_names = ["npu"],
  target.backend_capabilities.npu.supports_cast = false,
  target.backend_capabilities.npu.supports_dequant_boundary = true,
  target.backend_capabilities.npu.supports_layout_transform = false
} {
  func.func @dequant_required(%arg0: tensor<?x64x64xi8>)
      -> tensor<?x64x64xf32> attributes {
    representation.effective_dtype = "int8",
    representation.source_backend = "npu"
  } {
    %0 = "compute.dequant_op"(%arg0) {layout.transform_required = false}
         : (tensor<?x64x64xi8>) -> tensor<?x64x64xf32>
    return %0 : tensor<?x64x64xf32>
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Module 3 (coreml backend: supports_layout_transform=false, supports_cast=true)
//
// Case 5: @layout_unsupported — layout.transform_required=true but not supported.
//   needsLayoutTransform=true, supportsLayoutTransform=false →
//     layout_transform_required=false, materialization_required=false,
//     unsupported_reason="layout_transform_required_but_unsupported"
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @layout_unsupported

// CHECK:      "compute.conv"
// CHECK-SAME: boundary.cast_required = false
// CHECK-SAME: boundary.dequant_required = false
// CHECK-SAME: boundary.layout_transform_required = false
// CHECK-SAME: boundary.materialization_required = false
// CHECK-SAME: boundary.reason = "layout_transform_unsupported"
// CHECK-SAME: boundary.truth_boundary = "boundary_planning_only_no_ir_materialization"
// CHECK-SAME: boundary.unsupported_reason = "layout_transform_required_but_unsupported"

// -----

module attributes {
  target.backend_capability_names = ["coreml"],
  target.backend_capabilities.coreml.supports_cast = true,
  target.backend_capabilities.coreml.supports_dequant_boundary = false,
  target.backend_capabilities.coreml.supports_layout_transform = false
} {
  func.func @layout_unsupported(%arg0: tensor<?x3x64x64xf32>)
      -> tensor<?x3x64x64xf32> attributes {
    representation.effective_dtype = "fp32",
    representation.source_backend = "coreml"
  } {
    %0 = "compute.conv"(%arg0) {layout.transform_required = true}
         : (tensor<?x3x64x64xf32>) -> tensor<?x3x64x64xf32>
    return %0 : tensor<?x3x64x64xf32>
  }
}
