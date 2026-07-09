// FileCheck tests for BoundaryMaterializationPass.
//
// Run:
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --load-dialect-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(boundary-materialization-pipeline)' \
//     boundary_materialization.mlir | \
//   FileCheck boundary_materialization.mlir --input-file=- --split-input-file

// -----

// Test 1: boundary.cast_required = true on an f32-producing op in an
// fp16-planned function materializes hir.cast, redirects every use
// (consumer op and func.return) to the cast result, and updates the
// function result type. The cast carries full provenance back to the
// planning decision.
//
// CHECK-LABEL: func.func @materialize_cast
// CHECK-SAME: -> tensor<4x8xf16>
// CHECK-SAME: boundary.materialization.deferred_count = 0
// CHECK-SAME: boundary.materialization.materialized_count = 1
// CHECK-SAME: boundary.materialization.status = "completed"
// CHECK: %[[SRC:.*]] = "compute.matmul"
// CHECK-SAME: boundary.cast_materialized = true
// CHECK-SAME: boundary.materialization.status = "materialized"
// CHECK-SAME: boundary.materialized_ops = ["cast"]
// CHECK-NEXT: %[[CAST:.*]] = hir.cast %[[SRC]]
// CHECK-SAME: cast.from_dtype = "f32"
// CHECK-SAME: cast.to_dtype = "f16"
// CHECK-SAME: compiler.materialized = true
// CHECK-SAME: materialized.by = "boundary-materialization"
// CHECK-SAME: materialized.from_decision = "boundary.cast_required"
// CHECK-SAME: materialized.of_op = "compute.matmul"
// CHECK-SAME: materialized.truth_boundary = "compiler_materialized_boundary_op_not_runtime_executed"
// CHECK-SAME: (tensor<4x8xf32>) -> tensor<4x8xf16>
// CHECK: "compute.consumer"(%[[CAST]])
// CHECK: return %[[CAST]] : tensor<4x8xf16>

module {
  func.func @materialize_cast(%arg0: tensor<4x8xf16>) -> tensor<4x8xf32>
      attributes {representation.effective_dtype = "fp16"} {
    %0 = "compute.matmul"(%arg0, %arg0) {
      boundary.cast_required = true,
      boundary.materialization_required = true,
      selected_plan.candidate_type = "direct_lower"
    } : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf32>
    %1 = "compute.consumer"(%0) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

// Test 2: no boundary requirement -> no materialization, no summary attrs,
// function left untouched.
//
// CHECK-LABEL: func.func @no_boundary_required
// CHECK-NOT: hir.cast
// CHECK-NOT: boundary.materialization

module {
  func.func @no_boundary_required(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16>
      attributes {representation.effective_dtype = "fp16"} {
    %0 = "compute.matmul"(%arg0, %arg0) {
      boundary.cast_required = false,
      boundary.dequant_required = false,
      boundary.layout_transform_required = false
    } : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}

// -----

// Test 3: dequant and layout-transform requirements are recorded as
// deferred (planned but not yet materializable without inventing
// metadata), never faked into IR.
//
// CHECK-LABEL: func.func @dequant_and_layout_deferred
// CHECK-SAME: boundary.materialization.deferred_count = 2
// CHECK-SAME: boundary.materialization.materialized_count = 0
// CHECK: "compute.matmul"
// CHECK-SAME: boundary.materialization.deferred = ["dequant", "layout_transform"]
// CHECK-NOT: hir.dequantize
// CHECK-NOT: hir.cast

module {
  func.func @dequant_and_layout_deferred(%arg0: tensor<4x8xf16>)
      -> tensor<4x8xf16>
      attributes {representation.effective_dtype = "int8"} {
    %0 = "compute.matmul"(%arg0, %arg0) {
      boundary.cast_required = false,
      boundary.dequant_required = true,
      boundary.layout_transform_required = true,
      selected_plan.candidate_type = "representation_conversion"
    } : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}

// -----

// Test 4: an op whose selected plan is "unsupported" never has boundary
// ops materialized — a plan with no viable lowering path must not be
// speculatively transformed.
//
// CHECK-LABEL: func.func @unsupported_not_materialized
// CHECK-SAME: boundary.materialization.deferred_count = 0
// CHECK-SAME: boundary.materialization.materialized_count = 0
// CHECK: "compute.exotic"
// CHECK-SAME: boundary.materialization.skipped_reason = "selected_plan_unsupported"
// CHECK-NOT: hir.cast

module {
  func.func @unsupported_not_materialized(%arg0: tensor<4x8xf16>)
      -> tensor<4x8xf32>
      attributes {representation.effective_dtype = "fp16"} {
    %0 = "compute.exotic"(%arg0) {
      boundary.cast_required = true,
      selected_plan.candidate_type = "unsupported"
    } : (tensor<4x8xf16>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}
