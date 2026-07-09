// Diagnostic tests for BoundaryMaterializationPass: malformed or
// inconsistent planning attrs must produce diagnostics, never a silent
// no-op.
//
// Run:
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --load-dialect-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(boundary-materialization-pipeline)' \
//     --verify-diagnostics \
//     boundary_materialization_invalid.mlir

// -----

// Case 1: cast required but the function was never given a planned
// representation dtype -> hard error, pass failure.

module {
  func.func @missing_effective_dtype(%arg0: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    // expected-error @below {{boundary-materialization: boundary.cast_required = true but the function has no float representation.effective_dtype (got '')}}
    %0 = "compute.matmul"(%arg0, %arg0) {
      boundary.cast_required = true
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

// Case 2: cast required but the op already produces the planned dtype ->
// contradictory plan, hard error.

module {
  func.func @cast_to_same_dtype(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
      attributes {representation.effective_dtype = "fp32"} {
    // expected-error @below {{boundary-materialization: boundary.cast_required = true but source and target element types are both 'f32'}}
    %0 = "compute.matmul"(%arg0, %arg0) {
      boundary.cast_required = true
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

// -----

// Case 3: materialization_required set without any specific boundary flag
// -> inconsistent planning output, warning (nothing to materialize).

module {
  func.func @requirement_without_flag(%arg0: tensor<4x8xf16>)
      -> tensor<4x8xf16>
      attributes {representation.effective_dtype = "fp16"} {
    // expected-warning @below {{boundary-materialization: boundary.materialization_required = true but no specific boundary requirement flag is set; nothing materialized}}
    %0 = "compute.matmul"(%arg0, %arg0) {
      boundary.materialization_required = true
    } : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}
