// FileCheck tests for PlanSelectionPass.
//
// Run:
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(plan-selection-pipeline)' \
//     plan_selection.mlir | \
//   FileCheck plan_selection.mlir --input-file=- --split-input-file

// -----

// Test 1: direct_lower (penalty 0) beats algebraic_decomposition (penalty 5).
//
// The decomposition candidate must appear in compiler.selection_rejections, not
// in compiler.selected_candidates. The flat selected_plan.* attrs confirm the winner.
//
// CHECK-LABEL: @direct_lower_beats_decomposition
// CHECK: "compute.matmul"
// CHECK-SAME: compiler.selection_rejections
// CHECK-SAME: candidate_type = "algebraic_decomposition"
// CHECK-SAME: selected_plan.candidate_type = "direct_lower"
// CHECK-SAME: selected_plan.penalty_score = 0
// CHECK-SAME: selected_plan.reason = "lowest_penalty_evaluated"

module {
  func.func @direct_lower_beats_decomposition(%arg0: tensor<1x256xf16>)
      -> tensor<1x256xf16> {
    %mm = "compute.matmul"(%arg0, %arg0) {
      compiler.evaluated_candidates = [{
        candidate_type = "direct_lower",
        evaluation.penalty_score = 0 : i64,
        evaluation.status = "evaluated",
        required_boundary_ops = [],
        source_op = "matmul"
      }, {
        candidate_type = "algebraic_decomposition",
        evaluation.penalty_score = 5 : i64,
        evaluation.status = "evaluated",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 2: representation_conversion (penalty 6, tier 0) beats backend_fallback
// (penalty 20, tier 1) even though both have evaluation.status=evaluated.
// Rule 4: backend_fallback is only selected when no non-fallback evaluated
// candidate exists.
//
// CHECK-LABEL: @repr_conversion_beats_backend_fallback
// CHECK: "compute.matmul"
// CHECK-SAME: compiler.selection_rejections
// CHECK-SAME: candidate_type = "backend_fallback"
// CHECK-SAME: selected_plan.candidate_type = "representation_conversion"
// CHECK-SAME: selected_plan.penalty_score = 6
// CHECK-SAME: selected_plan.reason = "lowest_penalty_evaluated"

module {
  func.func @repr_conversion_beats_backend_fallback(
      %arg0: tensor<1x256xf16>, %arg1: tensor<256x256xi8>)
      -> tensor<1x256xf16> {
    %mm = "compute.matmul"(%arg0, %arg1) {
      compiler.evaluated_candidates = [{
        candidate_type = "representation_conversion",
        evaluation.penalty_score = 6 : i64,
        evaluation.status = "evaluated",
        required_boundary_ops = ["dequant_weight"],
        source_op = "matmul"
      }, {
        candidate_type = "backend_fallback",
        evaluation.penalty_score = 20 : i64,
        evaluation.status = "evaluated",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<1x256xf16>, tensor<256x256xi8>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 3: backend_fallback is selected when it is the only evaluated candidate
// (no direct_lower, no non-fallback alternative exists).
// compiler.selection_rejections must be empty.
//
// CHECK-LABEL: @backend_fallback_last_resort
// CHECK: "compute.gelu"
// CHECK-SAME: compiler.selection_rejections = []
// CHECK-SAME: selected_plan.candidate_type = "backend_fallback"
// CHECK-SAME: selected_plan.penalty_score = 20
// CHECK-SAME: selected_plan.reason = "backend_fallback_last_resort"

module {
  func.func @backend_fallback_last_resort(%arg0: tensor<1x256xf16>)
      -> tensor<1x256xf16> {
    %g = "compute.gelu"(%arg0) {
      compiler.evaluated_candidates = [{
        candidate_type = "backend_fallback",
        evaluation.penalty_score = 20 : i64,
        evaluation.status = "evaluated",
        required_boundary_ops = [],
        source_op = "gelu"
      }]
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}

// -----

// Test 4: unsupported is selected (with reason=no_valid_lowering_path) only
// when no evaluated or partially_evaluated candidate exists.
// evaluation.status=rejected means CandidateEvaluationPass marked it as the
// unsupported sentinel.
//
// CHECK-LABEL: @unsupported_no_valid_candidate
// CHECK: "compute.gelu"
// CHECK-SAME: selected_plan.candidate_type = "unsupported"
// CHECK-SAME: selected_plan.penalty_score = 100
// CHECK-SAME: selected_plan.reason = "no_valid_lowering_path"

module {
  func.func @unsupported_no_valid_candidate(%arg0: tensor<1x256xf16>)
      -> tensor<1x256xf16> {
    %g = "compute.gelu"(%arg0) {
      compiler.evaluated_candidates = [{
        candidate_type = "unsupported",
        evaluation.penalty_score = 100 : i64,
        evaluation.status = "rejected",
        required_boundary_ops = [],
        source_op = "gelu"
      }]
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}

// -----

// Test 5: Tie-break is deterministic when two candidates have the same penalty
// score. cast_conversion (tiebreak priority 2) beats layout_conversion
// (tiebreak priority 3) at the same penalty of 4.
//
// CHECK-LABEL: @tiebreak_cast_over_layout
// CHECK: "compute.matmul"
// CHECK-SAME: compiler.selection_rejections
// CHECK-SAME: candidate_type = "layout_conversion"
// CHECK-SAME: selected_plan.candidate_type = "cast_conversion"
// CHECK-SAME: selected_plan.penalty_score = 4

module {
  func.func @tiebreak_cast_over_layout(%arg0: tensor<1x256xf16>)
      -> tensor<1x256xf16> {
    %mm = "compute.matmul"(%arg0, %arg0) {
      compiler.evaluated_candidates = [{
        candidate_type = "layout_conversion",
        evaluation.penalty_score = 4 : i64,
        evaluation.status = "evaluated",
        required_boundary_ops = [],
        source_op = "matmul"
      }, {
        candidate_type = "cast_conversion",
        evaluation.penalty_score = 4 : i64,
        evaluation.status = "evaluated",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}
