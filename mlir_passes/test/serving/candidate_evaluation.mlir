// FileCheck tests for CandidateEvaluationPass.
//
// Run:
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(candidate-evaluation-pipeline)' \
//     candidate_evaluation.mlir | \
//   FileCheck candidate_evaluation.mlir --input-file=- --split-input-file

// -----

// Test 1: direct_lower has lower penalty than algebraic_decomposition.
//
// Two functions in the same module; the test verifies both penalty values and
// that the relative ordering holds (0 < 5).
//
// CHECK-LABEL: @direct_lower_penalty
// CHECK: "compute.matmul"
// CHECK-SAME: compiler.evaluated_candidates
// CHECK-SAME: candidate_type = "direct_lower"
// CHECK-SAME: evaluation.penalty_score = 0
// CHECK-SAME: evaluation.status = "evaluated"
//
// CHECK-LABEL: @decomposition_penalty
// CHECK: "compute.gelu"
// CHECK-SAME: compiler.evaluated_candidates
// CHECK-SAME: candidate_type = "algebraic_decomposition"
// CHECK-SAME: evaluation.penalty_score = 5
// CHECK-SAME: evaluation.status = "evaluated"

module {
  func.func @direct_lower_penalty(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16> {
    %mm = "compute.matmul"(%arg0, %arg0) {
      compiler.candidates = [{
        candidate_reason = "kernel_exact_match",
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul",
        truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"
      }],
      compiler.candidates.count = 1 : i64,
      compiler.candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated",
      compiler.rejected_candidates = []
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }

  func.func @decomposition_penalty(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16> {
    %g = "compute.gelu"(%arg0) {
      compiler.candidates = [{
        candidate_reason = "from_alternative_lowering",
        candidate_type = "algebraic_decomposition",
        required_boundary_ops = [],
        source_op = "gelu",
        truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"
      }],
      compiler.candidates.count = 1 : i64,
      compiler.candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated",
      compiler.rejected_candidates = []
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}

// -----

// Test 2: representation_conversion with dequant boundary incurs a boundary
// penalty on top of the base cost (base=3, dequant=+3, total=6).
//
// CHECK-LABEL: @representation_conversion_with_dequant
// CHECK: "compute.matmul"
// CHECK-SAME: compiler.evaluated_candidates
// CHECK-SAME: candidate_type = "representation_conversion"
// CHECK-SAME: evaluation.penalty_score = 6
// CHECK-SAME: evaluation.reason = "representation_conversion_base_penalty+dequant_boundary"
// CHECK-SAME: evaluation.status = "evaluated"

module {
  func.func @representation_conversion_with_dequant(
      %arg0: tensor<1x256xf16>, %arg1: tensor<256x256xi8>)
      -> tensor<1x256xf16> {
    %mm = "compute.matmul"(%arg0, %arg1) {
      compiler.candidates = [{
        candidate_reason = "from_alternative_lowering",
        candidate_type = "representation_conversion",
        required_boundary_ops = ["dequant_weight"],
        source_op = "matmul",
        truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"
      }],
      compiler.candidates.count = 1 : i64,
      compiler.candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated",
      compiler.rejected_candidates = []
    } : (tensor<1x256xf16>, tensor<256x256xi8>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 3: backend_fallback candidate receives a high penalty (20) but is still
// evaluated (status="evaluated"), not rejected.
//
// CHECK-LABEL: @backend_fallback_high_penalty
// CHECK: "compute.gelu"
// CHECK-SAME: compiler.evaluated_candidates
// CHECK-SAME: candidate_type = "backend_fallback"
// CHECK-SAME: evaluation.penalty_score = 20
// CHECK-SAME: evaluation.status = "evaluated"

module {
  func.func @backend_fallback_high_penalty(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16> {
    %g = "compute.gelu"(%arg0) {
      compiler.candidates = [{
        candidate_reason = "last_resort_fallback",
        candidate_type = "backend_fallback",
        required_boundary_ops = [],
        source_op = "gelu",
        truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"
      }],
      compiler.candidates.count = 1 : i64,
      compiler.candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated",
      compiler.rejected_candidates = []
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}

// -----

// Test 4: unsupported candidate receives the highest penalty (100) and
// evaluation.status="rejected" to signal no viable lowering path.
//
// CHECK-LABEL: @unsupported_highest_penalty
// CHECK: "compute.gelu"
// CHECK-SAME: compiler.evaluated_candidates
// CHECK-SAME: candidate_type = "unsupported"
// CHECK-SAME: evaluation.penalty_score = 100
// CHECK-SAME: evaluation.status = "rejected"

module {
  func.func @unsupported_highest_penalty(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16> {
    %g = "compute.gelu"(%arg0) {
      compiler.candidates = [{
        candidate_reason = "no_viable_lowering_path",
        candidate_type = "unsupported",
        required_boundary_ops = [],
        source_op = "gelu",
        truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"
      }],
      compiler.candidates.count = 1 : i64,
      compiler.candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated",
      compiler.rejected_candidates = []
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}

// -----

// Test 5: compiler.rejected_candidates are not promoted to compiler.evaluated_candidates.
//
// The op has a valid direct_lower candidate and an invalid algebraic_decomposition
// in rejected_candidates. After evaluation, compiler.evaluated_candidates contains
// only the direct_lower, and compiler.rejected_candidates remains unchanged.
//
// CHECK-LABEL: @rejected_remain_rejected
// CHECK: "compute.matmul"
// CHECK-SAME: compiler.evaluated_candidates
// CHECK-SAME: candidate_type = "direct_lower"
// CHECK-SAME: evaluation.penalty_score = 0
// CHECK-SAME: compiler.rejected_candidates
// CHECK-SAME: candidate_type = "algebraic_decomposition"
// CHECK-SAME: rejection_reason = "missing_kernels:sigmoid"

module {
  func.func @rejected_remain_rejected(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16> {
    %mm = "compute.matmul"(%arg0, %arg0) {
      compiler.candidates = [{
        candidate_reason = "kernel_exact_match",
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul",
        truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"
      }],
      compiler.candidates.count = 1 : i64,
      compiler.candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated",
      compiler.rejected_candidates = [{
        candidate_type = "algebraic_decomposition",
        rejection_reason = "missing_kernels:sigmoid",
        source_op = "matmul",
        truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"
      }]
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}
