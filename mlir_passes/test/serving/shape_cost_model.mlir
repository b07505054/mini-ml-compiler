// FileCheck tests for shape_cost_model_v2 (ServingCostModelPass layer 3).
//
// Run:
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(candidate-evaluation-pipeline)' \
//     shape_cost_model.mlir | \
//   FileCheck shape_cost_model.mlir --input-file=- --split-input-file
//
// (The final test additionally runs plan-selection-pipeline.)
//
// All values below are static compiler estimates from tensor shapes and
// declared theoretical peak numbers — not measured benchmarks.

// -----

// Test 1: FLOPs and costs scale with shape. Small matmul (4x8 -> 4x16):
// flops = 2*4*8*16 = 1024; fp16 bytes: input 64 + weight 256 + output 128
// = 448; AI = 1024*1000/448 = 2285 milli. With peak_flops_fp16 = 1e12 and
// bandwidth = 1e11: compute = 1 ns, memory = 4 ns, total = max + 0 = 4 ns.
// Large matmul (256x512 -> 256x1024): flops = 268435456; bytes 1835008;
// compute = 268435 ns; memory = 18350 ns; total = 268435 ns.
// Default ranking is untouched: evaluation.penalty_score stays V0 (= 0).
//
// CHECK-LABEL: func.func @matmul_flops_scale_with_shape
// CHECK: %[[SMALL:.*]] = "compute.matmul"
// CHECK-SAME: evaluation.penalty_score = 0
// CHECK-SAME: evaluation.shape_cost.arithmetic_intensity_milli = 2285
// CHECK-SAME: evaluation.shape_cost.estimated_boundary_cost_nanos = 0
// CHECK-SAME: evaluation.shape_cost.estimated_compute_cost_nanos = 1
// CHECK-SAME: evaluation.shape_cost.estimated_memory_cost_nanos = 4
// CHECK-SAME: evaluation.shape_cost.estimated_total_cost_nanos = 4
// CHECK-SAME: evaluation.shape_cost.flops_estimate = 1024
// CHECK-SAME: evaluation.shape_cost.input_bytes_estimate = 64
// CHECK-SAME: evaluation.shape_cost.model_version = "shape_cost_model_v2"
// CHECK-SAME: evaluation.shape_cost.output_bytes_estimate = 128
// CHECK-SAME: evaluation.shape_cost.status = "estimated"
// CHECK-SAME: evaluation.shape_cost.total_memory_bytes_estimate = 448
// CHECK-SAME: evaluation.shape_cost.truth_boundary = "static_shape_derived_declared_profile_not_measured_not_runtime_validated"
// CHECK-SAME: evaluation.shape_cost.weight_bytes_estimate = 256
// CHECK-SAME: compiler.shape_profile.op_kind = "matmul_like"
// CHECK-SAME: compiler.shape_profile.ranking_mode = "v0_heuristic"
// CHECK-SAME: compiler.shape_profile.status = "static_shapes"
// CHECK: "compute.matmul"
// CHECK-SAME: evaluation.shape_cost.estimated_compute_cost_nanos = 268435
// CHECK-SAME: evaluation.shape_cost.estimated_memory_cost_nanos = 18350
// CHECK-SAME: evaluation.shape_cost.estimated_total_cost_nanos = 268435
// CHECK-SAME: evaluation.shape_cost.flops_estimate = 268435456
// CHECK-SAME: evaluation.shape_cost.total_memory_bytes_estimate = 1835008

module attributes {
  target.static_cost_profile.peak_flops_fp16 = 1.0e12 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e11 : f64
} {
  func.func @matmul_flops_scale_with_shape(
      %a: tensor<4x8xf16>, %b: tensor<256x512xf16>)
      -> (tensor<4x16xf16>, tensor<256x1024xf16>) {
    %0 = "compute.matmul"(%a) {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    %1 = "compute.matmul"(%b) {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    return %0, %1 : tensor<4x16xf16>, tensor<256x1024xf16>
  }
}

// -----

// Test 2: dtype changes bytes. Same 4x8 -> 4x16 matmul; candidate dtype
// fp32 gives total 128 + 512 + 256 = 896 bytes; fp16 gives 448. Without
// declared profile numbers the model emits facts only — no time estimates,
// no fabricated nanoseconds.
//
// CHECK-LABEL: func.func @dtype_changes_bytes
// CHECK: "compute.matmul"
// CHECK-SAME: evaluation.shape_cost.input_bytes_estimate = 128
// CHECK-SAME: evaluation.shape_cost.status = "facts_only_no_profile_numbers"
// CHECK-SAME: evaluation.shape_cost.total_memory_bytes_estimate = 896
// CHECK-SAME: evaluation.shape_cost.weight_bytes_estimate = 512
// CHECK-SAME: evaluation.shape_cost.input_bytes_estimate = 64
// CHECK-SAME: evaluation.shape_cost.total_memory_bytes_estimate = 448
// CHECK-NOT: estimated_total_cost_nanos

module {
  func.func @dtype_changes_bytes(%a: tensor<4x8xf16>) -> tensor<4x16xf16> {
    %0 = "compute.matmul"(%a) {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        dtype = "fp32",
        required_boundary_ops = [],
        source_op = "matmul"
      }, {
        candidate_type = "direct_lower",
        dtype = "fp16",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 3: dynamic shapes fall back honestly to the V1 fixed model — no
// shape_cost attrs, status recorded, V0 penalty untouched.
//
// CHECK-LABEL: func.func @dynamic_shape_falls_back
// CHECK: "compute.matmul"
// CHECK-NOT: evaluation.shape_cost
// CHECK-SAME: evaluation.penalty_score = 0
// CHECK-SAME: compiler.shape_profile.op_kind = "matmul_like"
// CHECK-SAME: compiler.shape_profile.status = "dynamic_dims_unresolved"

module {
  func.func @dynamic_shape_falls_back(%a: tensor<?x8xf16>)
      -> tensor<?x16xf16> {
    %0 = "compute.matmul"(%a) {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<?x8xf16>) -> tensor<?x16xf16>
    return %0 : tensor<?x16xf16>
  }
}

// -----

// Test 4: existing quantization metadata narrows the weight byte estimate.
// Op A (no quant metadata): fp16 weight bytes = 256. Op B carries
// quant.weight_dtype = "int8" (from QuantizationStrategyPlanningPass):
// weight bytes = 128, total = 64 + 128 + 128 = 320 -> memory 3 ns; its
// dequant_weight boundary moves 2 x 256 float-weight bytes = 512 bytes
// -> boundary 5 ns; total = max(1, 3) + 5 = 8 ns.
//
// CHECK-LABEL: func.func @quant_weight_dtype_changes_bytes
// CHECK: "compute.matmul"
// CHECK-SAME: evaluation.shape_cost.weight_bytes_estimate = 256
// CHECK: "compute.matmul"
// CHECK-SAME: evaluation.shape_cost.estimated_boundary_cost_nanos = 5
// CHECK-SAME: evaluation.shape_cost.estimated_total_cost_nanos = 8
// CHECK-SAME: evaluation.shape_cost.total_memory_bytes_estimate = 320
// CHECK-SAME: evaluation.shape_cost.weight_bytes_estimate = 128

module attributes {
  target.static_cost_profile.peak_flops_fp16 = 1.0e12 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e11 : f64
} {
  func.func @quant_weight_dtype_changes_bytes(%a: tensor<4x8xf16>)
      -> (tensor<4x16xf16>, tensor<4x16xf16>) {
    %0 = "compute.matmul"(%a) {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    %1 = "compute.matmul"(%a) {
      quant.weight_dtype = "int8",
      compiler.candidates = [{
        candidate_type = "representation_conversion",
        required_boundary_ops = ["dequant_weight"],
        source_op = "matmul"
      }]
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0, %1 : tensor<4x16xf16>, tensor<4x16xf16>
  }
}

// -----

// Test 5 (co-design): serving.cost_model.mode = "shape_aware_v2" ranks by
// the shape/dtype/profile-derived estimate. The op produces f32; candidate A
// (direct_lower, fp32) moves 3670016 bytes -> 367002 ns memory-bound total.
// Candidate B (cast_conversion to the fp16 effective dtype) moves half the
// bytes -> compute-bound total 268435 ns. V0 would pick A (penalty 0 vs 2);
// the shape-aware estimate picks B, with the V0 score preserved as
// evaluation.penalty_score_v0.
//
// CHECK-LABEL: func.func @v2_mode_changes_selection
// CHECK: "compute.matmul"
// CHECK-SAME: evaluation.penalty_score_v0 = 2
// CHECK-SAME: compiler.shape_profile.ranking_mode = "shape_aware_v2_estimated_total_nanos"
// CHECK-SAME: selected_plan.candidate_type = "cast_conversion"
// CHECK-SAME: selected_plan.penalty_score = 268435
// CHECK-SAME: selected_plan.shape_cost.estimated_memory_cost_nanos = 183501
// CHECK-SAME: selected_plan.shape_cost.estimated_total_cost_nanos = 268435
// CHECK-SAME: selected_plan.shape_cost.flops_estimate = 268435456
// CHECK-SAME: selected_plan.shape_cost.model_version = "shape_cost_model_v2"
// CHECK-SAME: selected_plan.shape_cost.total_memory_bytes_estimate = 1835008
// CHECK-SAME: selected_plan.shape_cost.truth_boundary = "static_shape_derived_declared_profile_not_measured_not_runtime_validated"

module attributes {
  serving.cost_model.mode = "shape_aware_v2",
  target.static_cost_profile.peak_flops_fp32 = 1.0e12 : f64,
  target.static_cost_profile.peak_flops_fp16 = 1.0e12 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e10 : f64
} {
  func.func @v2_mode_changes_selection(%a: tensor<256x512xf32>)
      -> tensor<256x1024xf32>
      attributes {representation.effective_dtype = "fp16"} {
    %0 = "compute.matmul"(%a) {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul"
      }, {
        candidate_type = "cast_conversion",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<256x512xf32>) -> tensor<256x1024xf32>
    return %0 : tensor<256x1024xf32>
  }
}
