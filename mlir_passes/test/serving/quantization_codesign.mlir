// FileCheck tests for QuantizationCoDesignPass
// (quantization_codesign_contract_v1).
//
// Run (the pipeline runs the pass TWICE — passing checks therefore also
// prove idempotent re-runs produce identical attrs):
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(quant-codesign-pipeline,quant-codesign-pipeline)' \
//     quantization_codesign.mlir | \
//   FileCheck quantization_codesign.mlir --input-file=- --split-input-file
//
// All numbers are static planning estimates from declared profiles and
// tensor shapes. No calibration, no measured accuracy, and no quantized
// runtime execution exist in this repository.

// -----

// Test 1: no quant.codesign.policy -> the pass is inert; existing behavior
// and artifacts are untouched.
//
// CHECK-LABEL: func.func @inert_without_policy
// CHECK-NOT: quant_codesign

module {
  func.func @inert_without_policy(%a: tensor<4x8xf16>) -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 2: runtime-variable weights are graph-illegal for weight-only
// quantization — rejected before any candidate is considered.
//
// CHECK-LABEL: func.func @runtime_variable_weight_rejected
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.rejection_reasons = ["weight_not_compile_time_constant"]
// CHECK-SAME: quant_codesign.status = "rejected_weight_not_constant"

module attributes {
  quant.codesign.policy = "systems_cost_only",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"]
} {
  func.func @runtime_variable_weight_rejected(%a: tensor<4x8xf16>)
      -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = false
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 3: small shape — the materialized float dequant intermediate
// (2 x 256 = 512 boundary bytes; no dispatchable kernel exists) outweighs
// the 128-byte weight saving: before 4 ns, after 8 ns, benefit -4 ns.
// Quantization honestly does NOT win. Unknown granularity/group/axis/
// symmetric metadata is absent, and scale/zero-point sources say so.
//
// CHECK-LABEL: func.func @small_shape_boundary_overhead_loses
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.backend_legality = "legal"
// CHECK-SAME: quant_codesign.candidate.activation_dtype = "fp16"
// CHECK-SAME: quant_codesign.candidate.representation = "weight_only_int8"
// CHECK-SAME: quant_codesign.candidate.weight_dtype = "int8"
// CHECK-SAME: quant_codesign.est.boundary_bytes = 512
// CHECK-SAME: quant_codesign.est.systems_benefit_nanos = -4
// CHECK-SAME: quant_codesign.est.total_cost_after_nanos = 8
// CHECK-SAME: quant_codesign.est.total_cost_before_nanos = 4
// CHECK-SAME: quant_codesign.est.weight_bytes_after = 128
// CHECK-SAME: quant_codesign.est.weight_bytes_before = 256
// CHECK-SAME: quant_codesign.kernel_support.status = "no_kernel_registry_declared"
// CHECK-SAME: quant_codesign.materialization.required = true
// CHECK-SAME: quant_codesign.materialization.status = "deferred_missing_quant_params"
// CHECK-SAME: quant_codesign.rejection_reasons = ["materialized_dequant_boundary_traffic_exceeds_weight_savings"]
// CHECK-SAME: quant_codesign.scale_source = "not_available_no_calibration"
// CHECK-SAME: quant_codesign.status = "rejected_no_systems_benefit"
// CHECK-SAME: quant_codesign.zero_point_source = "not_available_no_calibration"

module attributes {
  quant.codesign.policy = "systems_cost_only",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"],
  target.static_cost_profile.peak_flops_fp16 = 1.0e12 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e11 : f64
} {
  func.func @small_shape_boundary_overhead_loses(%a: tensor<4x8xf16>)
      -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 4: large memory-bound matmul with a DISPATCHABLE fixture weight-only
// kernel: the materialized dequant intermediate disappears (boundary 0),
// weight traffic halves (1048576 -> 524288 bytes), and the estimate drops
// 18350 -> 13107 ns: selected under require_dispatchable_kernel. The
// unmodeled inline dequant/unpack cost is explicitly listed as excluded,
// not silently treated as free.
//
// CHECK-LABEL: func.func @large_memory_bound_selected_with_kernel
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.est.boundary_bytes = 0
// CHECK-SAME: quant_codesign.est.excluded_terms = ["inline_dequant_unpack_conversion_cost"]
// CHECK-SAME: quant_codesign.est.systems_benefit_nanos = 5243
// CHECK-SAME: quant_codesign.est.total_cost_after_nanos = 13107
// CHECK-SAME: quant_codesign.est.total_cost_before_nanos = 18350
// CHECK-SAME: quant_codesign.est.weight_bytes_after = 524288
// CHECK-SAME: quant_codesign.est.weight_bytes_before = 1048576
// CHECK-SAME: quant_codesign.kernel_support.kernel_id = "fixture_cpu_matmul_wq_int8"
// CHECK-SAME: quant_codesign.kernel_support.source = "fixture"
// CHECK-SAME: quant_codesign.kernel_support.status = "runtime_dispatchable"
// CHECK-SAME: quant_codesign.materialization.required = false
// CHECK-SAME: quant_codesign.materialization.status = "not_required_kernel_consumes_quantized_weights"
// CHECK-SAME: quant_codesign.status = "selected"

module attributes {
  quant.codesign.policy = "require_dispatchable_kernel",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"],
  target.static_cost_profile.peak_flops_fp16 = 1.0e14 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e11 : f64,
  target.runtime_kernels = [{
    backend = "cpu",
    kernel_id = "fixture_cpu_matmul_wq_int8",
    op_name = "matmul",
    source = "fixture",
    supported_dtypes = ["fp16"],
    supported_quant_modes = ["weight_only"]
  }]
} {
  func.func @large_memory_bound_selected_with_kernel(%a: tensor<256x512xf16>)
      -> tensor<256x1024xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    return %0 : tensor<256x1024xf16>
  }
}

// -----

// Test 5: the backend DECLARES weight-only support but the declared runtime
// kernel registry has no matching kernel — a library capability never
// masquerades as a dispatchable kernel.
//
// CHECK-LABEL: func.func @backend_capable_but_not_dispatchable
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.kernel_support.status = "backend_supported_but_not_dispatchable"
// CHECK-SAME: quant_codesign.rejection_reasons = ["backend_capability_is_not_a_dispatchable_kernel"]
// CHECK-SAME: quant_codesign.status = "backend_supported_but_not_dispatchable"

module attributes {
  quant.codesign.policy = "require_dispatchable_kernel",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"],
  target.runtime_kernels = [{
    backend = "metal",
    kernel_id = "metal_rmsnorm_f32_v1",
    op_name = "rmsnorm",
    source = "handwritten_runtime",
    supported_dtypes = ["fp32"],
    supported_quant_modes = ["none"]
  }]
} {
  func.func @backend_capable_but_not_dispatchable(%a: tensor<4x8xf16>)
      -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 6: require_dispatchable_kernel with NO registry declared at all —
// an explicit deferral, never silent.
//
// CHECK-LABEL: func.func @no_registry_defers_kernel_requirement
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.kernel_support.status = "no_kernel_registry_declared"
// CHECK-SAME: quant_codesign.status = "deferred_no_runtime_kernel"

module attributes {
  quant.codesign.policy = "require_dispatchable_kernel",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"]
} {
  func.func @no_registry_defers_kernel_requirement(%a: tensor<4x8xf16>)
      -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 7: require_accuracy_evidence always defers today — no calibrated or
// measured accuracy evidence exists anywhere in this repository.
//
// CHECK-LABEL: func.func @accuracy_evidence_required_defers
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.accuracy_evidence.status = "no_accuracy_evidence"
// CHECK-SAME: quant_codesign.rejection_reasons = ["no_calibrated_or_measured_accuracy_evidence_in_repository"]
// CHECK-SAME: quant_codesign.status = "deferred_missing_accuracy_evidence"

module attributes {
  quant.codesign.policy = "require_accuracy_evidence",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"]
} {
  func.func @accuracy_evidence_required_defers(%a: tensor<4x8xf16>)
      -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 8: dynamic shapes stay honest — no bytes, no nanos, an explicit
// deferral under a cost-gated policy.
//
// The first CHECK-SAME matches contract_version immediately adjacent to
// kernel_support.status — est.* attrs sort between them, so contiguity
// proves no cost estimates were fabricated for the dynamic shape.
//
// CHECK-LABEL: func.func @dynamic_shape_defers_cost_estimates
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.contract_version = "quantization_codesign_contract_v1", quant_codesign.kernel_support.status = "no_kernel_registry_declared"
// CHECK-SAME: quant_codesign.rejection_reasons = ["shapes_not_static"]
// CHECK-SAME: quant_codesign.status = "deferred_missing_cost_estimates"

module attributes {
  quant.codesign.policy = "systems_cost_only",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"],
  target.static_cost_profile.peak_flops_fp16 = 1.0e12 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e11 : f64
} {
  func.func @dynamic_shape_defers_cost_estimates(%a: tensor<?x8xf16>)
      -> tensor<?x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<?x8xf16>) -> tensor<?x16xf16>
    return %0 : tensor<?x16xf16>
  }
}

// -----

// Test 9: planning_only evaluates and reports evidence but never selects.
// The forced-AWQ module attrs upgrade the evidence status only to
// algorithm_declared_not_calibrated — an algorithm DECLARATION is not an
// implementation and not accuracy evidence.
//
// CHECK-LABEL: func.func @planning_only_with_declared_algorithm
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.accuracy_evidence.artifact_ref = "artifacts/qwen_awq"
// CHECK-SAME: quant_codesign.accuracy_evidence.status = "algorithm_declared_not_calibrated"
// CHECK-SAME: quant_codesign.algorithm.name = "awq"
// CHECK-SAME: quant_codesign.algorithm.status = "declared_external"
// CHECK-SAME: quant_codesign.status = "planning_only_not_selected"

module attributes {
  quant.codesign.policy = "planning_only",
  quantization.algorithm = "awq",
  quantization.quantized_model_artifact_ref = "artifacts/qwen_awq",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"]
} {
  func.func @planning_only_with_declared_algorithm(%a: tensor<4x8xf16>)
      -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// -----

// Test 10: a backend that declares no quantized representation is not
// legal — and non-matmul ops are out of V1 scope entirely (no attrs).
//
// CHECK-LABEL: func.func @backend_not_legal
// CHECK: "compute.matmul"
// CHECK-SAME: quant_codesign.backend_legality = "not_legal"
// CHECK-SAME: quant_codesign.status = "rejected_backend_not_legal"
// CHECK: "compute.relu"
// CHECK-NOT: quant_codesign

module attributes {
  quant.codesign.policy = "systems_cost_only"
} {
  func.func @backend_not_legal(%a: tensor<4x8xf16>)
      -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    %1 = "compute.relu"(%0) : (tensor<4x16xf16>) -> tensor<4x16xf16>
    return %1 : tensor<4x16xf16>
  }
}
