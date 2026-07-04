// RUN: mlir-opt --allow-unregistered-dialect \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(lowering-decision-planning-pipeline)' \
// RUN:   --split-input-file %s | FileCheck %s
//
// LoweringDecisionPlanningPass tests (Commit J).
// Each section pre-annotates kernel.* attrs (as KernelAvailabilityPlanningPass would)
// and checks the resulting lowering.* decision attrs.
// Truth boundary: lowering_decision_static_not_backend_codegen_verified

// Test 1: kernel.lowering_status = lowerable -> lowering.decision = direct_lower.

// CHECK-LABEL: @direct_lower
// CHECK: "compute.matmul"
// CHECK-SAME: lowering.decision = "direct_lower"
// CHECK-SAME: lowering.reason = "kernel_available_for_op_dtype_layout"
// CHECK-SAME: lowering.required_rewrite = ""
// CHECK-SAME: lowering.requires_cast = false
// CHECK-SAME: lowering.requires_dequant = false
// CHECK-SAME: lowering.requires_layout_transform = false
// CHECK-SAME: lowering.target_backend = "cpu"
// CHECK-SAME: lowering.target_kernel = "cpu_matmul_fp16"
// CHECK-SAME: lowering.target_kernel_library = "cpu_blas"
// CHECK-SAME: lowering.truth_boundary = "lowering_decision_static_not_backend_codegen_verified"

func.func @direct_lower(%x: tensor<1x256xf16>) -> tensor<1x256xf16> attributes {
  representation.source_backend = "cpu"
} {
  %mm = "compute.matmul"(%x, %x) {
    kernel.backend = "cpu",
    kernel.decision_reason = "exact_kernel_match",
    kernel.exists = true,
    kernel.fallback_backend = "cpu_simd",
    kernel.library = "cpu_blas",
    kernel.lowering_status = "lowerable",
    kernel.name = "cpu_matmul_fp16",
    kernel.required_rewrite = "",
    kernel.truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
  return %mm : tensor<1x256xf16>
}

// -----

// Test 2: kernel.lowering_status = rewrite_candidate -> lowering.decision = rewrite_then_lower.
// required_rewrite is propagated verbatim.

// CHECK-LABEL: @rewrite_then_lower
// CHECK: "compute.matmul"
// CHECK-SAME: lowering.decision = "rewrite_then_lower"
// CHECK-SAME: lowering.reason = "rewrite_then_lower_via_int8_to_fp16_cast"
// CHECK-SAME: lowering.required_rewrite = "int8_to_fp16_cast"
// CHECK-SAME: lowering.target_backend = "cpu"
// CHECK-SAME: lowering.target_kernel = "cpu_matmul_rewrite"
// CHECK-SAME: lowering.target_kernel_library = "cpu_blas"
// CHECK-SAME: lowering.truth_boundary = "lowering_decision_static_not_backend_codegen_verified"

func.func @rewrite_then_lower(%x: tensor<1x256xi8>) -> tensor<1x256xi8> attributes {
  representation.source_backend = "cpu"
} {
  %mm = "compute.matmul"(%x, %x) {
    kernel.backend = "cpu",
    kernel.decision_reason = "rewrite_pattern_available",
    kernel.exists = false,
    kernel.fallback_backend = "",
    kernel.library = "cpu_blas",
    kernel.lowering_status = "rewrite_candidate",
    kernel.name = "cpu_matmul_rewrite",
    kernel.required_rewrite = "int8_to_fp16_cast",
    kernel.truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  } : (tensor<1x256xi8>, tensor<1x256xi8>) -> tensor<1x256xi8>
  return %mm : tensor<1x256xi8>
}

// -----

// Test 3: fallback_required + boundary.dequant_required=true -> dequant_then_lower.
// Rule 3 takes priority over Rule 4 when both dequant and fallback_backend are set.

// CHECK-LABEL: @dequant_then_lower
// CHECK: "compute.matmul"
// CHECK-SAME: lowering.decision = "dequant_then_lower"
// CHECK-SAME: lowering.reason = "dequant_required_before_fallback_to_cpu_simd"
// CHECK-SAME: lowering.required_rewrite = ""
// CHECK-SAME: lowering.requires_dequant = true
// CHECK-SAME: lowering.target_backend = "cpu_simd"
// CHECK-SAME: lowering.target_kernel = ""
// CHECK-SAME: lowering.target_kernel_library = ""
// CHECK-SAME: lowering.truth_boundary = "lowering_decision_static_not_backend_codegen_verified"

func.func @dequant_then_lower(%x: tensor<1x256xi8>) -> tensor<1x256xi8> attributes {
  representation.source_backend = "cpu"
} {
  %mm = "compute.matmul"(%x, %x) {
    boundary.dequant_required = true,
    kernel.backend = "cpu",
    kernel.decision_reason = "no_exact_kernel_fallback_to_cpu_simd",
    kernel.exists = false,
    kernel.fallback_backend = "cpu_simd",
    kernel.library = "",
    kernel.lowering_status = "fallback_required",
    kernel.name = "",
    kernel.required_rewrite = "",
    kernel.truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  } : (tensor<1x256xi8>, tensor<1x256xi8>) -> tensor<1x256xi8>
  return %mm : tensor<1x256xi8>
}

// -----

// Test 4: fallback_required + fallback_backend + no dequant -> fallback_backend.

// CHECK-LABEL: @fallback_backend
// CHECK: "compute.matmul"
// CHECK-SAME: lowering.decision = "fallback_backend"
// CHECK-SAME: lowering.reason = "fallback_to_cpu_simd"
// CHECK-SAME: lowering.required_rewrite = ""
// CHECK-SAME: lowering.requires_dequant = false
// CHECK-SAME: lowering.target_backend = "cpu_simd"
// CHECK-SAME: lowering.target_kernel = ""
// CHECK-SAME: lowering.target_kernel_library = ""
// CHECK-SAME: lowering.truth_boundary = "lowering_decision_static_not_backend_codegen_verified"

func.func @fallback_backend(%x: tensor<1x256xi8>) -> tensor<1x256xi8> attributes {
  representation.source_backend = "cpu"
} {
  %mm = "compute.matmul"(%x, %x) {
    boundary.dequant_required = false,
    kernel.backend = "cpu",
    kernel.decision_reason = "no_exact_kernel_fallback_to_cpu_simd",
    kernel.exists = false,
    kernel.fallback_backend = "cpu_simd",
    kernel.library = "",
    kernel.lowering_status = "fallback_required",
    kernel.name = "",
    kernel.required_rewrite = "",
    kernel.truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  } : (tensor<1x256xi8>, tensor<1x256xi8>) -> tensor<1x256xi8>
  return %mm : tensor<1x256xi8>
}

// -----

// Test 5: kernel.lowering_status = unsupported -> lowering.decision = unsupported.
// No kernel, no rewrite, no fallback.

// CHECK-LABEL: @unsupported
// CHECK: "compute.matmul"
// CHECK-SAME: lowering.decision = "unsupported"
// CHECK-SAME: lowering.reason = "no_kernel_no_rewrite_no_fallback"
// CHECK-SAME: lowering.required_rewrite = ""
// CHECK-SAME: lowering.requires_dequant = false
// CHECK-SAME: lowering.target_backend = "npu"
// CHECK-SAME: lowering.target_kernel = ""
// CHECK-SAME: lowering.target_kernel_library = ""
// CHECK-SAME: lowering.truth_boundary = "lowering_decision_static_not_backend_codegen_verified"

func.func @unsupported(%x: tensor<1x256xi8>) -> tensor<1x256xi8> attributes {
  representation.source_backend = "npu"
} {
  %mm = "compute.matmul"(%x, %x) {
    kernel.backend = "npu",
    kernel.decision_reason = "op_not_found_in_kernel_library",
    kernel.exists = false,
    kernel.fallback_backend = "",
    kernel.library = "",
    kernel.lowering_status = "unsupported",
    kernel.name = "",
    kernel.required_rewrite = "",
    kernel.truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  } : (tensor<1x256xi8>, tensor<1x256xi8>) -> tensor<1x256xi8>
  return %mm : tensor<1x256xi8>
}
