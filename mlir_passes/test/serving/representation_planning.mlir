// FileCheck test for RepresentationPlanningPass.
//
// Run with the HIR plugin loaded (split-input-file allows multiple modules):
//   mlir-opt %s --split-input-file --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(representation-planning-pipeline)' \
//   | FileCheck %s
//
// Behaviors under test:
//   1. Normal path: plan_dtype in backend supported_dtypes -> target_profile source.
//   2. Dtype conflict: plan_dtype not supported -> fallback to first dtype + conflict attrs.
//   3. Missing capability data: backend not in target.backend_capability_names -> conflict.
//   4. No execution_provider.primary: func is not annotated (skip).
//
// RepresentationPlanningPass reads:
//   - execution_provider.primary from func.func (set by ExecutionProviderPlanningPass)
//   - quantization.plan_dtype from the module
//   - target.backend_capability_names from the module
//   - target.backend_capabilities.{backend}.* flat-prefix attrs from the module
//
// Truth boundary: representation_plan_static_not_validated_against_kernel_performance

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 1: Normal path — plan_dtype supported by backend.
// quantization.plan_dtype = "fp16"; cuda backend supports ["fp16", "int8"].
// Expected: effective_dtype = "fp16", source = "target_profile", no conflict.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @normal_dtype_match
// CHECK-SAME:  representation.effective_dtype = "fp16"
// CHECK-SAME:  representation.effective_dtype_source = "target_profile"
// CHECK-SAME:  representation.preferred_activation_layout = "NCHW"
// CHECK-SAME:  representation.preferred_weight_layout = "KCRS"
// CHECK-SAME:  representation.source_backend = "cuda"
// CHECK-SAME:  representation.truth_boundary = "representation_plan_static_not_validated_against_kernel_performance"
// CHECK-NOT:   representation.conflict_reason

module attributes {
  quantization.plan_dtype = "fp16",
  target.backend_capability_names = ["cuda"],
  target.backend_capabilities.cuda.supported_dtypes = ["fp16", "int8"],
  target.backend_capabilities.cuda.preferred_activation_layouts = ["NCHW"],
  target.backend_capabilities.cuda.preferred_weight_layouts = ["KCRS"]
} {
  func.func @normal_dtype_match() attributes {
    execution_provider.primary = "cuda"
  } {
    return
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 2: Dtype conflict — plan_dtype not supported by backend.
// quantization.plan_dtype = "int8"; cpu_only backend supports ["fp32"] only.
// Expected: effective_dtype = "fp32" (first supported), conflict attrs emitted.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @dtype_conflict
// CHECK-SAME:  representation.conflict_fallback_dtype = "fp32"
// CHECK-SAME:  representation.conflict_reason = "backend_lacks_plan_dtype"
// CHECK-SAME:  representation.effective_dtype = "fp32"
// CHECK-SAME:  representation.effective_dtype_source = "fallback_first_supported"
// CHECK-SAME:  representation.preferred_activation_layout = "NHWC"
// CHECK-SAME:  representation.preferred_weight_layout = "OHWI"
// CHECK-SAME:  representation.source_backend = "cpu_only"
// CHECK-SAME:  representation.truth_boundary = "representation_plan_static_not_validated_against_kernel_performance"

// -----

module attributes {
  quantization.plan_dtype = "int8",
  target.backend_capability_names = ["cpu_only"],
  target.backend_capabilities.cpu_only.supported_dtypes = ["fp32"],
  target.backend_capabilities.cpu_only.preferred_activation_layouts = ["NHWC"],
  target.backend_capabilities.cpu_only.preferred_weight_layouts = ["OHWI"]
} {
  func.func @dtype_conflict() attributes {
    execution_provider.primary = "cpu_only"
  } {
    return
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 3: Missing capability data — backend selected but not in capability names.
// execution_provider.primary = "metal"; target.backend_capability_names = ["cuda"].
// quantization.plan_dtype = "fp16".
// Expected: effective_dtype = "fp16", conflict_reason = "backend_capability_missing",
//           layouts = "unknown".
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @missing_capability
// CHECK-SAME:  representation.conflict_reason = "backend_capability_missing"
// CHECK-SAME:  representation.effective_dtype = "fp16"
// CHECK-SAME:  representation.effective_dtype_source = "quantization_plan_no_backend_capability_data"
// CHECK-SAME:  representation.preferred_activation_layout = "unknown"
// CHECK-SAME:  representation.preferred_weight_layout = "unknown"
// CHECK-SAME:  representation.source_backend = "metal"
// CHECK-SAME:  representation.truth_boundary = "representation_plan_static_not_validated_against_kernel_performance"

// -----

module attributes {
  quantization.plan_dtype = "fp16",
  target.backend_capability_names = ["cuda"],
  target.backend_capabilities.cuda.supported_dtypes = ["fp16", "int8"],
  target.backend_capabilities.cuda.preferred_activation_layouts = ["NCHW"],
  target.backend_capabilities.cuda.preferred_weight_layouts = ["KCRS"]
} {
  func.func @missing_capability() attributes {
    execution_provider.primary = "metal"
  } {
    return
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 4: No execution_provider.primary — func is skipped entirely.
// No representation.* attrs should appear.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @no_primary
// CHECK-NOT:   representation.effective_dtype
// CHECK-NOT:   representation.conflict_reason

// -----

module attributes {
  quantization.plan_dtype = "fp16",
  target.backend_capability_names = ["cuda"]
} {
  func.func @no_primary() {
    return
  }
}
