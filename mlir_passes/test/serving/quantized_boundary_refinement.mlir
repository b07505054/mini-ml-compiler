// RUN: mlir-opt --allow-unregistered-dialect \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(quantized-boundary-refinement-pipeline)' \
// RUN:   --split-input-file %s | FileCheck %s
//
// QuantizedBoundaryRefinementPass tests (Commit P).
// Each section pre-annotates ops with quant.strategy / lowering.decision attrs
// (simulating QuantizationStrategyPlanningPass + LoweringDecisionPlanningPass output)
// and runs only QuantizedBoundaryRefinementPass to verify boundary.weight_* output.
//
// Truth boundary: quantized_boundary_refinement_static_not_materialized

// Test 1: weight_only_int8 op falls back to cpu_ref which has an empty
// supported_quant_modes list (= no declared quant support, consistent with
// QuantizationStrategyPlanningPass semantics) -> weight_dequant_required=true.

// CHECK-LABEL: @weight_only_fallback_backend_no_int8
// CHECK: "compute.matmul"
// CHECK-SAME: boundary.weight_dequant_reason = "fallback_backend_lacks_quantized_weight_support"
// CHECK-SAME: boundary.weight_dequant_required = true
// CHECK-SAME: boundary.weight_dtype_mismatch = true

module attributes {
  target.backend_capabilities.cpu_ref.supported_quant_modes = []
} {
  func.func @weight_only_fallback_backend_no_int8(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes { representation.source_backend = "ane_int8" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      lowering.decision = "fallback_backend",
      lowering.target_backend = "cpu_ref",
      quant.activation_dtype = "fp16",
      quant.strategy = "weight_only_int8",
      quant.weight_dtype = "int8"
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 2: weight_only_int8 op direct_lowers to a backend with weight_only support
// -> kernel exact match already verified capability -> weight_dequant_required=false.
// The pass trusts lowering.decision=direct_lower and does not re-check backend caps.

// CHECK-LABEL: @weight_only_direct_lower
// CHECK: "compute.matmul"
// CHECK-SAME: boundary.fallback_backend_supports_weight_dtype = true
// CHECK-SAME: boundary.weight_dequant_reason = "kernel_exact_match_backend_supports_weight_quantization"
// CHECK-SAME: boundary.weight_dequant_required = false
// CHECK-SAME: boundary.weight_dtype_mismatch = false

module attributes {
  target.backend_capabilities.ane_int8.supported_quant_modes = ["weight_only", "static_int8"]
} {
  func.func @weight_only_direct_lower(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes { representation.source_backend = "ane_int8" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      lowering.decision = "direct_lower",
      lowering.target_backend = "ane_int8",
      quant.activation_dtype = "fp16",
      quant.strategy = "weight_only_int8",
      quant.weight_dtype = "int8"
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 3: fp16_fallback op -> not weight-quantized -> weight_dequant_required=false,
// reason=not_weight_quantized, regardless of lowering decision or backend.

// CHECK-LABEL: @fp16_fallback_no_weight_dequant
// CHECK: "compute.matmul"
// CHECK-SAME: boundary.fallback_backend_supports_weight_dtype = true
// CHECK-SAME: boundary.weight_dequant_reason = "not_weight_quantized"
// CHECK-SAME: boundary.weight_dequant_required = false
// CHECK-SAME: boundary.weight_dtype_mismatch = false

module attributes {
  target.backend_capabilities.cpu_ref.supported_quant_modes = []
} {
  func.func @fp16_fallback_no_weight_dequant(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes { representation.source_backend = "cpu_ref" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      lowering.decision = "fallback_backend",
      lowering.target_backend = "cpu_ref",
      quant.activation_dtype = "fp16",
      quant.strategy = "fp16_fallback",
      quant.weight_dtype = "fp16"
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 4: weight_only_int8 with no lowering.decision attr (standalone pass, upstream
// LoweringDecisionPlanningPass has not run) -> target_backend_unknown, no requirement
// asserted (safe/optimistic default).

// CHECK-LABEL: @weight_only_no_lowering_decision
// CHECK: "compute.matmul"
// CHECK-SAME: boundary.fallback_backend_supports_weight_dtype = true
// CHECK-SAME: boundary.weight_dequant_reason = "target_backend_unknown"
// CHECK-SAME: boundary.weight_dequant_required = false
// CHECK-SAME: boundary.weight_dtype_mismatch = false

module {
  func.func @weight_only_no_lowering_decision(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes { representation.source_backend = "ane_int8" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      quant.activation_dtype = "fp16",
      quant.strategy = "weight_only_int8",
      quant.weight_dtype = "int8"
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}
