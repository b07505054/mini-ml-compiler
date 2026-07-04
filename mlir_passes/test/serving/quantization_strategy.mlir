// Tests for QuantizationStrategyPlanningPass.
// Each module below is separated by a split-input-file marker.

// Test 1: Constant-weight matmul on backend with int8 support.
// coreml backend: supported_quant_modes=["static_int8","weight_only"], weights_are_constant=true.
// Rule 1 (prefer weight-only) => strategy=weight_only_int8, weight_dtype=int8, activation=fp16.
//
// CHECK-LABEL: @constant_weight_int8_backend
// CHECK: "compute.matmul"
// CHECK-SAME: quant.activation_dtype = "fp16"
// CHECK-SAME: quant.strategy = "weight_only_int8"
// CHECK-SAME: quant.truth_boundary = "quantization_strategy_static_not_accuracy_calibrated"
// CHECK-SAME: quant.weight_dtype = "int8"

module attributes {
  target.backend_capability_names = ["coreml"],
  target.backend_capabilities.coreml.supported_quant_modes = ["static_int8", "weight_only"],
  target.backend_capabilities.coreml.supported_dtypes = ["fp16", "int8"],
  target.backend_capabilities.coreml.accumulation_dtypes = ["fp32"],
  target.backend_capabilities.coreml.allowed_quant_granularity = ["per_channel"],
  target.backend_capabilities.coreml.supports_dequant_boundary = true
} {
  func.func @constant_weight_int8_backend(%arg0: tensor<1x768xf16>) -> tensor<1x768xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "coreml",
      representation.weights_are_constant = true
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<1x768xf16>, tensor<1x768xf16>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}

// -----

// Test 2: Backend lacking activation int8 — fp16 activation, reason explains backend_lacks_activation_int8.
// gpu backend: no int8 in supported_quant_modes or supported_dtypes.
// => strategy=fp16_fallback, activation_dtype=fp16, fallback_reason=backend_lacks_activation_int8.
//
// CHECK-LABEL: @no_int8_backend
// CHECK: "compute.matmul"
// CHECK-SAME: quant.activation_dtype = "fp16"
// CHECK-SAME: quant.fallback_reason = "backend_lacks_activation_int8"
// CHECK-SAME: quant.strategy = "fp16_fallback"

module attributes {
  target.backend_capability_names = ["gpu"],
  target.backend_capabilities.gpu.supported_quant_modes = [],
  target.backend_capabilities.gpu.supported_dtypes = ["fp16", "fp32"],
  target.backend_capabilities.gpu.accumulation_dtypes = ["fp32"]
} {
  func.func @no_int8_backend(%arg0: tensor<1x768xf16>) -> tensor<1x768xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "gpu",
      representation.weights_are_constant = true
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<1x768xf16>, tensor<1x768xf16>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}

// -----

// Test 3: Softmax falls back to fp16 with accuracy_sensitive_op reason.
// Rule 5: softmax is accuracy-sensitive regardless of backend int8 support.
// => strategy=fp16_fallback, accuracy_risk=medium, fallback_reason=accuracy_sensitive_op.
//
// CHECK-LABEL: @softmax_accuracy_sensitive
// CHECK: "compute.softmax"
// CHECK-SAME: quant.accuracy_risk = "medium"
// CHECK-SAME: quant.fallback_reason = "accuracy_sensitive_op"
// CHECK-SAME: quant.strategy = "fp16_fallback"

module attributes {
  target.backend_capability_names = ["ane"],
  target.backend_capabilities.ane.supported_quant_modes = ["static_int8"],
  target.backend_capabilities.ane.supported_dtypes = ["fp16", "int8"],
  target.backend_capabilities.ane.accumulation_dtypes = ["int32"]
} {
  func.func @softmax_accuracy_sensitive(%arg0: tensor<1x768xf16>) -> tensor<1x768xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "ane"
    }
  {
    %0 = "compute.softmax"(%arg0) : (tensor<1x768xf16>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}

// -----

// Test 4: Non-constant weight fails weight quantization.
// Backend has int8 support but representation.weights_are_constant = false.
// => strategy=fp16_fallback, fallback_reason=requires_constant_weight.
//
// CHECK-LABEL: @non_constant_weight
// CHECK: "compute.matmul"
// CHECK-SAME: quant.fallback_reason = "requires_constant_weight"
// CHECK-SAME: quant.strategy = "fp16_fallback"

module attributes {
  target.backend_capability_names = ["coreml"],
  target.backend_capabilities.coreml.supported_quant_modes = ["static_int8", "weight_only"],
  target.backend_capabilities.coreml.supported_dtypes = ["fp16", "int8"],
  target.backend_capabilities.coreml.accumulation_dtypes = ["fp32"]
} {
  func.func @non_constant_weight(%arg0: tensor<1x768xf16>) -> tensor<1x768xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "coreml",
      representation.weights_are_constant = false
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<1x768xf16>, tensor<1x768xf16>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}

// -----

// Test 5: Op producing int8 output while effective dtype is fp16 marks dequant boundary.
// Rule 6: quantized op output feeds fp16/fp32 region => requires_dequant_boundary=true.
// The op result type is tensor<1x768xi8> (int8) but effective_dtype is fp16.
//
// CHECK-LABEL: @quantized_to_float_boundary
// CHECK: "compute.matmul"
// CHECK-SAME: quant.output_dtype = "int8"
// CHECK-SAME: quant.requires_dequant_boundary = true
// CHECK-SAME: quant.strategy = "weight_only_int8"

module attributes {
  target.backend_capability_names = ["coreml"],
  target.backend_capabilities.coreml.supported_quant_modes = ["static_int8"],
  target.backend_capabilities.coreml.supported_dtypes = ["fp16", "int8"],
  target.backend_capabilities.coreml.accumulation_dtypes = ["int32"],
  target.backend_capabilities.coreml.supports_dequant_boundary = true
} {
  func.func @quantized_to_float_boundary(%arg0: tensor<1x768xf16>) -> tensor<1x768xi8>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "coreml",
      representation.weights_are_constant = true
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<1x768xf16>, tensor<1x768xf16>) -> tensor<1x768xi8>
    return %0 : tensor<1x768xi8>
  }
}
