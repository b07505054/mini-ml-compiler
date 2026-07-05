// RUN: mlir-opt --allow-unregistered-dialect \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(kernel-availability-planning-pipeline)' \
// RUN:   --split-input-file %s | FileCheck %s
//
// KernelAvailabilityPlanningPass tests (Commit I).
// Each section exercises one matching rule.
// Truth boundary: kernel_availability_static_library_model_not_runtime_verified

// Test 1: Exact match (fp16 + NCHW) -> lowerable.

// CHECK-LABEL: @exact_kernel_match
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.backend = "cpu"
// CHECK-SAME: kernel.decision_reason = "exact_kernel_match"
// CHECK-SAME: kernel.exists = true
// CHECK-SAME: kernel.fallback_backend = "cpu_simd"
// CHECK-SAME: kernel.library = "cpu_blas"
// CHECK-SAME: kernel.lowering_status = "lowerable"
// CHECK-SAME: kernel.name = "cpu_matmul_fp16"
// CHECK-SAME: kernel.required_rewrite = ""
// CHECK-SAME: kernel.truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"

module attributes {
  target.kernel_libraries.cpu = [{
    fallback_backend = "cpu_simd",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "cpu_blas",
    kernel_name = "cpu_matmul_fp16",
    op_type = "matmul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = ["NCHW"],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @exact_kernel_match(%x: tensor<1x256xf16>) -> tensor<1x256xf16> attributes {
    representation.source_backend = "cpu"
  } {
    %mm = "compute.matmul"(%x, %x) {
      layout.effective_layout = "NCHW",
      quant.activation_dtype = "fp16"
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 2: Quantized dtype not in supported_dtypes -> fallback_required.
// Kernel only supports fp16; op requests int8. fallback_backend = "cpu_simd".

// CHECK-LABEL: @missing_quantized_kernel
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.backend = "cpu"
// CHECK-SAME: kernel.decision_reason = "no_exact_kernel_fallback_to_cpu_simd"
// CHECK-SAME: kernel.exists = false
// CHECK-SAME: kernel.fallback_backend = "cpu_simd"
// CHECK-SAME: kernel.lowering_status = "fallback_required"

module attributes {
  target.kernel_libraries.cpu = [{
    fallback_backend = "cpu_simd",
    fallback_kernel = "cpu_simd_matmul",
    fusion_patterns = [],
    kernel_library = "cpu_blas",
    kernel_name = "cpu_matmul_fp16",
    op_type = "matmul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @missing_quantized_kernel(%x: tensor<1x256xi8>) -> tensor<1x256xi8> attributes {
    representation.source_backend = "cpu"
  } {
    %mm = "compute.matmul"(%x, %x) {
      quant.activation_dtype = "int8"
    } : (tensor<1x256xi8>, tensor<1x256xi8>) -> tensor<1x256xi8>
    return %mm : tensor<1x256xi8>
  }
}

// -----

// Test 3: Rewrite pattern declared -> rewrite_candidate.
// No fallback_backend. Kernel has rewrite_patterns = ["int8_to_fp16_cast"].
// Expected: lowering_status = "rewrite_candidate", required_rewrite = "int8_to_fp16_cast".

// CHECK-LABEL: @rewrite_pattern_match
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.backend = "cpu"
// CHECK-SAME: kernel.decision_reason = "rewrite_pattern_available"
// CHECK-SAME: kernel.exists = false
// CHECK-SAME: kernel.lowering_status = "rewrite_candidate"
// CHECK-SAME: kernel.required_rewrite = "int8_to_fp16_cast"

module attributes {
  target.kernel_libraries.cpu = [{
    fallback_backend = "",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "cpu_blas",
    kernel_name = "cpu_matmul_rewrite",
    op_type = "matmul",
    requires_constant_weight = false,
    rewrite_patterns = ["int8_to_fp16_cast"],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @rewrite_pattern_match(%x: tensor<1x256xi8>) -> tensor<1x256xi8> attributes {
    representation.source_backend = "cpu"
  } {
    %mm = "compute.matmul"(%x, %x) {
      quant.activation_dtype = "int8"
    } : (tensor<1x256xi8>, tensor<1x256xi8>) -> tensor<1x256xi8>
    return %mm : tensor<1x256xi8>
  }
}

// -----

// Test 4: Arm Compute Library style - explicit (op, layout, dtype) tuple.
// Backend "arm_compute" has conv2d kernel for fp16 + NHWC.
// Op carries matching attrs -> exact match -> lowerable.

// CHECK-LABEL: @arm_style_layout_dtype_match
// CHECK: "compute.conv2d"
// CHECK-SAME: kernel.backend = "arm_compute"
// CHECK-SAME: kernel.decision_reason = "exact_kernel_match"
// CHECK-SAME: kernel.exists = true
// CHECK-SAME: kernel.library = "arm_compute_lib"
// CHECK-SAME: kernel.lowering_status = "lowerable"
// CHECK-SAME: kernel.name = "arm_compute_conv2d_nhwc_fp16"

module attributes {
  target.kernel_libraries.arm_compute = [{
    fallback_backend = "cpu",
    fallback_kernel = "",
    fusion_patterns = ["conv2d_bias_relu"],
    kernel_library = "arm_compute_lib",
    kernel_name = "arm_compute_conv2d_nhwc_fp16",
    op_type = "conv2d",
    requires_constant_weight = true,
    rewrite_patterns = [],
    source_level = "public_docs",
    supported_dtypes = ["fp16"],
    supported_layouts = ["NHWC"],
    supported_quant_modes = [],
    supports_dynamic_shape = false,
    supports_fusion = true,
    truth_boundary = "arm_compute_lib_op_coverage_github_docs"
  }]
} {
  func.func @arm_style_layout_dtype_match(%x: tensor<1x32x32x64xf16>) -> tensor<1x32x32x64xf16> attributes {
    representation.source_backend = "arm_compute"
  } {
    %conv = "compute.conv2d"(%x, %x) {
      layout.effective_layout = "NHWC",
      quant.activation_dtype = "fp16"
    } : (tensor<1x32x32x64xf16>, tensor<1x32x32x64xf16>) -> tensor<1x32x32x64xf16>
    return %conv : tensor<1x32x32x64xf16>
  }
}

// -----

// Test 5: Apple CoreML - public_docs source_level, truth_boundary says no ANE internals.
// kernel_library = "coreml_builtin" (the public CoreML abstraction layer).
// truth_boundary = "coreml_public_api_no_ane_kernel_internals".
// Expected: kernel.exists = true, library = "coreml_builtin" (not a private ANE name).

// CHECK-LABEL: @apple_coreml_no_ane_internals
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.backend = "coreml"
// CHECK-SAME: kernel.exists = true
// CHECK-SAME: kernel.library = "coreml_builtin"
// CHECK-SAME: kernel.lowering_status = "lowerable"
// CHECK-SAME: kernel.truth_boundary = "coreml_public_api_no_ane_kernel_internals"

module attributes {
  target.kernel_libraries.coreml = [{
    fallback_backend = "cpu",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "coreml_builtin",
    kernel_name = "coreml_matmul",
    op_type = "matmul",
    requires_constant_weight = true,
    rewrite_patterns = [],
    source_level = "public_docs",
    supported_dtypes = ["fp16", "fp32"],
    supported_layouts = ["abstracted_by_coreml"],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "coreml_public_api_no_ane_kernel_internals"
  }]
} {
  func.func @apple_coreml_no_ane_internals(%x: tensor<1x256xf16>) -> tensor<1x256xf16> attributes {
    representation.source_backend = "coreml"
  } {
    %mm = "compute.matmul"(%x, %x) {
      layout.effective_layout = "abstracted_by_coreml",
      quant.activation_dtype = "fp16"
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test N1: weight_only_int8 quant strategy + requires_constant_weight=true +
// weight.constant_satisfied=true -> exact match -> lowerable.
// Verifies opQuantMode is derived from quant.strategy ("weight_only_int8" -> "weight_only"),
// not from quant.activation_quant_mode.

// CHECK-LABEL: @weight_only_constant_satisfied_true
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.decision_reason = "exact_kernel_match"
// CHECK-SAME: kernel.exists = true
// CHECK-SAME: kernel.lowering_status = "lowerable"

module attributes {
  target.kernel_libraries.ane_int8 = [{
    fallback_backend = "cpu",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "ane_weight_only_lib",
    kernel_name = "ane_weight_only_matmul",
    op_type = "matmul",
    requires_constant_weight = true,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = ["weight_only"],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @weight_only_constant_satisfied_true(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes { representation.source_backend = "ane_int8" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      quant.activation_dtype = "fp16",
      quant.strategy = "weight_only_int8",
      weight.constant_satisfied = true
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test N2: weight_only_int8 + requires_constant_weight=true +
// weight.constant_satisfied=false -> constWeightFailed -> fallback_required.
// Verifies enforceConstantWeight when weight.constant_satisfied attr is explicitly set.

// CHECK-LABEL: @weight_only_constant_satisfied_false
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.decision_reason = "requires_constant_weight_fallback_to_cpu"
// CHECK-SAME: kernel.exists = false
// CHECK-SAME: kernel.fallback_backend = "cpu"
// CHECK-SAME: kernel.lowering_status = "fallback_required"

module attributes {
  target.kernel_libraries.ane_int8 = [{
    fallback_backend = "cpu",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "ane_weight_only_lib",
    kernel_name = "ane_weight_only_matmul",
    op_type = "matmul",
    requires_constant_weight = true,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = ["weight_only"],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @weight_only_constant_satisfied_false(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes { representation.source_backend = "ane_int8" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      quant.activation_dtype = "fp16",
      quant.strategy = "weight_only_int8",
      weight.constant_satisfied = false
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test N3: fp16_fallback strategy -> opQuantMode="none" -> empty supported_quant_modes
// matches any mode -> exact match -> lowerable.
// Verifies backward compat: fp16_fallback ops find kernels with open quant mode lists.

// CHECK-LABEL: @fp16_fallback_open_quant_modes
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.decision_reason = "exact_kernel_match"
// CHECK-SAME: kernel.exists = true
// CHECK-SAME: kernel.lowering_status = "lowerable"

module attributes {
  target.kernel_libraries.cpu = [{
    fallback_backend = "cpu_ref",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "cpu_blas",
    kernel_name = "cpu_matmul_fp16_generic",
    op_type = "matmul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @fp16_fallback_open_quant_modes(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes { representation.source_backend = "cpu" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      quant.activation_dtype = "fp16",
      quant.strategy = "fp16_fallback"
    } : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test N4: static_int8 strategy -> opQuantMode="static_int8" -> matched against
// supported_quant_modes=["static_int8"] -> exact match -> lowerable.
// Verifies the static_int8 branch in opQuantMode derivation.

// CHECK-LABEL: @static_int8_quant_mode_match
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.decision_reason = "exact_kernel_match"
// CHECK-SAME: kernel.exists = true
// CHECK-SAME: kernel.lowering_status = "lowerable"

module attributes {
  target.kernel_libraries.ane_static = [{
    fallback_backend = "",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "ane_static_lib",
    kernel_name = "ane_static_int8_matmul",
    op_type = "matmul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["int8"],
    supported_layouts = [],
    supported_quant_modes = ["static_int8"],
    supports_dynamic_shape = true,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @static_int8_quant_mode_match(%arg0: tensor<1x256xi8>) -> tensor<1x256xi8>
    attributes { representation.source_backend = "ane_static" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      quant.activation_dtype = "int8",
      quant.strategy = "static_int8"
    } : (tensor<1x256xi8>, tensor<1x256xi8>) -> tensor<1x256xi8>
    return %mm : tensor<1x256xi8>
  }
}

// -----

// Test N5: supports_dynamic_shape=false + op has dynamic result shape ->
// dynamicShapeFailed -> decision_reason=dynamic_shape_unsupported.
// Verifies hasDynamicShape detection via ShapedType::hasStaticShape on result types.

// CHECK-LABEL: @dynamic_shape_rejected
// CHECK: "compute.matmul"
// CHECK-SAME: kernel.decision_reason = "dynamic_shape_unsupported"
// CHECK-SAME: kernel.exists = false
// CHECK-SAME: kernel.lowering_status = "fallback_required"

module attributes {
  target.kernel_libraries.cpu_static = [{
    fallback_backend = "cpu",
    fallback_kernel = "",
    fusion_patterns = [],
    kernel_library = "cpu_blas_static",
    kernel_name = "cpu_static_matmul",
    op_type = "matmul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supports_dynamic_shape = false,
    supports_fusion = false,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @dynamic_shape_rejected(%arg0: tensor<?x256xf16>) -> tensor<?x256xf16>
    attributes { representation.source_backend = "cpu_static" }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      quant.activation_dtype = "fp16"
    } : (tensor<?x256xf16>, tensor<?x256xf16>) -> tensor<?x256xf16>
    return %mm : tensor<?x256xf16>
  }
}
