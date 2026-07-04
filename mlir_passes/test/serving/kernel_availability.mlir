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
