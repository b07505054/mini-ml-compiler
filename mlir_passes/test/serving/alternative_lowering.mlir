// RUN: mlir-opt --allow-unregistered-dialect \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(alternative-lowering-planning-pipeline)' \
// RUN:   --split-input-file %s | FileCheck %s
//
// AlternativeLoweringPlanningPass tests (Commit P).
// Each section pre-annotates ops with kernel.exists/lowering_status (from
// KernelAvailabilityPlanningPass) and runs only AlternativeLoweringPlanningPass.
//
// Truth boundary: alternative_lowering_static_not_materialized_not_cost_evaluated

// Test 1: gelu with no direct kernel. Kernel library contains mul/fp16 and
// sigmoid/fp16. The pass should generate a valid algebraic_decomposition candidate
// (gelu → mul + sigmoid + mul) because all target kernels are available.

// CHECK-LABEL: @gelu_valid_decomposition
// CHECK: "compute.gelu"
// CHECK-SAME: alternative.available = true
// CHECK-SAME: alternative.candidates
// CHECK-SAME: alternative_type = "algebraic_decomposition"
// CHECK-SAME: validation_status = "valid"

module attributes {
  target.backend_capabilities.cpu.supports_cast = false,
  target.backend_capabilities.cpu.supports_dequant_boundary = false,
  target.backend_capabilities.cpu.supports_layout_transform = false,
  target.kernel_libraries.cpu = [{
    fallback_backend = "",
    kernel_library = "cpu_math",
    kernel_name = "cpu_mul_fp16",
    op_type = "mul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }, {
    fallback_backend = "",
    kernel_library = "cpu_math",
    kernel_name = "cpu_sigmoid_fp16",
    op_type = "sigmoid",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @gelu_valid_decomposition(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "cpu"
    }
  {
    %g = "compute.gelu"(%arg0) {
      kernel.exists = false,
      kernel.fallback_backend = "",
      kernel.lowering_status = "unsupported",
      quant.activation_dtype = "fp16",
      quant.strategy = "fp16_fallback"
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}

// -----

// Test 2: gelu with no direct kernel. Kernel library has mul/fp16 but NOT
// sigmoid/fp16. The algebraic_decomposition candidate is invalid because sigmoid
// kernel is missing.

// CHECK-LABEL: @gelu_missing_sigmoid_kernel
// CHECK: "compute.gelu"
// CHECK-SAME: alternative.available = false
// CHECK-SAME: alternative.candidates
// CHECK-SAME: alternative_type = "algebraic_decomposition"
// CHECK-SAME: validation_status = "invalid_missing_kernel"

module attributes {
  target.backend_capabilities.cpu.supports_cast = false,
  target.backend_capabilities.cpu.supports_dequant_boundary = false,
  target.backend_capabilities.cpu.supports_layout_transform = false,
  target.kernel_libraries.cpu = [{
    fallback_backend = "",
    kernel_library = "cpu_math",
    kernel_name = "cpu_mul_fp16",
    op_type = "mul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @gelu_missing_sigmoid_kernel(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "cpu"
    }
  {
    %g = "compute.gelu"(%arg0) {
      kernel.exists = false,
      kernel.fallback_backend = "",
      kernel.lowering_status = "unsupported",
      quant.activation_dtype = "fp16",
      quant.strategy = "fp16_fallback"
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}

// -----

// Test 3: matmul with weight_only_int8 strategy. No weight_only kernel exists, but
// an fp16 matmul kernel is available and the backend declares supports_dequant_boundary.
// The pass generates a valid representation_conversion candidate
// (dequant_weight + fp16_matmul).

// CHECK-LABEL: @weight_only_representation_conversion
// CHECK: "compute.matmul"
// CHECK-SAME: alternative.available = true
// CHECK-SAME: alternative.candidates
// CHECK-SAME: alternative_type = "representation_conversion"
// CHECK-SAME: validation_status = "valid"

module attributes {
  target.backend_capabilities.cpu.supports_cast = false,
  target.backend_capabilities.cpu.supports_dequant_boundary = true,
  target.backend_capabilities.cpu.supports_layout_transform = false,
  target.kernel_libraries.cpu = [{
    fallback_backend = "",
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
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @weight_only_representation_conversion(
      %arg0: tensor<1x256xf16>, %arg1: tensor<256x256xi8>)
      -> tensor<1x256xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "cpu"
    }
  {
    %mm = "compute.matmul"(%arg0, %arg1) {
      kernel.exists = false,
      kernel.fallback_backend = "",
      kernel.lowering_status = "fallback_required",
      quant.activation_dtype = "fp16",
      quant.strategy = "weight_only_int8",
      quant.weight_dtype = "int8"
    } : (tensor<1x256xf16>, tensor<256x256xi8>) -> tensor<1x256xf16>
    return %mm : tensor<1x256xf16>
  }
}

// -----

// Test 4: matmul missing in NCHW layout. The backend (arm_compute) declares
// supports_layout_transform = true, and the kernel library has a matmul kernel
// for NHWC. The pass generates a valid layout_conversion candidate
// (layout_transform + matmul in NHWC).

// CHECK-LABEL: @layout_conversion_candidate
// CHECK: "compute.matmul"
// CHECK-SAME: alternative.available = true
// CHECK-SAME: alternative.candidates
// CHECK-SAME: alternative_type = "layout_conversion"
// CHECK-SAME: validation_status = "valid"

module attributes {
  target.backend_capabilities.arm_compute.supports_cast = false,
  target.backend_capabilities.arm_compute.supports_dequant_boundary = false,
  target.backend_capabilities.arm_compute.supports_layout_transform = true,
  target.kernel_libraries.arm_compute = [{
    fallback_backend = "cpu",
    kernel_library = "arm_compute_lib",
    kernel_name = "acl_matmul_fp16_nhwc",
    op_type = "matmul",
    requires_constant_weight = false,
    rewrite_patterns = [],
    source_level = "declared_profile",
    supported_dtypes = ["fp16"],
    supported_layouts = ["NHWC"],
    supported_quant_modes = [],
    supports_dynamic_shape = true,
    truth_boundary = "kernel_availability_static_library_model_not_runtime_verified"
  }]
} {
  func.func @layout_conversion_candidate(%arg0: tensor<1x256x8x8xf16>) -> tensor<1x256x8x8xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "arm_compute"
    }
  {
    %mm = "compute.matmul"(%arg0, %arg0) {
      kernel.exists = false,
      kernel.fallback_backend = "cpu",
      kernel.lowering_status = "fallback_required",
      layout.effective_layout = "NCHW",
      quant.activation_dtype = "fp16",
      quant.strategy = "fp16_fallback"
    } : (tensor<1x256x8x8xf16>, tensor<1x256x8x8xf16>) -> tensor<1x256x8x8xf16>
    return %mm : tensor<1x256x8x8xf16>
  }
}

// -----

// Test 5: gelu with no ANE kernel library and a declared cpu fallback_backend.
// Algebraic decomposition is invalid (no kernels), but a backend_fallback candidate
// is generated and marked risk = last_resort.

// CHECK-LABEL: @backend_fallback_last_resort
// CHECK: "compute.gelu"
// CHECK-SAME: alternative.available = true
// CHECK-SAME: alternative_type = "backend_fallback"
// CHECK-SAME: risk = "last_resort"

module attributes {
  target.kernel_libraries.ane = []
} {
  func.func @backend_fallback_last_resort(%arg0: tensor<1x256xf16>) -> tensor<1x256xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.source_backend = "ane"
    }
  {
    %g = "compute.gelu"(%arg0) {
      kernel.exists = false,
      kernel.fallback_backend = "cpu",
      kernel.lowering_status = "fallback_required",
      quant.activation_dtype = "fp16",
      quant.strategy = "fp16_fallback"
    } : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %g : tensor<1x256xf16>
  }
}
