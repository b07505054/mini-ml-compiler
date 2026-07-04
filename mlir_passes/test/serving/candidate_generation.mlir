// Tests for CandidateGenerationPass.
// Each module below is separated by a split-input-file marker.

// Test 1: Backend with fp16/bf16/int8 generates multiple dtype candidates.
// gpu backend: preferred_dtypes=["fp16"], acceptable_dtypes=["bf16","int8"].
// One preferred activation layout ("NHWC") => 3 dtype x 1 layout = 3 candidates.
//
// CHECK-LABEL: @dtype_variants
// CHECK: candidates = [
// CHECK-SAME: dtype = "fp16"
// CHECK-SAME: dtype = "bf16"
// CHECK-SAME: dtype = "int8"
// CHECK: candidates.count = 3 : i64
// CHECK: candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"

module attributes {
  target.backend_capability_names = ["gpu"],
  target.backend_capabilities.gpu.backend_api = "opencl",
  target.backend_capabilities.gpu.preferred_dtypes = ["fp16"],
  target.backend_capabilities.gpu.acceptable_dtypes = ["bf16", "int8"],
  target.backend_capabilities.gpu.supported_dtypes = ["fp16", "bf16", "int8"],
  target.backend_capabilities.gpu.accumulation_dtypes = ["fp32"],
  target.backend_capabilities.gpu.preferred_activation_layouts = ["NHWC"],
  target.backend_capabilities.gpu.acceptable_activation_layouts = [],
  target.backend_capabilities.gpu.preferred_weight_layouts = ["OHWI"],
  target.backend_capabilities.gpu.layout_agnostic_ops = ["relu"],
  target.backend_capabilities.gpu.supported_quant_modes = ["static_int8"],
  target.backend_capabilities.gpu.supports_fusion_patterns = [],
  target.backend_capabilities.gpu.preferred_fusion_patterns = [],
  target.backend_capabilities.gpu.supports_layout_transform = true,
  target.backend_capabilities.gpu.supports_cast = true,
  target.backend_capabilities.gpu.supports_dequant_boundary = false,
  target.backend_capabilities.gpu.supports_requantize = false,
  target.backend_capabilities.gpu.source_level = "declared_profile",
  target.backend_capabilities.gpu.truth_boundary = "test_profile"
} {
  func.func @dtype_variants(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32>
    attributes {
      representation.effective_dtype = "fp32"
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<1x768xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>
    return %0 : tensor<1x768xf32>
  }
}

// -----

// Test 2: Preferred + acceptable activation layouts generate layout variants.
// tensor_core backend: preferred=["NHWC"], acceptable=["NHWC","NCHW"].
// 1 dtype x 2 layouts => 2 candidates.
//
// CHECK-LABEL: @layout_variants
// CHECK: candidates = [
// CHECK-SAME: activation_layout = "NHWC"
// CHECK-SAME: activation_layout = "NCHW"
// CHECK: candidates.count = 2 : i64

module attributes {
  target.backend_capability_names = ["tensor_core"],
  target.backend_capabilities.tensor_core.backend_api = "cuda",
  target.backend_capabilities.tensor_core.preferred_dtypes = ["bf16"],
  target.backend_capabilities.tensor_core.acceptable_dtypes = [],
  target.backend_capabilities.tensor_core.supported_dtypes = ["bf16"],
  target.backend_capabilities.tensor_core.accumulation_dtypes = ["fp32"],
  target.backend_capabilities.tensor_core.preferred_activation_layouts = ["NHWC"],
  target.backend_capabilities.tensor_core.acceptable_activation_layouts = ["NHWC", "NCHW"],
  target.backend_capabilities.tensor_core.preferred_weight_layouts = ["blocked_kc"],
  target.backend_capabilities.tensor_core.layout_agnostic_ops = [],
  target.backend_capabilities.tensor_core.supported_quant_modes = [],
  target.backend_capabilities.tensor_core.supports_fusion_patterns = ["matmul_bias_relu"],
  target.backend_capabilities.tensor_core.preferred_fusion_patterns = ["matmul_bias_gelu"],
  target.backend_capabilities.tensor_core.supports_layout_transform = true,
  target.backend_capabilities.tensor_core.supports_cast = true,
  target.backend_capabilities.tensor_core.supports_dequant_boundary = false,
  target.backend_capabilities.tensor_core.supports_requantize = false,
  target.backend_capabilities.tensor_core.source_level = "public_docs",
  target.backend_capabilities.tensor_core.truth_boundary = "test_profile"
} {
  func.func @layout_variants(%arg0: tensor<1x768xbf16>) -> tensor<1x768xbf16>
    attributes {
      representation.effective_dtype = "bf16"
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<1x768xbf16>, tensor<1x768xbf16>) -> tensor<1x768xbf16>
    return %0 : tensor<1x768xbf16>
  }
}

// -----

// Test 3: Unsupported op marks candidate fail — candidate is NOT removed.
// npu backend has unsupported_ops = ["gather"].
// The func body contains hir.gather.
// All candidates should be constraint_status = "fail" with "unsupported_op:gather".
//
// CHECK-LABEL: @unsupported_op_marks_fail
// CHECK: candidates = [
// CHECK-SAME: constraint_status = "fail"
// CHECK-SAME: failed_constraints = "unsupported_op:gather"
// Candidate is present (not discarded):
// CHECK: candidates.count = 1 : i64

module attributes {
  target.backend_capability_names = ["npu"],
  target.backend_capabilities.npu.backend_api = "qnn",
  target.backend_capabilities.npu.preferred_dtypes = ["int8"],
  target.backend_capabilities.npu.acceptable_dtypes = [],
  target.backend_capabilities.npu.supported_dtypes = ["int8"],
  target.backend_capabilities.npu.accumulation_dtypes = ["int32"],
  target.backend_capabilities.npu.preferred_activation_layouts = ["NHWC"],
  target.backend_capabilities.npu.acceptable_activation_layouts = [],
  target.backend_capabilities.npu.preferred_weight_layouts = ["OHWI"],
  target.backend_capabilities.npu.layout_agnostic_ops = [],
  target.backend_capabilities.npu.supported_quant_modes = ["static_int8"],
  target.backend_capabilities.npu.supports_fusion_patterns = [],
  target.backend_capabilities.npu.preferred_fusion_patterns = [],
  target.backend_capabilities.npu.unsupported_ops = ["gather"],
  target.backend_capabilities.npu.requires_static_shape = false,
  target.backend_capabilities.npu.supports_layout_transform = false,
  target.backend_capabilities.npu.supports_cast = true,
  target.backend_capabilities.npu.supports_dequant_boundary = true,
  target.backend_capabilities.npu.supports_requantize = false,
  target.backend_capabilities.npu.source_level = "public_docs",
  target.backend_capabilities.npu.truth_boundary = "test_profile"
} {
  func.func @unsupported_op_marks_fail(%arg0: tensor<4x256xi8>, %idx: tensor<4xi32>)
      -> tensor<4x256xi8>
    attributes {
      representation.effective_dtype = "int8"
    }
  {
    %0 = "compute.gather"(%arg0, %idx) : (tensor<4x256xi8>, tensor<4xi32>) -> tensor<4x256xi8>
    return %0 : tensor<4x256xi8>
  }
}

// -----

// Test 4: Static-shape requirement fails when func has dynamic tensor shapes.
// amx backend: requires_static_shape = true.
// Func arg has type tensor<?x768xbf16> (dynamic first dimension).
//
// CHECK-LABEL: @static_shape_constraint
// CHECK: candidates = [
// CHECK-SAME: constraint_status = "fail"
// CHECK-SAME: failed_constraints = "requires_static_shape"
// 2 candidates (bf16 and int8), both marked fail:
// CHECK: candidates.count = 2 : i64

module attributes {
  target.backend_capability_names = ["amx"],
  target.backend_capabilities.amx.backend_api = "onednn",
  target.backend_capabilities.amx.preferred_dtypes = ["bf16"],
  target.backend_capabilities.amx.acceptable_dtypes = [],
  target.backend_capabilities.amx.supported_dtypes = ["bf16", "int8"],
  target.backend_capabilities.amx.accumulation_dtypes = ["fp32"],
  target.backend_capabilities.amx.preferred_activation_layouts = ["NCHW"],
  target.backend_capabilities.amx.acceptable_activation_layouts = [],
  target.backend_capabilities.amx.preferred_weight_layouts = ["KN"],
  target.backend_capabilities.amx.layout_agnostic_ops = [],
  target.backend_capabilities.amx.supported_quant_modes = ["static_int8"],
  target.backend_capabilities.amx.supports_fusion_patterns = [],
  target.backend_capabilities.amx.preferred_fusion_patterns = [],
  target.backend_capabilities.amx.requires_static_shape = true,
  target.backend_capabilities.amx.supports_layout_transform = true,
  target.backend_capabilities.amx.supports_cast = true,
  target.backend_capabilities.amx.supports_dequant_boundary = false,
  target.backend_capabilities.amx.supports_requantize = false,
  target.backend_capabilities.amx.source_level = "public_docs",
  target.backend_capabilities.amx.truth_boundary = "test_profile"
} {
  func.func @static_shape_constraint(%arg0: tensor<?x768xbf16>) -> tensor<?x768xbf16>
    attributes {
      representation.effective_dtype = "bf16"
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<?x768xbf16>, tensor<?x768xbf16>) -> tensor<?x768xbf16>
    return %0 : tensor<?x768xbf16>
  }
}

// -----

// Test 5: Constant-weight requirement fails when representation declares
//         weights are NOT constant.
// coreml backend: requires_constant_weight = true.
// Func has representation.weights_are_constant = false.
//
// CHECK-LABEL: @constant_weight_constraint
// CHECK: candidates = [
// CHECK-SAME: constraint_status = "fail"
// CHECK-SAME: failed_constraints = "requires_constant_weight"
// 2 candidates (fp16 and fp32), both marked fail:
// CHECK: candidates.count = 2 : i64

module attributes {
  target.backend_capability_names = ["coreml"],
  target.backend_capabilities.coreml.backend_api = "coreml",
  target.backend_capabilities.coreml.preferred_dtypes = ["fp16"],
  target.backend_capabilities.coreml.acceptable_dtypes = [],
  target.backend_capabilities.coreml.supported_dtypes = ["fp16", "fp32"],
  target.backend_capabilities.coreml.accumulation_dtypes = ["fp32"],
  target.backend_capabilities.coreml.preferred_activation_layouts = ["abstracted_by_coreml"],
  target.backend_capabilities.coreml.acceptable_activation_layouts = [],
  target.backend_capabilities.coreml.preferred_weight_layouts = ["abstracted_by_coreml"],
  target.backend_capabilities.coreml.layout_agnostic_ops = [],
  target.backend_capabilities.coreml.supported_quant_modes = [],
  target.backend_capabilities.coreml.supports_fusion_patterns = [],
  target.backend_capabilities.coreml.preferred_fusion_patterns = [],
  target.backend_capabilities.coreml.requires_constant_weight = true,
  target.backend_capabilities.coreml.supports_layout_transform = false,
  target.backend_capabilities.coreml.supports_cast = true,
  target.backend_capabilities.coreml.supports_dequant_boundary = false,
  target.backend_capabilities.coreml.supports_requantize = false,
  target.backend_capabilities.coreml.source_level = "public_docs",
  target.backend_capabilities.coreml.truth_boundary = "test_profile"
} {
  func.func @constant_weight_constraint(%arg0: tensor<1x768xf16>) -> tensor<1x768xf16>
    attributes {
      representation.effective_dtype = "fp16",
      representation.weights_are_constant = false
    }
  {
    %0 = "compute.matmul"(%arg0, %arg0) : (tensor<1x768xf16>, tensor<1x768xf16>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}
