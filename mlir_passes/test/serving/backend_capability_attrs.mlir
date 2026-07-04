// FileCheck test for target.backend_capabilities.{backend}.* MLIR module attrs.
//
// Verifies:
// 1. Backend capability attrs survive the serving-optimization-pipeline unchanged.
// 2. Known capability fields are present in the output.
// 3. Cost fields that were NOT set are absent from the output (not zero).
//
// The module below represents what attachToModule() produces for a profile with:
//   - One "coreml" backend (public_docs, layout abstracted, no cost fields)
//   - One "cpu" backend (public_docs, NHWC, alignment declared, no cost fields)
//
// Run:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(serving-optimization-pipeline)' \
//   | FileCheck %s

// -- Backend capability index ------------------------------------------------
// CHECK: target.backend_capability_names = ["coreml", "cpu"]

// -- coreml backend: public_docs, layout abstracted by Core ML runtime --------
// CHECK: target.backend_capabilities.coreml.backend_api = "coreml"
// CHECK: target.backend_capabilities.coreml.preferred_activation_layouts = ["abstracted_by_coreml"]
// CHECK: target.backend_capabilities.coreml.source_level = "public_docs"
// CHECK: target.backend_capabilities.coreml.supports_cast = true
// CHECK: target.backend_capabilities.coreml.supports_dequant_boundary = true
// CHECK: target.backend_capabilities.coreml.supports_layout_transform = false
// CHECK: target.backend_capabilities.coreml.truth_boundary = "dtype_from_coreml_docs_layout_managed_by_coreml_runtime_cost_unknown"

// Cost fields absent for coreml backend (unknown = absent, not zero).
// CHECK-NOT: target.backend_capabilities.coreml.layout_transform_cost_ms
// CHECK-NOT: target.backend_capabilities.coreml.cast_cost_ms
// CHECK-NOT: target.backend_capabilities.coreml.quantize_cost_ms
// CHECK-NOT: target.backend_capabilities.coreml.dequantize_cost_ms
// CHECK-NOT: target.backend_capabilities.coreml.requantize_cost_ms
// CHECK-NOT: target.backend_capabilities.coreml.backend_transfer_cost_ms

// -- cpu backend: public_docs, NHWC, 32-element alignment --------------------
// CHECK: target.backend_capabilities.cpu.backend_api = "arm_compute"
// CHECK: target.backend_capabilities.cpu.preferred_activation_layouts = ["NHWC"]
// CHECK: target.backend_capabilities.cpu.required_k_alignment = 32
// CHECK: target.backend_capabilities.cpu.required_n_alignment = 32
// CHECK: target.backend_capabilities.cpu.source_level = "public_docs"
// CHECK: target.backend_capabilities.cpu.supports_dequant_boundary = true
// CHECK: target.backend_capabilities.cpu.supports_requantize = true

// Cost fields absent for cpu backend (unknown = absent, not zero).
// CHECK-NOT: target.backend_capabilities.cpu.layout_transform_cost_ms
// CHECK-NOT: target.backend_capabilities.cpu.cast_cost_ms

module attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 4 : i64,
  llm.hidden_size = 512 : i64,
  target.profile_id = "apple-a17pro-mobile",
  target.preferred_backend = "coreml",
  target.allowed_backends = ["coreml", "metal", "cpu"],
  target.supported_precisions = ["fp16"],
  target.paged_kv_compatible_backends = [],
  target.static_shape_support = true,

  // Backend capability index.
  target.backend_capability_names = ["coreml", "cpu"],

  // coreml: Core ML abstract backend.
  // Source: developer.apple.com/documentation/coreml
  // Layout is managed by Core ML runtime — compiler does not control it.
  // Cost fields absent: ANE/GPU internals are not publicly documented.
  target.backend_capabilities.coreml.backend_api = "coreml",
  target.backend_capabilities.coreml.supported_ops = ["matmul", "conv2d", "relu", "softmax"],
  target.backend_capabilities.coreml.supported_dtypes = ["fp32", "fp16", "int8"],
  target.backend_capabilities.coreml.accumulation_dtypes = ["fp32"],
  target.backend_capabilities.coreml.supported_quant_modes = ["static_int8", "weight_only"],
  target.backend_capabilities.coreml.preferred_activation_layouts = ["abstracted_by_coreml"],
  target.backend_capabilities.coreml.preferred_weight_layouts = ["abstracted_by_coreml"],
  target.backend_capabilities.coreml.layout_agnostic_ops = ["relu", "add", "softmax", "reshape"],
  target.backend_capabilities.coreml.supports_layout_transform = false,
  target.backend_capabilities.coreml.supports_cast = true,
  target.backend_capabilities.coreml.supports_dequant_boundary = true,
  target.backend_capabilities.coreml.supports_requantize = false,
  target.backend_capabilities.coreml.supports_fusion_patterns = [],
  target.backend_capabilities.coreml.source_level = "public_docs",
  target.backend_capabilities.coreml.truth_boundary = "dtype_from_coreml_docs_layout_managed_by_coreml_runtime_cost_unknown",

  // cpu: Arm Compute Library CPU backend.
  // Source: github.com/ARM-software/ComputeLibrary — NHWC preferred for convolutions.
  // required_k/n_alignment: declared_profile (derived from NEON 128-bit = 32 int8 lanes).
  // Cost fields absent: per-op throughput varies by Cortex-A model and NEON config.
  target.backend_capabilities.cpu.backend_api = "arm_compute",
  target.backend_capabilities.cpu.supported_ops = ["matmul", "conv2d", "relu", "add", "softmax", "reshape"],
  target.backend_capabilities.cpu.supported_dtypes = ["fp32", "fp16", "int8", "uint8"],
  target.backend_capabilities.cpu.accumulation_dtypes = ["fp32"],
  target.backend_capabilities.cpu.supported_quant_modes = ["static_int8", "dynamic_int8"],
  target.backend_capabilities.cpu.preferred_activation_layouts = ["NHWC"],
  target.backend_capabilities.cpu.preferred_weight_layouts = ["OHWI"],
  target.backend_capabilities.cpu.layout_agnostic_ops = ["relu", "add", "reshape", "gather", "softmax"],
  target.backend_capabilities.cpu.supports_layout_transform = true,
  target.backend_capabilities.cpu.supports_cast = true,
  target.backend_capabilities.cpu.supports_dequant_boundary = true,
  target.backend_capabilities.cpu.supports_requantize = true,
  target.backend_capabilities.cpu.supports_fusion_patterns = ["matmul_bias_relu"],
  target.backend_capabilities.cpu.required_k_alignment = 32 : i64,
  target.backend_capabilities.cpu.required_n_alignment = 32 : i64,
  target.backend_capabilities.cpu.source_level = "public_docs",
  target.backend_capabilities.cpu.truth_boundary = "layout_from_arm_compute_library_docs_alignment_declared_profile_cost_unknown"
} {
  func.func @prefill(%tokens: tensor<?xi32>) -> tensor<?x512xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 64 : i64,
      serving.output_tokens = 32 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x512xf16>
    return %0 : tensor<?x512xf16>
  }

  func.func @decode(%token: tensor<1xi32>) -> tensor<1x512xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.phase = "decode",
      serving.prompt_tokens = 64 : i64,
      serving.output_tokens = 32 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x512xf16>
    return %0 : tensor<1x512xf16>
  }
}
