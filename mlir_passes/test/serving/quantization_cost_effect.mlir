// FileCheck test: quantization.plan_dtype flows into kv and serving cost attrs.
//
// Run with the HIR plugin loaded:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(quantization-planning,serving-phase-analysis,kv-layout-planning)' \
//     --split-input-file | FileCheck %s
//
// Three cases:
//   A: llm.dtype="fp16", supported=["fp16"]  → dtype_bytes=2.0 on kv and quantization attrs
//   B: llm.dtype="int8", supported=["fp16","int8"] → dtype_bytes=1.0
//   C: no quantization attrs                  → fallback dtype_bytes=2.0

// Case A — fp16 plan dtype; byte estimates use 2 bytes/element.
// CHECK-LABEL: func.func @prefill_fp16
// CHECK-SAME:  kv.dtype_bytes = 2.
// CHECK-SAME:  kv.layout = "paged"
// CHECK-SAME:  quantization.dtype_bytes = 2.
// CHECK-SAME:  quantization.effective_dtype = "fp16"
// CHECK-SAME:  quantization.truth_boundary = "precision_selection_from_target_profile_not_calibrated"
// CHECK-SAME:  serving.policy

module attributes {
  llm.dtype = "fp16",
  target.supported_precisions = ["fp16"],
  llm.num_layers = 4 : i64,
  llm.hidden_size = 64 : i64
} {
  func.func @prefill_fp16(%x: tensor<1x64xf16>) -> tensor<1x64xf16> {
    %0 = "llm.attention_prefill"(%x, %x, %x) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 32 : i64,
      serving.output_tokens = 16 : i64
    } : (tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
    return %0 : tensor<1x64xf16>
  }
}

// -----

// Case B — int8 plan dtype; byte estimates use 1 byte/element (halved KV footprint).
// CHECK-LABEL: func.func @prefill_int8
// CHECK-SAME:  kv.dtype_bytes = 1.
// CHECK-SAME:  quantization.dtype_bytes = 1.
// CHECK-SAME:  quantization.effective_dtype = "int8"
// CHECK-SAME:  quantization.truth_boundary = "precision_selection_from_target_profile_not_calibrated"

module attributes {
  llm.dtype = "int8",
  target.supported_precisions = ["fp16", "int8"],
  llm.num_layers = 4 : i64,
  llm.hidden_size = 64 : i64
} {
  func.func @prefill_int8(%x: tensor<1x64xf16>) -> tensor<1x64xf16> {
    %0 = "llm.attention_prefill"(%x, %x, %x) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 32 : i64,
      serving.output_tokens = 16 : i64
    } : (tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
    return %0 : tensor<1x64xf16>
  }
}

// -----

// Case C — no quantization attrs; fallback to fp16 / 2 bytes/element.
// CHECK-LABEL: func.func @prefill_default
// CHECK-SAME:  kv.dtype_bytes = 2.
// CHECK-SAME:  quantization.dtype_bytes = 2.
// CHECK-SAME:  quantization.effective_dtype = "fp16"
// CHECK-SAME:  quantization.truth_boundary = "precision_selection_from_target_profile_not_calibrated"

module attributes {
  llm.num_layers = 4 : i64,
  llm.hidden_size = 64 : i64
} {
  func.func @prefill_default(%x: tensor<1x64xf16>) -> tensor<1x64xf16> {
    %0 = "llm.attention_prefill"(%x, %x, %x) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 32 : i64,
      serving.output_tokens = 16 : i64
    } : (tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
    return %0 : tensor<1x64xf16>
  }
}
