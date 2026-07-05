// Apple/CoreML quantization-lowering demo (Commit L).
//
// Shows the complete compiler decision chain on one realistic graph:
//   hardware profile (apple-quant-lowering-demo)
//   -> serving-phase-analysis  (serving.policy = "colocated")
//   -> kv-layout-planning      (paged, coreml in pagedKVCompatibleBackends)
//   -> execution-provider-planning (coreml = primary, target_preferred)
//   -> representation-planning
//   -> layout-planning
//   -> boundary-planning
//   -> quantization-strategy-planning
//   -> kernel-availability-planning
//   -> lowering-decision-planning
//   -> execution plan JSON
//
// Expected per-op outcomes:
//
//   llm.attention_prefill   unsupported (no coreml kernel for attention)
//   hir.matmul              weight_only_int8 / fallback_backend
//                             requiresConstantWeight=true on coreml_matmul but
//                             weight is a runtime tensor -> kernel_exists=false
//                             -> falls back to cpu
//   hir.conv2d              weight_only_int8 / fallback_backend  (same reason)
//   hir.softmax             fp16_fallback / direct_lower
//                             requiresConstantWeight=false, fp16 in supportedDtypes
//                             -> kernel_exists=true -> lowerable -> direct_lower
//   hir.gelu                unsupported (not in coreml kernel library)
//
// Truth boundary: lowering_decision_static_not_backend_codegen_verified
//                 quantization_strategy_static_not_accuracy_calibrated

module attributes {
  llm.model      = "apple-coreml-quant-lowering-demo",
  llm.num_layers = 4 : i64,
  llm.hidden_size = 512 : i64
} {
  func.func @prefill(%x: tensor<1x512xf16>) -> tensor<1x512xf16> {
    // Attention op anchors serving-phase-analysis and paged-KV selection.
    %attn = "llm.attention_prefill"(%x, %x, %x) {
      kv_cache.role = "producer",
      serving.phase = "prefill"
    } : (tensor<1x512xf16>, tensor<1x512xf16>, tensor<1x512xf16>) -> tensor<1x512xf16>

    // Matmul — weight_only_int8 (activation_dtype=fp16), fallback_backend.
    %mm = "hir.matmul"(%attn, %attn) :
        (tensor<1x512xf16>, tensor<1x512xf16>) -> tensor<1x512xf16>

    // Conv2d — weight_only_int8, fallback_backend (same reason as matmul).
    %conv = "hir.conv2d"(%mm, %mm) :
        (tensor<1x512xf16>, tensor<1x512xf16>) -> tensor<1x512xf16>

    // Softmax — accuracy-sensitive: fp16_fallback strategy, direct_lower lowering.
    %sm = "hir.softmax"(%conv) :
        (tensor<1x512xf16>) -> tensor<1x512xf16>

    // Gelu — not in coreml kernelLibrary: unsupported.
    %gelu = "hir.gelu"(%sm) :
        (tensor<1x512xf16>) -> tensor<1x512xf16>

    return %gelu : tensor<1x512xf16>
  }
}
