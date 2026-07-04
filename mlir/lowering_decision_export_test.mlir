// Test fixture for LoweringDecisionPlanningPass export (Commit K).
// Used with lowering_decision_export_test.json profile.
//
// Designed to produce 4 of 5 lowering decision outcomes through compile-for-target:
//   hir.matmul   -> direct_lower    (fp16 kernel matches fp16 activation)
//   hir.conv2d   -> rewrite_then_lower (fp32-only kernel + rewrite pattern for fp16)
//   hir.softmax  -> fallback_backend   (fp32-only kernel + fallback, no rewrite)
//   hir.gelu     -> unsupported        (not in kernel library)
//
// dequant_then_lower is covered by ServingExecutionPlanBuilderTest (pre-annotated MLIR).
// Truth boundary: lowering_decision_export_static_not_backend_codegen_verified

module attributes {
  llm.model = "lowering-decision-export-test",
  llm.num_layers = 2 : i64,
  llm.hidden_size = 128 : i64
} {
  func.func @prefill(%x: tensor<1x128xf16>) -> tensor<1x128xf16> {
    // Triggers ServingPhaseAnalysisPass -> serving.policy = "colocated".
    %attn = "llm.attention_prefill"(%x, %x, %x) {
      kv_cache.role = "producer",
      serving.phase = "prefill"
    } : (tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>) -> tensor<1x128xf16>

    // hir.matmul: quant.activation_dtype = "fp16" (weight_only_int8).
    // Kernel library: matmul supports ["fp16"] -> exact match -> lowerable -> direct_lower.
    %mm = "hir.matmul"(%attn, %attn) : (tensor<1x128xf16>, tensor<1x128xf16>) -> tensor<1x128xf16>

    // hir.conv2d: quant.activation_dtype = "fp16".
    // Kernel library: conv2d supports ["fp32"] + rewrite ["fp16_to_fp32_cast"]
    // -> no exact match, has rewrite -> rewrite_candidate -> rewrite_then_lower.
    %conv = "hir.conv2d"(%mm, %mm) : (tensor<1x128xf16>, tensor<1x128xf16>) -> tensor<1x128xf16>

    // hir.softmax: accuracy-sensitive -> fp16_fallback -> activation_dtype = "fp16".
    // Kernel library: softmax supports ["fp32"] + fallback_backend="cpu_ref", no rewrite
    // -> no exact match, no rewrite, has fallback -> fallback_required
    // -> boundary.dequant_required = false (effectiveDtype=fp16, opDtype=fp16 both float)
    // -> fallback_backend.
    %sm = "hir.softmax"(%conv) : (tensor<1x128xf16>) -> tensor<1x128xf16>

    // hir.gelu: not in kernel library -> unsupported.
    %gelu = "hir.gelu"(%sm) : (tensor<1x128xf16>) -> tensor<1x128xf16>

    return %gelu : tensor<1x128xf16>
  }
}
