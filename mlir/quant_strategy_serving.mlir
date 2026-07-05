// Test fixture for QuantizationStrategyPlanningPass export (Commit H).
// Contains llm.attention_prefill to trigger serving.policy annotation plus
// hir.matmul (quantizable) and hir.softmax (accuracy-sensitive) to exercise
// both weight_only_int8 and fp16_fallback quant strategy paths.

module attributes {
  llm.model = "quant-strategy-test",
  llm.num_layers = 4 : i64,
  llm.hidden_size = 256 : i64
} {
  func.func @quant_strategy_prefill(%x: tensor<1x256xf16>) -> tensor<1x256xf16> {
    // Triggers ServingPhaseAnalysisPass -> serving.policy annotation.
    %attn = "llm.attention_prefill"(%x, %x, %x) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 16 : i64,
      serving.output_tokens = 8 : i64
    } : (tensor<1x256xf16>, tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    // hir.matmul: weight.is_constant=true tells WeightClassificationPlanningPass to
    // classify the weight operand as constant (Rule 2 op-level override) -> weight_only_int8.
    %mm = "hir.matmul"(%attn, %attn) { weight.is_constant = true }
        : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
    // hir.softmax: accuracy-sensitive -> fp16_fallback.
    %sm = "hir.softmax"(%mm) : (tensor<1x256xf16>) -> tensor<1x256xf16>
    return %sm : tensor<1x256xf16>
  }
}
