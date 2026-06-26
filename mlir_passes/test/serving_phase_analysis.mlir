// RUN: mlir-opt %s --allow-unregistered-dialect \
// RUN:   --load-dialect-plugin=%plugin \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(serving-phase-analysis)' \
// RUN: | FileCheck %s

// ServingPhaseAnalysisPass annotates func.func operations that contain
// llm.attention_prefill or llm.attention_decode with formula-synthetic serving
// cost and policy attributes. Cost source: formula_synthetic. Truth boundary:
// estimated; not measured latency for any model or hardware.
//
// With prompt_tokens=128, output_tokens=64, num_layers=12, hidden_size=768:
//   compute_ms     = 0.08*128 + 0.12*64 = 17.92
//   col_queue_ms   = 0.02*128 = 2.56       -> colocated_total = 20.48
//   pd_queue_ms    = 0.02*128*0.4 + 2.0 = 3.024
//   kv_mb          = 12*2*768*2*192/1024/1024 ≈ 6.75   kv_xfer ≈ 0.281
//                                           -> pd_split_total ≈ 21.225
//   margin ≈ 0.745 ms  (3.64% of min) -> confidence = "low"
//   policy = "colocated"

// Attributes are printed in alphabetical order by the MLIR printer.
// CHECK-LABEL: func.func @tiny_gpt_prefill
// CHECK-SAME:  serving.colocated_total_ms
// CHECK-SAME:  serving.confidence = "low"
// CHECK-SAME:  serving.cost_source = "formula_synthetic"
// CHECK-SAME:  serving.decision_margin_ms
// CHECK-SAME:  serving.pd_split_total_ms
// CHECK-SAME:  serving.policy = "colocated"
// CHECK-SAME:  serving.truth_boundary = "estimated_cost_not_measured_latency"

// CHECK-LABEL: func.func @tiny_gpt_decode
// CHECK-SAME:  serving.colocated_total_ms
// CHECK-SAME:  serving.confidence = "low"
// CHECK-SAME:  serving.cost_source = "formula_synthetic"
// CHECK-SAME:  serving.decision_margin_ms
// CHECK-SAME:  serving.pd_split_total_ms
// CHECK-SAME:  serving.policy = "colocated"
// CHECK-SAME:  serving.truth_boundary = "estimated_cost_not_measured_latency"

module attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  llm.num_heads = 12 : i64,
  llm.intermediate_size = 3072 : i64,
  llm.vocab_size = 50257 : i64
} {
  func.func @tiny_gpt_prefill(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.embed"(%tokens) : (tensor<?xi32>) -> tensor<?x768xf16>
    %1 = "llm.rmsnorm"(%0) : (tensor<?x768xf16>) -> tensor<?x768xf16>
    %2 = "llm.attention_prefill"(%1, %1, %1) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?x768xf16>, tensor<?x768xf16>, tensor<?x768xf16>) -> tensor<?x768xf16>
    %3 = "llm.mlp"(%2) : (tensor<?x768xf16>) -> tensor<?x768xf16>
    return %3 : tensor<?x768xf16>
  }

  func.func @tiny_gpt_decode(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.embed"(%token) : (tensor<1xi32>) -> tensor<1x768xf16>
    %1 = "llm.rmsnorm"(%0) : (tensor<1x768xf16>) -> tensor<1x768xf16>
    %2 = "llm.attention_decode"(%1, %1, %1) {
      kv_cache.role = "consumer",
      serving.phase = "decode",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1x768xf16>, tensor<1x768xf16>, tensor<1x768xf16>) -> tensor<1x768xf16>
    %3 = "llm.mlp"(%2) : (tensor<1x768xf16>) -> tensor<1x768xf16>
    return %3 : tensor<1x768xf16>
  }
}
