#pragma once

#include "frontend/graph_builder.h"

#include <cstdint>
#include <string>

// Serving-aware model configuration for LLM graph construction.
// Fields match the QwenModelSpec consumed by the qwen-to-serving-mlir MLIR tool;
// default values correspond to Qwen-0.5B architecture constants.
//
// This struct intentionally contains only architecture metadata — it does NOT
// hold trained weights, tokenizer state, or HuggingFace checkpoint paths.
struct LLMModelSpec {
    std::string model_name          = "qwen_0_5b";
    int64_t num_layers              = 24;
    int64_t hidden_size             = 1024;
    int64_t num_attention_heads     = 16;
    int64_t num_key_value_heads     = 16;   // <num_attention_heads for GQA
    int64_t intermediate_size       = 2816;
    int64_t vocab_size              = 151936;
    int64_t max_seq_len             = 512;
    int64_t batch_size              = 1;
    std::string truth_boundary =
        "llm_graph_serving_ir_from_model_config_not_hf_weights";
};

// Serving mode for LLM graph construction.
//   Prefill — processes the full prompt; writes new K/V entries to cache
//             via KVCacheWrite.
//   Decode  — generates one token; reads existing K/V from cache
//             via KVCacheRead. Adds a persistent kv_cache_in tensor.
enum class LLMGraphMode {
    Prefill,
    Decode,
};

// Builds a serving-aware Graph IR for a single transformer layer.
//
// IMPORTANT: This builder does NOT:
//   - import HuggingFace weights or checkpoints
//   - lower a full multi-layer transformer graph (all num_layers layers)
//   - perform tokenization or produce executable inference code
//
// It produces a metadata-only compiler Graph IR skeleton for one transformer
// layer, using architecture constants from LLMModelSpec. The graph is
// structurally equivalent to the MLIR emitted by qwen-to-serving-mlir, but
// represented in the C++ Graph IR.
//
// All tensors are metadata-only (shape + name, no allocated data).
// LLM weight tensors are far too large to allocate at compiler frontend time
// (embedding table alone is ~620 MB at Qwen-0.5B scale).
//
// Op sequence (single prefill layer, 19 tensors, 8 nodes):
//   Embedding → RMSNorm → QKVProjection → KVCacheWrite
//     → CausalAttention → RMSNorm → MLP → Linear (LM head)
//
// Op sequence (single decode layer, 20 tensors, 8 nodes):
//   Embedding → RMSNorm → QKVProjection → KVCacheRead
//     → CausalAttention → RMSNorm → MLP → Linear (LM head)
class LLMGraphBuilder : public GraphBuilder {
public:
    explicit LLMGraphBuilder(LLMModelSpec spec = {});

    // Builds the prefill graph (default mode).
    Graph build() override;

    // Builds the graph for the requested serving mode.
    Graph build(LLMGraphMode mode);

private:
    LLMModelSpec spec_;
};
