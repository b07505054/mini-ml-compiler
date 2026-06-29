#pragma once

#include <string>
#include <vector>

// Backend target for a node in the graph IR.
// Unknown means no placement decision has been recorded yet; downstream
// passes (GraphLowerer) fall back to their own op-type classification.
enum class BackendKind {
    Unknown,
    CPU,
    Metal,
};

enum class OpType {
    Input,
    MatMul,
    Add,
    ReLU,
    FusedMatMulBias,
    FusedMatMulAddReLU,
    Attention,
    CausalAttention,
    LayerNorm,
    FusedAttention,
    TiledAttention,
    Conv2D, 
    BatchNorm,
    MaxPool,
    Flatten,
    Linear,
    FusedConvBatchNormReLU,

    // LLM serving ops — added for LLMGraphBuilder
    Embedding,       // vocabulary embedding lookup
    RMSNorm,         // RMS layer normalization
    QKVProjection,   // fused Q/K/V linear projection
    KVCacheWrite,    // write K/V to cache (prefill path)
    KVCacheRead,     // read K/V from cache (decode path)
    MLP,             // feed-forward / MLP block
};

struct Node {
    std::string name;
    OpType op;
    std::vector<int> inputs;
    std::vector<int> outputs;
    BackendKind assigned_backend = BackendKind::Unknown;

    Node() = default;

    Node(std::string name, OpType op, std::vector<int> inputs, std::vector<int> outputs)
        : name(std::move(name)),
          op(op),
          inputs(std::move(inputs)),
          outputs(std::move(outputs)) {}
};