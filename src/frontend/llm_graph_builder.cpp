#include "frontend/llm_graph_builder.h"

#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"

// Build a metadata-only Tensor: shape and name are set but data is never
// allocated. LLM weight tensors are far too large to allocate at compiler
// frontend time (e.g., the Qwen-0.5B embedding table is ~620 MB).
// All shape information is available for downstream compiler passes.
static Tensor make_meta(const std::string& name,
                        std::vector<int> shape,
                        bool persistent = false) {
    Tensor t;           // default ctor: data stays empty
    t.name       = name;
    t.shape      = std::move(shape);
    t.dtype      = DType::Float32;
    t.persistent = persistent;
    return t;
}

LLMGraphBuilder::LLMGraphBuilder(LLMModelSpec spec) : spec_(std::move(spec)) {}

Graph LLMGraphBuilder::build() {
    return build(LLMGraphMode::Prefill);
}

Graph LLMGraphBuilder::build(LLMGraphMode mode) {
    Graph graph;

    // Derived dimensions
    const int bs       = static_cast<int>(spec_.batch_size);
    const int seq      = static_cast<int>(spec_.max_seq_len);
    const int hs       = static_cast<int>(spec_.hidden_size);
    const int n_q      = static_cast<int>(spec_.num_attention_heads);
    const int n_kv     = static_cast<int>(spec_.num_key_value_heads);
    const int head_dim = (n_q > 0) ? hs / n_q : hs;
    const int kv_dim   = n_kv * head_dim;   // may differ from hs under GQA
    const int qkv_dim  = hs + 2 * kv_dim;  // fused Q+K+V projection width
    const int inter    = static_cast<int>(spec_.intermediate_size);
    const int vocab    = static_cast<int>(spec_.vocab_size);

    // ---- Tensors -------------------------------------------------
    // All tensors are metadata-only (no data allocation).

    int tokens = graph.add_tensor(
        make_meta("tokens", {bs, seq}));

    int embedding_weight = graph.add_tensor(
        make_meta("embedding_weight", {vocab, hs}, /*persistent=*/true));

    int embed_out = graph.add_tensor(
        make_meta("embed_out", {bs, seq, hs}));

    int rmsnorm1_weight = graph.add_tensor(
        make_meta("rmsnorm1_weight", {hs}, /*persistent=*/true));

    int norm1_out = graph.add_tensor(
        make_meta("norm1_out", {bs, seq, hs}));

    // Fused QKV projection weight: [Q_dim + K_dim + V_dim, hidden_size]
    int qkv_weight = graph.add_tensor(
        make_meta("qkv_weight", {qkv_dim, hs}, /*persistent=*/true));

    // Separate Q, K, V activation tensors (three outputs of QKVProjection)
    int q_out = graph.add_tensor(
        make_meta("q_out", {bs, seq, hs}));

    int k_out = graph.add_tensor(
        make_meta("k_out", {bs, seq, kv_dim}));

    int v_out = graph.add_tensor(
        make_meta("v_out", {bs, seq, kv_dim}));

    // Mode-specific KV cache tensors.
    // Prefill writes new K/V to cache; decode reads existing K/V from cache.
    int kv_cache_in  = -1;
    int kv_cache_out = -1;

    if (mode == LLMGraphMode::Decode) {
        // Persistent tensor representing the existing KV cache state.
        kv_cache_in = graph.add_tensor(
            make_meta("kv_cache_in", {bs, seq, kv_dim}, /*persistent=*/true));
    }

    kv_cache_out = graph.add_tensor(
        make_meta("kv_cache_out", {bs, seq, kv_dim}));

    int attn_out = graph.add_tensor(
        make_meta("attn_out", {bs, seq, hs}));

    int rmsnorm2_weight = graph.add_tensor(
        make_meta("rmsnorm2_weight", {hs}, /*persistent=*/true));

    int norm2_out = graph.add_tensor(
        make_meta("norm2_out", {bs, seq, hs}));

    // MLP weight tensors (SwiGLU / gated MLP)
    int gate_weight = graph.add_tensor(
        make_meta("gate_weight", {inter, hs}, /*persistent=*/true));

    int up_weight = graph.add_tensor(
        make_meta("up_weight", {inter, hs}, /*persistent=*/true));

    int down_weight = graph.add_tensor(
        make_meta("down_weight", {hs, inter}, /*persistent=*/true));

    int mlp_out = graph.add_tensor(
        make_meta("mlp_out", {bs, seq, hs}));

    int lm_head_weight = graph.add_tensor(
        make_meta("lm_head_weight", {vocab, hs}, /*persistent=*/true));

    int logits = graph.add_tensor(
        make_meta("logits", {bs, vocab}));

    // ---- Nodes ---------------------------------------------------

    graph.add_node(Node(
        "embedding", OpType::Embedding,
        {tokens, embedding_weight},
        {embed_out}));

    graph.add_node(Node(
        "rmsnorm_1", OpType::RMSNorm,
        {embed_out, rmsnorm1_weight},
        {norm1_out}));

    // QKVProjection produces three activation tensors: Q, K, V
    graph.add_node(Node(
        "qkv_projection", OpType::QKVProjection,
        {norm1_out, qkv_weight},
        {q_out, k_out, v_out}));

    // Prefill: write K and V to the KV cache.
    // Decode:  read existing K/V from the KV cache.
    if (mode == LLMGraphMode::Prefill) {
        graph.add_node(Node(
            "kv_cache_write", OpType::KVCacheWrite,
            {k_out, v_out},
            {kv_cache_out}));
    } else {
        graph.add_node(Node(
            "kv_cache_read", OpType::KVCacheRead,
            {kv_cache_in},
            {kv_cache_out}));
    }

    graph.add_node(Node(
        "causal_attention", OpType::CausalAttention,
        {q_out, kv_cache_out},
        {attn_out}));

    graph.add_node(Node(
        "rmsnorm_2", OpType::RMSNorm,
        {attn_out, rmsnorm2_weight},
        {norm2_out}));

    // MLP block: gate, up, and down projections as a single fused node
    graph.add_node(Node(
        "mlp", OpType::MLP,
        {norm2_out, gate_weight, up_weight, down_weight},
        {mlp_out}));

    // LM head: final linear projection to vocabulary
    graph.add_node(Node(
        "lm_head", OpType::Linear,
        {mlp_out, lm_head_weight},
        {logits}));

    return graph;
}
