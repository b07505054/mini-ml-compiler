#include "frontend/llm_graph_builder.h"
#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"
#include "runtime/graph_lowerer.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// ---- helpers -------------------------------------------------------

static void check(bool cond, const char* label) {
    if (!cond) {
        std::cerr << "FAIL: " << label << "\n";
        std::exit(1);
    }
    std::cout << "  pass: " << label << "\n";
}

// Small spec that avoids any risk of large allocation in tests.
static LLMModelSpec small_spec() {
    LLMModelSpec s;
    s.model_name          = "test_llm";
    s.num_layers          = 1;
    s.hidden_size         = 64;
    s.num_attention_heads = 4;
    s.num_key_value_heads = 4;
    s.intermediate_size   = 128;
    s.vocab_size          = 100;
    s.max_seq_len         = 8;
    s.batch_size          = 1;
    return s;
}

static bool has_node_with_op(const Graph& g, OpType op) {
    for (const auto& n : g.nodes) {
        if (n.op == op) return true;
    }
    return false;
}

static bool has_tensor_named(const Graph& g, const std::string& name) {
    for (const auto& t : g.tensors) {
        if (t.name == name) return true;
    }
    return false;
}

// ---- tests ---------------------------------------------------------

static void test_builds_without_throwing() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(!g.tensors.empty(), "build() produces non-empty tensors");
    check(!g.nodes.empty(),   "build() produces non-empty nodes");
}

static void test_tensor_count() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    // Expected: tokens, embed_weight, embed_out, rmsnorm1_w, norm1_out,
    //           qkv_weight, q_out, k_out, v_out, kv_cache_out, attn_out,
    //           rmsnorm2_w, norm2_out, gate_w, up_w, down_w, mlp_out,
    //           lm_head_weight, logits  = 19
    check(g.tensors.size() == 19, "tensor count == 19");
}

static void test_node_count() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    // embedding, rmsnorm_1, qkv_projection, kv_cache_write,
    // causal_attention, rmsnorm_2, mlp, lm_head  = 8
    check(g.nodes.size() == 8, "node count == 8");
}

static void test_contains_embedding() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(has_node_with_op(g, OpType::Embedding), "graph contains Embedding");
}

static void test_contains_rmsnorm() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(has_node_with_op(g, OpType::RMSNorm), "graph contains RMSNorm");
}

static void test_contains_qkv_projection() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(has_node_with_op(g, OpType::QKVProjection), "graph contains QKVProjection");
}

static void test_contains_kv_cache_write() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(has_node_with_op(g, OpType::KVCacheWrite), "graph contains KVCacheWrite");
}

static void test_contains_causal_attention() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(has_node_with_op(g, OpType::CausalAttention), "graph contains CausalAttention");
}

static void test_contains_mlp() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(has_node_with_op(g, OpType::MLP), "graph contains MLP");
}

static void test_lm_head_is_linear() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    // lm_head is the last node and uses OpType::Linear
    check(g.nodes.back().op   == OpType::Linear, "last node op is Linear");
    check(g.nodes.back().name == "lm_head",      "last node name is lm_head");
}

static void test_has_logits_tensor() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    check(has_tensor_named(g, "logits"), "graph has 'logits' tensor");
}

static void test_logits_shape() {
    LLMModelSpec s = small_spec();
    LLMGraphBuilder b(s);
    Graph g = b.build();
    // logits shape: {batch_size, vocab_size}
    const Tensor* logits = nullptr;
    for (const auto& t : g.tensors) {
        if (t.name == "logits") { logits = &t; break; }
    }
    check(logits != nullptr, "logits tensor found");
    check(logits->shape.size() == 2, "logits is 2-D");
    check(logits->shape[0] == 1,                       "logits batch dim == 1");
    check(logits->shape[1] == static_cast<int>(s.vocab_size), "logits vocab dim correct");
}

static void test_tokens_shape() {
    LLMModelSpec s = small_spec();
    LLMGraphBuilder b(s);
    Graph g = b.build();
    check(!g.tensors.empty(), "tensors non-empty");
    const auto& tok = g.tensors[0];
    check(tok.name == "tokens", "first tensor is tokens");
    check(tok.shape.size() == 2, "tokens is 2-D");
    check(tok.shape[0] == 1,                        "tokens batch dim == 1");
    check(tok.shape[1] == static_cast<int>(s.max_seq_len), "tokens seq dim correct");
}

static void test_all_tensors_are_metadata_only() {
    // LLMGraphBuilder must never allocate tensor data — all data vectors
    // must be empty. This is intentional: LLM weight tensors are too large
    // to materialise at compiler frontend time.
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    for (const auto& t : g.tensors) {
        check(t.data.empty(),
              ("tensor '" + t.name + "' has no allocated data (metadata-only)").c_str());
    }
}

static void test_qkv_projection_has_three_outputs() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    for (const auto& n : g.nodes) {
        if (n.op == OpType::QKVProjection) {
            check(n.outputs.size() == 3,
                  "qkv_projection node has 3 output tensors (q, k, v)");
            return;
        }
    }
    check(false, "qkv_projection node must exist");
}

static void test_weight_tensors_are_persistent() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    for (const auto& t : g.tensors) {
        if (t.name == "embedding_weight" ||
            t.name == "lm_head_weight"   ||
            t.name == "qkv_weight"       ||
            t.name == "gate_weight"      ||
            t.name == "up_weight"        ||
            t.name == "down_weight") {
            check(t.persistent,
                  ("weight tensor '" + t.name + "' is persistent").c_str());
        }
    }
    // activation tensors are not persistent
    for (const auto& t : g.tensors) {
        if (t.name == "embed_out" || t.name == "attn_out" || t.name == "mlp_out") {
            check(!t.persistent,
                  ("activation tensor '" + t.name + "' is not persistent").c_str());
        }
    }
}

static void test_two_rmsnorm_nodes() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build();
    int count = 0;
    for (const auto& n : g.nodes) {
        if (n.op == OpType::RMSNorm) ++count;
    }
    check(count == 2, "graph contains exactly 2 RMSNorm nodes");
}

static void test_custom_spec_changes_shapes() {
    LLMModelSpec s = small_spec();
    s.hidden_size         = 128;
    s.num_attention_heads = 8;
    s.num_key_value_heads = 8;
    s.vocab_size          = 200;
    s.max_seq_len         = 16;

    LLMGraphBuilder b(s);
    Graph g = b.build();

    // Still 19 tensors / 8 nodes regardless of spec
    check(g.tensors.size() == 19, "custom spec: tensor count == 19");
    check(g.nodes.size()   == 8,  "custom spec: node count == 8");

    // Check embed_out shape: {1, 16, 128}
    const Tensor* eo = nullptr;
    for (const auto& t : g.tensors) {
        if (t.name == "embed_out") { eo = &t; break; }
    }
    check(eo != nullptr, "embed_out tensor found");
    check(eo->shape.size()==3 && eo->shape[1]==16 && eo->shape[2]==128,
          "embed_out shape {1,16,128}");
}

// ---- prefill / decode mode tests -----------------------------------

static void test_default_build_is_prefill() {
    LLMGraphBuilder b(small_spec());
    Graph g_default  = b.build();
    Graph g_prefill  = b.build(LLMGraphMode::Prefill);
    // Both must produce the same op sequence.
    check(g_default.nodes.size() == g_prefill.nodes.size(),
          "default build() and build(Prefill) have same node count");
    for (size_t i = 0; i < g_default.nodes.size(); ++i) {
        check(g_default.nodes[i].op == g_prefill.nodes[i].op,
              ("node op matches at index " + std::to_string(i)).c_str());
    }
}

static void test_prefill_has_kv_cache_write_not_read() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Prefill);
    check( has_node_with_op(g, OpType::KVCacheWrite),
          "prefill graph contains KVCacheWrite");
    check(!has_node_with_op(g, OpType::KVCacheRead),
          "prefill graph does not contain KVCacheRead");
}

static void test_decode_has_kv_cache_read_not_write() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Decode);
    check( has_node_with_op(g, OpType::KVCacheRead),
          "decode graph contains KVCacheRead");
    check(!has_node_with_op(g, OpType::KVCacheWrite),
          "decode graph does not contain KVCacheWrite");
}

static void test_decode_node_count() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Decode);
    // Same 8 nodes as prefill; only the KV cache op differs.
    check(g.nodes.size() == 8, "decode graph node count == 8");
}

static void test_decode_tensor_count() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Decode);
    // Decode adds kv_cache_in (persistent) → 19 prefill tensors + 1 = 20.
    check(g.tensors.size() == 20, "decode graph tensor count == 20");
}

static void test_decode_has_kv_cache_in_tensor() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Decode);
    check(has_tensor_named(g, "kv_cache_in"),
          "decode graph has kv_cache_in tensor");
    // kv_cache_in must be persistent (represents stored cache state).
    for (const auto& t : g.tensors) {
        if (t.name == "kv_cache_in") {
            check(t.persistent, "kv_cache_in is persistent");
            break;
        }
    }
}

static void test_decode_tensors_are_metadata_only() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Decode);
    for (const auto& t : g.tensors) {
        check(t.data.empty(),
              ("decode tensor '" + t.name + "' has no allocated data").c_str());
    }
}

// ---- backend classification tests ----------------------------------
// Verify that QKVProjection and MLP are classified as "Metal" by
// GraphLowerer::backend_for_op, which is the authoritative classification
// that BackendPlacementPass now mirrors.

static void test_backend_classification_qkv_projection_is_metal() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Prefill);

    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);

    for (const auto& op : lowered.ops) {
        if (op.lowered_op_type == "LoweredQKVProjection") {
            check(op.backend == "Metal",
                  "QKVProjection classified as Metal by GraphLowerer");
            return;
        }
    }
    check(false, "LoweredQKVProjection op must exist in lowered graph");
}

static void test_backend_classification_mlp_is_metal() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Prefill);

    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);

    for (const auto& op : lowered.ops) {
        if (op.lowered_op_type == "LoweredMLP") {
            check(op.backend == "Metal",
                  "MLP classified as Metal by GraphLowerer");
            return;
        }
    }
    check(false, "LoweredMLP op must exist in lowered graph");
}

static void test_backend_classification_embedding_is_cpu() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Prefill);

    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);

    for (const auto& op : lowered.ops) {
        if (op.lowered_op_type == "LoweredEmbedding") {
            check(op.backend == "CPU",
                  "Embedding classified as CPU by GraphLowerer");
            return;
        }
    }
    check(false, "LoweredEmbedding op must exist in lowered graph");
}

static void test_backend_classification_rmsnorm_is_cpu() {
    LLMGraphBuilder b(small_spec());
    Graph g = b.build(LLMGraphMode::Prefill);

    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);

    bool found = false;
    for (const auto& op : lowered.ops) {
        if (op.lowered_op_type == "LoweredRMSNorm") {
            check(op.backend == "CPU",
                  "RMSNorm classified as CPU by GraphLowerer");
            found = true;
            break;
        }
    }
    check(found, "LoweredRMSNorm op must exist in lowered graph");
}

// ---- main ----------------------------------------------------------

int main() {
    std::cout << "=== LLMGraphBuilder tests ===\n";

    test_builds_without_throwing();
    test_tensor_count();
    test_node_count();
    test_contains_embedding();
    test_contains_rmsnorm();
    test_contains_qkv_projection();
    test_contains_kv_cache_write();
    test_contains_causal_attention();
    test_contains_mlp();
    test_lm_head_is_linear();
    test_has_logits_tensor();
    test_logits_shape();
    test_tokens_shape();
    test_all_tensors_are_metadata_only();
    test_qkv_projection_has_three_outputs();
    test_weight_tensors_are_persistent();
    test_two_rmsnorm_nodes();
    test_custom_spec_changes_shapes();

    // Mode tests
    test_default_build_is_prefill();
    test_prefill_has_kv_cache_write_not_read();
    test_decode_has_kv_cache_read_not_write();
    test_decode_node_count();
    test_decode_tensor_count();
    test_decode_has_kv_cache_in_tensor();
    test_decode_tensors_are_metadata_only();

    // Backend classification tests
    test_backend_classification_qkv_projection_is_metal();
    test_backend_classification_mlp_is_metal();
    test_backend_classification_embedding_is_cpu();
    test_backend_classification_rmsnorm_is_cpu();

    std::cout << "\nAll LLMGraphBuilder tests passed.\n";
    return 0;
}
