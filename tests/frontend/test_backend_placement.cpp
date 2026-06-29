#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"
#include "pass/compiler_pipeline_passes.h"
#include "runtime/graph_lowerer.h"

#include <iostream>
#include <string>

// ---- helpers -------------------------------------------------------

static void check(bool cond, const char* label) {
    if (!cond) {
        std::cerr << "FAIL: " << label << "\n";
        std::exit(1);
    }
    std::cout << "  pass: " << label << "\n";
}

// Build a minimal one-node graph with the given OpType.
// Two tensors are added so GraphLowerer can resolve memory_offset from
// node.outputs[0] without an out-of-bounds access.
static Graph make_graph(OpType op) {
    Graph g;
    Tensor t0;  t0.name = "in";  t0.shape = {4, 4};
    Tensor t1;  t1.name = "out"; t1.shape = {4, 4};
    int in_id  = g.add_tensor(t0);
    int out_id = g.add_tensor(t1);
    g.add_node(Node("node", op, {in_id}, {out_id}));
    return g;
}

// Run BackendPlacementPass on a single-node graph and return the
// assigned_backend on that node.
static BackendKind placement_for(OpType op) {
    Graph g = make_graph(op);
    BackendPlacementPass pass;
    pass.run(g);
    return g.nodes[0].assigned_backend;
}

// Lower a graph with no prior pass and return the backend string on the
// first lowered op.
static std::string lowered_backend_for(OpType op) {
    Graph g = make_graph(op);
    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);
    return lowered.ops[0].backend;
}

// ---- BackendPlacementPass tests ------------------------------------

static void test_placement_matmul_is_metal() {
    check(placement_for(OpType::MatMul) == BackendKind::Metal,
          "BackendPlacementPass: MatMul -> Metal");
}

static void test_placement_conv2d_is_metal() {
    check(placement_for(OpType::Conv2D) == BackendKind::Metal,
          "BackendPlacementPass: Conv2D -> Metal");
}

static void test_placement_fused_conv_bn_relu_is_metal() {
    check(placement_for(OpType::FusedConvBatchNormReLU) == BackendKind::Metal,
          "BackendPlacementPass: FusedConvBatchNormReLU -> Metal");
}

static void test_placement_qkv_projection_is_metal() {
    check(placement_for(OpType::QKVProjection) == BackendKind::Metal,
          "BackendPlacementPass: QKVProjection -> Metal");
}

static void test_placement_mlp_is_metal() {
    check(placement_for(OpType::MLP) == BackendKind::Metal,
          "BackendPlacementPass: MLP -> Metal");
}

static void test_placement_linear_is_metal() {
    check(placement_for(OpType::Linear) == BackendKind::Metal,
          "BackendPlacementPass: Linear -> Metal");
}

static void test_placement_fused_matmul_bias_is_metal() {
    check(placement_for(OpType::FusedMatMulBias) == BackendKind::Metal,
          "BackendPlacementPass: FusedMatMulBias -> Metal");
}

static void test_placement_fused_attention_is_metal() {
    check(placement_for(OpType::FusedAttention) == BackendKind::Metal,
          "BackendPlacementPass: FusedAttention -> Metal");
}

static void test_placement_tiled_attention_is_metal() {
    check(placement_for(OpType::TiledAttention) == BackendKind::Metal,
          "BackendPlacementPass: TiledAttention -> Metal");
}

static void test_placement_embedding_is_cpu() {
    check(placement_for(OpType::Embedding) == BackendKind::CPU,
          "BackendPlacementPass: Embedding -> CPU");
}

static void test_placement_rmsnorm_is_cpu() {
    check(placement_for(OpType::RMSNorm) == BackendKind::CPU,
          "BackendPlacementPass: RMSNorm -> CPU");
}

static void test_placement_kv_cache_write_is_cpu() {
    check(placement_for(OpType::KVCacheWrite) == BackendKind::CPU,
          "BackendPlacementPass: KVCacheWrite -> CPU");
}

static void test_placement_kv_cache_read_is_cpu() {
    check(placement_for(OpType::KVCacheRead) == BackendKind::CPU,
          "BackendPlacementPass: KVCacheRead -> CPU");
}

static void test_placement_causal_attention_is_cpu() {
    check(placement_for(OpType::CausalAttention) == BackendKind::CPU,
          "BackendPlacementPass: CausalAttention -> CPU");
}

static void test_placement_relu_is_cpu() {
    check(placement_for(OpType::ReLU) == BackendKind::CPU,
          "BackendPlacementPass: ReLU -> CPU");
}

static void test_placement_add_is_cpu() {
    check(placement_for(OpType::Add) == BackendKind::CPU,
          "BackendPlacementPass: Add -> CPU");
}

static void test_placement_result_not_unknown() {
    // After the pass runs, no node should remain Unknown.
    for (OpType op : {
            OpType::MatMul, OpType::Add, OpType::ReLU,
            OpType::Embedding, OpType::RMSNorm, OpType::MLP,
            OpType::KVCacheWrite, OpType::KVCacheRead
    }) {
        check(placement_for(op) != BackendKind::Unknown,
              "BackendPlacementPass: assigned_backend is never Unknown after run");
    }
}

// ---- GraphLowerer tests -------------------------------------------

static void test_lowerer_fallback_matmul_is_metal() {
    // No pass run; assigned_backend is Unknown; lowerer uses backend_for_op.
    check(lowered_backend_for(OpType::MatMul) == "Metal",
          "GraphLowerer fallback: MatMul -> Metal");
}

static void test_lowerer_fallback_embedding_is_cpu() {
    check(lowered_backend_for(OpType::Embedding) == "CPU",
          "GraphLowerer fallback: Embedding -> CPU");
}

static void test_lowerer_respects_metal_override() {
    // Manually set a CPU op to Metal and verify lowerer honours it.
    Graph g = make_graph(OpType::Embedding);
    g.nodes[0].assigned_backend = BackendKind::Metal;
    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);
    check(lowered.ops[0].backend == "Metal",
          "GraphLowerer: respects assigned_backend Metal (overrides fallback CPU)");
}

static void test_lowerer_respects_cpu_override() {
    // Manually force a Metal op to CPU.
    Graph g = make_graph(OpType::MatMul);
    g.nodes[0].assigned_backend = BackendKind::CPU;
    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);
    check(lowered.ops[0].backend == "CPU",
          "GraphLowerer: respects assigned_backend CPU (overrides fallback Metal)");
}

static void test_lowerer_uses_pass_decision() {
    // Run BackendPlacementPass then lower; final backend must match the pass.
    Graph g = make_graph(OpType::QKVProjection);
    BackendPlacementPass pass;
    pass.run(g);
    // Pass should have set Metal for QKVProjection.
    check(g.nodes[0].assigned_backend == BackendKind::Metal,
          "GraphLowerer+pass: QKVProjection assigned Metal by pass");

    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);
    check(lowered.ops[0].backend == "Metal",
          "GraphLowerer+pass: lowered backend matches pass decision (Metal)");
}

static void test_lowerer_pass_cpu_decision_propagates() {
    Graph g = make_graph(OpType::RMSNorm);
    BackendPlacementPass pass;
    pass.run(g);
    check(g.nodes[0].assigned_backend == BackendKind::CPU,
          "GraphLowerer+pass: RMSNorm assigned CPU by pass");

    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(g);
    check(lowered.ops[0].backend == "CPU",
          "GraphLowerer+pass: lowered backend matches pass decision (CPU)");
}

// ---- main ----------------------------------------------------------

int main() {
    std::cout << "=== BackendPlacement tests ===\n";

    // BackendPlacementPass — Metal ops
    test_placement_matmul_is_metal();
    test_placement_conv2d_is_metal();
    test_placement_fused_conv_bn_relu_is_metal();
    test_placement_qkv_projection_is_metal();
    test_placement_mlp_is_metal();
    test_placement_linear_is_metal();
    test_placement_fused_matmul_bias_is_metal();
    test_placement_fused_attention_is_metal();
    test_placement_tiled_attention_is_metal();

    // BackendPlacementPass — CPU ops
    test_placement_embedding_is_cpu();
    test_placement_rmsnorm_is_cpu();
    test_placement_kv_cache_write_is_cpu();
    test_placement_kv_cache_read_is_cpu();
    test_placement_causal_attention_is_cpu();
    test_placement_relu_is_cpu();
    test_placement_add_is_cpu();

    // No Unknown remaining after pass
    test_placement_result_not_unknown();

    // GraphLowerer fallback (Unknown → backend_for_op)
    test_lowerer_fallback_matmul_is_metal();
    test_lowerer_fallback_embedding_is_cpu();

    // GraphLowerer override tests
    test_lowerer_respects_metal_override();
    test_lowerer_respects_cpu_override();

    // End-to-end: pass writes → lowerer reads
    test_lowerer_uses_pass_decision();
    test_lowerer_pass_cpu_decision_propagates();

    std::cout << "\nAll BackendPlacement tests passed.\n";
    return 0;
}
