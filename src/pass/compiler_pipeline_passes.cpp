#include "pass/compiler_pipeline_passes.h"
#include "ir/op_type_utils.h"
#include "analysis/shape_inference.h"
#include "runtime/memory_planner.h"

#include <iostream>

const char* ShapeInferencePass::name() const {
    return "ShapeInferencePass";
}

void ShapeInferencePass::run(Graph& graph) {
    ShapeInference infer;
    infer.run(graph);
}

const char* DTypePropagationPass::name() const {
    return "DTypePropagationPass";
}

void DTypePropagationPass::run(Graph& graph) {
    std::cout << "[DTypePropagationPass] Propagating tensor dtypes: default float32\n";
}

const char* FusionCandidatePass::name() const {
    return "FusionCandidatePass";
}

void FusionCandidatePass::run(Graph& graph) {
    std::cout << "[FusionCandidatePass] Searching fusion candidates\n";
    for (size_t i = 0; i + 2 < graph.nodes.size(); ++i) {
        auto& n0 = graph.nodes[i];
        auto& n1 = graph.nodes[i + 1];
        auto& n2 = graph.nodes[i + 2];

        if (
            n0.op == OpType::Conv2D &&
            n1.op == OpType::BatchNorm &&
            n2.op == OpType::ReLU
        ) {
            std::cout
                << "  rewriting: Conv2D + BatchNorm + ReLU -> FusedConvBatchNormReLU\n";

            n0.op = OpType::FusedConvBatchNormReLU;

            n0.outputs = n2.outputs;

            n1.name = "__fused__";
            n2.name = "__fused__";
        }
    }
    for (size_t i = 0; i + 2 < graph.nodes.size(); ++i) {
        auto& n0 = graph.nodes[i];
        auto& n1 = graph.nodes[i + 1];
        auto& n2 = graph.nodes[i + 2];

        if (
            n0.op == OpType::MatMul &&
            n1.op == OpType::Add &&
            n2.op == OpType::ReLU
        ) {
            std::cout
                << "  rewriting: MatMul + Add + ReLU -> FusedMatMulAddReLU\n";

            n0.op = OpType::FusedMatMulAddReLU;

            n0.outputs = n2.outputs;

            n1.name = "__fused__";
            n2.name = "__fused__";
        }
    }

    for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
        auto& n0 = graph.nodes[i];
        auto& n1 = graph.nodes[i + 1];

        if (
            n0.op == OpType::MatMul &&
            n1.op == OpType::Add
        ) {
            std::cout
                << "  rewriting: MatMul + Add -> FusedMatMulBias\n";

            n0.op = OpType::FusedMatMulBias;

            n0.outputs = n1.outputs;

            n1.name = "__fused__";
        }
    }

    std::vector<Node> rewritten;

    for (const auto& node : graph.nodes) {
        if (node.name != "__fused__") {
            rewritten.push_back(node);
        }
    }

    graph.nodes = rewritten;
}

const char* MemoryPlanningPass::name() const {
    return "MemoryPlanningPass";
}

void MemoryPlanningPass::run(Graph& graph) {
    MemoryPlanner planner;
    planner.plan(graph);
}

const char* BackendPlacementPass::name() const {
    return "BackendPlacementPass";
}

void BackendPlacementPass::run(Graph& graph) {
    std::cout << "[BackendPlacementPass] Assigning backend placement\n";

    for (auto& node : graph.nodes) {
        if (
            node.op == OpType::MatMul                  ||
            node.op == OpType::Linear                  ||
            node.op == OpType::Conv2D                  ||
            node.op == OpType::FusedMatMulBias         ||
            node.op == OpType::FusedMatMulAddReLU      ||
            node.op == OpType::FusedConvBatchNormReLU  ||
            node.op == OpType::FusedAttention          ||
            node.op == OpType::TiledAttention          ||
            node.op == OpType::QKVProjection           ||
            node.op == OpType::MLP
        ) {
            node.assigned_backend = BackendKind::Metal;
            std::cout << "  " << node.name << " -> Metal\n";
        } else {
            node.assigned_backend = BackendKind::CPU;
            std::cout << "  " << node.name << " -> CPU\n";
        }
    }
}

const char* SchedulingPass::name() const {
    return "SchedulingPass";
}

void SchedulingPass::run(Graph& graph) {
    std::cout
        << "[SchedulingPass] "
        << "Building static topological execution schedule\n";

    int order = 0;

    for (const auto& node : graph.nodes) {
        std::cout
            << "  ["
            << order++
            << "] "
            << node.name
            << " | "
            << op_type_to_string(node.op)
            << "\n";
    }
}