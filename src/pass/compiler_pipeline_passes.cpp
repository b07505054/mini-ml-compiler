#include "pass/compiler_pipeline_passes.h"

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
        const auto& a = graph.nodes[i];
        const auto& b = graph.nodes[i + 1];
        const auto& c = graph.nodes[i + 2];

        if (a.op == OpType::MatMul &&
            b.op == OpType::Add &&
            c.op == OpType::ReLU) {
            std::cout << "  candidate: "
                      << a.name
                      << " + "
                      << b.name
                      << " + "
                      << c.name
                      << "\n";
        }
    }
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

    for (const auto& node : graph.nodes) {
        if (node.op == OpType::MatMul) {
            std::cout << "  "
                      << node.name
                      << " -> MockGPU/Metal candidate\n";
        } else {
            std::cout << "  "
                      << node.name
                      << " -> CPU fallback\n";
        }
    }
}

const char* SchedulingPass::name() const {
    return "SchedulingPass";
}

void SchedulingPass::run(Graph& graph) {
    std::cout << "[SchedulingPass] Building static topological execution schedule\n";

    int order = 0;

    for (const auto& node : graph.nodes) {
        std::cout << "  ["
                  << order++
                  << "] "
                  << node.name
                  << "\n";
    }
}