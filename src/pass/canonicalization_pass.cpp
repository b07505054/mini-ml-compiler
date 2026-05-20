#include "pass/canonicalization_pass.h"

#include <iostream>

void CanonicalizationPass::run(Graph& graph) {
    std::cout
        << "[CanonicalizationPass] "
        << "Searching canonicalization opportunities\n";

    for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
        const auto& n0 = graph.nodes[i];
        const auto& n1 = graph.nodes[i + 1];

        if (
            n0.op == OpType::ReLU &&
            n1.op == OpType::ReLU
        ) {
            std::cout
                << "  candidate: ReLU(ReLU(x)) -> ReLU(x) around nodes "
                << n0.name
                << " and "
                << n1.name
                << "\n";
        }
    }

    for (const auto& node : graph.nodes) {
        if (node.op == OpType::Add) {
            std::cout
                << "  candidate: Add(x, 0) -> x if one input is constant zero at node "
                << node.name
                << "\n";
        }

        if (node.op == OpType::MatMul) {
            std::cout
                << "  candidate: MatMul(x, I) -> x if RHS is identity at node "
                << node.name
                << "\n";
        }
    }
}
