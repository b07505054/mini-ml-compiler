#include "pass/fusion_pass.h"

#include <iostream>

void FusionPass::run(Graph& graph) {
    std::vector<Node> optimized;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        if (i + 2 < graph.nodes.size() &&
            graph.nodes[i].op == OpType::MatMul &&
            graph.nodes[i + 1].op == OpType::Add &&
            graph.nodes[i + 2].op == OpType::ReLU) {

            Node fused(
                "fused_matmul_add_relu",
                OpType::FusedMatMulAddReLU,
                {
                    graph.nodes[i].inputs[0],      // input
                    graph.nodes[i].inputs[1],      // weight
                    graph.nodes[i + 1].inputs[1]   // bias
                },
                graph.nodes[i + 2].outputs         // final output
            );

            optimized.push_back(fused);
            i += 2;

            std::cout << "[FusionPass] Applied MatMul + Add + ReLU fusion\n";
        } else {
            optimized.push_back(graph.nodes[i]);
        }
    }

    graph.nodes = std::move(optimized);
}