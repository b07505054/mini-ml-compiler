#include "pass/algebraic_simplification_pass.h"

#include <iostream>

void AlgebraicSimplificationPass::run(Graph& graph) {
    int simplified = 0;

    for (auto& node : graph.nodes) {
        if (node.op != OpType::ReLU || node.inputs.empty()) {
            continue;
        }

        int input_tensor = node.inputs[0];

        for (const auto& producer : graph.nodes) {
            if (producer.op != OpType::ReLU || producer.outputs.empty()) {
                continue;
            }

            if (producer.outputs[0] == input_tensor) {
                // ReLU(ReLU(x)) -> ReLU(x)
                node.inputs[0] = producer.inputs[0];
                simplified++;
                break;
            }
        }
    }

    std::cout << "[AlgebraicSimplificationPass] Simplified "
              << simplified
              << " patterns\n";
}