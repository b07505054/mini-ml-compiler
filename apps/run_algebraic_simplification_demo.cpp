#include "ir/graph.h"
#include "pass/pass_manager.h"
#include "pass/algebraic_simplification_pass.h"
#include "pass/dead_node_elimination_pass.h"

#include <iostream>
#include <memory>

int main() {
    Graph graph;

    int input = graph.add_tensor(Tensor("input", {2, 2}));
    int relu1_out = graph.add_tensor(Tensor("relu1_out", {}));
    int output = graph.add_tensor(Tensor("output", {}));

    graph.get_tensor(input).persistent = true;

    graph.add_node(Node("relu1", OpType::ReLU, {input}, {relu1_out}));
    graph.add_node(Node("relu2", OpType::ReLU, {relu1_out}, {output}));

    std::cout << "Before Algebraic Simplification:\n";
    graph.dump();

    PassManager pm;
    pm.add_pass(std::make_unique<AlgebraicSimplificationPass>());
    pm.add_pass(std::make_unique<DeadNodeEliminationPass>());
    pm.run(graph);

    std::cout << "\nAfter Algebraic Simplification + DCE:\n";
    graph.dump();

    return 0;
}