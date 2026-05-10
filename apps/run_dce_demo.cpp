#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "pass/pass_manager.h"
#include "pass/dead_node_elimination_pass.h"

#include <iostream>
#include <memory>

int main() {
    Graph graph;

    int input = graph.add_tensor(Tensor("input", {2, 2}));
    int weight = graph.add_tensor(Tensor("weight", {2, 2}));
    int bias = graph.add_tensor(Tensor("bias", {2, 2}));

    int live_matmul = graph.add_tensor(Tensor("live_matmul", {}));
    int output = graph.add_tensor(Tensor("output", {}));

    int dead_matmul = graph.add_tensor(Tensor("dead_matmul", {}));
    int dead_output = graph.add_tensor(Tensor("dead_output", {}));

    graph.get_tensor(input).persistent = true;
    graph.get_tensor(weight).persistent = true;
    graph.get_tensor(bias).persistent = true;

    graph.add_node(Node("live_matmul", OpType::MatMul, {input, weight}, {live_matmul}));
    graph.add_node(Node("live_add", OpType::Add, {live_matmul, bias}, {output}));

    // Dead branch: not connected to final graph output.
    graph.add_node(Node("dead_matmul", OpType::MatMul, {input, weight}, {dead_matmul}));
    graph.add_node(Node("dead_relu", OpType::ReLU, {dead_matmul}, {dead_output}));

    std::cout << "Before DCE:\n";
    graph.dump();

    PassManager pm;
    pm.add_pass(std::make_unique<DeadNodeEliminationPass>());
    pm.run(graph);

    std::cout << "\nAfter DCE:\n";
    graph.dump();

    return 0;
}