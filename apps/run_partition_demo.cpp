#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "runtime/lowering.h"
#include "runtime/graph_partitioner.h"
#include "runtime/backend_utils.h"

#include <iostream>

int main() {
    Graph graph;

    int input = graph.add_tensor(Tensor("input", {2, 3}));
    int weight = graph.add_tensor(Tensor("weight", {3, 2}));
    int bias = graph.add_tensor(Tensor("bias", {2, 2}));
    int matmul_out = graph.add_tensor(Tensor("matmul_out", {}));
    int add_out = graph.add_tensor(Tensor("add_out", {}));
    int output = graph.add_tensor(Tensor("output", {}));

    graph.get_tensor(input).persistent = true;
    graph.get_tensor(weight).persistent = true;
    graph.get_tensor(bias).persistent = true;

    graph.get_tensor(input).data = {
        1, 2, 3,
        4, 5, 6
    };

    graph.get_tensor(weight).data = {
        1, 2,
        3, 4,
        5, 6
    };

    graph.get_tensor(bias).data = {
        1, 1,
        1, 1
    };

    graph.add_node(Node("matmul", OpType::MatMul, {input, weight}, {matmul_out}));
    graph.add_node(Node("add", OpType::Add, {matmul_out, bias}, {add_out}));
    graph.add_node(Node("relu", OpType::ReLU, {add_out}, {output}));

    ShapeInference infer;
    infer.run(graph);

    MemoryPlanner memory_planner;
    memory_planner.plan(graph);

    GraphVerifier verifier;
    if (!verifier.verify(graph)) {
        return 1;
    }

    ExecutionPlan plan = lower_to_execution_plan(graph);

    GraphPartitioner partitioner;
    auto partitions = partitioner.partition(plan);

    std::cout << "\n=== Graph Partitions ===\n";

    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition " << i
                  << " | Backend: "
                  << backend_name(partitions[i].backend)
                  << " | Nodes: ";

        for (const auto& node : partitions[i].nodes) {
            std::cout << node.name << " ";
        }

        std::cout << "\n";
    }

    return 0;
}