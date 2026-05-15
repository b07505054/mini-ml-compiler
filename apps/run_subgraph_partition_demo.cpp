#include "ir/graph.h"
#include "runtime/graph_partitioner.h"
#include "runtime/backend_utils.h"

#include <iostream>

int main() {
    Graph graph;

    int input1 = graph.add_tensor(Tensor("input1", {2, 3}));
    int input2 = graph.add_tensor(Tensor("input2", {2, 3}));
    int weight = graph.add_tensor(Tensor("weight", {3, 2}));

    int matmul1_out = graph.add_tensor(Tensor("matmul1_out", {}));
    int matmul2_out = graph.add_tensor(Tensor("matmul2_out", {}));
    int add_out = graph.add_tensor(Tensor("add_out", {}));
    int output = graph.add_tensor(Tensor("output", {}));

    graph.add_node(
        Node(
            "matmul1",
            OpType::MatMul,
            {input1, weight},
            {matmul1_out}
        )
    );

    graph.add_node(
        Node(
            "matmul2",
            OpType::MatMul,
            {input2, weight},
            {matmul2_out}
        )
    );

    graph.add_node(
        Node(
            "add",
            OpType::Add,
            {matmul1_out, matmul2_out},
            {add_out}
        )
    );

    graph.add_node(
        Node(
            "relu",
            OpType::ReLU,
            {add_out},
            {output}
        )
    );

    GraphPartitioner partitioner;

    auto subgraphs = partitioner.partition(graph);

    std::cout << "=== Subgraph Delegation Report ===\n\n";

    for (size_t i = 0; i < subgraphs.size(); ++i) {
        const auto& sg = subgraphs[i];

        std::cout << "Subgraph " << i
                  << " -> "
                  << backend_name(sg.backend)
                  << "\n";

        for (const auto& node : sg.nodes) {
            std::cout << "  "
                      << node.name
                      << "\n";
        }

        std::cout << "\n";
    }

    return 0;
}