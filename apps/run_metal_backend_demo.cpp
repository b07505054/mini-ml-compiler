#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"
#include "runtime/executor.h"
#include "runtime/lowering.h"
#include "runtime/backend_type.h"

#include <iostream>

int main() {
    Graph graph;

    int input = graph.add_tensor(
        Tensor("input", {2, 2})
    );

    int output = graph.add_tensor(
        Tensor("output", {})
    );

    graph.get_tensor(input).data = {
        1.0f, -2.0f,
        3.0f, -4.0f
    };

    graph.add_node(
        Node(
            "relu",
            OpType::ReLU,
            {input},
            {output}
        )
    );

    ExecutionPlan plan =
        lower_to_execution_plan(graph);

    Executor executor;

    executor.run(
        graph,
        plan,
        true,
        true,
        BackendType::Metal
    );

    std::cout << "Metal backend demo complete.\n";

    return 0;
}