#include "ir/graph.h"
#include "runtime/executor.h"
#include "runtime/lowering.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"

#include <iostream>

int main() {
    Graph graph;

    int input = graph.add_tensor(Tensor("input", {2, 4}));
    int output = graph.add_tensor(Tensor("output", {}));

    graph.get_tensor(input).persistent = true;

    graph.get_tensor(input).data = {
        1, 2, 3, 4,
        10, 20, 30, 40
    };

    graph.add_node(
        Node(
            "layernorm",
            OpType::LayerNorm,
            {input},
            {output}
        )
    );

    ShapeInference infer;
    infer.run(graph);

    GraphVerifier verifier;
    verifier.verify(graph);

    auto plan = lower_to_execution_plan(graph);

    Executor executor;
    executor.run(graph, plan);

    std::cout << "\nLayerNorm Output:\n";

    for (float x : graph.get_tensor(output).data) {
        std::cout << x << " ";
    }

    std::cout << "\n";

    return 0;
}