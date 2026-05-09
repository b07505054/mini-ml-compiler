#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "runtime/lowering.h"
#include "runtime/executor.h"
#include "runtime/backend_type.h"

#include <algorithm>
#include <iostream>

int main() {
    Graph graph;

    int input = graph.add_tensor(Tensor("input", {2, 3}));
    int weight = graph.add_tensor(Tensor("weight", {3, 2}));
    int output = graph.add_tensor(Tensor("output", {}));

    graph.get_tensor(input).persistent = true;
    graph.get_tensor(weight).persistent = true;

    graph.get_tensor(input).data = {
        1, 2, 3,
        4, 5, 6
    };

    graph.get_tensor(weight).data = {
        1, 2,
        3, 4,
        5, 6
    };

    graph.add_node(Node("matmul", OpType::MatMul, {input, weight}, {output}));

    ShapeInference infer;
    infer.run(graph);

    MemoryPlanner memory_planner;
    memory_planner.plan(graph);

    GraphVerifier verifier;
    if (!verifier.verify(graph)) {
        return 1;
    }

    ExecutionPlan plan = lower_to_execution_plan(graph);

    Executor executor;

    std::cout << "\n=== Run on CPU Backend ===\n";
    executor.run(graph, plan, true, true, BackendType::CPU);

    std::cout << "\nCPU Output:\n";
    for (float x : graph.get_tensor(output).data) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    std::fill(
        graph.get_tensor(output).data.begin(),
        graph.get_tensor(output).data.end(),
        0.0f
    );

    std::cout << "\n=== Run on MockGPU Backend ===\n";
    executor.run(graph, plan, true, true, BackendType::MockGPU);

    std::cout << "\nMockGPU Output:\n";
    for (float x : graph.get_tensor(output).data) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    return 0;
}