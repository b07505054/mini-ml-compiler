#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "runtime/lowering.h"
#include "runtime/async_executor.h"

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

    AsyncExecutor executor;
    executor.run(graph, plan, true, true);

    std::cout << "\nAsync Output:\n";

    for (float x : graph.get_tensor(output).data) {
        std::cout << x << " ";
    }

    std::cout << "\n";

    return 0;
}