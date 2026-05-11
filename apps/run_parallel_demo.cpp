#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "runtime/lowering.h"
#include "runtime/parallel_executor.h"

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

    graph.get_tensor(input1).persistent = true;
    graph.get_tensor(input2).persistent = true;
    graph.get_tensor(weight).persistent = true;

    graph.get_tensor(input1).data = {
        1, 2, 3,
        4, 5, 6
    };

    graph.get_tensor(input2).data = {
        1, 1, 1,
        2, 2, 2
    };

    graph.get_tensor(weight).data = {
        1, 2,
        3, 4,
        5, 6
    };

    graph.add_node(Node("matmul1", OpType::MatMul, {input1, weight}, {matmul1_out}));
    graph.add_node(Node("matmul2", OpType::MatMul, {input2, weight}, {matmul2_out}));
    graph.add_node(Node("add", OpType::Add, {matmul1_out, matmul2_out}, {add_out}));
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

    ParallelExecutor executor;
    executor.run(graph, plan, true, true);

    std::cout << "\nParallel Output:\n";

    for (float x : graph.get_tensor(output).data) {
        std::cout << x << " ";
    }

    std::cout << "\n";

    return 0;
}