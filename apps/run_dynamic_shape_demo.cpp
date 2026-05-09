#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "runtime/lowering.h"
#include "runtime/executor.h"

#include <iostream>

void run_case(int batch_size) {
    Graph graph;

    int input = graph.add_tensor(Tensor("input", {batch_size, 3}));
    int weight = graph.add_tensor(Tensor("weight", {3, 2}));
    int bias = graph.add_tensor(Tensor("bias", {batch_size, 2}));
    int matmul_out = graph.add_tensor(Tensor("matmul_out", {}));
    int add_out = graph.add_tensor(Tensor("add_out", {}));
    int output = graph.add_tensor(Tensor("output", {}));

    graph.get_tensor(input).persistent = true;
    graph.get_tensor(weight).persistent = true;
    graph.get_tensor(bias).persistent = true;

    graph.get_tensor(input).data.resize(batch_size * 3);
    graph.get_tensor(bias).data.resize(batch_size * 2);

    for (int i = 0; i < batch_size * 3; ++i) {
        graph.get_tensor(input).data[i] = static_cast<float>((i % 6) + 1);
    }

    graph.get_tensor(weight).data = {
        1, 2,
        3, 4,
        5, 6
    };

    for (int i = 0; i < batch_size * 2; ++i) {
        graph.get_tensor(bias).data[i] = 1.0f;
    }

    graph.add_node(Node("matmul", OpType::MatMul, {input, weight}, {matmul_out}));
    graph.add_node(Node("add", OpType::Add, {matmul_out, bias}, {add_out}));
    graph.add_node(Node("relu", OpType::ReLU, {add_out}, {output}));

    ShapeInference infer;
    infer.run(graph);

    MemoryPlanner memory_planner;
    memory_planner.plan(graph);

    GraphVerifier verifier;
    if (!verifier.verify(graph)) {
        std::cerr << "Verification failed\n";
        return;
    }

    ExecutionPlan plan = lower_to_execution_plan(graph);

    Executor executor;
    executor.run(graph, plan, true, true);

    std::cout << "\nDynamic batch size: " << batch_size << "\n";
    std::cout << "Output shape: ["
              << graph.get_tensor(output).shape[0]
              << ", "
              << graph.get_tensor(output).shape[1]
              << "]\n";

    std::cout << "Output:\n";
    for (float x : graph.get_tensor(output).data) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== Dynamic Shape Runtime Demo ===\n";

    run_case(1);
    run_case(2);
    run_case(4);

    return 0;
}