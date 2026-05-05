#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "pass/pass_manager.h"
#include "pass/fusion_pass.h"
#include "runtime/lowering.h"
#include "runtime/executor.h"
#include "utils/timer.h"

#include <iostream>
#include <memory>

int main() {
    const int M = 256;
    const int K = 256;
    const int N = 256;

    Graph graph;

    int input = graph.add_tensor(Tensor("input", {M, K}));
    int weight = graph.add_tensor(Tensor("weight", {K, N}));
    int bias = graph.add_tensor(Tensor("bias", {M, N}));
    int matmul_out = graph.add_tensor(Tensor("matmul_out", {}));
    int add_out = graph.add_tensor(Tensor("add_out", {}));
    int output = graph.add_tensor(Tensor("output", {}));

    for (int i = 0; i < graph.get_tensor(input).numel(); ++i) {
        graph.get_tensor(input).data[i] = 1.0f;
    }

    for (int i = 0; i < graph.get_tensor(weight).numel(); ++i) {
        graph.get_tensor(weight).data[i] = 0.01f;
    }

    for (int i = 0; i < graph.get_tensor(bias).numel(); ++i) {
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
        return 1;
    }

    PassManager pm;
    pm.add_pass(std::make_unique<FusionPass>());
    pm.run(graph);

    infer.run(graph);

    if (!verifier.verify(graph)) {
        return 1;
    }

    ExecutionPlan plan = lower_to_execution_plan(graph);

    Executor executor;

    const int warmup = 10;
    const int runs = 100;

    for (int i = 0; i < warmup; ++i) {
        executor.run(graph, plan, false);
    }

    Timer timer;
    timer.start();

    for (int i = 0; i < runs; ++i) {
        executor.run(graph, plan, false);
    }

    double total_ms = timer.stop_ms();
    double avg_ms = total_ms / runs;

    std::cout << "=== MatMul Benchmark ===\n";
    std::cout << "Shape: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "Runs: " << runs << "\n";
    std::cout << "Total latency: " << total_ms << " ms\n";
    std::cout << "Average latency: " << avg_ms << " ms\n";

    std::cout << "Output[0]: " << graph.get_tensor(output).data[0] << "\n";

    return 0;
}