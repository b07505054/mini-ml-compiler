#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "runtime/lowering.h"
#include "runtime/executor.h"

#include <iostream>

int main() {
    Graph graph;

    int Q = graph.add_tensor(Tensor("Q", {2, 4}));
    int K = graph.add_tensor(Tensor("K", {2, 4}));
    int V = graph.add_tensor(Tensor("V", {2, 4}));
    int Out = graph.add_tensor(Tensor("Out", {}));

    graph.get_tensor(Q).persistent = true;
    graph.get_tensor(K).persistent = true;
    graph.get_tensor(V).persistent = true;

    graph.get_tensor(Q).data = {
        1, 0, 0, 0,
        0, 1, 0, 0
    };

    graph.get_tensor(K).data = {
        1, 0, 0, 0,
        0, 1, 0, 0
    };

    graph.get_tensor(V).data = {
        10, 0, 0, 0,
        0, 20, 0, 0
    };

    graph.add_node(Node("tiled_attention", OpType::TiledAttention, {Q, K, V}, {Out}));

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
    executor.run(graph, plan, true, true);

    std::cout << "\nTiled Attention Output:\n";
    for (float x : graph.get_tensor(Out).data) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    return 0;
}