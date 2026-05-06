#include "frontend/mlir_importer.h"
#include "analysis/shape_inference.h"
#include "analysis/graph_verifier.h"
#include "runtime/memory_planner.h"
#include "pass/pass_manager.h"
#include "pass/fusion_pass.h"
#include "runtime/lowering.h"
#include "runtime/executor.h"

#include <iostream>
#include <memory>

int main() {
    Graph graph = load_mlir_graph("../models/tiny_mlp.mlir");

    // Set input data after import.
    graph.get_tensor(0).data = {1, 1, 1, 1};

    std::cout << "Imported graph:\n";
    graph.dump();

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
    executor.run(graph, plan, true);

    std::cout << "\nOutput:\n";
    for (const auto& tensor : graph.tensors) {
        if (tensor.name == "output") {
            for (float x : tensor.data) {
                std::cout << x << " ";
            }
            std::cout << "\n";
        }
    }

    return 0;
}