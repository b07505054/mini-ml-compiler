#include "runtime/executor.h"

#include <iostream>

Executor::Executor()
    : registry(create_default_registry()) {}

void Executor::run(Graph& graph, const ExecutionPlan& plan, bool verbose) {
    if (verbose) {
        std::cout << "[Executor] Running execution plan with operator registry\n";
    }

    for (const auto& node : plan.ordered_nodes) {
        registry.dispatch(node.op, graph, node);
    }
}