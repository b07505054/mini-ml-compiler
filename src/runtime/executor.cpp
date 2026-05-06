#include "runtime/executor.h"
#include <chrono>
#include <iostream>

Executor::Executor()
    : registry(create_default_registry()) {}

void Executor::run(Graph& graph, const ExecutionPlan& plan, bool verbose, bool profile) {
    if (verbose) {
        std::cout << "[Executor] Running execution plan with operator registry\n";
    }

    for (const auto& node : plan.ordered_nodes) {
        if (verbose) {
            std::cout << "[Executor] Dispatching node: " << node.name << "\n";
        }

        if (profile) {
            auto t1 = std::chrono::high_resolution_clock::now();
            registry.dispatch(node.op, graph, node);
            auto t2 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            profiler.record(node.name, ms);
        } else {
            registry.dispatch(node.op, graph, node);
        }
    }

    if (profile) {
        profiler.print_summary();
    }
}