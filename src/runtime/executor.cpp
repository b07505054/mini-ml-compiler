#include "runtime/executor.h"

#include <iostream>
#include <chrono>

Executor::Executor() = default;

Backend& Executor::select_backend(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:
            return cpu_backend;

        case BackendType::MockGPU:
            return mock_gpu_backend;

        default:
            return cpu_backend;
    }
}

void Executor::run(
    Graph& graph,
    const ExecutionPlan& plan,
    bool verbose,
    bool profile,
    BackendType backend_type
) {
    Backend& backend = select_backend(backend_type);
    if (profile) {
        profiler.reset();
    }

    if (verbose) {
        std::cout << "[Executor] Running execution plan on backend: "
                  << backend.name() << "\n";
    }

    for (const auto& node : plan.ordered_nodes) {
        if (verbose) {
            std::cout << "[Executor] Dispatching node: "
                      << node.name
                      << " -> "
                      << backend.name()
                      << "\n";
        }

        if (profile) {
            auto t1 = std::chrono::high_resolution_clock::now();

            backend.execute(graph, node);

            auto t2 = std::chrono::high_resolution_clock::now();

            double ms =
                std::chrono::duration<double, std::milli>(t2 - t1).count();

            profiler.record(node.name, backend.name(), ms);
        } else {
            backend.execute(graph, node);
        }
    }

    if (profile) {
        profiler.print_summary();
    }
}

void Executor::run_scheduled(
    Graph& graph,
    const ExecutionPlan& plan,
    bool verbose,
    bool profile
) {
    if (profile) {
        profiler.reset();
    }

    if (verbose) {
        std::cout << "[Executor] Running execution plan with backend scheduler\n";
    }

    for (const auto& node : plan.ordered_nodes) {
        BackendType backend_type = scheduler.select_backend(node);
        Backend& backend = select_backend(backend_type);

        if (verbose) {
            std::cout << "[Scheduler] "
                      << node.name
                      << " -> "
                      << backend.name()
                      << "\n";
        }

        if (profile) {
            auto t1 = std::chrono::high_resolution_clock::now();

            backend.execute(graph, node);

            auto t2 = std::chrono::high_resolution_clock::now();

            double ms =
                std::chrono::duration<double, std::milli>(t2 - t1).count();

            profiler.record(node.name, backend.name(), ms);
        } else {
            backend.execute(graph, node);
        }
    }

    if (profile) {
        profiler.print_summary();
        profiler.export_json("../trace/runtime_trace.json");
    }
}