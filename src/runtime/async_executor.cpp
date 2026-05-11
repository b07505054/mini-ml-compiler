#include "runtime/async_executor.h"

#include <chrono>
#include <future>
#include <iostream>

AsyncExecutor::AsyncExecutor() = default;

Backend& AsyncExecutor::select_backend(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:
            return cpu_backend;

        case BackendType::MockGPU:
            return mock_gpu_backend;

        default:
            return cpu_backend;
    }
}

void AsyncExecutor::run(
    Graph& graph,
    const ExecutionPlan& plan,
    bool verbose,
    bool profile
) {
    if (profile) {
        profiler.reset();
    }

    if (verbose) {
        std::cout << "[AsyncExecutor] Running execution plan with async task dispatch\n";
    }

    for (const auto& node : plan.ordered_nodes) {
        BackendType backend_type = scheduler.select_backend(node);
        Backend& backend = select_backend(backend_type);

        if (verbose) {
            std::cout << "[AsyncExecutor] Launching async node: "
                      << node.name
                      << " -> "
                      << backend.name()
                      << "\n";
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto task = std::async(
            std::launch::async,
            [&graph, &backend, node]() {
                backend.execute(graph, node);
            }
        );

        task.get();

        auto end = std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        if (profile) {
            profiler.record(node.name, backend.name(), ms);
        }
    }

    if (profile) {
        profiler.print_summary();
        profiler.export_json("../trace/async_runtime_trace.json");
    }
}