#include "runtime/parallel_executor.h"

#include <chrono>
#include <future>
#include <iostream>
#include <queue>

ParallelExecutor::ParallelExecutor() = default;

Backend& ParallelExecutor::select_backend(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:
            return cpu_backend;

        case BackendType::MockGPU:
            return mock_gpu_backend;

        default:
            return cpu_backend;
    }
}

std::unordered_map<int, std::vector<int>> ParallelExecutor::build_consumers(
    const ExecutionPlan& plan
) const {
    std::unordered_map<int, std::vector<int>> consumers;

    for (int i = 0; i < static_cast<int>(plan.ordered_nodes.size()); ++i) {
        const auto& producer = plan.ordered_nodes[i];

        for (int out : producer.outputs) {
            for (int j = 0; j < static_cast<int>(plan.ordered_nodes.size()); ++j) {
                if (i == j) {
                    continue;
                }

                const auto& consumer = plan.ordered_nodes[j];

                for (int in : consumer.inputs) {
                    if (in == out) {
                        consumers[i].push_back(j);
                    }
                }
            }
        }
    }

    return consumers;
}

std::unordered_map<int, int> ParallelExecutor::build_dependency_count(
    const ExecutionPlan& plan
) const {
    std::unordered_map<int, int> deps;

    for (int i = 0; i < static_cast<int>(plan.ordered_nodes.size()); ++i) {
        deps[i] = 0;
    }

    for (int i = 0; i < static_cast<int>(plan.ordered_nodes.size()); ++i) {
        const auto& node = plan.ordered_nodes[i];

        for (int j = 0; j < static_cast<int>(plan.ordered_nodes.size()); ++j) {
            if (i == j) {
                continue;
            }

            const auto& maybe_producer = plan.ordered_nodes[j];

            for (int in : node.inputs) {
                for (int out : maybe_producer.outputs) {
                    if (in == out) {
                        deps[i]++;
                    }
                }
            }
        }
    }

    return deps;
}

void ParallelExecutor::run(
    Graph& graph,
    const ExecutionPlan& plan,
    bool verbose,
    bool profile
) {
    if (profile) {
        profiler.reset();
    }

    auto consumers = build_consumers(plan);
    auto deps = build_dependency_count(plan);

    std::unordered_set<int> completed;
    std::unordered_set<int> launched;

    if (verbose) {
        std::cout << "[ParallelExecutor] Running dependency-aware parallel execution\n";
    }

    while (completed.size() < plan.ordered_nodes.size()) {
        std::vector<int> ready;

        for (int i = 0; i < static_cast<int>(plan.ordered_nodes.size()); ++i) {
            if (!launched.count(i) && deps[i] == 0) {
                ready.push_back(i);
                launched.insert(i);
            }
        }

        if (ready.empty()) {
            std::cerr << "[ParallelExecutor] Error: no ready nodes. Graph may contain a cycle.\n";
            return;
        }

        std::vector<std::future<void>> futures;

        for (int idx : ready) {
            const auto& node = plan.ordered_nodes[idx];

            BackendType backend_type = provider_scheduler.select_backend(node);
            Backend& backend = select_backend(backend_type);

            if (verbose) {
                std::cout << "[ParallelExecutor] Launching ready node: "
                          << node.name
                          << " -> "
                          << backend.name()
                          << "\n";
            }

            futures.push_back(
                std::async(
                    std::launch::async,
                    [&graph, &backend, &node, &profiler = this->profiler, profile]() {
                        auto t1 = std::chrono::high_resolution_clock::now();

                        backend.execute(graph, node);

                        auto t2 = std::chrono::high_resolution_clock::now();

                        if (profile) {
                            double ms =
                                std::chrono::duration<double, std::milli>(t2 - t1).count();

                            profiler.record(node.name, backend.name(), ms);
                        }
                    }
                )
            );
        }

        for (auto& f : futures) {
            f.get();
        }

        for (int idx : ready) {
            completed.insert(idx);

            for (int consumer : consumers[idx]) {
                deps[consumer]--;
            }
        }
    }

    if (profile) {
        profiler.print_summary();
        profiler.export_json("../trace/parallel_runtime_trace.json");
    }
}