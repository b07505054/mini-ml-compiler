#pragma once

#include "ir/graph.h"
#include "runtime/execution_plan.h"
#include "runtime/cpu_backend.h"
#include "runtime/mock_gpu_backend.h"
#include "runtime/provider_scheduler.h"
#include "runtime/backend_type.h"
#include "utils/profiler.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

class ParallelExecutor {
public:
    ParallelExecutor();

    void run(
        Graph& graph,
        const ExecutionPlan& plan,
        bool verbose = true,
        bool profile = false
    );

private:
    CPUBackend cpu_backend;
    MockGPUBackend mock_gpu_backend;
    ProviderScheduler provider_scheduler;
    Profiler profiler;

    Backend& select_backend(BackendType backend_type);

    std::unordered_map<int, std::vector<int>> build_consumers(
        const ExecutionPlan& plan
    ) const;

    std::unordered_map<int, int> build_dependency_count(
        const ExecutionPlan& plan
    ) const;
};