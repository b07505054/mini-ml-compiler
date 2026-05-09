#pragma once

#include "ir/graph.h"
#include "runtime/execution_plan.h"
#include "runtime/cpu_backend.h"
#include "runtime/mock_gpu_backend.h"
#include "runtime/backend_type.h"
#include "utils/profiler.h"
#include "runtime/backend_scheduler.h"

class Executor {
public:
    Executor();

    void run(
        Graph& graph,
        const ExecutionPlan& plan,
        bool verbose = true,
        bool profile = false,
        BackendType backend_type = BackendType::CPU
    );
    void run_scheduled(
        Graph& graph,
        const ExecutionPlan& plan,
        bool verbose = true,
        bool profile = false
    );
private:
    CPUBackend cpu_backend;
    MockGPUBackend mock_gpu_backend;
    BackendScheduler scheduler;
    Profiler profiler;

    Backend& select_backend(BackendType backend_type);
};