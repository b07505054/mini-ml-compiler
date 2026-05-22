#pragma once

#include "ir/graph.h"
#include "runtime/backend.h"
#include "runtime/backend_type.h"
#include "runtime/cpu_backend.h"
#include "runtime/execution_schedule.h"
#ifdef __APPLE__
#include "runtime/metal_backend.h"
#endif
#include "runtime/mock_gpu_backend.h"
#include "runtime/runtime_tracer.h"

class ScheduleExecutor {
public:
    void run(
        Graph& graph,
        const ExecutionSchedule& schedule,
        bool verbose = true,
        bool trace = true
    );

private:
    CPUBackend cpu_backend;
    MockGPUBackend mock_gpu_backend;
    #ifdef __APPLE__
        MetalBackend metal_backend;
    #endif
    RuntimeTracer tracer;

    Backend& select_backend(
        const std::string& backend_name
    );

    const Node& get_node_by_op_id(
        const Graph& graph,
        int op_id
    ) const;

    double now_ms() const;
};