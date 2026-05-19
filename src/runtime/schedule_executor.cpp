#include "runtime/schedule_executor.h"

#include <chrono>
#include <iostream>

Backend& ScheduleExecutor::select_backend(
    const std::string& backend_name
) {
    if (backend_name == "Metal") {
        return metal_backend;
    }

    if (backend_name == "MockGPU") {
        return mock_gpu_backend;
    }

    return cpu_backend;
}

const Node& ScheduleExecutor::get_node_by_op_id(
    const Graph& graph,
    int op_id
) const {
    return graph.nodes.at(
        static_cast<size_t>(op_id)
    );
}

double ScheduleExecutor::now_ms() const {
    auto t =
        std::chrono::high_resolution_clock
            ::now()
            .time_since_epoch();

    return std::chrono::duration<
        double,
        std::milli
    >(t).count();
}

void ScheduleExecutor::run(
    Graph& graph,
    const ExecutionSchedule& schedule,
    bool verbose,
    bool trace
) {
    if (verbose) {
        std::cout
            << "\n=== ScheduleExecutor Runtime ===\n";
    }

    for (const auto& entry : schedule.entries) {
        const Node& node =
            get_node_by_op_id(
                graph,
                entry.op_id
            );

        Backend& backend =
            select_backend(
                entry.backend
            );

        if (verbose) {
            std::cout
                << "[ScheduleExecutor] order="
                << entry.start_order
                << " op="
                << entry.op_name
                << " backend="
                << backend.name()
                << " mem_offset="
                << entry.memory_offset
                << "\n";
        }

        double start =
            now_ms();

        backend.execute(
            graph,
            node
        );

        double end =
            now_ms();

        if (trace) {
            tracer.add_event({
                entry.op_name,
                backend.name(),
                entry.memory_offset,
                start,
                end,
                end - start
            });
        }
    }

    if (trace) {
        tracer.dump();

        tracer.export_json(
            "../trace/scheduled_runtime_trace.json"
        );
    }
}