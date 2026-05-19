#include "runtime/runtime_tracer.h"

#include <fstream>
#include <iostream>

void RuntimeTracer::add_event(
    const ExecutionTraceEvent& e
) {
    events.push_back(e);
}

void RuntimeTracer::dump() const {
    std::cout
        << "\n=== Runtime Execution Trace ===\n";

    for (const auto& e : events) {
        std::cout
            << e.op_name
            << " | "
            << e.backend
            << " | latency="
            << e.latency_ms
            << " ms"
            << "\n";
    }
}

void RuntimeTracer::export_json(
    const std::string& path
) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < events.size(); ++i) {
        const auto& e = events[i];

        out << "  {\n";

        out << "    \"op_name\": \""
            << e.op_name
            << "\",\n";

        out << "    \"backend\": \""
            << e.backend
            << "\",\n";

        out << "    \"memory_offset\": "
            << e.memory_offset
            << ",\n";

        out << "    \"start_ms\": "
            << e.start_ms
            << ",\n";

        out << "    \"end_ms\": "
            << e.end_ms
            << ",\n";

        out << "    \"latency_ms\": "
            << e.latency_ms
            << "\n";

        out << "  }";

        if (i + 1 < events.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout
        << "[RuntimeTracer] Exported trace to "
        << path
        << "\n";
}