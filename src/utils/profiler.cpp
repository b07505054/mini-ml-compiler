#include "utils/profiler.h"

#include <iostream>
#include <iomanip>
#include <fstream>

void Profiler::record(const std::string& op_name, double latency_ms) {
    events.push_back({op_name, "Unknown", latency_ms});
}

void Profiler::record(
    const std::string& op_name,
    const std::string& backend,
    double latency_ms
) {
    events.push_back({op_name, backend, latency_ms});
}

void Profiler::print_summary() const {
    std::cout << "\n=== Runtime Profiling Summary ===\n";

    double total = 0.0;

    for (const auto& e : events) {
        total += e.latency_ms;
    }

    for (const auto& e : events) {
        double pct = (total > 0.0)
            ? (100.0 * e.latency_ms / total)
            : 0.0;

        std::cout
            << std::setw(24)
            << std::left
            << (e.op_name + " [" + e.backend + "]")
            << " : "
            << std::setw(10)
            << e.latency_ms
            << " ms  ("
            << pct
            << "%)\n";
    }

    std::cout
        << "Total latency: "
        << total
        << " ms\n";
}

void Profiler::export_json(const std::string& path) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < events.size(); ++i) {
        const auto& e = events[i];

        out << "  {\n";
        out << "    \"op_name\": \"" << e.op_name << "\",\n";
        out << "    \"backend\": \"" << e.backend << "\",\n";
        out << "    \"latency_ms\": " << e.latency_ms << "\n";
        out << "  }";

        if (i + 1 < events.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout << "[Profiler] Exported runtime trace to "
              << path
              << "\n";
}

void Profiler::reset() {
    events.clear();
}