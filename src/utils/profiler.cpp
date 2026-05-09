#include "utils/profiler.h"

#include <iostream>
#include <iomanip>

void Profiler::record(const std::string& op_name, double latency_ms) {
    events.push_back({op_name, latency_ms});
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
            << e.op_name
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

void Profiler::reset() {
    events.clear();
}