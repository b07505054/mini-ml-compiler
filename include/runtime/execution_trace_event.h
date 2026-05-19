#pragma once

#include <string>

struct ExecutionTraceEvent {
    std::string op_name;

    std::string backend;

    int memory_offset;

    double start_ms;

    double end_ms;

    double latency_ms;
};