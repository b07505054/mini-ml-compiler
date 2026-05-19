#pragma once

#include "runtime/execution_trace_event.h"

#include <string>
#include <vector>

class RuntimeTracer {
public:
    void add_event(
        const ExecutionTraceEvent& e
    );

    void dump() const;

    void export_json(
        const std::string& path
    ) const;

private:
    std::vector<ExecutionTraceEvent> events;
};