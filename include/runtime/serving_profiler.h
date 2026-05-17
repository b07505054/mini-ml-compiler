#pragma once

#include "runtime/llm_request.h"
#include "runtime/serving_trace_event.h"
#include <vector>

class ServingProfiler {
public:
    void add_finished_request(
        const LLMRequest& request
    );

    void print_metrics() const;
    void add_trace_event(
        int request_id,
        const std::string& phase,
        int step,
        double timestamp_ms
    );

    void export_trace_json(
        const std::string& path
    ) const;

private:
    std::vector<LLMRequest> finished_requests;
    std::vector<ServingTraceEvent> trace_events;
};