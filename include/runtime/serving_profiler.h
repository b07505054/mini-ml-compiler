#pragma once

#include "runtime/llm_request.h"

#include <vector>

class ServingProfiler {
public:
    void add_finished_request(
        const LLMRequest& request
    );

    void print_metrics() const;

private:
    std::vector<LLMRequest> finished_requests;
};