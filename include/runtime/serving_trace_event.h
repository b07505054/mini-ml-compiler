#pragma once

#include <string>

struct ServingTraceEvent {
    int request_id;

    std::string phase;

    int step;

    double timestamp_ms;
};