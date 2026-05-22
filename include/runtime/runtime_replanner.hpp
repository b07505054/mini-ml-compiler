#pragma once

#include "compiler/cost_based_planner.hpp"

#include <string>
#include <vector>

struct RuntimeObservation {
    std::string backend;
    float observed_latency_ms;
    bool overloaded = false;
};

class RuntimeReplanner {
public:
    PlannerCandidate replan(
        const PlannerCandidate& current,
        const std::vector<RuntimeObservation>& observations
    );
};