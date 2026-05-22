#pragma once

#include "ir/graph.h"
#include "pass/cost_report.h"

#include <string>
#include <vector>

struct PlannerCandidate {
    std::string name;

    float total_latency_ms = 0.f;
    float switch_cost_ms = 0.f;
    float gpu_occupancy = 0.f;
    float memory_pressure_mb = 0.f;

    bool chosen = false;

    std::vector<std::string> assignments;
};

class CostBasedPlanner {
public:
    PlannerCandidate choose_best_plan(
        Graph& graph,
        const CostReport& report
    );

private:
    PlannerCandidate evaluate_candidate(
        const std::string& name,
        Graph graph,
        const CostReport& report,
        const std::vector<std::string>& backends
    );

    float estimate_op_latency(
        const std::string& op_type,
        const std::string& backend
    );

    void export_candidates_json(
        const std::vector<PlannerCandidate>& candidates,
        const std::string& path
    );
};