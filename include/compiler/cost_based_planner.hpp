#pragma once

#include "ir/graph.h"
#include "pass/cost_report.h"

#include <string>
#include <vector>

struct PlannerOpCost {
    std::string op_name;
    std::string op_type;
    std::string backend;
    std::string cost_source;
    std::string decision_reason;

    float latency_ms = 0.f;
    float launch_overhead_ms = 0.f;
    float compute_time_ms = 0.f;
    float memory_time_ms = 0.f;
    float transfer_cost_ms = 0.f;
    size_t estimated_flops = 0;
    size_t estimated_bytes = 0;
};

struct PlannerCandidate {
    std::string name;

    float total_latency_ms = 0.f;
    float switch_cost_ms = 0.f;
    float transfer_cost_ms = 0.f;
    float gpu_occupancy = 0.f;
    float memory_pressure_mb = 0.f;

    bool chosen = false;

    std::vector<std::string> assignments;
    std::vector<PlannerOpCost> op_costs;
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

    PlannerOpCost estimate_op_latency(
        const std::string& op_name,
        const std::string& op_type,
        const std::string& backend,
        const CostReport& report
    );

    void export_candidates_json(
        const std::vector<PlannerCandidate>& candidates,
        const std::string& path
    );
};
