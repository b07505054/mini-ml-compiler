#include "compiler/cost_based_planner.hpp"

#include "ir/op_type_utils.h"

#include <fstream>
#include <iostream>

float CostBasedPlanner::estimate_op_latency(
    const std::string& op_type,
    const std::string& backend
) {
    float base = 0.1f;

    if (op_type == "FusedConvBatchNormReLU") {
        base = 0.32f;
    } else if (op_type == "Conv2D") {
        base = 0.36f;
    } else if (op_type == "MaxPool") {
        base = 0.18f;
    } else if (op_type == "Flatten") {
        base = 0.05f;
    } else if (op_type == "Linear") {
        base = 0.21f;
    }

    if (backend == "CPU") {
        base *= 4.0f;
    }

    return base;
}

PlannerCandidate CostBasedPlanner::evaluate_candidate(
    const std::string& name,
    Graph graph,
    const CostReport& report,
    const std::vector<std::string>& backends
) {
    (void) report;

    PlannerCandidate c;
    c.name = name;

    float total = 0.0f;
    float switch_cost = 0.0f;
    float gpu_time = 0.0f;

    std::string prev_backend;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        if (i >= backends.size()) {
            continue;
        }

        std::string backend = backends[i];

        std::string op_type =
            op_type_to_string(node.op);

        float latency =
            estimate_op_latency(
                op_type,
                backend
            );

        if (
            !prev_backend.empty() &&
            prev_backend != backend
        ) {
            total += 0.02f;
            switch_cost += 0.02f;
        }

        total += latency;

        if (backend == "Metal") {
            gpu_time += latency;
        }

        prev_backend = backend;

        c.assignments.push_back(
            node.name + " -> " + backend
        );
    }

    c.total_latency_ms = total;
    c.switch_cost_ms = switch_cost;

    if (total > 0.0f) {
        c.gpu_occupancy =
            gpu_time / total;
    }

    c.memory_pressure_mb = 18.0f;

    return c;
}

void CostBasedPlanner::export_candidates_json(
    const std::vector<PlannerCandidate>& candidates,
    const std::string& path
) {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];

        out << "  {\n";

        out << "    \"name\": \""
            << c.name
            << "\",\n";

        out << "    \"total_latency_ms\": "
            << c.total_latency_ms
            << ",\n";

        out << "    \"switch_cost_ms\": "
            << c.switch_cost_ms
            << ",\n";

        out << "    \"gpu_occupancy\": "
            << c.gpu_occupancy
            << ",\n";

        out << "    \"memory_pressure_mb\": "
            << c.memory_pressure_mb
            << ",\n";

        out << "    \"chosen\": "
            << (c.chosen ? "true" : "false")
            << ",\n";

        out << "    \"assignments\": [";

        for (size_t j = 0; j < c.assignments.size(); ++j) {
            out << "\""
                << c.assignments[j]
                << "\"";

            if (j + 1 < c.assignments.size()) {
                out << ", ";
            }
        }

        out << "]\n";

        out << "  }";

        if (i + 1 < candidates.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout
        << "[CostBasedPlanner] Exported planner candidates to "
        << path
        << "\n";
}

PlannerCandidate CostBasedPlanner::choose_best_plan(
    Graph& graph,
    const CostReport& report
) {
    std::vector<PlannerCandidate> candidates;

    candidates.push_back(
        evaluate_candidate(
            "current",
            graph,
            report,
            {
                "Metal",
                "CPU",
                "CPU",
                "Metal"
            }
        )
    );

    candidates.push_back(
        evaluate_candidate(
            "all_metal",
            graph,
            report,
            {
                "Metal",
                "Metal",
                "Metal",
                "Metal"
            }
        )
    );

    candidates.push_back(
        evaluate_candidate(
            "metal_pool_only",
            graph,
            report,
            {
                "Metal",
                "Metal",
                "CPU",
                "Metal"
            }
        )
    );

    size_t best_index = 0;

    for (size_t i = 1; i < candidates.size(); ++i) {
        if (
            candidates[i].total_latency_ms <
            candidates[best_index].total_latency_ms
        ) {
            best_index = i;
        }
    }

    candidates[best_index].chosen = true;

    for (const auto& c : candidates) {
        std::cout
            << "\n[PlannerCandidate] "
            << c.name
            << "\n";

        std::cout
            << "  total_latency_ms="
            << c.total_latency_ms
            << "\n";

        std::cout
            << "  switch_cost_ms="
            << c.switch_cost_ms
            << "\n";

        std::cout
            << "  gpu_occupancy="
            << c.gpu_occupancy
            << "\n";

        std::cout
            << "  memory_pressure_mb="
            << c.memory_pressure_mb
            << "\n";

        for (const auto& a : c.assignments) {
            std::cout
                << "    "
                << a
                << "\n";
        }
    }

    std::cout
        << "\n=== BEST PLAN ===\n"
        << candidates[best_index].name
        << "\n";

    export_candidates_json(
        candidates,
        "../trace/cv_cost_based_planner.json"
    );

    return candidates[best_index];
}