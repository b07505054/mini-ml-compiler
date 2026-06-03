#include "compiler/cost_based_planner.hpp"

#include "ir/op_type_utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

struct BackendCostModel {
    float bandwidth_gbps;
    float compute_tflops;
    float launch_overhead_ms;
};

BackendCostModel backend_cost_model(const std::string& backend) {
    if (backend == "Metal") {
        return {
            120.0f,
            2.0f,
            0.08f
        };
    }
    return {
        25.0f,
        0.15f,
        0.005f
    };
}

const CostReportEntry* find_report_entry(
    const CostReport& report,
    const std::string& op_name,
    const std::string& op_type
) {
    auto by_name = std::find_if(
        report.entries.begin(),
        report.entries.end(),
        [&](const CostReportEntry& entry) {
            return entry.op_name == op_name;
        }
    );
    if (by_name != report.entries.end()) {
        return &*by_name;
    }

    auto by_type = std::find_if(
        report.entries.begin(),
        report.entries.end(),
        [&](const CostReportEntry& entry) {
            return entry.op_type == op_type;
        }
    );
    if (by_type != report.entries.end()) {
        return &*by_type;
    }

    return nullptr;
}

float ms_from_bytes(size_t bytes, float bandwidth_gbps) {
    if (bandwidth_gbps <= 0.0f) {
        return 0.0f;
    }
    return static_cast<float>(bytes) / (bandwidth_gbps * 1'000'000.0f);
}

float ms_from_flops(size_t flops, float compute_tflops) {
    if (compute_tflops <= 0.0f) {
        return 0.0f;
    }
    return static_cast<float>(flops) / (compute_tflops * 1'000'000'000.0f);
}

float fallback_hardcoded_latency(
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

} // namespace

PlannerOpCost CostBasedPlanner::estimate_op_latency(
    const std::string& op_name,
    const std::string& op_type,
    const std::string& backend,
    const CostReport& report
) {
    PlannerOpCost cost;
    cost.op_name = op_name;
    cost.op_type = op_type;
    cost.backend = backend;

    const CostReportEntry* entry =
        find_report_entry(
            report,
            op_name,
            op_type
        );

    if (!entry) {
        cost.latency_ms =
            fallback_hardcoded_latency(
                op_type,
                backend
            );
        cost.cost_source = "fallback_static_table";
        cost.decision_reason =
            "no_cost_report_entry_conservative_fallback";
        return cost;
    }

    cost.estimated_flops =
        entry->estimated_flops;
    cost.estimated_bytes =
        entry->estimated_read_bytes
        + entry->estimated_write_bytes;

    if (
        entry->actual_latency_ms > 0.0 &&
        (
            entry->actual_backend == backend ||
            entry->actual_backend == "N/A"
        )
    ) {
        cost.latency_ms =
            static_cast<float>(
                entry->actual_latency_ms
            );
        cost.cost_source =
            "runtime_observed_latency";
        cost.decision_reason =
            "CostReport actual latency overrides static model";
        return cost;
    }

    BackendCostModel model =
        backend_cost_model(
            backend
        );

    cost.launch_overhead_ms =
        model.launch_overhead_ms;
    if (
        entry->backend == backend &&
        entry->estimated_kernel_launch_cost > 0.0
    ) {
        cost.launch_overhead_ms =
            static_cast<float>(
                entry->estimated_kernel_launch_cost
            );
    }

    cost.compute_time_ms =
        ms_from_flops(
            cost.estimated_flops,
            model.compute_tflops
        );

    cost.memory_time_ms =
        ms_from_bytes(
            cost.estimated_bytes,
            model.bandwidth_gbps
        );

    if (entry->backend != backend) {
        cost.transfer_cost_ms =
            ms_from_bytes(
                cost.estimated_bytes,
                40.0f
            );
    }

    cost.latency_ms =
        cost.launch_overhead_ms
        + cost.compute_time_ms
        + cost.memory_time_ms
        + cost.transfer_cost_ms;

    if (cost.latency_ms <= 0.0f) {
        cost.latency_ms =
            fallback_hardcoded_latency(
                op_type,
                backend
            );
        cost.cost_source =
            "fallback_static_table";
        cost.decision_reason =
            "CostReport entry lacked FLOPs/bytes; used conservative fallback";
        return cost;
    }

    cost.cost_source =
        "cost_report_static_model";
    cost.decision_reason =
        entry->backend == backend
            ? "CostReport FLOPs/bytes calibrated backend estimate"
            : "Candidate backend differs from CostReport placement; added transfer cost";

    return cost;
}

PlannerCandidate CostBasedPlanner::evaluate_candidate(
    const std::string& name,
    Graph graph,
    const CostReport& report,
    const std::vector<std::string>& backends
) {
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

        PlannerOpCost op_cost =
            estimate_op_latency(
                node.name,
                op_type,
                backend,
                report
            );

        if (
            !prev_backend.empty() &&
            prev_backend != backend
        ) {
            total += 0.02f;
            switch_cost += 0.02f;
        }

        total += op_cost.latency_ms;
        c.transfer_cost_ms +=
            op_cost.transfer_cost_ms;

        if (backend == "Metal") {
            gpu_time += op_cost.latency_ms;
        }

        prev_backend = backend;

        c.assignments.push_back(
            node.name + " -> " + backend
        );
        c.op_costs.push_back(
            op_cost
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

    out << "{\n";
    out << "  \"format\": \"cost_based_planner.v2\",\n";
    out << "  \"cost_model\": {\n";
    out << "    \"source\": \"CostReport FLOPs/bytes plus backend bandwidth/compute/launch model\",\n";
    out << "    \"metal_bandwidth_gbps\": 120,\n";
    out << "    \"metal_compute_tflops\": 2,\n";
    out << "    \"cpu_bandwidth_gbps\": 25,\n";
    out << "    \"cpu_compute_tflops\": 0.15,\n";
    out << "    \"backend_transfer_bandwidth_gbps\": 40\n";
    out << "  },\n";
    out << "  \"candidates\": [\n";

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];

        out << "    {\n";

        out << "      \"name\": \""
            << c.name
            << "\",\n";

        out << "      \"total_latency_ms\": "
            << c.total_latency_ms
            << ",\n";

        out << "      \"switch_cost_ms\": "
            << c.switch_cost_ms
            << ",\n";

        out << "      \"transfer_cost_ms\": "
            << c.transfer_cost_ms
            << ",\n";

        out << "      \"gpu_occupancy\": "
            << c.gpu_occupancy
            << ",\n";

        out << "      \"memory_pressure_mb\": "
            << c.memory_pressure_mb
            << ",\n";

        out << "      \"chosen\": "
            << (c.chosen ? "true" : "false")
            << ",\n";

        out << "      \"assignments\": [";

        for (size_t j = 0; j < c.assignments.size(); ++j) {
            out << "\""
                << c.assignments[j]
                << "\"";

            if (j + 1 < c.assignments.size()) {
                out << ", ";
            }
        }

        out << "],\n";

        out << "      \"op_costs\": [\n";
        for (size_t j = 0; j < c.op_costs.size(); ++j) {
            const auto& op = c.op_costs[j];
            out << "        {\n";
            out << "          \"op_name\": \"" << op.op_name << "\",\n";
            out << "          \"op_type\": \"" << op.op_type << "\",\n";
            out << "          \"backend\": \"" << op.backend << "\",\n";
            out << "          \"latency_ms\": " << op.latency_ms << ",\n";
            out << "          \"launch_overhead_ms\": " << op.launch_overhead_ms << ",\n";
            out << "          \"compute_time_ms\": " << op.compute_time_ms << ",\n";
            out << "          \"memory_time_ms\": " << op.memory_time_ms << ",\n";
            out << "          \"transfer_cost_ms\": " << op.transfer_cost_ms << ",\n";
            out << "          \"estimated_flops\": " << op.estimated_flops << ",\n";
            out << "          \"estimated_bytes\": " << op.estimated_bytes << ",\n";
            out << "          \"cost_source\": \"" << op.cost_source << "\",\n";
            out << "          \"decision_reason\": \"" << op.decision_reason << "\"\n";
            out << "        }";
            if (j + 1 < c.op_costs.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "      ]\n";

        out << "    }";

        if (i + 1 < candidates.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";

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
        std::cout
            << "  transfer_cost_ms="
            << c.transfer_cost_ms
            << "\n";

        for (const auto& a : c.assignments) {
            std::cout
                << "    "
                << a
                << "\n";
        }
        for (const auto& op : c.op_costs) {
            std::cout
                << "    cost "
                << op.op_name
                << " latency_ms="
                << op.latency_ms
                << " source="
                << op.cost_source
                << " reason="
                << op.decision_reason
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
