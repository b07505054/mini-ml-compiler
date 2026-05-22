#include "runtime/runtime_replanner.hpp"

#include <fstream>
#include <iostream>

void export_runtime_replan_json(
    const PlannerCandidate& before,
    const PlannerCandidate& after,
    const std::vector<RuntimeObservation>& observations,
    const std::string& path
) {
    std::ofstream out(path);

    out << "{\n";

    out << "  \"trigger\": \"runtime_backend_overload\",\n";

    out << "  \"observations\": [\n";

    for (size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];

        out << "    {\n";
        out << "      \"backend\": \""
            << obs.backend
            << "\",\n";
        out << "      \"observed_latency_ms\": "
            << obs.observed_latency_ms
            << ",\n";
        out << "      \"overloaded\": "
            << (obs.overloaded ? "true" : "false")
            << "\n";
        out << "    }";

        if (i + 1 < observations.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "  ],\n";

    auto write_plan =
        [&](const std::string& name,
            const PlannerCandidate& plan) {
            out << "  \""
                << name
                << "\": {\n";

            out << "    \"name\": \""
                << plan.name
                << "\",\n";

            out << "    \"total_latency_ms\": "
                << plan.total_latency_ms
                << ",\n";

            out << "    \"switch_cost_ms\": "
                << plan.switch_cost_ms
                << ",\n";

            out << "    \"gpu_occupancy\": "
                << plan.gpu_occupancy
                << ",\n";

            out << "    \"assignments\": [";

            for (size_t i = 0; i < plan.assignments.size(); ++i) {
                out << "\""
                    << plan.assignments[i]
                    << "\"";

                if (i + 1 < plan.assignments.size()) {
                    out << ", ";
                }
            }

            out << "]\n";

            out << "  }";
        };

    write_plan(
        "before",
        before
    );

    out << ",\n";

    write_plan(
        "after",
        after
    );

    out << "\n}\n";

    std::cout
        << "[RuntimeReplanner] Exported replan trace to "
        << path
        << "\n";
}

PlannerCandidate RuntimeReplanner::replan(
    const PlannerCandidate& current,
    const std::vector<RuntimeObservation>& observations
) {
    PlannerCandidate updated = current;

    bool metal_overloaded = false;

    for (const auto& obs : observations) {
        std::cout
            << "[RuntimeObservation] "
            << obs.backend
            << " latency="
            << obs.observed_latency_ms
            << " ms"
            << "\n";

        if (
            obs.backend == "Metal" &&
            obs.overloaded
        ) {
            metal_overloaded = true;

            std::cout
                << "  detected overload on Metal backend\n";
        }
    }

    if (!metal_overloaded) {
        export_runtime_replan_json(
            current,
            updated,
            observations,
            "../trace/cv_runtime_replan.json"
        );

        return updated;
    }

    updated.name =
        "runtime_replanned_cpu_fallback";

    updated.assignments.clear();

    updated.total_latency_ms = 2.10f;
    updated.switch_cost_ms = 0.0f;
    updated.gpu_occupancy = 0.0f;

    updated.assignments.push_back(
        "conv1 -> CPU"
    );

    updated.assignments.push_back(
        "pool1 -> CPU"
    );

    updated.assignments.push_back(
        "flatten -> CPU"
    );

    updated.assignments.push_back(
        "linear -> CPU"
    );

    std::cout
        << "\n=== Runtime Replanned ===\n";

    for (const auto& a : updated.assignments) {
        std::cout
            << "  "
            << a
            << "\n";
    }

    export_runtime_replan_json(
        current,
        updated,
        observations,
        "../trace/cv_runtime_replan.json"
    );

    return updated;
}