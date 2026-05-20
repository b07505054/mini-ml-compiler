#include "pass/cost_report.h"

#include <fstream>
#include <iostream>

void CostReport::dump() const {
    std::cout
        << "\n=== Compiler Cost Report ===\n";

    for (const auto& e : entries) {
        std::cout
            << e.op_name
            << " | "
            << e.op_type
            << " | backend="
            << e.backend
            << " | read_bytes="
            << e.estimated_read_bytes
            << " | write_bytes="
            << e.estimated_write_bytes
            << " | flops="
            << e.estimated_flops
            << " | intensity="
            << e.arithmetic_intensity
            << " | launch_cost="
            << e.estimated_kernel_launch_cost
            << " | backend_switch_cost="
            << e.estimated_backend_switch_cost
            << " | actual_backend="
            << e.actual_backend
            << " | actual_latency_ms="
            << e.actual_latency_ms
            << "\n";
        if (!e.fusion_note.empty()) {
            std::cout
                << "    fusion: "
                << e.fusion_note
                << "\n";
        }
    }
}

void CostReport::export_json(
    const std::string& path
) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];

        out << "  {\n";

        out << "    \"op_name\": \""
            << e.op_name
            << "\",\n";

        out << "    \"op_type\": \""
            << e.op_type
            << "\",\n";

        out << "    \"backend\": \""
            << e.backend
            << "\",\n";

        out << "    \"estimated_read_bytes\": "
            << e.estimated_read_bytes
            << ",\n";

        out << "    \"estimated_write_bytes\": "
            << e.estimated_write_bytes
            << ",\n";

        out << "    \"estimated_flops\": "
            << e.estimated_flops
            << ",\n";

        out << "    \"arithmetic_intensity\": "
            << e.arithmetic_intensity
            << ",\n";

        out << "    \"estimated_kernel_launch_cost\": "
            << e.estimated_kernel_launch_cost
            << ",\n";

        out << "    \"estimated_backend_switch_cost\": "
            << e.estimated_backend_switch_cost
            << ",\n";
        out << "    \"actual_backend\": \""
            << e.actual_backend
            << "\",\n";

        out << "    \"actual_latency_ms\": "
            << e.actual_latency_ms
            << ",\n";
        out << "    \"fusion_note\": \""
            << e.fusion_note
            << "\"\n";

        out << "  }";

        if (i + 1 < entries.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout
        << "[CostReport] Exported report to "
        << path
        << "\n";
}