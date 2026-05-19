#include "runtime/execution_schedule.h"
#include <fstream>
#include <iostream>

void ExecutionSchedule::dump() const {
    std::cout
        << "\n=== Static Execution Schedule ===\n";

    for (const auto& e : entries) {
        std::cout
            << "[" << e.start_order << "] "
            << e.op_name
            << " | "
            << e.op_type
            << " | backend="
            << e.backend
            << " | mem_offset="
            << e.memory_offset
            << "\n";
    }
}
void ExecutionSchedule::export_json(
    const std::string& path
) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];

        out << "  {\n";

        out << "    \"op_id\": "
            << e.op_id
            << ",\n";

        out << "    \"op_name\": \""
            << e.op_name
            << "\",\n";

        out << "    \"op_type\": \""
            << e.op_type
            << "\",\n";

        out << "    \"backend\": \""
            << e.backend
            << "\",\n";

        out << "    \"inputs\": [";

        for (size_t j = 0; j < e.inputs.size(); ++j) {
            out << e.inputs[j];

            if (j + 1 < e.inputs.size()) {
                out << ", ";
            }
        }

        out << "],\n";

        out << "    \"outputs\": [";

        for (size_t j = 0; j < e.outputs.size(); ++j) {
            out << e.outputs[j];

            if (j + 1 < e.outputs.size()) {
                out << ", ";
            }
        }

        out << "],\n";

        out << "    \"start_order\": "
            << e.start_order
            << ",\n";

        out << "    \"memory_offset\": "
            << e.memory_offset
            << "\n";

        out << "  }";

        if (i + 1 < entries.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout << "[ExecutionSchedule] Exported schedule to "
              << path
              << "\n";
}