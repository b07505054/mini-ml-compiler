#include "runtime/execution_plan_v2.h"

#include <fstream>
#include <iostream>

void ExecutionPlanV2::dump() const {
    std::cout
        << "\n=== ExecutionPlan IR v2 ===\n";

    for (const auto& s : steps) {
        std::cout
            << "[step "
            << s.step_id
            << "] "
            << s.lowered_op_type
            << " | backend="
            << s.backend
            << " | mem_offset="
            << s.memory_offset
            << " | deps=[";

        for (size_t i = 0; i < s.dependency_steps.size(); ++i) {
            std::cout << s.dependency_steps[i];

            if (i + 1 < s.dependency_steps.size()) {
                std::cout << ", ";
            }
        }

        std::cout
            << "]"
            << " | launch="
            << s.launch_config
            << "\n";
    }
}

void ExecutionPlanV2::export_json(
    const std::string& path
) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < steps.size(); ++i) {
        const auto& s = steps[i];

        out << "  {\n";

        out << "    \"step_id\": "
            << s.step_id
            << ",\n";

        out << "    \"lowered_op_id\": "
            << s.lowered_op_id
            << ",\n";

        out << "    \"lowered_op_type\": \""
            << s.lowered_op_type
            << "\",\n";

        out << "    \"backend\": \""
            << s.backend
            << "\",\n";

        out << "    \"inputs\": [";

        for (size_t j = 0; j < s.inputs.size(); ++j) {
            out << s.inputs[j];

            if (j + 1 < s.inputs.size()) {
                out << ", ";
            }
        }

        out << "],\n";

        out << "    \"outputs\": [";

        for (size_t j = 0; j < s.outputs.size(); ++j) {
            out << s.outputs[j];

            if (j + 1 < s.outputs.size()) {
                out << ", ";
            }
        }

        out << "],\n";

        out << "    \"dependency_steps\": [";

        for (size_t j = 0; j < s.dependency_steps.size(); ++j) {
            out << s.dependency_steps[j];

            if (j + 1 < s.dependency_steps.size()) {
                out << ", ";
            }
        }

        out << "],\n";

        out << "    \"memory_offset\": "
            << s.memory_offset
            << ",\n";

        out << "    \"launch_config\": \""
            << s.launch_config
            << "\"\n";

        out << "  }";

        if (i + 1 < steps.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout
        << "[ExecutionPlanV2] Exported plan to "
        << path
        << "\n";
}