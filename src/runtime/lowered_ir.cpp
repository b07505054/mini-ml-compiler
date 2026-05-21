#include "runtime/lowered_ir.h"

#include <fstream>
#include <iostream>

void LoweredGraph::dump() const {
    std::cout << "\n=== Lowered Graph IR ===\n";

    for (const auto& op : ops) {
        std::cout
            << "[" << op.op_id << "] "
            << op.source_op_name
            << " -> "
            << op.lowered_op_type
            << " | backend="
            << op.backend
            << " | mem_offset="
            << op.memory_offset
            << "\n";
    }
}

void LoweredGraph::export_json(
    const std::string& path
) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < ops.size(); ++i) {
        const auto& op = ops[i];

        out << "  {\n";
        out << "    \"op_id\": " << op.op_id << ",\n";
        out << "    \"source_op_name\": \"" << op.source_op_name << "\",\n";
        out << "    \"lowered_op_type\": \"" << op.lowered_op_type << "\",\n";
        out << "    \"backend\": \"" << op.backend << "\",\n";

        out << "    \"inputs\": [";
        for (size_t j = 0; j < op.inputs.size(); ++j) {
            out << op.inputs[j];

            if (j + 1 < op.inputs.size()) {
                out << ", ";
            }
        }
        out << "],\n";

        out << "    \"outputs\": [";
        for (size_t j = 0; j < op.outputs.size(); ++j) {
            out << op.outputs[j];

            if (j + 1 < op.outputs.size()) {
                out << ", ";
            }
        }
        out << "],\n";

        out << "    \"memory_offset\": " << op.memory_offset << "\n";
        out << "  }";

        if (i + 1 < ops.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout
        << "[LoweredGraph] Exported lowered IR to "
        << path
        << "\n";
}