#include "compiler/subgraph_partitioner.hpp"

#include <fstream>
#include <iostream>

std::vector<SubgraphPartition> SubgraphPartitioner::partition(
    const std::vector<ScheduleEntry>& schedule
) const {
    std::vector<SubgraphPartition> result;

    if (schedule.empty()) {
        return result;
    }

    int current_id = 0;

    SubgraphPartition current;
    current.subgraph_id = current_id;
    current.backend = schedule[0].backend;

    for (const auto& step : schedule) {
        if (step.backend != current.backend) {
            result.push_back(current);

            current_id++;

            current = {};
            current.subgraph_id = current_id;
            current.backend = step.backend;
        }

        current.ops.push_back(step.op_name);
    }

    result.push_back(current);

    return result;
}

void SubgraphPartitioner::export_json(
    const std::vector<SubgraphPartition>& partitions,
    const std::string& path
) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < partitions.size(); ++i) {
        const auto& p = partitions[i];

        out << "  {\n";

        out << "    \"subgraph_id\": "
            << p.subgraph_id
            << ",\n";

        out << "    \"backend\": \""
            << p.backend
            << "\",\n";

        out << "    \"ops\": [";

        for (size_t j = 0; j < p.ops.size(); ++j) {
            out << "\""
                << p.ops[j]
                << "\"";

            if (j + 1 < p.ops.size()) {
                out << ", ";
            }
        }

        out << "]\n";

        out << "  }";

        if (i + 1 < partitions.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";

    std::cout
        << "[SubgraphPartitioner] Exported partitions to "
        << path
        << "\n";
}