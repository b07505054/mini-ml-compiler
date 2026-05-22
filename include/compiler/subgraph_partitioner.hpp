#pragma once

#include "runtime/execution_schedule.h"

#include <string>
#include <vector>

struct SubgraphPartition {
    int subgraph_id;

    std::string backend;

    std::vector<std::string> ops;
};

class SubgraphPartitioner {
public:
    std::vector<SubgraphPartition> partition(
        const std::vector<ScheduleEntry>& schedule
    ) const;

    void export_json(
        const std::vector<SubgraphPartition>& partitions,
        const std::string& path
    ) const;
};