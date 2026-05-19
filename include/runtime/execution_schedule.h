#pragma once

#include <string>
#include <vector>

struct ScheduleEntry {
    int op_id;

    std::string op_name;

    std::string op_type;

    std::string backend;

    std::vector<int> inputs;

    std::vector<int> outputs;

    int start_order;

    int memory_offset;
};

struct ExecutionSchedule {
    std::vector<ScheduleEntry> entries;

    void dump() const;
    void export_json(
        const std::string& path
    ) const;
};