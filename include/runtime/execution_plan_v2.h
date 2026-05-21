#pragma once

#include <string>
#include <vector>

struct ExecutionStep {
    int step_id;

    int lowered_op_id;

    std::string lowered_op_type;

    std::string backend;

    std::vector<int> inputs;

    std::vector<int> outputs;

    std::vector<int> dependency_steps;

    int memory_offset;

    std::string launch_config;
};

struct ExecutionPlanV2 {
    std::vector<ExecutionStep> steps;

    void dump() const;

    void export_json(
        const std::string& path
    ) const;
};