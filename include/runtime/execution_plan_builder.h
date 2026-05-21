#pragma once

#include "runtime/lowered_ir.h"
#include "runtime/execution_plan_v2.h"

class ExecutionPlanBuilder {
public:
    ExecutionPlanV2 build(
        const LoweredGraph& lowered
    ) const;

private:
    std::vector<int> find_dependencies(
        const LoweredGraph& lowered,
        size_t op_index
    ) const;

    std::string launch_config_for(
        const LoweredOp& op
    ) const;
};