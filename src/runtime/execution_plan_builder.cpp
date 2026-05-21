#include "runtime/execution_plan_builder.h"

#include <algorithm>

std::vector<int> ExecutionPlanBuilder::find_dependencies(
    const LoweredGraph& lowered,
    size_t op_index
) const {
    std::vector<int> deps;

    const auto& op =
        lowered.ops[op_index];

    for (size_t j = 0; j < op_index; ++j) {
        const auto& prev =
            lowered.ops[j];

        for (int input : op.inputs) {
            if (
                std::find(
                    prev.outputs.begin(),
                    prev.outputs.end(),
                    input
                ) != prev.outputs.end()
            ) {
                deps.push_back(
                    static_cast<int>(j)
                );
            }
        }
    }

    return deps;
}

std::string ExecutionPlanBuilder::launch_config_for(
    const LoweredOp& op
) const {
    if (op.backend == "Metal") {
        return "threadgroup=256";
    }

    return "host-call";
}

ExecutionPlanV2 ExecutionPlanBuilder::build(
    const LoweredGraph& lowered
) const {
    ExecutionPlanV2 plan;

    for (size_t i = 0; i < lowered.ops.size(); ++i) {
        const auto& op =
            lowered.ops[i];

        ExecutionStep step;

        step.step_id =
            static_cast<int>(i);

        step.lowered_op_id =
            op.op_id;

        step.lowered_op_type =
            op.lowered_op_type;

        step.backend =
            op.backend;

        step.inputs =
            op.inputs;

        step.outputs =
            op.outputs;

        step.dependency_steps =
            find_dependencies(
                lowered,
                i
            );

        step.memory_offset =
            op.memory_offset;

        step.launch_config =
            launch_config_for(
                op
            );

        plan.steps.push_back(step);
    }

    return plan;
}