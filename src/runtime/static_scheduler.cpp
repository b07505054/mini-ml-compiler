#include "runtime/static_scheduler.h"

ExecutionSchedule build_static_schedule(
    const Graph& graph
) {
    ExecutionSchedule sched;

    int order = 0;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node =
            graph.nodes[i];

        ScheduleEntry entry;

        entry.op_id =
            static_cast<int>(i);

        entry.op_name =
            node.name;

        switch (node.op) {
            case OpType::MatMul:
                entry.op_type = "MatMul";
                entry.backend = "Metal";
                break;

            case OpType::Add:
                entry.op_type = "Add";
                entry.backend = "CPU";
                break;

            case OpType::ReLU:
                entry.op_type = "ReLU";
                entry.backend = "CPU";
                break;

            case OpType::FusedMatMulBias:
                entry.op_type = "FusedMatMulBias";
                entry.backend = "Metal";
                break;

            case OpType::FusedMatMulAddReLU:
                entry.op_type = "FusedMatMulAddReLU";
                entry.backend = "Metal";
                break;
            
            case OpType::Conv2D:
                entry.op_type = "Conv2D";
                entry.backend = "Metal";
                break;

            case OpType::BatchNorm:
                entry.op_type = "BatchNorm";
                entry.backend = "CPU";
                break;

            case OpType::MaxPool:
                entry.op_type = "MaxPool";
                entry.backend = "CPU";
                break;

            case OpType::Flatten:
                entry.op_type = "Flatten";
                entry.backend = "CPU";
                break;

            case OpType::Linear:
                entry.op_type = "Linear";
                entry.backend = "Metal";
                break;

            case OpType::FusedConvBatchNormReLU:
                entry.op_type = "FusedConvBatchNormReLU";
                entry.backend = "Metal";
                break;

            default:
                entry.op_type = "Unknown";
                entry.backend = "CPU";
                break;
        }

        entry.inputs =
            node.inputs;

        entry.outputs =
            node.outputs;

        entry.start_order =
            order++;

        if (!node.outputs.empty()) {
            int tid =
                node.outputs[0];

            entry.memory_offset =
                graph.tensors[tid]
                    .memory_offset;
        } else {
            entry.memory_offset = -1;
        }

        sched.entries.push_back(entry);
    }

    return sched;
}