#include "runtime/graph_lowerer.h"

#include "ir/op_type_utils.h"

std::string GraphLowerer::lowered_type_for_op(
    OpType op
) const {
    switch (op) {
        case OpType::MatMul:
            return "LoweredMatMul";

        case OpType::Add:
            return "LoweredElementwiseAdd";

        case OpType::ReLU:
            return "LoweredElementwiseReLU";

        case OpType::FusedMatMulBias:
            return "LoweredFusedMatMulBias";

        case OpType::FusedMatMulAddReLU:
            return "LoweredFusedMatMulAddReLU";

        case OpType::Attention:
            return "LoweredAttention";

        case OpType::CausalAttention:
            return "LoweredCausalAttention";

        case OpType::FusedAttention:
            return "LoweredFusedAttention";

        case OpType::TiledAttention:
            return "LoweredTiledAttention";

        case OpType::Conv2D:
            return "LoweredConv2D";

        case OpType::BatchNorm:
            return "LoweredBatchNorm";

        case OpType::MaxPool:
            return "LoweredMaxPool";

        case OpType::Flatten:
            return "LoweredFlatten";

        case OpType::Linear:
            return "LoweredLinear";

        case OpType::FusedConvBatchNormReLU:
            return "LoweredFusedConvBatchNormReLU";

        case OpType::Embedding:
            return "LoweredEmbedding";

        case OpType::RMSNorm:
            return "LoweredRMSNorm";

        case OpType::QKVProjection:
            return "LoweredQKVProjection";

        case OpType::KVCacheWrite:
            return "LoweredKVCacheWrite";

        case OpType::KVCacheRead:
            return "LoweredKVCacheRead";

        case OpType::MLP:
            return "LoweredMLP";

        default:
            return "LoweredGenericOp";
    }
}

std::string GraphLowerer::backend_for_op(
    OpType op
) const {
    if (
        op == OpType::MatMul ||
        op == OpType::Linear ||
        op == OpType::Conv2D ||
        op == OpType::FusedMatMulBias ||
        op == OpType::FusedMatMulAddReLU ||
        op == OpType::FusedConvBatchNormReLU ||
        op == OpType::FusedAttention ||
        op == OpType::TiledAttention ||
        op == OpType::QKVProjection ||
        op == OpType::MLP
    ) {
        return "Metal";
    }

    return "CPU";
}

LoweredGraph GraphLowerer::lower(
    const Graph& graph
) const {
    LoweredGraph lowered;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node =
            graph.nodes[i];

        LoweredOp op;

        op.op_id =
            static_cast<int>(i);

        op.source_op_name =
            node.name;

        op.lowered_op_type =
            lowered_type_for_op(
                node.op
            );

        if (node.assigned_backend == BackendKind::Metal) {
            op.backend = "Metal";
        } else if (node.assigned_backend == BackendKind::CPU) {
            op.backend = "CPU";
        } else {
            op.backend = backend_for_op(node.op);
        }

        op.inputs =
            node.inputs;

        op.outputs =
            node.outputs;

        if (!node.outputs.empty()) {
            int output_id =
                node.outputs[0];

            op.memory_offset =
                graph.tensors.at(
                    static_cast<size_t>(output_id)
                ).memory_offset;
        } else {
            op.memory_offset = -1;
        }

        lowered.ops.push_back(op);
    }

    return lowered;
}