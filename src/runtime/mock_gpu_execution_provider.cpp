#include "runtime/mock_gpu_execution_provider.h"

bool MockGPUExecutionProvider::can_execute(const Node& node) const {
    switch (node.op) {
        case OpType::MatMul:
        case OpType::FusedMatMulAddReLU:
        case OpType::Attention:
        case OpType::CausalAttention:
        case OpType::FusedAttention:
            return true;

        default:
            return false;
    }
}

BackendType MockGPUExecutionProvider::backend_type() const {
    return BackendType::MockGPU;
}

const char* MockGPUExecutionProvider::name() const {
    return "MockGPUExecutionProvider";
}