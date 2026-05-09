#include "runtime/backend_scheduler.h"

BackendType BackendScheduler::select_backend(const Node& node) const {
    switch (node.op) {
        case OpType::MatMul:
        case OpType::FusedMatMulAddReLU:
        case OpType::Attention:
        case OpType::CausalAttention:
            return BackendType::MockGPU;

        case OpType::Add:
        case OpType::ReLU:
        default:
            return BackendType::CPU;
    }
}