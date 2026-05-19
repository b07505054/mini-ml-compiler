#pragma once

#include "ir/node.h"

#include <string>

inline std::string op_type_to_string(OpType t) {
    switch (t) {
        case OpType::Input:
            return "Input";

        case OpType::MatMul:
            return "MatMul";

        case OpType::Add:
            return "Add";

        case OpType::ReLU:
            return "ReLU";

        case OpType::FusedMatMulBias:
            return "FusedMatMulBias";

        case OpType::FusedMatMulAddReLU:
            return "FusedMatMulAddReLU";

        case OpType::Attention:
            return "Attention";

        case OpType::CausalAttention:
            return "CausalAttention";

        case OpType::LayerNorm:
            return "LayerNorm";

        case OpType::FusedAttention:
            return "FusedAttention";

        case OpType::TiledAttention:
            return "TiledAttention";
    }

    return "Unknown";
}