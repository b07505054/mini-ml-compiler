#pragma once

#include <string>
#include <vector>

enum class OpType {
    Input,
    MatMul,
    Add,
    ReLU,
    FusedMatMulAddReLU,
    Attention,
    CausalAttention
};

struct Node {
    std::string name;
    OpType op;
    std::vector<int> inputs;
    std::vector<int> outputs;

    Node() = default;

    Node(std::string name, OpType op, std::vector<int> inputs, std::vector<int> outputs)
        : name(std::move(name)),
          op(op),
          inputs(std::move(inputs)),
          outputs(std::move(outputs)) {}
};