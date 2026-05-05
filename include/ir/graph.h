#pragma once

#include "ir/node.h"
#include "ir/tensor.h"

#include <vector>
#include <iostream>

class Graph {
public:
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;

    int add_tensor(const Tensor& tensor) {
        tensors.push_back(tensor);
        return static_cast<int>(tensors.size()) - 1;
    }

    void add_node(const Node& node) {
        nodes.push_back(node);
    }

    Tensor& get_tensor(int id) {
        return tensors.at(id);
    }

    const Tensor& get_tensor(int id) const {
        return tensors.at(id);
    }

    void dump() const {
        std::cout << "=== Graph IR ===\n";

        std::cout << "Tensors:\n";
        for (size_t i = 0; i < tensors.size(); ++i) {
            std::cout << "  [" << i << "] " << tensors[i].name << "\n";
        }

        std::cout << "Nodes:\n";
        for (const auto& node : nodes) {
            std::cout << "  " << node.name << "\n";
        }
    }
};