#pragma once

#include "ir/graph.h"
#include "ir/node.h"

#include <functional>
#include <unordered_map>
#include <stdexcept>

using KernelFn = std::function<void(Graph&, const Node&)>;

class OpRegistry {
public:
    void register_op(OpType op, KernelFn fn) {
        registry[op] = std::move(fn);
    }

    void dispatch(OpType op, Graph& graph, const Node& node) const {
        auto it = registry.find(op);

        if (it == registry.end()) {
            throw std::runtime_error("No kernel registered for op");
        }

        it->second(graph, node);
    }

private:
    std::unordered_map<OpType, KernelFn> registry;
};

OpRegistry create_default_registry();