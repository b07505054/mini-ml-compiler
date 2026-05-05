#include "runtime/memory_planner.h"

#include <iostream>

void MemoryPlanner::plan(Graph& graph) {
    size_t total_elements = 0;

    for (auto& tensor : graph.tensors) {
        if (tensor.data.empty() && tensor.numel() > 0) {
            total_elements += tensor.numel();
        }
    }

    arena.resize(total_elements);

    size_t offset = 0;

    for (auto& tensor : graph.tensors) {
        if (tensor.data.empty() && tensor.numel() > 0) {
            tensor.data.assign(arena.begin() + offset,
                               arena.begin() + offset + tensor.numel());

            offset += tensor.numel();
        }
    }

    std::cout << "[MemoryPlanner] Planned arena memory for "
              << total_elements << " float elements\n";
}