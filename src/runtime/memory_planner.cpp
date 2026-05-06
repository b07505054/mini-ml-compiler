#include "runtime/memory_planner.h"

#include <iostream>
#include <algorithm>

void MemoryPlanner::plan(Graph& graph) {

    // initialize
    for (auto& tensor : graph.tensors) {
        tensor.first_use = -1;
        tensor.last_use = -1;
    }

    // lifetime analysis
    for (int node_idx = 0; node_idx < graph.nodes.size(); ++node_idx) {

        const auto& node = graph.nodes[node_idx];

        for (int tid : node.inputs) {

            auto& tensor = graph.get_tensor(tid);

            if (tensor.first_use == -1) {
                tensor.first_use = node_idx;
            }

            tensor.last_use = node_idx;
        }

        for (int tid : node.outputs) {

            auto& tensor = graph.get_tensor(tid);

            if (tensor.first_use == -1) {
                tensor.first_use = node_idx;
            }

            tensor.last_use = node_idx;
        }
    }

    // simple memory reuse
    int next_offset = 0;

    for (int i = 0; i < graph.tensors.size(); ++i) {
        auto& current = graph.tensors[i];

        bool reused = false;

        for (int j = 0; j < i; ++j) {
            auto& previous = graph.tensors[j];

            if (!previous.persistent &&
                !current.persistent &&
                previous.last_use < current.first_use) {

                current.memory_offset = previous.memory_offset;
                reused = true;
                break;
            }
        }

        if (!reused) {
            current.memory_offset = next_offset;

            int size = current.numel();
            if (size == 0) size = 1;

            next_offset += size;
        }

        if (current.data.empty() && current.numel() > 0) {
            current.data.resize(current.numel(), 0.0f);
        }
    }

    std::cout << "[MemoryPlanner] Tensor lifetime analysis\n";

    for (const auto& tensor : graph.tensors) {

        std::cout
            << "  "
            << tensor.name
            << " | first="
            << tensor.first_use
            << " last="
            << tensor.last_use
            << " offset="
            << tensor.memory_offset
            << "\n";
    }

    std::cout
        << "[MemoryPlanner] Planned arena memory: "
        << next_offset
        << " float elements\n";
}