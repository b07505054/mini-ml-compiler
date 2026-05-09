#include "runtime/memory_planner.h"

#include <iostream>
#include <algorithm>
#include <vector>

struct ReuseEvent {
    std::string current;
    std::string reused_from;
    int offset;
};

static int tensor_size(const Tensor& tensor) {
    int size = tensor.numel();
    return size > 0 ? size : 1;
}

void MemoryPlanner::plan(Graph& graph) {
    for (auto& tensor : graph.tensors) {
        tensor.first_use = -1;
        tensor.last_use = -1;
        tensor.memory_offset = -1;
    }

    for (int node_idx = 0; node_idx < static_cast<int>(graph.nodes.size()); ++node_idx) {
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

    int next_offset = 0;
    int naive_memory = 0;

    std::vector<ReuseEvent> reuse_events;

    for (auto& tensor : graph.tensors) {
        naive_memory += tensor_size(tensor);
    }

    for (int i = 0; i < static_cast<int>(graph.tensors.size()); ++i) {
        auto& current = graph.tensors[i];

        bool reused = false;

        for (int j = 0; j < i; ++j) {
            auto& previous = graph.tensors[j];

            bool can_reuse =
                !previous.persistent &&
                !current.persistent &&
                previous.last_use < current.first_use &&
                tensor_size(previous) >= tensor_size(current);

            if (can_reuse) {
                current.memory_offset = previous.memory_offset;
                reused = true;

                reuse_events.push_back({
                    current.name,
                    previous.name,
                    current.memory_offset
                });

                break;
            }
        }

        if (!reused) {
            current.memory_offset = next_offset;
            next_offset += tensor_size(current);
        }

        if (current.data.empty() && current.numel() > 0) {
            current.data.resize(current.numel(), 0.0f);
        }
    }

    int planned_memory = next_offset;
    int saved_memory = naive_memory - planned_memory;

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
            << " persistent="
            << (tensor.persistent ? "true" : "false")
            << " size="
            << tensor_size(tensor)
            << "\n";
    }

    std::cout << "[MemoryPlanner] Reuse events\n";

    if (reuse_events.empty()) {
        std::cout << "  none\n";
    } else {
        for (const auto& event : reuse_events) {
            std::cout
                << "  "
                << event.current
                << " reuses buffer from "
                << event.reused_from
                << " at offset "
                << event.offset
                << "\n";
        }
    }

    std::cout
        << "[MemoryPlanner] Naive memory: "
        << naive_memory
        << " float elements\n";

    std::cout
        << "[MemoryPlanner] Planned peak memory: "
        << planned_memory
        << " float elements\n";

    std::cout
        << "[MemoryPlanner] Saved memory: "
        << saved_memory
        << " float elements\n";
}