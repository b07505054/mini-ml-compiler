#include "pass/dead_node_elimination_pass.h"

#include <iostream>
#include <unordered_set>
#include <vector>
#include <algorithm>

void DeadNodeEliminationPass::run(Graph& graph) {
    std::unordered_set<int> used_tensors;

    // Treat tensor named "output" as graph output.
    for (int i = 0; i < static_cast<int>(graph.tensors.size()); ++i) {
        if (graph.tensors[i].name == "output") {
            used_tensors.insert(i);
        }
    }

    std::vector<Node> kept_nodes;

    for (int i = static_cast<int>(graph.nodes.size()) - 1; i >= 0; --i) {
        const auto& node = graph.nodes[i];

        bool needed = false;

        for (int out : node.outputs) {
            if (used_tensors.count(out)) {
                needed = true;
                break;
            }
        }

        if (needed) {
            kept_nodes.push_back(node);

            for (int in : node.inputs) {
                used_tensors.insert(in);
            }
        }
    }

    std::reverse(kept_nodes.begin(), kept_nodes.end());

    int removed =
        static_cast<int>(graph.nodes.size()) -
        static_cast<int>(kept_nodes.size());

    graph.nodes = kept_nodes;

    std::cout << "[DeadNodeEliminationPass] Removed "
              << removed
              << " dead nodes\n";
}