#include "runtime/graph_partitioner.h"

std::vector<Subgraph> GraphPartitioner::partition(
    const Graph& graph
) const {
    std::vector<Subgraph> subgraphs;

    if (graph.nodes.empty()) {
        return subgraphs;
    }

    Subgraph current;

    current.backend =
        scheduler.select_backend(graph.nodes[0]);

    for (const auto& node : graph.nodes) {
        BackendType backend =
            scheduler.select_backend(node);

        if (backend != current.backend) {
            subgraphs.push_back(current);

            current = {};

            current.backend = backend;
        }

        current.nodes.push_back(node);
    }

    if (!current.nodes.empty()) {
        subgraphs.push_back(current);
    }

    return subgraphs;
}