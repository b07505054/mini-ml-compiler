#include "runtime/graph_partitioner.h"

std::vector<GraphPartition> GraphPartitioner::partition(const ExecutionPlan& plan) const {
    std::vector<GraphPartition> partitions;

    for (const auto& node : plan.ordered_nodes) {
        BackendType backend = scheduler.select_backend(node);

        if (partitions.empty() || partitions.back().backend != backend) {
            GraphPartition p;
            p.backend = backend;
            p.nodes.push_back(node);
            partitions.push_back(p);
        } else {
            partitions.back().nodes.push_back(node);
        }
    }

    return partitions;
}