#pragma once

#include "ir/graph.h"
#include "runtime/subgraph.h"
#include "runtime/provider_scheduler.h"

#include <vector>

class GraphPartitioner {
public:
    std::vector<Subgraph> partition(
        const Graph& graph
    ) const;

private:
    ProviderScheduler scheduler;
};