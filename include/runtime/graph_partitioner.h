#pragma once

#include "runtime/execution_plan.h"
#include "runtime/backend_scheduler.h"
#include "runtime/backend_type.h"

#include <vector>

struct GraphPartition {
    BackendType backend;
    std::vector<Node> nodes;
};

class GraphPartitioner {
public:
    std::vector<GraphPartition> partition(const ExecutionPlan& plan) const;

private:
    BackendScheduler scheduler;
};