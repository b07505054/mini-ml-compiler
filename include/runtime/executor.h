#pragma once

#include "ir/graph.h"
#include "runtime/execution_plan.h"
#include "runtime/op_registry.h"

class Executor {
public:
    Executor();

    void run(Graph& graph, const ExecutionPlan& plan, bool verbose = true);

private:
    OpRegistry registry;
};