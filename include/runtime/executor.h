#pragma once

#include "ir/graph.h"
#include "runtime/execution_plan.h"
#include "runtime/op_registry.h"
#include "utils/profiler.h"

class Executor {
public:
    Executor();

    void run(Graph& graph, const ExecutionPlan& plan, bool verbose = true, bool profile = false);

private:
    OpRegistry registry;
    Profiler profiler;
};