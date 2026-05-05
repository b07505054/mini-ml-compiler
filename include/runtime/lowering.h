#pragma once

#include "ir/graph.h"
#include "runtime/execution_plan.h"

ExecutionPlan lower_to_execution_plan(const Graph& graph);