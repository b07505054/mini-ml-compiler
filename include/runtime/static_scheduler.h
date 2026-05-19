#pragma once

#include "ir/graph.h"
#include "runtime/execution_schedule.h"

ExecutionSchedule build_static_schedule(
    const Graph& graph
);