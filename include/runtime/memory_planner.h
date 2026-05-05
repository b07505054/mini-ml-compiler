#pragma once

#include "ir/graph.h"

#include <vector>
#include <cstddef>

class MemoryPlanner {
public:
    void plan(Graph& graph);

private:
    std::vector<float> arena;
};