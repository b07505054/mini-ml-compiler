#pragma once

#include "ir/node.h"

#include <vector>

struct ExecutionPlan {
    std::vector<Node> ordered_nodes;
};