#pragma once

#include "ir/node.h"
#include "runtime/backend_type.h"

#include <vector>

struct Subgraph {
    BackendType backend;

    std::vector<Node> nodes;
};