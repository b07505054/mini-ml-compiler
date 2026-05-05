#pragma once

#include "ir/graph.h"

class GraphVerifier {
public:
    bool verify(const Graph& graph) const;
};