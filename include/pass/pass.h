#pragma once

#include "ir/graph.h"

class Pass {
public:
    virtual ~Pass() = default;
    virtual void run(Graph& graph) = 0;
};