#pragma once

#include "ir/graph.h"
#include "ir/node.h"

class Backend {
public:
    virtual ~Backend() = default;

    virtual void execute(Graph& graph, const Node& node) = 0;
    virtual const char* name() const = 0;
};