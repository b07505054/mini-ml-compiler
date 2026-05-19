#pragma once

#include "ir/graph.h"

#include <string>

class CompilerPass {
public:
    virtual ~CompilerPass() = default;

    virtual const char* name() const = 0;

    virtual void run(Graph& graph) = 0;
};