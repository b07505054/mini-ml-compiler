#pragma once

#include "ir/graph.h"

#include <string>

class MLIREmitter {
public:
    std::string emit(const Graph& graph) const;
};