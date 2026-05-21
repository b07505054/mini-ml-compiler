#pragma once

#include "ir/graph.h"
#include "runtime/lowered_ir.h"

class GraphLowerer {
public:
    LoweredGraph lower(
        const Graph& graph
    ) const;

private:
    std::string lowered_type_for_op(
        OpType op
    ) const;

    std::string backend_for_op(
        OpType op
    ) const;
};