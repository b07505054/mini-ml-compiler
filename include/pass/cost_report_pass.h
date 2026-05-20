#pragma once

#include "ir/graph.h"
#include "pass/cost_report.h"

class CostReportPass {
public:
    CostReport run(
        const Graph& graph
    ) const;
};