#pragma once

#include "pass/cost_report.h"

#include <string>

void merge_runtime_trace_into_cost_report(
    CostReport& report,
    const std::string& runtime_trace_path
);