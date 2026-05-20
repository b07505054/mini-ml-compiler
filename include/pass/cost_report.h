#pragma once

#include <string>
#include <vector>

struct CostReportEntry {
    std::string op_name;

    std::string op_type;

    std::string backend;

    size_t estimated_read_bytes;

    size_t estimated_write_bytes;

    size_t estimated_flops;

    double arithmetic_intensity;

    double estimated_kernel_launch_cost;

    double estimated_backend_switch_cost;

    std::string actual_backend;

    double actual_latency_ms;

    std::string fusion_note;
};

struct CostReport {
    std::vector<CostReportEntry> entries;

    void dump() const;

    void export_json(
        const std::string& path
    ) const;
};