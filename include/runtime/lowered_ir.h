#pragma once

#include <string>
#include <vector>

struct LoweredOp {
    int op_id;

    std::string source_op_name;

    std::string lowered_op_type;

    std::string backend;

    std::vector<int> inputs;

    std::vector<int> outputs;

    int memory_offset;
};

struct LoweredGraph {
    std::vector<LoweredOp> ops;

    void dump() const;

    void export_json(
        const std::string& path
    ) const;
};