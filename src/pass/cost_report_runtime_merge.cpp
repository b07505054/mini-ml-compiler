#include "pass/cost_report_runtime_merge.h"

#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string extract_string_value(
    const std::string& line
) {
    auto first =
        line.find("\"", line.find(":"));

    auto second =
        line.find("\"", first + 1);

    if (
        first == std::string::npos ||
        second == std::string::npos
    ) {
        return "";
    }

    return line.substr(
        first + 1,
        second - first - 1
    );
}

double extract_double_value(
    const std::string& line
) {
    auto colon =
        line.find(":");

    if (colon == std::string::npos) {
        return 0.0;
    }

    std::string value =
        line.substr(colon + 1);

    if (!value.empty() && value.back() == ',') {
        value.pop_back();
    }

    return std::stod(value);
}

} // namespace

void merge_runtime_trace_into_cost_report(
    CostReport& report,
    const std::string& runtime_trace_path
) {
    std::ifstream in(runtime_trace_path);

    if (!in) {
        return;
    }

    std::string line;

    std::string op_name;
    std::string backend;
    double latency = -1.0;

    while (std::getline(in, line)) {
        if (
            line.find("\"op_name\"") !=
            std::string::npos
        ) {
            op_name =
                extract_string_value(line);
        }

        if (
            line.find("\"backend\"") !=
            std::string::npos
        ) {
            backend =
                extract_string_value(line);
        }

        if (
            line.find("\"latency_ms\"") !=
            std::string::npos
        ) {
            latency =
                extract_double_value(line);

            for (auto& entry : report.entries) {
                if (entry.op_name == op_name) {
                    entry.actual_backend =
                        backend;

                    entry.actual_latency_ms =
                        latency;
                }
            }

            op_name.clear();
            backend.clear();
            latency = -1.0;
        }
    }
}