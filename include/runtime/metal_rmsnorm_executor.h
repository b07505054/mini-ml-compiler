#pragma once

#include <string>
#include <vector>

struct MetalRMSNormExecutionResult {
    std::string device;
    std::vector<float> output;
    std::vector<double> latencies_ms;
};

class MetalRMSNormExecutor {
public:
    explicit MetalRMSNormExecutor(const std::string& shader_path);
    ~MetalRMSNormExecutor();

    MetalRMSNormExecutor(const MetalRMSNormExecutor&) = delete;
    MetalRMSNormExecutor& operator=(const MetalRMSNormExecutor&) = delete;

    MetalRMSNormExecutionResult execute(
        const std::vector<float>& input,
        const std::vector<float>& weight,
        int tokens,
        int hidden,
        float epsilon,
        int warmup_runs,
        int timed_runs
    );

private:
    struct Impl;
    Impl* impl_;
};
