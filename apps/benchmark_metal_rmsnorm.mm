#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5f;
constexpr int kWarmupRuns = 10;
constexpr int kMeasuredRuns = 50;
constexpr NSUInteger kThreadsPerGroup = 256;

struct ShapeResult {
    int tokens;
    int hidden;
    double metal_p50_ms;
    double metal_p95_ms;
    double cpu_p50_ms;
    double cpu_p95_ms;
    double speedup;
    double max_abs_diff;
    double effective_bandwidth_gbps;
    bool correct;
};

std::filesystem::path find_repo_root() {
    std::filesystem::path current = std::filesystem::current_path();
    for (int i = 0; i < 4; ++i) {
        if (std::filesystem::exists(current / "metal" / "rmsnorm.metal")) {
            return current;
        }
        current = current.parent_path();
    }
    throw std::runtime_error("could not find repo root containing metal/rmsnorm.metal");
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(fraction * (values.size() - 1));
    return values[index];
}

void fill_inputs(std::vector<float>& input, std::vector<float>& weight) {
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 31) - 15) * 0.03125);
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = 0.75f + static_cast<float>(i % 17) * 0.01f;
    }
}

void rmsnorm_cpu(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::vector<float>& output,
    int tokens,
    int hidden
) {
    for (int token = 0; token < tokens; ++token) {
        const size_t base = static_cast<size_t>(token) * hidden;
        double square_sum = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const double value = input[base + i];
            square_sum += value * value;
        }
        const float inverse_rms =
            1.0f / std::sqrt(static_cast<float>(square_sum / hidden) + kEpsilon);
        for (int i = 0; i < hidden; ++i) {
            output[base + i] = input[base + i] * inverse_rms * weight[i];
        }
    }
}

double max_abs_diff(
    const std::vector<float>& expected,
    const float* actual
) {
    double result = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        result = std::max(result, std::fabs(
            static_cast<double>(expected[i]) - actual[i]
        ));
    }
    return result;
}

ShapeResult benchmark_shape(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    int tokens,
    int hidden
) {
    const size_t elements = static_cast<size_t>(tokens) * hidden;
    std::vector<float> input(elements);
    std::vector<float> weight(hidden);
    std::vector<float> cpu_output(elements);
    fill_inputs(input, weight);
    rmsnorm_cpu(input, weight, cpu_output, tokens, hidden);

    id<MTLBuffer> input_buffer =
        [device newBufferWithBytes:input.data()
                           length:elements * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer =
        [device newBufferWithBytes:weight.data()
                           length:weight.size() * sizeof(float)
                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [device newBufferWithLength:elements * sizeof(float)
                            options:MTLResourceStorageModeShared];

    const uint32_t hidden_value = static_cast<uint32_t>(hidden);
    const float epsilon = kEpsilon;
    auto dispatch = [&]() {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_value length:sizeof(hidden_value) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder setThreadgroupMemoryLength:kThreadsPerGroup * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    };

    for (int i = 0; i < kWarmupRuns; ++i) {
        dispatch();
    }

    std::vector<double> metal_latencies;
    for (int i = 0; i < kMeasuredRuns; ++i) {
        const auto start = std::chrono::steady_clock::now();
        dispatch();
        const auto end = std::chrono::steady_clock::now();
        metal_latencies.push_back(
            std::chrono::duration<double, std::milli>(end - start).count()
        );
    }

    std::vector<double> cpu_latencies;
    for (int i = 0; i < kWarmupRuns + kMeasuredRuns; ++i) {
        const auto start = std::chrono::steady_clock::now();
        rmsnorm_cpu(input, weight, cpu_output, tokens, hidden);
        const auto end = std::chrono::steady_clock::now();
        if (i >= kWarmupRuns) {
            cpu_latencies.push_back(
                std::chrono::duration<double, std::milli>(end - start).count()
            );
        }
    }

    const double metal_p50 = percentile(metal_latencies, 0.50);
    const double metal_p95 = percentile(metal_latencies, 0.95);
    const double cpu_p50 = percentile(cpu_latencies, 0.50);
    const double cpu_p95 = percentile(cpu_latencies, 0.95);
    const double diff = max_abs_diff(
        cpu_output,
        static_cast<const float*>([output_buffer contents])
    );
    const double bytes =
        static_cast<double>((2 * elements + hidden) * sizeof(float));
    const double bandwidth = bytes / (metal_p50 * 1.0e6);

    return {
        tokens,
        hidden,
        metal_p50,
        metal_p95,
        cpu_p50,
        cpu_p95,
        cpu_p50 / metal_p50,
        diff,
        bandwidth,
        diff <= 1.0e-4,
    };
}

void write_report(
    const std::filesystem::path& path,
    const std::string& device_name,
    const std::vector<ShapeResult>& results
) {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"artifact_type\": \"runtime_kernel_profile\",\n";
    out << "  \"source\": \"apps/benchmark_metal_rmsnorm.mm\",\n";
    out << "  \"profile_status\": \"measured\",\n";
    out << "  \"environment\": {\n";
    out << "    \"device\": \"" << device_name << "\",\n";
    out << "    \"backend\": \"Metal\",\n";
    out << "    \"dtype\": \"f32\",\n";
    out << "    \"warmup_runs\": " << kWarmupRuns << ",\n";
    out << "    \"timed_runs\": " << kMeasuredRuns << "\n";
    out << "  },\n";
    out << "  \"kernel_benchmarks\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        out << "    {\n";
        out << "      \"fusion_candidate\": \"rmsnorm\",\n";
        out << "      \"representative_shape\": {\"tokens\": " << result.tokens
            << ", \"hidden\": " << result.hidden << ", \"dtype\": \"f32\"},\n";
        out << "      \"custom_kernel\": \"fused_rmsnorm_metal\",\n";
        out << "      \"custom_backend\": \"Metal\",\n";
        out << "      \"fallback_kernel\": \"cpu_rmsnorm\",\n";
        out << "      \"fallback_backend\": \"CPU\",\n";
        out << "      \"custom_latency_ms\": " << result.metal_p50_ms << ",\n";
        out << "      \"custom_p95_latency_ms\": " << result.metal_p95_ms << ",\n";
        out << "      \"fallback_latency_ms\": " << result.cpu_p50_ms << ",\n";
        out << "      \"fallback_p95_latency_ms\": " << result.cpu_p95_ms << ",\n";
        out << "      \"speedup\": " << result.speedup << ",\n";
        out << "      \"max_abs_diff\": " << result.max_abs_diff << ",\n";
        out << "      \"effective_bandwidth_gbps\": "
            << result.effective_bandwidth_gbps << ",\n";
        out << "      \"correct\": " << (result.correct ? "true" : "false") << ",\n";
        out << "      \"selection_ready\": "
            << (result.correct && result.speedup > 1.0 ? "true" : "false") << "\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal device unavailable\n";
            return 1;
        }

        const std::filesystem::path repo_root = find_repo_root();
        const std::filesystem::path shader_path = repo_root / "metal" / "rmsnorm.metal";
        NSError* error = nil;
        NSString* source = [NSString
            stringWithContentsOfFile:[NSString stringWithUTF8String:shader_path.c_str()]
            encoding:NSUTF8StringEncoding
            error:&error];
        if (!source) {
            std::cerr << "Failed to load " << shader_path << "\n";
            return 1;
        }

        id<MTLLibrary> library =
            [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"rmsnorm_f32"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::cerr << "Failed to compile Metal RMSNorm: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        std::vector<ShapeResult> results;
        bool all_correct = true;
        for (int tokens : {1, 16, 128}) {
            for (int hidden : {768, 1024, 4096, 8192}) {
                ShapeResult result =
                    benchmark_shape(device, queue, pipeline, tokens, hidden);
                results.push_back(result);
                all_correct = all_correct && result.correct;
                std::cout
                    << tokens << "x" << hidden
                    << " Metal p50=" << result.metal_p50_ms << " ms"
                    << " CPU p50=" << result.cpu_p50_ms << " ms"
                    << " speedup=" << result.speedup << "x"
                    << " max_diff=" << result.max_abs_diff
                    << " " << (result.correct ? "PASS" : "FAIL")
                    << "\n";
            }
        }

        const std::filesystem::path report_path =
            repo_root / "trace" / "metal_rmsnorm_benchmark.json";
        write_report(report_path, [[device name] UTF8String], results);
        std::cout << "Wrote " << report_path << "\n";
        return all_correct ? 0 : 1;
    }
}
