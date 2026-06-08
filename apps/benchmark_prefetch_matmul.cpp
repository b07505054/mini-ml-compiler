#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"
#include "utils/timer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    double rank = (values.size() - 1) * p;
    auto lo = static_cast<size_t>(rank);
    auto hi = std::min(lo + 1, values.size() - 1);
    double frac = rank - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * frac;
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

bool close_enough(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        return false;
    }
    for (size_t i = 0; i < a.data.size(); ++i) {
        if (std::fabs(a.data[i] - b.data[i]) > 1e-4f) {
            return false;
        }
    }
    return true;
}

void fill_inputs(Tensor& a, Tensor& b, Tensor& bias) {
    for (int i = 0; i < a.numel(); ++i) {
        a.data[i] = static_cast<float>((i % 17) - 8) * 0.01f;
    }
    for (int i = 0; i < b.numel(); ++i) {
        b.data[i] = static_cast<float>((i % 13) + 1) * 0.015f;
    }
    for (int i = 0; i < bias.numel(); ++i) {
        bias.data[i] = (i % 7 == 0) ? -0.2f : 0.05f;
    }
}

template <typename Fn>
std::vector<double> measure(Fn fn, int warmup, int runs) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        Timer timer;
        timer.start();
        fn();
        times.push_back(timer.stop_ms());
    }
    return times;
}

std::filesystem::path find_trace_dir() {
    for (const std::filesystem::path& dir : {"trace", "../trace"}) {
        if (std::filesystem::exists(dir) || std::filesystem::exists(dir.parent_path())) {
            std::filesystem::create_directories(dir);
            return dir;
        }
    }
    std::filesystem::create_directories("trace");
    return "trace";
}

void write_markdown(
    const std::filesystem::path& path,
    double baseline_p95,
    double prefetch_p95,
    double speedup,
    bool correct,
    bool selected
) {
    std::ofstream out(path);
    out << "# CPU Software Prefetch MatMul Benchmark\n\n";
    out << "Status: `measured`\n\n";
    out << "- Input: `HIR fused MatMul-Bias-ReLU CPU backend workload`\n";
    out << "- Decision: `profile-guided choice between tiled CPU kernel and prefetch tiled CPU kernel`\n";
    out << "- Metric: `p50/p95 latency, speedup, correctness, estimated bytes moved`\n\n";
    out << "| Kernel | p95 ms |\n";
    out << "|---|---:|\n";
    out << "| fused_matmul_add_relu_optimized | " << baseline_p95 << " |\n";
    out << "| fused_matmul_add_relu_prefetch | " << prefetch_p95 << " |\n\n";
    out << "- Speedup: `" << speedup << "x`\n";
    out << "- Correct: `" << (correct ? "true" : "false") << "`\n";
    out << "- Selected prefetch candidate: `" << (selected ? "true" : "false") << "`\n";
}

} // namespace

int main() {
    const int M = 256;
    const int K = 256;
    const int N = 256;
    const int warmup = 10;
    const int runs = 50;
    const int prefetch_distance = 16;

    Tensor A("A", {M, K});
    Tensor B("B", {K, N});
    Tensor Bias("Bias", {M, N});
    Tensor Baseline("Baseline", {M, N});
    Tensor Prefetch("Prefetch", {M, N});

    fill_inputs(A, B, Bias);

    fused_matmul_add_relu_optimized(A, B, Bias, Baseline);
    fused_matmul_add_relu_prefetch(A, B, Bias, Prefetch, prefetch_distance);
    bool correct = close_enough(Baseline, Prefetch);

    auto baseline_times = measure(
        [&]() { fused_matmul_add_relu_optimized(A, B, Bias, Baseline); },
        warmup,
        runs
    );
    auto prefetch_times = measure(
        [&]() { fused_matmul_add_relu_prefetch(A, B, Bias, Prefetch, prefetch_distance); },
        warmup,
        runs
    );

    double baseline_mean = mean(baseline_times);
    double prefetch_mean = mean(prefetch_times);
    double baseline_p50 = percentile(baseline_times, 0.50);
    double baseline_p95 = percentile(baseline_times, 0.95);
    double prefetch_p50 = percentile(prefetch_times, 0.50);
    double prefetch_p95 = percentile(prefetch_times, 0.95);
    double speedup = baseline_mean / std::max(prefetch_mean, 1e-9);
    bool selected = correct && prefetch_p95 < baseline_p95;

    std::filesystem::path trace_dir = find_trace_dir();
    std::filesystem::path json_path = trace_dir / "prefetch_matmul_benchmark.json";
    std::filesystem::path md_path = trace_dir / "prefetch_matmul_benchmark.md";

    size_t estimated_bytes = static_cast<size_t>(M) * K * sizeof(float)
        + static_cast<size_t>(K) * N * sizeof(float)
        + static_cast<size_t>(M) * N * sizeof(float) * 2;

    std::ofstream out(json_path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"artifact_type\": \"prefetch_kernel_benchmark\",\n";
    out << "  \"technology\": \"cpu_software_prefetching\",\n";
    out << "  \"status\": \"measured\",\n";
    out << "  \"input\": \"HIR fused MatMul-Bias-ReLU CPU backend workload\",\n";
    out << "  \"decision\": \"profile-guided choice between tiled CPU kernel and prefetch tiled CPU kernel\",\n";
    out << "  \"metric\": \"p50/p95 latency, speedup, correctness, estimated bytes moved\",\n";
    out << "  \"fusion_candidate\": \"matmul_bias_relu_prefetch\",\n";
    out << "  \"candidate_kernel\": \"fused_matmul_add_relu_prefetch\",\n";
    out << "  \"fallback_kernel\": \"fused_matmul_add_relu_optimized\",\n";
    out << "  \"custom_kernel\": \"fused_matmul_add_relu_prefetch\",\n";
    out << "  \"custom_backend\": \"CPU\",\n";
    out << "  \"shape\": {\"m\": " << M << ", \"k\": " << K << ", \"n\": " << N << ", \"dtype\": \"f32\"},\n";
    out << "  \"prefetch_distance\": " << prefetch_distance << ",\n";
    out << "  \"estimated_bytes_moved\": " << estimated_bytes << ",\n";
    out << "  \"baseline_mean_ms\": " << baseline_mean << ",\n";
    out << "  \"baseline_p50_ms\": " << baseline_p50 << ",\n";
    out << "  \"baseline_p95_ms\": " << baseline_p95 << ",\n";
    out << "  \"prefetch_mean_ms\": " << prefetch_mean << ",\n";
    out << "  \"prefetch_p50_ms\": " << prefetch_p50 << ",\n";
    out << "  \"prefetch_p95_ms\": " << prefetch_p95 << ",\n";
    out << "  \"custom_latency_ms\": " << prefetch_mean << ",\n";
    out << "  \"fallback_latency_ms\": " << baseline_mean << ",\n";
    out << "  \"speedup\": " << speedup << ",\n";
    out << "  \"correct\": " << (correct ? "true" : "false") << ",\n";
    out << "  \"selection_ready\": " << (selected ? "true" : "false") << ",\n";
    out << "  \"selection_reason\": \"" << (selected ? "prefetch_p95_faster" : "fallback_p95_not_improved") << "\",\n";
    out << "  \"kernel_benchmarks\": [\n";
    out << "    {\n";
    out << "      \"fusion_candidate\": \"matmul_bias_relu_prefetch\",\n";
    out << "      \"custom_kernel\": \"fused_matmul_add_relu_prefetch\",\n";
    out << "      \"custom_backend\": \"CPU\",\n";
    out << "      \"fallback_kernel\": \"fused_matmul_add_relu_optimized\",\n";
    out << "      \"custom_latency_ms\": " << prefetch_mean << ",\n";
    out << "      \"fallback_latency_ms\": " << baseline_mean << ",\n";
    out << "      \"speedup\": " << speedup << ",\n";
    out << "      \"correct\": " << (correct ? "true" : "false") << ",\n";
    out << "      \"selection_ready\": " << (selected ? "true" : "false") << ",\n";
    out << "      \"shape\": {\"m\": " << M << ", \"k\": " << K << ", \"n\": " << N << ", \"dtype\": \"f32\"},\n";
    out << "      \"prefetch_distance\": " << prefetch_distance << "\n";
    out << "    }\n";
    out << "  ]\n";
    out << "}\n";

    write_markdown(md_path, baseline_p95, prefetch_p95, speedup, correct, selected);

    std::cout << "=== CPU Software Prefetch MatMul Benchmark ===\n";
    std::cout << "Shape: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "Baseline p95: " << baseline_p95 << " ms\n";
    std::cout << "Prefetch p95: " << prefetch_p95 << " ms\n";
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "Wrote " << json_path << "\n";

    return correct ? 0 : 1;
}
