#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"
#include "utils/timer.h"

#include <iostream>
#include <cmath>

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
}

int main() {
    const int M = 256;
    const int K = 256;
    const int N = 256;

    Tensor A("A", {M, K});
    Tensor B("B", {K, N});
    Tensor Bias("Bias", {M, N});
    Tensor OutBaseline("OutBaseline", {M, N});
    Tensor OutOptimized("OutOptimized", {M, N});

    for (int i = 0; i < A.numel(); ++i) A.data[i] = 1.0f;
    for (int i = 0; i < B.numel(); ++i) B.data[i] = 0.01f;
    for (int i = 0; i < Bias.numel(); ++i) Bias.data[i] = 1.0f;

    const int warmup = 5;
    const int runs = 50;

    for (int i = 0; i < warmup; ++i) {
        fused_matmul_add_relu_baseline(A, B, Bias, OutBaseline);
    }

    Timer timer;
    timer.start();

    for (int i = 0; i < runs; ++i) {
        fused_matmul_add_relu_baseline(A, B, Bias, OutBaseline);
    }

    double baseline_total_ms = timer.stop_ms();
    double baseline_avg_ms = baseline_total_ms / runs;

    for (int i = 0; i < warmup; ++i) {
        fused_matmul_add_relu_optimized(A, B, Bias, OutOptimized);
    }

    timer.start();

    for (int i = 0; i < runs; ++i) {
        fused_matmul_add_relu_optimized(A, B, Bias, OutOptimized);
    }

    double optimized_total_ms = timer.stop_ms();
    double optimized_avg_ms = optimized_total_ms / runs;

    bool correct = true;
    for (int i = 0; i < OutBaseline.numel(); ++i) {
        if (!close_enough(OutBaseline.data[i], OutOptimized.data[i])) {
            correct = false;
            break;
        }
    }

    double speedup = baseline_avg_ms / optimized_avg_ms;

    std::cout << "=== Kernel Benchmark ===\n";
    std::cout << "Shape: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "Runs: " << runs << "\n";
    std::cout << "Baseline avg latency: " << baseline_avg_ms << " ms\n";
    std::cout << "Optimized avg latency: " << optimized_avg_ms << " ms\n";
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "Output[0]: " << OutOptimized.data[0] << "\n";

    return correct ? 0 : 1;
}