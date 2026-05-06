#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"
#include "runtime/thread_pool.h"
#include "utils/timer.h"

#include <iostream>
#include <vector>
#include <cmath>

bool close_enough(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
}

double benchmark_pool(
    const Tensor& A,
    const Tensor& B,
    const Tensor& Bias,
    Tensor& Out,
    ThreadPool& pool,
    int num_tasks,
    int runs
) {
    Timer timer;
    timer.start();

    for (int i = 0; i < runs; ++i) {
        fused_matmul_add_relu_threaded_pool(A, B, Bias, Out, pool, num_tasks);
    }

    return timer.stop_ms() / runs;
}

int main() {
    const int M = 512;
    const int K = 512;
    const int N = 512;

    Tensor A("A", {M, K});
    Tensor B("B", {K, N});
    Tensor Bias("Bias", {M, N});
    Tensor Baseline("Baseline", {M, N});
    Tensor Out("Out", {M, N});

    for (int i = 0; i < A.numel(); ++i) A.data[i] = 1.0f;
    for (int i = 0; i < B.numel(); ++i) B.data[i] = 0.01f;
    for (int i = 0; i < Bias.numel(); ++i) Bias.data[i] = 1.0f;

    const int warmup = 3;
    const int runs = 20;

    for (int i = 0; i < warmup; ++i) {
        fused_matmul_add_relu_optimized(A, B, Bias, Baseline);
    }

    Timer baseline_timer;
    baseline_timer.start();

    for (int i = 0; i < runs; ++i) {
        fused_matmul_add_relu_optimized(A, B, Bias, Baseline);
    }

    double baseline_avg_ms = baseline_timer.stop_ms() / runs;

    std::cout << "=== Persistent Thread Pool Benchmark ===\n";
    std::cout << "Shape: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "Runs: " << runs << "\n";
    std::cout << "Single-thread tiled avg latency: " << baseline_avg_ms << " ms\n";

    std::vector<int> worker_counts = {1, 2, 4, 8};
    std::vector<int> row_blocks = {32, 64, 128, 256};

    for (int row_block : row_blocks) {
        std::cout << "\nRow block size: " << row_block << "\n";

        for (int workers : worker_counts) {
            ThreadPool pool(workers);

            for (int i = 0; i < warmup; ++i) {
                fused_matmul_add_relu_threaded_pool(A, B, Bias, Out, pool, row_block);
            }

            double avg_ms = benchmark_pool(A, B, Bias, Out, pool, row_block, runs);
            double speedup = baseline_avg_ms / avg_ms;

            bool correct = true;
            for (int i = 0; i < Out.numel(); ++i) {
                if (!close_enough(Baseline.data[i], Out.data[i])) {
                    correct = false;
                    break;
                }
            }

            std::cout << workers << " workers avg latency: "
                    << avg_ms << " ms, speedup: "
                    << speedup << "x, correctness: "
                    << (correct ? "PASSED" : "FAILED") << "\n";
        }
    }
    std::cout << "Output[0]: " << Out.data[0] << "\n";

    return 0;
}