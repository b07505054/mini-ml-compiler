#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"

#include <chrono>
#include <iostream>

int main() {

    const int N = 1 << 24;

    Tensor input("input", {N});
    Tensor output1("output1", {N});
    Tensor output2("output2", {N});

    input.data.resize(N);

    for (int i = 0; i < N; ++i) {
        input.data[i] = (i % 17) - 8;
    }

    const int runs = 50;

    // baseline
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; ++r) {
        relu(input, output1);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // avx2
    auto t3 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; ++r) {
        relu_avx2(input, output2);
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    double baseline_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / runs;

    double avx_ms =
        std::chrono::duration<double, std::milli>(t4 - t3).count() / runs;

    bool correct = true;

    for (int i = 0; i < N; ++i) {
        if (output1.data[i] != output2.data[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "=== SIMD ReLU Benchmark ===\n";

    std::cout << "Tensor size: " << N << "\n";

    std::cout << "Baseline avg latency: "
              << baseline_ms
              << " ms\n";

    std::cout << "AVX2 avg latency: "
              << avx_ms
              << " ms\n";

    std::cout << "Speedup: "
              << baseline_ms / avx_ms
              << "x\n";

    std::cout << "Correctness: "
              << (correct ? "PASSED" : "FAILED")
              << "\n";

    return 0;
}