#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"

#include <chrono>
#include <iostream>
#include <random>
#include <cmath>

int main() {
    std::cout << "Starting AVX2 MatMul benchmark...\n";

    constexpr int SIZE = 128;
    constexpr int RUNS = 5;

    Tensor A("A", {SIZE, SIZE});
    Tensor B("B", {SIZE, SIZE});
    Tensor C1("C1", {SIZE, SIZE});
    Tensor C2("C2", {SIZE, SIZE});

    A.data.resize(SIZE * SIZE);
    B.data.resize(SIZE * SIZE);
    C1.data.assign(SIZE * SIZE, 0.0f);
    C2.data.assign(SIZE * SIZE, 0.0f);
    for (int i = 0; i < SIZE * SIZE; ++i) {
        A.data[i] = 1.0f;
        B.data[i] = 0.01f;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < RUNS; ++r) {
        matmul(A, B, C1);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < RUNS; ++r) {
        matmul_avx2(A, B, C2);
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    double scalar_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / RUNS;

    double avx2_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count() / RUNS;

    bool correct = true;
    for (size_t i = 0; i < C1.data.size(); ++i) {
        if (std::fabs(C1.data[i] - C2.data[i]) > 1e-3f) {
            correct = false;
            break;
        }
    }

    std::cout << "=== AVX2 MatMul Benchmark ===\n";
    std::cout << "Shape: " << SIZE << "x" << SIZE << "\n";
    std::cout << "Scalar avg latency: " << scalar_ms << " ms\n";
    std::cout << "AVX2 avg latency: " << avx2_ms << " ms\n";
    std::cout << "Speedup: " << scalar_ms / avx2_ms << "x\n";
    std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "Output[0]: " << C2.data[0] << "\n";

    return 0;
}