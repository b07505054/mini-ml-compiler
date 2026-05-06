#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"

#include <chrono>
#include <iostream>
#include <cmath>

int main() {
    const int N = 1 << 24;

    Tensor A("A", {N});
    Tensor B("B", {N});
    Tensor C_scalar("C_scalar", {N});
    Tensor C_avx("C_avx", {N});

    for (int i = 0; i < N; ++i) {
        A.data[i] = static_cast<float>(i % 100);
        B.data[i] = static_cast<float>((i * 2) % 100);
    }

    const int runs = 50;

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; ++r) {
        add(A, B, C_scalar);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    auto t3 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; ++r) {
        add_avx2(A, B, C_avx);
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    double scalar_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / runs;

    double avx_ms =
        std::chrono::duration<double, std::milli>(t4 - t3).count() / runs;

    bool correct = true;

    for (int i = 0; i < N; ++i) {
        if (std::fabs(C_scalar.data[i] - C_avx.data[i]) > 1e-5f) {
            correct = false;
            break;
        }
    }

    std::cout << "=== SIMD Add Benchmark ===\n";
    std::cout << "Tensor size: " << N << "\n";
    std::cout << "Scalar avg latency: " << scalar_ms << " ms\n";
    std::cout << "AVX2 avg latency: " << avx_ms << " ms\n";
    std::cout << "Speedup: " << scalar_ms / avx_ms << "x\n";
    std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";

    return 0;
}