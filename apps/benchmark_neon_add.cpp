#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"

#include <chrono>
#include <iostream>

int main() {

    constexpr int SIZE = 1 << 24;

    Tensor A("A", {SIZE});
    Tensor B("B", {SIZE});
    Tensor C("C", {SIZE});

    A.data.resize(SIZE);
    B.data.resize(SIZE);

    for (int i = 0; i < SIZE; ++i) {
        A.data[i] = 1.0f;
        B.data[i] = 2.0f;
    }

    auto start =
        std::chrono::high_resolution_clock::now();

    add_neon(A, B, C);

    auto end =
        std::chrono::high_resolution_clock::now();

    double latency =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

#ifdef __ARM_NEON
    const char* backend = "ARM NEON";
#else
    const char* backend = "Fallback Scalar";
#endif

    std::cout << "=== NEON Vector Add Benchmark ===\n";

    std::cout
        << "Backend: "
        << backend
        << "\n";

    std::cout
        << "Latency: "
        << latency
        << " ms\n";

    std::cout
        << "Output[0]: "
        << C.data[0]
        << "\n";

    return 0;
}