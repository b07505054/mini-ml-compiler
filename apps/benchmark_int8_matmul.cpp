#include "ir/tensor.h"
#include "ir/quant_tensor.h"
#include "kernels/cpu_kernels.h"

#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>

int main() {
    constexpr int SIZE = 128;
    constexpr int RUNS = 10;

    Tensor A("A", {SIZE, SIZE});
    Tensor B("B", {SIZE, SIZE});
    Tensor C_float("C_float", {SIZE, SIZE});
    Tensor C_int8("C_int8", {SIZE, SIZE});

    for (int i = 0; i < A.numel(); ++i) {
        A.data[i] = 0.01f * static_cast<float>((i % 17) - 8);
        B.data[i] = 0.01f * static_cast<float>((i % 13) - 6);
    }

    float scale_a = 0.01f;
    float scale_b = 0.01f;

    QuantTensor A_q = quantize_tensor_symmetric(A, scale_a);
    QuantTensor B_q = quantize_tensor_symmetric(B, scale_b);

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < RUNS; ++r) {
        matmul(A, B, C_float);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < RUNS; ++r) {
        matmul_int8(A_q, B_q, C_int8);
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    double float_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / RUNS;

    double int8_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count() / RUNS;

    float max_abs_error = 0.0f;

    for (int i = 0; i < C_float.numel(); ++i) {
        max_abs_error = std::max(
            max_abs_error,
            std::fabs(C_float.data[i] - C_int8.data[i])
        );
    }

    std::cout << "=== INT8 MatMul Benchmark ===\n";
    std::cout << "Shape: " << SIZE << "x" << SIZE << "\n";
    std::cout << "Float MatMul avg latency: " << float_ms << " ms\n";
    std::cout << "INT8 MatMul avg latency: " << int8_ms << " ms\n";
    std::cout << "Speedup: " << float_ms / int8_ms << "x\n";
    std::cout << "Max abs error: " << max_abs_error << "\n";
    std::cout << "Output[0] float: " << C_float.data[0] << "\n";
    std::cout << "Output[0] int8: " << C_int8.data[0] << "\n";

    return 0;
}