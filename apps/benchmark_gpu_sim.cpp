#include "ir/tensor.h"
#include "runtime/gpu_sim.h"
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
    Tensor C("C", {M, N});

    for (int i = 0; i < A.numel(); ++i) A.data[i] = 1.0f;
    for (int i = 0; i < B.numel(); ++i) B.data[i] = 0.01f;

    int TILE = 16;

    Dim3 grid(N / TILE, M / TILE);
    Dim3 block(TILE, TILE);

    auto kernel = [&](int bx, int by, int tx, int ty) {

        int row = by * TILE + ty;
        int col = bx * TILE + tx;

        if (row >= M || col >= N) return;

        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A.data[row * K + k] * B.data[k * N + col];
        }

        C.data[row * N + col] = sum;
    };

    const int runs = 50;

    Timer timer;
    timer.start();

    for (int i = 0; i < runs; ++i) {
        GPUSimulator::launch(grid, block, kernel);
    }

    double avg_ms = timer.stop_ms() / runs;

    std::cout << "=== GPU-style Execution (Simulated) ===\n";
    std::cout << "Average latency: " << avg_ms << " ms\n";
    std::cout << "Output[0]: " << C.data[0] << "\n";

    return 0;
}