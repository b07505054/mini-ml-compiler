#include "ir/tensor.h"
#include "runtime/gpu_sim.h"
#include "utils/timer.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

int main() {
    const int M = 256;
    const int K = 256;
    const int N = 256;
    const int TILE = 16;

    Tensor A("A", {M, K});
    Tensor B("B", {K, N});
    Tensor C("C", {M, N});

    for (int i = 0; i < A.numel(); ++i) A.data[i] = 1.0f;
    for (int i = 0; i < B.numel(); ++i) B.data[i] = 0.01f;

    Dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    Dim3 block(TILE, TILE);

    auto kernel = [&](int bx, int by, int tx, int ty) {
        int row = by * TILE + ty;
        int col = bx * TILE + tx;

        if (row >= M || col >= N) return;

        float sum = 0.0f;

        // Simulated tiled GPU execution:
        // each block computes one TILE x TILE output tile.
        for (int tile_k = 0; tile_k < K; tile_k += TILE) {
            std::vector<float> local_a(TILE);
            std::vector<float> local_b(TILE);

            for (int kk = 0; kk < TILE; ++kk) {
                int k = tile_k + kk;

                local_a[kk] = (row < M && k < K)
                    ? A.data[row * K + k]
                    : 0.0f;

                local_b[kk] = (k < K && col < N)
                    ? B.data[k * N + col]
                    : 0.0f;
            }

            for (int kk = 0; kk < TILE; ++kk) {
                sum += local_a[kk] * local_b[kk];
            }
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

    std::cout << "=== GPU-style Tiled Execution (Simulated) ===\n";
    std::cout << "Shape: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "Tile size: " << TILE << "x" << TILE << "\n";
    std::cout << "Average latency: " << avg_ms << " ms\n";
    std::cout << "Output[0]: " << C.data[0] << "\n";

    return 0;
}