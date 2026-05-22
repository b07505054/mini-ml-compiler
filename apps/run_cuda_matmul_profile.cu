#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at "                               \
                      << __FILE__                             \
                      << ":"                                  \
                      << __LINE__                             \
                      << std::endl;                           \
            std::exit(1);                                     \
        }                                                     \
    } while (0)

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (
        row < M &&
        col < N
    ) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum +=
                A[row * K + k]
                * B[k * N + col];
        }

        C[row * N + col] =
            sum;
    }
}

double percentile(
    std::vector<float> v,
    double p
) {
    std::sort(
        v.begin(),
        v.end()
    );

    size_t idx =
        static_cast<size_t>(
            p * (v.size() - 1)
        );

    return v[idx];
}

int main() {
    constexpr int M = 1024;
    constexpr int K = 1024;
    constexpr int N = 1024;

    constexpr int warmup_runs = 5;
    constexpr int measured_runs = 50;

    size_t bytesA =
        M * K * sizeof(float);

    size_t bytesB =
        K * N * sizeof(float);

    size_t bytesC =
        M * N * sizeof(float);

    std::vector<float> hA(M * K, 1.0f);
    std::vector<float> hB(K * N, 1.0f);
    std::vector<float> hC(M * N, 0.0f);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(
        cudaMemcpy(
            dA,
            hA.data(),
            bytesA,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            dB,
            hB.data(),
            bytesB,
            cudaMemcpyHostToDevice
        )
    );

    dim3 threads(
        16,
        16
    );

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> latencies_ms;

    for (int i = 0;
         i < warmup_runs + measured_runs;
         ++i) {

        CUDA_CHECK(
            cudaEventRecord(start)
        );

        matmul_kernel<<<blocks, threads>>>(
            dA,
            dB,
            dC,
            M,
            N,
            K
        );

        CUDA_CHECK(
            cudaEventRecord(stop)
        );

        CUDA_CHECK(
            cudaEventSynchronize(stop)
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        float ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            )
        );

        if (i >= warmup_runs) {
            latencies_ms.push_back(ms);
        }
    }

    CUDA_CHECK(
        cudaMemcpy(
            hC.data(),
            dC,
            bytesC,
            cudaMemcpyDeviceToHost
        )
    );

    bool correct = true;

    for (int i = 0; i < 16; ++i) {
        if (std::abs(hC[i] - 1024.0f) > 1e-3f) {
            correct = false;
            break;
        }
    }

    double avg =
        std::accumulate(
            latencies_ms.begin(),
            latencies_ms.end(),
            0.0
        ) / latencies_ms.size();

    double p50 =
        percentile(
            latencies_ms,
            0.50
        );

    double p95 =
        percentile(
            latencies_ms,
            0.95
        );

    double p99 =
        percentile(
            latencies_ms,
            0.99
        );

    double flops =
        2.0
        * M
        * N
        * K;

    double gflops =
        flops
        / (avg / 1000.0)
        / 1e9;

    std::cout
        << "\n=== CUDA MatMul Kernel Profile ===\n";

    std::cout
        << "shape: "
        << M
        << "x"
        << K
        << " * "
        << K
        << "x"
        << N
        << "\n";

    std::cout
        << "runs: "
        << measured_runs
        << "\n";

    std::cout
        << "correct: "
        << (correct ? "true" : "false")
        << "\n";

    std::cout
        << "avg_ms: "
        << avg
        << "\n";

    std::cout
        << "p50_ms: "
        << p50
        << "\n";

    std::cout
        << "p95_ms: "
        << p95
        << "\n";

    std::cout
        << "p99_ms: "
        << p99
        << "\n";

    std::cout
        << "estimated_GFLOPs: "
        << gflops
        << "\n";

    std::ofstream out(
        "../trace/cuda_matmul_profile.json"
    );

    out << "{\n";
    out << "  \"kernel\": \"matmul_naive\",\n";
    out << "  \"M\": " << M << ",\n";
    out << "  \"K\": " << K << ",\n";
    out << "  \"N\": " << N << ",\n";
    out << "  \"runs\": " << measured_runs << ",\n";
    out << "  \"correct\": "
        << (correct ? "true" : "false")
        << ",\n";
    out << "  \"avg_ms\": " << avg << ",\n";
    out << "  \"p50_ms\": " << p50 << ",\n";
    out << "  \"p95_ms\": " << p95 << ",\n";
    out << "  \"p99_ms\": " << p99 << ",\n";
    out << "  \"estimated_GFLOPs\": "
        << gflops
        << "\n";
    out << "}\n";

    std::cout
        << "[CUDAProfiler] Exported profile to "
        << "../trace/cuda_matmul_profile.json\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}