#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx < N) {
        C[idx] =
            A[idx] + B[idx];
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
    constexpr int N =
        1 << 24;

    constexpr int warmup_runs =
        10;

    constexpr int measured_runs =
        100;

    size_t bytes =
        N * sizeof(float);

    std::vector<float> hA(N, 1.0f);
    std::vector<float> hB(N, 2.0f);
    std::vector<float> hC(N, 0.0f);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &dA,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &dB,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &dC,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            dA,
            hA.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            dB,
            hB.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    int threads =
        256;

    int blocks =
        (N + threads - 1)
        / threads;

    std::vector<float> latencies_ms;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );

    for (int i = 0;
         i < warmup_runs + measured_runs;
         ++i) {

        CUDA_CHECK(
            cudaEventRecord(start)
        );

        vector_add_kernel<<<blocks, threads>>>(
            dA,
            dB,
            dC,
            N
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
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    bool correct = true;

    for (int i = 0; i < 16; ++i) {
        if (hC[i] != 3.0f) {
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

    double total_bytes =
        static_cast<double>(
            3 * bytes
        );

    double bandwidth_gbps =
        total_bytes
        / (avg / 1000.0)
        / 1e9;

    std::cout
        << "\n=== CUDA VectorAdd Kernel Profile ===\n";

    std::cout
        << "elements: "
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
        << "effective_bandwidth_GBps: "
        << bandwidth_gbps
        << "\n";

    std::ofstream out(
        "../trace/cuda_vector_add_profile.json"
    );

    out << "{\n";
    out << "  \"kernel\": \"vector_add\",\n";
    out << "  \"elements\": " << N << ",\n";
    out << "  \"runs\": " << measured_runs << ",\n";
    out << "  \"correct\": "
        << (correct ? "true" : "false")
        << ",\n";
    out << "  \"avg_ms\": " << avg << ",\n";
    out << "  \"p50_ms\": " << p50 << ",\n";
    out << "  \"p95_ms\": " << p95 << ",\n";
    out << "  \"p99_ms\": " << p99 << ",\n";
    out << "  \"effective_bandwidth_GBps\": "
        << bandwidth_gbps
        << "\n";
    out << "}\n";

    std::cout
        << "[CUDAProfiler] Exported profile to "
        << "../trace/cuda_vector_add_profile.json\n";

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    CUDA_CHECK(
        cudaFree(dA)
    );

    CUDA_CHECK(
        cudaFree(dB)
    );

    CUDA_CHECK(
        cudaFree(dC)
    );

    return 0;
}