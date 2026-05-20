#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>

double percentile(
    std::vector<double> v,
    double p
) {
    std::sort(v.begin(), v.end());

    size_t idx =
        static_cast<size_t>(
            p * (v.size() - 1)
        );

    return v[idx];
}

int main() {
    @autoreleasepool {

        id<MTLDevice> device =
            MTLCreateSystemDefaultDevice();

        if (!device) {
            std::cerr
                << "Metal unavailable\n";

            return 1;
        }

        std::cout
            << "Metal device: "
            << [[device name] UTF8String]
            << "\n";

        constexpr size_t N =
            1024 * 1024;

        std::vector<float> A(N, 1.0f);
        std::vector<float> B(N, 2.0f);

        id<MTLBuffer> bufferA =
            [device newBufferWithBytes:A.data()
                                length:N * sizeof(float)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferB =
            [device newBufferWithBytes:B.data()
                                length:N * sizeof(float)
                               options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue =
            [device newCommandQueue];

        constexpr int warmup_runs = 10;
        constexpr int measured_runs = 100;

        std::vector<double> latencies_ms;

        for (int i = 0; i < warmup_runs + measured_runs; ++i) {

            auto t0 =
                std::chrono::high_resolution_clock::now();

            id<MTLCommandBuffer> cmd =
                [queue commandBuffer];

            // fake GPU workload
            [cmd commit];

            [cmd waitUntilCompleted];

            auto t1 =
                std::chrono::high_resolution_clock::now();

            double ms =
                std::chrono::duration<double,
                                      std::milli>(
                    t1 - t0
                ).count();

            if (i >= warmup_runs) {
                latencies_ms.push_back(ms);
            }
        }

        double avg =
            std::accumulate(
                latencies_ms.begin(),
                latencies_ms.end(),
                0.0
            ) / latencies_ms.size();

        double p50 =
            percentile(latencies_ms, 0.50);

        double p95 =
            percentile(latencies_ms, 0.95);

        double p99 =
            percentile(latencies_ms, 0.99);

        std::cout
            << "\n=== Metal Kernel Profile ===\n";

        std::cout
            << "runs: "
            << measured_runs
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

        std::ofstream out(
            "../trace/metal_kernel_profile.json"
        );

        out << "{\n";

        out << "  \"device\": \""
            << [[device name] UTF8String]
            << "\",\n";

        out << "  \"avg_ms\": "
            << avg
            << ",\n";

        out << "  \"p50_ms\": "
            << p50
            << ",\n";

        out << "  \"p95_ms\": "
            << p95
            << ",\n";

        out << "  \"p99_ms\": "
            << p99
            << "\n";

        out << "}\n";

        std::cout
            << "[MetalProfiler] Exported profile to "
            << "../trace/metal_kernel_profile.json\n";
    }

    return 0;
}