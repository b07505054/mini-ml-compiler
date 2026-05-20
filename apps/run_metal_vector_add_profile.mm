#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

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
            std::cerr << "Metal unavailable\n";
            return 1;
        }

        std::cout << "Metal device: "
                  << [[device name] UTF8String]
                  << "\n";

        NSString* sourcePath =
            @"../metal/vector_add.metal";

        NSError* error = nil;

        NSString* source =
            [NSString stringWithContentsOfFile:sourcePath
                                      encoding:NSUTF8StringEncoding
                                         error:&error];

        if (!source) {
            std::cerr << "Failed to load shader\n";
            return 1;
        }

        id<MTLLibrary> library =
            [device newLibraryWithSource:source
                                  options:nil
                                    error:&error];

        if (!library) {
            std::cerr << "Failed to compile shader: "
                      << [[error localizedDescription] UTF8String]
                      << "\n";
            return 1;
        }

        id<MTLFunction> function =
            [library newFunctionWithName:@"vector_add"];

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];

        if (!pipeline) {
            std::cerr << "Failed to create pipeline\n";
            return 1;
        }

        constexpr size_t N = 1024 * 1024;

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

        id<MTLBuffer> bufferC =
            [device newBufferWithLength:N * sizeof(float)
                                options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue =
            [device newCommandQueue];

        constexpr int warmup_runs = 10;
        constexpr int measured_runs = 100;

        std::vector<double> latencies_ms;

        MTLSize gridSize =
            MTLSizeMake(N, 1, 1);

        MTLSize threadGroupSize =
            MTLSizeMake(256, 1, 1);

        for (int i = 0; i < warmup_runs + measured_runs; ++i) {
            auto t0 =
                std::chrono::high_resolution_clock::now();

            id<MTLCommandBuffer> cmd =
                [queue commandBuffer];

            id<MTLComputeCommandEncoder> encoder =
                [cmd computeCommandEncoder];

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufferA offset:0 atIndex:0];
            [encoder setBuffer:bufferB offset:0 atIndex:1];
            [encoder setBuffer:bufferC offset:0 atIndex:2];

            [encoder dispatchThreads:gridSize
                threadsPerThreadgroup:threadGroupSize];

            [encoder endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];

            auto t1 =
                std::chrono::high_resolution_clock::now();

            double ms =
                std::chrono::duration<double, std::milli>(
                    t1 - t0
                ).count();

            if (i >= warmup_runs) {
                latencies_ms.push_back(ms);
            }
        }

        float* result =
            static_cast<float*>([bufferC contents]);

        bool correct = true;

        for (size_t i = 0; i < 16; ++i) {
            if (result[i] != 3.0f) {
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
            percentile(latencies_ms, 0.50);

        double p95 =
            percentile(latencies_ms, 0.95);

        double p99 =
            percentile(latencies_ms, 0.99);

        std::cout << "\n=== Metal VectorAdd Kernel Profile ===\n";
        std::cout << "elements: " << N << "\n";
        std::cout << "runs: " << measured_runs << "\n";
        std::cout << "correct: " << (correct ? "true" : "false") << "\n";
        std::cout << "avg_ms: " << avg << "\n";
        std::cout << "p50_ms: " << p50 << "\n";
        std::cout << "p95_ms: " << p95 << "\n";
        std::cout << "p99_ms: " << p99 << "\n";

        std::ofstream out(
            "../trace/metal_vector_add_profile.json"
        );

        out << "{\n";
        out << "  \"device\": \""
            << [[device name] UTF8String]
            << "\",\n";
        out << "  \"kernel\": \"vector_add\",\n";
        out << "  \"elements\": " << N << ",\n";
        out << "  \"runs\": " << measured_runs << ",\n";
        out << "  \"correct\": "
            << (correct ? "true" : "false")
            << ",\n";
        out << "  \"avg_ms\": " << avg << ",\n";
        out << "  \"p50_ms\": " << p50 << ",\n";
        out << "  \"p95_ms\": " << p95 << ",\n";
        out << "  \"p99_ms\": " << p99 << "\n";
        out << "}\n";

        std::cout
            << "[MetalProfiler] Exported profile to "
            << "../trace/metal_vector_add_profile.json\n";
    }

    return 0;
}