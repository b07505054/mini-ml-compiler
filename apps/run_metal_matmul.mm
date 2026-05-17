#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <vector>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (!device) {
            std::cerr << "Metal device not found\n";
            return 1;
        }

        std::cout << "Metal device: "
                  << [[device name] UTF8String]
                  << "\n";

        NSString* sourcePath = @"../metal/matmul.metal";
        NSError* error = nil;

        NSString* source =
            [NSString stringWithContentsOfFile:sourcePath
                                      encoding:NSUTF8StringEncoding
                                         error:&error];

        if (!source) {
            std::cerr << "Failed to load Metal shader\n";
            return 1;
        }

        id<MTLLibrary> library =
            [device newLibraryWithSource:source
                                  options:nil
                                    error:&error];

        if (!library) {
            std::cerr << "Failed to compile Metal shader: "
                      << [[error localizedDescription] UTF8String]
                      << "\n";
            return 1;
        }

        id<MTLFunction> function =
            [library newFunctionWithName:@"matmul_kernel"];

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];

        if (!pipeline) {
            std::cerr << "Failed to create pipeline\n";
            return 1;
        }

        const uint32_t M = 2;
        const uint32_t K = 3;
        const uint32_t N = 2;

        std::vector<float> A = {
            1, 2, 3,
            4, 5, 6
        };

        std::vector<float> B = {
            1, 2,
            3, 4,
            5, 6
        };

        std::vector<float> C(M * N, 0.0f);

        id<MTLBuffer> bufferA =
            [device newBufferWithBytes:A.data()
                                length:A.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferB =
            [device newBufferWithBytes:B.data()
                                length:B.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferC =
            [device newBufferWithLength:C.size() * sizeof(float)
                                options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferM =
            [device newBufferWithBytes:&M
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferN =
            [device newBufferWithBytes:&N
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferK =
            [device newBufferWithBytes:&K
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        [encoder setBuffer:bufferM offset:0 atIndex:3];
        [encoder setBuffer:bufferN offset:0 atIndex:4];
        [encoder setBuffer:bufferK offset:0 atIndex:5];

        MTLSize gridSize =
            MTLSizeMake(N, M, 1);

        MTLSize threadGroupSize =
            MTLSizeMake(8, 8, 1);

        [encoder dispatchThreads:gridSize
            threadsPerThreadgroup:threadGroupSize];

        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* result =
            static_cast<float*>([bufferC contents]);

        std::cout << "Metal MatMul Output:\n";

        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                std::cout << result[i * N + j] << " ";
            }

            std::cout << "\n";
        }

        std::cout << "Expected:\n";
        std::cout << "22 28\n";
        std::cout << "49 64\n";

        return 0;
    }
}