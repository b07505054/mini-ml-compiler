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

        NSString* sourcePath = @"../metal/vector_add.metal";
        NSError* error = nil;

        NSString* source = [NSString stringWithContentsOfFile:sourcePath
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
            [library newFunctionWithName:@"vector_add"];

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];

        if (!pipeline) {
            std::cerr << "Failed to create pipeline\n";
            return 1;
        }

        constexpr int N = 8;

        std::vector<float> a(N, 1.0f);
        std::vector<float> b(N, 2.0f);
        std::vector<float> c(N, 0.0f);

        id<MTLBuffer> bufferA =
            [device newBufferWithBytes:a.data()
                                length:N * sizeof(float)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferB =
            [device newBufferWithBytes:b.data()
                                length:N * sizeof(float)
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferC =
            [device newBufferWithLength:N * sizeof(float)
                                options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue =
            [device newCommandQueue];

        id<MTLCommandBuffer> commandBuffer =
            [queue commandBuffer];

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(N, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(N, 1, 1);

        [encoder dispatchThreads:gridSize
            threadsPerThreadgroup:threadGroupSize];

        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* result =
            static_cast<float*>([bufferC contents]);

        std::cout << "Metal Vector Add Output:\n";

        for (int i = 0; i < N; ++i) {
            std::cout << result[i] << " ";
        }

        std::cout << "\n";

        return 0;
    }
}