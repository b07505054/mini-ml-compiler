#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runtime/metal_rmsnorm_executor.h"

#include <chrono>
#include <stdexcept>

struct MetalRMSNormExecutor::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
};

MetalRMSNormExecutor::MetalRMSNormExecutor(const std::string& shader_path)
    : impl_(new Impl()) {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        throw std::runtime_error("Metal device unavailable");
    }

    NSError* error = nil;
    NSString* source = [NSString
        stringWithContentsOfFile:[NSString stringWithUTF8String:shader_path.c_str()]
        encoding:NSUTF8StringEncoding
        error:&error];
    if (!source) {
        throw std::runtime_error("failed to load Metal RMSNorm shader");
    }

    id<MTLLibrary> library =
        [impl_->device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"rmsnorm_f32"];
    impl_->pipeline =
        [impl_->device newComputePipelineStateWithFunction:function error:&error];
    impl_->queue = [impl_->device newCommandQueue];
    if (!impl_->pipeline || !impl_->queue) {
        throw std::runtime_error("failed to initialize Metal RMSNorm pipeline");
    }
}

MetalRMSNormExecutor::~MetalRMSNormExecutor() {
    delete impl_;
}

MetalRMSNormExecutionResult MetalRMSNormExecutor::execute(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int tokens,
    int hidden,
    float epsilon,
    int warmup_runs,
    int timed_runs
) {
    if (tokens <= 0 || hidden <= 0 ||
        input.size() != static_cast<size_t>(tokens) * hidden ||
        weight.size() != static_cast<size_t>(hidden)) {
        throw std::invalid_argument("invalid Metal RMSNorm input shape");
    }

    constexpr NSUInteger threads_per_group = 256;
    id<MTLBuffer> input_buffer =
        [impl_->device newBufferWithBytes:input.data()
                                  length:input.size() * sizeof(float)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer =
        [impl_->device newBufferWithBytes:weight.data()
                                  length:weight.size() * sizeof(float)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [impl_->device newBufferWithLength:input.size() * sizeof(float)
                                   options:MTLResourceStorageModeShared];
    const uint32_t hidden_value = static_cast<uint32_t>(hidden);

    auto dispatch = [&]() {
        id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:impl_->pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_value length:sizeof(hidden_value) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder setThreadgroupMemoryLength:threads_per_group * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    };

    for (int i = 0; i < warmup_runs; ++i) {
        dispatch();
    }

    std::vector<double> latencies;
    for (int i = 0; i < timed_runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        dispatch();
        const auto end = std::chrono::steady_clock::now();
        latencies.push_back(
            std::chrono::duration<double, std::milli>(end - start).count()
        );
    }

    const float* output = static_cast<const float*>([output_buffer contents]);
    return {
        [[impl_->device name] UTF8String],
        std::vector<float>(output, output + input.size()),
        std::move(latencies),
    };
}
