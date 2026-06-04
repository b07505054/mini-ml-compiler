#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    threadgroup float* partial_sum [[threadgroup(0)]]
) {
    const uint base = token * hidden;
    float sum = 0.0f;

    for (uint i = lane; i < hidden; i += threads) {
        const float value = input[base + i];
        sum += value * value;
    }

    partial_sum[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial_sum[lane] += partial_sum[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inverse_rms = rsqrt(partial_sum[0] / float(hidden) + epsilon);
    for (uint i = lane; i < hidden; i += threads) {
        output[base + i] = input[base + i] * inverse_rms * weight[i];
    }
}
