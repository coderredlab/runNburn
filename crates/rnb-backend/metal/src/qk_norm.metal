#include <metal_stdlib>
using namespace metal;

// Per-head RMSNorm: out[h*head_dim + i] =
//   (in[h*head_dim + i] / sqrt(mean(in_head^2) + eps)) * weight[i]
// rnb-cpu kernels/norm.rs rms_norm_into 1:1 (weight[head_dim] 는 head 공유).
// threadgroup 1개 = head 1개 (grid = num_heads). tg_size 256 고정
// (2의 거듭제곱 → tree reduction 정확, SIMD width 32 배수). grid-stride 라
// head_dim 임의 OK.
kernel void qk_norm(
    device const float* in     [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      head_dim [[buffer(3)]],
    constant float&     eps      [[buffer(4)]],
    uint head    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    uint base = head * head_dim;
    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = in[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(partial[0] / (float)head_dim + eps);
    for (uint i = tid; i < head_dim; i += tg_size) {
        out[base + i] = in[base + i] * inv_rms * weight[i];
    }
}
