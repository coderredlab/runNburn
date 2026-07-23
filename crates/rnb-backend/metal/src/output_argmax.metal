#include <metal_stdlib>
using namespace metal;

kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* token_out [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float best_vals[256];
    threadgroup uint best_idxs[256];

    float best = -INFINITY;
    uint best_idx = 0u;
    for (uint i = tid; i < N; i += 256u) {
        float v = logits[i];
        if (v > best || (v == best && i > best_idx)) {
            best = v;
            best_idx = i;
        }
    }

    best_vals[tid] = best;
    best_idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other = best_vals[tid + stride];
            uint other_idx = best_idxs[tid + stride];
            if (other > best_vals[tid] || (other == best_vals[tid] && other_idx > best_idxs[tid])) {
                best_vals[tid] = other;
                best_idxs[tid] = other_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        token_out[0] = best_idxs[0];
    }
}
