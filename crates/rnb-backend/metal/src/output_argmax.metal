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

// Batched DFlash draft output: top-1 token and its softmax probability. One
// threadgroup handles one logits row; host sets row-specific buffer offsets.
kernel void top1_probability_f32(
    device const float* logits [[buffer(0)]],
    device uint* token_out [[buffer(1)]],
    device float* probability_out [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float values[256];
    threadgroup uint indices[256];

    float best = -INFINITY;
    uint best_idx = 0u;
    for (uint i = tid; i < N; i += 256u) {
        float value = logits[i];
        if (value > best || (value == best && i > best_idx)) {
            best = value;
            best_idx = i;
        }
    }
    values[tid] = best;
    indices[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other = values[tid + stride];
            uint other_idx = indices[tid + stride];
            if (other > values[tid] || (other == values[tid] && other_idx > indices[tid])) {
                values[tid] = other;
                indices[tid] = other_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = 0.0f;
    const float max_logit = values[0];
    for (uint i = tid; i < N; i += 256u) {
        sum += exp(logits[i] - max_logit);
    }
    values[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        token_out[0] = indices[0];
        probability_out[0] = 1.0f / values[0];
    }
}

kernel void argmax_f32_excluding(
    device const float* logits [[buffer(0)]],
    device uint* token_out [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& excluded [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float best_vals[256];
    threadgroup uint best_idxs[256];

    float best = -INFINITY;
    uint best_idx = 0u;
    for (uint i = tid; i < N; i += 256u) {
        if (i == excluded) {
            continue;
        }
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
