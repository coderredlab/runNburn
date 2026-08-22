#pragma once

#include <float.h>

template <unsigned D>
__device__ __forceinline__ void rnb_kvarn_hadamard(float* values) {
    const unsigned lane = threadIdx.x;
    for (unsigned span = 1u; span < D; span <<= 1u) {
        if (lane < D / 2u) {
            const unsigned base = (lane / span) * (span << 1u);
            const unsigned offset = lane % span;
            const float left = values[base + offset];
            const float right = values[base + span + offset];
            values[base + offset] = left + right;
            values[base + span + offset] = left - right;
        }
        __syncthreads();
    }
    values[lane] *= rsqrtf(static_cast<float>(D));
    __syncthreads();
}

__host__ __device__ constexpr unsigned rnb_log2u(unsigned value) {
    return value <= 1u ? 0u : 1u + rnb_log2u(value >> 1u);
}

// Tree reduction preserving the original pairwise addition order (stride D/2
// down to 1, lower lane absorbs upper) with half of the original barriers:
// three staged shared-memory rounds, an in-warp shuffle tail, one broadcast
// round-trip so every lane receives the same dot result, and caller-side
// buffer alternation replacing the old trailing __syncthreads.
template <unsigned D>
__device__ __forceinline__ float rnb_kvarn_dot_reduce(
    float value, float (*partial)[D], unsigned buf) {
    constexpr unsigned LOG2_D = rnb_log2u(D);
    constexpr unsigned SHARED_STEPS = LOG2_D - 5u;
    static_assert(LOG2_D >= 6u, "D too small for staged reduce");
    const unsigned lane = threadIdx.x;
    partial[buf][lane] = value;
    __syncthreads();
    unsigned stride = D >> 1u;
    #pragma unroll
    for (unsigned step = 0u; step < SHARED_STEPS; ++step) {
        if (lane < stride) {
            partial[buf][lane] += partial[buf][lane + stride];
        }
        __syncthreads();
        stride >>= 1u;
    }
    // Surviving sums live in lanes < 32; the leftover strides stay inside
    // warp 0, so register shuffles finish them with identical pairings.
    // Lanes >= 32 hold stale scratch; masking to +0.0 keeps the shuffle tail
    // free of contamination.
    float value_in_warp = lane < 32u ? partial[buf][lane] : 0.0f;
    value_in_warp += __shfl_down_sync(0xffffffffu, value_in_warp, 16u);
    value_in_warp += __shfl_down_sync(0xffffffffu, value_in_warp, 8u);
    value_in_warp += __shfl_down_sync(0xffffffffu, value_in_warp, 4u);
    value_in_warp += __shfl_down_sync(0xffffffffu, value_in_warp, 2u);
    value_in_warp += __shfl_down_sync(0xffffffffu, value_in_warp, 1u);
    if (lane == 0u) {
        partial[buf][0] = value_in_warp;
    }
    __syncthreads();
    return partial[buf][0];
}

template <unsigned D>
__device__ void rnb_kvarn_attention_decode_impl(
    float* __restrict__ out,
    const float* __restrict__ q,
    const unsigned char* __restrict__ packed_blocks,
    const unsigned short* __restrict__ sink_key,
    const unsigned short* __restrict__ sink_value,
    const unsigned short* __restrict__ tail_key,
    const unsigned short* __restrict__ tail_value,
    const unsigned short* __restrict__ current_key,
    const unsigned short* __restrict__ current_value,
    unsigned kv_len,
    unsigned tail_start,
    unsigned sink_len,
    unsigned tail_len,
    unsigned num_blocks,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned group,
    unsigned value_bits,
    unsigned block_bytes,
    unsigned key_packed_offset,
    unsigned key_scale_offset,
    unsigned key_zero_offset,
    unsigned key_token_scale_offset,
    unsigned value_packed_offset,
    unsigned value_channel_scale_offset,
    unsigned value_token_scale_offset,
    unsigned value_zero_offset,
    float scale,
    float softcap,
    unsigned current_tokens,
    unsigned sliding_window,
    float* __restrict__ split_scratch,
    unsigned split_z) {
    const unsigned lane = threadIdx.x;
    const unsigned head = blockIdx.x;
    const unsigned query_token = blockIdx.y;
    const unsigned query_rows = current_tokens > 0u ? current_tokens : 1u;
    if (lane >= D || head >= num_heads || query_token >= query_rows || num_kv_heads == 0u) {
        return;
    }

    __shared__ float transform[D];
    // Double-buffered reduction scratch: consecutive dot reductions alternate
    // buffers, so no trailing barrier is needed before token N+1 overwrites
    // the other plane.
    __shared__ float partial[2][D];
    unsigned reduce_buf = 0u;

    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_head = head / heads_per_group;
    const unsigned row_width = num_kv_heads * D;
    const unsigned visible_current =
        current_tokens > 0u && query_token < current_tokens ? query_token + 1u : 0u;
    const unsigned visible_len = kv_len + visible_current;
    const unsigned window_start =
        sliding_window > 0u && visible_len > sliding_window ? visible_len - sliding_window : 0u;
    const unsigned z_index = blockIdx.z;
    const unsigned blocks_per_slice =
        split_z > 0u ? (num_blocks + split_z - 1u) / split_z : num_blocks;
    const unsigned slice_begin = blocks_per_slice * z_index;
    const unsigned slice_end = min(slice_begin + blocks_per_slice, num_blocks);
    const unsigned regions = split_z + 3u;
    const unsigned region_sink = split_z;
    const unsigned region_tail = split_z + 1u;
    const unsigned region_current = split_z + 2u;
    const size_t query_base =
        (static_cast<size_t>(query_token) * num_heads + head) * D;
    const float q_original = q[query_base + lane];
    transform[lane] = q_original;
    __syncthreads();
    rnb_kvarn_hadamard<D>(transform);
    const float q_rotated = transform[lane];

    float sink_max = -FLT_MAX;
    float sink_sum = 0.0f;
    float sink_acc = 0.0f;
    for (unsigned token = 0u; z_index == 0u && token < sink_len; ++token) {
        if (token < window_start || token >= kv_len) {
            continue;
        }
        const unsigned base = token * row_width + kv_head * D;
        const float key = __half2float(__ushort_as_half(sink_key[base + lane]));
        float score = rnb_kvarn_dot_reduce<D>(q_original * key, partial, reduce_buf) * scale;
        reduce_buf ^= 1u;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        const float next_max = fmaxf(sink_max, score);
        const float old_scale = sink_max == -FLT_MAX ? 0.0f : expf(sink_max - next_max);
        const float probability = expf(score - next_max);
        const float value = __half2float(__ushort_as_half(sink_value[base + lane]));
        sink_acc = sink_acc * old_scale + probability * value;
        sink_sum = sink_sum * old_scale + probability;
        sink_max = next_max;
    }

    float quantized_max = -FLT_MAX;
    float quantized_sum = 0.0f;
    float quantized_acc = 0.0f;
    const unsigned key_packed_head = D * (group / 2u);
    const unsigned value_pack = 8u / value_bits;
    const unsigned value_packed_row = D / value_pack;
    const unsigned value_packed_head = group * value_packed_row;
    for (unsigned block = slice_begin; block < slice_end; ++block) {
        const unsigned char* record = packed_blocks + static_cast<size_t>(block) * block_bytes;
        const unsigned short* key_scale = reinterpret_cast<const unsigned short*>(record + key_scale_offset);
        const unsigned short* key_zero = reinterpret_cast<const unsigned short*>(record + key_zero_offset);
        const unsigned short* key_token_scale = reinterpret_cast<const unsigned short*>(record + key_token_scale_offset);
        const unsigned short* value_channel_scale = reinterpret_cast<const unsigned short*>(record + value_channel_scale_offset);
        const unsigned short* value_token_scale = reinterpret_cast<const unsigned short*>(record + value_token_scale_offset);
        const unsigned short* value_zero = reinterpret_cast<const unsigned short*>(record + value_zero_offset);
        for (unsigned token = 0u; token < group; ++token) {
            const unsigned global_token = sink_len + block * group + token;
            if (global_token < window_start || global_token >= kv_len) {
                continue;
            }
            const unsigned key_byte_index = kv_head * key_packed_head + lane * (group / 2u) + token / 2u;
            const unsigned key_byte = record[key_packed_offset + key_byte_index];
            const unsigned key_quant = token & 1u ? key_byte >> 4u : key_byte & 0x0fu;
            const unsigned key_channel = kv_head * D + lane;
            const float key_scale_value = __half2float(__ushort_as_half(key_scale[key_channel]));
            const float key_zero_value = __half2float(__ushort_as_half(key_zero[key_channel]));
            const float key_token_value = __half2float(__ushort_as_half(key_token_scale[kv_head * group + token]));
            const float key = (static_cast<float>(key_quant) * key_scale_value + key_zero_value) * key_token_value;
            float score = rnb_kvarn_dot_reduce<D>(q_rotated * key, partial, reduce_buf) * scale;
            reduce_buf ^= 1u;
            if (softcap > 0.0f) {
                score = softcap * tanhf(score / softcap);
            }
            const float next_max = fmaxf(quantized_max, score);
            const float old_scale = quantized_max == -FLT_MAX ? 0.0f : expf(quantized_max - next_max);
            const float probability = expf(score - next_max);

            const unsigned value_byte_index = kv_head * value_packed_head + token * value_packed_row + lane / value_pack;
            const unsigned value_byte = record[value_packed_offset + value_byte_index];
            const unsigned value_mask = (1u << value_bits) - 1u;
            const unsigned value_quant = (value_byte >> ((lane % value_pack) * value_bits)) & value_mask;
            const float value_channel = __half2float(__ushort_as_half(value_channel_scale[kv_head * D + lane]));
            const float value_token = __half2float(__ushort_as_half(value_token_scale[kv_head * group + token]));
            const float value_zero_point = __half2float(__ushort_as_half(value_zero[kv_head * group + token]));
            const float value = (static_cast<float>(value_quant) * value_token + value_zero_point) * value_channel;
            quantized_acc = quantized_acc * old_scale + probability * value;
            quantized_sum = quantized_sum * old_scale + probability;
            quantized_max = next_max;
        }
    }

    transform[lane] = quantized_acc;
    __syncthreads();
    rnb_kvarn_hadamard<D>(transform);
    const float quantized_acc_original = transform[lane];

    float tail_max = -FLT_MAX;
    float tail_sum = 0.0f;
    float tail_acc = 0.0f;
    for (unsigned token = 0u; z_index == 0u && token < tail_len; ++token) {
        const unsigned global_token = tail_start + token;
        if (global_token < window_start || global_token >= kv_len) {
            continue;
        }
        const unsigned base = token * row_width + kv_head * D;
        const float key = __half2float(__ushort_as_half(tail_key[base + lane]));
        float score = rnb_kvarn_dot_reduce<D>(q_original * key, partial, reduce_buf) * scale;
        reduce_buf ^= 1u;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        const float next_max = fmaxf(tail_max, score);
        const float old_scale = tail_max == -FLT_MAX ? 0.0f : expf(tail_max - next_max);
        const float probability = expf(score - next_max);
        const float value = __half2float(__ushort_as_half(tail_value[base + lane]));
        tail_acc = tail_acc * old_scale + probability * value;
        tail_sum = tail_sum * old_scale + probability;
        tail_max = next_max;
    }

    float current_max = -FLT_MAX;
    float current_sum = 0.0f;
    float current_acc = 0.0f;
    for (unsigned token = 0u; z_index == 0u && token < visible_current; ++token) {
        const unsigned global_token = kv_len + token;
        if (global_token < window_start) {
            continue;
        }
        const unsigned base = token * row_width + kv_head * D;
        const float key = __half2float(__ushort_as_half(current_key[base + lane]));
        float score = rnb_kvarn_dot_reduce<D>(q_original * key, partial, reduce_buf) * scale;
        reduce_buf ^= 1u;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        const float next_max = fmaxf(current_max, score);
        const float old_scale = current_max == -FLT_MAX ? 0.0f : expf(current_max - next_max);
        const float probability = expf(score - next_max);
        const float value = __half2float(__ushort_as_half(current_value[base + lane]));
        current_acc = current_acc * old_scale + probability * value;
        current_sum = current_sum * old_scale + probability;
        current_max = next_max;
    }

    if (split_z > 1u) {
        // Multi-slice path: stash this slice's partial softmax (acc rotated
        // back to the original domain so the merge kernel stays linear) into
        // the scratch regions; z==0 also records sink/tail/current.
        transform[lane] = quantized_acc;
        __syncthreads();
        rnb_kvarn_hadamard<D>(transform);
        const size_t slice_slot =
            (static_cast<size_t>(query_token) * regions + z_index) * num_heads + head;
        split_scratch[slice_slot * 2u] = quantized_max;
        split_scratch[slice_slot * 2u + 1u] = quantized_sum;
        split_scratch[regions * num_heads * 2u + slice_slot * D + lane] = transform[lane];
        if (z_index == 0u) {
            const size_t sink_slot =
                (static_cast<size_t>(query_token) * regions + region_sink) * num_heads + head;
            const size_t tail_slot =
                (static_cast<size_t>(query_token) * regions + region_tail) * num_heads + head;
            const size_t current_slot =
                (static_cast<size_t>(query_token) * regions + region_current) * num_heads + head;
            split_scratch[sink_slot * 2u] = sink_max;
            split_scratch[sink_slot * 2u + 1u] = sink_sum;
            split_scratch[regions * num_heads * 2u + sink_slot * D + lane] = sink_acc;
            split_scratch[tail_slot * 2u] = tail_max;
            split_scratch[tail_slot * 2u + 1u] = tail_sum;
            split_scratch[regions * num_heads * 2u + tail_slot * D + lane] = tail_acc;
            split_scratch[current_slot * 2u] = current_max;
            split_scratch[current_slot * 2u + 1u] = current_sum;
            split_scratch[regions * num_heads * 2u + current_slot * D + lane] = current_acc;
        }
        return;
    }
    const float first_max = fmaxf(sink_max, quantized_max);
    const float sink_weight = sink_sum > 0.0f ? expf(sink_max - first_max) : 0.0f;
    const float quantized_weight = quantized_sum > 0.0f ? expf(quantized_max - first_max) : 0.0f;
    float merged_acc = sink_acc * sink_weight + quantized_acc_original * quantized_weight;
    float merged_sum = sink_sum * sink_weight + quantized_sum * quantized_weight;
    float merged_max = first_max;

    if (tail_sum > 0.0f) {
        const float next_max = merged_sum > 0.0f ? fmaxf(merged_max, tail_max) : tail_max;
        const float old_weight = merged_sum > 0.0f ? expf(merged_max - next_max) : 0.0f;
        const float tail_weight = expf(tail_max - next_max);
        merged_acc = merged_acc * old_weight + tail_acc * tail_weight;
        merged_sum = merged_sum * old_weight + tail_sum * tail_weight;
        merged_max = next_max;
    }
    if (current_sum > 0.0f) {
        const float next_max = merged_sum > 0.0f ? fmaxf(merged_max, current_max) : current_max;
        const float old_weight = merged_sum > 0.0f ? expf(merged_max - next_max) : 0.0f;
        const float current_weight = expf(current_max - next_max);
        merged_acc = merged_acc * old_weight + current_acc * current_weight;
        merged_sum = merged_sum * old_weight + current_sum * current_weight;
    }
    out[query_base + lane] = merged_sum > 0.0f ? merged_acc / merged_sum : 0.0f;
}

#define RNB_KVARN_ARGUMENTS \
    float* out, const float* q, const unsigned char* packed_blocks, \
    const unsigned short* sink_key, const unsigned short* sink_value, \
    const unsigned short* tail_key, const unsigned short* tail_value, \
    const unsigned short* current_key, const unsigned short* current_value, \
    unsigned kv_len, unsigned tail_start, unsigned sink_len, unsigned tail_len, \
    unsigned num_blocks, unsigned num_heads, unsigned num_kv_heads, unsigned group, \
    unsigned value_bits, unsigned block_bytes, unsigned key_packed_offset, \
    unsigned key_scale_offset, unsigned key_zero_offset, unsigned key_token_scale_offset, \
    unsigned value_packed_offset, unsigned value_channel_scale_offset, \
    unsigned value_token_scale_offset, unsigned value_zero_offset, float scale, \
    float softcap, unsigned current_tokens, unsigned sliding_window, \
    float* split_scratch, unsigned split_z

#define RNB_KVARN_PASS \
    out, q, packed_blocks, sink_key, sink_value, tail_key, tail_value, current_key, \
    current_value, kv_len, tail_start, sink_len, tail_len, num_blocks, num_heads, \
    num_kv_heads, group, value_bits, block_bytes, key_packed_offset, key_scale_offset, \
    key_zero_offset, key_token_scale_offset, value_packed_offset, \
    value_channel_scale_offset, value_token_scale_offset, value_zero_offset, scale, \
    softcap, current_tokens, sliding_window, split_scratch, split_z

extern "C" __global__ void rnb_kvarn_attention_decode_hd128(RNB_KVARN_ARGUMENTS) {
    rnb_kvarn_attention_decode_impl<128>(RNB_KVARN_PASS);
}

extern "C" __global__ void rnb_kvarn_attention_decode_hd256(RNB_KVARN_ARGUMENTS) {
    rnb_kvarn_attention_decode_impl<256>(RNB_KVARN_PASS);
}

extern "C" __global__ void rnb_kvarn_attention_decode_hd512(RNB_KVARN_ARGUMENTS) {
    rnb_kvarn_attention_decode_impl<512>(RNB_KVARN_PASS);
}

// Combines per-slice partial softmaxes from the split decode kernel. Regions
// are consumed in ascending token order (quantized slices, then sink, tail,
// current) with the same online-merge formulas as the single-pass version;
// results match up to floating-point association inside the online softmax.
template <unsigned D>
__device__ void rnb_kvarn_attention_decode_merge_impl(
    float* __restrict__ out,
    const float* __restrict__ split_scratch,
    unsigned split_z,
    unsigned num_heads) {
    const unsigned lane = threadIdx.x;
    const unsigned head = blockIdx.x;
    const unsigned query_token = blockIdx.y;
    if (lane >= D || head >= num_heads) {
        return;
    }
    const unsigned regions = split_z + 3u;
    const size_t region_base = static_cast<size_t>(query_token) * regions * num_heads;
    float merged_max = -FLT_MAX;
    float merged_sum = 0.0f;
    float merged_acc = 0.0f;
    auto merge_region = [&](unsigned region) {
        const size_t slot = region_base + region * num_heads + head;
        const float region_max = split_scratch[slot * 2u];
        const float region_sum = split_scratch[slot * 2u + 1u];
        if (region_sum <= 0.0f) {
            return;
        }
        const float region_acc =
            split_scratch[regions * num_heads * 2u + slot * D + lane];
        if (merged_sum <= 0.0f) {
            merged_max = region_max;
            merged_sum = region_sum;
            merged_acc = region_acc;
            return;
        }
        const float next_max = fmaxf(merged_max, region_max);
        const float old_weight = expf(merged_max - next_max);
        const float region_weight = expf(region_max - next_max);
        merged_acc = merged_acc * old_weight + region_acc * region_weight;
        merged_sum = merged_sum * old_weight + region_sum * region_weight;
        merged_max = next_max;
    };
    for (unsigned slice = 0u; slice < split_z; ++slice) {
        merge_region(slice);
    }
    merge_region(split_z);       // sink
    merge_region(split_z + 1u);  // tail
    merge_region(split_z + 2u);  // current
    const size_t query_base =
        (static_cast<size_t>(query_token) * num_heads + head) * D;
    out[query_base + lane] = merged_sum > 0.0f ? merged_acc / merged_sum : 0.0f;
}

#define RNB_KVARN_MERGE_ARGUMENTS \
    float* out, const float* split_scratch, unsigned split_z, unsigned num_heads

#define RNB_KVARN_MERGE_PASS \
    out, split_scratch, split_z, num_heads

extern "C" __global__ void rnb_kvarn_attention_decode_merge_hd128(RNB_KVARN_MERGE_ARGUMENTS) {
    rnb_kvarn_attention_decode_merge_impl<128>(RNB_KVARN_MERGE_PASS);
}

extern "C" __global__ void rnb_kvarn_attention_decode_merge_hd256(RNB_KVARN_MERGE_ARGUMENTS) {
    rnb_kvarn_attention_decode_merge_impl<256>(RNB_KVARN_MERGE_PASS);
}

extern "C" __global__ void rnb_kvarn_attention_decode_merge_hd512(RNB_KVARN_MERGE_ARGUMENTS) {
    rnb_kvarn_attention_decode_merge_impl<512>(RNB_KVARN_MERGE_PASS);
}

#undef RNB_KVARN_PASS
#undef RNB_KVARN_ARGUMENTS
#undef RNB_KVARN_MERGE_PASS
#undef RNB_KVARN_MERGE_ARGUMENTS
