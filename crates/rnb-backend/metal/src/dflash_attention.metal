#include <metal_stdlib>
using namespace metal;

// Muse DFlash block attention. One SIMD-group owns one (query row, query head).
// The current draft block is intentionally non-causal: every query sees every block row,
// while old committed rows are clipped by the query-relative sliding window.
kernel void dflash_attention_hd128(
    device const float* query [[buffer(0)]],
    device const half* context_key [[buffer(1)]],
    device const half* context_value [[buffer(2)]],
    device const float* block_key [[buffer(3)]],
    device const float* block_value [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& context_len [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& position [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& num_kv_heads [[buffer(10)]],
    constant uint& sliding_window [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint HEAD_DIM = 128u;
    constexpr uint SIMD_WIDTH = 32u;

    const uint query_index = group.x;
    const uint head = group.y;
    if (query_index >= seq_len || head >= num_heads) {
        return;
    }

    const uint kv_head = head / (num_heads / num_kv_heads);
    const uint query_base = (query_index * num_heads + head) * HEAD_DIM;
    const uint context_start = position - context_len;
    const uint query_position = position + query_index;
    const uint first_position = query_position + 1u > sliding_window
        ? query_position + 1u - sliding_window
        : 0u;
    const uint first_key = first_position > context_start
        ? first_position - context_start
        : 0u;
    const uint key_count = context_len + seq_len;

    float query_lane[4];
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint part = 0u; part < 4u; ++part) {
        query_lane[part] = query[query_base + lane + part * SIMD_WIDTH];
    }

    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint key_index = first_key; key_index < key_count; ++key_index) {
        const bool committed = key_index < context_len;
        const uint row = committed ? key_index : key_index - context_len;
        const uint key_base = (row * num_kv_heads + kv_head) * HEAD_DIM;
        float dot = 0.0f;
        for (uint part = 0u; part < 4u; ++part) {
            const uint offset = key_base + lane + part * SIMD_WIDTH;
            const float key_value = committed
                ? (float)context_key[offset]
                : block_key[offset];
            dot += query_lane[part] * key_value;
        }
        const float score = simd_sum(dot) * scale;

        if (score > maximum) {
            const float old_scale = isfinite(maximum) ? exp(maximum - score) : 0.0f;
            denominator = denominator * old_scale + 1.0f;
            for (uint part = 0u; part < 4u; ++part) {
                const uint offset = key_base + lane + part * SIMD_WIDTH;
                const float value = committed
                    ? (float)context_value[offset]
                    : block_value[offset];
                accumulator[part] = accumulator[part] * old_scale + value;
            }
            maximum = score;
        } else {
            const float probability = exp(score - maximum);
            denominator += probability;
            for (uint part = 0u; part < 4u; ++part) {
                const uint offset = key_base + lane + part * SIMD_WIDTH;
                const float value = committed
                    ? (float)context_value[offset]
                    : block_value[offset];
                accumulator[part] += probability * value;
            }
        }
    }

    const float inverse_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (uint part = 0u; part < 4u; ++part) {
        output[query_base + lane + part * SIMD_WIDTH] = accumulator[part] * inverse_denominator;
    }
}

// Muse target continuation attention with the same mixed-precision contract as
// rnb-cpu attention_batch_f16: Q is rounded to F16, QK dot and AV accumulator
// stay in F16, while online-softmax maxima and sums stay in F32.
kernel void muse_target_attention_f16_hd128(
    device const float* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& kv_len [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& sliding_window [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint head_group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint HEAD_DIM = 128u;
    constexpr uint KV_TILE = 32u;
    constexpr uint DOT_LANES = 8u;

    const uint query_index = head_group / num_heads;
    const uint head = head_group % num_heads;
    if (query_index >= seq_len) {
        return;
    }

    const uint kv_head = head / (num_heads / num_kv_heads);
    const uint query_base = (query_index * num_heads + head) * HEAD_DIM;
    const uint global_pos = kv_len - seq_len + query_index;
    const uint score_index = thread_index / DOT_LANES;
    const uint dot_lane = thread_index % DOT_LANES;

    threadgroup float dot_lanes[KV_TILE * DOT_LANES];
    threadgroup float probabilities[KV_TILE];
    threadgroup half accumulator[HEAD_DIM];
    threadgroup float shared_row_max;
    threadgroup float shared_row_sum;
    threadgroup float shared_rescale;

    if (thread_index < HEAD_DIM) {
        accumulator[thread_index] = 0.0h;
    }
    if (thread_index == 0u) {
        shared_row_max = -INFINITY;
        shared_row_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0u; tile_start < kv_len; tile_start += KV_TILE) {
        const uint key_index = tile_start + score_index;
        half sum0 = 0.0h;
        half sum1 = 0.0h;
        half sum2 = 0.0h;
        half sum3 = 0.0h;
        if (score_index < KV_TILE && key_index <= global_pos) {
            const bool masked_by_window =
                sliding_window > 0u && key_index + sliding_window <= global_pos;
            if (!masked_by_window) {
                const uint key_base =
                    (key_index * num_kv_heads + kv_head) * HEAD_DIM;
                for (uint offset = 0u; offset < HEAD_DIM; offset += 32u) {
                    sum0 = fma(
                        (half)query[query_base + offset + dot_lane],
                        key[key_base + offset + dot_lane],
                        sum0);
                    sum1 = fma(
                        (half)query[query_base + offset + 8u + dot_lane],
                        key[key_base + offset + 8u + dot_lane],
                        sum1);
                    sum2 = fma(
                        (half)query[query_base + offset + 16u + dot_lane],
                        key[key_base + offset + 16u + dot_lane],
                        sum2);
                    sum3 = fma(
                        (half)query[query_base + offset + 24u + dot_lane],
                        key[key_base + offset + 24u + dot_lane],
                        sum3);
                }
            }
        }
        dot_lanes[thread_index] = (float)((sum0 + sum1) + (sum2 + sum3));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index < KV_TILE) {
            const uint lane_base = thread_index * DOT_LANES;
            const float pair0 =
                dot_lanes[lane_base] + dot_lanes[lane_base + 4u];
            const float pair1 =
                dot_lanes[lane_base + 1u] + dot_lanes[lane_base + 5u];
            const float pair2 =
                dot_lanes[lane_base + 2u] + dot_lanes[lane_base + 6u];
            const float pair3 =
                dot_lanes[lane_base + 3u] + dot_lanes[lane_base + 7u];
            const uint reduction_key_index = tile_start + thread_index;
            const bool valid_key =
                reduction_key_index < kv_len &&
                reduction_key_index <= global_pos &&
                !(sliding_window > 0u &&
                  reduction_key_index + sliding_window <= global_pos);
            probabilities[thread_index] = valid_key
                ? ((pair0 + pair1) + (pair2 + pair3)) * scale
                : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index == 0u) {
            const uint tile_len = min(KV_TILE, kv_len - tile_start);
            float tile_max = -INFINITY;
            for (uint index = 0u; index < tile_len; ++index) {
                tile_max = max(tile_max, probabilities[index]);
            }
            const float new_max = max(shared_row_max, tile_max);
            const float rescale = isfinite(shared_row_max)
                ? exp(shared_row_max - new_max)
                : 0.0f;
            shared_row_sum *= rescale;
            shared_rescale = rescale;
            float tile_sum = 0.0f;
            for (uint index = 0u; index < tile_len; ++index) {
                const float probability = isfinite(probabilities[index])
                    ? exp(probabilities[index] - new_max)
                    : 0.0f;
                probabilities[index] = probability;
                tile_sum += probability;
            }
            shared_row_sum += tile_sum;
            shared_row_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index < HEAD_DIM) {
            half acc = accumulator[thread_index] * (half)shared_rescale;
            const uint tile_len = min(KV_TILE, kv_len - tile_start);
            for (uint index = 0u; index < tile_len; ++index) {
                const uint key_row = tile_start + index;
                const uint value_base =
                    (key_row * num_kv_heads + kv_head) * HEAD_DIM;
                acc = fma(
                    value[value_base + thread_index],
                    (half)probabilities[index],
                    acc);
            }
            accumulator[thread_index] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index < HEAD_DIM) {
        const float inverse_sum =
            shared_row_sum > 0.0f ? 1.0f / shared_row_sum : 0.0f;
        output[query_base + thread_index] =
            (float)accumulator[thread_index] * inverse_sum;
    }
}
