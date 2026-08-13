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
