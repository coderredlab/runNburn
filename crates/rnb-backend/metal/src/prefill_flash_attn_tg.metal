#include <metal_stdlib>
using namespace metal;

constant constexpr uint Q = 8u;
constant constexpr uint C = 64u;
constant constexpr uint HD = 256u;
constant constexpr uint NSG = 4u;
constant constexpr uint SIMD_WIDTH = 32u;

// Dense causal GQA prefill attention for HD=256. Q, scores, and the f32 output
// accumulator stay in threadgroup memory; K/V are read directly from the cache.
kernel void attn_prefill_flash_tg(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& kv_len [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    threadgroup half* scratch [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const uint query_block = group.x;
    const uint head = group.y;
    if (head >= num_heads) return;

    const uint query_row0 = query_block * Q;
    const uint heads_per_group = num_heads / num_kv_heads;
    const uint kv_head = head / heads_per_group;
    const uint kv_dim = num_kv_heads * HD;

    threadgroup half* query = scratch;
    threadgroup float* accumulator = reinterpret_cast<threadgroup float*>(query + Q * HD);
    threadgroup float* scores = accumulator + Q * HD;

    for (uint i = tid; i < Q * HD; i += NSG * SIMD_WIDTH) {
        const uint row = i / HD;
        const uint dim = i % HD;
        const uint global_row = query_row0 + row;
        query[i] = global_row < seq_len
            ? half(q[(global_row * num_heads + head) * HD + dim])
            : half(0.0f);
        accumulator[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_sum[Q / NSG] = { 0.0f, 0.0f };
    float row_max[Q / NSG] = { -INFINITY, -INFINITY };
    const uint query_last = min(query_row0 + Q - 1u, seq_len - 1u);
    const uint max_global_position = (kv_len - seq_len) + query_last;

    for (uint key_col0 = 0u; key_col0 <= max_global_position; key_col0 += C) {
        // QK^T. Each SIMD-group owns two disjoint 8-column score tiles and
        // computes all eight query rows for each tile.
        for (uint chunk = 0u; chunk < C / (8u * NSG); ++chunk) {
            const uint key_col = key_col0 + (chunk * NSG + simdgroup_id) * 8u;
            simdgroup_float8x8 qk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (uint dim = 0u; dim < HD; dim += 16u) {
                simdgroup_half8x8 query_matrix[2];
                simdgroup_half8x8 key_matrix[2];
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(query_matrix[0], query + dim, HD);
                simdgroup_load(query_matrix[1], query + dim + 8u, HD);
                simdgroup_load(
                    key_matrix[0],
                    k_cache + key_col * kv_dim + kv_head * HD + dim,
                    kv_dim,
                    0,
                    true);
                simdgroup_load(
                    key_matrix[1],
                    k_cache + key_col * kv_dim + kv_head * HD + dim + 8u,
                    kv_dim,
                    0,
                    true);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(qk, query_matrix[0], key_matrix[0], qk);
                simdgroup_multiply_accumulate(qk, query_matrix[1], key_matrix[1], qk);
            }
            simdgroup_store(
                qk,
                scores + (chunk * NSG + simdgroup_id) * 8u,
                C,
                0,
                false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax. Each SIMD-group owns two query rows.
        for (uint local = 0u; local < Q / NSG; ++local) {
            const uint row = local * NSG + simdgroup_id;
            const uint global_row = query_row0 + row;
            const uint global_position = global_row < seq_len
                ? (kv_len - seq_len) + global_row
                : 0u;
            const uint score_col = lane * 2u;
            float2 value = float2(
                scores[row * C + score_col],
                scores[row * C + score_col + 1u]) * scale;
            if (global_row >= seq_len || key_col0 + score_col > global_position) {
                value[0] = -INFINITY;
            }
            if (global_row >= seq_len || key_col0 + score_col + 1u > global_position) {
                value[1] = -INFINITY;
            }

            const float old_max = row_max[local];
            const float new_max = simd_max(max(old_max, max(value[0], value[1])));
            const float rescale = old_max > -INFINITY ? exp(old_max - new_max) : 0.0f;
            const float2 probability = exp(value - new_max);
            row_sum[local] = row_sum[local] * rescale
                + simd_sum(probability[0] + probability[1]);
            row_max[local] = new_max;
            scores[row * C + score_col] = probability[0];
            scores[row * C + score_col + 1u] = probability[1];
            for (uint dim = lane; dim < HD; dim += SIMD_WIDTH) {
                accumulator[row * HD + dim] *= rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P*V. Each SIMD-group owns one quarter of the output columns and
        // accumulates all eight query rows in f32 SIMD matrices.
        simdgroup_float8x8 output_matrix[HD / (8u * NSG)];
        for (uint tile = 0u; tile < HD / (8u * NSG); ++tile) {
            const uint output_col = (tile * NSG + simdgroup_id) * 8u;
            simdgroup_load(output_matrix[tile], accumulator + output_col, HD);
        }
        for (uint key_tile = 0u; key_tile < C / 8u; ++key_tile) {
            simdgroup_float8x8 probability_matrix;
            simdgroup_load(probability_matrix, scores + key_tile * 8u, C);
            const uint value_row0 = key_col0 + key_tile * 8u;
            for (uint tile = 0u; tile < HD / (8u * NSG); ++tile) {
                const uint output_col = (tile * NSG + simdgroup_id) * 8u;
                simdgroup_half8x8 value_matrix;
                simdgroup_load(
                    value_matrix,
                    v_cache + value_row0 * kv_dim + kv_head * HD + output_col,
                    kv_dim);
                simdgroup_multiply_accumulate(
                    output_matrix[tile],
                    probability_matrix,
                    value_matrix,
                    output_matrix[tile]);
            }
        }
        for (uint tile = 0u; tile < HD / (8u * NSG); ++tile) {
            const uint output_col = (tile * NSG + simdgroup_id) * 8u;
            simdgroup_store(output_matrix[tile], accumulator + output_col, HD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint local = 0u; local < Q / NSG; ++local) {
        const uint row = local * NSG + simdgroup_id;
        const uint global_row = query_row0 + row;
        if (global_row < seq_len) {
            for (uint dim = lane; dim < HD; dim += SIMD_WIDTH) {
                out[(global_row * num_heads + head) * HD + dim]
                    = accumulator[row * HD + dim] / row_sum[local];
            }
        }
    }
}
