#include <metal_stdlib>
using namespace metal;

constant constexpr uint KV_TILE = 8u;
constant constexpr uint HEAD_DIM = 256u;
constant constexpr uint SIMD_WIDTH = 32u;
constant constexpr uint HEADS_PER_GROUP = 8u;

// Batched long-context f16 KV decode attention for HD=256, GQA factor=8.
//
// One threadgroup owns one (KV head, sequence split). One SIMD-group handles
// one (batch lane, query head), preserving the reference kernel's per-query
// key order and partial row layout. K/V tiles are loaded once into threadgroup
// memory and reused by all batch lanes and query heads.
kernel void attn_decode_f16_gqa_batch_splitk_part(
    device const float*  q            [[buffer(0)]],
    device const ushort* k_cache      [[buffer(1)]],
    device const ushort* v_cache      [[buffer(2)]],
    device float*        partial_acc  [[buffer(3)]],
    device float*        partial_m    [[buffer(4)]],
    device float*        partial_s    [[buffer(5)]],
    constant uint&       num_heads    [[buffer(6)]],
    constant uint&       num_kv_heads [[buffer(7)]],
    constant uint&       head_dim     [[buffer(8)]],
    device const uint*   kv_lens      [[buffer(9)]],
    constant float&      scale        [[buffer(10)]],
    constant uint&       num_splits   [[buffer(11)]],
    constant uint&       batch        [[buffer(12)]],
    threadgroup ushort*  kv_tile      [[threadgroup(0)]],
    uint2 gid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint lane       [[thread_index_in_simdgroup]],
    uint simdgroup  [[simdgroup_index_in_threadgroup]])
{
    uint kv_h = gid.x;
    uint split = gid.y;
    uint heads_per_group = num_heads / num_kv_heads;
    if (kv_h >= num_kv_heads || split >= num_splits || batch < 2u ||
        head_dim != HEAD_DIM || heads_per_group != HEADS_PER_GROUP) {
        return;
    }

    uint batch_idx = simdgroup / HEADS_PER_GROUP;
    uint query_head = simdgroup % HEADS_PER_GROUP;
    uint h = kv_h * HEADS_PER_GROUP + query_head;
    uint kv_len = kv_lens[batch_idx];
    uint chunk = (kv_len + num_splits - 1u) / num_splits;
    uint start = split * chunk;
    uint end = min(kv_len, start + chunk);
    uint row = (batch_idx * num_splits + split) * num_heads + h;
    uint q_off = (batch_idx * num_heads + h) * HEAD_DIM;
    uint kv_dim = num_kv_heads * HEAD_DIM;

    uint union_start = UINT_MAX;
    uint union_end = 0u;
    for (uint b = 0u; b < batch; b++) {
        uint lane_kv_len = kv_lens[b];
        uint lane_chunk = (lane_kv_len + num_splits - 1u) / num_splits;
        uint lane_start = split * lane_chunk;
        uint lane_end = min(lane_kv_len, lane_start + lane_chunk);
        union_start = min(union_start, lane_start);
        union_end = max(union_end, lane_end);
    }

    float qf[HEAD_DIM / SIMD_WIDTH];
    float acc[HEAD_DIM / SIMD_WIDTH];
    for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
        uint d = lane + i * SIMD_WIDTH;
        qf[i] = (float)(half)q[q_off + d];
        acc[i] = 0.0f;
    }

    threadgroup ushort* k_tile = kv_tile;
    threadgroup ushort* v_tile = kv_tile + KV_TILE * HEAD_DIM;
    uint threadgroup_width = batch * HEADS_PER_GROUP * SIMD_WIDTH;

    float m = -INFINITY;
    float s = 0.0f;
    for (uint tile_start = union_start; tile_start < union_end; tile_start += KV_TILE) {
        uint tile_len = min(KV_TILE, union_end - tile_start);
        uint tile_elements = tile_len * HEAD_DIM;
        for (uint x = tid; x < tile_elements; x += threadgroup_width) {
            uint t = x / HEAD_DIM;
            uint d = x % HEAD_DIM;
            uint j = tile_start + t;
            uint kv_off = j * kv_dim + kv_h * HEAD_DIM + d;
            k_tile[x] = k_cache[kv_off];
            v_tile[x] = v_cache[kv_off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < tile_len; t++) {
            uint j = tile_start + t;
            if (j < start || j >= end) {
                continue;
            }

            uint tile_off = t * HEAD_DIM;
            float partial = 0.0f;
            for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
                uint d = lane + i * SIMD_WIDTH;
                float kf = (float)as_type<half>(k_tile[tile_off + d]);
                partial += qf[i] * kf;
            }
            float x = simd_sum(partial) * scale;

            if (x > m) {
                bool rescale = (m > -INFINITY);
                float alpha = rescale ? exp(m - x) : 1.0f;
                if (rescale) s *= alpha;
                for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
                    uint d = lane + i * SIMD_WIDTH;
                    float a = acc[i];
                    if (rescale) a *= alpha;
                    float vv = (float)as_type<half>(v_tile[tile_off + d]);
                    acc[i] = a + vv;
                }
                s += 1.0f;
                m = x;
            } else {
                float p = exp(x - m);
                for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
                    uint d = lane + i * SIMD_WIDTH;
                    float vv = (float)as_type<half>(v_tile[tile_off + d]);
                    acc[i] += vv * p;
                }
                s += p;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        partial_m[row] = m;
        partial_s[row] = s;
    }
    for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
        uint d = lane + i * SIMD_WIDTH;
        partial_acc[row * HEAD_DIM + d] = acc[i];
    }
}
