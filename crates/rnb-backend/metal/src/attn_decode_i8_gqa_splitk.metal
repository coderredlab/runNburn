#include <metal_stdlib>
using namespace metal;

constant constexpr uint KV_TILE = 8u;
constant constexpr uint HEAD_DIM = 256u;
constant constexpr uint SIMD_WIDTH = 32u;
constant constexpr uint HEADS_PER_GROUP = 8u;

// Long-context int8 KV decode attention for HD=256, GQA factor=8.
//
// One threadgroup owns one (KV head, sequence split). Its eight SIMD-groups
// process the eight query heads independently, so each lane keeps only one
// query/output accumulator. Quantized K/V tiles and their scales are loaded
// once into threadgroup memory and reused by every query-head SIMD-group.
// The per-head key order and split partial layout match attn_decode_i8_splitk.
kernel void attn_decode_i8_gqa_splitk_part(
    device const float* q            [[buffer(0)]],
    device const char*  k_cache      [[buffer(1)]],
    device const char*  v_cache      [[buffer(2)]],
    device const float* k_scale      [[buffer(3)]],
    device const float* v_scale      [[buffer(4)]],
    device float*       partial_acc  [[buffer(5)]],
    device float*       partial_m    [[buffer(6)]],
    device float*       partial_s    [[buffer(7)]],
    constant uint&      num_heads    [[buffer(8)]],
    constant uint&      num_kv_heads [[buffer(9)]],
    constant uint&      head_dim     [[buffer(10)]],
    constant uint&      kv_len       [[buffer(11)]],
    constant float&     scale        [[buffer(12)]],
    constant uint&      num_splits   [[buffer(13)]],
    uint2 gid      [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    uint kv_h = gid.x;
    uint split = gid.y;
    uint heads_per_group = num_heads / num_kv_heads;
    if (kv_h >= num_kv_heads || split >= num_splits ||
        head_dim != HEAD_DIM || heads_per_group != HEADS_PER_GROUP) {
        return;
    }

    uint h = kv_h * HEADS_PER_GROUP + simdgroup;
    uint row = split * num_heads + h;
    uint kv_dim = num_kv_heads * HEAD_DIM;
    uint q_off = h * HEAD_DIM;

    uint chunk = (kv_len + num_splits - 1u) / num_splits;
    uint start = split * chunk;
    uint end = min(kv_len, start + chunk);

    float qf[HEAD_DIM / SIMD_WIDTH];
    float acc[HEAD_DIM / SIMD_WIDTH];
    for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
        uint d = lane + i * SIMD_WIDTH;
        qf[i] = (float)(half)q[q_off + d];
        acc[i] = 0.0f;
    }

    threadgroup char k_tile[KV_TILE * HEAD_DIM];
    threadgroup char v_tile[KV_TILE * HEAD_DIM];
    threadgroup float k_scale_tile[KV_TILE];
    threadgroup float v_scale_tile[KV_TILE];

    float m = -INFINITY;
    float s = 0.0f;
    for (uint tile_start = start; tile_start < end; tile_start += KV_TILE) {
        uint tile_len = min(KV_TILE, end - tile_start);
        for (uint t = 0u; t < tile_len; t++) {
            uint j = tile_start + t;
            uint kv_off = j * kv_dim + kv_h * HEAD_DIM;
            uint tile_off = t * HEAD_DIM;
            k_tile[tile_off + tid] = k_cache[kv_off + tid];
            v_tile[tile_off + tid] = v_cache[kv_off + tid];
        }
        if (tid < tile_len) {
            uint sidx = (tile_start + tid) * num_kv_heads + kv_h;
            k_scale_tile[tid] = k_scale[sidx];
            v_scale_tile[tid] = v_scale[sidx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < tile_len; t++) {
            uint tile_off = t * HEAD_DIM;
            float partial = 0.0f;
            for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
                uint d = lane + i * SIMD_WIDTH;
                partial += qf[i] * (float)k_tile[tile_off + d];
            }
            float x = simd_sum(partial) * scale * k_scale_tile[t];
            float vsc = v_scale_tile[t];

            if (x > m) {
                bool rescale = (m > -INFINITY);
                float alpha = rescale ? exp(m - x) : 1.0f;
                if (rescale) s *= alpha;
                for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
                    uint d = lane + i * SIMD_WIDTH;
                    float a = acc[i];
                    if (rescale) a *= alpha;
                    float vv = (float)v_tile[tile_off + d] * vsc;
                    acc[i] = a + vv;
                }
                s += 1.0f;
                m = x;
            } else {
                float p = exp(x - m);
                for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; i++) {
                    uint d = lane + i * SIMD_WIDTH;
                    float vv = (float)v_tile[tile_off + d] * vsc;
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
