#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint KV_TILE = 16u;
constant constexpr uint HEAD_DIM = 256u;
constant constexpr uint SIMD_WIDTH = 32u;
constant constexpr uint HEAD_GROUPS = 8u;
constant constexpr uint SIMD_GROUPS = 8u;
constant constexpr uint QK_GROUPS = 1u;

static inline void produce_qk(
    threadgroup half* query,
    threadgroup half* k_tile,
    threadgroup float* scores)
{
    auto query_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        query, dextents<int32_t, 2>(HEAD_DIM, HEAD_GROUPS));
    auto key_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        k_tile, dextents<int32_t, 2>(KV_TILE, HEAD_DIM));
    auto score_tensor = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        scores, dextents<int32_t, 2>(KV_TILE, HEAD_GROUPS));
    constexpr auto descriptor = matmul2d_descriptor(
        HEAD_GROUPS, KV_TILE, HEAD_DIM, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<QK_GROUPS>> operation;
    auto output = operation.template get_destination_cooperative_tensor<
        decltype(query_tensor), decltype(key_tensor), float>();
    operation.run(query_tensor, key_tensor, output);
    output.store(score_tensor);
}

// M5 HD=256, GQA factor=8 int8 KV decode candidate.
//
// All SIMD-groups stage one shared half K/int8 V tile. One SIMD-group computes
// one 8x16 QK tile with Metal 4 tensor operations. A single buffer keeps
// threadgroup memory below 17 KiB; all eight query-head accumulators stay in
// registers.
kernel void attn_decode_i8_gqa_matrix_splitk_part(
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
    threadgroup uchar*  scratch      [[threadgroup(0)]],
    uint2 gid      [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    const uint kv_h = gid.x;
    const uint split = gid.y;
    const uint heads_per_group = num_heads / num_kv_heads;
    if (kv_h >= num_kv_heads || split >= num_splits ||
        head_dim != HEAD_DIM || heads_per_group != HEAD_GROUPS) {
        return;
    }

    const uint query_elements = HEAD_GROUPS * HEAD_DIM;
    const uint tile_elements = KV_TILE * HEAD_DIM;
    threadgroup half* query = reinterpret_cast<threadgroup half*>(scratch);
    threadgroup half* k_tile = query + query_elements;
    threadgroup char* v_tile =
        reinterpret_cast<threadgroup char*>(k_tile + tile_elements);
    threadgroup float* scores =
        reinterpret_cast<threadgroup float*>(v_tile + tile_elements);
    threadgroup float* k_scale_tile = scores + HEAD_GROUPS * KV_TILE;
    threadgroup float* v_scale_tile = k_scale_tile + KV_TILE;

    const uint base_head = kv_h * HEAD_GROUPS;
    for (uint i = tid; i < query_elements; i += SIMD_GROUPS * SIMD_WIDTH) {
        const uint query_head = i / HEAD_DIM;
        const uint dim = i % HEAD_DIM;
        query[i] = half(q[(base_head + query_head) * HEAD_DIM + dim]);
    }

    float acc[HEAD_DIM / SIMD_WIDTH];
    for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; ++i) {
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float s = 0.0f;

    const uint kv_dim = num_kv_heads * HEAD_DIM;
    const uint chunk = (kv_len + num_splits - 1u) / num_splits;
    const uint start = split * chunk;
    const uint end = min(kv_len, start + chunk);

    for (uint tile_start = start; tile_start < end; tile_start += KV_TILE) {
        const uint tile_len = min(KV_TILE, end - tile_start);
        for (uint i = tid; i < tile_elements; i += SIMD_GROUPS * SIMD_WIDTH) {
            const uint tile_row = i / HEAD_DIM;
            const uint dim = i % HEAD_DIM;
            if (tile_row < tile_len) {
                const uint kv_offset =
                    (tile_start + tile_row) * kv_dim + kv_h * HEAD_DIM + dim;
                k_tile[dim * KV_TILE + tile_row] = half(k_cache[kv_offset]);
                v_tile[i] = v_cache[kv_offset];
            } else {
                k_tile[dim * KV_TILE + tile_row] = half(0.0f);
                v_tile[i] = 0;
            }
        }
        if (tid < KV_TILE) {
            if (tid < tile_len) {
                const uint scale_index = (tile_start + tid) * num_kv_heads + kv_h;
                k_scale_tile[tid] = k_scale[scale_index];
                v_scale_tile[tid] = v_scale[scale_index];
            } else {
                k_scale_tile[tid] = 0.0f;
                v_scale_tile[tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup < QK_GROUPS) {
            produce_qk(query, k_tile, scores);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_row = 0u; tile_row < tile_len; ++tile_row) {
            const float x = scores[simdgroup * KV_TILE + tile_row]
                * scale * k_scale_tile[tile_row];
            const float value_scale = v_scale_tile[tile_row];
            const uint tile_offset = tile_row * HEAD_DIM;
            if (x > m) {
                const bool rescale = m > -INFINITY;
                const float alpha = rescale ? exp(m - x) : 1.0f;
                if (rescale) s *= alpha;
                for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; ++i) {
                    const uint dim = lane + i * SIMD_WIDTH;
                    float value = acc[i];
                    if (rescale) value *= alpha;
                    const float v = float(v_tile[tile_offset + dim]) * value_scale;
                    acc[i] = value + v;
                }
                s += 1.0f;
                m = x;
            } else {
                const float probability = exp(x - m);
                for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; ++i) {
                    const uint dim = lane + i * SIMD_WIDTH;
                    const float v = float(v_tile[tile_offset + dim]) * value_scale;
                    acc[i] += v * probability;
                }
                s += probability;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint h = base_head + simdgroup;
    const uint row = split * num_heads + h;
    if (lane == 0u) {
        partial_m[row] = m;
        partial_s[row] = s;
    }
    for (uint i = 0u; i < HEAD_DIM / SIMD_WIDTH; ++i) {
        const uint dim = lane + i * SIMD_WIDTH;
        partial_acc[row * HEAD_DIM + dim] = acc[i];
    }
}
