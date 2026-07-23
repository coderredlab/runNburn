#include <metal_stdlib>
using namespace metal;

static inline device const uchar* qwen_moe_selected_weight9(
    uint slot,
    device const uchar* w0,
    device const uchar* w1,
    device const uchar* w2,
    device const uchar* w3,
    device const uchar* w4,
    device const uchar* w5,
    device const uchar* w6,
    device const uchar* w7,
    device const uchar* w8)
{
    switch (slot) {
        case 0u: return w0;
        case 1u: return w1;
        case 2u: return w2;
        case 3u: return w3;
        case 4u: return w4;
        case 5u: return w5;
        case 6u: return w6;
        case 7u: return w7;
        default: return w8;
    }
}

#define QWEN_MOE_MAX_WEIGHT_TABLE 257

struct QwenMoeWeightTable {
    array<device const uchar*, QWEN_MOE_MAX_WEIGHT_TABLE> weight [[id(0)]];
};

static inline device const uchar* qwen_moe_table_weight(
    constant QwenMoeWeightTable& table,
    device const uint* expert_ids,
    uint slot)
{
    return table.weight[expert_ids[slot]];
}

kernel void qwen_moe_decode_route_shared(
    device const float* logits             [[buffer(0)]],
    device const float* input              [[buffer(1)]],
    device const float* shared_input_scale [[buffer(2)]],
    device uint*        expert_ids         [[buffer(3)]],
    device float*       route_weights      [[buffer(4)]],
    constant uint&      n_expert           [[buffer(5)]],
    constant uint&      n_used             [[buffer(6)]],
    constant uint&      hidden_dim         [[buffer(7)]],
    constant uint&      shared_expert_id   [[buffer(8)]],
    ushort tid [[thread_index_in_threadgroup]])
{
    if (n_expert > 256u || n_used == 0u || n_used > 31u) return;

    threadgroup float sorted_values[256];
    threadgroup uint sorted_ids[256];
    threadgroup float shared_dot_partials[256];

    float shared_dot = 0.0f;
    for (uint i = (uint)tid; i < hidden_dim; i += 256u) {
        shared_dot += input[i] * shared_input_scale[i];
    }
    shared_dot_partials[tid] = shared_dot;

    float logit = (uint)tid < n_expert ? logits[tid] : -INFINITY;
    bool selectable = !isnan(logit) && logit != -INFINITY;
    sorted_values[tid] = selectable ? logit : -INFINITY;
    sorted_ids[tid] = selectable ? (uint)tid : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fixed-size bitonic network: higher logit first, then lower expert id.
    for (uint size = 2u; size <= 256u; size <<= 1u) {
        for (uint stride = size >> 1u; stride != 0u; stride >>= 1u) {
            uint peer = (uint)tid ^ stride;
            if (peer > (uint)tid) {
                float lhs_value = sorted_values[tid];
                float rhs_value = sorted_values[peer];
                uint lhs_id = sorted_ids[tid];
                uint rhs_id = sorted_ids[peer];
                bool lhs_better = lhs_value > rhs_value ||
                    (lhs_value == rhs_value && lhs_id < rhs_id);
                bool descending = (((uint)tid & size) == 0u);
                if (lhs_better != descending) {
                    sorted_values[tid] = rhs_value;
                    sorted_values[peer] = lhs_value;
                    sorted_ids[tid] = rhs_id;
                    sorted_ids[peer] = lhs_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if ((uint)tid < stride) {
            shared_dot_partials[tid] += shared_dot_partials[(uint)tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float selected_max = sorted_values[0];
        float selected_sum = 0.0f;
        for (uint rank = 0u; rank < n_used; rank++) {
            selected_sum += exp(sorted_values[rank] - selected_max);
        }

        for (uint rank = 0u; rank < n_used; rank++) {
            expert_ids[rank] = sorted_ids[rank];
            route_weights[rank] = selected_sum != 0.0f
                ? exp(sorted_values[rank] - selected_max) / selected_sum
                : 0.0f;
        }
        expert_ids[n_used] = shared_expert_id;
        route_weights[n_used] =
            1.0f / (1.0f + exp(-shared_dot_partials[0]));
    }
}


kernel void qwen_moe_decode_q4k_slots(
    device const uchar* sparse_weight_bytes [[buffer(0)]],
    device const uchar* shared_weight_bytes [[buffer(1)]],
    device const float* input               [[buffer(2)]],
    device float*       out                 [[buffer(3)]],
    device const uint*  expert_ids          [[buffer(4)]],
    constant uint&      N                   [[buffer(5)]],
    constant uint&      K                   [[buffer(6)]],
    constant ulong&     per_expert_bytes    [[buffer(7)]],
    constant uint&      shared_expert_id    [[buffer(8)]],
    constant ulong&     sparse_byte_offset  [[buffer(9)]],
    constant ulong&     shared_byte_offset  [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = (group.x * threads.y + (uint)sg) * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    const uint expert = expert_ids[slot];
    device const uchar* base = expert == shared_expert_id
        ? shared_weight_bytes + shared_byte_offset
        : sparse_weight_bytes + sparse_byte_offset + (ulong)expert * per_expert_bytes;
    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 144u);
    device const uchar* x1 = x0 + nb * 144u;
    device const float* y4 = input + ix * 256u + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        {
            device const uchar*  blk = x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        y4 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q4k_down_slots(
    device const uchar* sparse_weight_bytes [[buffer(0)]],
    device const uchar* shared_weight_bytes [[buffer(1)]],
    device const float* input               [[buffer(2)]],
    device float*       out                 [[buffer(3)]],
    device const uint*  expert_ids          [[buffer(4)]],
    constant uint&      N                   [[buffer(5)]],
    constant uint&      K                   [[buffer(6)]],
    constant ulong&     per_expert_bytes    [[buffer(7)]],
    constant uint&      shared_expert_id    [[buffer(8)]],
    constant ulong&     sparse_byte_offset  [[buffer(9)]],
    constant ulong&     shared_byte_offset  [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = (group.x * threads.y + (uint)sg) * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    const uint expert = expert_ids[slot];
    device const uchar* base = expert == shared_expert_id
        ? shared_weight_bytes + shared_byte_offset
        : sparse_weight_bytes + sparse_byte_offset + (ulong)expert * per_expert_bytes;
    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 144u);
    device const uchar* x1 = x0 + nb * 144u;
    device const float* y4 = input + slot * K + ix * 256u + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        {
            device const uchar*  blk = x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        y4 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q4k_pair_slots(
    device const uchar* gate_sparse_weight_bytes [[buffer(0)]],
    device const uchar* gate_shared_weight_bytes [[buffer(1)]],
    device const uchar* up_sparse_weight_bytes   [[buffer(2)]],
    device const uchar* up_shared_weight_bytes   [[buffer(3)]],
    device const float* input                    [[buffer(4)]],
    device float*       gate_out                 [[buffer(5)]],
    device float*       up_out                   [[buffer(6)]],
    device const uint*  expert_ids               [[buffer(7)]],
    constant uint&      N                        [[buffer(8)]],
    constant uint&      K                        [[buffer(9)]],
    constant ulong&     gate_per_expert_bytes    [[buffer(10)]],
    constant ulong&     up_per_expert_bytes      [[buffer(11)]],
    constant uint&      shared_expert_id         [[buffer(12)]],
    constant ulong&     gate_sparse_byte_offset  [[buffer(13)]],
    constant ulong&     gate_shared_byte_offset  [[buffer(14)]],
    constant ulong&     up_sparse_byte_offset    [[buffer(15)]],
    constant ulong&     up_shared_byte_offset    [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    const uint expert = expert_ids[slot];
    device const uchar* gate_base = expert == shared_expert_id
        ? gate_shared_weight_bytes + gate_shared_byte_offset
        : gate_sparse_weight_bytes + gate_sparse_byte_offset + (ulong)expert * gate_per_expert_bytes;
    device const uchar* up_base = expert == shared_expert_id
        ? up_shared_weight_bytes + up_shared_byte_offset
        : up_sparse_weight_bytes + up_sparse_byte_offset + (ulong)expert * up_per_expert_bytes;

    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* gate_x0 = gate_base + first_row * (nb * 144u);
    device const uchar* gate_x1 = gate_x0 + nb * 144u;
    device const uchar* up_x0 = up_base + first_row * (nb * 144u);
    device const uchar* up_x1 = up_x0 + nb * 144u;
    device const float* y4 = input + ix * 256u + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        {
            device const uchar*  blk = gate_x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            gate_sum0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                         (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                         (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                         (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                         (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        {
            device const uchar*  blk = up_x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            up_sum0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                       (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                       (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                       (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                       (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        if (has_row1) {
            {
                device const uchar*  blk = gate_x1 + ib * 144u;
                device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
                device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
                device const half*   dh  = (device const half*)blk;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const ushort* q2 = q1 + 32u;
                float4 acc1 = {0.f,0.f,0.f,0.f};
                float4 acc2 = {0.f,0.f,0.f,0.f};
                for (ushort i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
                }
                gate_sum1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                             (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                             (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                             (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                             (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
            }

            {
                device const uchar*  blk = up_x1 + ib * 144u;
                device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
                device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
                device const half*   dh  = (device const half*)blk;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const ushort* q2 = q1 + 32u;
                float4 acc1 = {0.f,0.f,0.f,0.f};
                float4 acc2 = {0.f,0.f,0.f,0.f};
                for (ushort i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
                }
                up_sum1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                           (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                           (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                           (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                           (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
            }
        }

        y4 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float gate0 = simd_sum(gate_sum0);
    float up0 = simd_sum(up_sum0);
    if (lane == 0u) {
        gate_out[out_base + first_row] = gate0;
        up_out[out_base + first_row] = up0;
    }
    if (has_row1) {
        float gate1 = simd_sum(gate_sum1);
        float up1 = simd_sum(up_sum1);
        if (lane == 0u) {
            gate_out[out_base + first_row + 1u] = gate1;
            up_out[out_base + first_row + 1u] = up1;
        }
    }
}

kernel void qwen_moe_decode_q4k_selected_slots(
    device const uchar* w0    [[buffer(0)]],
    device const uchar* w1    [[buffer(1)]],
    device const uchar* w2    [[buffer(2)]],
    device const uchar* w3    [[buffer(3)]],
    device const uchar* w4    [[buffer(4)]],
    device const uchar* w5    [[buffer(5)]],
    device const uchar* w6    [[buffer(6)]],
    device const uchar* w7    [[buffer(7)]],
    device const uchar* w8    [[buffer(8)]],
    device const float* input [[buffer(9)]],
    device float*       out   [[buffer(10)]],
    constant uint&      N     [[buffer(11)]],
    constant uint&      K     [[buffer(12)]],
    constant uint&      slots [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint row = group.x;
    if (slot >= slots || row >= N) return;

    device const uchar* base = qwen_moe_selected_weight9(
        slot, w0, w1, w2, w3, w4, w5, w6, w7, w8);
    uint num_blocks = K / 256u;
    float acc = 0.0f;

    bool pow2 = (num_blocks & (num_blocks - 1u)) == 0u;
    bool sub_block = pow2 && num_blocks >= 2u && num_blocks <= 32u;

    if (sub_block) {
        uint m = min(4u, 32u / num_blocks);
        uint t = num_blocks * m;
        uint groups_per_lane = 4u / m;
        if (lane < t) {
            uint block_idx = lane / m;
            uint sub_idx   = lane % m;
            uint g_start   = sub_idx * groups_per_lane;

            device const uchar* blk = base + (row * num_blocks + block_idx) * 144u;
            ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
            ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
            float d    = (float)as_type<half>(d_bits);
            float dmin = (float)as_type<half>(dmin_bits);

            device const uchar* sc = blk + 4;
            float scales_f[8];
            float mins_f[8];
            for (uint j = 0; j < 8u; j++) {
                uchar s, mm;
                if (j < 4u) {
                    s  = sc[j]      & 63u;
                    mm = sc[j + 4u] & 63u;
                } else {
                    s  = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                    mm = (sc[j + 4u] >> 4u)   | ((sc[j]      >> 6u) << 4u);
                }
                scales_f[j] = (float)s;
                mins_f[j]   = (float)mm;
            }

            device const uchar* qs = blk + 16;
            uint x_base = block_idx * 256u;
            for (uint gi = 0; gi < groups_per_lane; gi++) {
                uint g = g_start + gi;
                uint is = g * 2u;
                float d1 = d * scales_f[is];
                float m1 = dmin * mins_f[is];
                float d2 = d * scales_f[is + 1u];
                float m2 = dmin * mins_f[is + 1u];

                uint q_off = g * 32u;
                uint y_off = g * 64u;
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] & 0x0Fu);
                    acc += (d1 * q - m1) * input[x_base + y_off + l];
                }
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] >> 4u);
                    acc += (d2 * q - m2) * input[x_base + y_off + 32u + l];
                }
            }
        }
    } else {
        for (uint b = lane; b < num_blocks; b += 32u) {
            device const uchar* blk = base + (row * num_blocks + b) * 144u;
            ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
            ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
            float d    = (float)as_type<half>(d_bits);
            float dmin = (float)as_type<half>(dmin_bits);

            device const uchar* sc = blk + 4;
            float scales_f[8];
            float mins_f[8];
            for (uint j = 0; j < 8u; j++) {
                uchar s, mm;
                if (j < 4u) {
                    s  = sc[j]      & 63u;
                    mm = sc[j + 4u] & 63u;
                } else {
                    s  = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                    mm = (sc[j + 4u] >> 4u)   | ((sc[j]      >> 6u) << 4u);
                }
                scales_f[j] = (float)s;
                mins_f[j]   = (float)mm;
            }

            device const uchar* qs = blk + 16;
            uint x_base = b * 256u;
            for (uint g = 0; g < 4u; g++) {
                uint is = g * 2u;
                float d1 = d * scales_f[is];
                float m1 = dmin * mins_f[is];
                float d2 = d * scales_f[is + 1u];
                float m2 = dmin * mins_f[is + 1u];

                uint q_off = g * 32u;
                uint y_off = g * 64u;
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] & 0x0Fu);
                    acc += (d1 * q - m1) * input[x_base + y_off + l];
                }
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] >> 4u);
                    acc += (d2 * q - m2) * input[x_base + y_off + 32u + l];
                }
            }
        }
    }

    float total = simd_sum(acc);
    if (lane == 0u) out[slot * N + row] = total;
}

kernel void qwen_moe_decode_q4k_selected_slots_coalesced(
    device const uchar* w0    [[buffer(0)]],
    device const uchar* w1    [[buffer(1)]],
    device const uchar* w2    [[buffer(2)]],
    device const uchar* w3    [[buffer(3)]],
    device const uchar* w4    [[buffer(4)]],
    device const uchar* w5    [[buffer(5)]],
    device const uchar* w6    [[buffer(6)]],
    device const uchar* w7    [[buffer(7)]],
    device const uchar* w8    [[buffer(8)]],
    device const float* input [[buffer(9)]],
    device float*       out   [[buffer(10)]],
    constant uint&      N     [[buffer(11)]],
    constant uint&      K     [[buffer(12)]],
    constant uint&      slots [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    device const uchar* base = qwen_moe_selected_weight9(
        slot, w0, w1, w2, w3, w4, w5, w6, w7, w8);
    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 144u);
    device const uchar* x1 = x0 + nb * 144u;
    device const float* y4 = input + ix * 256u + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        {
            device const uchar*  blk = x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        y4 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q4k_selected_slots_nsg2(
    device const uchar* w0    [[buffer(0)]],
    device const uchar* w1    [[buffer(1)]],
    device const uchar* w2    [[buffer(2)]],
    device const uchar* w3    [[buffer(3)]],
    device const uchar* w4    [[buffer(4)]],
    device const uchar* w5    [[buffer(5)]],
    device const uchar* w6    [[buffer(6)]],
    device const uchar* w7    [[buffer(7)]],
    device const uchar* w8    [[buffer(8)]],
    device const float* input [[buffer(9)]],
    device float*       out   [[buffer(10)]],
    constant uint&      N     [[buffer(11)]],
    constant uint&      K     [[buffer(12)]],
    constant uint&      slots [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = (group.x * 2u + (uint)sg) * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    device const uchar* base = qwen_moe_selected_weight9(
        slot, w0, w1, w2, w3, w4, w5, w6, w7, w8);
    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 144u);
    device const uchar* x1 = x0 + nb * 144u;
    device const float* y4 = input + ix * 256u + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        {
            device const uchar*  blk = x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        y4 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q4k_selected_pair_slots(
    device const uchar* gate_w0 [[buffer(0)]],
    device const uchar* gate_w1 [[buffer(1)]],
    device const uchar* gate_w2 [[buffer(2)]],
    device const uchar* gate_w3 [[buffer(3)]],
    device const uchar* gate_w4 [[buffer(4)]],
    device const uchar* gate_w5 [[buffer(5)]],
    device const uchar* gate_w6 [[buffer(6)]],
    device const uchar* gate_w7 [[buffer(7)]],
    device const uchar* gate_w8 [[buffer(8)]],
    device const uchar* up_w0   [[buffer(9)]],
    device const uchar* up_w1   [[buffer(10)]],
    device const uchar* up_w2   [[buffer(11)]],
    device const uchar* up_w3   [[buffer(12)]],
    device const uchar* up_w4   [[buffer(13)]],
    device const uchar* up_w5   [[buffer(14)]],
    device const uchar* up_w6   [[buffer(15)]],
    device const uchar* up_w7   [[buffer(16)]],
    device const uchar* up_w8   [[buffer(17)]],
    device const float* input   [[buffer(18)]],
    device float*       gate_out [[buffer(19)]],
    device float*       up_out   [[buffer(20)]],
    constant uint&      N        [[buffer(21)]],
    constant uint&      K        [[buffer(22)]],
    constant uint&      slots    [[buffer(23)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    device const uchar* gate_base = qwen_moe_selected_weight9(
        slot, gate_w0, gate_w1, gate_w2, gate_w3, gate_w4, gate_w5, gate_w6, gate_w7, gate_w8);
    device const uchar* up_base = qwen_moe_selected_weight9(
        slot, up_w0, up_w1, up_w2, up_w3, up_w4, up_w5, up_w6, up_w7, up_w8);
    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* gate_x0 = gate_base + first_row * (nb * 144u);
    device const uchar* gate_x1 = gate_x0 + nb * 144u;
    device const uchar* up_x0 = up_base + first_row * (nb * 144u);
    device const uchar* up_x1 = up_x0 + nb * 144u;
    device const float* y4 = input + ix * 256u + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        {
            device const uchar*  blk = gate_x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            gate_sum0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                         (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                         (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                         (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                         (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        {
            device const uchar*  blk = up_x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            up_sum0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                       (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                       (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                       (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                       (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        if (has_row1) {
            {
                device const uchar*  blk = gate_x1 + ib * 144u;
                device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
                device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
                device const half*   dh  = (device const half*)blk;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const ushort* q2 = q1 + 32u;
                float4 acc1 = {0.f,0.f,0.f,0.f};
                float4 acc2 = {0.f,0.f,0.f,0.f};
                for (ushort i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
                }
                gate_sum1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                             (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                             (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                             (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                             (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
            }

            {
                device const uchar*  blk = up_x1 + ib * 144u;
                device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
                device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
                device const half*   dh  = (device const half*)blk;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const ushort* q2 = q1 + 32u;
                float4 acc1 = {0.f,0.f,0.f,0.f};
                float4 acc2 = {0.f,0.f,0.f,0.f};
                for (ushort i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
                }
                up_sum1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                           (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                           (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                           (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                           (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
            }
        }

        y4 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float gate0 = simd_sum(gate_sum0);
    float up0 = simd_sum(up_sum0);
    if (lane == 0u) {
        gate_out[out_base + first_row] = gate0;
        up_out[out_base + first_row] = up0;
    }
    if (has_row1) {
        float gate1 = simd_sum(gate_sum1);
        float up1 = simd_sum(up_sum1);
        if (lane == 0u) {
            gate_out[out_base + first_row + 1u] = gate1;
            up_out[out_base + first_row + 1u] = up1;
        }
    }
}

kernel void qwen_moe_decode_silu_slots(
    device float* gate_up [[buffer(0)]],
    device const float* up [[buffer(1)]],
    constant uint& total [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    const float x = gate_up[gid];
    gate_up[gid] = (x / (1.0f + exp(-x))) * up[gid];
}

kernel void qwen_moe_decode_q5k_slots(
    device const uchar* sparse_weight_bytes [[buffer(0)]],
    device const uchar* shared_weight_bytes [[buffer(1)]],
    device const float* input               [[buffer(2)]],
    device float*       out                 [[buffer(3)]],
    device const uint*  expert_ids          [[buffer(4)]],
    constant uint&      N                   [[buffer(5)]],
    constant uint&      K                   [[buffer(6)]],
    constant ulong&     per_expert_bytes    [[buffer(7)]],
    constant uint&      shared_expert_id    [[buffer(8)]],
    constant ulong&     sparse_byte_offset  [[buffer(9)]],
    constant ulong&     shared_byte_offset  [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = (group.x * threads.y + (uint)sg) * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;
    const uint expert = expert_ids[slot];
    device const uchar* base = expert == shared_expert_id
        ? shared_weight_bytes + shared_byte_offset
        : sparse_weight_bytes + sparse_byte_offset + (ulong)expert * per_expert_bytes;

    // 2-way K split (ix = lane%2) keeps all 32 lanes active when K=512 (nb=2 super-blocks).
    // The old llama 4-way (ix=lane%4) idled half the simd-group at nb<4. Each old 4-way
    // tid's 8-element inner loop is halved across two lanes (`half`); simd_sum recombines
    // the partial dot products, so the math is bit-for-bit the same partition of the sum.
    const ushort ix   = lane % 2u;        // super-block stride (0,1)
    const ushort t    = lane / 2u;        // 0..15
    const ushort hf   = t % 2u;           // low/high half of the 8-elem inner loop
    const ushort tid4 = t / 2u;           // 0..7  (old 4-way tid)
    const ushort iq   = tid4 / 4u;        // 0 or 1
    const ushort ir   = tid4 % 4u;        // 0..3
    const ushort l0       = 8u * ir;
    const ushort lstart   = 4u * hf;
    const ushort q_offset = 32u * iq + l0;
    const ushort y_offset = 64u * iq + l0;
    const uchar hm1 = (uchar)(1u << (2u * iq));
    const uchar hm2 = (uchar)(hm1 << 1);
    const uchar hm3 = (uchar)(hm1 << 4);
    const uchar hm4 = (uchar)(hm2 << 4);
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 176u);
    device const uchar* x1 = x0 + nb * 176u;
    device const float* y1 = input + slot * K + ix * 256u + y_offset;

    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 2u) {
        device const float* y2 = y1 + 128u;
        float yl0[4], yl8[4], yh0[4], yh8[4];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort l = 0; l < 4; ++l) {
            ushort ll = lstart + l;
            yl0[l] = y1[ll +  0]; sumy[0] += yl0[l];
            yl8[l] = y1[ll + 32]; sumy[1] += yl8[l];
            yh0[l] = y2[ll +  0]; sumy[2] += yh0[l];
            yh8[l] = y2[ll + 32]; sumy[3] += yh8[l];
        }

        {
            device const uchar*  blk = x0 + ib * 176u;
            device const half*   dh  = (device const half*)blk;
            device const ushort* a   = (device const ushort*)(blk + 4u) + iq;
            device const uchar*  q1  = (blk + 48u) + q_offset;
            device const uchar*  qh  = (blk + 16u) + l0;
            device const uchar*  q2  = q1 + 64u;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort l = 0; l < 4; ++l) {
                ushort ll = lstart + l;
                uchar h = qh[ll];
                acc1[0] += yl0[l] * (q1[ll] & 0x0F);
                acc1[1] += yl8[l] * (q1[ll] & 0xF0);
                acc1[2] += yh0[l] * (q2[ll] & 0x0F);
                acc1[3] += yh8[l] * (q2[ll] & 0xF0);
                acc2[0] += (h & hm1) ? yl0[l] : 0.f;
                acc2[1] += (h & hm2) ? yl8[l] : 0.f;
                acc2[2] += (h & hm3) ? yh0[l] : 0.f;
                acc2[3] += (h & hm4) ? yh8[l] : 0.f;
            }
            sumf0 += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                     sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                     sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                     sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                     (float)dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 176u;
            device const half*   dh  = (device const half*)blk;
            device const ushort* a   = (device const ushort*)(blk + 4u) + iq;
            device const uchar*  q1  = (blk + 48u) + q_offset;
            device const uchar*  qh  = (blk + 16u) + l0;
            device const uchar*  q2  = q1 + 64u;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort l = 0; l < 4; ++l) {
                ushort ll = lstart + l;
                uchar h = qh[ll];
                acc1[0] += yl0[l] * (q1[ll] & 0x0F);
                acc1[1] += yl8[l] * (q1[ll] & 0xF0);
                acc1[2] += yh0[l] * (q2[ll] & 0x0F);
                acc1[3] += yh8[l] * (q2[ll] & 0xF0);
                acc2[0] += (h & hm1) ? yl0[l] : 0.f;
                acc2[1] += (h & hm2) ? yl8[l] : 0.f;
                acc2[2] += (h & hm3) ? yh0[l] : 0.f;
                acc2[3] += (h & hm4) ? yh8[l] : 0.f;
            }
            sumf1 += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                     sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                     sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                     sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                     (float)dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }

        y1 += 2u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q5k_selected_slots(
    device const uchar* w0    [[buffer(0)]],
    device const uchar* w1    [[buffer(1)]],
    device const uchar* w2    [[buffer(2)]],
    device const uchar* w3    [[buffer(3)]],
    device const uchar* w4    [[buffer(4)]],
    device const uchar* w5    [[buffer(5)]],
    device const uchar* w6    [[buffer(6)]],
    device const uchar* w7    [[buffer(7)]],
    device const uchar* w8    [[buffer(8)]],
    device const float* input [[buffer(9)]],
    device float*       out   [[buffer(10)]],
    constant uint&      N     [[buffer(11)]],
    constant uint&      K     [[buffer(12)]],
    constant uint&      slots [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;
    device const uchar* base = qwen_moe_selected_weight9(
        slot, w0, w1, w2, w3, w4, w5, w6, w7, w8);

    const ushort ix  = lane % 4u;
    const ushort tid = lane / 4u;
    const ushort iq  = tid / 4u;
    const ushort ir  = tid % 4u;
    const ushort l0       = 8u * ir;
    const ushort q_offset = 32u * iq + l0;
    const ushort y_offset = 64u * iq + l0;
    const uchar hm1 = (uchar)(1u << (2u * iq));
    const uchar hm2 = (uchar)(hm1 << 1);
    const uchar hm3 = (uchar)(hm1 << 4);
    const uchar hm4 = (uchar)(hm2 << 4);
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 176u);
    device const uchar* x1 = x0 + nb * 176u;
    device const float* y1 = input + slot * K + ix * 256u + y_offset;

    float yl[16];
    float yh[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        device const float* y2 = y1 + 128u;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        {
            device const uchar*  blk = x0 + ib * 176u;
            device const half*   dh  = (device const half*)blk;
            device const ushort* a   = (device const ushort*)(blk + 4u) + iq;
            device const uchar*  q1  = (blk + 48u) + q_offset;
            device const uchar*  qh  = (blk + 16u) + l0;
            device const uchar*  q2  = q1 + 64u;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort l = 0; l < 8; ++l) {
                uchar h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }
            sumf0 += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                     sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                     sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                     sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                     (float)dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 176u;
            device const half*   dh  = (device const half*)blk;
            device const ushort* a   = (device const ushort*)(blk + 4u) + iq;
            device const uchar*  q1  = (blk + 48u) + q_offset;
            device const uchar*  qh  = (blk + 16u) + l0;
            device const uchar*  q2  = q1 + 64u;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort l = 0; l < 8; ++l) {
                uchar h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }
            sumf1 += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                     sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                     sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                     sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                     (float)dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }

        y1 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q6k_slots(
    device const uchar* sparse_weight_bytes [[buffer(0)]],
    device const uchar* shared_weight_bytes [[buffer(1)]],
    device const float* input               [[buffer(2)]],
    device float*       out                 [[buffer(3)]],
    device const uint*  expert_ids          [[buffer(4)]],
    constant uint&      N                   [[buffer(5)]],
    constant uint&      K                   [[buffer(6)]],
    constant ulong&     per_expert_bytes    [[buffer(7)]],
    constant uint&      shared_expert_id    [[buffer(8)]],
    constant ulong&     sparse_byte_offset  [[buffer(9)]],
    constant ulong&     shared_byte_offset  [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = (group.x * threads.y + (uint)sg) * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;
    const uint expert = expert_ids[slot];
    device const uchar* base = expert == shared_expert_id
        ? shared_weight_bytes + shared_byte_offset
        : sparse_weight_bytes + sparse_byte_offset + (ulong)expert * per_expert_bytes;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const ushort tid = lane / 2u;
    const ushort ix  = lane % 2u;
    const ushort ip  = tid / 8u;
    const ushort il  = tid % 8u;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;
    const uint nb = K / 256u;

    device const uchar* x0 = base + first_row * (nb * 210u);
    device const uchar* x1 = x0 + nb * 210u;
    float yl[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = ix; ib < nb; ib += 2u) {
        device const float* y = input + slot * K + ib * 256u + y_offset;
        for (ushort l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        {
            device const uchar* blk = x0 + ib * 210u;
            device const uchar* q1  = (blk + 0u)   + q_offset_l;
            device const uchar* q2  = q1 + 32u;
            device const uchar* qh  = (blk + 128u) + q_offset_h;
            device const char*  sc  = (device const char*)(blk + 192u) + is;
            device const half*  dh  = (device const half*)(blk + 208u);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf0 += (float)dh[0] * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }

        if (has_row1) {
            device const uchar* blk = x1 + ib * 210u;
            device const uchar* q1  = (blk + 0u)   + q_offset_l;
            device const uchar* q2  = q1 + 32u;
            device const uchar* qh  = (blk + 128u) + q_offset_h;
            device const char*  sc  = (device const char*)(blk + 192u) + is;
            device const half*  dh  = (device const half*)(blk + 208u);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf1 += (float)dh[0] * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}


kernel void qwen_moe_decode_q6k_selected_slots(
    device const uchar* w0    [[buffer(0)]],
    device const uchar* w1    [[buffer(1)]],
    device const uchar* w2    [[buffer(2)]],
    device const uchar* w3    [[buffer(3)]],
    device const uchar* w4    [[buffer(4)]],
    device const uchar* w5    [[buffer(5)]],
    device const uchar* w6    [[buffer(6)]],
    device const uchar* w7    [[buffer(7)]],
    device const uchar* w8    [[buffer(8)]],
    device const float* input [[buffer(9)]],
    device float*       out   [[buffer(10)]],
    constant uint&      N     [[buffer(11)]],
    constant uint&      K     [[buffer(12)]],
    constant uint&      slots [[buffer(13)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;
    device const uchar* base = qwen_moe_selected_weight9(
        slot, w0, w1, w2, w3, w4, w5, w6, w7, w8);

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const ushort tid = lane / 2u;
    const ushort ix  = lane % 2u;
    const ushort ip  = tid / 8u;
    const ushort il  = tid % 8u;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;
    const uint nb = K / 256u;

    device const uchar* x0 = base + first_row * (nb * 210u);
    device const uchar* x1 = x0 + nb * 210u;
    float yl[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = ix; ib < nb; ib += 2u) {
        device const float* y = input + slot * K + ib * 256u + y_offset;
        for (ushort l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        {
            device const uchar* blk = x0 + ib * 210u;
            device const uchar* q1  = (blk + 0u)   + q_offset_l;
            device const uchar* q2  = q1 + 32u;
            device const uchar* qh  = (blk + 128u) + q_offset_h;
            device const char*  sc  = (device const char*)(blk + 192u) + is;
            device const half*  dh  = (device const half*)(blk + 208u);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf0 += (float)dh[0] * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }

        if (has_row1) {
            device const uchar* blk = x1 + ib * 210u;
            device const uchar* q1  = (blk + 0u)   + q_offset_l;
            device const uchar* q2  = q1 + 32u;
            device const uchar* qh  = (blk + 128u) + q_offset_h;
            device const char*  sc  = (device const char*)(blk + 192u) + is;
            device const half*  dh  = (device const half*)(blk + 208u);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf1 += (float)dh[0] * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q4k_table_slots(
    constant QwenMoeWeightTable& table [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device float*       out            [[buffer(2)]],
    device const uint*  expert_ids     [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      K              [[buffer(5)]],
    constant uint&      slots          [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    device const uchar* base = qwen_moe_table_weight(table, expert_ids, slot);
    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 144u);
    device const uchar* x1 = x0 + nb * 144u;
    device const float* y4 = input + ix * 256u + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        {
            device const uchar*  blk = x0 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf0 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 144u;
            device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half*   dh  = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf1 += (float)dh[0] * ((acc1[0]+1.f/256.f*acc1[1])*sc8[0] +
                                     (acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f +
                                     (acc2[0]+1.f/256.f*acc2[1])*sc8[4] +
                                     (acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f) -
                     (float)dh[1] * (sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
        }

        y4 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q5k_table_slots(
    constant QwenMoeWeightTable& table [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device float*       out            [[buffer(2)]],
    device const uint*  expert_ids     [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      K              [[buffer(5)]],
    constant uint&      slots          [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;
    device const uchar* base = qwen_moe_table_weight(table, expert_ids, slot);

    const ushort ix  = lane % 4u;
    const ushort tid = lane / 4u;
    const ushort iq  = tid / 4u;
    const ushort ir  = tid % 4u;
    const ushort l0       = 8u * ir;
    const ushort q_offset = 32u * iq + l0;
    const ushort y_offset = 64u * iq + l0;
    const uchar hm1 = (uchar)(1u << (2u * iq));
    const uchar hm2 = (uchar)(hm1 << 1);
    const uchar hm3 = (uchar)(hm1 << 4);
    const uchar hm4 = (uchar)(hm2 << 4);
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = base + first_row * (nb * 176u);
    device const uchar* x1 = x0 + nb * 176u;
    device const float* y1 = input + slot * K + ix * 256u + y_offset;

    float yl[16];
    float yh[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        device const float* y2 = y1 + 128u;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (ushort l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        {
            device const uchar*  blk = x0 + ib * 176u;
            device const half*   dh  = (device const half*)blk;
            device const ushort* a   = (device const ushort*)(blk + 4u) + iq;
            device const uchar*  q1  = (blk + 48u) + q_offset;
            device const uchar*  qh  = (blk + 16u) + l0;
            device const uchar*  q2  = q1 + 64u;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort l = 0; l < 8; ++l) {
                uchar h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }
            sumf0 += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                     sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                     sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                     sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                     (float)dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }

        if (has_row1) {
            device const uchar*  blk = x1 + ib * 176u;
            device const half*   dh  = (device const half*)blk;
            device const ushort* a   = (device const ushort*)(blk + 4u) + iq;
            device const uchar*  q1  = (blk + 48u) + q_offset;
            device const uchar*  qh  = (blk + 16u) + l0;
            device const uchar*  q2  = q1 + 64u;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (ushort l = 0; l < 8; ++l) {
                uchar h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }
            sumf1 += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                     sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                     sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                     sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                     (float)dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }

        y1 += 4u * 256u;
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_q6k_table_slots(
    constant QwenMoeWeightTable& table [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device float*       out            [[buffer(2)]],
    device const uint*  expert_ids     [[buffer(3)]],
    constant uint&      N              [[buffer(4)]],
    constant uint&      K              [[buffer(5)]],
    constant uint&      slots          [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint first_row = group.x * 2u;
    if (slot >= slots || first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;
    device const uchar* base = qwen_moe_table_weight(table, expert_ids, slot);

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const ushort tid = lane / 2u;
    const ushort ix  = lane % 2u;
    const ushort ip  = tid / 8u;
    const ushort il  = tid % 8u;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;
    const uint nb = K / 256u;

    device const uchar* x0 = base + first_row * (nb * 210u);
    device const uchar* x1 = x0 + nb * 210u;
    float yl[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = ix; ib < nb; ib += 2u) {
        device const float* y = input + slot * K + ib * 256u + y_offset;
        for (ushort l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        {
            device const uchar* blk = x0 + ib * 210u;
            device const uchar* q1  = (blk + 0u)   + q_offset_l;
            device const uchar* q2  = q1 + 32u;
            device const uchar* qh  = (blk + 128u) + q_offset_h;
            device const char*  sc  = (device const char*)(blk + 192u) + is;
            device const half*  dh  = (device const half*)(blk + 208u);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf0 += (float)dh[0] * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }

        if (has_row1) {
            device const uchar* blk = x1 + ib * 210u;
            device const uchar* q1  = (blk + 0u)   + q_offset_l;
            device const uchar* q2  = q1 + 32u;
            device const uchar* qh  = (blk + 128u) + q_offset_h;
            device const char*  sc  = (device const char*)(blk + 192u) + is;
            device const half*  dh  = (device const half*)(blk + 208u);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf1 += (float)dh[0] * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }
    }

    const uint out_base = slot * N;
    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[out_base + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[out_base + first_row + 1u] = t1;
    }
}

kernel void qwen_moe_decode_reduce_slots(
    device const float* down_slots    [[buffer(0)]],
    device const float* route_weights [[buffer(1)]],
    device float*       out           [[buffer(2)]],
    constant uint&      N             [[buffer(3)]],
    constant uint&      slots         [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= N) return;
    float acc = 0.0f;
    for (uint slot = 0; slot < slots; ++slot) {
        acc += route_weights[slot] * down_slots[slot * N + row];
    }
    out[row] = acc;
}

// ---------------------------------------------------------------------------
// Batched shared-expert scatter-add (MoE FFN B-fusion, milestone 5).
//
// 각 lane(threadgroup)에 대해:
//   rw   = sigmoid( dot(normed[lane], shared_input_scale) )    (route_shared 재현)
//   hidden[lane] += rw * down_shared[lane]
// down_shared[lane] 는 dense shared-expert(gate/up/silu/down)를 B-column GEMV 로
// weight 1회 읽기로 계산한 결과(bcol out 레이아웃 out[lane*n_embd + row]).
// route weight 는 route_shared 와 동일한 256-thread tree reduction 으로 bit-identical.
// grid = B threadgroup, tg = 256 thread.
// ---------------------------------------------------------------------------
kernel void qwen_moe_decode_shared_add(
    device float*       hidden              [[buffer(0)]],  // [B*n_embd] in/out
    device const float* normed              [[buffer(1)]],  // [B*n_embd]
    device const float* down_shared         [[buffer(2)]],  // [B*n_embd]
    device const float* shared_input_scale  [[buffer(3)]],  // [n_embd]
    constant uint&      n_embd              [[buffer(4)]],
    uint  lane_id [[threadgroup_position_in_grid]],
    ushort tid    [[thread_index_in_threadgroup]])
{
    threadgroup float partials[256];
    device const float* nm = normed + (uint)lane_id * n_embd;
    float dot = 0.0f;
    for (uint i = (uint)tid; i < n_embd; i += 256u) {
        dot += nm[i] * shared_input_scale[i];
    }
    partials[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if ((uint)tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rw = 1.0f / (1.0f + exp(-partials[0]));
    device float* h = hidden + (uint)lane_id * n_embd;
    device const float* ds = down_shared + (uint)lane_id * n_embd;
    for (uint i = (uint)tid; i < n_embd; i += 256u) {
        h[i] += rw * ds[i];
    }
}

// ---------------------------------------------------------------------------
// reduce+residual fusion (chain decode). reduce_slots 가 out 에 쓰고 이후 별도
// residual dispatch 가 hidden += out 하던 두 단계를, hidden 에 route 가중 합을 직접
// accumulate 하는 한 dispatch 로 합쳐 dispatch 경계 barrier 한 개를 제거한다. hidden
// 은 caller 가 byte offset 으로 바인딩한다(현재 chain single-token 은 slot 0=offset 0).
// 수학은 qwen_moe_decode_reduce_slots + residual_add 와 bit-identical.
kernel void qwen_moe_decode_reduce_add_slots(
    device const float* down_slots    [[buffer(0)]],
    device const float* route_weights [[buffer(1)]],
    device float*       hidden        [[buffer(2)]],
    constant uint&      N             [[buffer(3)]],
    constant uint&      slots         [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= N) return;
    float acc = 0.0f;
    for (uint slot = 0; slot < slots; ++slot) {
        acc += route_weights[slot] * down_slots[slot * N + row];
    }
    hidden[row] += acc;
}
