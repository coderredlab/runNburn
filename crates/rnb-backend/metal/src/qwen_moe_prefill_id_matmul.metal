#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void qwen_moe_llama_id_map0(
    device const int  *selected_experts [[buffer(0)]],
    device uint       *tpe              [[buffer(1)]],
    device int        *ids              [[buffer(2)]],
    constant uint     &n_tokens         [[buffer(3)]],
    constant uint     &n_expert_used    [[buffer(4)]],
    constant uint     &n_expert         [[buffer(5)]],
    ushort expert [[thread_position_in_threadgroup]])
{
    if ((uint)expert >= n_expert) {
        return;
    }

    uint local = 0;
    device int *expert_ids = ids + (uint)expert * n_tokens;
    for (uint token = 0; token < n_tokens; ++token) {
        for (uint rank = 0; rank < n_expert_used; ++rank) {
            if (selected_experts[token * n_expert_used + rank] == (int)expert) {
                expert_ids[local++] = (int)(token * n_expert_used + rank);
            }
        }
    }
    tpe[expert] = local;
}

kernel void qwen_moe_llama_id_build_blocks(
    device const uint *tpe              [[buffer(0)]],
    device uint       *block_experts    [[buffer(1)]],
    device uint       *block_local0     [[buffer(2)]],
    device uint       *indirect_args    [[buffer(3)]],
    constant uint     &n_expert         [[buffer(4)]],
    constant uint     &hidden_dim       [[buffer(5)]],
    constant uint     &ffn_dim          [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) {
        return;
    }

    uint block_count = 0u;
    for (uint expert = 0u; expert < n_expert; ++expert) {
        const uint count = tpe[expert];
        for (uint local0 = 0u; local0 < count; local0 += 64u) {
            block_experts[block_count] = expert;
            block_local0[block_count] = local0;
            ++block_count;
        }
    }

    indirect_args[0] = (ffn_dim + 127u) / 128u;
    indirect_args[1] = block_count;
    indirect_args[2] = 1u;
    indirect_args[4] = (hidden_dim + 63u) / 64u;
    indirect_args[5] = block_count;
    indirect_args[6] = 1u;
}

kernel void gemm_q4k_tensorops_id(
    device const uchar *weight_bytes        [[buffer(0)]],
    device const float *input               [[buffer(1)]],
    device float       *out                 [[buffer(2)]],
    device const uint  *token_ids           [[buffer(3)]],
    device const uint  *expert_offsets      [[buffer(4)]],
    device const uint  *expert_counts       [[buffer(5)]],
    constant uint      &N                   [[buffer(6)]],
    constant uint      &K                   [[buffer(7)]],
    constant uint      &EXPERT_STRIDE_BYTES [[buffer(8)]],
    device const uint  *block_experts       [[buffer(9)]],
    device const uint  *block_local0        [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 32u, KC = 64u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint slot_base = expert_offsets[expert];
    uint nb_super = K / 256u;
    uint nchunk = K / KC;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  A_stage[64 * 64];
    threadgroup half  B_stage[64 * 32];
    threadgroup float C_stage[64 * 32];

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint local = local0 + t;
            if (local < count) {
                uint token = token_ids[slot_base + local];
                A_stage[i] = (half)input[token * K + c * KC + kk];
            } else {
                A_stage[i] = (half)0;
            }
        }
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                uint sb = c / 4u;
                uint g = c % 4u;
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    B_stage[l * BN + r] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    B_stage[(32u + l) * BN + r] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(64, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 64));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 64, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint local = local0 + t;
        uint row = row0 + r;
        if (local < count && row < N) {
            uint slot = slot_base + local;
            out[slot * N + row] = C_stage[t * BN + r];
        }
    }
}

kernel void qwen_moe_id_partial_reduce_scatter(
    device const float *partial       [[buffer(0)]],
    device float       *out           [[buffer(1)]],
    device const uint  *token_ids     [[buffer(2)]],
    device const float *route_weights [[buffer(3)]],
    constant uint      &ffn_tiles     [[buffer(4)]],
    constant uint      &tile_slots    [[buffer(5)]],
    constant uint      &hidden_tile   [[buffer(6)]],
    constant uint      &hidden_dim    [[buffer(7)]],
    constant uint      &hidden0       [[buffer(8)]],
    constant uint      &group_start   [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = tile_slots * hidden_tile;
    if (gid >= total) return;

    uint local = gid / hidden_tile;
    uint h = gid - local * hidden_tile;
    float sum = 0.0f;
    for (uint ft = 0; ft < ffn_tiles; ft++) {
        uint idx = (ft * tile_slots + local) * hidden_tile + h;
        sum += partial[idx];
    }

    uint slot = group_start + local;
    uint token = token_ids[slot];
    out[token * hidden_dim + hidden0 + h] += route_weights[slot] * sum;
}

inline float qwen_moe_q4k_value(device const uchar *expert_weight, uint row, uint kidx, uint K)
{
    uint nb_super = K / 256u;
    uint sb = kidx / 256u;
    uint inb = kidx - sb * 256u;
    uint group = inb / 32u;
    uint lane = inb - group * 32u;
    device const uchar *blk = expert_weight + (row * nb_super + sb) * 144u;
    ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
    ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
    float d = (float)as_type<half>(d_bits);
    float dmin = (float)as_type<half>(dmin_bits);
    device const uchar *scales = blk + 4;
    uint sc;
    uint mn;
    if (group < 4u) {
        sc = (uint)(scales[group] & 63u);
        mn = (uint)(scales[group + 4u] & 63u);
    } else {
        sc = (uint)((scales[group + 4u] & 0x0Fu) | ((scales[group - 4u] >> 6u) << 4u));
        mn = (uint)((scales[group + 4u] >> 4u) | ((scales[group] >> 6u) << 4u));
    }
    uchar packed = blk[16u + (group / 2u) * 32u + lane];
    uint q = (group & 1u) == 0u ? (uint)(packed & 0x0Fu) : (uint)(packed >> 4u);
    return d * (float)sc * (float)q - dmin * (float)mn;
}

kernel void qwen_moe_id_q4_gate_up_tile(
    device const uchar *gate_weight_bytes       [[buffer(0)]],
    device const uchar *up_weight_bytes         [[buffer(1)]],
    device const float *input                   [[buffer(2)]],
    device float       *gate_tile               [[buffer(3)]],
    device float       *up_tile                 [[buffer(4)]],
    device const uint  *token_ids               [[buffer(5)]],
    device const uint  *expert_offsets          [[buffer(6)]],
    device const uint  *expert_counts           [[buffer(7)]],
    device const uint  *block_experts           [[buffer(8)]],
    device const uint  *block_local0            [[buffer(9)]],
    constant uint      &N                       [[buffer(10)]],
    constant uint      &K                       [[buffer(11)]],
    constant uint      &GATE_WEIGHT_BYTE_OFFSET [[buffer(12)]],
    constant uint      &UP_WEIGHT_BYTE_OFFSET   [[buffer(13)]],
    constant uint      &GATE_EXPERT_STRIDE      [[buffer(14)]],
    constant uint      &UP_EXPERT_STRIDE        [[buffer(15)]],
    constant uint      &FFN_TILE0               [[buffer(16)]],
    constant uint      &FFN_TILE                [[buffer(17)]],
    constant uint      &TILE_SLOTS              [[buffer(18)]],
    uint gid [[thread_position_in_grid]])
{
    uint elems_per_block = TILE_SLOTS * FFN_TILE;
    uint block = gid / elems_per_block;
    uint inner = gid - block * elems_per_block;
    uint local_in_block = inner / FFN_TILE;
    uint col = inner - local_in_block * FFN_TILE;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint local = local0 + local_in_block;
    uint row = FFN_TILE0 + col;
    if (local >= count || row >= N) return;

    uint slot_base = expert_offsets[expert];
    uint slot = slot_base + local;
    uint token = token_ids[slot];
    device const uchar *gate_expert =
        gate_weight_bytes + GATE_WEIGHT_BYTE_OFFSET + expert * GATE_EXPERT_STRIDE;
    device const uchar *up_expert =
        up_weight_bytes + UP_WEIGHT_BYTE_OFFSET + expert * UP_EXPERT_STRIDE;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint kk = 0; kk < K; kk++) {
        float x = (float)((half)input[token * K + kk]);
        float gate_w = (float)((half)qwen_moe_q4k_value(gate_expert, row, kk, K));
        float up_w = (float)((half)qwen_moe_q4k_value(up_expert, row, kk, K));
        gate_acc += x * gate_w;
        up_acc += x * up_w;
    }
    gate_tile[slot * FFN_TILE + col] = gate_acc;
    up_tile[slot * FFN_TILE + col] = up_acc;
}

kernel void qwen_moe_id_silu_mul_tile(
    device const float *gate_tile [[buffer(0)]],
    device const float *up_tile   [[buffer(1)]],
    device float       *act_tile  [[buffer(2)]],
    constant uint      &ELEMS     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= ELEMS) return;
    float g = gate_tile[gid];
    act_tile[gid] = (g / (1.0f + exp(-g))) * up_tile[gid];
}

inline float qwen_moe_q6k_value(device const uchar *weight_bytes, uint row, uint kidx, uint K)
{
    uint nb_super = K / 256u;
    uint sb = kidx / 256u;
    uint inb = kidx - sb * 256u;
    uint half_idx = inb / 128u;
    uint within = inb - half_idx * 128u;
    device const uchar *blk = weight_bytes + (row * nb_super + sb) * 210u;
    device const uchar *ql = blk;
    device const uchar *qh = blk + 128;
    device const char  *sc = (device const char *)(blk + 192);
    ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
    float d = (float)as_type<half>(d_bits);
    uint ql_base = half_idx * 64u;
    uint qh_base = half_idx * 32u;
    uint sc_base = half_idx * 8u;
    uint l;
    int q;
    uint scale_idx;
    if (within < 32u) {
        l = within;
        q = (int)((ql[ql_base + l] & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
        scale_idx = sc_base + l / 16u;
    } else if (within < 64u) {
        l = within - 32u;
        q = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
        scale_idx = sc_base + l / 16u + 2u;
    } else if (within < 96u) {
        l = within - 64u;
        q = (int)((ql[ql_base + l] >> 4u) | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
        scale_idx = sc_base + l / 16u + 4u;
    } else {
        l = within - 96u;
        q = (int)((ql[ql_base + l + 32u] >> 4u) | (((qh[qh_base + l] >> 6u) & 3u) << 4u));
        scale_idx = sc_base + l / 16u + 6u;
    }
    return d * (float)sc[scale_idx] * (float)(q - 32);
}

kernel void qwen_moe_id_q6_down_partial_tile(
    device const uchar *weight_bytes [[buffer(0)]],
    device const float *act_tile     [[buffer(1)]],
    device float       *partial      [[buffer(2)]],
    constant uint      &WEIGHT_BYTE_OFFSET [[buffer(3)]],
    constant uint      &HIDDEN_DIM   [[buffer(4)]],
    constant uint      &HIDDEN0      [[buffer(5)]],
    constant uint      &HIDDEN_TILE  [[buffer(6)]],
    constant uint      &FFN_DIM      [[buffer(7)]],
    constant uint      &FFN_TILE0    [[buffer(8)]],
    constant uint      &FFN_TILE     [[buffer(9)]],
    constant uint      &SLOTS        [[buffer(10)]],
    constant uint      &FFN_TILE_IDX [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = SLOTS * HIDDEN_TILE;
    if (gid >= total) return;
    uint slot = gid / HIDDEN_TILE;
    uint h = gid - slot * HIDDEN_TILE;
    uint row = HIDDEN0 + h;
    if (row >= HIDDEN_DIM) return;
    device const uchar *weights = weight_bytes + WEIGHT_BYTE_OFFSET;
    float acc = 0.0f;
    for (uint kk = 0; kk < FFN_TILE; kk++) {
        uint kidx = FFN_TILE0 + kk;
        acc += act_tile[slot * FFN_TILE + kk] * qwen_moe_q6k_value(weights, row, kidx, FFN_DIM);
    }
    partial[(FFN_TILE_IDX * SLOTS + slot) * HIDDEN_TILE + h] = acc;
}

kernel void gemm_q4k_tensorops_id_v2_64x64(
    device const uchar *weight_bytes        [[buffer(0)]],
    device const float *input               [[buffer(1)]],
    device float       *out                 [[buffer(2)]],
    device const uint  *token_ids           [[buffer(3)]],
    device const uint  *expert_offsets      [[buffer(4)]],
    device const uint  *expert_counts       [[buffer(5)]],
    constant uint      &N                   [[buffer(6)]],
    constant uint      &K                   [[buffer(7)]],
    constant uint      &EXPERT_STRIDE_BYTES [[buffer(8)]],
    device const uint  *block_experts       [[buffer(9)]],
    device const uint  *block_local0        [[buffer(10)]],
    threadgroup char   *shmem               [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 64u, NK = 64u, NUM_THREADS = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint slot_base = expert_offsets[expert];
    uint nb_super = K / 256u;
    uint nchunk = K / NK;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  *input_stage  = (threadgroup half *)shmem;
    threadgroup half  *weight_stage = input_stage + BM * NK;
    threadgroup float *c_stage      = (threadgroup float *)(weight_stage + BN * NK);

    auto tInput = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)NK, (int)BM));
    auto tWeight = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)NK, (int)BN));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tInput), decltype(tWeight), float>();

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * NK; i += NUM_THREADS) {
            uint local = i / NK;
            uint kk = i % NK;
            if (local0 + local < count) {
                uint token = token_ids[slot_base + local0 + local];
                input_stage[i] = (half)input[token * K + c * NK + kk];
            } else {
                input_stage[i] = (half)0;
            }
        }

        uint sb = c / 4u;
        uint g = c % 4u;
        for (uint w = tid; w < BN; w += NUM_THREADS) {
            uint row = row0 + w;
            if (row < N) {
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    weight_stage[w * NK + l] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    weight_stage[w * NK + 32u + l] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) {
                    weight_stage[w * NK + k] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mInput = tInput.slice(0, 0);
        auto mWeight = tWeight.slice(0, 0);
        mm.run(mInput, mWeight, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        c_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < BM * BN; i += NUM_THREADS) {
        uint local = i / BN;
        uint r = i % BN;
        uint row = row0 + r;
        if (local0 + local < count && row < N) {
            uint slot = slot_base + local0 + local;
            out[slot * N + row] = c_stage[local * BN + r];
        }
    }
}

kernel void gemm_q4k_tensorops_id_v2_64x64_f16(
    device const uchar *weight_bytes        [[buffer(0)]],
    device const float *input               [[buffer(1)]],
    device half        *out                 [[buffer(2)]],
    device const uint  *token_ids           [[buffer(3)]],
    device const uint  *expert_offsets      [[buffer(4)]],
    device const uint  *expert_counts       [[buffer(5)]],
    constant uint      &N                   [[buffer(6)]],
    constant uint      &K                   [[buffer(7)]],
    constant uint      &EXPERT_STRIDE_BYTES [[buffer(8)]],
    device const uint  *block_experts       [[buffer(9)]],
    device const uint  *block_local0        [[buffer(10)]],
    threadgroup char   *shmem               [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 64u, NK = 64u, NUM_THREADS = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint slot_base = expert_offsets[expert];
    uint nb_super = K / 256u;
    uint nchunk = K / NK;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  *input_stage  = (threadgroup half *)shmem;
    threadgroup half  *weight_stage = input_stage + BM * NK;
    threadgroup float *c_stage      = (threadgroup float *)(weight_stage + BN * NK);

    auto tInput = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)NK, (int)BM));
    auto tWeight = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)NK, (int)BN));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tInput), decltype(tWeight), float>();

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * NK; i += NUM_THREADS) {
            uint local = i / NK;
            uint kk = i % NK;
            if (local0 + local < count) {
                uint token = token_ids[slot_base + local0 + local];
                input_stage[i] = (half)input[token * K + c * NK + kk];
            } else {
                input_stage[i] = (half)0;
            }
        }

        uint sb = c / 4u;
        uint g = c % 4u;
        for (uint w = tid; w < BN; w += NUM_THREADS) {
            uint row = row0 + w;
            if (row < N) {
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    weight_stage[w * NK + l] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    weight_stage[w * NK + 32u + l] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) {
                    weight_stage[w * NK + k] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mInput = tInput.slice(0, 0);
        auto mWeight = tWeight.slice(0, 0);
        mm.run(mInput, mWeight, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        c_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < BM * BN; i += NUM_THREADS) {
        uint local = i / BN;
        uint r = i % BN;
        uint row = row0 + r;
        if (local0 + local < count && row < N) {
            uint slot = slot_base + local0 + local;
            out[slot * N + row] = (half)c_stage[local * BN + r];
        }
    }
}

kernel void gemm_q4k_tensorops_id_v2_64x128_f16(
    device const uchar *weight_bytes        [[buffer(0)]],
    device const float *input               [[buffer(1)]],
    device half        *out                 [[buffer(2)]],
    device const uint  *token_ids           [[buffer(3)]],
    device const uint  *expert_offsets      [[buffer(4)]],
    device const uint  *expert_counts       [[buffer(5)]],
    constant uint      &N                   [[buffer(6)]],
    constant uint      &K                   [[buffer(7)]],
    constant uint      &EXPERT_STRIDE_BYTES [[buffer(8)]],
    device const uint  *block_experts       [[buffer(9)]],
    device const uint  *block_local0        [[buffer(10)]],
    threadgroup char   *shmem               [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 128u, NK = 64u, NUM_THREADS = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint slot_base = expert_offsets[expert];
    uint nb_super = K / 256u;
    uint nchunk = K / NK;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  *input_stage  = (threadgroup half *)shmem;
    threadgroup half  *weight_stage = input_stage + BM * NK;
    threadgroup float *c_stage      = (threadgroup float *)(weight_stage + BN * NK);

    auto tInput = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)NK, (int)BM));
    auto tWeight = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)NK, (int)BN));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tInput), decltype(tWeight), float>();

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * NK; i += NUM_THREADS) {
            uint local = i / NK;
            uint kk = i % NK;
            if (local0 + local < count) {
                uint token = token_ids[slot_base + local0 + local];
                input_stage[i] = (half)input[token * K + c * NK + kk];
            } else {
                input_stage[i] = (half)0;
            }
        }

        uint sb = c / 4u;
        uint g = c % 4u;
        for (uint w = tid; w < BN; w += NUM_THREADS) {
            uint row = row0 + w;
            if (row < N) {
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    weight_stage[w * NK + l] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    weight_stage[w * NK + 32u + l] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) {
                    weight_stage[w * NK + k] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mInput = tInput.slice(0, 0);
        auto mWeight = tWeight.slice(0, 0);
        mm.run(mInput, mWeight, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        c_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < BM * BN; i += NUM_THREADS) {
        uint local = i / BN;
        uint r = i % BN;
        uint row = row0 + r;
        if (local0 + local < count && row < N) {
            uint slot = slot_base + local0 + local;
            out[slot * N + row] = (half)c_stage[local * BN + r];
        }
    }
}

static inline void qwen_moe_chain_dequant_q4k_64(
    device const uchar *blk,
    uint g,
    threadgroup half *dst)
{
    ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
    ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
    float d    = (float)as_type<half>(d_bits);
    float dmin = (float)as_type<half>(dmin_bits);
    device const uchar *sc = blk + 4;
    uint is = g * 2u;
    uint i1 = is + 1u;
    uchar s0, m0, s1, m1;
    if (is < 4u) {
        s0 = sc[is] & 63u;
        m0 = sc[is + 4u] & 63u;
    } else {
        s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
        m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u);
    }
    if (i1 < 4u) {
        s1 = sc[i1] & 63u;
        m1 = sc[i1 + 4u] & 63u;
    } else {
        s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
        m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u);
    }
    float d1 = d * (float)s0;
    float mm1 = dmin * (float)m0;
    float d2 = d * (float)s1;
    float mm2 = dmin * (float)m1;
    device const uchar *qs = blk + 16 + g * 32u;
    for (uint l = 0; l < 32u; l++) {
        float ql = (float)(qs[l] & 0x0Fu);
        dst[l] = (half)(d1 * ql - mm1);
        float qh = (float)(qs[l] >> 4u);
        dst[32u + l] = (half)(d2 * qh - mm2);
    }
}

static inline void qwen_moe_chain_dequant_q5k_64(
    device const uchar *blk,
    uint g,
    threadgroup half *dst)
{
    ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
    ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
    float d    = (float)as_type<half>(d_bits);
    float dmin = (float)as_type<half>(dmin_bits);
    device const uchar *sc = blk + 4;
    uint is = g * 2u;
    uint i1 = is + 1u;
    uchar s0, m0, s1, m1;
    if (is < 4u) {
        s0 = sc[is] & 63u;
        m0 = sc[is + 4u] & 63u;
    } else {
        s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
        m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u);
    }
    if (i1 < 4u) {
        s1 = sc[i1] & 63u;
        m1 = sc[i1 + 4u] & 63u;
    } else {
        s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
        m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u);
    }
    float d1 = d * (float)s0;
    float mm1 = dmin * (float)m0;
    float d2 = d * (float)s1;
    float mm2 = dmin * (float)m1;
    device const uchar *qh = blk + 16;
    device const uchar *ql = blk + 48 + g * 32u;
    uchar u1 = (uchar)(1u << (2u * g));
    uchar u2 = (uchar)(2u << (2u * g));
    for (uint l = 0; l < 32u; l++) {
        float high1 = (qh[l] & u1) ? 16.0f : 0.0f;
        float qlow = (float)(ql[l] & 0x0Fu) + high1;
        dst[l] = (half)(d1 * qlow - mm1);
        float high2 = (qh[l] & u2) ? 16.0f : 0.0f;
        float qhigh = (float)(ql[l] >> 4u) + high2;
        dst[32u + l] = (half)(d2 * qhigh - mm2);
    }
}

template<
    uint BLOCK_BYTES,
    void (*dequantize_group)(device const uchar *, uint, threadgroup half *),
    typename input_t,
    bool INPUT_BY_TOKEN,
    bool COMPACT_BLOCKS,
    bool WIDE_COLS>
static inline void qwen_moe_chain_large_qk_impl(
    device const uchar *weight_bytes,
    device const input_t *input,
    device const uint *tpe,
    device const int *ids,
    device const uint *block_experts,
    device const uint *block_local0,
    device float *out,
    constant uint &N,
    constant uint &K,
    constant uint &N_TOKENS,
    constant uint &TOP_K,
    constant uint &EXPERT_STRIDE_BYTES,
    threadgroup char *shmem,
    uint3 tgid,
    uint tid)
{
    constexpr uint BM = 64u, BN = WIDE_COLS ? 128u : 64u;
    constexpr uint NK = 64u, NUM_THREADS = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = COMPACT_BLOCKS ? block_experts[block] : tgid.z;
    uint local0 = COMPACT_BLOCKS ? block_local0[block] : tgid.y * BM;
    uint count = tpe[expert];
    const uint min_expert_count = N_TOKENS >= 1024u ? 1u : 16u;
    if (count < min_expert_count || local0 >= count) {
        return;
    }
    uint nb_super = K / 256u;
    uint nchunk = K / NK;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  *input_stage  = (threadgroup half *)shmem;
    threadgroup half  *weight_stage = input_stage + BM * NK;
    threadgroup float *c_stage = WIDE_COLS
        ? (threadgroup float *)shmem
        : (threadgroup float *)(weight_stage + BN * NK);
    auto tInput = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)NK, (int)BM));
    auto tWeight = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)NK, (int)BN));
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tInput), decltype(tWeight), float>();

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * NK; i += NUM_THREADS) {
            uint local = i / NK;
            uint kk = i % NK;
            if (local0 + local < count) {
                uint route_slot = (uint)ids[expert * N_TOKENS + local0 + local];
                uint input_slot = INPUT_BY_TOKEN ? route_slot / TOP_K : route_slot;
                input_stage[i] = (half)input[input_slot * K + c * NK + kk];
            } else {
                input_stage[i] = (half)0;
            }
        }
        uint sb = c / 4u;
        uint g = c % 4u;
        for (uint w = tid; w < BN; w += NUM_THREADS) {
            uint row = row0 + w;
            if (row < N) {
                device const uchar *blk =
                    expert_weight + (row * nb_super + sb) * BLOCK_BYTES;
                dequantize_group(blk, g, weight_stage + w * NK);
            } else {
                for (uint k = 0; k < NK; k++) {
                    weight_stage[w * NK + k] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mInput = tInput.slice(0, 0);
        auto mWeight = tWeight.slice(0, 0);
        mm.run(mInput, mWeight, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        c_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BM * BN; i += NUM_THREADS) {
        uint local = i / BN;
        uint row = row0 + i % BN;
        if (local0 + local < count && row < N) {
            uint slot = (uint)ids[expert * N_TOKENS + local0 + local];
            out[slot * N + row] = c_stage[local * BN + i % BN];
        }
    }
}

#define QWEN_MOE_CHAIN_LARGE_ARGS(INPUT_T) \
    device const uchar *weight_bytes [[buffer(0)]], \
    device const INPUT_T *input [[buffer(1)]], \
    device const uint *tpe [[buffer(2)]], \
    device const int *ids [[buffer(3)]], \
    device float *out [[buffer(4)]], \
    constant uint &N [[buffer(5)]], \
    constant uint &K [[buffer(6)]], \
    constant uint &N_TOKENS [[buffer(7)]], \
    constant uint &TOP_K [[buffer(8)]], \
    constant uint &EXPERT_STRIDE_BYTES [[buffer(9)]], \
    device const uint *block_experts [[buffer(10)]], \
    device const uint *block_local0  [[buffer(11)]], \
    threadgroup char *shmem [[threadgroup(0)]], \
    uint3 tgid [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]

kernel void qwen_moe_chain_large_q4k_f32(QWEN_MOE_CHAIN_LARGE_ARGS(float))
{
    qwen_moe_chain_large_qk_impl<
        144u, qwen_moe_chain_dequant_q4k_64, float, true, true, false>(
        weight_bytes, input, tpe, ids, block_experts, block_local0, out,
        N, K, N_TOKENS, TOP_K, EXPERT_STRIDE_BYTES, shmem, tgid, tid);
}

kernel void qwen_moe_chain_large_q4k_f16(QWEN_MOE_CHAIN_LARGE_ARGS(half))
{
    qwen_moe_chain_large_qk_impl<
        144u, qwen_moe_chain_dequant_q4k_64, half, false, true, true>(
        weight_bytes, input, tpe, ids, block_experts, block_local0, out,
        N, K, N_TOKENS, TOP_K, EXPERT_STRIDE_BYTES, shmem, tgid, tid);
}

kernel void qwen_moe_chain_large_q5k_f16(QWEN_MOE_CHAIN_LARGE_ARGS(half))
{
    qwen_moe_chain_large_qk_impl<
        176u, qwen_moe_chain_dequant_q5k_64, half, false, true, false>(
        weight_bytes, input, tpe, ids, block_experts, block_local0, out,
        N, K, N_TOKENS, TOP_K, EXPERT_STRIDE_BYTES, shmem, tgid, tid);
}

kernel void qwen_moe_chain_large_q4k_f16_dense(QWEN_MOE_CHAIN_LARGE_ARGS(half))
{
    qwen_moe_chain_large_qk_impl<
        144u, qwen_moe_chain_dequant_q4k_64, half, false, false, true>(
        weight_bytes, input, tpe, ids, block_experts, block_local0, out,
        N, K, N_TOKENS, TOP_K, EXPERT_STRIDE_BYTES, shmem, tgid, tid);
}



template<bool COMPACT_BLOCKS, bool WIDE_ROWS>
static inline void qwen_moe_chain_large_q6k_impl(
    device const uchar *weight_bytes,
    device const half *input,
    device const uint *tpe,
    device const int *ids,
    device const uint *block_experts,
    device const uint *block_local0,
    device float *out,
    constant uint &N,
    constant uint &K,
    constant uint &N_TOKENS,
    constant uint &TOP_K,
    constant uint &EXPERT_STRIDE_BYTES,
    threadgroup char *shmem,
    uint3 tgid,
    uint tid)
{
    constexpr uint BM = 64u, BN = WIDE_ROWS ? 64u : 32u;
    constexpr uint NK = WIDE_ROWS ? 64u : 128u;
    constexpr uint NUM_THREADS = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = COMPACT_BLOCKS ? block_experts[block] : tgid.z;
    uint local0 = COMPACT_BLOCKS ? block_local0[block] : tgid.y * BM;
    uint count = tpe[expert];
    const uint min_expert_count = N_TOKENS >= 1024u ? 1u : 16u;
    if (count < min_expert_count || local0 >= count) {
        return;
    }
    uint nb_super = K / 256u;
    uint nchunk = K / NK;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  *input_stage  = (threadgroup half *)shmem;
    threadgroup half  *weight_stage = input_stage + BM * NK;
    threadgroup float *c_stage      = (threadgroup float *)(weight_stage + BN * NK);
    auto tInput = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)NK, (int)BM));
    auto tWeight = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)NK, (int)BN));
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tInput), decltype(tWeight), float>();

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * NK; i += NUM_THREADS) {
            uint local = i / NK;
            uint kk = i % NK;
            if (local0 + local < count) {
                uint slot = (uint)ids[expert * N_TOKENS + local0 + local];
                input_stage[i] = input[slot * K + c * NK + kk];
            } else {
                input_stage[i] = (half)0;
            }
        }
        uint sb = WIDE_ROWS ? c / 4u : c / 2u;
        uint half_index = WIDE_ROWS ? (c % 4u) / 2u : c % 2u;
        for (uint w = tid; w < BN; w += NUM_THREADS) {
            uint row = row0 + w;
            if (row < N) {
                device const uchar *blk =
                    expert_weight + (row * nb_super + sb) * 210u;
                device const uchar *ql = blk;
                device const uchar *qh = blk + 128;
                device const char *sc = (device const char *)(blk + 192);
                ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
                float d = (float)as_type<half>(d_bits);
                uint ql_base = half_index * 64u;
                uint qh_base = half_index * 32u;
                uint sc_base = half_index * 8u;
                threadgroup half *dst = weight_stage + w * NK;
                for (uint l = 0; l < 32u; l++) {
                    uint is = l / 16u;
                    int q1 = (int)((ql[ql_base + l] & 0x0Fu) |
                        (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                    int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) |
                        (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                    if (WIDE_ROWS) {
                        bool high_nibbles = (c & 1u) != 0u;
                        int q0 = high_nibbles
                            ? (int)((ql[ql_base + l] >> 4u) |
                                (((qh[qh_base + l] >> 4u) & 3u) << 4u))
                            : q1;
                        int q1_wide = high_nibbles
                            ? (int)((ql[ql_base + l + 32u] >> 4u) |
                                (((qh[qh_base + l] >> 6u) & 3u) << 4u))
                            : q2;
                        uint scale_base = sc_base + (high_nibbles ? 4u : 0u);
                        dst[l] =
                            (half)(d * (float)sc[scale_base + is] * (float)(q0 - 32));
                        dst[l + 32u] = (half)(
                            d * (float)sc[scale_base + is + 2u] * (float)(q1_wide - 32));
                    } else {
                        int q3 = (int)((ql[ql_base + l] >> 4u) |
                            (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                        int q4 = (int)((ql[ql_base + l + 32u] >> 4u) |
                            (((qh[qh_base + l] >> 6u) & 3u) << 4u));
                        dst[l] =
                            (half)(d * (float)sc[sc_base + is] * (float)(q1 - 32));
                        dst[l + 32u] = (half)(
                            d * (float)sc[sc_base + is + 2u] * (float)(q2 - 32));
                        dst[l + 64u] = (half)(
                            d * (float)sc[sc_base + is + 4u] * (float)(q3 - 32));
                        dst[l + 96u] = (half)(
                            d * (float)sc[sc_base + is + 6u] * (float)(q4 - 32));
                    }
                }
            } else {
                for (uint k = 0; k < NK; k++) {
                    weight_stage[w * NK + k] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mInput = tInput.slice(0, 0);
        auto mWeight = tWeight.slice(0, 0);
        mm.run(mInput, mWeight, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        c_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BM * BN; i += NUM_THREADS) {
        uint local = i / BN;
        uint row = row0 + i % BN;
        if (local0 + local < count && row < N) {
            uint slot = (uint)ids[expert * N_TOKENS + local0 + local];
            out[slot * N + row] = c_stage[local * BN + i % BN];
        }
    }
}

kernel void qwen_moe_chain_large_q6k_f16(QWEN_MOE_CHAIN_LARGE_ARGS(half))
{
    qwen_moe_chain_large_q6k_impl<true, true>(
        weight_bytes, input, tpe, ids, block_experts, block_local0, out,
        N, K, N_TOKENS, TOP_K, EXPERT_STRIDE_BYTES, shmem, tgid, tid);
}

kernel void qwen_moe_chain_large_q6k_f16_dense(QWEN_MOE_CHAIN_LARGE_ARGS(half))
{
    qwen_moe_chain_large_q6k_impl<false, true>(
        weight_bytes, input, tpe, ids, block_experts, block_local0, out,
        N, K, N_TOKENS, TOP_K, EXPERT_STRIDE_BYTES, shmem, tgid, tid);
}

#undef QWEN_MOE_CHAIN_LARGE_ARGS

kernel void gemm_q6k_tensorops_id(
    device const uchar *weight_bytes        [[buffer(0)]],
    device const float *input               [[buffer(1)]],
    device float       *out                 [[buffer(2)]],
    device const uint  *token_ids           [[buffer(3)]],
    device const uint  *expert_offsets      [[buffer(4)]],
    device const uint  *expert_counts       [[buffer(5)]],
    constant uint      &N                   [[buffer(6)]],
    constant uint      &K                   [[buffer(7)]],
    constant uint      &EXPERT_STRIDE_BYTES [[buffer(8)]],
    device const uint  *block_experts       [[buffer(9)]],
    device const uint  *block_local0        [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 32u, KC = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint slot_base = expert_offsets[expert];
    uint nb_super = K / 256u;
    uint nchunk = K / KC;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  A_stage[64 * 128];
    threadgroup half  B_stage[128 * 32];
    threadgroup float C_stage[64 * 32];

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint local = local0 + t;
            if (local < count) {
                uint slot = slot_base + local;
                A_stage[i] = (half)input[slot * K + c * KC + kk];
            } else {
                A_stage[i] = (half)0;
            }
        }
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                uint sb = c / 2u;
                uint n  = c % 2u;
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 210u;
                device const uchar *ql = blk;
                device const uchar *qh = blk + 128;
                device const char  *sc = (device const char *)(blk + 192);
                ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
                float d = (float)as_type<half>(d_bits);
                uint ql_base = n * 64u;
                uint qh_base = n * 32u;
                uint sc_base = n * 8u;
                for (uint l = 0; l < 32u; l++) {
                    uint is = l / 16u;
                    int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                    int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                    int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                    int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u));
                    float w1 = d * (float)sc[sc_base + is]      * (float)(q1 - 32);
                    float w2 = d * (float)sc[sc_base + is + 2u] * (float)(q2 - 32);
                    float w3 = d * (float)sc[sc_base + is + 4u] * (float)(q3 - 32);
                    float w4 = d * (float)sc[sc_base + is + 6u] * (float)(q4 - 32);
                    B_stage[(l)       * BN + r] = (half)w1;
                    B_stage[(l + 32u) * BN + r] = (half)w2;
                    B_stage[(l + 64u) * BN + r] = (half)w3;
                    B_stage[(l + 96u) * BN + r] = (half)w4;
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(128, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 128));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 128, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint local = local0 + t;
        uint row = row0 + r;
        if (local < count && row < N) {
            uint slot = slot_base + local;
            out[slot * N + row] = C_stage[t * BN + r];
        }
    }
}

kernel void qwen_moe_v3_q4_down(
    device const uchar *weight_bytes        [[buffer(0)]],
    device const half  *input               [[buffer(1)]],
    device float       *out                 [[buffer(2)]],
    device const uint  *dst_slots           [[buffer(3)]],
    device const uint  *expert_offsets      [[buffer(4)]],
    device const uint  *expert_counts       [[buffer(5)]],
    constant uint      &N                   [[buffer(6)]],
    constant uint      &K                   [[buffer(7)]],
    constant uint      &EXPERT_STRIDE_BYTES [[buffer(8)]],
    device const uint  *block_experts       [[buffer(9)]],
    device const uint  *block_local0        [[buffer(10)]],
    threadgroup char   *shmem               [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 64u, NK = 64u, NUM_THREADS = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint slot_base = expert_offsets[expert];
    uint nb_super = K / 256u;
    uint nchunk = K / NK;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  *input_stage  = (threadgroup half *)shmem;
    threadgroup half  *weight_stage = input_stage + BM * NK;
    threadgroup float *c_stage      = (threadgroup float *)(weight_stage + BN * NK);

    auto tInput = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)NK, (int)BM));
    auto tWeight = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)NK, (int)BN));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tInput), decltype(tWeight), float>();

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * NK; i += NUM_THREADS) {
            uint local = i / NK;
            uint kk = i % NK;
            if (local0 + local < count) {
                uint slot = slot_base + local0 + local;
                input_stage[i] = input[slot * K + c * NK + kk];
            } else {
                input_stage[i] = (half)0;
            }
        }

        uint sb = c / 4u;
        uint g = c % 4u;
        for (uint w = tid; w < BN; w += NUM_THREADS) {
            uint row = row0 + w;
            if (row < N) {
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    weight_stage[w * NK + l] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    weight_stage[w * NK + 32u + l] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) {
                    weight_stage[w * NK + k] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mInput = tInput.slice(0, 0);
        auto mWeight = tWeight.slice(0, 0);
        mm.run(mInput, mWeight, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        c_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < BM * BN; i += NUM_THREADS) {
        uint local = i / BN;
        uint r = i % BN;
        uint row = row0 + r;
        if (local0 + local < count && row < N) {
            uint slot = slot_base + local0 + local;
            out[dst_slots[slot] * N + row] = c_stage[local * BN + r];
        }
    }
}

kernel void qwen_moe_v3_q6_down(
    device const uchar *weight_bytes        [[buffer(0)]],
    device const half  *input               [[buffer(1)]],
    device float       *out                 [[buffer(2)]],
    device const uint  *dst_slots           [[buffer(3)]],
    device const uint  *expert_offsets      [[buffer(4)]],
    device const uint  *expert_counts       [[buffer(5)]],
    constant uint      &N                   [[buffer(6)]],
    constant uint      &K                   [[buffer(7)]],
    constant uint      &EXPERT_STRIDE_BYTES [[buffer(8)]],
    device const uint  *block_experts       [[buffer(9)]],
    device const uint  *block_local0        [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 32u, KC = 128u;
    uint row0 = tgid.x * BN;
    uint block = tgid.y;
    uint expert = block_experts[block];
    uint local0 = block_local0[block];
    uint count = expert_counts[expert];
    uint slot_base = expert_offsets[expert];
    uint nb_super = K / 256u;
    uint nchunk = K / KC;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  A_stage[64 * 128];
    threadgroup half  B_stage[128 * 32];
    threadgroup float C_stage[64 * 32];

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint local = local0 + t;
            if (local < count) {
                uint slot = slot_base + local;
                A_stage[i] = input[slot * K + c * KC + kk];
            } else {
                A_stage[i] = (half)0;
            }
        }
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                uint sb = c / 2u;
                uint n  = c % 2u;
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 210u;
                device const uchar *ql = blk;
                device const uchar *qh = blk + 128;
                device const char  *sc = (device const char *)(blk + 192);
                ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
                float d = (float)as_type<half>(d_bits);
                uint ql_base = n * 64u;
                uint qh_base = n * 32u;
                uint sc_base = n * 8u;
                for (uint l = 0; l < 32u; l++) {
                    uint is = l / 16u;
                    int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                    int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                    int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                    int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u));
                    float w1 = d * (float)sc[sc_base + is]      * (float)(q1 - 32);
                    float w2 = d * (float)sc[sc_base + is + 2u] * (float)(q2 - 32);
                    float w3 = d * (float)sc[sc_base + is + 4u] * (float)(q3 - 32);
                    float w4 = d * (float)sc[sc_base + is + 6u] * (float)(q4 - 32);
                    B_stage[(l)       * BN + r] = (half)w1;
                    B_stage[(l + 32u) * BN + r] = (half)w2;
                    B_stage[(l + 64u) * BN + r] = (half)w3;
                    B_stage[(l + 96u) * BN + r] = (half)w4;
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(128, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 128));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 128, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint local = local0 + t;
        uint row = row0 + r;
        if (local < count && row < N) {
            uint slot = slot_base + local;
            uint dst = dst_slots[slot];
            out[dst * N + row] = C_stage[t * BN + r];
        }
    }
}

kernel void qwen_moe_v4_q4_down_scatter(
    device const uchar *weight_bytes          [[buffer(0)]],
    device const half  *input                 [[buffer(1)]],
    device float       *out                   [[buffer(2)]],
    device const uint  *token_ids_sorted      [[buffer(3)]],
    device const float *route_weights_sorted  [[buffer(4)]],
    device const uint  *expert_rank_offsets   [[buffer(5)]],
    device const uint  *expert_rank_counts    [[buffer(6)]],
    constant uint      &N                     [[buffer(7)]],
    constant uint      &K                     [[buffer(8)]],
    constant uint      &EXPERT_STRIDE_BYTES   [[buffer(9)]],
    device const uint  *rank_block_experts    [[buffer(10)]],
    device const uint  *rank_block_local0     [[buffer(11)]],
    constant uint      &RANK_BLOCK_OFFSET     [[buffer(12)]],
    constant uint      &RANK_BLOCK_COUNT      [[buffer(13)]],
    constant uint      &RANK                  [[buffer(14)]],
    constant uint      &N_EXPERT_USED         [[buffer(15)]],
    threadgroup char   *shmem                 [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 64u, NK = 64u, NUM_THREADS = 128u;
    if (tgid.y >= RANK_BLOCK_COUNT) return;
    uint row0 = tgid.x * BN;
    uint block = RANK_BLOCK_OFFSET + tgid.y;
    uint expert = rank_block_experts[block];
    uint local0 = rank_block_local0[block];
    uint pair = expert * N_EXPERT_USED + RANK;
    uint count = expert_rank_counts[pair];
    uint slot_base = expert_rank_offsets[pair];
    uint nb_super = K / 256u;
    uint nchunk = K / NK;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  *input_stage  = (threadgroup half *)shmem;
    threadgroup half  *weight_stage = input_stage + BM * NK;
    threadgroup float *c_stage      = (threadgroup float *)(weight_stage + BN * NK);

    auto tInput = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)NK, (int)BM));
    auto tWeight = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)NK, (int)BN));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tInput), decltype(tWeight), float>();

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * NK; i += NUM_THREADS) {
            uint local = local0 + i / NK;
            uint kk = i % NK;
            if (local < count) {
                uint slot = slot_base + local;
                input_stage[i] = input[slot * K + c * NK + kk];
            } else {
                input_stage[i] = (half)0;
            }
        }

        uint sb = c / 4u;
        uint g = c % 4u;
        for (uint w = tid; w < BN; w += NUM_THREADS) {
            uint row = row0 + w;
            if (row < N) {
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    weight_stage[w * NK + l] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    weight_stage[w * NK + 32u + l] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) {
                    weight_stage[w * NK + k] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mInput = tInput.slice(0, 0);
        auto mWeight = tWeight.slice(0, 0);
        mm.run(mInput, mWeight, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        c_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < BM * BN; i += NUM_THREADS) {
        uint local = local0 + i / BN;
        uint r = i % BN;
        uint row = row0 + r;
        if (local < count && row < N) {
            uint slot = slot_base + local;
            uint token = token_ids_sorted[slot];
            float route = route_weights_sorted[slot];
            out[token * N + row] += route * c_stage[(i / BN) * BN + r];
        }
    }
}

kernel void qwen_moe_v4_q6_down_scatter(
    device const uchar *weight_bytes          [[buffer(0)]],
    device const half  *input                 [[buffer(1)]],
    device float       *out                   [[buffer(2)]],
    device const uint  *token_ids_sorted      [[buffer(3)]],
    device const float *route_weights_sorted  [[buffer(4)]],
    device const uint  *expert_rank_offsets   [[buffer(5)]],
    device const uint  *expert_rank_counts    [[buffer(6)]],
    constant uint      &N                     [[buffer(7)]],
    constant uint      &K                     [[buffer(8)]],
    constant uint      &EXPERT_STRIDE_BYTES   [[buffer(9)]],
    device const uint  *rank_block_experts    [[buffer(10)]],
    device const uint  *rank_block_local0     [[buffer(11)]],
    constant uint      &RANK_BLOCK_OFFSET     [[buffer(12)]],
    constant uint      &RANK_BLOCK_COUNT      [[buffer(13)]],
    constant uint      &RANK                  [[buffer(14)]],
    constant uint      &N_EXPERT_USED         [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint BM = 64u, BN = 32u, KC = 128u;
    if (tgid.y >= RANK_BLOCK_COUNT) return;
    uint row0 = tgid.x * BN;
    uint block = RANK_BLOCK_OFFSET + tgid.y;
    uint expert = rank_block_experts[block];
    uint local0 = rank_block_local0[block];
    uint pair = expert * N_EXPERT_USED + RANK;
    uint count = expert_rank_counts[pair];
    uint slot_base = expert_rank_offsets[pair];
    uint nb_super = K / 256u;
    uint nchunk = K / KC;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;

    threadgroup half  A_stage[64 * 128];
    threadgroup half  B_stage[128 * 32];
    threadgroup float C_stage[64 * 32];

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint local = local0 + t;
            if (local < count) {
                uint slot = slot_base + local;
                A_stage[i] = input[slot * K + c * KC + kk];
            } else {
                A_stage[i] = (half)0;
            }
        }
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                uint sb = c / 2u;
                uint n  = c % 2u;
                device const uchar *blk = expert_weight + (row * nb_super + sb) * 210u;
                device const uchar *ql = blk;
                device const uchar *qh = blk + 128;
                device const char  *sc = (device const char *)(blk + 192);
                ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
                float d = (float)as_type<half>(d_bits);
                uint ql_base = n * 64u;
                uint qh_base = n * 32u;
                uint sc_base = n * 8u;
                for (uint l = 0; l < 32u; l++) {
                    uint is = l / 16u;
                    int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                    int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                    int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                    int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u));
                    float w1 = d * (float)sc[sc_base + is]      * (float)(q1 - 32);
                    float w2 = d * (float)sc[sc_base + is + 2u] * (float)(q2 - 32);
                    float w3 = d * (float)sc[sc_base + is + 4u] * (float)(q3 - 32);
                    float w4 = d * (float)sc[sc_base + is + 6u] * (float)(q4 - 32);
                    B_stage[(l)       * BN + r] = (half)w1;
                    B_stage[(l + 32u) * BN + r] = (half)w2;
                    B_stage[(l + 64u) * BN + r] = (half)w3;
                    B_stage[(l + 96u) * BN + r] = (half)w4;
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(128, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 128));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 128, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint local = local0 + t;
        uint row = row0 + r;
        if (local < count && row < N) {
            uint slot = slot_base + local;
            uint token = token_ids_sorted[slot];
            float route = route_weights_sorted[slot];
            out[token * N + row] += route * C_stage[t * BN + r];
        }
    }
}

kernel void qwen_moe_v3_token_rank_combine(
    device const float *down_token_rank [[buffer(0)]],
    device const float *route_weights   [[buffer(1)]],
    device float       *out             [[buffer(2)]],
    constant uint      &seq_len         [[buffer(3)]],
    constant uint      &n_expert_used   [[buffer(4)]],
    constant uint      &hidden_dim      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = seq_len * hidden_dim;
    if (gid >= total) return;
    uint token = gid / hidden_dim;
    uint h = gid - token * hidden_dim;
    float sum = 0.0f;
    for (uint rank = 0; rank < n_expert_used; rank++) {
        uint tr = token * n_expert_used + rank;
        sum += route_weights[tr] * down_token_rank[tr * hidden_dim + h];
    }
    out[token * hidden_dim + h] += sum;
}

// llama.cpp kernel_mul_mm_id parity path (64 output rows x 32 routes x 32 K).
// The quantized blocks remain in their GGML byte layout and are dequantized
// directly into the 4 KiB weight half of dynamic threadgroup memory.
struct qwen_llama_block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

struct qwen_llama_block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

struct qwen_llama_block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

struct qwen_llama_block_q8_0 {
    half d;
    char qs[32];
};

// Byte-compatible with rnb_cpu::gemm::Q8KBlock.
struct qwen_llama_block_q8_K {
    float d;
    char qs[256];
    short bsums[16];
};

kernel void qwen_moe_llama_quantize_q8k_f32(
    device const float *input [[buffer(0)]],
    device qwen_llama_block_q8_K *output [[buffer(1)]],
    constant uint &n_blocks [[buffer(2)]],
    uint block_index [[thread_position_in_grid]])
{
#pragma clang fp reassociate(off) contract(off)
    if (block_index >= n_blocks) {
        return;
    }

    device const float *x = input + block_index * 256u;
    device qwen_llama_block_q8_K *y = output + block_index;
    float amax = 0.0f;
    float max_val = 0.0f;
    for (uint i = 0; i < 256u; ++i) {
        const float ax = fabs(x[i]);
        if (ax > amax) {
            amax = ax;
            max_val = x[i];
        }
    }
    if (amax == 0.0f) {
        y->d = 0.0f;
        for (uint i = 0; i < 256u; ++i) {
            y->qs[i] = 0;
        }
        for (uint i = 0; i < 16u; ++i) {
            y->bsums[i] = 0;
        }
        return;
    }

    const float iscale = -127.0f / max_val;
    y->d = 1.0f / iscale;
    for (uint group = 0; group < 16u; ++group) {
        int bsum = 0;
        for (uint i = 0; i < 16u; ++i) {
            const uint index = group * 16u + i;
            const int rounded = (int)rint(iscale * x[index]);
            const char q = (char)min(127, rounded);
            y->qs[index] = q;
            bsum += (int)q;
        }
        y->bsums[group] = (short)bsum;
    }
}

static inline uchar2 qwen_llama_get_scale_min_k4(
    int j,
    int k,
    device const uchar *q)
{
    return j < 4
        ? uchar2(q[j + k] & 63u, q[j + 4 + k] & 63u)
        : uchar2((q[j + 4 + k] & 0x0fu) | ((q[j - 4 + k] & 0xc0u) >> 2u),
                 (q[j + 4 + k] >> 4u) | ((q[j + k] & 0xc0u) >> 2u));
}

static inline float qwen_llama_dot_q4_K_q8_K(
    device const qwen_llama_block_q4_K *weights,
    device const qwen_llama_block_q8_K *activations,
    uint n_blocks)
{
#pragma clang fp reassociate(off) contract(off)
    float sumf = 0.0f;
    for (uint block = 0; block < n_blocks; ++block) {
        device const qwen_llama_block_q4_K *x = weights + block;
        device const qwen_llama_block_q8_K *y = activations + block;
        uchar sc[8];
        uchar mn[8];
        for (int j = 0; j < 8; ++j) {
            const uchar2 scale_min = qwen_llama_get_scale_min_k4(j, 0, x->scales);
            sc[j] = scale_min[0];
            mn[j] = scale_min[1];
        }

        const float d = y->d * (float)x->d;
        const float dmin = y->d * (float)x->dmin;
        int summ = 0;
        for (uint j = 0; j < 8u; ++j) {
            const int bsum =
                (int)y->bsums[2u * j] + (int)y->bsums[2u * j + 1u];
            summ += (int)mn[j] * bsum;
        }
        sumf -= dmin * (float)summ;

        int sumi1 = 0;
        int sumi2 = 0;
        for (uint group = 0; group < 4u; ++group) {
            int dot_low = 0;
            int dot_high = 0;
            for (uint i = 0; i < 32u; ++i) {
                const uchar q = x->qs[group * 32u + i];
                dot_low +=
                    (int)(q & 0x0fu) * (int)y->qs[group * 64u + i];
                dot_high +=
                    (int)(q >> 4u) * (int)y->qs[group * 64u + 32u + i];
            }
            sumi1 += dot_low * (int)sc[2u * group];
            sumi2 += dot_high * (int)sc[2u * group + 1u];
        }
        sumf += d * (float)(sumi1 + sumi2);
    }
    return sumf;
}

static inline float qwen_llama_dot_q6_K_q8_K(
    device const qwen_llama_block_q6_K *weights,
    device const qwen_llama_block_q8_K *activations,
    uint n_blocks)
{
#pragma clang fp reassociate(off) contract(off)
    float sum = 0.0f;
    for (uint block = 0; block < n_blocks; ++block) {
        device const qwen_llama_block_q6_K *x = weights + block;
        device const qwen_llama_block_q8_K *y = activations + block;
        int isum_mins = 0;
        for (uint j = 0; j < 16u; ++j) {
            isum_mins += (int)y->bsums[j] * (int)x->scales[j];
        }

        int isum = 0;
        for (uint batch = 0; batch < 2u; ++batch) {
            const uint ql_base = batch * 64u;
            const uint qh_base = batch * 32u;
            const uint q8_base = batch * 128u;
            const uint scale_base = batch * 8u;
            for (uint group = 0; group < 4u; ++group) {
                int dot_low = 0;
                int dot_high = 0;
                const uint qh_group = (group & 1u) * 16u;
                const uint low_shift = group < 2u ? 0u : 2u;
                const uint high_shift = group < 2u ? 4u : 6u;
                for (uint i = 0; i < 16u; ++i) {
                    const uchar ql = x->ql[ql_base + group * 16u + i];
                    const uchar qh = x->qh[qh_base + qh_group + i];
                    const int q_low = (int)(
                        (ql & 0x0fu) | (((qh >> low_shift) & 0x03u) << 4u));
                    const int q_high = (int)(
                        (ql >> 4u) | (((qh >> high_shift) & 0x03u) << 4u));
                    dot_low +=
                        q_low * (int)y->qs[q8_base + group * 16u + i];
                    dot_high +=
                        q_high * (int)y->qs[q8_base + 64u + group * 16u + i];
                }
                isum += dot_low * (int)x->scales[scale_base + group];
                isum += dot_high * (int)x->scales[scale_base + 4u + group];
            }
        }
        sum += (float)x->d * y->d * (float)(isum - 32 * isum_mins);
    }
    return sum;
}

static inline void qwen_llama_dequantize_q4_K(
    device const qwen_llama_block_q4_K *xb,
    short il,
    thread half4x4 &reg)
{
    device const uchar *q = xb->qs;
    short is = (il / 4) * 2;
    q += (il / 4) * 32 + 16 * (il & 1);
    il &= 3;
    const uchar2 sc = qwen_llama_get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? (float)xb->d : (float)xb->d / 16.0f;
    const float dl = d * (float)sc[0];
    const float ml = (float)xb->dmin * (float)sc[1];
    const ushort mask = il < 2 ? 0x0f : 0xf0;

    for (short i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (float)(q[i] & mask) - ml);
    }
}

static inline void qwen_llama_dequantize_q5_K(
    device const qwen_llama_block_q5_K *xb,
    short il,
    thread half4x4 &reg)
{
    device const uchar *q = xb->qs;
    device const uchar *qh = xb->qh;
    short is = (il / 4) * 2;
    q += 32 * (il / 4) + 16 * (il & 1);
    qh += 16 * (il & 1);
    const uchar ul = (uchar)(1u << (il / 2));
    il &= 3;
    const uchar2 sc = qwen_llama_get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? (float)xb->d : (float)xb->d / 16.0f;
    const float dl = d * (float)sc[0];
    const float ml = (float)xb->dmin * (float)sc[1];
    const ushort mask = il < 2 ? 0x0f : 0xf0;
    const float qh_value = il < 2 ? 16.0f : 256.0f;

    for (short i = 0; i < 16; ++i) {
        const float quant = (float)(q[i] & mask) + ((qh[i] & ul) ? qh_value : 0.0f);
        reg[i / 4][i % 4] = (half)(dl * quant - ml);
    }
}

static inline void qwen_llama_dequantize_q6_K(
    device const qwen_llama_block_q6_K *xb,
    short il,
    thread half4x4 &reg)
{
    const half d_all = xb->d;
    device const ushort *ql = (device const ushort *)xb->ql;
    device const ushort *qh = (device const ushort *)xb->qh;
    device const char *scales = xb->scales;

    ql += 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh += 16 * (il / 8) + 8 * (il & 1);
    const float sc = (float)scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;

    const uint mask_high =
        il > 1 ? (il > 2 ? 0xc0c0c0c0u : 0x30303030u)
               : (il > 0 ? 0x0c0c0c0cu : 0x03030303u);
    const uint mask_low = il > 1 ? 0xf0f0f0f0u : 0x0f0f0f0fu;
    const float ml = (float)d_all * sc * 32.0f;
    const float dl0 = (float)d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl1 / 256.0f;
    const float dl3 = dl2 / 256.0f;
    const uchar shift_high_right = il > 2 ? 2 : 0;
    const uchar shift_high_left = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uchar shift_low_right = il > 1 ? 4 : 0;

    for (short i = 0; i < 4; ++i) {
        const uint low = ((uint)ql[2 * i] | ((uint)ql[2 * i + 1] << 16)) & mask_low;
        const uint high = ((uint)qh[2 * i] | ((uint)qh[2 * i + 1] << 16)) & mask_high;
        const uint q = ((high << shift_high_left) >> shift_high_right) |
                       (low >> shift_low_right);
        reg[i][0] = (half)(dl0 * (float)(q & 0xffu) - ml);
        reg[i][1] = (half)(dl1 * (float)(q & 0xff00u) - ml);
        reg[i][2] = (half)(dl2 * (float)(q & 0xff0000u) - ml);
        reg[i][3] = (half)(dl3 * (float)(q & 0xff000000u) - ml);
    }
}

static inline void qwen_llama_dequantize_q8_0(
    device const qwen_llama_block_q8_0 *xb,
    short il,
    thread half4x4 &reg)
{
    const float d = (float)xb->d;
    for (short i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)((float)xb->qs[i + 16 * il] * d);
    }
}

template<
    bool SMALL_ONLY,
    typename block_q,
    void (*dequantize_func)(device const block_q *, short, thread half4x4 &),
    typename input_t,
    typename input_2x4_t>
static inline void qwen_moe_llama_mul_mm_id_impl(
    device const uchar *weight_bytes,
    device const input_t *input,
    device const uint *tpe,
    device const int *ids,
    device float *out,
    constant uint &N,
    constant uint &K,
    constant uint &N_TOKENS,
    constant uint &TOP_K,
    constant uint &EXPERT_STRIDE_BYTES,
    threadgroup char *shmem,
    uint3 tgid,
    ushort tid,
    ushort lane,
    ushort simdgroup)
{
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short NL = 16;

    const uint expert = tgid.z;
    const uint row0 = tgid.y * NR0;
    const uint local0 = tgid.x * NR1;
    const uint count = tpe[expert];
    const uint min_expert_count = N_TOKENS >= 1024u ? 1u : 16u;
    if (local0 >= count || (SMALL_ONLY && count >= min_expert_count)) {
        return;
    }

    const short nr0 = (short)min((uint)NR0, N - row0);
    const short nr1 = (short)min((uint)NR1, count - local0);
    const short lr0 = min((short)(tid / NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tid / NL1), (short)(nr1 - 1));
    const short il0 = tid % NL0;
    short il = il0;
    const short iy = 8 * (tid % NL1);

    const uint route_slot = (uint)ids[expert * N_TOKENS + local0 + lr1];
    const uint token = route_slot / TOP_K;
    const uint blocks_per_row = K / 256u;
    device const uchar *expert_weight = weight_bytes + expert * EXPERT_STRIDE_BYTES;
    device const block_q *x =
        (device const block_q *)expert_weight + (row0 + lr0) * blocks_per_row;
    device const input_t *y = input + token * K + iy;

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(NR1, NK));
    constexpr auto descriptor = matmul2d_descriptor(
        NR1, NR0, NK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 weight_values;
        dequantize_func(x, il, weight_values);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = i % 8;
            const short ly = (tid / NL0) % 8;
            sa[NK * (8 * sy + ly) + 8 * sx + lx] = weight_values[i / 4][i % 4];
        }

        const short sx = tid % NL1;
        const short sy = (tid / NL1) / 8;
        const short ly = (tid / NL1) % 8;
        const input_2x4_t input_values = *((device const input_2x4_t *)y);
        *((threadgroup half2x4 *)(sb + NK * (8 * sy + ly) + 8 * sx)) =
            (half2x4)input_values;

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x = il < 2 ? x + 1 : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = simdgroup; j < nr1; j += 4) {
        const uint slot = (uint)ids[expert * N_TOKENS + local0 + j];
        device float *dst = out + slot * N + row0;
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float *src = sc + j * NR0;
        threadgroup float4 *src4 = (threadgroup float4 *)src;

        int i = lane;
        for (; i < nr0 / 4; i += 32) {
            dst4[i] = src4[i];
        }
        i = 4 * (nr0 / 4) + lane;
        for (; i < nr0; i += 32) {
            dst[i] = src[i];
        }
    }
}

#define QWEN_LLAMA_ID_KERNEL_ARGS(INPUT_T) \
    device const uchar *weight_bytes [[buffer(0)]], \
    device const INPUT_T *input [[buffer(1)]], \
    device const uint *tpe [[buffer(2)]], \
    device const int *ids [[buffer(3)]], \
    device float *out [[buffer(4)]], \
    constant uint &N [[buffer(5)]], \
    constant uint &K [[buffer(6)]], \
    constant uint &N_TOKENS [[buffer(7)]], \
    constant uint &TOP_K [[buffer(8)]], \
    constant uint &EXPERT_STRIDE_BYTES [[buffer(9)]], \
    threadgroup char *shmem [[threadgroup(0)]], \
    uint3 tgid [[threadgroup_position_in_grid]], \
    ushort tid [[thread_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], \
    ushort simdgroup [[simdgroup_index_in_threadgroup]]

#define QWEN_LLAMA_ID_CALL(SMALL_ONLY, BLOCK_T, DEQUANT, INPUT_T, INPUT_VEC_T) \
    qwen_moe_llama_mul_mm_id_impl<SMALL_ONLY, BLOCK_T, DEQUANT, INPUT_T, INPUT_VEC_T>( \
        weight_bytes, input, tpe, ids, out, N, K, N_TOKENS, TOP_K, \
        EXPERT_STRIDE_BYTES, shmem, tgid, tid, lane, simdgroup)

kernel void qwen_moe_llama_mul_mm_id_q4k_f32(QWEN_LLAMA_ID_KERNEL_ARGS(float))
{
    QWEN_LLAMA_ID_CALL(
        false, qwen_llama_block_q4_K, qwen_llama_dequantize_q4_K, float, float2x4);
}

kernel void qwen_moe_llama_mul_mm_id_q4k_f16(QWEN_LLAMA_ID_KERNEL_ARGS(half))
{
    QWEN_LLAMA_ID_CALL(
        false, qwen_llama_block_q4_K, qwen_llama_dequantize_q4_K, half, half2x4);
}

kernel void qwen_moe_llama_mul_mm_id_q5k_f32(QWEN_LLAMA_ID_KERNEL_ARGS(float))
{
    QWEN_LLAMA_ID_CALL(
        false, qwen_llama_block_q5_K, qwen_llama_dequantize_q5_K, float, float2x4);
}

kernel void qwen_moe_llama_mul_mm_id_q5k_f16(QWEN_LLAMA_ID_KERNEL_ARGS(half))
{
    QWEN_LLAMA_ID_CALL(
        false, qwen_llama_block_q5_K, qwen_llama_dequantize_q5_K, half, half2x4);
}

kernel void qwen_moe_llama_mul_mm_id_q6k_f32(QWEN_LLAMA_ID_KERNEL_ARGS(float))
{
    QWEN_LLAMA_ID_CALL(
        false, qwen_llama_block_q6_K, qwen_llama_dequantize_q6_K, float, float2x4);
}

kernel void qwen_moe_llama_mul_mm_id_q6k_f16(QWEN_LLAMA_ID_KERNEL_ARGS(half))
{
    QWEN_LLAMA_ID_CALL(
        false, qwen_llama_block_q6_K, qwen_llama_dequantize_q6_K, half, half2x4);
}


kernel void qwen_moe_chain_small_q4k_f32(QWEN_LLAMA_ID_KERNEL_ARGS(float))
{
    QWEN_LLAMA_ID_CALL(
        true, qwen_llama_block_q4_K, qwen_llama_dequantize_q4_K, float, float2x4);
}

kernel void qwen_moe_chain_small_q5k_f32(QWEN_LLAMA_ID_KERNEL_ARGS(float))
{
    QWEN_LLAMA_ID_CALL(
        true, qwen_llama_block_q5_K, qwen_llama_dequantize_q5_K, float, float2x4);
}

kernel void qwen_moe_chain_small_q6k_f32(QWEN_LLAMA_ID_KERNEL_ARGS(float))
{
    QWEN_LLAMA_ID_CALL(
        true, qwen_llama_block_q6_K, qwen_llama_dequantize_q6_K, float, float2x4);
}

template<
    typename block_q,
    float (*dot_func)(
        device const block_q *,
        device const qwen_llama_block_q8_K *,
        uint)>
static inline void qwen_moe_llama_mul_mm_id_q8k_impl(
    device const uchar *weight_bytes,
    device const qwen_llama_block_q8_K *input,
    device const uint *tpe,
    device const int *ids,
    device float *out,
    constant uint &N,
    constant uint &K,
    constant uint &N_TOKENS,
    constant uint &TOP_K,
    constant uint &EXPERT_STRIDE_BYTES,
    uint3 gid)
{
    const uint row = gid.x;
    const uint local = gid.y;
    const uint expert = gid.z;
    const uint count = tpe[expert];
    const uint min_expert_count = N_TOKENS >= 1024u ? 1u : 16u;
    if (row >= N || local >= count || count >= min_expert_count) {
        return;
    }

    const uint slot = (uint)ids[expert * N_TOKENS + local];
    const uint input_row = slot / TOP_K;
    const uint n_blocks = K / 256u;
    device const uchar *expert_weight =
        weight_bytes + expert * EXPERT_STRIDE_BYTES;
    device const block_q *weight_row =
        (device const block_q *)expert_weight + row * n_blocks;
    device const qwen_llama_block_q8_K *input_row_blocks =
        input + input_row * n_blocks;
    out[slot * N + row] = dot_func(weight_row, input_row_blocks, n_blocks);
}

#define QWEN_LLAMA_ID_Q8K_KERNEL_ARGS \
    device const uchar *weight_bytes [[buffer(0)]], \
    device const qwen_llama_block_q8_K *input [[buffer(1)]], \
    device const uint *tpe [[buffer(2)]], \
    device const int *ids [[buffer(3)]], \
    device float *out [[buffer(4)]], \
    constant uint &N [[buffer(5)]], \
    constant uint &K [[buffer(6)]], \
    constant uint &N_TOKENS [[buffer(7)]], \
    constant uint &TOP_K [[buffer(8)]], \
    constant uint &EXPERT_STRIDE_BYTES [[buffer(9)]], \
    uint3 gid [[thread_position_in_grid]]

kernel void qwen_moe_llama_mul_mm_id_q4k_q8k(
    QWEN_LLAMA_ID_Q8K_KERNEL_ARGS)
{
    qwen_moe_llama_mul_mm_id_q8k_impl<
        qwen_llama_block_q4_K,
        qwen_llama_dot_q4_K_q8_K>(
            weight_bytes, input, tpe, ids, out, N, K, N_TOKENS, TOP_K,
            EXPERT_STRIDE_BYTES, gid);
}

kernel void qwen_moe_llama_mul_mm_id_q6k_q8k(
    QWEN_LLAMA_ID_Q8K_KERNEL_ARGS)
{
    qwen_moe_llama_mul_mm_id_q8k_impl<
        qwen_llama_block_q6_K,
        qwen_llama_dot_q6_K_q8_K>(
            weight_bytes, input, tpe, ids, out, N, K, N_TOKENS, TOP_K,
            EXPERT_STRIDE_BYTES, gid);
}

#undef QWEN_LLAMA_ID_Q8K_KERNEL_ARGS

kernel void qwen_moe_chain_cast_large_slots_f32_f16(
    device const float *input [[buffer(0)]],
    device const uint *tpe [[buffer(1)]],
    device const int *ids [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant uint &N_TOKENS [[buffer(4)]],
    constant uint &DIM [[buffer(5)]],
    constant uint &INPUT_BY_TOKEN [[buffer(6)]],
    constant uint &TOP_K [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint local = gid.y;
    uint expert = gid.z;
    uint count = tpe[expert];
    const uint min_expert_count = N_TOKENS >= 1024u ? 1u : 16u;
    if (column >= DIM || local >= count || count < min_expert_count) {
        return;
    }
    uint slot = (uint)ids[expert * N_TOKENS + local];
    uint input_row = INPUT_BY_TOKEN != 0u ? slot / TOP_K : slot;
    out[slot * DIM + column] = (half)input[input_row * DIM + column];
}
#undef QWEN_LLAMA_ID_CALL
#undef QWEN_LLAMA_ID_KERNEL_ARGS


template<
    typename block_q,
    void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
static inline void qwen_moe_shared_mul_mm_qk_f32_impl(
    device const uchar *weight_bytes,
    device const float *input,
    device float *out,
    constant uint &N,
    constant uint &K,
    constant uint &M,
    threadgroup char *shmem,
    uint3 tgid,
    ushort tid,
    ushort lane,
    ushort simdgroup)
{
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short NL = 16;

    const uint row0 = tgid.y * NR0;
    const uint token0 = tgid.x * NR1;
    if (row0 >= N || token0 >= M) {
        return;
    }

    const short nr0 = (short)min((uint)NR0, N - row0);
    const short nr1 = (short)min((uint)NR1, M - token0);
    const short lr0 = min((short)(tid / NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tid / NL1), (short)(nr1 - 1));
    const short il0 = tid % NL0;
    short il = il0;
    const short iy = 8 * (tid % NL1);

    const uint token = token0 + lr1;
    const uint blocks_per_row = (K + 255u) / 256u;
    device const block_q *x =
        (device const block_q *)weight_bytes + (row0 + lr0) * blocks_per_row;
    device const float *y = input + token * K;

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(NR1, NK));
    constexpr auto descriptor = matmul2d_descriptor(
        NR1, NR0, NK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 weight_values;
        dequantize_func(x, il, weight_values);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = i % 8;
            const short ly = (tid / NL0) % 8;
            sa[NK * (8 * sy + ly) + 8 * sx + lx] = weight_values[i / 4][i % 4];
        }

        const short sx = tid % NL1;
        const short sy = (tid / NL1) / 8;
        const short ly = (tid / NL1) % 8;
        threadgroup half *input_tile = sb + NK * (8 * sy + ly) + 8 * sx;
        if (loop_k + (uint)iy + 8u <= K) {
            const float2x4 input_values =
                *((device const float2x4 *)(y + loop_k + (uint)iy));
            *((threadgroup half2x4 *)input_tile) = (half2x4)input_values;
        } else {
            for (short i = 0; i < 8; ++i) {
                const uint input_k = loop_k + (uint)iy + (uint)i;
                input_tile[i] = input_k < K ? (half)y[input_k] : 0.0h;
            }
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x = il < 2 ? x + 1 : x;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = simdgroup; j < nr1; j += 4) {
        device float *dst = out + (token0 + j) * N + row0;
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float *src = sc + j * NR0;
        threadgroup float4 *src4 = (threadgroup float4 *)src;

        int i = lane;
        for (; i < nr0 / 4; i += 32) {
            dst4[i] = src4[i];
        }
        i = 4 * (nr0 / 4) + lane;
        for (; i < nr0; i += 32) {
            dst[i] = src[i];
        }
    }
}

#define QWEN_SHARED_QK_KERNEL_ARGS \
    device const uchar *weight_bytes [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *out [[buffer(2)]], \
    constant uint &N [[buffer(3)]], \
    constant uint &K [[buffer(4)]], \
    constant uint &M [[buffer(5)]], \
    threadgroup char *shmem [[threadgroup(0)]], \
    uint3 tgid [[threadgroup_position_in_grid]], \
    ushort tid [[thread_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], \
    ushort simdgroup [[simdgroup_index_in_threadgroup]]

#define QWEN_SHARED_QK_CALL(BLOCK_T, DEQUANT) \
    qwen_moe_shared_mul_mm_qk_f32_impl<BLOCK_T, DEQUANT>( \
        weight_bytes, input, out, N, K, M, shmem, tgid, tid, lane, simdgroup)

kernel void qwen_moe_shared_mul_mm_q4k_f32(QWEN_SHARED_QK_KERNEL_ARGS)
{
    QWEN_SHARED_QK_CALL(qwen_llama_block_q4_K, qwen_llama_dequantize_q4_K);
}

kernel void qwen_moe_shared_mul_mm_q6k_f32(QWEN_SHARED_QK_KERNEL_ARGS)
{
    QWEN_SHARED_QK_CALL(qwen_llama_block_q6_K, qwen_llama_dequantize_q6_K);
}

#undef QWEN_SHARED_QK_CALL
#undef QWEN_SHARED_QK_KERNEL_ARGS

kernel void qwen_moe_llama_shared_mul_mm_q8_0_f32(
    device const uchar *weight_bytes [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N_TOKENS [[buffer(5)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]])
{
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK = 32;
    constexpr short NL0 = NK / 16;
    constexpr short NL1 = NK / 8;
    constexpr short NL = 2;

    const uint row0 = tgid.y * NR0;
    const uint token0 = tgid.x * NR1;
    if (token0 >= N_TOKENS) {
        return;
    }

    const short nr0 = (short)min((uint)NR0, N - row0);
    const short nr1 = (short)min((uint)NR1, N_TOKENS - token0);
    const short lr0 = min((short)(tid / NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tid / NL1), (short)(nr1 - 1));
    const short il0 = tid % NL0;
    short il = il0;
    const short iy = 8 * (tid % NL1);

    const uint token = token0 + lr1;
    const uint blocks_per_row = K / 32u;
    device const qwen_llama_block_q8_0 *x =
        (device const qwen_llama_block_q8_0 *)weight_bytes +
        (row0 + lr0) * blocks_per_row;
    device const float *y = input + token * K + iy;

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(NR1, NK));
    constexpr auto descriptor = matmul2d_descriptor(
        NR1, NR0, NK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 weight_values;
        qwen_llama_dequantize_q8_0(x, il, weight_values);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = i % 8;
            const short ly = (tid / NL0) % 8;
            sa[NK * (8 * sy + ly) + 8 * sx + lx] = weight_values[i / 4][i % 4];
        }

        const short sx = tid % NL1;
        const short sy = (tid / NL1) / 8;
        const short ly = (tid / NL1) % 8;
        const float2x4 input_values = *((device const float2x4 *)y);
        *((threadgroup half2x4 *)(sb + NK * (8 * sy + ly) + 8 * sx)) =
            (half2x4)input_values;

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x = il < 2 ? x + 1 : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = simdgroup; j < nr1; j += 4) {
        device float *dst = out + (token0 + j) * N + row0;
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float *src = sc + j * NR0;
        threadgroup float4 *src4 = (threadgroup float4 *)src;

        int i = lane;
        for (; i < nr0 / 4; i += 32) {
            dst4[i] = src4[i];
        }
        i = 4 * (nr0 / 4) + lane;
        for (; i < nr0; i += 32) {
            dst[i] = src[i];
        }
    }
}

kernel void qwen_moe_llama_swiglu_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
#pragma clang fp reassociate(off) contract(off)
    if (gid >= n_elements) {
        return;
    }

    const float gate_value = gate[gid];
    const float up_value = up[gid];
    out[gid] =
        (gate_value / (1.0f + precise::exp(-gate_value))) * up_value;
}

kernel void qwen_moe_llama_weighted_rank_reduce_f32(
    device const float *slot_values [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n_tokens [[buffer(3)]],
    constant uint &n_rank [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
#pragma clang fp reassociate(off) contract(off)
    const uint total = n_tokens * n_rows;
    if (gid >= total || n_rank == 0u) {
        return;
    }

    const uint token = gid / n_rows;
    const uint row = gid - token * n_rows;
    const uint token_rank_base = token * n_rank;
    float product = slot_values[token_rank_base * n_rows + row] *
        weights[token_rank_base];
    float value = product;
    for (uint rank = 1u; rank < n_rank; ++rank) {
        const uint slot = token_rank_base + rank;
        product = slot_values[slot * n_rows + row] * weights[slot];
        value = value + product;
    }
    out[gid] = value;
}

kernel void qwen_moe_llama_expert_order_reduce_f32(
    device const float *slot_values [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const uint *selected_experts [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &n_tokens [[buffer(4)]],
    constant uint &n_rank [[buffer(5)]],
    constant uint &n_rows [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
#pragma clang fp reassociate(off) contract(off)
    const uint total = n_tokens * n_rows;
    if (gid >= total || n_rank == 0u || n_rank > 8u) {
        return;
    }

    const uint token = gid / n_rows;
    const uint row = gid - token * n_rows;
    const uint token_rank_base = token * n_rank;
    uint order[8];
    for (uint rank = 0u; rank < n_rank; ++rank) {
        order[rank] = rank;
        uint pos = rank;
        while (pos > 0u &&
               selected_experts[token_rank_base + order[pos - 1u]] >
                   selected_experts[token_rank_base + order[pos]]) {
            const uint previous = order[pos - 1u];
            order[pos - 1u] = order[pos];
            order[pos] = previous;
            --pos;
        }
    }

    float value = 0.0f;
    for (uint sorted_rank = 0u; sorted_rank < n_rank; ++sorted_rank) {
        const uint slot = token_rank_base + order[sorted_rank];
        const float product = weights[slot] * slot_values[slot * n_rows + row];
        value = value + product;
    }
    out[gid] = value;
}
