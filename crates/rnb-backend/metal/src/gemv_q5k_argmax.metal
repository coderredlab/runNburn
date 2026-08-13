#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#include <metal_stdlib>
using namespace metal;

// Muse output tail: llama.cpp Q5_K NR0=1 geometry plus one local
// argmax result per two-SIMD-group threadgroup. The projection never materializes
// vocab-sized logits; each threadgroup writes only its better row.
kernel void gemv_q5k_argmax_partial(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device float*       partial_value [[buffer(2)]],
    device uint*        partial_index [[buffer(3)]],
    constant uint&      N            [[buffer(4)]],
    constant uint&      K            [[buffer(5)]],
    constant uint&      weight_byte_offset [[buffer(6)]],
    constant uint&      excluded     [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    const uint row = group * 2u + (uint)sg;
    const bool valid = row < N && row != excluded;

    const ushort ix  = lane % 4u;
    const ushort tid = lane / 4u;
    const ushort iq  = tid / 4u;
    const ushort ir  = tid % 4u;
    const ushort l0 = 8u * ir;
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

    device const uchar* x = weight_bytes + weight_byte_offset + row * (nb * 176u);
    device const float* y1 = input + ix * 256u + y_offset;
    float yl[16];
    float yh[16];
    float sumf = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    if (valid) {
        for (uint ib = ix; ib < nb; ib += 4u) {
            device const float* y2 = y1 + 128u;
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (ushort l = 0; l < 8; ++l) {
                yl[l+0] = y1[l+0];  sumy[0] += yl[l+0];
                yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
                yh[l+0] = y2[l+0];  sumy[2] += yh[l+0];
                yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
            }

            device const uchar* blk = x + ib * 176u;
            device const half* dh = (device const half*)blk;
            device const ushort* a = (device const ushort*)(blk + 4u) + iq;
            device const uchar* q1 = (blk + 48u) + q_offset;
            device const uchar* qh = (blk + 16u) + l0;
            device const uchar* q2 = q1 + 64u;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            FOR_UNROLL (ushort l = 0; l < 8; ++l) {
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
            sumf += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                    sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                    sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                    sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                    (float)dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
            y1 += 4u * 256u;
        }
    }

    const float total = valid ? simd_sum(sumf) : -INFINITY;
    threadgroup float candidates[2];
    if (lane == 0u) candidates[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        const bool second_valid = group * 2u + 1u < N && group * 2u + 1u != excluded;
        const bool choose_second = second_valid && candidates[1] >= candidates[0];
        partial_value[group] = choose_second ? candidates[1] : candidates[0];
        partial_index[group] = group * 2u + (choose_second ? 1u : 0u);
    }
}

// The first-stage result count is at most ceil(202048 / 2) for the Muse vocab.
// One 1024-thread threadgroup scans it and preserves the existing highest-index tie break.
// One threadgroup consumes one candidate per SIMD-group. Compared with the
// single-group scan, this keeps the large partial array distributed across all
// SIMD-groups and reduces only their local winners in threadgroup memory.
kernel void argmax_partial_f32(
    device const float* partial_value [[buffer(0)]],
    device const uint*  partial_index [[buffer(1)]],
    device uint*        out           [[buffer(2)]],
    constant uint&      count         [[buffer(3)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint tid = (uint)sg * 32u + (uint)lane;
    const uint stride = (uint)nsg * 32u;
    float best_value = -INFINITY;
    uint best_index = 0u;
    for (uint i = tid; i < count; i += stride) {
        const float value = partial_value[i];
        const uint index = partial_index[i];
        if (value > best_value || (value == best_value && index > best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        const float other_value = simd_shuffle_down(best_value, offset);
        const uint other_index = simd_shuffle_down(best_index, offset);
        if (lane < offset &&
            (other_value > best_value ||
             (other_value == best_value && other_index > best_index))) {
            best_value = other_value;
            best_index = other_index;
        }
    }

    threadgroup float values[32];
    threadgroup uint indices[32];
    if (lane == 0u) {
        values[sg] = best_value;
        indices[sg] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float total_value = values[0];
        uint total_index = indices[0];
        for (ushort i = 1u; i < nsg; ++i) {
            if (values[i] > total_value ||
                (values[i] == total_value && indices[i] > total_index)) {
                total_value = values[i];
                total_index = indices[i];
            }
        }
        out[0] = total_index;
    }
}
