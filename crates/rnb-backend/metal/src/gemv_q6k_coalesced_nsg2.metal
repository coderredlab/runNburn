#include <metal_stdlib>
using namespace metal;

// Q6_K coalesced GEMV with two SIMD-groups per threadgroup.
// Each SIMD-group computes two output rows, so one threadgroup covers four rows.
kernel void gemv_q6k_coalesced_nsg2(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device float*       out          [[buffer(2)]],
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint   group [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sg    [[simdgroup_index_in_threadgroup]])
{
    const uint first_row = (group * 2u + (uint)sg) * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

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

    device const uchar* x0 = weight_bytes + weight_byte_offset + first_row * (nb * 210u);
    device const uchar* x1 = x0 + nb * 210u;

    float yl[16];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = ix; ib < nb; ib += 2u) {
        device const float* y = input + ib * 256u + y_offset;
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

    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[first_row + 1u] = t1;
    }
}
