#include <metal_stdlib>
using namespace metal;

// Q4_K coalesced GEMV with two SIMD-groups per threadgroup.
// Each SIMD-group computes two output rows, so one threadgroup covers four rows.
kernel void gemv_q4k_coalesced_nsg2(
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

    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = weight_bytes + weight_byte_offset + first_row * (nb * 144u);
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

    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[first_row + 1u] = t1;
    }
}
