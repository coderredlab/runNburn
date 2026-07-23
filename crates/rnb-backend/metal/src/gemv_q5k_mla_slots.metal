#include <metal_stdlib>
using namespace metal;

// pm117: GLM MLA prefill slot-batch Q5_K GEMV. slot(token*HEADS + head)이 grid y 축.
// weight 는 head 별로 다르고 (head = slot % HEADS), input/out 은 slot 별 연속 배치.
// row 처리 구조는 gemv_q5k_coalesced (llama.cpp nr0=2 multi-row + activation reuse)
// 그대로 — slot 오프셋만 추가. dense (HEADS=1) 에서는 head 항이 0 이 된다.
//
// Q5_K super-block layout (176 bytes):
//   0-1 d / 2-3 dmin / 4-15 scales[12] / 16-47 qh[32] / 48-175 ql[128].
kernel void gemv_q5k_mla_slots(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device float*       out          [[buffer(2)]],
    constant uint&      N            [[buffer(3)]],  // rows per head
    constant uint&      K            [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      HEADS        [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const uint slot = tg.y;
    const uint head = slot % HEADS;
    const uint first_row = tg.x * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    const ushort ix  = lane % 4u;   // 0..3
    const ushort tid = lane / 4u;   // 0..7
    const ushort iq  = tid / 4u;    // 0 or 1
    const ushort ir  = tid % 4u;    // 0..3

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

    device const uchar* x0 = weight_bytes + weight_byte_offset
        + (head * N + first_row) * (nb * 176u);
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

        // row 0
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

        // row 1 (있을 때만 — weight OOB 방지)
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

    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[slot * N + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[slot * N + first_row + 1u] = t1;
    }
}
