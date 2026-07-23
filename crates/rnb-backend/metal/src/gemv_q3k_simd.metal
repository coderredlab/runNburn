#include <metal_stdlib>
using namespace metal;

// pm123: SIMD-group 협력 Q3_K GEMV (decode). 각 threadgroup(=1 SIMD-group, 32 lane)이
// 하나의 출력 row 담당. lane 이 super-block 을 stride 32 로 분할(correctness-first;
// sub-block lane-saturation 최적화는 후속). dequant 은 rnb-cpu dequantize_q3_k 1:1.
//
// Q3_K super-block layout (110 bytes, matches BlockQ3_K repr(C)):
//   0-31 hmask[32] / 32-95 qs[64] / 96-107 scales[12](6-bit packed) / 108-109 d(f16).
// 128-elem group(nn=0,1): j 0..4(shift=2j), half 0..2(16), out idx = nn*128 + j*32 + half*16 + l.
kernel void gemv_q3k_simd(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device float*       out          [[buffer(2)]],
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;
    const uint num_blocks = K / 256u;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    float acc = 0.0f;

    for (uint b = lane; b < num_blocks; b += 32u) {
        device const uchar* blk =
            weight_bytes + weight_byte_offset + (row * num_blocks + b) * 110u;
        device const uchar* hm  = blk;         // hmask[32]
        device const uchar* qs  = blk + 32u;   // qs[64]
        device const uchar* scb = blk + 96u;   // scales[12]
        ushort d_bits = (ushort)blk[108] | ((ushort)blk[109] << 8);
        float d = (float)as_type<half>(d_bits);
        // 6-bit scale unpack (rnb-cpu dequantize_q3_k 1:1)
        uint a0 = (uint)scb[0] | ((uint)scb[1] << 8) | ((uint)scb[2] << 16) | ((uint)scb[3] << 24);
        uint a1 = (uint)scb[4] | ((uint)scb[5] << 8) | ((uint)scb[6] << 16) | ((uint)scb[7] << 24);
        uint a2 = (uint)scb[8] | ((uint)scb[9] << 8) | ((uint)scb[10] << 16) | ((uint)scb[11] << 24);
        uint tmp = a2;
        uint scw[4];
        scw[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        scw[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        scw[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        scw[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        uint x_base = b * 256u;
        for (uint n = 0; n < 2u; n++) {
            uint q_off = n * 32u;
            uint y_base = n * 128u;
            for (uint jj = 0; jj < 4u; jj++) {
                uint shift = jj * 2u;
                uchar mbit = (uchar)(1u << (n * 4u + jj));
                uint is0 = n * 8u + jj * 2u;
                uint is1 = is0 + 1u;
                int s0 = (int)(char)((scw[is0 >> 2u] >> ((is0 & 3u) * 8u)) & 0xffu);
                int s1 = (int)(char)((scw[is1 >> 2u] >> ((is1 & 3u) * 8u)) & 0xffu);
                float dl0 = d * (float)(s0 - 32);
                float dl1 = d * (float)(s1 - 32);
                for (uint l = 0; l < 16u; l++) {
                    int q0 = (int)((qs[q_off + l] >> shift) & 3u);
                    int hv0 = (hm[l] & mbit) ? 0 : 4;
                    acc += dl0 * (float)(q0 - hv0) * input[x_base + y_base + jj * 32u + l];
                    int q1 = (int)((qs[q_off + l + 16u] >> shift) & 3u);
                    int hv1 = (hm[l + 16u] & mbit) ? 0 : 4;
                    acc += dl1 * (float)(q1 - hv1) * input[x_base + y_base + jj * 32u + 16u + l];
                }
            }
        }
    }

    float total = simd_sum(acc);
    if (lane == 0) out[row] = total;
}
