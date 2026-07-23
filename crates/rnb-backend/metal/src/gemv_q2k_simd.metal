#include <metal_stdlib>
using namespace metal;

// pm123: SIMD-group 협력 Q2_K GEMV (decode). 각 threadgroup(=1 SIMD-group, 32 lane)이
// 하나의 출력 row 담당. lane 이 super-block 을 stride 32 로 분할(correctness-first).
// dequant 은 rnb-cpu dequantize_q2_k 1:1 (scale=low4·d, min=high4·dmin, 2-bit qs).
//
// Q2_K super-block layout (84 bytes, matches BlockQ2_K repr(C)):
//   0-15 scales[16] / 16-79 qs[64] / 80-81 d(f16) / 82-83 dmin(f16).
// 128-elem group(nn=0,1): jj 0..8, q_base=nn*32+(jj&1)*16, shift=(jj>>1)*2,
//   out idx = nn*128 + jj*16 + l.
kernel void gemv_q2k_simd(
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
    float acc = 0.0f;

    for (uint b = lane; b < num_blocks; b += 32u) {
        device const uchar* blk =
            weight_bytes + weight_byte_offset + (row * num_blocks + b) * 84u;
        device const uchar* scb = blk;         // scales[16]
        device const uchar* qs  = blk + 16u;   // qs[64]
        ushort d_bits  = (ushort)blk[80] | ((ushort)blk[81] << 8);
        ushort dm_bits = (ushort)blk[82] | ((ushort)blk[83] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dm_bits);
        uint x_base = b * 256u;
        for (uint n = 0; n < 2u; n++) {
            uint y_base = n * 128u;
            for (uint jj = 0; jj < 8u; jj++) {
                uint sc = (uint)scb[n * 8u + jj];
                float scale = d * (float)(sc & 0x0Fu);
                float mn = dmin * (float)(sc >> 4u);
                uint qbase = n * 32u + (jj & 1u) * 16u;
                uint shift = (jj >> 1u) * 2u;
                for (uint l = 0; l < 16u; l++) {
                    int q = (int)((qs[qbase + l] >> shift) & 3u);
                    acc += ((float)q * scale - mn) * input[x_base + y_base + jj * 16u + l];
                }
            }
        }
    }

    float total = simd_sum(acc);
    if (lane == 0) out[row] = total;
}
