#include <metal_stdlib>
using namespace metal;

// Q4_K super-block layout (144 bytes, matches BlockQ4_K repr(C)):
//   offset  0-1  : d     (f16 little-endian)
//   offset  2-3  : dmin  (f16 little-endian)
//   offset  4-15 : scales[12]  (6-bit packed sub-block scales/mins)
//   offset 16-143: qs[128]     (4-bit packed quants, 256 weights)
//
// Dequant rule (rnb-cpu dequantize_q4_k 1:1 이식):
//   8 sub-block scales/mins ← get_scale_min_k4:
//     j < 4: sc = scales[j] & 63,  m = scales[j+4] & 63
//     j >= 4: sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//             m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
//   4 groups × 64 elements:
//     group g: is = g*2
//       y[g*64   + l]  = d * sc[is]   * (qs[g*32+l] & 0xF) - dmin * m[is]   (l=0..31)
//       y[g*64+32+ l]  = d * sc[is+1] * (qs[g*32+l] >> 4)  - dmin * m[is+1] (l=0..31)
//
// Kernel: 각 thread = 하나의 출력 row.
// N=1, K=256(한 블록)인 경우 thread 0 이 block_bytes[0..143] 전체를 처리.

kernel void gemv_q4k(
    device const uchar* weight_bytes [[buffer(0)]],  // N * 144 bytes (N Q4_K blocks per row)
    device const float* input        [[buffer(1)]],  // K f32
    device float*       out          [[buffer(2)]],  // N f32
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],  // K = num_blocks * 256
    constant uint&      weight_byte_offset [[buffer(5)]],  // zero-copy NoCopy: page 내 weight 시작 offset
    uint                row          [[thread_position_in_grid]])
{
    if (row >= N) return;

    uint num_blocks = K / 256u;
    float acc = 0.0f;

    for (uint b = 0; b < num_blocks; b++) {
        // 이 행(row)의 b번째 블록 시작 오프셋
        device const uchar* blk = weight_bytes + weight_byte_offset + (row * num_blocks + b) * 144u;

        // d, dmin: f16 → float (little-endian half 읽기)
        ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
        ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        // scales[12] (offset 4..15)
        device const uchar* sc = blk + 4;

        // get_scale_min_k4: 8개 sub-block sc/m 추출 (rnb-cpu 규칙 1:1)
        float scales_f[8];
        float mins_f[8];
        for (uint j = 0; j < 8u; j++) {
            uchar s, m;
            if (j < 4u) {
                s = sc[j]     & 63u;
                m = sc[j + 4u] & 63u;
            } else {
                s = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                m = (sc[j + 4u] >> 4u)   | ((sc[j]       >> 6u) << 4u);
            }
            scales_f[j] = (float)s;
            mins_f[j]   = (float)m;
        }

        // qs[128] (offset 16..143)
        device const uchar* qs = blk + 16;

        // input 의 이 블록 해당 부분 시작
        uint x_base = b * 256u;

        // 4 groups, 각 64 elements
        for (uint g = 0; g < 4u; g++) {
            uint is      = g * 2u;
            float d1     = d * scales_f[is];
            float m1     = dmin * mins_f[is];
            float d2     = d * scales_f[is + 1u];
            float m2     = dmin * mins_f[is + 1u];

            uint q_off = g * 32u;
            uint y_off = g * 64u;

            // low nibble 32개 (sub-block is)
            for (uint l = 0; l < 32u; l++) {
                float q = (float)(qs[q_off + l] & 0x0Fu);
                float w = d1 * q - m1;
                acc += w * input[x_base + y_off + l];
            }
            // high nibble 32개 (sub-block is+1)
            for (uint l = 0; l < 32u; l++) {
                float q = (float)(qs[q_off + l] >> 4u);
                float w = d2 * q - m2;
                acc += w * input[x_base + y_off + 32u + l];
            }
        }
    }

    out[row] = acc;
}
