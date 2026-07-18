#include <metal_stdlib>
using namespace metal;

// Q6_K super-block layout (210 bytes, matches BlockQ6_K repr(C)):
//   offset   0-127 : ql[128]    (low 4 bits)
//   offset 128-191 : qh[64]     (high 2 bits)
//   offset 192-207 : scales[16] (i8, signed)
//   offset 208-209 : d          (f16 little-endian)
//
// Dequant rule (rnb-cpu dequantize_q6_k 1:1 이식):
//   2 groups of 128 elements (n = 0, 1):
//     ql_base = n*64, qh_base = n*32, sc_base = n*8, y_base = n*128
//     for l in 0..32:  is = l / 16
//       q1 = (ql[ql_base+l]    & 0x0F) | (((qh[qh_base+l] >> 0) & 3) << 4)
//       q2 = (ql[ql_base+l+32] & 0x0F) | (((qh[qh_base+l] >> 2) & 3) << 4)
//       q3 = (ql[ql_base+l]    >> 4)   | (((qh[qh_base+l] >> 4) & 3) << 4)
//       q4 = (ql[ql_base+l+32] >> 4)   | (((qh[qh_base+l] >> 6) & 3) << 4)
//       y[y_base+l   ] = d * sc[sc_base+is  ] * (q1 - 32)
//       y[y_base+l+32] = d * sc[sc_base+is+2] * (q2 - 32)
//       y[y_base+l+64] = d * sc[sc_base+is+4] * (q3 - 32)
//       y[y_base+l+96] = d * sc[sc_base+is+6] * (q4 - 32)
//
// Kernel: 각 thread = 하나의 출력 row. block 을 dequant 하며 input 과 내적 누적.

kernel void gemv_q6k(
    device const uchar* weight_bytes [[buffer(0)]],  // N * 210 bytes (N Q6_K blocks per row)
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
        device const uchar* blk = weight_bytes + weight_byte_offset + (row * num_blocks + b) * 210u;

        device const uchar* ql = blk;                            // 0..127
        device const uchar* qh = blk + 128;                      // 128..191
        device const char*  sc = (device const char*)(blk + 192); // 192..207 (signed i8)

        // d: f16 → float (little-endian half 읽기)
        ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
        float d = (float)as_type<half>(d_bits);

        // input 의 이 블록 해당 부분 시작
        uint x_base = b * 256u;

        // 2 groups × 128 elements
        for (uint n = 0; n < 2u; n++) {
            uint ql_base = n * 64u;
            uint qh_base = n * 32u;
            uint sc_base = n * 8u;
            uint y_base  = n * 128u;

            for (uint l = 0; l < 32u; l++) {
                uint is = l / 16u; // 0 for first 16, 1 for next 16

                int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u));

                float w1 = d * (float)sc[sc_base + is]       * (float)(q1 - 32);
                float w2 = d * (float)sc[sc_base + is + 2u]  * (float)(q2 - 32);
                float w3 = d * (float)sc[sc_base + is + 4u]  * (float)(q3 - 32);
                float w4 = d * (float)sc[sc_base + is + 6u]  * (float)(q4 - 32);

                acc += w1 * input[x_base + y_base + l];
                acc += w2 * input[x_base + y_base + l + 32u];
                acc += w3 * input[x_base + y_base + l + 64u];
                acc += w4 * input[x_base + y_base + l + 96u];
            }
        }
    }

    out[row] = acc;
}
