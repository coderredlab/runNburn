#include <metal_stdlib>
using namespace metal;

// Q5_K super-block layout (176 bytes, BlockQ5_K repr(C)):
//   0-1   : d    (f16)        2-3   : dmin (f16)
//   4-15  : scales[12] (6-bit packed, get_scale_min_k4 와 동일)
//   16-47 : qh[32] (high bits, 1 bit/weight)
//   48-175: qs[128] (low 4 bits)
// Dequant (rnb-cpu dequantize_q5_k 1:1): Q4_K + qh high bit(+16).
//   group g(0..4): is=2g, u1=1<<2g, u2=2<<2g
//     y[g*64+l]    = d*sc[is]   * ((ql[g*32+l]&0xF) + (qh[l]&u1?16:0)) - dmin*m[is]
//     y[g*64+32+l] = d*sc[is+1] * ((ql[g*32+l]>>4)  + (qh[l]&u2?16:0)) - dmin*m[is+1]
// Kernel: thread 1개 = 출력 row 1개. weight_byte_offset = NoCopy page 내 시작.
kernel void gemv_q5k(
    device const uchar* weight_bytes      [[buffer(0)]],  // N * (K/256 * 176) bytes
    device const float* input             [[buffer(1)]],
    device float*       out               [[buffer(2)]],
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],   // K = num_blocks * 256
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint                row               [[thread_position_in_grid]])
{
    if (row >= N) return;
    uint num_blocks = K / 256u;
    float acc = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        device const uchar* blk =
            weight_bytes + weight_byte_offset + (row * num_blocks + b) * 176u;
        ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
        ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        device const uchar* sc = blk + 4;
        float scales_f[8];
        float mins_f[8];
        for (uint j = 0; j < 8u; j++) {
            uchar s, m;
            if (j < 4u) {
                s = sc[j]      & 63u;
                m = sc[j + 4u] & 63u;
            } else {
                s = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                m = (sc[j + 4u] >> 4u)   | ((sc[j]       >> 6u) << 4u);
            }
            scales_f[j] = (float)s;
            mins_f[j]   = (float)m;
        }

        device const uchar* qh = blk + 16;
        device const uchar* ql = blk + 48;
        uint x_base = b * 256u;

        for (uint g = 0; g < 4u; g++) {
            uint is  = g * 2u;
            float d1 = d * scales_f[is];
            float m1 = dmin * mins_f[is];
            float d2 = d * scales_f[is + 1u];
            float m2 = dmin * mins_f[is + 1u];
            uint ql_off = g * 32u;
            uint y_off  = g * 64u;
            uchar u1 = (uchar)(1u << (2u * g));
            uchar u2 = (uchar)(2u << (2u * g));

            for (uint l = 0; l < 32u; l++) {
                float high = (qh[l] & u1) ? 16.0f : 0.0f;
                float q = (float)(ql[ql_off + l] & 0x0Fu) + high;
                acc += (d1 * q - m1) * input[x_base + y_off + l];
            }
            for (uint l = 0; l < 32u; l++) {
                float high = (qh[l] & u2) ? 16.0f : 0.0f;
                float q = (float)(ql[ql_off + l] >> 4u) + high;
                acc += (d2 * q - m2) * input[x_base + y_off + 32u + l];
            }
        }
    }
    out[row] = acc;
}
