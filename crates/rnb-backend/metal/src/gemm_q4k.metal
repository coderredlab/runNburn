#include <metal_stdlib>
using namespace metal;

// Q4_K batch GEMM (naive correctness PoC): gemv_q4k.metal 의 dequant 를 1:1 이식하고
// M(token) 축만 추가한다. weight layout(144B/super-block)·dequant 규칙은 gemv_q4k.metal 과 동일.
//
//   out[tok*N + row] = sum_K dequant(weight[row])[k] * input[tok*K + k]
//
// grid = 2D: thread_position_in_grid.x = row(0..N, threadgroup width 패딩 있음 → bound check),
//            thread_position_in_grid.y = tok(0..M).
// 각 thread = (row, tok) 1쌍의 출력 하나. 튜닝(coalesced/tiled/activation reuse)은 GREEN 후 별도.

kernel void gemm_q4k(
    device const uchar* weight_bytes [[buffer(0)]],  // N * (K/256)*144 bytes
    device const float* input        [[buffer(1)]],  // M * K f32 (row-major)
    device float*       out          [[buffer(2)]],  // M * N f32 (row-major)
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],  // K = num_blocks * 256
    constant uint&      weight_byte_offset [[buffer(5)]],  // zero-copy NoCopy: page 내 weight offset
    constant uint&      M            [[buffer(6)]],
    uint2               gid          [[thread_position_in_grid]])  // gid.x=row, gid.y=tok
{
    const uint row = gid.x;
    const uint tok = gid.y;
    if (row >= N || tok >= M) return;

    uint num_blocks = K / 256u;
    float acc = 0.0f;

    for (uint b = 0; b < num_blocks; b++) {
        // 이 행(row)의 b번째 블록 시작 오프셋 (gemv_q4k.metal 과 동일)
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

        // input 의 이 토큰·이 블록 해당 부분 시작 (M축: tok*K offset 추가)
        uint x_base = tok * K + b * 256u;

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

    out[tok * N + row] = acc;
}
