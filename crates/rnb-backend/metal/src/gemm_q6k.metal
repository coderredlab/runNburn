#include <metal_stdlib>
using namespace metal;

// Q6_K batch GEMM (naive correctness PoC): gemv_q6k.metal 의 dequant 를 1:1 이식하고
// M(token) 축만 추가한다. weight layout(210B/super-block)·dequant 규칙은 gemv_q6k.metal 과 동일.
//
//   out[tok*N + row] = sum_K dequant(weight[row])[k] * input[tok*K + k]
//
// grid = 2D: thread_position_in_grid.x = row(0..N, threadgroup width 패딩 → bound check),
//            thread_position_in_grid.y = tok(0..M).

kernel void gemm_q6k(
    device const uchar* weight_bytes [[buffer(0)]],  // N * (K/256)*210 bytes
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
        // 이 행(row)의 b번째 블록 시작 오프셋 (gemv_q6k.metal 과 동일)
        device const uchar* blk = weight_bytes + weight_byte_offset + (row * num_blocks + b) * 210u;

        device const uchar* ql = blk;                            // 0..127
        device const uchar* qh = blk + 128;                      // 128..191
        device const char*  sc = (device const char*)(blk + 192); // 192..207 (signed i8)

        // d: f16 → float (little-endian half 읽기)
        ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
        float d = (float)as_type<half>(d_bits);

        // input 의 이 토큰·이 블록 해당 부분 시작 (M축: tok*K offset 추가)
        uint x_base = tok * K + b * 256u;

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

    out[tok * N + row] = acc;
}
