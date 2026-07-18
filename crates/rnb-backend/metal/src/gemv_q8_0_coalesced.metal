#include <metal_stdlib>
using namespace metal;

// llama.cpp kernel_mul_mv_q8_0_f32_impl 의 NR0=2 multi-row 이식 (coalesced + activation reuse).
// 1 threadgroup(=1 simdgroup, 32 lane)이 output row 2개 처리. NQ=8 quants/thread.
// lane 을 ix=lane/4(0..7) × il=lane%4(0..3) 로 나눠 block(K/32 개)을 stride-8 순회.
// input(yl)은 두 row 공유(activation reuse). grid=ceil(N/2), tg=32.
//
// Q8_0 block layout (34 bytes, matches rnb-cpu BlockQ8_0 + dequantize_q8_0):
//   0-1 d(half) / 2-33 qs[32](i8). y[i] = qs[i] * d. block 당 32 elem (super-block 아님).
//   num_blocks = K / 32.
kernel void gemv_q8_0_coalesced(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device float*       out          [[buffer(2)]],
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]])
{
    const uint first_row = group * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    constexpr ushort NQ = 8u;
    const uint nb = K / 32u;

    const ushort ix = lane / 4u;   // 0..7  (block index within simdgroup stride)
    const ushort il = lane % 4u;   // 0..3  (NQ-chunk within a 32-elem block)

    device const uchar* x0 = weight_bytes + weight_byte_offset + first_row * (nb * 34u);
    device const uchar* x1 = x0 + nb * 34u;

    device const float* yb = input + (uint)ix * 32u + (uint)il * NQ;

    float yl[NQ];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = ix; ib < nb; ib += 8u) {
        for (ushort i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        // row 0
        {
            device const uchar* blk = x0 + ib * 34u;
            ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
            float d = (float)as_type<half>(d_bits);
            device const char* qs = (device const char*)(blk + 2u) + (uint)il * NQ;
            float sumq = 0.f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * yl[i];
            }
            sumf0 += sumq * d;
        }

        // row 1 (있을 때만 — weight OOB 방지)
        if (has_row1) {
            device const uchar* blk = x1 + ib * 34u;
            ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
            float d = (float)as_type<half>(d_bits);
            device const char* qs = (device const char*)(blk + 2u) + (uint)il * NQ;
            float sumq = 0.f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * yl[i];
            }
            sumf1 += sumq * d;
        }

        yb += 8u * 32u;
    }

    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[first_row + 1u] = t1;
    }
}
