#include <metal_stdlib>
using namespace metal;

// pm113: GLM MLA prefill slot-batch GEMV. slot(token*HEADS + head)이 grid y 축.
// weight 는 head 별로 다르고 (head = slot % HEADS), input/out 은 slot 별 연속 배치.
// row 처리 구조는 gemv_q8_0_coalesced (NR0=2 multi-row + activation reuse) 그대로.
//
// Q8_0 block layout (34 bytes): 0-1 d(half) / 2-33 qs[32](i8). num_blocks = K/32.
kernel void gemv_q8_0_mla_slots(
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

    constexpr ushort NQ = 8u;
    const uint nb = K / 32u;

    const ushort ix = lane / 4u;   // 0..7  (block index within simdgroup stride)
    const ushort il = lane % 4u;   // 0..3  (NQ-chunk within a 32-elem block)

    device const uchar* x0 = weight_bytes + weight_byte_offset
        + (head * N + first_row) * (nb * 34u);
    device const uchar* x1 = x0 + nb * 34u;

    device const float* yb = input + slot * K + (uint)ix * 32u + (uint)il * NQ;

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
    if (lane == 0u) out[slot * N + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[slot * N + first_row + 1u] = t1;
    }
}
