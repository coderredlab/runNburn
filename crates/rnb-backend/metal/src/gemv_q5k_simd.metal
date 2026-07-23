#include <metal_stdlib>
using namespace metal;

// SIMD-group 협력 Q5_K GEMV (memory-bandwidth 최적화, pm21 P1 lane-saturation).
//
// 각 threadgroup(=1 SIMD-group, 32 lane)이 하나의 출력 row 를 담당.
// dequant 규칙은 gemv_q5k.metal 과 1:1 동일(rnb-cpu dequantize_q5_k = Q4_K + qh high bit).
//
// Q5_K super-block layout (176 bytes):
//   0-1 d / 2-3 dmin / 4-15 scales[12](6bit packed) / 16-47 qh[32] / 48-175 ql[128].
//   group g(0..4): is=2g, u1=1<<2g, u2=2<<2g — qh 비트 마스크가 group g 에 의존.
//
// num_blocks 가 2의 거듭제곱(2..32)이면 block 을 m 개 lane 이 **group(64 elem) 단위**
// 로 협력 분할(lane idle 제거). m = min(4, 32/num_blocks), lane 당 ≥1 group → u1/u2(g
// 의존) 마스크가 정확(group 경계 안 깸). T = num_blocks*m ≤ 32 (simd_sum 정확).
// 예: K=4096(nb16)→m2,T32 / K=2048(nb8)→m4,T32 / nb≥32·비-2거듭제곱→stride.
kernel void gemv_q5k_simd(
    device const uchar* weight_bytes      [[buffer(0)]],
    device const float* input             [[buffer(1)]],
    device float*       out               [[buffer(2)]],
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;

    uint num_blocks = K / 256u;
    float acc = 0.0f;

    bool pow2 = (num_blocks & (num_blocks - 1u)) == 0u;
    bool sub_block = pow2 && num_blocks >= 2u && num_blocks <= 32u;

    // 한 group g(0..3, 64 elem)의 dequant·dot 을 acc 에 누적(stride/sub-block 공통).
    #define Q5K_GROUP(blk, x_base, g)                                                  \
    {                                                                                  \
        ushort d_bits    = (ushort)(blk)[0] | ((ushort)(blk)[1] << 8);                 \
        ushort dmin_bits = (ushort)(blk)[2] | ((ushort)(blk)[3] << 8);                 \
        float d    = (float)as_type<half>(d_bits);                                     \
        float dmin = (float)as_type<half>(dmin_bits);                                  \
        device const uchar* sc = (blk) + 4;                                            \
        uchar s1, mm1, s2, mm2;                                                         \
        {                                                                              \
            uint j = (g) * 2u;                                                          \
            if (j < 4u) { s1 = sc[j] & 63u; mm1 = sc[j + 4u] & 63u; }                   \
            else { s1 = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);              \
                   mm1 = (sc[j + 4u] >> 4u) | ((sc[j] >> 6u) << 4u); }                  \
            uint j2 = j + 1u;                                                           \
            if (j2 < 4u) { s2 = sc[j2] & 63u; mm2 = sc[j2 + 4u] & 63u; }                \
            else { s2 = (sc[j2 + 4u] & 0x0Fu) | ((sc[j2 - 4u] >> 6u) << 4u);            \
                   mm2 = (sc[j2 + 4u] >> 4u) | ((sc[j2] >> 6u) << 4u); }                \
        }                                                                              \
        float d1 = d * (float)s1;  float m1 = dmin * (float)mm1;                        \
        float d2 = d * (float)s2;  float m2 = dmin * (float)mm2;                        \
        device const uchar* qh = (blk) + 16;                                           \
        device const uchar* ql = (blk) + 48;                                           \
        uint ql_off = (g) * 32u;                                                        \
        uint y_off  = (g) * 64u;                                                        \
        uchar u1 = (uchar)(1u << (2u * (g)));                                           \
        uchar u2 = (uchar)(2u << (2u * (g)));                                           \
        for (uint l = 0; l < 32u; l++) {                                                \
            float high = (qh[l] & u1) ? 16.0f : 0.0f;                                   \
            float q = (float)(ql[ql_off + l] & 0x0Fu) + high;                           \
            acc += (d1 * q - m1) * input[(x_base) + y_off + l];                         \
        }                                                                              \
        for (uint l = 0; l < 32u; l++) {                                                \
            float high = (qh[l] & u2) ? 16.0f : 0.0f;                                   \
            float q = (float)(ql[ql_off + l] >> 4u) + high;                             \
            acc += (d2 * q - m2) * input[(x_base) + y_off + 32u + l];                   \
        }                                                                              \
    }

    if (sub_block) {
        uint m = min(4u, 32u / num_blocks);      // block 당 lane 수 (1..4, group 4개)
        uint t = num_blocks * m;                 // 활성 lane 수 (≤ 32)
        uint groups_per_lane = 4u / m;           // lane 당 group 수 (1,2,4)
        if (lane < t) {
            uint block_idx = lane / m;
            uint sub_idx   = lane % m;
            uint g_start   = sub_idx * groups_per_lane;
            device const uchar* blk =
                weight_bytes + weight_byte_offset + (row * num_blocks + block_idx) * 176u;
            uint x_base = block_idx * 256u;
            for (uint gi = 0; gi < groups_per_lane; gi++) {
                uint g = g_start + gi;
                Q5K_GROUP(blk, x_base, g);
            }
        }
    } else {
        // fallback: lane 이 block 을 stride 32 로 분할(num_blocks ≥ 32 또는 비-2거듭제곱).
        for (uint b = lane; b < num_blocks; b += 32u) {
            device const uchar* blk =
                weight_bytes + weight_byte_offset + (row * num_blocks + b) * 176u;
            uint x_base = b * 256u;
            Q5K_GROUP(blk, x_base, 0u);
            Q5K_GROUP(blk, x_base, 1u);
            Q5K_GROUP(blk, x_base, 2u);
            Q5K_GROUP(blk, x_base, 3u);
        }
    }
    #undef Q5K_GROUP

    float total = simd_sum(acc);
    if (lane == 0) {
        out[row] = total;
    }
}
