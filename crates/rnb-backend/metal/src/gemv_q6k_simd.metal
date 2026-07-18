#include <metal_stdlib>
using namespace metal;

// SIMD-group 협력 Q6_K GEMV (memory-bandwidth 최적화, pm21 P1 lane-saturation).
//
// 각 threadgroup(=1 SIMD-group, 32 lane)이 하나의 출력 row 를 담당.
// dequant 규칙은 gemv_q6k.metal 과 1:1 동일(rnb-cpu dequantize_q6_k).
//
// num_blocks 가 2의 거듭제곱(2..32)이면 block 을 m 개 lane 이 **group(128 elem)
// 단위**로 협력 분할(pm21 P1, lane idle 제거). Q6_K 는 block 당 group 2개라
// m = min(2, 32/num_blocks), lane 당 ≥1 group → group 내부 4-way(+0/+32/+64/+96)
// dequant 을 한 lane 이 통째로 처리해 안 깨짐. T = num_blocks*m ≤ 32 (simd_sum 정확).
// 예: K=4096(nb16) → m2, T32 (block 당 2 lane, lane 당 1 group).
//     K=12288(nb48) → stride (nb>32, 비-2거듭제곱).
// num_blocks ≥ 32 또는 비-2거듭제곱이면 기존 block-stride(b += 32) fallback.
//
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
kernel void gemv_q6k_simd(
    device const uchar* weight_bytes [[buffer(0)]],  // N * 210 bytes (N Q6_K blocks per row)
    device const float* input        [[buffer(1)]],  // K f32
    device float*       out          [[buffer(2)]],  // N f32
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],  // K = num_blocks * 256
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;

    uint num_blocks = K / 256u;
    float acc = 0.0f;

    bool pow2 = (num_blocks & (num_blocks - 1u)) == 0u;
    bool sub_block = pow2 && num_blocks >= 2u && num_blocks <= 32u;

    // 한 group n(0,1, 128 elem)의 dequant·dot 을 acc 에 누적(stride/sub-block 공통).
    #define Q6K_GROUP(blk, x_base, n)                                                  \
    {                                                                                  \
        device const uchar* ql = (blk);                                                \
        device const uchar* qh = (blk) + 128u;                                         \
        device const char*  sc = (device const char*)((blk) + 192u);                   \
        ushort d_bits = (ushort)(blk)[208] | ((ushort)(blk)[209] << 8);                \
        float d = (float)as_type<half>(d_bits);                                        \
        uint ql_base = (n) * 64u;                                                       \
        uint qh_base = (n) * 32u;                                                       \
        uint sc_base = (n) * 8u;                                                        \
        uint y_base  = (n) * 128u;                                                      \
        for (uint l = 0; l < 32u; l++) {                                                \
            uint is = l / 16u;                                                          \
            int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u)); \
            int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u)); \
            int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u)); \
            int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u)); \
            float w1 = d * (float)sc[sc_base + is]       * (float)(q1 - 32);            \
            float w2 = d * (float)sc[sc_base + is + 2u]  * (float)(q2 - 32);            \
            float w3 = d * (float)sc[sc_base + is + 4u]  * (float)(q3 - 32);            \
            float w4 = d * (float)sc[sc_base + is + 6u]  * (float)(q4 - 32);            \
            acc += w1 * input[(x_base) + y_base + l];                                   \
            acc += w2 * input[(x_base) + y_base + l + 32u];                             \
            acc += w3 * input[(x_base) + y_base + l + 64u];                             \
            acc += w4 * input[(x_base) + y_base + l + 96u];                             \
        }                                                                              \
    }

    if (sub_block) {
        uint m = min(2u, 32u / num_blocks);      // block 당 lane 수 (1..2, group 2개)
        uint t = num_blocks * m;                 // 활성 lane 수 (≤ 32)
        uint groups_per_lane = 2u / m;           // lane 당 group 수 (1,2)
        if (lane < t) {
            uint block_idx = lane / m;
            uint sub_idx   = lane % m;
            uint n_start   = sub_idx * groups_per_lane;
            device const uchar* blk =
                weight_bytes + weight_byte_offset + (row * num_blocks + block_idx) * 210u;
            uint x_base = block_idx * 256u;
            for (uint ni = 0; ni < groups_per_lane; ni++) {
                uint n = n_start + ni;
                Q6K_GROUP(blk, x_base, n);
            }
        }
    } else {
        // fallback: lane 이 block 을 stride 32 로 분할(num_blocks ≥ 32 또는 비-2거듭제곱).
        for (uint b = lane; b < num_blocks; b += 32u) {
            device const uchar* blk =
                weight_bytes + weight_byte_offset + (row * num_blocks + b) * 210u;
            uint x_base = b * 256u;
            Q6K_GROUP(blk, x_base, 0u);
            Q6K_GROUP(blk, x_base, 1u);
        }
    }
    #undef Q6K_GROUP

    // SIMD-group 32 lane partial sum → row 총합.
    float total = simd_sum(acc);
    if (lane == 0) {
        out[row] = total;
    }
}
