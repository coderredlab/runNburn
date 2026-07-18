#include <metal_stdlib>
using namespace metal;

// SIMD-group 협력 Q4_K GEMV (memory-bandwidth 최적화).
//
// 각 threadgroup(=1 SIMD-group, 32 lane)이 하나의 출력 row 를 담당.
// dequant 규칙은 gemv_q4k.metal 과 1:1 동일(rnb-cpu dequantize_q4_k).
//
// decode GEMV 는 memory-bandwidth-bound 라(roofline intensity ~4 ≪ 임계 ~50),
// 32-lane 협력으로 bandwidth 활용. K 가 작으면(num_blocks < 32) "block 당 1 lane"
// 분할은 lane 절반 이상이 idle(pm21: K=4096 → num_blocks=16 → 50% idle). 이를
// 막기 위해 num_blocks 가 2의 거듭제곱(2..32)이면 **block 을 m 개 lane 이 group
// (64 elem) 단위로 협력 분할**(pm21 P1, lane-saturation):
//   m = min(4, 32/num_blocks)  — lane 당 최소 1 group(64 elem) 보장 → nibble/scale
//                                경계 안 깸(Q4_K scale 은 32 elem, group 은 64 elem).
//   T = num_blocks * m ≤ 32    — simd_sum(32 lane 고정합)이 정확(부분합 누락 없음).
// num_blocks ≥ 32 또는 비-2거듭제곱이면 기존 block-stride(b += 32) fallback.
//
// 예: K=4096(nb16) → m2, T32 (block 당 2 lane, lane 당 2 group).
//     K=2048(nb8)  → m4, T32 (block 당 4 lane, lane 당 1 group).
//     K=12288(nb48)→ stride (nb>32).
kernel void gemv_q4k_simd(
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

    uint num_blocks = K / 256u;
    float acc = 0.0f;

    // lane-saturation 분기: num_blocks 가 2의 거듭제곱(2..32)이면 group 단위 sub-block.
    bool pow2 = (num_blocks & (num_blocks - 1u)) == 0u;
    bool sub_block = pow2 && num_blocks >= 2u && num_blocks <= 32u;

    if (sub_block) {
        uint m = min(4u, 32u / num_blocks);      // block 당 lane 수 (1..4)
        uint t = num_blocks * m;                 // 활성 lane 수 (≤ 32)
        uint groups_per_lane = 4u / m;           // lane 당 group 수 (1,2,4)
        if (lane < t) {
            uint block_idx = lane / m;
            uint sub_idx   = lane % m;
            uint g_start   = sub_idx * groups_per_lane;

            device const uchar* blk =
                weight_bytes + weight_byte_offset + (row * num_blocks + block_idx) * 144u;

            ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
            ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
            float d    = (float)as_type<half>(d_bits);
            float dmin = (float)as_type<half>(dmin_bits);

            device const uchar* sc = blk + 4;
            float scales_f[8];
            float mins_f[8];
            for (uint j = 0; j < 8u; j++) {
                uchar s, mm;
                if (j < 4u) {
                    s  = sc[j]      & 63u;
                    mm = sc[j + 4u] & 63u;
                } else {
                    s  = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                    mm = (sc[j + 4u] >> 4u)   | ((sc[j]      >> 6u) << 4u);
                }
                scales_f[j] = (float)s;
                mins_f[j]   = (float)mm;
            }

            device const uchar* qs = blk + 16;
            uint x_base = block_idx * 256u;

            for (uint gi = 0; gi < groups_per_lane; gi++) {
                uint g = g_start + gi;
                uint is = g * 2u;
                float d1 = d * scales_f[is];
                float m1 = dmin * mins_f[is];
                float d2 = d * scales_f[is + 1u];
                float m2 = dmin * mins_f[is + 1u];

                uint q_off = g * 32u;
                uint y_off = g * 64u;

                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] & 0x0Fu);
                    acc += (d1 * q - m1) * input[x_base + y_off + l];
                }
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] >> 4u);
                    acc += (d2 * q - m2) * input[x_base + y_off + 32u + l];
                }
            }
        }
    } else {
        // fallback: lane 이 block 을 stride 32 로 분할(num_blocks ≥ 32 또는 비-2거듭제곱).
        for (uint b = lane; b < num_blocks; b += 32u) {
            device const uchar* blk =
                weight_bytes + weight_byte_offset + (row * num_blocks + b) * 144u;

            ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
            ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
            float d    = (float)as_type<half>(d_bits);
            float dmin = (float)as_type<half>(dmin_bits);

            device const uchar* sc = blk + 4;
            float scales_f[8];
            float mins_f[8];
            for (uint j = 0; j < 8u; j++) {
                uchar s, mm;
                if (j < 4u) {
                    s  = sc[j]      & 63u;
                    mm = sc[j + 4u] & 63u;
                } else {
                    s  = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                    mm = (sc[j + 4u] >> 4u)   | ((sc[j]      >> 6u) << 4u);
                }
                scales_f[j] = (float)s;
                mins_f[j]   = (float)mm;
            }

            device const uchar* qs = blk + 16;
            uint x_base = b * 256u;

            for (uint g = 0; g < 4u; g++) {
                uint is = g * 2u;
                float d1 = d * scales_f[is];
                float m1 = dmin * mins_f[is];
                float d2 = d * scales_f[is + 1u];
                float m2 = dmin * mins_f[is + 1u];

                uint q_off = g * 32u;
                uint y_off = g * 64u;

                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] & 0x0Fu);
                    acc += (d1 * q - m1) * input[x_base + y_off + l];
                }
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] >> 4u);
                    acc += (d2 * q - m2) * input[x_base + y_off + 32u + l];
                }
            }
        }
    }

    // SIMD-group 32 lane partial sum → row 총합.
    float total = simd_sum(acc);
    if (lane == 0) {
        out[row] = total;
    }
}
