#include <metal_stdlib>
using namespace metal;

// Q4_K batch GEMM, dequant 공유(pm33 튜닝): threadgroup 1개 = weight row 1개.
// superblock(256)을 **1번만** dequant → threadgroup memory dq[256] → 전 token(tid grid-stride)이
// 공유 dot. naive(각 (row,tok) thread가 row 전체 dequant → M배 중복)의 dequant M배를 1배로.
//
//   out[tok*N + row] = sum_K dequant(weight[row])[k] * input[tok*K + k]
//
// grid = N threadgroups(각 1 row), threads_per_tg = TG(256). 부분합 acc[M]은 동적 threadgroup memory.
// acc[M]*4 + dq 1KB + scales 64B ≤ 32KB(M5 한계) → M ≤ ~7900(prefill chunk 내).

kernel void gemm_q4k_shared(
    device const uchar* weight_bytes [[buffer(0)]],  // N * (K/256)*144 bytes
    device const float* input        [[buffer(1)]],  // M * K f32 (row-major)
    device float*       out          [[buffer(2)]],  // M * N f32 (row-major)
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],  // K = num_blocks * 256
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      M            [[buffer(6)]],
    threadgroup float*  acc          [[threadgroup(0)]], // M floats, 동적 length(setThreadgroupMemoryLength)
    uint row    [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    if (row >= N) return;
    const uint num_blocks = K / 256u;

    threadgroup float dq[256];   // 현재 superblock 의 dequant 결과(전 token 공유)
    threadgroup float sc_f[8];   // 8 sub-block scale
    threadgroup float mn_f[8];   // 8 sub-block min

    // acc[M] = 0 초기화
    for (uint t = tid; t < M; t += tgsize) {
        acc[t] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* wbase = weight_bytes + weight_byte_offset;

    for (uint sb = 0; sb < num_blocks; sb++) {
        device const uchar* blk = wbase + (row * num_blocks + sb) * 144u;

        // d, dmin (모든 thread 동일하게 읽음 — uniform)
        ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
        ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);
        device const uchar* sc = blk + 4;

        // 8 sub-block scale/min: thread 0..7 이 1개씩 (get_scale_min_k4, gemv_q4k.metal 1:1)
        if (tid < 8u) {
            uint j = tid;
            uchar s, m;
            if (j < 4u) {
                s = sc[j]     & 63u;
                m = sc[j + 4u] & 63u;
            } else {
                s = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                m = (sc[j + 4u] >> 4u)   | ((sc[j]       >> 6u) << 4u);
            }
            sc_f[j] = (float)s;
            mn_f[j] = (float)m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 256 출력 element dequant: thread j(0..255) 1개씩(tgsize<256 이면 grid-stride).
        // j → group g=j/64, half jj=j%64. low(jj<32): is=2g, qs nibble low. high: is=2g+1, nibble high.
        device const uchar* qs = blk + 16;
        for (uint j = tid; j < 256u; j += tgsize) {
            uint g  = j / 64u;
            uint jj = j % 64u;
            uint q_off = g * 32u;
            uint is;
            uint q;
            if (jj < 32u) {
                is = g * 2u;
                q  = qs[q_off + jj] & 0x0Fu;
            } else {
                is = g * 2u + 1u;
                q  = qs[q_off + (jj - 32u)] >> 4u;
            }
            dq[j] = d * sc_f[is] * (float)q - dmin * mn_f[is];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dot: 각 thread 가 자기 token 들(tid grid-stride)에 대해 이 superblock 부분합 누적.
        uint x_off = sb * 256u;
        for (uint tok = tid; tok < M; tok += tgsize) {
            device const float* y = input + tok * K + x_off;
            float a = 0.0f;
            for (uint j = 0; j < 256u; j++) {
                a += dq[j] * y[j];
            }
            acc[tok] += a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 결과 기록
    for (uint tok = tid; tok < M; tok += tgsize) {
        out[tok * N + row] = acc[tok];
    }
}
