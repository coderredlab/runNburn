#include <metal_stdlib>
using namespace metal;

// Q6_K batch GEMM, dequant 공유(pm33 튜닝): gemm_q4k_shared 패턴 + Q6_K dequant(stride 210).
// threadgroup=weight row 1개, superblock(256) dequant 1번 → threadgroup memory dq[256] → 전 token 공유.
// Q6_K scale 은 직접 i8 16개라 unpack 불필요(Q4_K 의 6-bit scale unpack 단계 없음).
//
// element j(0..255) 매핑(gemv_q6k.metal 1:1): n=j/128(group), within=j%128, l=within%32, quad=within/32.
//   quad0: ql[ql_base+l]&0xF   | (qh>>0&3)<<4, sc[sc_base+is]
//   quad1: ql[ql_base+l+32]&0xF| (qh>>2&3)<<4, sc[sc_base+is+2]
//   quad2: ql[ql_base+l]>>4    | (qh>>4&3)<<4, sc[sc_base+is+4]
//   quad3: ql[ql_base+l+32]>>4 | (qh>>6&3)<<4, sc[sc_base+is+6]   (is=l/16)

kernel void gemm_q6k_shared(
    device const uchar* weight_bytes [[buffer(0)]],  // N * (K/256)*210 bytes
    device const float* input        [[buffer(1)]],  // M * K f32
    device float*       out          [[buffer(2)]],  // M * N f32
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      M            [[buffer(6)]],
    threadgroup float*  acc          [[threadgroup(0)]], // M floats(동적)
    uint row    [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    if (row >= N) return;
    const uint num_blocks = K / 256u;

    threadgroup float dq[256];

    for (uint t = tid; t < M; t += tgsize) {
        acc[t] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* wbase = weight_bytes + weight_byte_offset;

    for (uint sb = 0; sb < num_blocks; sb++) {
        device const uchar* blk = wbase + (row * num_blocks + sb) * 210u;
        device const uchar* ql = blk;                             // 0..127
        device const uchar* qh = blk + 128;                       // 128..191
        device const char*  sc = (device const char*)(blk + 192); // 192..207 (signed i8)
        ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
        float d = (float)as_type<half>(d_bits);

        // 256 출력 element dequant: thread j(grid-stride). d uniform/sc global 이라 사전 barrier 불필요.
        for (uint j = tid; j < 256u; j += tgsize) {
            uint n      = j / 128u;
            uint within = j % 128u;
            uint l      = within % 32u;
            uint quad   = within / 32u;
            uint ql_base = n * 64u;
            uint qh_base = n * 32u;
            uint sc_base = n * 8u;
            uint is = l / 16u;

            int q;
            uchar qhb = qh[qh_base + l];
            if (quad == 0u) {
                q = (int)((ql[ql_base + l]       & 0x0Fu) | (((qhb >> 0u) & 3u) << 4u));
            } else if (quad == 1u) {
                q = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qhb >> 2u) & 3u) << 4u));
            } else if (quad == 2u) {
                q = (int)((ql[ql_base + l]       >> 4u)   | (((qhb >> 4u) & 3u) << 4u));
            } else {
                q = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qhb >> 6u) & 3u) << 4u));
            }
            int sc_idx = (int)(sc_base + is + quad * 2u);
            dq[j] = d * (float)sc[sc_idx] * (float)(q - 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

    for (uint tok = tid; tok < M; tok += tgsize) {
        out[tok * N + row] = acc[tok];
    }
}
