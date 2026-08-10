#pragma once

static __device__ __forceinline__ float rnb_q4k_block_dot_f32_lane(
    const unsigned char* __restrict__ block,
    const float* __restrict__ input,
    unsigned lane) {
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
    float acc = 0.0f;
#pragma unroll
    for (unsigned group = 0; group < 4u; ++group) {
        const unsigned j0 = group * 2u;
        const unsigned j1 = j0 + 1u;
        unsigned sc0;
        unsigned mn0;
        unsigned sc1;
        unsigned mn1;
        if (j0 < 4u) {
            sc0 = block[4u + j0] & 63u;
            mn0 = block[8u + j0] & 63u;
            sc1 = block[4u + j1] & 63u;
            mn1 = block[8u + j1] & 63u;
        } else {
            sc0 = (block[8u + j0] & 0x0fu) | ((block[j0] >> 6) << 4);
            mn0 = (block[8u + j0] >> 4) | ((block[4u + j0] >> 6) << 4);
            sc1 = (block[8u + j1] & 0x0fu) | ((block[j1] >> 6) << 4);
            mn1 = (block[8u + j1] >> 4) | ((block[4u + j1] >> 6) << 4);
        }
        const unsigned q = block[16u + group * 32u + lane];
        const float y0 = (d * (float)sc0) * (float)(q & 0x0fu) - dmin * (float)mn0;
        const float y1 = (d * (float)sc1) * (float)(q >> 4) - dmin * (float)mn1;
        acc += y0 * input[group * 64u + lane];
        acc += y1 * input[group * 64u + lane + 32u];
    }
    return acc;
}

struct RnbMtp2Q4WideLane {
    float d;
    float dmin;
    float sc;
    float mn;
    int q_pack0;
    int q_pack1;
};

static __device__ __forceinline__ RnbMtp2Q4WideLane rnb_mtp2_q4k_wide_lane_decode(
    const unsigned char* __restrict__ block,
    unsigned j,
    unsigned elem) {
    RnbMtp2Q4WideLane out;
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
    out.d = __half2float(__ushort_as_half((unsigned short)raw_d));
    out.dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
    unsigned sc;
    unsigned mn;
    if (j < 4u) {
        sc = block[4u + j] & 63u;
        mn = block[4u + j + 4u] & 63u;
    } else {
        sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
        mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
    }
    out.sc = (float)sc;
    out.mn = (float)mn;
    const unsigned char* q_ptr = block + 16u + (j >> 1) * 32u + elem;
    const uint2 q_raw = *reinterpret_cast<const uint2*>(q_ptr);
    const unsigned shift = (j & 1u) * 4u;
    out.q_pack0 = (int)((q_raw.x >> shift) & 0x0f0f0f0fu);
    out.q_pack1 = (int)((q_raw.y >> shift) & 0x0f0f0f0fu);
    return out;
}
