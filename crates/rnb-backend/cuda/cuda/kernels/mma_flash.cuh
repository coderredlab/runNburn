#pragma once
#include <cuda_fp16.h>

// cu108: 직접 tensor core flash attention (sm_86 mma.sync). 코드베이스 첫 직접 mma 커널.
// q4k_gemv.cu 가 include → Q4K_GEMV_PARALLEL_PTX 모듈에 합류 (launch_cached_gemv 가 symbol 찾게).
extern "C" __global__ void rnb_mma_flash_probe(int* out) {
#if __CUDA_ARCH__ >= 800
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 800;
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 0;
#endif
}

// Task1 PoC: m16n8k16 QK^T 단일 타일. fragment 레이아웃 reference 검증용.
// Q: 16x16 row-major f16, K: 8x16 row-major f16 (row = kv token), S: 16x8 f32 row-major.
extern "C" __global__ void rnb_mma_qkt_tile(
    float* __restrict__ s_out,
    const __half* __restrict__ q,
    const __half* __restrict__ k) {
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    unsigned a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = gid + (i & 1) * 8;
        const int col = tid * 2 + (i >> 1) * 8;
        a[i] = (unsigned)__half_as_ushort(q[row * 16 + col + 1]) << 16
             | (unsigned)__half_as_ushort(q[row * 16 + col + 0]);
    }
    unsigned b[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int col = gid;                 // kv token 0..7
        const int row = tid * 2 + j * 8;     // contraction 0..15
        b[j] = (unsigned)__half_as_ushort(k[col * 16 + row + 1]) << 16
             | (unsigned)__half_as_ushort(k[col * 16 + row + 0]);
    }
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = gid + (i >> 1) * 8;
        const int col = tid * 2 + (i & 1);
        s_out[row * 8 + col] = c[i];
    }
#endif
}

// Task2 PoC: QK^T + online softmax (row max/sum reduce via shfl). p_out = exp(s*scale - rowmax).
extern "C" __global__ void rnb_mma_qkt_softmax_tile(
    float* __restrict__ p_out,
    const __half* __restrict__ q,
    const __half* __restrict__ k) {
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    unsigned a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = gid + (i & 1) * 8;
        const int col = tid * 2 + (i >> 1) * 8;
        a[i] = (unsigned)__half_as_ushort(q[row * 16 + col + 1]) << 16
             | (unsigned)__half_as_ushort(q[row * 16 + col + 0]);
    }
    unsigned b[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int col = gid;
        const int row = tid * 2 + j * 8;
        b[j] = (unsigned)__half_as_ushort(k[col * 16 + row + 1]) << 16
             | (unsigned)__half_as_ushort(k[col * 16 + row + 0]);
    }
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    const float scale = rsqrtf(16.0f);
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] *= scale;
    // row max: c0,c1 = row gid; c2,c3 = row gid+8. 같은 row 8 col = tid 0..3 4-thread reduce.
    float m0 = fmaxf(c[0], c[1]);
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    float m1 = fmaxf(c[2], c[3]);
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    float p[4] = {expf(c[0] - m0), expf(c[1] - m0), expf(c[2] - m1), expf(c[3] - m1)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = gid + (i >> 1) * 8;
        const int col = tid * 2 + (i & 1);
        p_out[row * 8 + col] = p[i];
    }
#endif
}

// Task3 PoC: P@V (m16n8k8). P(16x8 f32 row-major) @ V(8x16 f16, row=token) → O(16x16 f32).
// head_dim 16 = N=8 ×2. P→A operand 는 register 직접 변환 (m16n8k8 K=8 = QK^T N=8 동일 위치).
// V B operand(col-major) + kv_head stride(num_kv_heads=1) 검증.
extern "C" __global__ void rnb_mma_pv_tile(
    float* __restrict__ o_out,
    const float* __restrict__ p,
    const __half* __restrict__ v) {
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    // A operand from P (row-major 16x8): a0=(row gid, col tid*2/+1), a1=(row gid+8, ...)
    unsigned a[2];
    {
        __half lo0 = __float2half(p[gid * 8 + tid * 2 + 0]);
        __half hi0 = __float2half(p[gid * 8 + tid * 2 + 1]);
        a[0] = (unsigned)__half_as_ushort(hi0) << 16 | __half_as_ushort(lo0);
        __half lo1 = __float2half(p[(gid + 8) * 8 + tid * 2 + 0]);
        __half hi1 = __float2half(p[(gid + 8) * 8 + tid * 2 + 1]);
        a[1] = (unsigned)__half_as_ushort(hi1) << 16 | __half_as_ushort(lo1);
    }
    float o[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int hd = nt * 8 + (int)gid;  // N (head_dim) = gid; nt = N-tile
        // B operand from V (col-major): b0 = pack(V[token=tid*2][hd], V[token=tid*2+1][hd])
        __half lo = v[(tid * 2 + 0) * 16 + hd];
        __half hi = v[(tid * 2 + 1) * 16 + hd];
        unsigned bv = (unsigned)__half_as_ushort(hi) << 16 | __half_as_ushort(lo);
        float cc[4] = {0.f, 0.f, 0.f, 0.f};
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
            : "=f"(cc[0]), "=f"(cc[1]), "=f"(cc[2]), "=f"(cc[3])
            : "r"(a[0]), "r"(a[1]), "r"(bv),
              "f"(cc[0]), "f"(cc[1]), "f"(cc[2]), "f"(cc[3]));
#pragma unroll
        for (int i = 0; i < 4; ++i) o[nt][i] = cc[i];
    }
    // store O (16x16): D layout c0=(gid,tid*2) c1=(gid,tid*2+1) c2=(gid+8,..) c3=(gid+8,..)
    //                  head_dim col = nt*8 + {tid*2, tid*2+1}
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        o_out[gid * 16 + nt * 8 + tid * 2 + 0] = o[nt][0];
        o_out[gid * 16 + nt * 8 + tid * 2 + 1] = o[nt][1];
        o_out[(gid + 8) * 16 + nt * 8 + tid * 2 + 0] = o[nt][2];
        o_out[(gid + 8) * 16 + nt * 8 + tid * 2 + 1] = o[nt][3];
    }
#endif
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ float rnb_mma_mask(float v, int pos, int j, int window) {
    // window+causal: j > pos (미래) || j < pos+1-window (window 밖) → -inf
    return (j > pos || j < pos + 1 - window) ? -1e30f : v;
}
#endif

// Task4 PoC: flash 루프 (kv Bc=8 타일 순회 + online rescale + row별 window mask).
// Q/K/V: head_dim 16, Br=16 query (1 warp). O(16x16). row별 pos = q_start + row.
extern "C" __global__ void rnb_mma_flash_tile(
    float* __restrict__ o_out,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    int kv_len,
    int window,
    int q_start) {
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    const float scale = rsqrtf(16.0f);
    // Q → A operand (m16n8k16 QK^T), 타일 불변
    unsigned qa[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = gid + (i & 1) * 8;
        const int col = tid * 2 + (i >> 1) * 8;
        qa[i] = (unsigned)__half_as_ushort(q[row * 16 + col + 1]) << 16
              | (unsigned)__half_as_ushort(q[row * 16 + col + 0]);
    }
    float row_max0 = -1e30f, row_max1 = -1e30f;
    float row_sum0 = 0.f, row_sum1 = 0.f;
    float o[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const int pos0 = q_start + (int)gid;
    const int pos1 = q_start + (int)gid + 8;
    for (int kv0 = 0; kv0 < kv_len; kv0 += 8) {
        unsigned kb[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = gid;
            const int row = tid * 2 + j * 8;
            kb[j] = (unsigned)__half_as_ushort(k[(kv0 + col) * 16 + row + 1]) << 16
                  | (unsigned)__half_as_ushort(k[(kv0 + col) * 16 + row + 0]);
        }
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(kb[0]), "r"(kb[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] *= scale;
        const int kj0 = kv0 + (int)tid * 2;
        const int kj1 = kv0 + (int)tid * 2 + 1;
        c[0] = rnb_mma_mask(c[0], pos0, kj0, window);
        c[1] = rnb_mma_mask(c[1], pos0, kj1, window);
        c[2] = rnb_mma_mask(c[2], pos1, kj0, window);
        c[3] = rnb_mma_mask(c[3], pos1, kj1, window);
        float tm0 = fmaxf(c[0], c[1]);
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        float tm1 = fmaxf(c[2], c[3]);
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float nm0 = fmaxf(row_max0, tm0);
        const float nm1 = fmaxf(row_max1, tm1);
        const float corr0 = (row_max0 <= -1e30f) ? 0.f : expf(row_max0 - nm0);
        const float corr1 = (row_max1 <= -1e30f) ? 0.f : expf(row_max1 - nm1);
        float p[4] = {expf(c[0] - nm0), expf(c[1] - nm0), expf(c[2] - nm1), expf(c[3] - nm1)};
        float ps0 = p[0] + p[1];
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        float ps1 = p[2] + p[3];
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        row_sum0 = row_sum0 * corr0 + ps0;
        row_sum1 = row_sum1 * corr1 + ps1;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            o[nt][0] *= corr0;
            o[nt][1] *= corr0;
            o[nt][2] *= corr1;
            o[nt][3] *= corr1;
        }
        unsigned pa[2];
        {
            __half lo0 = __float2half(p[0]), hi0 = __float2half(p[1]);
            pa[0] = (unsigned)__half_as_ushort(hi0) << 16 | __half_as_ushort(lo0);
            __half lo1 = __float2half(p[2]), hi1 = __float2half(p[3]);
            pa[1] = (unsigned)__half_as_ushort(hi1) << 16 | __half_as_ushort(lo1);
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int hd = nt * 8 + (int)gid;
            __half lo = v[(kv0 + (int)tid * 2 + 0) * 16 + hd];
            __half hi = v[(kv0 + (int)tid * 2 + 1) * 16 + hd];
            unsigned bv = (unsigned)__half_as_ushort(hi) << 16 | __half_as_ushort(lo);
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                : "+f"(o[nt][0]), "+f"(o[nt][1]), "+f"(o[nt][2]), "+f"(o[nt][3])
                : "r"(pa[0]), "r"(pa[1]), "r"(bv));
        }
        row_max0 = nm0;
        row_max1 = nm1;
    }
    const float inv0 = row_sum0 > 0.f ? 1.f / row_sum0 : 0.f;
    const float inv1 = row_sum1 > 0.f ? 1.f / row_sum1 : 0.f;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        o_out[gid * 16 + nt * 8 + tid * 2 + 0] = o[nt][0] * inv0;
        o_out[gid * 16 + nt * 8 + tid * 2 + 1] = o[nt][1] * inv0;
        o_out[(gid + 8) * 16 + nt * 8 + tid * 2 + 0] = o[nt][2] * inv1;
        o_out[(gid + 8) * 16 + nt * 8 + tid * 2 + 1] = o[nt][3] * inv1;
    }
#endif
}

// Task5: production hd256 SWA mma flash. head_dim 256 (QK^T 16 k-step, P@V 32 N-tile),
// Br=64 (4 warp, warp당 16 q rows), GQA kv_heads, window mask. f16 K/V 직접.
// 레이아웃: q/out [pos][head][256], k/v [token][kv_head][256] (jbatch4 동일).
extern "C" __global__ void rnb_attention_prefill_flash_hd256_window_mma(
    float* __restrict__ out,
    const float* __restrict__ q,  // q_post_dev 는 f32 (jbatch4 와 동일), 내부에서 f16 변환
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window) {
#if __CUDA_ARCH__ >= 800
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    const unsigned h = blockIdx.y;
    if (h >= num_heads || num_kv_heads == 0u) return;
    const unsigned q_base = blockIdx.x * 64u + warp * 16u;
    const unsigned heads_per_kv = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_kv;
    const unsigned kstride = num_kv_heads * 256u;
    const unsigned qrow0 = q_base + gid;
    const unsigned qrow1 = q_base + gid + 8u;
    const int pos0 = (int)(kv_len - seq_len) + (int)qrow0;
    const int pos1 = (int)(kv_len - seq_len) + (int)qrow1;
    float row_max0 = -1e30f, row_max1 = -1e30f, row_sum0 = 0.f, row_sum1 = 0.f;
    float o[32][4];
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    }
    // K/V smem 타일 (4 warp 공유, k-step/N-tile 재사용). +8 pad bank conflict 회피.
    __shared__ __half ksm[8][264];
    __shared__ __half vsm[8][264];
    for (unsigned kv0 = 0; kv0 < kv_len; kv0 += 8u) {
        // 협력 K/V load (128 thread → 8 token × 256 head_dim)
        for (int idx = (int)threadIdx.x; idx < 8 * 256; idx += 128) {
            const int tok = idx >> 8;
            const int dd = idx & 255;
            const unsigned token = kv0 + (unsigned)tok;
            __half kk = (__half)0, vv = (__half)0;
            if (token < kv_len) {
                kk = k[token * kstride + kv_h * 256u + dd];
                vv = v[token * kstride + kv_h * 256u + dd];
            }
            ksm[tok][dd] = kk;
            vsm[tok][dd] = vv;
        }
        __syncthreads();
        // QK^T: head_dim 256 = 16 k-step 누적
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        for (int ks = 0; ks < 256; ks += 16) {
            unsigned qa[4], kb[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned qr = gid + (i & 1) * 8u;
                const int col = ks + (int)tid * 2 + (i >> 1) * 8;
                const unsigned qpos = q_base + qr;
                __half q0 = (__half)0, q1 = (__half)0;
                if (qpos < seq_len) {
                    q0 = __float2half(q[qpos * num_heads * 256u + h * 256u + col + 0]);
                    q1 = __float2half(q[qpos * num_heads * 256u + h * 256u + col + 1]);
                }
                qa[i] = (unsigned)__half_as_ushort(q1) << 16 | __half_as_ushort(q0);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int row = ks + (int)tid * 2 + j * 8;
                kb[j] = (unsigned)__half_as_ushort(ksm[gid][row + 1]) << 16
                      | __half_as_ushort(ksm[gid][row + 0]);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(kb[0]), "r"(kb[1]));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] *= scale;
        const int kj0 = (int)kv0 + (int)tid * 2;
        const int kj1 = (int)kv0 + (int)tid * 2 + 1;
        // token 범위 밖(kj >= kv_len)도 mask (pos < kv_len 이면 j>pos 로 잡히지만 명시)
        c[0] = (kj0 >= (int)kv_len) ? -1e30f : rnb_mma_mask(c[0], pos0, kj0, (int)window);
        c[1] = (kj1 >= (int)kv_len) ? -1e30f : rnb_mma_mask(c[1], pos0, kj1, (int)window);
        c[2] = (kj0 >= (int)kv_len) ? -1e30f : rnb_mma_mask(c[2], pos1, kj0, (int)window);
        c[3] = (kj1 >= (int)kv_len) ? -1e30f : rnb_mma_mask(c[3], pos1, kj1, (int)window);
        float tm0 = fmaxf(c[0], c[1]);
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        float tm1 = fmaxf(c[2], c[3]);
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float nm0 = fmaxf(row_max0, tm0);
        const float nm1 = fmaxf(row_max1, tm1);
        const float corr0 = (row_max0 <= -1e30f) ? 0.f : expf(row_max0 - nm0);
        const float corr1 = (row_max1 <= -1e30f) ? 0.f : expf(row_max1 - nm1);
        float p[4] = {expf(c[0] - nm0), expf(c[1] - nm0), expf(c[2] - nm1), expf(c[3] - nm1)};
        float ps0 = p[0] + p[1];
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        float ps1 = p[2] + p[3];
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        row_sum0 = row_sum0 * corr0 + ps0;
        row_sum1 = row_sum1 * corr1 + ps1;
        unsigned pa[2];
        {
            __half lo0 = __float2half(p[0]), hi0 = __float2half(p[1]);
            pa[0] = (unsigned)__half_as_ushort(hi0) << 16 | __half_as_ushort(lo0);
            __half lo1 = __float2half(p[2]), hi1 = __float2half(p[3]);
            pa[1] = (unsigned)__half_as_ushort(hi1) << 16 | __half_as_ushort(lo1);
        }
#pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= corr0;
            o[nt][1] *= corr0;
            o[nt][2] *= corr1;
            o[nt][3] *= corr1;
            const int hd = nt * 8 + (int)gid;
            __half lo = vsm[tid * 2 + 0][hd];
            __half hi = vsm[tid * 2 + 1][hd];
            unsigned bv = (unsigned)__half_as_ushort(hi) << 16 | __half_as_ushort(lo);
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                : "+f"(o[nt][0]), "+f"(o[nt][1]), "+f"(o[nt][2]), "+f"(o[nt][3])
                : "r"(pa[0]), "r"(pa[1]), "r"(bv));
        }
        row_max0 = nm0;
        row_max1 = nm1;
        __syncthreads();  // 다음 kv tile smem load 전 재사용 완료 보장
    }
    const float inv0 = row_sum0 > 0.f ? 1.f / row_sum0 : 0.f;
    const float inv1 = row_sum1 > 0.f ? 1.f / row_sum1 : 0.f;
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        const int hd = nt * 8 + (int)tid * 2;
        if (qrow0 < seq_len) {
            out[qrow0 * num_heads * 256u + h * 256u + hd + 0] = o[nt][0] * inv0;
            out[qrow0 * num_heads * 256u + h * 256u + hd + 1] = o[nt][1] * inv0;
        }
        if (qrow1 < seq_len) {
            out[qrow1 * num_heads * 256u + h * 256u + hd + 0] = o[nt][2] * inv1;
            out[qrow1 * num_heads * 256u + h * 256u + hd + 1] = o[nt][3] * inv1;
        }
    }
#endif
}

// hd256 stream-K partial: query64 tensor-core tile over one KV chunk.
// Emits the same unnormalized (max, sum, accumulator) ABI as the F32 split kernel.
extern "C" __global__ void rnb_attention_prefill_flash_hd256_window_mma_stream_k_partials(
    float* __restrict__ partial_acc,
    float* __restrict__ partial_meta,
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window,
    unsigned chunk_size,
    unsigned num_chunks) {
#if __CUDA_ARCH__ >= 800
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2;
    const unsigned tid = lane & 3u;
    const unsigned h = blockIdx.y;
    const unsigned chunk = blockIdx.z;
    if (h >= num_heads || num_kv_heads == 0u || chunk >= num_chunks
        || chunk_size == 0u || kv_len < seq_len || window < kv_len) {
        return;
    }

    const unsigned query_tile_base = blockIdx.x * 64u;
    const unsigned q_local_base = warp * 16u;
    const unsigned qrow0 = query_tile_base + q_local_base + gid;
    const unsigned qrow1 = qrow0 + 8u;
    const unsigned heads_per_kv = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_kv;
    const unsigned kstride = num_kv_heads * 256u;
    const unsigned chunk_start = chunk * chunk_size;
    unsigned chunk_end = chunk_start + chunk_size;
    if (chunk_end > kv_len) {
        chunk_end = kv_len;
    }
    const int pos0 = (int)(kv_len - seq_len) + (int)qrow0;
    const int pos1 = (int)(kv_len - seq_len) + (int)qrow1;
    const int window_start0 = pos0 + 1 > (int)window ? pos0 + 1 - (int)window : 0;
    const int window_start1 = pos1 + 1 > (int)window ? pos1 + 1 - (int)window : 0;
    const bool has_values0 = qrow0 < seq_len && chunk_start < chunk_end
        && (int)chunk_start <= pos0 && (int)chunk_end > window_start0;
    const bool has_values1 = qrow1 < seq_len && chunk_start < chunk_end
        && (int)chunk_start <= pos1 && (int)chunk_end > window_start1;

    float row_max0 = -1e30f;
    float row_max1 = -1e30f;
    float row_sum0 = 0.0f;
    float row_sum1 = 0.0f;
    float o[32][4];
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        o[nt][0] = 0.0f;
        o[nt][1] = 0.0f;
        o[nt][2] = 0.0f;
        o[nt][3] = 0.0f;
    }

    // Q is invariant across KV chunks. Convert the query64 tile once per CTA.
    __shared__ __half qsm[64][264];
    __shared__ __half ksm[8][264];
    __shared__ __half vsm[8][264];
    for (int idx = (int)threadIdx.x; idx < 64 * 256; idx += 128) {
        const unsigned q_local = (unsigned)idx >> 8;
        const unsigned dd = (unsigned)idx & 255u;
        const unsigned qpos = query_tile_base + q_local;
        qsm[q_local][dd] = qpos < seq_len
            ? __float2half(q[qpos * num_heads * 256u + h * 256u + dd])
            : (__half)0;
    }
    __syncthreads();

    for (unsigned kv0 = chunk_start; kv0 < chunk_end; kv0 += 8u) {
        for (int idx = (int)threadIdx.x; idx < 8 * 256; idx += 128) {
            const unsigned tok = (unsigned)idx >> 8;
            const unsigned dd = (unsigned)idx & 255u;
            const unsigned token = kv0 + tok;
            __half kk = (__half)0;
            __half vv = (__half)0;
            if (token < chunk_end) {
                kk = k[token * kstride + kv_h * 256u + dd];
                vv = v[token * kstride + kv_h * 256u + dd];
            }
            ksm[tok][dd] = kk;
            vsm[tok][dd] = vv;
        }
        __syncthreads();

        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int ks = 0; ks < 256; ks += 16) {
            unsigned qa[4];
            unsigned qla[4];
            unsigned kb[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned qr = gid + (unsigned)(i & 1) * 8u;
                const int col = ks + (int)tid * 2 + (i >> 1) * 8;
                const unsigned q_local = q_local_base + qr;
                const __half q0 = qsm[q_local][col + 0];
                const __half q1 = qsm[q_local][col + 1];
                qa[i] = (unsigned)__half_as_ushort(q1) << 16
                    | __half_as_ushort(q0);
                const unsigned qpos = query_tile_base + q_local;
                const float q0_original = qpos < seq_len
                    ? q[qpos * num_heads * 256u + h * 256u + col + 0]
                    : 0.0f;
                const float q1_original = qpos < seq_len
                    ? q[qpos * num_heads * 256u + h * 256u + col + 1]
                    : 0.0f;
                const __half q0_low = __float2half(q0_original - __half2float(q0));
                const __half q1_low = __float2half(q1_original - __half2float(q1));
                qla[i] = (unsigned)__half_as_ushort(q1_low) << 16
                    | __half_as_ushort(q0_low);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = ks + (int)tid * 2 + i * 8;
                kb[i] = (unsigned)__half_as_ushort(ksm[gid][row + 1]) << 16
                    | __half_as_ushort(ksm[gid][row + 0]);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(kb[0]), "r"(kb[1]));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(qla[0]), "r"(qla[1]), "r"(qla[2]), "r"(qla[3]), "r"(kb[0]), "r"(kb[1]));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[i] *= scale;
        }
        const int kj0 = (int)kv0 + (int)tid * 2;
        const int kj1 = kj0 + 1;
        c[0] = kj0 >= (int)chunk_end ? -1e30f : rnb_mma_mask(c[0], pos0, kj0, (int)window);
        c[1] = kj1 >= (int)chunk_end ? -1e30f : rnb_mma_mask(c[1], pos0, kj1, (int)window);
        c[2] = kj0 >= (int)chunk_end ? -1e30f : rnb_mma_mask(c[2], pos1, kj0, (int)window);
        c[3] = kj1 >= (int)chunk_end ? -1e30f : rnb_mma_mask(c[3], pos1, kj1, (int)window);

        float tile_max0 = fmaxf(c[0], c[1]);
        tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, 1));
        tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, 2));
        float tile_max1 = fmaxf(c[2], c[3]);
        tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, 1));
        tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, 2));
        const float new_max0 = fmaxf(row_max0, tile_max0);
        const float new_max1 = fmaxf(row_max1, tile_max1);
        const float correction0 = row_max0 <= -1e30f ? 0.0f : expf(row_max0 - new_max0);
        const float correction1 = row_max1 <= -1e30f ? 0.0f : expf(row_max1 - new_max1);
        float probability[4] = {
            expf(c[0] - new_max0),
            expf(c[1] - new_max0),
            expf(c[2] - new_max1),
            expf(c[3] - new_max1),
        };
        float probability_sum0 = probability[0] + probability[1];
        probability_sum0 += __shfl_xor_sync(0xffffffffu, probability_sum0, 1);
        probability_sum0 += __shfl_xor_sync(0xffffffffu, probability_sum0, 2);
        float probability_sum1 = probability[2] + probability[3];
        probability_sum1 += __shfl_xor_sync(0xffffffffu, probability_sum1, 1);
        probability_sum1 += __shfl_xor_sync(0xffffffffu, probability_sum1, 2);
        row_sum0 = row_sum0 * correction0 + probability_sum0;
        row_sum1 = row_sum1 * correction1 + probability_sum1;
        const __half p00 = __float2half(probability[0]);
        const __half p01 = __float2half(probability[1]);
        const __half p10 = __float2half(probability[2]);
        const __half p11 = __float2half(probability[3]);
        const __half p00_low = __float2half(probability[0] - __half2float(p00));
        const __half p01_low = __float2half(probability[1] - __half2float(p01));
        const __half p10_low = __float2half(probability[2] - __half2float(p10));
        const __half p11_low = __float2half(probability[3] - __half2float(p11));
        const unsigned pa0 = (unsigned)__half_as_ushort(p01) << 16 | __half_as_ushort(p00);
        const unsigned pa1 = (unsigned)__half_as_ushort(p11) << 16 | __half_as_ushort(p10);
        const unsigned pa0_low = (unsigned)__half_as_ushort(p01_low) << 16
            | __half_as_ushort(p00_low);
        const unsigned pa1_low = (unsigned)__half_as_ushort(p11_low) << 16
            | __half_as_ushort(p10_low);

#pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= correction0;
            o[nt][1] *= correction0;
            o[nt][2] *= correction1;
            o[nt][3] *= correction1;
            const int hd = nt * 8 + (int)gid;
            const __half value0 = vsm[tid * 2 + 0][hd];
            const __half value1 = vsm[tid * 2 + 1][hd];
            const unsigned bv = (unsigned)__half_as_ushort(value1) << 16
                | __half_as_ushort(value0);
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                : "+f"(o[nt][0]), "+f"(o[nt][1]), "+f"(o[nt][2]), "+f"(o[nt][3])
                : "r"(pa0), "r"(pa1), "r"(bv));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                : "+f"(o[nt][0]), "+f"(o[nt][1]), "+f"(o[nt][2]), "+f"(o[nt][3])
                : "r"(pa0_low), "r"(pa1_low), "r"(bv));
        }
        row_max0 = new_max0;
        row_max1 = new_max1;
        __syncthreads();
    }

    const float empty_max = -3.4028234663852886e38f;
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        const unsigned hd = (unsigned)nt * 8u + tid * 2u;
        if (qrow0 < seq_len) {
            const unsigned partial_row = (qrow0 * num_heads + h) * num_chunks + chunk;
            partial_acc[partial_row * 256u + hd + 0u] = has_values0 ? o[nt][0] : 0.0f;
            partial_acc[partial_row * 256u + hd + 1u] = has_values0 ? o[nt][1] : 0.0f;
        }
        if (qrow1 < seq_len) {
            const unsigned partial_row = (qrow1 * num_heads + h) * num_chunks + chunk;
            partial_acc[partial_row * 256u + hd + 0u] = has_values1 ? o[nt][2] : 0.0f;
            partial_acc[partial_row * 256u + hd + 1u] = has_values1 ? o[nt][3] : 0.0f;
        }
    }
    if (tid == 0u) {
        if (qrow0 < seq_len) {
            const unsigned partial_row = (qrow0 * num_heads + h) * num_chunks + chunk;
            partial_meta[partial_row * 2u] = has_values0 ? row_max0 : empty_max;
            partial_meta[partial_row * 2u + 1u] = has_values0 ? row_sum0 : 0.0f;
        }
        if (qrow1 < seq_len) {
            const unsigned partial_row = (qrow1 * num_heads + h) * num_chunks + chunk;
            partial_meta[partial_row * 2u] = has_values1 ? row_max1 : empty_max;
            partial_meta[partial_row * 2u + 1u] = has_values1 ? row_sum1 : 0.0f;
        }
    }
#endif
}


// cu113: hd512 FULL attention (Gemma4 full layer). hd256 window 커널 확장 —
// head_dim 512, QK^T 32 k-step, P@V 64 N-tile. window 없음 = causal only.
// O register o[64][4] = 256 f32/thread (hd256 의 2배): occupancy 위험, ptxas -v 로
// spill 확인. hd512 FULL 은 prefill 의 24배 격차 주범(SWA 9.77ms vs FULL 50~95ms).
// 레이아웃: q/out [pos][head][512], k/v [token][kv_head][512] (jbatch4 동일).
extern "C" __global__ void rnb_attention_prefill_flash_hd512_mma(
    float* __restrict__ out,
    const float* __restrict__ q,  // q_post_dev f32, 내부 f16 변환
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
#if __CUDA_ARCH__ >= 800
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2, tid = lane & 3u;
    const unsigned h = blockIdx.y;
    if (h >= num_heads || num_kv_heads == 0u) return;
    const unsigned q_base = blockIdx.x * 64u + warp * 16u;
    const unsigned heads_per_kv = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_kv;
    const unsigned kstride = num_kv_heads * 512u;
    const unsigned qrow0 = q_base + gid;
    const unsigned qrow1 = q_base + gid + 8u;
    const int pos0 = (int)(kv_len - seq_len) + (int)qrow0;
    const int pos1 = (int)(kv_len - seq_len) + (int)qrow1;
    float row_max0 = -1e30f, row_max1 = -1e30f, row_sum0 = 0.f, row_sum1 = 0.f;
    float o[64][4];
#pragma unroll
    for (int nt = 0; nt < 64; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    }
    // K/V smem 타일 (4 warp 공유). +8 pad bank conflict 회피. head_dim 512.
    __shared__ __half ksm[8][520];
    __shared__ __half vsm[8][520];
    for (unsigned kv0 = 0; kv0 < kv_len; kv0 += 8u) {
        // 협력 K/V load (128 thread → 8 token × 512 head_dim)
        for (int idx = (int)threadIdx.x; idx < 8 * 512; idx += 128) {
            const int tok = idx >> 9;  // / 512
            const int dd = idx & 511;
            const unsigned token = kv0 + (unsigned)tok;
            __half kk = (__half)0, vv = (__half)0;
            if (token < kv_len) {
                kk = k[token * kstride + kv_h * 512u + dd];
                vv = v[token * kstride + kv_h * 512u + dd];
            }
            ksm[tok][dd] = kk;
            vsm[tok][dd] = vv;
        }
        __syncthreads();
        // QK^T: head_dim 512 = 32 k-step 누적
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        for (int ks = 0; ks < 512; ks += 16) {
            unsigned qa[4], kb[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned qr = gid + (i & 1) * 8u;
                const int col = ks + (int)tid * 2 + (i >> 1) * 8;
                const unsigned qpos = q_base + qr;
                __half q0 = (__half)0, q1 = (__half)0;
                if (qpos < seq_len) {
                    q0 = __float2half(q[qpos * num_heads * 512u + h * 512u + col + 0]);
                    q1 = __float2half(q[qpos * num_heads * 512u + h * 512u + col + 1]);
                }
                qa[i] = (unsigned)__half_as_ushort(q1) << 16 | __half_as_ushort(q0);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int row = ks + (int)tid * 2 + j * 8;
                kb[j] = (unsigned)__half_as_ushort(ksm[gid][row + 1]) << 16
                      | __half_as_ushort(ksm[gid][row + 0]);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(kb[0]), "r"(kb[1]));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] *= scale;
        const int kj0 = (int)kv0 + (int)tid * 2;
        const int kj1 = (int)kv0 + (int)tid * 2 + 1;
        // FULL causal mask: j > pos 또는 범위 밖이면 -inf (window 없음)
        c[0] = (kj0 >= (int)kv_len || kj0 > pos0) ? -1e30f : c[0];
        c[1] = (kj1 >= (int)kv_len || kj1 > pos0) ? -1e30f : c[1];
        c[2] = (kj0 >= (int)kv_len || kj0 > pos1) ? -1e30f : c[2];
        c[3] = (kj1 >= (int)kv_len || kj1 > pos1) ? -1e30f : c[3];
        float tm0 = fmaxf(c[0], c[1]);
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        float tm1 = fmaxf(c[2], c[3]);
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float nm0 = fmaxf(row_max0, tm0);
        const float nm1 = fmaxf(row_max1, tm1);
        const float corr0 = (row_max0 <= -1e30f) ? 0.f : expf(row_max0 - nm0);
        const float corr1 = (row_max1 <= -1e30f) ? 0.f : expf(row_max1 - nm1);
        float p[4] = {expf(c[0] - nm0), expf(c[1] - nm0), expf(c[2] - nm1), expf(c[3] - nm1)};
        float ps0 = p[0] + p[1];
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        float ps1 = p[2] + p[3];
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        row_sum0 = row_sum0 * corr0 + ps0;
        row_sum1 = row_sum1 * corr1 + ps1;
        unsigned pa[2];
        {
            __half lo0 = __float2half(p[0]), hi0 = __float2half(p[1]);
            pa[0] = (unsigned)__half_as_ushort(hi0) << 16 | __half_as_ushort(lo0);
            __half lo1 = __float2half(p[2]), hi1 = __float2half(p[3]);
            pa[1] = (unsigned)__half_as_ushort(hi1) << 16 | __half_as_ushort(lo1);
        }
#pragma unroll
        for (int nt = 0; nt < 64; ++nt) {
            o[nt][0] *= corr0;
            o[nt][1] *= corr0;
            o[nt][2] *= corr1;
            o[nt][3] *= corr1;
            const int hd = nt * 8 + (int)gid;
            __half lo = vsm[tid * 2 + 0][hd];
            __half hi = vsm[tid * 2 + 1][hd];
            unsigned bv = (unsigned)__half_as_ushort(hi) << 16 | __half_as_ushort(lo);
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                : "+f"(o[nt][0]), "+f"(o[nt][1]), "+f"(o[nt][2]), "+f"(o[nt][3])
                : "r"(pa[0]), "r"(pa[1]), "r"(bv));
        }
        row_max0 = nm0;
        row_max1 = nm1;
        __syncthreads();
    }
    const float inv0 = row_sum0 > 0.f ? 1.f / row_sum0 : 0.f;
    const float inv1 = row_sum1 > 0.f ? 1.f / row_sum1 : 0.f;
#pragma unroll
    for (int nt = 0; nt < 64; ++nt) {
        const int hd = nt * 8 + (int)tid * 2;
        if (qrow0 < seq_len) {
            out[qrow0 * num_heads * 512u + h * 512u + hd + 0] = o[nt][0] * inv0;
            out[qrow0 * num_heads * 512u + h * 512u + hd + 1] = o[nt][1] * inv0;
        }
        if (qrow1 < seq_len) {
            out[qrow1 * num_heads * 512u + h * 512u + hd + 0] = o[nt][2] * inv1;
            out[qrow1 * num_heads * 512u + h * 512u + hd + 1] = o[nt][3] * inv1;
        }
    }
#endif
}

extern "C" __global__ void rnb_glm_mla_prefill_scores_f16(
    float* __restrict__ scores,
    const float* __restrict__ q_absorbed,
    const float* __restrict__ q_pe,
    const __half* __restrict__ cache,
    unsigned pos_start,
    unsigned seq_len,
    unsigned num_heads,
    unsigned kv_len,
    unsigned kv_rank,
    unsigned rope_dim,
    unsigned kv_width,
    float scale) {
    const unsigned query = blockIdx.x;
    const unsigned token = query / num_heads;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned key = blockIdx.y * 4u + warp;
    if (token >= seq_len || key >= pos_start + token + 1u || key >= kv_len) {
        return;
    }

    const float* q_latent = q_absorbed + query * kv_rank;
    const float* q_rope = q_pe + query * rope_dim;
    const __half* cached = cache + key * kv_width;
    float dot = 0.0f;
    for (unsigned dim = lane; dim < kv_rank; dim += 32u) {
        dot += q_latent[dim] * __half2float(cached[dim]);
    }
    for (unsigned dim = lane; dim < rope_dim; dim += 32u) {
        dot += q_rope[dim] * __half2float(cached[kv_rank + dim]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    if (lane == 0u) {
        scores[query * kv_len + key] = dot * scale;
    }
}

extern "C" __global__ void rnb_glm_mla_prefill_softmax(
    float* __restrict__ scores,
    unsigned pos_start,
    unsigned seq_len,
    unsigned num_heads,
    unsigned kv_len) {
    const unsigned query = blockIdx.x;
    const unsigned token = query / num_heads;
    if (token >= seq_len) {
        return;
    }
    const unsigned attend_len = min(kv_len, pos_start + token + 1u);
    float* row = scores + query * kv_len;
    __shared__ float reduce[256];

    float local_max = -3.402823466e+38F;
    for (unsigned key = threadIdx.x; key < attend_len; key += blockDim.x) {
        local_max = fmaxf(local_max, row[key]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduce[0];

    float local_sum = 0.0f;
    for (unsigned key = threadIdx.x; key < attend_len; key += blockDim.x) {
        const float probability = expf(row[key] - row_max);
        row[key] = probability;
        local_sum += probability;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (unsigned key = threadIdx.x; key < attend_len; key += blockDim.x) {
        row[key] *= inverse_sum;
    }
}

extern "C" __global__ void rnb_glm_mla_prefill_weighted_f16(
    float* __restrict__ output,
    const float* __restrict__ scores,
    const __half* __restrict__ cache,
    unsigned pos_start,
    unsigned seq_len,
    unsigned num_heads,
    unsigned kv_len,
    unsigned kv_rank,
    unsigned kv_width) {
    const unsigned query = blockIdx.x;
    const unsigned token = query / num_heads;
    if (token >= seq_len) {
        return;
    }
    const unsigned attend_len = min(kv_len, pos_start + token + 1u);
    const float* probabilities = scores + query * kv_len;
    float* out = output + query * kv_rank;
    for (unsigned dim = threadIdx.x; dim < kv_rank; dim += blockDim.x) {
        float sum = 0.0f;
        for (unsigned key = 0; key < attend_len; ++key) {
            sum += probabilities[key] * __half2float(cache[key * kv_width + dim]);
        }
        out[dim] = sum;
    }
}
