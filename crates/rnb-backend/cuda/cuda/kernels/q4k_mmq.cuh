#include <cuda_pipeline.h>

// Q4_K x Q8_1 tiled matrix multiply for Ampere-class integer tensor cores.
//
// One 8-warp CTA computes a 32-row x 32-sequence output tile. Compared with
// the legacy 64x8 carrier this spends more activation traffic to reuse each
// packed Q4_K weight tile across four sequence groups and halves CTA count.

extern "C" __global__ void rnb_q4k_q8_1_matmul_mmq_tile32(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 750
    (void)out;
    (void)weights;
    (void)input_qs;
    (void)input_ds;
    (void)rows;
    (void)blocks_per_row;
    (void)seq_len;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 32u;
    const unsigned seq_base = blockIdx.y * 32u;
    const unsigned warp_row_off = (warp & 1u) * 16u;
    const unsigned warp_seq_off = (warp >> 1) * 8u;

    __shared__ signed char a_tile[32 * 36];
    __shared__ signed char b_tile[32 * 36];
    __shared__ float x_d[32];
    __shared__ float x_dmin[32];
    __shared__ unsigned char x_sc[32];
    __shared__ unsigned char x_mn[32];
    __shared__ float y_d[32];
    __shared__ float y_s[32];

    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;
    const unsigned t_col_a = warp_seq_off + ((lane & 3u) << 1);
    const unsigned t_col_b = t_col_a + 1u;
    const unsigned row_a = row_base + warp_row_off + t_row_a;
    const unsigned row_b = row_base + warp_row_off + t_row_b;
    const unsigned seq_a = seq_base + t_col_a;
    const unsigned seq_b = seq_base + t_col_b;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        float block_d_a0 = 0.0f;
        float block_d_a1 = 0.0f;
        float block_d_b0 = 0.0f;
        float block_d_b1 = 0.0f;
        float block_m_a0 = 0.0f;
        float block_m_a1 = 0.0f;
        float block_m_b0 = 0.0f;
        float block_m_b1 = 0.0f;

        for (unsigned sub = 0; sub < 8u; ++sub) {
            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* a_dst = a_tile + load_row * 36u + load_off;

            if (global_row < rows) {
                const unsigned char* packed = weights + global_row * row_bytes + block * 144u;
                if (sub == 0u && load_off == 0u) {
                    const unsigned raw_d = static_cast<unsigned>(packed[0])
                        | (static_cast<unsigned>(packed[1]) << 8);
                    const unsigned raw_dmin = static_cast<unsigned>(packed[2])
                        | (static_cast<unsigned>(packed[3]) << 8);
                    x_d[load_row] = __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                    x_dmin[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_dmin)));
                }
                if (load_off == 0u) {
                    unsigned scale;
                    unsigned minimum;
                    if (sub < 4u) {
                        scale = packed[4u + sub] & 63u;
                        minimum = packed[8u + sub] & 63u;
                    } else {
                        scale = (packed[8u + sub] & 0x0fu)
                            | ((packed[sub] >> 6) << 4);
                        minimum = (packed[8u + sub] >> 4)
                            | ((packed[4u + sub] >> 6) << 4);
                    }
                    x_sc[load_row] = static_cast<unsigned char>(scale);
                    x_mn[load_row] = static_cast<unsigned char>(minimum);
                }
                const unsigned nibble_base = 16u + (sub >> 1) * 32u;
                const unsigned packed_qs = *reinterpret_cast<const unsigned*>(packed + nibble_base + load_off);
                const unsigned unpacked = ((sub & 1u) == 0u) ? (packed_qs & 0x0f0f0f0fu)
                                                               : ((packed_qs >> 4) & 0x0f0f0f0fu);
                *reinterpret_cast<unsigned*>(a_dst) = unpacked;
            } else {
                *reinterpret_cast<unsigned*>(a_dst) = 0u;
                if (sub == 0u && load_off == 0u) {
                    x_d[load_row] = 0.0f;
                    x_dmin[load_row] = 0.0f;
                }
                if (load_off == 0u) {
                    x_sc[load_row] = 0u;
                    x_mn[load_row] = 0u;
                }
            }

            // cu223: b-tile 로더가 자신이 나른 4B word 를 그 자리에서 dp4a 로
            // 접고 8-lane shuffle 로 chunk 합을 만들어 y_s 에 기록한다.
            // (기존에는 모든 warp 가 min-term 합을 dp4a 루프로 재계산 — LSU
            // 파이프의 최대 소비자였다. int 덧셈이라 결과는 bitwise 동일.)
            const unsigned load_seq = tid >> 3;
            const unsigned seq_off = (tid & 7u) * 4u;
            const unsigned global_seq = seq_base + load_seq;
            signed char* b_dst = b_tile + load_seq * 36u + seq_off;
            int b_word = 0;
            if (global_seq < seq_len) {
                const unsigned chunk = block * 8u + sub;
                const signed char* b_src = input_qs +
                    (global_seq * blocks_per_row * 256u) + chunk * 32u + seq_off;
                b_word = *reinterpret_cast<const int*>(b_src);
                *reinterpret_cast<int*>(b_dst) = b_word;
                if (seq_off == 0u) {
                    y_d[load_seq] = input_ds[global_seq * blocks_per_row * 8u + chunk];
                }
            } else {
                *reinterpret_cast<int*>(b_dst) = 0;
                if (seq_off == 0u) {
                    y_d[load_seq] = 0.0f;
                }
            }
            int chunk_sum = __dp4a(0x01010101, b_word, 0);
            chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 4u, 8);
            chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 2u, 8);
            chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 1u, 8);
            if ((tid & 7u) == 0u) {
                y_s[load_seq] = static_cast<float>(chunk_sum);
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int a0 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_a) * 36u + a_col_lo]);
            const int a1 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_b) * 36u + a_col_lo]);
            const int a2 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_a) * 36u + a_col_hi]);
            const int a3 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_b) * 36u + a_col_hi]);

            const unsigned b_seq = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_lo]);
            const int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_hi]);

            int d0 = 0;
            int d1 = 0;
            int d2 = 0;
            int d3 = 0;
            rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            const bool seq_a_valid = seq_a < seq_len;
            const bool seq_b_valid = seq_b < seq_len;
            const float sum_qy_a = y_s[t_col_a];
            const float sum_qy_b = y_s[t_col_b];
            const float dy_a = seq_a_valid ? y_d[t_col_a] : 0.0f;
            const float dy_b = seq_b_valid ? y_d[t_col_b] : 0.0f;
            const float scale_a = static_cast<float>(x_sc[warp_row_off + t_row_a]);
            const float scale_b = static_cast<float>(x_sc[warp_row_off + t_row_b]);
            const float min_a = static_cast<float>(x_mn[warp_row_off + t_row_a]);
            const float min_b = static_cast<float>(x_mn[warp_row_off + t_row_b]);

            block_d_a0 += dy_a * scale_a * static_cast<float>(d0);
            block_d_a1 += dy_b * scale_a * static_cast<float>(d1);
            block_d_b0 += dy_a * scale_b * static_cast<float>(d2);
            block_d_b1 += dy_b * scale_b * static_cast<float>(d3);
            block_m_a0 += dy_a * min_a * sum_qy_a;
            block_m_a1 += dy_b * min_a * sum_qy_b;
            block_m_b0 += dy_a * min_b * sum_qy_a;
            block_m_b1 += dy_b * min_b * sum_qy_b;
            __syncthreads();
        }

        const float d_a = x_d[warp_row_off + t_row_a];
        const float d_b = x_d[warp_row_off + t_row_b];
        const float dmin_a = x_dmin[warp_row_off + t_row_a];
        const float dmin_b = x_dmin[warp_row_off + t_row_b];
        acc[0] += d_a * block_d_a0 - dmin_a * block_m_a0;
        acc[1] += d_a * block_d_a1 - dmin_a * block_m_a1;
        acc[2] += d_b * block_d_b0 - dmin_b * block_m_b0;
        acc[3] += d_b * block_d_b1 - dmin_b * block_m_b1;
        __syncthreads();
    }

    if (row_a_valid && seq_a < seq_len) out[seq_a * rows + row_a] = acc[0];
    if (row_a_valid && seq_b < seq_len) out[seq_b * rows + row_a] = acc[1];
    if (row_b_valid && seq_a < seq_len) out[seq_a * rows + row_b] = acc[2];
    if (row_b_valid && seq_b < seq_len) out[seq_b * rows + row_b] = acc[3];
#endif
}

// cu226: 32-row x 64-sequence variant. ncu attribution after the cu223
// chunk-sum fusion kept the kernel LSU-bound with grid.y=36 CTAs re-loading
// and re-unpacking the same Q4_K weight tile per 32-seq slab. Doubling the
// CTA sequence width halves grid.y, so every a-tile global load, nibble
// unpack, and scale/min fetch is amortized over twice the output columns.
// Per-element accumulation order is identical to the tile32 kernel (same
// block/sub loop, same mma fragment mapping), so outputs stay bitwise equal.
// __launch_bounds__(256, 4) caps registers at 64: the free-allocation build
// used 72 registers, dropping occupancy to 3 CTAs/SM (50%) and leaving the
// memory pipes idle at 66% (tile32 saturated them at 86%) — the latency
// hiding lost to register pressure was worth more than the spill risk.
extern "C" __global__ void __launch_bounds__(256, 4) rnb_q4k_q8_1_matmul_mmq_tile32_seq64(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 750
    (void)out;
    (void)weights;
    (void)input_qs;
    (void)input_ds;
    (void)rows;
    (void)blocks_per_row;
    (void)seq_len;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 32u;
    const unsigned seq_base = blockIdx.y * 64u;
    const unsigned warp_row_off = (warp & 1u) * 16u;
    const unsigned warp_seq_off = (warp >> 1) * 16u;

    __shared__ signed char a_tile[32 * 36];
    __shared__ signed char b_tile[64 * 36];
    // cu227: raw Q4_K block staging — 144B rows padded to 160B (16B-aligned
    // cp.async rows, 40-word stride keeps the 8-thread qs reads bank-clean).
    __shared__ unsigned char raw_stage[32 * 160];
    __shared__ float x_d[32];
    __shared__ float x_dmin[32];
    __shared__ unsigned char x_sc[32];
    __shared__ unsigned char x_mn[32];
    __shared__ float y_d[64];
    __shared__ float y_s[64];

    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;
    const unsigned t_col_a = warp_seq_off + ((lane & 3u) << 1);
    const unsigned t_col_b = t_col_a + 1u;
    const unsigned t_col_c = t_col_a + 8u;
    const unsigned t_col_d = t_col_b + 8u;
    const unsigned row_a = row_base + warp_row_off + t_row_a;
    const unsigned row_b = row_base + warp_row_off + t_row_b;
    const unsigned seq_a = seq_base + t_col_a;
    const unsigned seq_b = seq_base + t_col_b;
    const unsigned seq_c = seq_base + t_col_c;
    const unsigned seq_d = seq_base + t_col_d;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        float block_d[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float block_m[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // cu227: stage the 32 raw 144B blocks once per block iteration with
        // 16B async copies (9 chunks/row, 288 total). The former per-sub
        // loader re-read the same qs window and scale bytes from global on
        // every sub — misaligned 32B windows (block*144+16 straddles sectors)
        // and scattered byte loads were 54% excessive sectors in ncu. All
        // sub-iterations now unpack from shared; arithmetic is unchanged, so
        // outputs stay bitwise equal.
        for (unsigned c = tid; c < 288u; c += 256u) {
            const unsigned s_row = c / 9u;
            const unsigned s_off = (c % 9u) * 16u;
            const unsigned g_row = row_base + s_row;
            if (g_row < rows) {
                __pipeline_memcpy_async(
                    raw_stage + s_row * 160u + s_off,
                    weights + g_row * row_bytes + block * 144u + s_off,
                    16);
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        for (unsigned sub = 0; sub < 8u; ++sub) {
            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* a_dst = a_tile + load_row * 36u + load_off;

            if (global_row < rows) {
                const unsigned char* packed = raw_stage + load_row * 160u;
                if (sub == 0u && load_off == 0u) {
                    const unsigned raw_d = static_cast<unsigned>(packed[0])
                        | (static_cast<unsigned>(packed[1]) << 8);
                    const unsigned raw_dmin = static_cast<unsigned>(packed[2])
                        | (static_cast<unsigned>(packed[3]) << 8);
                    x_d[load_row] = __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                    x_dmin[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_dmin)));
                }
                if (load_off == 0u) {
                    unsigned scale;
                    unsigned minimum;
                    if (sub < 4u) {
                        scale = packed[4u + sub] & 63u;
                        minimum = packed[8u + sub] & 63u;
                    } else {
                        scale = (packed[8u + sub] & 0x0fu)
                            | ((packed[sub] >> 6) << 4);
                        minimum = (packed[8u + sub] >> 4)
                            | ((packed[4u + sub] >> 6) << 4);
                    }
                    x_sc[load_row] = static_cast<unsigned char>(scale);
                    x_mn[load_row] = static_cast<unsigned char>(minimum);
                }
                const unsigned nibble_base = 16u + (sub >> 1) * 32u;
                const unsigned packed_qs = *reinterpret_cast<const unsigned*>(packed + nibble_base + load_off);
                const unsigned unpacked = ((sub & 1u) == 0u) ? (packed_qs & 0x0f0f0f0fu)
                                                               : ((packed_qs >> 4) & 0x0f0f0f0fu);
                *reinterpret_cast<unsigned*>(a_dst) = unpacked;
            } else {
                *reinterpret_cast<unsigned*>(a_dst) = 0u;
                if (sub == 0u && load_off == 0u) {
                    x_d[load_row] = 0.0f;
                    x_dmin[load_row] = 0.0f;
                }
                if (load_off == 0u) {
                    x_sc[load_row] = 0u;
                    x_mn[load_row] = 0u;
                }
            }

            // 64-seq b-tile: 256 threads cover 32 sequences per pass, so two
            // passes fill the slab. The cu223 loader-side chunk-sum fold runs
            // per pass (each 8-lane group folds the word it just carried).
            const unsigned chunk = block * 8u + sub;
#pragma unroll
            for (unsigned pass = 0; pass < 2u; ++pass) {
                const unsigned load_seq = (tid >> 3) + pass * 32u;
                const unsigned seq_off = (tid & 7u) * 4u;
                const unsigned global_seq = seq_base + load_seq;
                signed char* b_dst = b_tile + load_seq * 36u + seq_off;
                int b_word = 0;
                if (global_seq < seq_len) {
                    const signed char* b_src = input_qs +
                        (global_seq * blocks_per_row * 256u) + chunk * 32u + seq_off;
                    b_word = *reinterpret_cast<const int*>(b_src);
                    *reinterpret_cast<int*>(b_dst) = b_word;
                    if (seq_off == 0u) {
                        y_d[load_seq] = input_ds[global_seq * blocks_per_row * 8u + chunk];
                    }
                } else {
                    *reinterpret_cast<int*>(b_dst) = 0;
                    if (seq_off == 0u) {
                        y_d[load_seq] = 0.0f;
                    }
                }
                int chunk_sum = __dp4a(0x01010101, b_word, 0);
                chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 4u, 8);
                chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 2u, 8);
                chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 1u, 8);
                if ((tid & 7u) == 0u) {
                    y_s[load_seq] = static_cast<float>(chunk_sum);
                }
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int a0 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_a) * 36u + a_col_lo]);
            const int a1 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_b) * 36u + a_col_lo]);
            const int a2 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_a) * 36u + a_col_hi]);
            const int a3 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_b) * 36u + a_col_hi]);

            const float scale_a = static_cast<float>(x_sc[warp_row_off + t_row_a]);
            const float scale_b = static_cast<float>(x_sc[warp_row_off + t_row_b]);
            const float min_a = static_cast<float>(x_mn[warp_row_off + t_row_a]);
            const float min_b = static_cast<float>(x_mn[warp_row_off + t_row_b]);

            const unsigned b_seq0 = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
#pragma unroll
            for (unsigned half = 0; half < 2u; ++half) {
                const unsigned b_seq = b_seq0 + half * 8u;
                const int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_lo]);
                const int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_hi]);

                int d0 = 0;
                int d1 = 0;
                int d2 = 0;
                int d3 = 0;
                rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

                const unsigned col_lo = t_col_a + half * 8u;
                const unsigned col_hi = t_col_b + half * 8u;
                const bool seq_lo_valid = (seq_base + col_lo) < seq_len;
                const bool seq_hi_valid = (seq_base + col_hi) < seq_len;
                const float sum_qy_lo = y_s[col_lo];
                const float sum_qy_hi = y_s[col_hi];
                const float dy_lo = seq_lo_valid ? y_d[col_lo] : 0.0f;
                const float dy_hi = seq_hi_valid ? y_d[col_hi] : 0.0f;

                block_d[half * 4u + 0u] += dy_lo * scale_a * static_cast<float>(d0);
                block_d[half * 4u + 1u] += dy_hi * scale_a * static_cast<float>(d1);
                block_d[half * 4u + 2u] += dy_lo * scale_b * static_cast<float>(d2);
                block_d[half * 4u + 3u] += dy_hi * scale_b * static_cast<float>(d3);
                block_m[half * 4u + 0u] += dy_lo * min_a * sum_qy_lo;
                block_m[half * 4u + 1u] += dy_hi * min_a * sum_qy_hi;
                block_m[half * 4u + 2u] += dy_lo * min_b * sum_qy_lo;
                block_m[half * 4u + 3u] += dy_hi * min_b * sum_qy_hi;
            }
            __syncthreads();
        }

        const float d_a = x_d[warp_row_off + t_row_a];
        const float d_b = x_d[warp_row_off + t_row_b];
        const float dmin_a = x_dmin[warp_row_off + t_row_a];
        const float dmin_b = x_dmin[warp_row_off + t_row_b];
        acc[0] += d_a * block_d[0] - dmin_a * block_m[0];
        acc[1] += d_a * block_d[1] - dmin_a * block_m[1];
        acc[2] += d_b * block_d[2] - dmin_b * block_m[2];
        acc[3] += d_b * block_d[3] - dmin_b * block_m[3];
        acc[4] += d_a * block_d[4] - dmin_a * block_m[4];
        acc[5] += d_a * block_d[5] - dmin_a * block_m[5];
        acc[6] += d_b * block_d[6] - dmin_b * block_m[6];
        acc[7] += d_b * block_d[7] - dmin_b * block_m[7];
        __syncthreads();
    }

    if (row_a_valid && seq_a < seq_len) out[seq_a * rows + row_a] = acc[0];
    if (row_a_valid && seq_b < seq_len) out[seq_b * rows + row_a] = acc[1];
    if (row_b_valid && seq_a < seq_len) out[seq_a * rows + row_b] = acc[2];
    if (row_b_valid && seq_b < seq_len) out[seq_b * rows + row_b] = acc[3];
    if (row_a_valid && seq_c < seq_len) out[seq_c * rows + row_a] = acc[4];
    if (row_a_valid && seq_d < seq_len) out[seq_d * rows + row_a] = acc[5];
    if (row_b_valid && seq_c < seq_len) out[seq_c * rows + row_b] = acc[6];
    if (row_b_valid && seq_d < seq_len) out[seq_d * rows + row_b] = acc[7];
#endif
}

// cu228: 64-row x 64-sequence variant with a 512-thread CTA. At seq64 the
// b-side (activation) loads are two thirds of the loader traffic and are
// re-issued by every 32-row CTA along grid.x; doubling the row tile halves
// grid.x, so b-tile loads and chunk-sum folds amortize over twice the output
// rows (loader ops per output 0.375 -> 0.25). The 512-thread CTA keeps the
// per-thread accumulator layout of the 32x64 kernel (acc[8] + block accums),
// so registers stay in the same band and per-element accumulation order is
// identical to tile32 — outputs stay bitwise equal. __launch_bounds__(512, 2)
// keeps 2 CTAs/SM (1024 threads, same 66% occupancy as the 32x64 tile).
extern "C" __global__ void __launch_bounds__(512, 2) rnb_q4k_q8_1_matmul_mmq_tile64_seq64(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 750
    (void)out;
    (void)weights;
    (void)input_qs;
    (void)input_ds;
    (void)rows;
    (void)blocks_per_row;
    (void)seq_len;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 64u;
    const unsigned seq_base = blockIdx.y * 64u;
    const unsigned warp_row_off = (warp & 3u) * 16u;
    const unsigned warp_seq_off = (warp >> 2) * 16u;

    // Alternate sub-block slabs so the barrier before stage N compute also
    // retires stage N-1 before that parity is reused. This removes the second
    // block-wide barrier from every sub-block without changing math order.
    __shared__ signed char a_tile[2][64 * 36];
    __shared__ signed char b_tile[2][64 * 36];
    __shared__ unsigned char raw_stage[64 * 160];
    __shared__ float x_d[64];
    __shared__ float x_dmin[64];
    __shared__ unsigned char x_sc[2][64];
    __shared__ unsigned char x_mn[2][64];
    __shared__ float y_d[2][64];
    __shared__ float y_s[2][64];

    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;
    const unsigned t_col_a = warp_seq_off + ((lane & 3u) << 1);
    const unsigned t_col_b = t_col_a + 1u;
    const unsigned t_col_c = t_col_a + 8u;
    const unsigned t_col_d = t_col_b + 8u;
    const unsigned row_a = row_base + warp_row_off + t_row_a;
    const unsigned row_b = row_base + warp_row_off + t_row_b;
    const unsigned seq_a = seq_base + t_col_a;
    const unsigned seq_b = seq_base + t_col_b;
    const unsigned seq_c = seq_base + t_col_c;
    const unsigned seq_d = seq_base + t_col_d;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        float block_d[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float block_m[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // Stage the 64 raw 144B blocks once per block iteration (9 chunks
        // per row, 576 total across 512 threads).
        for (unsigned c = tid; c < 576u; c += 512u) {
            const unsigned s_row = c / 9u;
            const unsigned s_off = (c % 9u) * 16u;
            const unsigned g_row = row_base + s_row;
            if (g_row < rows) {
                __pipeline_memcpy_async(
                    raw_stage + s_row * 160u + s_off,
                    weights + g_row * row_bytes + block * 144u + s_off,
                    16);
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        unsigned packed_qs_pair = 0u;
        unsigned packed_scales0 = 0u;
        unsigned packed_scales1 = 0u;
        unsigned packed_scales2 = 0u;

        // Full unrolling resolves stage parity at compile time; a dynamic
        // shared-memory index costs more than the barrier this layout removes.
#pragma unroll
        for (unsigned sub = 0; sub < 8u; ++sub) {
            const unsigned stage = sub & 1u;
            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* a_dst = a_tile[stage] + load_row * 36u + load_off;

            if (global_row < rows) {
                const unsigned char* packed = raw_stage + load_row * 160u;
                if (sub == 0u && load_off == 0u) {
                    const unsigned raw_d = static_cast<unsigned>(packed[0])
                        | (static_cast<unsigned>(packed[1]) << 8);
                    const unsigned raw_dmin = static_cast<unsigned>(packed[2])
                        | (static_cast<unsigned>(packed[3]) << 8);
                    x_d[load_row] = __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                    x_dmin[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_dmin)));
                }
                if (load_off == 0u) {
                    if (sub == 0u) {
                        packed_scales0 = *reinterpret_cast<const unsigned*>(packed + 4u);
                        packed_scales1 = *reinterpret_cast<const unsigned*>(packed + 8u);
                        packed_scales2 = *reinterpret_cast<const unsigned*>(packed + 12u);
                    }
                    unsigned scale;
                    unsigned minimum;
                    if (sub < 4u) {
                        scale = (packed_scales0 >> (sub * 8u)) & 63u;
                        minimum = (packed_scales1 >> (sub * 8u)) & 63u;
                    } else {
                        const unsigned shift = (sub - 4u) * 8u;
                        const unsigned tail = (packed_scales2 >> shift) & 0xffu;
                        scale = (tail & 0x0fu)
                            | (((packed_scales0 >> shift) & 0xffu) >> 6 << 4);
                        minimum = (tail >> 4)
                            | (((packed_scales1 >> shift) & 0xffu) >> 6 << 4);
                    }
                    x_sc[stage][load_row] = static_cast<unsigned char>(scale);
                    x_mn[stage][load_row] = static_cast<unsigned char>(minimum);
                }
                const unsigned nibble_base = 16u + (sub >> 1) * 32u;
                if ((sub & 1u) == 0u) {
                    packed_qs_pair = *reinterpret_cast<const unsigned*>(
                        packed + nibble_base + load_off);
                }
                const unsigned unpacked = ((sub & 1u) == 0u)
                    ? (packed_qs_pair & 0x0f0f0f0fu)
                    : ((packed_qs_pair >> 4) & 0x0f0f0f0fu);
                *reinterpret_cast<unsigned*>(a_dst) = unpacked;
            } else {
                *reinterpret_cast<unsigned*>(a_dst) = 0u;
                if (sub == 0u && load_off == 0u) {
                    x_d[load_row] = 0.0f;
                    x_dmin[load_row] = 0.0f;
                }
                if (load_off == 0u) {
                    x_sc[stage][load_row] = 0u;
                    x_mn[stage][load_row] = 0u;
                }
            }

            // 64-seq b-slab in a single 512-thread pass; cu223 loader-side
            // chunk-sum fold per 8-lane group.
            const unsigned chunk = block * 8u + sub;
            {
                const unsigned load_seq = tid >> 3;
                const unsigned seq_off = (tid & 7u) * 4u;
                const unsigned global_seq = seq_base + load_seq;
                signed char* b_dst = b_tile[stage] + load_seq * 36u + seq_off;
                int b_word = 0;
                if (global_seq < seq_len) {
                    const signed char* b_src = input_qs +
                        (global_seq * blocks_per_row * 256u) + chunk * 32u + seq_off;
                    b_word = *reinterpret_cast<const int*>(b_src);
                    *reinterpret_cast<int*>(b_dst) = b_word;
                    if (seq_off == 0u) {
                        y_d[stage][load_seq] =
                            input_ds[global_seq * blocks_per_row * 8u + chunk];
                    }
                } else {
                    *reinterpret_cast<int*>(b_dst) = 0;
                    if (seq_off == 0u) {
                        y_d[stage][load_seq] = 0.0f;
                    }
                }
                int chunk_sum = __dp4a(0x01010101, b_word, 0);
                chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 4u, 8);
                chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 2u, 8);
                chunk_sum += __shfl_down_sync(0xffffffffu, chunk_sum, 1u, 8);
                if ((tid & 7u) == 0u) {
                    y_s[stage][load_seq] = static_cast<float>(chunk_sum);
                }
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int a0 = *reinterpret_cast<const int*>(
                &a_tile[stage][(warp_row_off + t_row_a) * 36u + a_col_lo]);
            const int a1 = *reinterpret_cast<const int*>(
                &a_tile[stage][(warp_row_off + t_row_b) * 36u + a_col_lo]);
            const int a2 = *reinterpret_cast<const int*>(
                &a_tile[stage][(warp_row_off + t_row_a) * 36u + a_col_hi]);
            const int a3 = *reinterpret_cast<const int*>(
                &a_tile[stage][(warp_row_off + t_row_b) * 36u + a_col_hi]);

            const float scale_a = static_cast<float>(x_sc[stage][warp_row_off + t_row_a]);
            const float scale_b = static_cast<float>(x_sc[stage][warp_row_off + t_row_b]);
            const float min_a = static_cast<float>(x_mn[stage][warp_row_off + t_row_a]);
            const float min_b = static_cast<float>(x_mn[stage][warp_row_off + t_row_b]);

            const unsigned b_seq0 = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
#pragma unroll
            for (unsigned half = 0; half < 2u; ++half) {
                const unsigned b_seq = b_seq0 + half * 8u;
                const int b0 =
                    *reinterpret_cast<const int*>(&b_tile[stage][b_seq * 36u + b_col_lo]);
                const int b1 =
                    *reinterpret_cast<const int*>(&b_tile[stage][b_seq * 36u + b_col_hi]);

                int d0 = 0;
                int d1 = 0;
                int d2 = 0;
                int d3 = 0;
                rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

                const unsigned col_lo = t_col_a + half * 8u;
                const unsigned col_hi = t_col_b + half * 8u;
                const bool seq_lo_valid = (seq_base + col_lo) < seq_len;
                const bool seq_hi_valid = (seq_base + col_hi) < seq_len;
                const float sum_qy_lo = y_s[stage][col_lo];
                const float sum_qy_hi = y_s[stage][col_hi];
                const float dy_lo = seq_lo_valid ? y_d[stage][col_lo] : 0.0f;
                const float dy_hi = seq_hi_valid ? y_d[stage][col_hi] : 0.0f;

                block_d[half * 4u + 0u] += dy_lo * scale_a * static_cast<float>(d0);
                block_d[half * 4u + 1u] += dy_hi * scale_a * static_cast<float>(d1);
                block_d[half * 4u + 2u] += dy_lo * scale_b * static_cast<float>(d2);
                block_d[half * 4u + 3u] += dy_hi * scale_b * static_cast<float>(d3);
                block_m[half * 4u + 0u] += dy_lo * min_a * sum_qy_lo;
                block_m[half * 4u + 1u] += dy_hi * min_a * sum_qy_hi;
                block_m[half * 4u + 2u] += dy_lo * min_b * sum_qy_lo;
                block_m[half * 4u + 3u] += dy_hi * min_b * sum_qy_hi;
            }
        }

        const float d_a = x_d[warp_row_off + t_row_a];
        const float d_b = x_d[warp_row_off + t_row_b];
        const float dmin_a = x_dmin[warp_row_off + t_row_a];
        const float dmin_b = x_dmin[warp_row_off + t_row_b];
        acc[0] += d_a * block_d[0] - dmin_a * block_m[0];
        acc[1] += d_a * block_d[1] - dmin_a * block_m[1];
        acc[2] += d_b * block_d[2] - dmin_b * block_m[2];
        acc[3] += d_b * block_d[3] - dmin_b * block_m[3];
        acc[4] += d_a * block_d[4] - dmin_a * block_m[4];
        acc[5] += d_a * block_d[5] - dmin_a * block_m[5];
        acc[6] += d_b * block_d[6] - dmin_b * block_m[6];
        acc[7] += d_b * block_d[7] - dmin_b * block_m[7];
        __syncthreads();
    }

    if (row_a_valid && seq_a < seq_len) out[seq_a * rows + row_a] = acc[0];
    if (row_a_valid && seq_b < seq_len) out[seq_b * rows + row_a] = acc[1];
    if (row_b_valid && seq_a < seq_len) out[seq_a * rows + row_b] = acc[2];
    if (row_b_valid && seq_b < seq_len) out[seq_b * rows + row_b] = acc[3];
    if (row_a_valid && seq_c < seq_len) out[seq_c * rows + row_a] = acc[4];
    if (row_a_valid && seq_d < seq_len) out[seq_d * rows + row_a] = acc[5];
    if (row_b_valid && seq_c < seq_len) out[seq_c * rows + row_b] = acc[6];
    if (row_b_valid && seq_d < seq_len) out[seq_d * rows + row_b] = acc[7];
#endif
}


// 128-row x 128-sequence Ampere MMQ tile. Eight warps each own one
// 16-row slab and sweep sixteen 8-column MMA fragments. Full-block Q4
// unpacking is reused across the complete sequence tile.
extern "C" __global__ void __launch_bounds__(256, 2)
rnb_q4k_q8_1_matmul_mmq_tile128_seq128(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ input_sums,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
    (void)out;
    (void)weights;
    (void)input_qs;
    (void)input_ds;
    (void)input_sums;
    (void)rows;
    (void)blocks_per_row;
    (void)seq_len;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned warp_row_off = warp * 16u;
    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;

    __shared__ int b_tile[8][16][8][8];
    __shared__ float2 x_dm_sc[8][128];
    __shared__ float y_d[8][128];
    __shared__ float y_s[8][128];
    const unsigned row_tiles = (rows + 127u) / 128u;
    const unsigned seq_tiles = (seq_len + 127u) / 128u;
    const unsigned tile_count = row_tiles * seq_tiles;
    for (unsigned tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const unsigned row_tile = tile / seq_tiles;
        const unsigned seq_tile = tile - row_tile * seq_tiles;
        const unsigned row_base = row_tile * 128u;
        const unsigned seq_base = seq_tile * 128u;

    const unsigned row_a = row_base + warp_row_off + t_row_a;
    const unsigned row_b = row_base + warp_row_off + t_row_b;
    float acc[64];
#pragma unroll
    for (unsigned i = 0; i < 64u; ++i) {
        acc[i] = 0.0f;
    }
    const unsigned row_bytes = blocks_per_row * 144u;
    for (unsigned block = 0; block < blocks_per_row; ++block) {
        if (tid < 128u) {
            const unsigned global_row = row_base + tid;
            if (global_row < rows) {
                const unsigned char* packed =
                    weights + global_row * row_bytes + block * 144u;
                const unsigned raw_d = static_cast<unsigned>(packed[0])
                    | (static_cast<unsigned>(packed[1]) << 8);
                const unsigned raw_dmin = static_cast<unsigned>(packed[2])
                    | (static_cast<unsigned>(packed[3]) << 8);
                const float d =
                    __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                const float dmin =
                    __half2float(__ushort_as_half(static_cast<unsigned short>(raw_dmin)));
#pragma unroll
                for (unsigned sub = 0; sub < 8u; ++sub) {
                    unsigned scale;
                    unsigned min_scale;
                    if (sub < 4u) {
                        scale = packed[4u + sub] & 63u;
                        min_scale = packed[8u + sub] & 63u;
                    } else {
                        scale =
                            (packed[8u + sub] & 0x0fu) | ((packed[sub] >> 6) << 4);
                        min_scale =
                            (packed[8u + sub] >> 4) | ((packed[4u + sub] >> 6) << 4);
                    }
                    x_dm_sc[sub][tid] = make_float2(
                        d * static_cast<float>(scale),
                        -dmin * static_cast<float>(min_scale));
                }
            } else {
#pragma unroll
                for (unsigned sub = 0; sub < 8u; ++sub) {
                    x_dm_sc[sub][tid] = make_float2(0.0f, 0.0f);
                }
            }
        }

        for (unsigned item = tid; item < 8192u; item += 256u) {
            const unsigned slot = item >> 10;
            const unsigned local = item & 1023u;
            const unsigned load_seq = local >> 3;
            const unsigned word = local & 7u;
            const unsigned global_seq = seq_base + load_seq;
            const unsigned chunk = block * 8u + slot;
            int b_word = 0;
            if (global_seq < seq_len) {
                const signed char* b_src = input_qs
                    + global_seq * blocks_per_row * 256u
                    + chunk * 32u + word * 4u;
                b_word = *reinterpret_cast<const int*>(b_src);
                if (word == 0u) {
                    const unsigned metadata =
                        global_seq * blocks_per_row * 8u + chunk;
                    y_d[slot][load_seq] = input_ds[metadata];
                    y_s[slot][load_seq] =
                        input_ds[metadata] * input_sums[metadata];
                }
            } else if (word == 0u) {
                y_d[slot][load_seq] = 0.0f;
                y_s[slot][load_seq] = 0.0f;
            }
            b_tile[slot][load_seq >> 3][word][load_seq & 7u] = b_word;
        }
        __syncthreads();

        const unsigned a_col_lo = (lane & 3u) * 4u;
        const unsigned a_col_hi = a_col_lo + 16u;
#pragma unroll
        for (unsigned pair = 0; pair < 4u; ++pair) {
            unsigned packed_a0 = 0u;
            unsigned packed_a1 = 0u;
            unsigned packed_a2 = 0u;
            unsigned packed_a3 = 0u;
            if (row_a < rows) {
                const unsigned char* packed =
                    weights + row_a * row_bytes + block * 144u + 16u + pair * 32u;
                packed_a0 = *reinterpret_cast<const unsigned*>(packed + a_col_lo);
                packed_a2 = *reinterpret_cast<const unsigned*>(packed + a_col_hi);
            }
            if (row_b < rows) {
                const unsigned char* packed =
                    weights + row_b * row_bytes + block * 144u + 16u + pair * 32u;
                packed_a1 = *reinterpret_cast<const unsigned*>(packed + a_col_lo);
                packed_a3 = *reinterpret_cast<const unsigned*>(packed + a_col_hi);
            }

#pragma unroll
            for (unsigned parity = 0; parity < 2u; ++parity) {
                const unsigned sub = pair * 2u + parity;
                const unsigned nibble_shift = parity * 4u;
                const int a0 = static_cast<int>(
                    (packed_a0 >> nibble_shift) & 0x0f0f0f0fu);
                const int a1 = static_cast<int>(
                    (packed_a1 >> nibble_shift) & 0x0f0f0f0fu);
                const int a2 = static_cast<int>(
                    (packed_a2 >> nibble_shift) & 0x0f0f0f0fu);
                const int a3 = static_cast<int>(
                    (packed_a3 >> nibble_shift) & 0x0f0f0f0fu);
                const float2 dm_a = x_dm_sc[sub][warp_row_off + t_row_a];
                const float2 dm_b = x_dm_sc[sub][warp_row_off + t_row_b];

#pragma unroll
                for (unsigned frag = 0; frag < 16u; ++frag) {
                    const unsigned seq_in_group = lane >> 2;
                    const unsigned word_lo = lane & 3u;
                    const int b0 = b_tile[sub][frag][word_lo][seq_in_group];
                    const int b1 = b_tile[sub][frag][word_lo + 4u][seq_in_group];
                    int dot0 = 0;
                    int dot1 = 0;
                    int dot2 = 0;
                    int dot3 = 0;
                    rnb_mma_m16n8k32_s8(
                        dot0, dot1, dot2, dot3,
                        a0, a1, a2, a3,
                        b0, b1,
                        0, 0, 0, 0);

                    const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
                    const unsigned col_b = col_a + 1u;
                    const float dy_a = y_d[sub][col_a];
                    const float dy_b = y_d[sub][col_b];
                    const float sum_a = y_s[sub][col_a];
                    const float sum_b = y_s[sub][col_b];
                    acc[frag * 4u + 0u] +=
                        dm_a.x * dy_a * static_cast<float>(dot0);
                    acc[frag * 4u + 0u] += dm_a.y * sum_a;
                    acc[frag * 4u + 1u] +=
                        dm_a.x * dy_b * static_cast<float>(dot1);
                    acc[frag * 4u + 1u] += dm_a.y * sum_b;
                    acc[frag * 4u + 2u] +=
                        dm_b.x * dy_a * static_cast<float>(dot2);
                    acc[frag * 4u + 2u] += dm_b.y * sum_a;
                    acc[frag * 4u + 3u] +=
                        dm_b.x * dy_b * static_cast<float>(dot3);
                    acc[frag * 4u + 3u] += dm_b.y * sum_b;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned frag = 0; frag < 16u; ++frag) {
        const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
        const unsigned col_b = col_a + 1u;
        const unsigned seq_a = seq_base + col_a;
        const unsigned seq_b = seq_base + col_b;
        if (row_a < rows && seq_a < seq_len) {
            out[seq_a * rows + row_a] = acc[frag * 4u + 0u];
        }
        if (row_a < rows && seq_b < seq_len) {
            out[seq_b * rows + row_a] = acc[frag * 4u + 1u];
        }
        if (row_b < rows && seq_a < seq_len) {
            out[seq_a * rows + row_b] = acc[frag * 4u + 2u];
        }
        if (row_b < rows && seq_b < seq_len) {
            out[seq_b * rows + row_b] = acc[frag * 4u + 3u];
        }
    }
    }
#endif
}

// Q8_1 activation transpose for the llama-style MMQ load contract. The
// source is token-major [seq][chunk][32]; the destination is
// chunk-major [chunk][seq][32], so a CTA can load one K slice for 64
// sequences with fully coalesced transactions. float2 stores the original
// f32 scale and the exact scale*integer-sum correction used by the current
// Q4_K arithmetic.
extern "C" __global__ void rnb_q8_1_transpose_chunks_by_seq(
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    signed char* __restrict__ output_qs,
    float2* __restrict__ output_meta,
    unsigned chunks_per_seq,
    unsigned seq_len) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned chunk = blockIdx.x * 8u + (threadIdx.x >> 5);
    const unsigned seq = blockIdx.y;
    if (chunk >= chunks_per_seq || seq >= seq_len) {
        return;
    }

    const unsigned source_chunk = seq * chunks_per_seq + chunk;
    const signed char value = input_qs[source_chunk * 32u + lane];
    const unsigned destination_chunk = chunk * seq_len + seq;
    output_qs[destination_chunk * 32u + lane] = value;

    int sum = static_cast<int>(value);
#pragma unroll
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
        const float d = input_ds[source_chunk];
        output_meta[destination_chunk] = make_float2(d, d * static_cast<float>(sum));
    }
}

// 128-row x 64-sequence Q4_K MMQ using the chunk-major Q8_1 layout above.
// Q4 words for all 128 rows are unpacked into shared memory with contiguous
// 128-byte global transactions. Q8 values are loaded in two 128-value halves,
// matching llama.cpp's load-tile structure while retaining runNburn's f32
// metadata and per-output accumulation order.
extern "C" __global__ void __launch_bounds__(256, 2)
rnb_q4k_q8_1_matmul_mmq_transposed_seq128(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float2* __restrict__ input_meta,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
    (void)out;
    (void)weights;
    (void)input_qs;
    (void)input_meta;
    (void)rows;
    (void)blocks_per_row;
    (void)seq_len;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned warp_row_off = warp * 16u;
    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;

    __shared__ unsigned x_packed[128][33];
    __shared__ float2 x_dm[8][128];
    __shared__ int y_qs[4][16][8][8];
    __shared__ float2 y_meta[4][128];

    const unsigned row_tiles = (rows + 127u) / 128u;
    const unsigned seq_tiles = (seq_len + 127u) / 128u;
    const unsigned tile_count = row_tiles * seq_tiles;
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const unsigned row_tile = tile / seq_tiles;
        const unsigned seq_tile = tile - row_tile * seq_tiles;
        const unsigned row_base = row_tile * 128u;
        const unsigned seq_base = seq_tile * 128u;
        const unsigned row_a = row_base + warp_row_off + t_row_a;
        const unsigned row_b = row_base + warp_row_off + t_row_b;
        float acc[64];
#pragma unroll
        for (unsigned i = 0; i < 64u; ++i) {
            acc[i] = 0.0f;
        }

        for (unsigned block = 0; block < blocks_per_row; ++block) {
#pragma unroll
            for (unsigned pass = 0; pass < 16u; ++pass) {
                const unsigned local_row = warp + pass * 8u;
                const unsigned global_row = row_base + local_row;
                unsigned packed_qs = 0u;
                if (global_row < rows) {
                    const unsigned char* packed =
                        weights + global_row * row_bytes + block * 144u;
                    packed_qs = *reinterpret_cast<const unsigned*>(
                        packed + 16u + lane * 4u);
                }
                x_packed[local_row][lane] = packed_qs;
            }

            if (tid < 128u) {
                const unsigned global_row = row_base + tid;
                if (global_row < rows) {
                    const unsigned char* packed =
                        weights + global_row * row_bytes + block * 144u;
                    const uint4 meta = *reinterpret_cast<const uint4*>(packed);
                    const float d = __half2float(
                        __ushort_as_half(static_cast<unsigned short>(meta.x & 0xffffu)));
                    const float dmin = __half2float(
                        __ushort_as_half(static_cast<unsigned short>(meta.x >> 16)));
#pragma unroll
                    for (unsigned sub = 0; sub < 8u; ++sub) {
                        unsigned scale;
                        unsigned min_scale;
                        if (sub < 4u) {
                            const unsigned shift = sub * 8u;
                            scale = (meta.y >> shift) & 63u;
                            min_scale = (meta.z >> shift) & 63u;
                        } else {
                            const unsigned shift = (sub - 4u) * 8u;
                            const unsigned lo = (meta.y >> shift) & 0xffu;
                            const unsigned hi = (meta.z >> shift) & 0xffu;
                            const unsigned mixed = (meta.w >> shift) & 0xffu;
                            scale = (mixed & 0x0fu) | ((lo >> 6) << 4);
                            min_scale = (mixed >> 4) | ((hi >> 6) << 4);
                        }
                        x_dm[sub][tid] = make_float2(
                            d * static_cast<float>(scale),
                            -dmin * static_cast<float>(min_scale));
                    }
                } else {
#pragma unroll
                    for (unsigned sub = 0; sub < 8u; ++sub) {
                        x_dm[sub][tid] = make_float2(0.0f, 0.0f);
                    }
                }
            }

#pragma unroll
            for (unsigned half = 0; half < 2u; ++half) {

                // Four 32-value sub-blocks x 128 sequences.
                for (unsigned item = tid; item < 4096u; item += 256u) {
                    const unsigned local_sub = item >> 10;
                    const unsigned local = item & 1023u;
                    const unsigned local_seq = local >> 3;
                    const unsigned word = local & 7u;
                    const unsigned global_seq = seq_base + local_seq;
                    int qword = 0;
                    if (global_seq < seq_len) {
                        const unsigned chunk = block * 8u + half * 4u + local_sub;
                        qword = *reinterpret_cast<const int*>(
                            input_qs + (chunk * seq_len + global_seq) * 32u + word * 4u);
                    }
                    y_qs[local_sub][local_seq >> 3][word][local_seq & 7u] = qword;
                }
                for (unsigned item = tid; item < 512u; item += 256u) {
                    const unsigned local_sub = item >> 7;
                    const unsigned local_seq = item & 127u;
                    const unsigned global_seq = seq_base + local_seq;
                    float2 meta = make_float2(0.0f, 0.0f);
                    if (global_seq < seq_len) {
                        const unsigned chunk = block * 8u + half * 4u + local_sub;
                        meta = input_meta[chunk * seq_len + global_seq];
                    }
                    y_meta[local_sub][local_seq] = meta;
                }
                __syncthreads();
                const unsigned word_lo = lane & 3u;


#pragma unroll
                for (unsigned local_sub = 0; local_sub < 4u; ++local_sub) {
                    const unsigned sub = half * 4u + local_sub;
                    const unsigned pair = sub >> 1;
                    const unsigned word_base = pair * 8u + word_lo;
                    const unsigned shift = (sub & 1u) * 4u;
                    const int a0 = static_cast<int>(
                        (x_packed[warp_row_off + t_row_a][word_base] >> shift) &
                        0x0f0f0f0fu);
                    const int a1 = static_cast<int>(
                        (x_packed[warp_row_off + t_row_b][word_base] >> shift) &
                        0x0f0f0f0fu);
                    const int a2 = static_cast<int>(
                        (x_packed[warp_row_off + t_row_a][word_base + 4u] >> shift) &
                        0x0f0f0f0fu);
                    const int a3 = static_cast<int>(
                        (x_packed[warp_row_off + t_row_b][word_base + 4u] >> shift) &
                        0x0f0f0f0fu);
                    const float2 dm_a = x_dm[sub][warp_row_off + t_row_a];
                    const float2 dm_b = x_dm[sub][warp_row_off + t_row_b];

#pragma unroll
                    for (unsigned frag = 0; frag < 16u; ++frag) {
                        const unsigned seq_in_group = lane >> 2;
                        const int b0 = y_qs[local_sub][frag][word_lo][seq_in_group];
                        const int b1 =
                            y_qs[local_sub][frag][word_lo + 4u][seq_in_group];
                        int dot0 = 0;
                        int dot1 = 0;
                        int dot2 = 0;
                        int dot3 = 0;
                        rnb_mma_m16n8k32_s8(
                            dot0, dot1, dot2, dot3,
                            a0, a1, a2, a3, b0, b1,
                            0, 0, 0, 0);

                        const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
                        const unsigned col_b = col_a + 1u;
                        const float2 meta_a = y_meta[local_sub][col_a];
                        const float2 meta_b = y_meta[local_sub][col_b];
                        acc[frag * 4u + 0u] +=
                            dm_a.x * meta_a.x * static_cast<float>(dot0);
                        acc[frag * 4u + 0u] += dm_a.y * meta_a.y;
                        acc[frag * 4u + 1u] +=
                            dm_a.x * meta_b.x * static_cast<float>(dot1);
                        acc[frag * 4u + 1u] += dm_a.y * meta_b.y;
                        acc[frag * 4u + 2u] +=
                            dm_b.x * meta_a.x * static_cast<float>(dot2);
                        acc[frag * 4u + 2u] += dm_b.y * meta_a.y;
                        acc[frag * 4u + 3u] +=
                            dm_b.x * meta_b.x * static_cast<float>(dot3);
                        acc[frag * 4u + 3u] += dm_b.y * meta_b.y;
                    }
                }
                __syncthreads();
            }
        }

#pragma unroll
        for (unsigned frag = 0; frag < 16u; ++frag) {
            const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
            const unsigned col_b = col_a + 1u;
            const unsigned seq_a = seq_base + col_a;
            const unsigned seq_b = seq_base + col_b;
            if (row_a < rows && seq_a < seq_len) {
                out[seq_a * rows + row_a] = acc[frag * 4u + 0u];
            }
            if (row_a < rows && seq_b < seq_len) {
                out[seq_b * rows + row_a] = acc[frag * 4u + 1u];
            }
            if (row_b < rows && seq_a < seq_len) {
                out[seq_a * rows + row_b] = acc[frag * 4u + 2u];
            }
            if (row_b < rows && seq_b < seq_len) {
                out[seq_b * rows + row_b] = acc[frag * 4u + 3u];
            }
        }
    }
#endif
}

// Adapted from llama.cpp/ggml-cuda's current Ampere Q4_K MMQ configuration:
// non-fallback, no-MUL_MAT_ID, I=128/J=128, eight warps and occupancy one.
// RnbBlockQ8_1Mmq keeps the upstream 128-value transposed+padded activation
// contract locally; the older runNburn transposed MMQ kernel above remains the
// diagnostic fallback.
extern "C" __global__ void __launch_bounds__(256, 1)
rnb_q4k_q8_1_matmul_mmq_llama_ampere_j128(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const RnbBlockQ8_1Mmq* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
    (void)out; (void)weights; (void)input; (void)rows; (void)blocks_per_row; (void)seq_len;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned warp_row_off = warp * 16u;
    const unsigned row_tiles = (rows + 127u) / 128u;
    const unsigned seq_tiles = (seq_len + 127u) / 128u;
    const unsigned tile_count = row_tiles * seq_tiles;
    const unsigned row_bytes = blocks_per_row * 144u;

    __shared__ unsigned x_packed[128][33];
    __shared__ float2 x_dm[8][128];
    __shared__ int y_qs[4][16][8][8];
    __shared__ float2 y_meta[4][128];

    for (unsigned tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const unsigned row_tile = tile / seq_tiles;
        const unsigned seq_tile = tile - row_tile * seq_tiles;
        const unsigned row_base = row_tile * 128u;
        const unsigned seq_base = seq_tile * 128u;
        const unsigned row_a = row_base + warp_row_off + (lane >> 2);
        const unsigned row_b = row_a + 8u;
        float acc[64];
#pragma unroll
        for (unsigned i = 0; i < 64u; ++i) acc[i] = 0.0f;

        for (unsigned block = 0; block < blocks_per_row; ++block) {
#pragma unroll
            for (unsigned pass = 0; pass < 16u; ++pass) {
                const unsigned local_row = warp + pass * 8u;
                const unsigned global_row = row_base + local_row;
                unsigned packed_qs = 0u;
                if (global_row < rows) {
                    packed_qs = *reinterpret_cast<const unsigned*>(
                        weights + global_row * row_bytes + block * 144u + 16u + lane * 4u);
                }
                x_packed[local_row][lane] = packed_qs;
            }
            if (tid < 128u) {
                const unsigned global_row = row_base + tid;
                if (global_row < rows) {
                    const unsigned char* packed = weights + global_row * row_bytes + block * 144u;
                    const uint4 meta = *reinterpret_cast<const uint4*>(packed);
                    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(meta.x)));
                    const float dmin = __half2float(
                        __ushort_as_half(static_cast<unsigned short>(meta.x >> 16)));
#pragma unroll
                    for (unsigned sub = 0; sub < 8u; ++sub) {
                        unsigned scale;
                        unsigned minimum;
                        if (sub < 4u) {
                            scale = (meta.y >> (sub * 8u)) & 63u;
                            minimum = (meta.z >> (sub * 8u)) & 63u;
                        } else {
                            const unsigned shift = (sub - 4u) * 8u;
                            const unsigned lo = (meta.y >> shift) & 0xffu;
                            const unsigned hi = (meta.z >> shift) & 0xffu;
                            const unsigned mixed = (meta.w >> shift) & 0xffu;
                            scale = (mixed & 0x0fu) | ((lo >> 6) << 4);
                            minimum = (mixed >> 4) | ((hi >> 6) << 4);
                        }
                        x_dm[sub][tid] = make_float2(
                            d * static_cast<float>(scale), -dmin * static_cast<float>(minimum));
                    }
                } else {
#pragma unroll
                    for (unsigned sub = 0; sub < 8u; ++sub) x_dm[sub][tid] = make_float2(0.0f, 0.0f);
                }
            }

#pragma unroll
            for (unsigned half = 0; half < 2u; ++half) {
                for (unsigned item = tid; item < 4096u; item += 256u) {
                    const unsigned local_sub = item >> 10;
                    const unsigned local = item & 1023u;
                    const unsigned local_seq = local >> 3;
                    const unsigned word = local & 7u;
                    const unsigned global_seq = seq_base + local_seq;
                    int qword = 0;
                    if (global_seq < seq_len) {
                        const RnbBlockQ8_1Mmq* source =
                            input + (unsigned long long)(block * 2u + half) * seq_len + global_seq;
                        qword = reinterpret_cast<const int*>(source->qs + local_sub * 32u)[word];
                    }
                    y_qs[local_sub][local_seq >> 3][word][local_seq & 7u] = qword;
                }
                for (unsigned item = tid; item < 512u; item += 256u) {
                    const unsigned local_sub = item >> 7;
                    const unsigned local_seq = item & 127u;
                    const unsigned global_seq = seq_base + local_seq;
                    float2 meta = make_float2(0.0f, 0.0f);
                    if (global_seq < seq_len) {
                        const RnbBlockQ8_1Mmq* source =
                            input + (unsigned long long)(block * 2u + half) * seq_len + global_seq;
                        meta = __half22float2(source->ds4[local_sub]);
                    }
                    y_meta[local_sub][local_seq] = meta;
                }
                __syncthreads();

                const unsigned word_lo = lane & 3u;
#pragma unroll
                for (unsigned local_sub = 0; local_sub < 4u; ++local_sub) {
                    const unsigned sub = half * 4u + local_sub;
                    const unsigned pair = sub >> 1;
                    const unsigned word_base = pair * 8u + word_lo;
                    const unsigned shift = (sub & 1u) * 4u;
                    const int a0 = static_cast<int>(
                        (x_packed[warp_row_off + (lane >> 2)][word_base] >> shift) & 0x0f0f0f0fu);
                    const int a1 = static_cast<int>(
                        (x_packed[warp_row_off + (lane >> 2) + 8u][word_base] >> shift) & 0x0f0f0f0fu);
                    const int a2 = static_cast<int>(
                        (x_packed[warp_row_off + (lane >> 2)][word_base + 4u] >> shift) & 0x0f0f0f0fu);
                    const int a3 = static_cast<int>(
                        (x_packed[warp_row_off + (lane >> 2) + 8u][word_base + 4u] >> shift) & 0x0f0f0f0fu);
                    const float2 dm_a = x_dm[sub][warp_row_off + (lane >> 2)];
                    const float2 dm_b = x_dm[sub][warp_row_off + (lane >> 2) + 8u];
#pragma unroll
                    for (unsigned frag = 0; frag < 16u; ++frag) {
                        const unsigned seq_in_group = lane >> 2;
                        const int b0 = y_qs[local_sub][frag][word_lo][seq_in_group];
                        const int b1 = y_qs[local_sub][frag][word_lo + 4u][seq_in_group];
                        int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
                        rnb_mma_m16n8k32_s8(dot0, dot1, dot2, dot3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);
                        const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
                        const unsigned col_b = col_a + 1u;
                        const float2 meta_a = y_meta[local_sub][col_a];
                        const float2 meta_b = y_meta[local_sub][col_b];
                        acc[frag * 4u] += dm_a.x * meta_a.x * static_cast<float>(dot0) + dm_a.y * meta_a.y;
                        acc[frag * 4u + 1u] += dm_a.x * meta_b.x * static_cast<float>(dot1) + dm_a.y * meta_b.y;
                        acc[frag * 4u + 2u] += dm_b.x * meta_a.x * static_cast<float>(dot2) + dm_b.y * meta_a.y;
                        acc[frag * 4u + 3u] += dm_b.x * meta_b.x * static_cast<float>(dot3) + dm_b.y * meta_b.y;
                    }
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (unsigned frag = 0; frag < 16u; ++frag) {
            const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
            const unsigned col_b = col_a + 1u;
            if (row_a < rows && seq_base + col_a < seq_len) out[(seq_base + col_a) * rows + row_a] = acc[frag * 4u];
            if (row_a < rows && seq_base + col_b < seq_len) out[(seq_base + col_b) * rows + row_a] = acc[frag * 4u + 1u];
            if (row_b < rows && seq_base + col_a < seq_len) out[(seq_base + col_a) * rows + row_b] = acc[frag * 4u + 2u];
            if (row_b < rows && seq_base + col_b < seq_len) out[(seq_base + col_b) * rows + row_b] = acc[frag * 4u + 3u];
        }
    }
#endif
}
