#include <cuda_pipeline.h>

// Q5_K x Q8_1 tiled matrix multiply for Ampere-class integer tensor cores.
//
// cu222: Q5_K was the last prefill-band quant without an MMQ generation —
// dev-input batches fell back to the q8dot wide GEMV family, which re-walks
// activations per row group instead of tiling (27B ssm_out: 1.37s of a 6.7s
// 1139-token prefill kernel budget). The 176-byte Q5_K block shares the
// Q4_K super-block scale/min packing, so this kernel is the Q4_K MMQ tile
// with the 5th bit OR-ed in from qh during the a-tile unpack:
//   q = (qs nibble) | (((qh[col] >> sub) & 1) << 4),  w = d*sc*q - dmin*mn
// Layout: d@0 dmin@2 scales@4(12B) qh@16(32B) qs@48(128B).
extern "C" __global__ void rnb_q5k_q8_1_matmul_mmq_tile32(
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
    const unsigned row_bytes = blocks_per_row * 176u;

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
                const unsigned char* packed = weights + global_row * row_bytes + block * 176u;
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
                const unsigned nibble_base = 48u + (sub >> 1) * 32u;
                const unsigned packed_qs = *reinterpret_cast<const unsigned*>(packed + nibble_base + load_off);
                unsigned unpacked = ((sub & 1u) == 0u) ? (packed_qs & 0x0f0f0f0fu)
                                                        : ((packed_qs >> 4) & 0x0f0f0f0fu);
                const unsigned qh_word = *reinterpret_cast<const unsigned*>(packed + 16u + load_off);
                unpacked |= ((qh_word >> sub) & 0x01010101u) << 4;
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

            // cu223: b-tile 로더가 4B word 를 dp4a + 8-lane shuffle 로 접어
            // chunk 합을 y_s 에 기록 — min-term dp4a 루프 제거 (bitwise 동일).
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

// cu226: 32-row x 64-sequence variant — same amortization as the Q4_K seq64
// tile (grid.y halved, a-tile load/unpack/scale traffic halved per output),
// with the Q5_K fifth-bit qh injection kept in the a-tile unpack. Per-element
// accumulation order matches tile32, so outputs stay bitwise equal.
// __launch_bounds__(256, 4) caps registers for 4 CTAs/SM (the Q4_K variant
// measured 72 regs free-allocation -> 50% occupancy and idle mem pipes).
extern "C" __global__ void __launch_bounds__(256, 4) rnb_q5k_q8_1_matmul_mmq_tile32_seq64(
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
    // cu227: raw Q5_K block staging — 176B rows padded to 224B (16B-aligned
    // cp.async rows; the 56-word stride shifts rows by 24 banks, keeping the
    // 8-thread qs/qh reads conflict-free across the 4 rows of a warp).
    __shared__ unsigned char raw_stage[32 * 224];
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
    const unsigned row_bytes = blocks_per_row * 176u;

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        float block_d[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float block_m[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // cu227: stage the 32 raw 176B blocks once per block iteration with
        // 16B async copies (11 chunks/row, 352 total); every sub-iteration
        // unpacks qs/qh/scales from shared. Arithmetic unchanged — bitwise.
        for (unsigned c = tid; c < 352u; c += 256u) {
            const unsigned s_row = c / 11u;
            const unsigned s_off = (c % 11u) * 16u;
            const unsigned g_row = row_base + s_row;
            if (g_row < rows) {
                __pipeline_memcpy_async(
                    raw_stage + s_row * 224u + s_off,
                    weights + g_row * row_bytes + block * 176u + s_off,
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
                const unsigned char* packed = raw_stage + load_row * 224u;
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
                const unsigned nibble_base = 48u + (sub >> 1) * 32u;
                const unsigned packed_qs = *reinterpret_cast<const unsigned*>(packed + nibble_base + load_off);
                unsigned unpacked = ((sub & 1u) == 0u) ? (packed_qs & 0x0f0f0f0fu)
                                                        : ((packed_qs >> 4) & 0x0f0f0f0fu);
                const unsigned qh_word = *reinterpret_cast<const unsigned*>(packed + 16u + load_off);
                unpacked |= ((qh_word >> sub) & 0x01010101u) << 4;
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

            // 64-seq b-slab: two 32-seq passes, cu223 loader-side chunk-sum
            // fold per pass.
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

// cu228: 64-row x 64-sequence variant with a 512-thread CTA — same b-side
// amortization as the Q4_K 64x64 tile (grid.x halved), with the Q5_K qh
// fifth-bit injection and cp.async staging kept (64 rows x 11 chunks).
// Per-element accumulation order matches tile32 — outputs stay bitwise equal.
extern "C" __global__ void __launch_bounds__(512, 2) rnb_q5k_q8_1_matmul_mmq_tile64_seq64(
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

    __shared__ signed char a_tile[64 * 36];
    __shared__ signed char b_tile[64 * 36];
    __shared__ unsigned char raw_stage[64 * 224];
    __shared__ float x_d[64];
    __shared__ float x_dmin[64];
    __shared__ unsigned char x_sc[64];
    __shared__ unsigned char x_mn[64];
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
    const unsigned row_bytes = blocks_per_row * 176u;

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        float block_d[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float block_m[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (unsigned c = tid; c < 704u; c += 512u) {
            const unsigned s_row = c / 11u;
            const unsigned s_off = (c % 11u) * 16u;
            const unsigned g_row = row_base + s_row;
            if (g_row < rows) {
                __pipeline_memcpy_async(
                    raw_stage + s_row * 224u + s_off,
                    weights + g_row * row_bytes + block * 176u + s_off,
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
                const unsigned char* packed = raw_stage + load_row * 224u;
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
                const unsigned nibble_base = 48u + (sub >> 1) * 32u;
                const unsigned packed_qs = *reinterpret_cast<const unsigned*>(packed + nibble_base + load_off);
                unsigned unpacked = ((sub & 1u) == 0u) ? (packed_qs & 0x0f0f0f0fu)
                                                        : ((packed_qs >> 4) & 0x0f0f0f0fu);
                const unsigned qh_word = *reinterpret_cast<const unsigned*>(packed + 16u + load_off);
                unpacked |= ((qh_word >> sub) & 0x01010101u) << 4;
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

            const unsigned chunk = block * 8u + sub;
            {
                const unsigned load_seq = tid >> 3;
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
