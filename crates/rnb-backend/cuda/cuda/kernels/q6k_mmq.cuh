// Q6_K x Q8_1 tiled matrix multiply for Ampere-class integer tensor cores.
//
// One 8-warp CTA computes a 32-row x 32-sequence output tile directly from
// the canonical 210-byte Q6_K block. The two 16-value Q6 sub-scales inside
// each Q8_1 chunk are evaluated by separate low/high MMA instructions.

extern "C" __global__ void rnb_q6k_q8_1_matmul_mmq_tile32(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
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
    __shared__ float weight_d[32];
    __shared__ signed char weight_scale_lo[32];
    __shared__ signed char weight_scale_hi[32];
    __shared__ float input_d[32];

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
    const unsigned row_bytes = blocks_per_row * 210u;

    for (unsigned block_index = 0; block_index < blocks_per_row; ++block_index) {
        float block_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (unsigned sub = 0; sub < 8u; ++sub) {
            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* a_dst = a_tile + load_row * 36u + load_off;

            if (global_row < rows) {
                const unsigned char* packed =
                    weights + global_row * row_bytes + block_index * 210u;
                if (sub == 0u && load_off == 0u) {
                    const unsigned raw_d = static_cast<unsigned>(packed[208])
                        | (static_cast<unsigned>(packed[209]) << 8);
                    weight_d[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                }
                if (load_off == 0u) {
                    weight_scale_lo[load_row] =
                        static_cast<signed char>(packed[192u + sub * 2u]);
                    weight_scale_hi[load_row] =
                        static_cast<signed char>(packed[193u + sub * 2u]);
                }
#pragma unroll
                for (unsigned i = 0; i < 4u; ++i) {
                    const unsigned elem = sub * 32u + load_off + i;
                    const unsigned half = elem >> 7;
                    const unsigned rem = elem & 127u;
                    const unsigned column = rem & 31u;
                    const unsigned ql_base = half * 64u;
                    const unsigned qh_base = 128u + half * 32u;
                    const unsigned qh = packed[qh_base + column];
                    unsigned q;
                    if (rem < 32u) {
                        q = (packed[ql_base + column] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    } else if (rem < 64u) {
                        q = (packed[ql_base + column + 32u] & 0x0fu)
                            | (((qh >> 2) & 3u) << 4);
                    } else if (rem < 96u) {
                        q = (packed[ql_base + column] >> 4) | (((qh >> 4) & 3u) << 4);
                    } else {
                        q = (packed[ql_base + column + 32u] >> 4)
                            | (((qh >> 6) & 3u) << 4);
                    }
                    a_dst[i] = static_cast<signed char>(static_cast<int>(q) - 32);
                }
            } else {
                *reinterpret_cast<unsigned*>(a_dst) = 0u;
                if (sub == 0u && load_off == 0u) {
                    weight_d[load_row] = 0.0f;
                }
                if (load_off == 0u) {
                    weight_scale_lo[load_row] = 0;
                    weight_scale_hi[load_row] = 0;
                }
            }

            const unsigned load_seq = tid >> 3;
            const unsigned seq_off = (tid & 7u) * 4u;
            const unsigned global_seq = seq_base + load_seq;
            signed char* b_dst = b_tile + load_seq * 36u + seq_off;
            if (global_seq < seq_len) {
                const unsigned chunk = block_index * 8u + sub;
                const signed char* b_src = input_qs
                    + global_seq * blocks_per_row * 256u + chunk * 32u + seq_off;
                *reinterpret_cast<unsigned*>(b_dst) = *reinterpret_cast<const unsigned*>(b_src);
                if (seq_off == 0u) {
                    input_d[load_seq] = input_ds[global_seq * blocks_per_row * 8u + chunk];
                }
            } else {
                *reinterpret_cast<unsigned*>(b_dst) = 0u;
                if (seq_off == 0u) {
                    input_d[load_seq] = 0.0f;
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

            const unsigned b_seq = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_lo]);
            const int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_hi]);

            int lo0 = 0;
            int lo1 = 0;
            int lo2 = 0;
            int lo3 = 0;
            rnb_mma_m16n8k32_s8(lo0, lo1, lo2, lo3, a0, a1, 0, 0, b0, 0, 0, 0, 0, 0);
            int hi0 = 0;
            int hi1 = 0;
            int hi2 = 0;
            int hi3 = 0;
            rnb_mma_m16n8k32_s8(hi0, hi1, hi2, hi3, 0, 0, a2, a3, 0, b1, 0, 0, 0, 0);

            const float dy_a = seq_a < seq_len ? input_d[t_col_a] : 0.0f;
            const float dy_b = seq_b < seq_len ? input_d[t_col_b] : 0.0f;
            const float scale_a_lo =
                static_cast<float>(weight_scale_lo[warp_row_off + t_row_a]);
            const float scale_a_hi =
                static_cast<float>(weight_scale_hi[warp_row_off + t_row_a]);
            const float scale_b_lo =
                static_cast<float>(weight_scale_lo[warp_row_off + t_row_b]);
            const float scale_b_hi =
                static_cast<float>(weight_scale_hi[warp_row_off + t_row_b]);
            block_acc[0] += dy_a * (scale_a_lo * static_cast<float>(lo0)
                + scale_a_hi * static_cast<float>(hi0));
            block_acc[1] += dy_b * (scale_a_lo * static_cast<float>(lo1)
                + scale_a_hi * static_cast<float>(hi1));
            block_acc[2] += dy_a * (scale_b_lo * static_cast<float>(lo2)
                + scale_b_hi * static_cast<float>(hi2));
            block_acc[3] += dy_b * (scale_b_lo * static_cast<float>(lo3)
                + scale_b_hi * static_cast<float>(hi3));
            __syncthreads();
        }

        const float d_a = weight_d[warp_row_off + t_row_a];
        const float d_b = weight_d[warp_row_off + t_row_b];
        acc[0] += d_a * block_acc[0];
        acc[1] += d_a * block_acc[1];
        acc[2] += d_b * block_acc[2];
        acc[3] += d_b * block_acc[3];
        __syncthreads();
    }

    if (row_a_valid && seq_a < seq_len) out[seq_a * rows + row_a] = acc[0];
    if (row_a_valid && seq_b < seq_len) out[seq_b * rows + row_a] = acc[1];
    if (row_b_valid && seq_a < seq_len) out[seq_a * rows + row_b] = acc[2];
    if (row_b_valid && seq_b < seq_len) out[seq_b * rows + row_b] = acc[3];
#endif
}

// cu226: 32-row x 64-sequence variant — same a-side amortization as the
// Q4_K/Q5_K seq64 tiles (grid.y halved; the expensive per-element Q6 unpack
// loop runs once for two 32-seq slabs). Per-element accumulation order
// matches tile32, so outputs stay bitwise equal. __launch_bounds__(256, 4)
// keeps 4 CTAs/SM against the doubled accumulator set.
extern "C" __global__ void __launch_bounds__(256, 4) rnb_q6k_q8_1_matmul_mmq_tile32_seq64(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
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
    __shared__ float weight_d[32];
    __shared__ signed char weight_scale_lo[32];
    __shared__ signed char weight_scale_hi[32];
    __shared__ float input_d[64];

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
    const unsigned row_bytes = blocks_per_row * 210u;

    for (unsigned block_index = 0; block_index < blocks_per_row; ++block_index) {
        float block_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (unsigned sub = 0; sub < 8u; ++sub) {
            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* a_dst = a_tile + load_row * 36u + load_off;

            if (global_row < rows) {
                const unsigned char* packed =
                    weights + global_row * row_bytes + block_index * 210u;
                if (sub == 0u && load_off == 0u) {
                    const unsigned raw_d = static_cast<unsigned>(packed[208])
                        | (static_cast<unsigned>(packed[209]) << 8);
                    weight_d[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                }
                if (load_off == 0u) {
                    weight_scale_lo[load_row] =
                        static_cast<signed char>(packed[192u + sub * 2u]);
                    weight_scale_hi[load_row] =
                        static_cast<signed char>(packed[193u + sub * 2u]);
                }
#pragma unroll
                for (unsigned i = 0; i < 4u; ++i) {
                    const unsigned elem = sub * 32u + load_off + i;
                    const unsigned half = elem >> 7;
                    const unsigned rem = elem & 127u;
                    const unsigned column = rem & 31u;
                    const unsigned ql_base = half * 64u;
                    const unsigned qh_base = 128u + half * 32u;
                    const unsigned qh = packed[qh_base + column];
                    unsigned q;
                    if (rem < 32u) {
                        q = (packed[ql_base + column] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    } else if (rem < 64u) {
                        q = (packed[ql_base + column + 32u] & 0x0fu)
                            | (((qh >> 2) & 3u) << 4);
                    } else if (rem < 96u) {
                        q = (packed[ql_base + column] >> 4) | (((qh >> 4) & 3u) << 4);
                    } else {
                        q = (packed[ql_base + column + 32u] >> 4)
                            | (((qh >> 6) & 3u) << 4);
                    }
                    a_dst[i] = static_cast<signed char>(static_cast<int>(q) - 32);
                }
            } else {
                *reinterpret_cast<unsigned*>(a_dst) = 0u;
                if (sub == 0u && load_off == 0u) {
                    weight_d[load_row] = 0.0f;
                }
                if (load_off == 0u) {
                    weight_scale_lo[load_row] = 0;
                    weight_scale_hi[load_row] = 0;
                }
            }

            // 64-seq b-slab: two 32-seq passes (Q6 has no min-term chunk sum).
            const unsigned chunk = block_index * 8u + sub;
#pragma unroll
            for (unsigned pass = 0; pass < 2u; ++pass) {
                const unsigned load_seq = (tid >> 3) + pass * 32u;
                const unsigned seq_off = (tid & 7u) * 4u;
                const unsigned global_seq = seq_base + load_seq;
                signed char* b_dst = b_tile + load_seq * 36u + seq_off;
                if (global_seq < seq_len) {
                    const signed char* b_src = input_qs
                        + global_seq * blocks_per_row * 256u + chunk * 32u + seq_off;
                    *reinterpret_cast<unsigned*>(b_dst) =
                        *reinterpret_cast<const unsigned*>(b_src);
                    if (seq_off == 0u) {
                        input_d[load_seq] = input_ds[global_seq * blocks_per_row * 8u + chunk];
                    }
                } else {
                    *reinterpret_cast<unsigned*>(b_dst) = 0u;
                    if (seq_off == 0u) {
                        input_d[load_seq] = 0.0f;
                    }
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

            const float scale_a_lo =
                static_cast<float>(weight_scale_lo[warp_row_off + t_row_a]);
            const float scale_a_hi =
                static_cast<float>(weight_scale_hi[warp_row_off + t_row_a]);
            const float scale_b_lo =
                static_cast<float>(weight_scale_lo[warp_row_off + t_row_b]);
            const float scale_b_hi =
                static_cast<float>(weight_scale_hi[warp_row_off + t_row_b]);

            const unsigned b_seq0 = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
#pragma unroll
            for (unsigned pair = 0; pair < 2u; ++pair) {
                const unsigned b_seq = b_seq0 + pair * 8u;
                const int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_lo]);
                const int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_hi]);

                int lo0 = 0;
                int lo1 = 0;
                int lo2 = 0;
                int lo3 = 0;
                rnb_mma_m16n8k32_s8(lo0, lo1, lo2, lo3, a0, a1, 0, 0, b0, 0, 0, 0, 0, 0);
                int hi0 = 0;
                int hi1 = 0;
                int hi2 = 0;
                int hi3 = 0;
                rnb_mma_m16n8k32_s8(hi0, hi1, hi2, hi3, 0, 0, a2, a3, 0, b1, 0, 0, 0, 0);

                const unsigned col_lo = t_col_a + pair * 8u;
                const unsigned col_hi = t_col_b + pair * 8u;
                const float dy_lo = (seq_base + col_lo) < seq_len ? input_d[col_lo] : 0.0f;
                const float dy_hi = (seq_base + col_hi) < seq_len ? input_d[col_hi] : 0.0f;

                block_acc[pair * 4u + 0u] += dy_lo * (scale_a_lo * static_cast<float>(lo0)
                    + scale_a_hi * static_cast<float>(hi0));
                block_acc[pair * 4u + 1u] += dy_hi * (scale_a_lo * static_cast<float>(lo1)
                    + scale_a_hi * static_cast<float>(hi1));
                block_acc[pair * 4u + 2u] += dy_lo * (scale_b_lo * static_cast<float>(lo2)
                    + scale_b_hi * static_cast<float>(hi2));
                block_acc[pair * 4u + 3u] += dy_hi * (scale_b_lo * static_cast<float>(lo3)
                    + scale_b_hi * static_cast<float>(hi3));
            }
            __syncthreads();
        }

        const float d_a = weight_d[warp_row_off + t_row_a];
        const float d_b = weight_d[warp_row_off + t_row_b];
        acc[0] += d_a * block_acc[0];
        acc[1] += d_a * block_acc[1];
        acc[2] += d_b * block_acc[2];
        acc[3] += d_b * block_acc[3];
        acc[4] += d_a * block_acc[4];
        acc[5] += d_a * block_acc[5];
        acc[6] += d_b * block_acc[6];
        acc[7] += d_b * block_acc[7];
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
// amortization as the Q4_K 64x64 tile (grid.x halved, b loads and input_d
// reads amortize over twice the output rows). The Q6 per-element unpack and
// lo/hi sub-scale MMA pair are unchanged; per-element accumulation order
// matches tile32, so outputs stay bitwise equal.
extern "C" __global__ void __launch_bounds__(512, 2) rnb_q6k_q8_1_matmul_mmq_tile64_seq64(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
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
    __shared__ float weight_d[64];
    __shared__ signed char weight_scale_lo[64];
    __shared__ signed char weight_scale_hi[64];
    __shared__ float input_d[64];

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
    const unsigned row_bytes = blocks_per_row * 210u;

    for (unsigned block_index = 0; block_index < blocks_per_row; ++block_index) {
        float block_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (unsigned sub = 0; sub < 8u; ++sub) {
            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* a_dst = a_tile + load_row * 36u + load_off;

            if (global_row < rows) {
                const unsigned char* packed =
                    weights + global_row * row_bytes + block_index * 210u;
                if (sub == 0u && load_off == 0u) {
                    const unsigned raw_d = static_cast<unsigned>(packed[208])
                        | (static_cast<unsigned>(packed[209]) << 8);
                    weight_d[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                }
                if (load_off == 0u) {
                    weight_scale_lo[load_row] =
                        static_cast<signed char>(packed[192u + sub * 2u]);
                    weight_scale_hi[load_row] =
                        static_cast<signed char>(packed[193u + sub * 2u]);
                }
#pragma unroll
                for (unsigned i = 0; i < 4u; ++i) {
                    const unsigned elem = sub * 32u + load_off + i;
                    const unsigned half = elem >> 7;
                    const unsigned rem = elem & 127u;
                    const unsigned column = rem & 31u;
                    const unsigned ql_base = half * 64u;
                    const unsigned qh_base = 128u + half * 32u;
                    const unsigned qh = packed[qh_base + column];
                    unsigned q;
                    if (rem < 32u) {
                        q = (packed[ql_base + column] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    } else if (rem < 64u) {
                        q = (packed[ql_base + column + 32u] & 0x0fu)
                            | (((qh >> 2) & 3u) << 4);
                    } else if (rem < 96u) {
                        q = (packed[ql_base + column] >> 4) | (((qh >> 4) & 3u) << 4);
                    } else {
                        q = (packed[ql_base + column + 32u] >> 4)
                            | (((qh >> 6) & 3u) << 4);
                    }
                    a_dst[i] = static_cast<signed char>(static_cast<int>(q) - 32);
                }
            } else {
                *reinterpret_cast<unsigned*>(a_dst) = 0u;
                if (sub == 0u && load_off == 0u) {
                    weight_d[load_row] = 0.0f;
                }
                if (load_off == 0u) {
                    weight_scale_lo[load_row] = 0;
                    weight_scale_hi[load_row] = 0;
                }
            }

            // 64-seq b-slab in a single 512-thread pass.
            const unsigned chunk = block_index * 8u + sub;
            {
                const unsigned load_seq = tid >> 3;
                const unsigned seq_off = (tid & 7u) * 4u;
                const unsigned global_seq = seq_base + load_seq;
                signed char* b_dst = b_tile + load_seq * 36u + seq_off;
                if (global_seq < seq_len) {
                    const signed char* b_src = input_qs
                        + global_seq * blocks_per_row * 256u + chunk * 32u + seq_off;
                    *reinterpret_cast<unsigned*>(b_dst) =
                        *reinterpret_cast<const unsigned*>(b_src);
                    if (seq_off == 0u) {
                        input_d[load_seq] = input_ds[global_seq * blocks_per_row * 8u + chunk];
                    }
                } else {
                    *reinterpret_cast<unsigned*>(b_dst) = 0u;
                    if (seq_off == 0u) {
                        input_d[load_seq] = 0.0f;
                    }
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

            const float scale_a_lo =
                static_cast<float>(weight_scale_lo[warp_row_off + t_row_a]);
            const float scale_a_hi =
                static_cast<float>(weight_scale_hi[warp_row_off + t_row_a]);
            const float scale_b_lo =
                static_cast<float>(weight_scale_lo[warp_row_off + t_row_b]);
            const float scale_b_hi =
                static_cast<float>(weight_scale_hi[warp_row_off + t_row_b]);

            const unsigned b_seq0 = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
#pragma unroll
            for (unsigned pair = 0; pair < 2u; ++pair) {
                const unsigned b_seq = b_seq0 + pair * 8u;
                const int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_lo]);
                const int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 36u + b_col_hi]);

                int lo0 = 0;
                int lo1 = 0;
                int lo2 = 0;
                int lo3 = 0;
                rnb_mma_m16n8k32_s8(lo0, lo1, lo2, lo3, a0, a1, 0, 0, b0, 0, 0, 0, 0, 0);
                int hi0 = 0;
                int hi1 = 0;
                int hi2 = 0;
                int hi3 = 0;
                rnb_mma_m16n8k32_s8(hi0, hi1, hi2, hi3, 0, 0, a2, a3, 0, b1, 0, 0, 0, 0);

                const unsigned col_lo = t_col_a + pair * 8u;
                const unsigned col_hi = t_col_b + pair * 8u;
                const float dy_lo = (seq_base + col_lo) < seq_len ? input_d[col_lo] : 0.0f;
                const float dy_hi = (seq_base + col_hi) < seq_len ? input_d[col_hi] : 0.0f;

                block_acc[pair * 4u + 0u] += dy_lo * (scale_a_lo * static_cast<float>(lo0)
                    + scale_a_hi * static_cast<float>(hi0));
                block_acc[pair * 4u + 1u] += dy_hi * (scale_a_lo * static_cast<float>(lo1)
                    + scale_a_hi * static_cast<float>(hi1));
                block_acc[pair * 4u + 2u] += dy_lo * (scale_b_lo * static_cast<float>(lo2)
                    + scale_b_hi * static_cast<float>(hi2));
                block_acc[pair * 4u + 3u] += dy_hi * (scale_b_lo * static_cast<float>(lo3)
                    + scale_b_hi * static_cast<float>(hi3));
            }
            __syncthreads();
        }

        const float d_a = weight_d[warp_row_off + t_row_a];
        const float d_b = weight_d[warp_row_off + t_row_b];
        acc[0] += d_a * block_acc[0];
        acc[1] += d_a * block_acc[1];
        acc[2] += d_b * block_acc[2];
        acc[3] += d_b * block_acc[3];
        acc[4] += d_a * block_acc[4];
        acc[5] += d_a * block_acc[5];
        acc[6] += d_b * block_acc[6];
        acc[7] += d_b * block_acc[7];
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

__device__ __forceinline__ int rnb_q6k_mmq_load_word(
    const unsigned char* packed,
    unsigned sub,
    unsigned word_in_sub) {
    const unsigned half = sub >> 2;
    const unsigned quarter = sub & 3u;
    const unsigned ql_offset =
        half * 64u + (quarter & 1u) * 32u + word_in_sub * 4u;
    const unsigned qh_offset =
        128u + half * 32u + word_in_sub * 4u;
    const unsigned short* ql16 =
        reinterpret_cast<const unsigned short*>(packed + ql_offset);
    const unsigned short* qh16 =
        reinterpret_cast<const unsigned short*>(packed + qh_offset);
    const unsigned ql = static_cast<unsigned>(ql16[0])
        | (static_cast<unsigned>(ql16[1]) << 16);
    const unsigned qh = static_cast<unsigned>(qh16[0])
        | (static_cast<unsigned>(qh16[1]) << 16);
    const unsigned ql_nibbles = quarter < 2u
        ? ql & 0x0f0f0f0fu
        : (ql >> 4) & 0x0f0f0f0fu;
    const unsigned qh_bits =
        (qh >> (quarter * 2u)) & 0x03030303u;
    return __vsubss4(
        static_cast<int>(ql_nibbles | (qh_bits << 4)),
        0x20202020);
}

// 128-row x 64-sequence Ampere MMQ tile. The quant layout and shape gate are
// generic Q6_K contracts; model-specific admission stays above this kernel.
// Eight warps each own one 16-row slab and reuse the fully unpacked Q6 block
// across eight 8-column MMA fragments. The narrower sequence tile halves the
// accumulator footprint so two CTAs can remain resident per SM.
extern "C" __global__ void __launch_bounds__(256, 2)
rnb_q6k_q8_1_matmul_mmq_tile128_seq64(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
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
    const unsigned row_base = blockIdx.x * 128u;
    const unsigned seq_base = blockIdx.y * 64u;
    const unsigned warp_row_off = warp * 16u;
    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;

    __shared__ int b_tile[8][8][8][8];
    __shared__ float x_d[128];
    __shared__ signed char x_scale_lo[8][128];
    __shared__ signed char x_scale_hi[8][128];
    __shared__ float y_d[8][64];

    const unsigned row_a = row_base + warp_row_off + t_row_a;
    const unsigned row_b = row_base + warp_row_off + t_row_b;
    float acc[32];
#pragma unroll
    for (unsigned i = 0; i < 32u; ++i) {
        acc[i] = 0.0f;
    }

    const unsigned row_bytes = blocks_per_row * 210u;
    for (unsigned block = 0; block < blocks_per_row; ++block) {
        if (tid < 128u) {
            const unsigned global_row = row_base + tid;
            if (global_row < rows) {
                const unsigned char* packed =
                    weights + global_row * row_bytes + block * 210u;
                const unsigned raw_d = static_cast<unsigned>(packed[208])
                    | (static_cast<unsigned>(packed[209]) << 8);
                x_d[tid] =
                    __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
#pragma unroll
                for (unsigned sub = 0; sub < 8u; ++sub) {
                    x_scale_lo[sub][tid] =
                        static_cast<signed char>(packed[192u + sub * 2u]);
                    x_scale_hi[sub][tid] =
                        static_cast<signed char>(packed[193u + sub * 2u]);
                }
            } else {
                x_d[tid] = 0.0f;
#pragma unroll
                for (unsigned sub = 0; sub < 8u; ++sub) {
                    x_scale_lo[sub][tid] = 0;
                    x_scale_hi[sub][tid] = 0;
                }
            }
        }

        for (unsigned item = tid; item < 4096u; item += 256u) {
            const unsigned slot = item >> 9;
            const unsigned local = item & 511u;
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
                    y_d[slot][load_seq] =
                        input_ds[global_seq * blocks_per_row * 8u + chunk];
                }
            } else if (word == 0u) {
                y_d[slot][load_seq] = 0.0f;
            }
            b_tile[slot][load_seq >> 3][word][load_seq & 7u] = b_word;
        }
        __syncthreads();

        const unsigned word_lo = lane & 3u;
        const unsigned word_hi = word_lo + 4u;
#pragma unroll
        for (unsigned sub = 0; sub < 8u; ++sub) {
            int a0 = 0;
            int a1 = 0;
            int a2 = 0;
            int a3 = 0;
            if (row_a < rows) {
                const unsigned char* packed =
                    weights + row_a * row_bytes + block * 210u;
                a0 = rnb_q6k_mmq_load_word(packed, sub, word_lo);
                a2 = rnb_q6k_mmq_load_word(packed, sub, word_hi);
            }
            if (row_b < rows) {
                const unsigned char* packed =
                    weights + row_b * row_bytes + block * 210u;
                a1 = rnb_q6k_mmq_load_word(packed, sub, word_lo);
                a3 = rnb_q6k_mmq_load_word(packed, sub, word_hi);
            }
            const float d_a = x_d[warp_row_off + t_row_a];
            const float d_b = x_d[warp_row_off + t_row_b];
            const float scale_a_lo =
                static_cast<float>(x_scale_lo[sub][warp_row_off + t_row_a]);
            const float scale_a_hi =
                static_cast<float>(x_scale_hi[sub][warp_row_off + t_row_a]);
            const float scale_b_lo =
                static_cast<float>(x_scale_lo[sub][warp_row_off + t_row_b]);
            const float scale_b_hi =
                static_cast<float>(x_scale_hi[sub][warp_row_off + t_row_b]);

#pragma unroll
            for (unsigned frag = 0; frag < 8u; ++frag) {
                const unsigned seq_in_group = lane >> 2;
                const int b0 = b_tile[sub][frag][word_lo][seq_in_group];
                const int b1 = b_tile[sub][frag][word_hi][seq_in_group];

                int lo0 = 0;
                int lo1 = 0;
                int lo2 = 0;
                int lo3 = 0;
                rnb_mma_m16n8k32_s8(
                    lo0, lo1, lo2, lo3,
                    a0, a1, 0, 0,
                    b0, 0,
                    0, 0, 0, 0);
                int hi0 = 0;
                int hi1 = 0;
                int hi2 = 0;
                int hi3 = 0;
                rnb_mma_m16n8k32_s8(
                    hi0, hi1, hi2, hi3,
                    0, 0, a2, a3,
                    0, b1,
                    0, 0, 0, 0);

                const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
                const unsigned col_b = col_a + 1u;
                const float dy_a = y_d[sub][col_a];
                const float dy_b = y_d[sub][col_b];
                acc[frag * 4u + 0u] += d_a * dy_a
                    * (scale_a_lo * static_cast<float>(lo0)
                        + scale_a_hi * static_cast<float>(hi0));
                acc[frag * 4u + 1u] += d_a * dy_b
                    * (scale_a_lo * static_cast<float>(lo1)
                        + scale_a_hi * static_cast<float>(hi1));
                acc[frag * 4u + 2u] += d_b * dy_a
                    * (scale_b_lo * static_cast<float>(lo2)
                        + scale_b_hi * static_cast<float>(hi2));
                acc[frag * 4u + 3u] += d_b * dy_b
                    * (scale_b_lo * static_cast<float>(lo3)
                        + scale_b_hi * static_cast<float>(hi3));
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned frag = 0; frag < 8u; ++frag) {
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
#endif
}

// Adapted from llama.cpp/ggml-cuda's current Ampere Q6_K MMQ configuration:
// non-fallback, no-MUL_MAT_ID, I=128/J=128, eight warps and occupancy one.
// The activation buffer is the local RnbBlockQ8_1Mmq layout from
// q8_1_mmq.cuh, not llama.cpp storage.
extern "C" __global__ void __launch_bounds__(256, 1)
rnb_q6k_q8_1_matmul_mmq_llama_ampere_j128(
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
    const unsigned row_bytes = blocks_per_row * 210u;

    __shared__ unsigned char x_raw[128][210];
    __shared__ int y_qs[4][16][8][8];
    __shared__ float y_d[4][128];

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
            for (unsigned item = tid; item < 128u * 210u; item += 256u) {
                const unsigned local_row = item / 210u;
                const unsigned offset = item - local_row * 210u;
                const unsigned global_row = row_base + local_row;
                x_raw[local_row][offset] = global_row < rows
                    ? weights[global_row * row_bytes + block * 210u + offset] : 0u;
            }
            __syncthreads();

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
                    float d = 0.0f;
                    if (global_seq < seq_len) {
                        const RnbBlockQ8_1Mmq* source =
                            input + (unsigned long long)(block * 2u + half) * seq_len + global_seq;
                        d = source->d4[local_sub];
                    }
                    y_d[local_sub][local_seq] = d;
                }
                __syncthreads();

                const unsigned word_lo = lane & 3u;
                const unsigned word_hi = word_lo + 4u;
#pragma unroll
                for (unsigned local_sub = 0; local_sub < 4u; ++local_sub) {
                    const unsigned sub = half * 4u + local_sub;
                    const unsigned char* packed_a = x_raw[warp_row_off + (lane >> 2)];
                    const unsigned char* packed_b = x_raw[warp_row_off + (lane >> 2) + 8u];
                    const int a0 = rnb_q6k_mmq_load_word(packed_a, sub, word_lo);
                    const int a1 = rnb_q6k_mmq_load_word(packed_b, sub, word_lo);
                    const int a2 = rnb_q6k_mmq_load_word(packed_a, sub, word_hi);
                    const int a3 = rnb_q6k_mmq_load_word(packed_b, sub, word_hi);
                    const float d_a = __half2float(__ushort_as_half(static_cast<unsigned short>(
                        x_raw[warp_row_off + (lane >> 2)][208]
                        | (static_cast<unsigned>(x_raw[warp_row_off + (lane >> 2)][209]) << 8))));
                    const float d_b = __half2float(__ushort_as_half(static_cast<unsigned short>(
                        x_raw[warp_row_off + (lane >> 2) + 8u][208]
                        | (static_cast<unsigned>(x_raw[warp_row_off + (lane >> 2) + 8u][209]) << 8))));
                    const float scale_a_lo = static_cast<float>(static_cast<signed char>(
                        x_raw[warp_row_off + (lane >> 2)][192u + sub * 2u]));
                    const float scale_a_hi = static_cast<float>(static_cast<signed char>(
                        x_raw[warp_row_off + (lane >> 2)][193u + sub * 2u]));
                    const float scale_b_lo = static_cast<float>(static_cast<signed char>(
                        x_raw[warp_row_off + (lane >> 2) + 8u][192u + sub * 2u]));
                    const float scale_b_hi = static_cast<float>(static_cast<signed char>(
                        x_raw[warp_row_off + (lane >> 2) + 8u][193u + sub * 2u]));
#pragma unroll
                    for (unsigned frag = 0; frag < 16u; ++frag) {
                        const unsigned seq_in_group = lane >> 2;
                        const int b0 = y_qs[local_sub][frag][word_lo][seq_in_group];
                        const int b1 = y_qs[local_sub][frag][word_hi][seq_in_group];
                        int lo0 = 0, lo1 = 0, lo2 = 0, lo3 = 0;
                        rnb_mma_m16n8k32_s8(lo0, lo1, lo2, lo3, a0, a1, 0, 0, b0, 0, 0, 0, 0, 0);
                        int hi0 = 0, hi1 = 0, hi2 = 0, hi3 = 0;
                        rnb_mma_m16n8k32_s8(hi0, hi1, hi2, hi3, 0, 0, a2, a3, 0, b1, 0, 0, 0, 0);
                        const unsigned col_a = frag * 8u + ((lane & 3u) << 1);
                        const unsigned col_b = col_a + 1u;
                        const float dy_a = y_d[local_sub][col_a];
                        const float dy_b = y_d[local_sub][col_b];
                        acc[frag * 4u] += d_a * dy_a * (scale_a_lo * lo0 + scale_a_hi * hi0);
                        acc[frag * 4u + 1u] += d_a * dy_b * (scale_a_lo * lo1 + scale_a_hi * hi1);
                        acc[frag * 4u + 2u] += d_b * dy_a * (scale_b_lo * lo2 + scale_b_hi * hi2);
                        acc[frag * 4u + 3u] += d_b * dy_b * (scale_b_lo * lo3 + scale_b_hi * hi3);
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
