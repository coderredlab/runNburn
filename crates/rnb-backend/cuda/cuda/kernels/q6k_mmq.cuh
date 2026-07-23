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

    __shared__ signed char a_tile[32 * 32];
    __shared__ signed char b_tile[32 * 32];
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
            signed char* a_dst = a_tile + load_row * 32u + load_off;

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
            signed char* b_dst = b_tile + load_seq * 32u + seq_off;
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
                &a_tile[(warp_row_off + t_row_a) * 32u + a_col_lo]);
            const int a1 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_b) * 32u + a_col_lo]);
            const int a2 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_a) * 32u + a_col_hi]);
            const int a3 = *reinterpret_cast<const int*>(
                &a_tile[(warp_row_off + t_row_b) * 32u + a_col_hi]);

            const unsigned b_seq = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_col_lo]);
            const int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_col_hi]);

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
