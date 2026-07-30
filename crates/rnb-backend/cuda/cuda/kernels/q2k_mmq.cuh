// Q2_K x Q8_1 tiled matrix multiply for Ampere-class integer tensor cores.
//
// One 8-warp CTA computes a 32-row x 32-sequence output tile directly from
// the canonical 84-byte Q2_K block (scales[16] | qs[64] | d | dmin). Q2_K
// carries a 4-bit scale and a 4-bit min per 16-value group, so each 32-wide
// Q8_1 chunk is evaluated as separate low/high MMA halves (like Q6_K) and
// the min term uses per-half activation sums (like Q4_K).

extern "C" __global__ void rnb_q2k_q8_1_matmul_mmq_tile32(
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
    __shared__ float x_d[32];
    __shared__ float x_dmin[32];
    __shared__ unsigned char x_s_lo[32];
    __shared__ unsigned char x_s_hi[32];
    __shared__ float y_d[32];

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
    const unsigned row_bytes = blocks_per_row * 84u;

    for (unsigned block_index = 0; block_index < blocks_per_row; ++block_index) {
        float block_d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float block_m[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (unsigned sub = 0; sub < 8u; ++sub) {
            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* a_dst = a_tile + load_row * 32u + load_off;

            if (global_row < rows) {
                const unsigned char* packed =
                    weights + global_row * row_bytes + block_index * 84u;
                if (sub == 0u && load_off == 0u) {
                    const unsigned raw_d = static_cast<unsigned>(packed[80])
                        | (static_cast<unsigned>(packed[81]) << 8);
                    const unsigned raw_dmin = static_cast<unsigned>(packed[82])
                        | (static_cast<unsigned>(packed[83]) << 8);
                    x_d[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
                    x_dmin[load_row] =
                        __half2float(__ushort_as_half(static_cast<unsigned short>(raw_dmin)));
                }
                if (load_off == 0u) {
                    x_s_lo[load_row] = packed[sub * 2u];
                    x_s_hi[load_row] = packed[sub * 2u + 1u];
                }
#pragma unroll
                for (unsigned i = 0; i < 4u; ++i) {
                    const unsigned elem = sub * 32u + load_off + i;
                    const unsigned q_index = (elem >> 7) * 32u + (elem & 31u);
                    const unsigned shift = ((elem & 127u) >> 5) * 2u;
                    a_dst[i] = static_cast<signed char>(
                        (packed[16u + q_index] >> shift) & 3u);
                }
            } else {
                *reinterpret_cast<unsigned*>(a_dst) = 0u;
                if (sub == 0u && load_off == 0u) {
                    x_d[load_row] = 0.0f;
                    x_dmin[load_row] = 0.0f;
                }
                if (load_off == 0u) {
                    x_s_lo[load_row] = 0u;
                    x_s_hi[load_row] = 0u;
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
                    y_d[load_seq] = input_ds[global_seq * blocks_per_row * 8u + chunk];
                }
            } else {
                *reinterpret_cast<unsigned*>(b_dst) = 0u;
                if (seq_off == 0u) {
                    y_d[load_seq] = 0.0f;
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

            int sum_lo_a = 0;
            int sum_hi_a = 0;
            int sum_lo_b = 0;
            int sum_hi_b = 0;
            const bool seq_a_valid = seq_a < seq_len;
            const bool seq_b_valid = seq_b < seq_len;
#pragma unroll
            for (int k = 0; k < 16; k += 4) {
                if (seq_a_valid) {
                    sum_lo_a = __dp4a(0x01010101,
                        *reinterpret_cast<const int*>(&b_tile[t_col_a * 32u + k]), sum_lo_a);
                    sum_hi_a = __dp4a(0x01010101,
                        *reinterpret_cast<const int*>(&b_tile[t_col_a * 32u + 16 + k]), sum_hi_a);
                }
                if (seq_b_valid) {
                    sum_lo_b = __dp4a(0x01010101,
                        *reinterpret_cast<const int*>(&b_tile[t_col_b * 32u + k]), sum_lo_b);
                    sum_hi_b = __dp4a(0x01010101,
                        *reinterpret_cast<const int*>(&b_tile[t_col_b * 32u + 16 + k]), sum_hi_b);
                }
            }

            const float dy_a = seq_a_valid ? y_d[t_col_a] : 0.0f;
            const float dy_b = seq_b_valid ? y_d[t_col_b] : 0.0f;
            const unsigned s_lo_a = x_s_lo[warp_row_off + t_row_a];
            const unsigned s_hi_a = x_s_hi[warp_row_off + t_row_a];
            const unsigned s_lo_b = x_s_lo[warp_row_off + t_row_b];
            const unsigned s_hi_b = x_s_hi[warp_row_off + t_row_b];
            const float sc_lo_a = static_cast<float>(s_lo_a & 0x0fu);
            const float sc_hi_a = static_cast<float>(s_hi_a & 0x0fu);
            const float mn_lo_a = static_cast<float>(s_lo_a >> 4);
            const float mn_hi_a = static_cast<float>(s_hi_a >> 4);
            const float sc_lo_b = static_cast<float>(s_lo_b & 0x0fu);
            const float sc_hi_b = static_cast<float>(s_hi_b & 0x0fu);
            const float mn_lo_b = static_cast<float>(s_lo_b >> 4);
            const float mn_hi_b = static_cast<float>(s_hi_b >> 4);

            block_d[0] += dy_a * (sc_lo_a * static_cast<float>(lo0)
                + sc_hi_a * static_cast<float>(hi0));
            block_d[1] += dy_b * (sc_lo_a * static_cast<float>(lo1)
                + sc_hi_a * static_cast<float>(hi1));
            block_d[2] += dy_a * (sc_lo_b * static_cast<float>(lo2)
                + sc_hi_b * static_cast<float>(hi2));
            block_d[3] += dy_b * (sc_lo_b * static_cast<float>(lo3)
                + sc_hi_b * static_cast<float>(hi3));
            block_m[0] += dy_a * (mn_lo_a * static_cast<float>(sum_lo_a)
                + mn_hi_a * static_cast<float>(sum_hi_a));
            block_m[1] += dy_b * (mn_lo_a * static_cast<float>(sum_lo_b)
                + mn_hi_a * static_cast<float>(sum_hi_b));
            block_m[2] += dy_a * (mn_lo_b * static_cast<float>(sum_lo_a)
                + mn_hi_b * static_cast<float>(sum_hi_a));
            block_m[3] += dy_b * (mn_lo_b * static_cast<float>(sum_lo_b)
                + mn_hi_b * static_cast<float>(sum_hi_b));
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
        __syncthreads();
    }

    if (row_a_valid && seq_a < seq_len) out[seq_a * rows + row_a] = acc[0];
    if (row_a_valid && seq_b < seq_len) out[seq_b * rows + row_a] = acc[1];
    if (row_b_valid && seq_a < seq_len) out[seq_a * rows + row_b] = acc[2];
    if (row_b_valid && seq_b < seq_len) out[seq_b * rows + row_b] = acc[3];
#endif
}
