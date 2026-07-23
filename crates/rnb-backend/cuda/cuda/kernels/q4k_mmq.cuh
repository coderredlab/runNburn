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

    __shared__ signed char a_tile[32 * 32];
    __shared__ signed char b_tile[32 * 32];
    __shared__ float x_d[32];
    __shared__ float x_dmin[32];
    __shared__ unsigned char x_sc[32];
    __shared__ unsigned char x_mn[32];
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
            signed char* a_dst = a_tile + load_row * 32u + load_off;

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

            const unsigned load_seq = tid >> 3;
            const unsigned seq_off = (tid & 7u) * 4u;
            const unsigned global_seq = seq_base + load_seq;
            signed char* b_dst = b_tile + load_seq * 32u + seq_off;
            if (global_seq < seq_len) {
                const unsigned chunk = block * 8u + sub;
                const signed char* b_src = input_qs +
                    (global_seq * blocks_per_row * 256u) + chunk * 32u + seq_off;
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

            int d0 = 0;
            int d1 = 0;
            int d2 = 0;
            int d3 = 0;
            rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            int sum_qy_a = 0;
            int sum_qy_b = 0;
            const bool seq_a_valid = seq_a < seq_len;
            const bool seq_b_valid = seq_b < seq_len;
#pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (seq_a_valid) {
                    const int y_a = *reinterpret_cast<const int*>(&b_tile[t_col_a * 32u + k]);
                    sum_qy_a = __dp4a(0x01010101, y_a, sum_qy_a);
                }
                if (seq_b_valid) {
                    const int y_b = *reinterpret_cast<const int*>(&b_tile[t_col_b * 32u + k]);
                    sum_qy_b = __dp4a(0x01010101, y_b, sum_qy_b);
                }
            }
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
            block_m_a0 += dy_a * min_a * static_cast<float>(sum_qy_a);
            block_m_a1 += dy_b * min_a * static_cast<float>(sum_qy_b);
            block_m_b0 += dy_a * min_b * static_cast<float>(sum_qy_a);
            block_m_b1 += dy_b * min_b * static_cast<float>(sum_qy_b);
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

