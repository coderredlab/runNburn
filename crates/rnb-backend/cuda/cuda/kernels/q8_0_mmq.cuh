// Q8_0 x Q8_1 tiled matrix multiply for Ampere-class integer tensor cores.
// One 8-warp CTA computes a 32-row x 32-sequence output tile while reusing
// each packed weight chunk across 32 input rows.

extern "C" __global__ void rnb_q8_0_q8_1_matmul_mmq_tile32(
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
    __shared__ float weight_d[32];
    __shared__ float input_d[32];

    const unsigned tile_row_a = lane >> 2;
    const unsigned tile_row_b = tile_row_a + 8u;
    const unsigned tile_seq_a = warp_seq_off + ((lane & 3u) << 1);
    const unsigned tile_seq_b = tile_seq_a + 1u;
    const unsigned row_a = row_base + warp_row_off + tile_row_a;
    const unsigned row_b = row_base + warp_row_off + tile_row_b;
    const unsigned seq_a = seq_base + tile_seq_a;
    const unsigned seq_b = seq_base + tile_seq_b;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;

    float acc_a0 = 0.0f;
    float acc_a1 = 0.0f;
    float acc_b0 = 0.0f;
    float acc_b1 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 34u;

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        const unsigned load_row = tid >> 3;
        const unsigned load_off = (tid & 7u) * 4u;
        const unsigned global_row = row_base + load_row;
        signed char* a_dst = a_tile + load_row * 32u + load_off;
        if (global_row < rows) {
            const unsigned char* packed = weights + global_row * row_bytes + block * 34u;
            const unsigned packed_qs =
                static_cast<unsigned>(packed[2u + load_off])
                | (static_cast<unsigned>(packed[3u + load_off]) << 8)
                | (static_cast<unsigned>(packed[4u + load_off]) << 16)
                | (static_cast<unsigned>(packed[5u + load_off]) << 24);
            *reinterpret_cast<unsigned*>(a_dst) = packed_qs;
            if (load_off == 0u) {
                const unsigned raw_d = static_cast<unsigned>(packed[0])
                    | (static_cast<unsigned>(packed[1]) << 8);
                weight_d[load_row] =
                    __half2float(__ushort_as_half(static_cast<unsigned short>(raw_d)));
            }
        } else {
            *reinterpret_cast<unsigned*>(a_dst) = 0u;
            if (load_off == 0u) {
                weight_d[load_row] = 0.0f;
            }
        }

        const unsigned load_seq = tid >> 3;
        const unsigned seq_off = (tid & 7u) * 4u;
        const unsigned global_seq = seq_base + load_seq;
        signed char* b_dst = b_tile + load_seq * 32u + seq_off;
        if (global_seq < seq_len) {
            const signed char* b_src =
                input_qs + global_seq * blocks_per_row * 32u + block * 32u + seq_off;
            *reinterpret_cast<unsigned*>(b_dst) = *reinterpret_cast<const unsigned*>(b_src);
            if (seq_off == 0u) {
                input_d[load_seq] = input_ds[global_seq * blocks_per_row + block];
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
            &a_tile[(warp_row_off + tile_row_a) * 32u + a_col_lo]);
        const int a1 = *reinterpret_cast<const int*>(
            &a_tile[(warp_row_off + tile_row_b) * 32u + a_col_lo]);
        const int a2 = *reinterpret_cast<const int*>(
            &a_tile[(warp_row_off + tile_row_a) * 32u + a_col_hi]);
        const int a3 = *reinterpret_cast<const int*>(
            &a_tile[(warp_row_off + tile_row_b) * 32u + a_col_hi]);

        const unsigned b_seq = warp_seq_off + (lane >> 2);
        const unsigned b_col_lo = (lane & 3u) * 4u;
        const unsigned b_col_hi = b_col_lo + 16u;
        const int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_col_lo]);
        const int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_col_hi]);

        int dot_a0 = 0;
        int dot_a1 = 0;
        int dot_b0 = 0;
        int dot_b1 = 0;
        rnb_mma_m16n8k32_s8(
            dot_a0,
            dot_a1,
            dot_b0,
            dot_b1,
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            0,
            0,
            0,
            0);

        const float scale_row_a = weight_d[warp_row_off + tile_row_a];
        const float scale_row_b = weight_d[warp_row_off + tile_row_b];
        const float scale_seq_a = seq_a < seq_len ? input_d[tile_seq_a] : 0.0f;
        const float scale_seq_b = seq_b < seq_len ? input_d[tile_seq_b] : 0.0f;
        acc_a0 += scale_row_a * scale_seq_a * static_cast<float>(dot_a0);
        acc_a1 += scale_row_a * scale_seq_b * static_cast<float>(dot_a1);
        acc_b0 += scale_row_b * scale_seq_a * static_cast<float>(dot_b0);
        acc_b1 += scale_row_b * scale_seq_b * static_cast<float>(dot_b1);
        __syncthreads();
    }

    if (row_a_valid && seq_a < seq_len) out[seq_a * rows + row_a] = acc_a0;
    if (row_a_valid && seq_b < seq_len) out[seq_b * rows + row_a] = acc_a1;
    if (row_b_valid && seq_a < seq_len) out[seq_a * rows + row_b] = acc_b0;
    if (row_b_valid && seq_b < seq_len) out[seq_b * rows + row_b] = acc_b1;
#endif
}

