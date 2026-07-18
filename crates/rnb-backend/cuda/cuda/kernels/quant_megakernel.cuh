extern "C" __global__ void rnb_q4k_ple_megakernel_m1(
    float* __restrict__ residual,
    float* __restrict__ out_scratch,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ norm_weight,
    float eps,
    int* __restrict__ counter,
    unsigned len,
    unsigned blocks_per_row,
    unsigned unit_offset,
    unsigned rows_per_block) {

    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_count = blockDim.x >> 5;
    const unsigned rows_per_warp = rows_per_block / warp_count;

    // Phase A: multi-block Q4K GEMV
    for (unsigned r = 0; r < rows_per_warp; ++r) {
        const unsigned row = blockIdx.x * rows_per_block + warp * rows_per_warp + r;
        if (row >= len) continue;

        float acc = 0.0f;
        const unsigned row_bytes = blocks_per_row * 144u;
        const unsigned char* row_ptr = weights + row * row_bytes;

        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 144u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned j = chunk >> 3;
                unsigned sc;
                unsigned mn;
                if (j < 4u) {
                    sc = block[4u + j] & 63u;
                    mn = block[4u + j + 4u] & 63u;
                } else {
                    sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                    mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
                }

                const unsigned elem = (chunk & 7u) * 4u;
                const unsigned q_index = (j >> 1) * 32u + elem;
                const unsigned char* q_ptr = block + 16u + q_index;
                const signed char* x_qs = input_qs + b * 256u + j * 32u + elem;
                const int q_pack = rnb_q4_pack4(q_ptr, j);
                const int x_pack = rnb_load_i32_aligned4(x_qs);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = input_ds[b * 8u + j];
                acc += x_d * ((d * (float)sc) * (float)dot - dmin * (float)mn * (float)x_sum);
            }
        }

        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }

        if (lane == 0u) {
            out_scratch[row] = acc;
        }
    }

    // Counter-based sync: signal Phase A completion
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&counter[0], 1);
    }

    // Phase B: block 0 polls counter then does RMSNorm + residual add
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            volatile int* ctr = (volatile int*)counter;
            while (ctr[0] < (int)gridDim.x) {
                // busy-wait
            }
        }
        __threadfence();
        __syncthreads();

        __shared__ float partial[256];
        float sum = 0.0f;
        for (unsigned i = threadIdx.x; i < len; i += blockDim.x) {
            const float v = out_scratch[i];
            sum += v * v;
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const float inv_rms = rsqrtf(partial[0] / (float)len + eps);
        for (unsigned i = threadIdx.x; i < len; i += blockDim.x) {
            const float scale = unit_offset != 0u ? (1.0f + norm_weight[i]) : norm_weight[i];
            residual[i] += out_scratch[i] * inv_rms * scale;
        }
    }
}
