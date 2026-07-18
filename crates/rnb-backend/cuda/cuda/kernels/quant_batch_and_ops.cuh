extern "C" __global__ void rnb_f32_to_f16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    unsigned len) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = __half_as_ushort(__float2half_rn(input[idx]));
    }
}

extern "C" __global__ void rnb_f16_to_f32(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    unsigned len) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = __half2float(__ushort_as_half(input[idx]));
    }
}

extern "C" __global__ void rnb_q4k_gemv_batch(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned row = blockIdx.x;
    const unsigned seq = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || seq >= seq_len || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_row = input + seq * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 144u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

        const unsigned j = tid >> 5;
        unsigned sc;
        unsigned mn;
        if (j < 4) {
            sc = block[4 + j] & 63u;
            mn = block[4 + j + 4] & 63u;
        } else {
            sc = (block[4 + j + 4] & 0x0fu) | ((block[4 + j - 4] >> 6) << 4);
            mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
        }

        const unsigned local = tid & 63u;
        const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
        unsigned q = block[16 + q_index];
        q = local < 32u ? (q & 0x0fu) : (q >> 4);

        const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
        acc += y * input_row[b * 256u + tid];
    }

    partial[tid] = acc;
    __syncthreads();

    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[seq * rows + row] = partial[0];
    }
}

extern "C" __global__ void rnb_q4k_gemv_batch_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    const bool valid = row < rows && seq < seq_len;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_row = input + seq * blocks_per_row * 256u;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 144u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned j = tid >> 5;
                unsigned sc;
                unsigned mn;
                if (j < 4u) {
                    sc = block[4u + j] & 63u;
                    mn = block[4u + j + 4u] & 63u;
                } else {
                    sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                    mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
                }

                const unsigned local = tid & 63u;
                const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
                unsigned q = block[16u + q_index];
                q = local < 32u ? (q & 0x0fu) : (q >> 4);

                const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
                acc += y * input_row[b * 256u + tid];
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

extern "C" __global__ void rnb_q4k_gemv_batch_seq4_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq_base = blockIdx.y * 4u;
    const bool row_valid = row < rows;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (row_valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 144u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned j = tid >> 5;
                unsigned sc;
                unsigned mn;
                if (j < 4u) {
                    sc = block[4u + j] & 63u;
                    mn = block[4u + j + 4u] & 63u;
                } else {
                    sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                    mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
                }

                const unsigned local = tid & 63u;
                const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
                unsigned q = block[16u + q_index];
                q = local < 32u ? (q & 0x0fu) : (q >> 4);
                const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
                const unsigned input_offset = b * 256u + tid;

                const unsigned seq0 = seq_base + 0u;
                if (seq0 < seq_len) {
                    acc0 += y * input[seq0 * blocks_per_row * 256u + input_offset];
                }
                const unsigned seq1 = seq_base + 1u;
                if (seq1 < seq_len) {
                    acc1 += y * input[seq1 * blocks_per_row * 256u + input_offset];
                }
                const unsigned seq2 = seq_base + 2u;
                if (seq2 < seq_len) {
                    acc2 += y * input[seq2 * blocks_per_row * 256u + input_offset];
                }
                const unsigned seq3 = seq_base + 3u;
                if (seq3 < seq_len) {
                    acc3 += y * input[seq3 * blocks_per_row * 256u + input_offset];
                }
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (row_valid && lane == 0u) {
        const unsigned seq0 = seq_base + 0u;
        if (seq0 < seq_len) {
            out[seq0 * rows + row] = acc0;
        }
        const unsigned seq1 = seq_base + 1u;
        if (seq1 < seq_len) {
            out[seq1 * rows + row] = acc1;
        }
        const unsigned seq2 = seq_base + 2u;
        if (seq2 < seq_len) {
            out[seq2 * rows + row] = acc2;
        }
        const unsigned seq3 = seq_base + 3u;
        if (seq3 < seq_len) {
            out[seq3 * rows + row] = acc3;
        }
    }
}

extern "C" __global__ void rnb_q4k_gemv_batch_q8dot_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    const bool valid = row < rows && seq < seq_len;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const signed char* seq_qs = input_qs + seq * blocks_per_row * 256u;
    const float* seq_ds = input_ds + seq * blocks_per_row * 8u;

    if (valid) {
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
                const signed char* x_qs = seq_qs + b * 256u + j * 32u + elem;
                const int q_pack = rnb_q4_pack4(q_ptr, j);
                const int x_pack = rnb_load_i32_aligned4(x_qs);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = seq_ds[b * 8u + j];
                acc += x_d * ((d * (float)sc) * (float)dot - dmin * (float)mn * (float)x_sum);
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

extern "C" __global__ void rnb_iq4_xs_gemv_batch_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    const bool valid = row < rows && seq < seq_len;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 136u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_row = input + seq * blocks_per_row * 256u;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 136u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned scales_h = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned ib = tid >> 5;
                const unsigned local = tid & 31u;
                const unsigned low =
                    (block[4u + (ib >> 1)] >> (4u * (ib & 1u))) & 0x0fu;
                const unsigned high = ((scales_h >> (2u * ib)) & 0x03u) << 4u;
                const float dl = d * ((float)(low | high) - 32.0f);
                const unsigned q_byte = block[8u + ib * 16u + (local & 15u)];
                const unsigned q = local < 16u ? (q_byte & 0x0fu) : (q_byte >> 4);
                acc += dl * rnb_iq4nl_value(q) * input_row[b * 256u + tid];
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

extern "C" __global__ void rnb_q4k_gemv_batch_q8dot_seq4_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq_base = blockIdx.y * 4u;
    const bool row_valid = row < rows;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (row_valid) {
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
                const int q_pack = rnb_q4_pack4(q_ptr, j);
                const float weight_scale = d * (float)sc;
                const float weight_min = dmin * (float)mn;

                const unsigned seq0 = seq_base + 0u;
                if (seq0 < seq_len) {
                    const unsigned x_base = seq0 * blocks_per_row * 256u + b * 256u + j * 32u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const int x_sum = __dp4a(0x01010101, x_pack, 0);
                    const float x_d = input_ds[seq0 * blocks_per_row * 8u + b * 8u + j];
                    acc0 += x_d * (weight_scale * (float)dot - weight_min * (float)x_sum);
                }
                const unsigned seq1 = seq_base + 1u;
                if (seq1 < seq_len) {
                    const unsigned x_base = seq1 * blocks_per_row * 256u + b * 256u + j * 32u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const int x_sum = __dp4a(0x01010101, x_pack, 0);
                    const float x_d = input_ds[seq1 * blocks_per_row * 8u + b * 8u + j];
                    acc1 += x_d * (weight_scale * (float)dot - weight_min * (float)x_sum);
                }
                const unsigned seq2 = seq_base + 2u;
                if (seq2 < seq_len) {
                    const unsigned x_base = seq2 * blocks_per_row * 256u + b * 256u + j * 32u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const int x_sum = __dp4a(0x01010101, x_pack, 0);
                    const float x_d = input_ds[seq2 * blocks_per_row * 8u + b * 8u + j];
                    acc2 += x_d * (weight_scale * (float)dot - weight_min * (float)x_sum);
                }
                const unsigned seq3 = seq_base + 3u;
                if (seq3 < seq_len) {
                    const unsigned x_base = seq3 * blocks_per_row * 256u + b * 256u + j * 32u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const int x_sum = __dp4a(0x01010101, x_pack, 0);
                    const float x_d = input_ds[seq3 * blocks_per_row * 8u + b * 8u + j];
                    acc3 += x_d * (weight_scale * (float)dot - weight_min * (float)x_sum);
                }
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (row_valid && lane == 0u) {
        const unsigned seq0 = seq_base + 0u;
        if (seq0 < seq_len) {
            out[seq0 * rows + row] = acc0;
        }
        const unsigned seq1 = seq_base + 1u;
        if (seq1 < seq_len) {
            out[seq1 * rows + row] = acc1;
        }
        const unsigned seq2 = seq_base + 2u;
        if (seq2 < seq_len) {
            out[seq2 * rows + row] = acc2;
        }
        const unsigned seq3 = seq_base + 3u;
        if (seq3 < seq_len) {
            out[seq3 * rows + row] = acc3;
        }
    }
}

extern "C" __global__ void rnb_q6k_gemv(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned n = tid >> 7;
        const unsigned rem = tid & 127u;
        const unsigned l = rem & 31u;
        const unsigned is = l >> 4;
        const unsigned ql_base = n * 64u;
        const unsigned qh_base = 128u + n * 32u;
        const unsigned sc_base = 192u + n * 8u;

        unsigned q;
        int sc;
        const unsigned qh = block[qh_base + l];
        if (rem < 32u) {
            q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is]);
        } else if (rem < 64u) {
            q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 2u]);
        } else if (rem < 96u) {
            q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 4u]);
        } else {
            q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 6u]);
        }

        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float y = d * (float)sc * (float)((int)q - 32);
        acc += y * input[b * 256u + tid];
    }

    partial[tid] = acc;
    __syncthreads();

    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = partial[0];
    }
}

// cu19: GPU Q6_K → F16 dequant. Mirrors host dequant_q6k_to_f16 so the result
// is bit-identical to the existing host path. Used by the transient_q6k_f16
// fallback when the resident F16 cache cannot hold the weight, keeping the
// host upload at Q6_K raw bytes (210 bytes/256 elements) instead of f16
// (512 bytes/256 elements) — 2.4× less PCIe traffic per weight, and
// no host-side dequant CPU cost.
extern "C" __global__ void rnb_q6k_dequant_f16(
    __half* __restrict__ out,
    const unsigned char* __restrict__ weights,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256u) {
        return;
    }
    const unsigned cols = blocks_per_row * 256u;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    __half* out_row = out + row * cols;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned n = tid >> 7;
        const unsigned rem = tid & 127u;
        const unsigned l = rem & 31u;
        const unsigned is = l >> 4;
        const unsigned ql_base = n * 64u;
        const unsigned qh_base = 128u + n * 32u;
        const unsigned sc_base = 192u + n * 8u;
        const unsigned qh = block[qh_base + l];
        unsigned q;
        int sc;
        if (rem < 32u) {
            q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is]);
        } else if (rem < 64u) {
            q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 2u]);
        } else if (rem < 96u) {
            q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 4u]);
        } else {
            q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 6u]);
        }
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float y = d * (float)sc * (float)((int)q - 32);
        out_row[b * 256u + tid] = __float2half(y);
    }
}

extern "C" __global__ void rnb_q6k_dequant_f32(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256u) {
        return;
    }
    const unsigned cols = blocks_per_row * 256u;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    float* out_row = out + row * cols;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned n = tid >> 7;
        const unsigned rem = tid & 127u;
        const unsigned l = rem & 31u;
        const unsigned is = l >> 4;
        const unsigned ql_base = n * 64u;
        const unsigned qh_base = 128u + n * 32u;
        const unsigned sc_base = 192u + n * 8u;
        const unsigned qh = block[qh_base + l];
        unsigned q;
        int sc;
        if (rem < 32u) {
            q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is]);
        } else if (rem < 64u) {
            q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 2u]);
        } else if (rem < 96u) {
            q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 4u]);
        } else {
            q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 6u]);
        }
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        out_row[b * 256u + tid] = d * (float)sc * (float)((int)q - 32);
    }
}

extern "C" __global__ void rnb_q6k_embedding_gather_f32(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned token_count) {
    const unsigned token_idx = blockIdx.x;
    const unsigned block_idx = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (token_idx >= token_count || block_idx >= blocks_per_row || tid >= 256u) {
        return;
    }

    const unsigned row = token_ids[token_idx];
    if (row >= rows) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* block = weights + row * row_bytes + block_idx * 210u;
    const unsigned n = tid >> 7;
    const unsigned rem = tid & 127u;
    const unsigned l = rem & 31u;
    const unsigned is = l >> 4;
    const unsigned ql_base = n * 64u;
    const unsigned qh_base = 128u + n * 32u;
    const unsigned sc_base = 192u + n * 8u;
    const unsigned qh = block[qh_base + l];
    unsigned q;
    int sc;
    if (rem < 32u) {
        q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
        sc = (int)((signed char)block[sc_base + is]);
    } else if (rem < 64u) {
        q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
        sc = (int)((signed char)block[sc_base + is + 2u]);
    } else if (rem < 96u) {
        q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
        sc = (int)((signed char)block[sc_base + is + 4u]);
    } else {
        q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
        sc = (int)((signed char)block[sc_base + is + 6u]);
    }
    const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    out[token_idx * blocks_per_row * 256u + block_idx * 256u + tid] =
        d * (float)sc * (float)((int)q - 32);
}

extern "C" __global__ void rnb_q6k_gemv_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    __shared__ float input_tile[256];

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        input_tile[threadIdx.x] = input[b * 256u + threadIdx.x];
        __syncthreads();
        if (valid) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned n = tid >> 7;
                const unsigned rem = tid & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;

                unsigned q;
                int sc;
                const unsigned qh = block[qh_base + l];
                if (rem < 32u) {
                    q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is]);
                } else if (rem < 64u) {
                    q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 2u]);
                } else if (rem < 96u) {
                    q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 4u]);
                } else {
                    q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 6u]);
                }
                const float y = d * (float)sc * (float)((int)q - 32);
                acc += y * input_tile[tid];
            }
        }
        __syncthreads();
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q6k_gemv_q8dot_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const unsigned n = elem >> 7;
                const unsigned rem = elem & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;
                int sc;
                if (rem < 32u) {
                    sc = (int)((signed char)block[sc_base + is]);
                } else if (rem < 64u) {
                    sc = (int)((signed char)block[sc_base + is + 2u]);
                } else if (rem < 96u) {
                    sc = (int)((signed char)block[sc_base + is + 4u]);
                } else {
                    sc = (int)((signed char)block[sc_base + is + 6u]);
                }

                int q_pack = 0;
                const int x_pack = rnb_load_i32_aligned4(input_qs + b * 256u + elem);
                #pragma unroll
                for (unsigned k = 0; k < 4u; ++k) {
                    const unsigned e = elem + k;
                    const unsigned e_rem = e & 127u;
                    const unsigned e_l = e_rem & 31u;
                    const unsigned qh = block[qh_base + e_l];
                    unsigned q;
                    if (e_rem < 32u) {
                        q = (block[ql_base + e_l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    } else if (e_rem < 64u) {
                        q = (block[ql_base + e_l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                    } else if (e_rem < 96u) {
                        q = (block[ql_base + e_l] >> 4) | (((qh >> 4) & 3u) << 4);
                    } else {
                        q = (block[ql_base + e_l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                    }
                    q_pack |= (int)(q & 0xffu) << (8u * k);
                }
                const int dot = __dp4a(q_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = input_ds[b * 8u + (elem >> 5)];
                acc += x_d * d * (float)sc * (float)(dot - 32 * x_sum);
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q6k_gemv_batch_q8dot_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    const bool valid = row < rows && seq < seq_len;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const signed char* seq_qs = input_qs + seq * blocks_per_row * 256u;
    const float* seq_ds = input_ds + seq * blocks_per_row * 8u;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const unsigned n = elem >> 7;
                const unsigned rem = elem & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;
                int sc;
                if (rem < 32u) {
                    sc = (int)((signed char)block[sc_base + is]);
                } else if (rem < 64u) {
                    sc = (int)((signed char)block[sc_base + is + 2u]);
                } else if (rem < 96u) {
                    sc = (int)((signed char)block[sc_base + is + 4u]);
                } else {
                    sc = (int)((signed char)block[sc_base + is + 6u]);
                }

                int q_pack = 0;
                const int x_pack = rnb_load_i32_aligned4(seq_qs + b * 256u + elem);
                #pragma unroll
                for (unsigned k = 0; k < 4u; ++k) {
                    const unsigned e = elem + k;
                    const unsigned e_rem = e & 127u;
                    const unsigned e_l = e_rem & 31u;
                    const unsigned qh = block[qh_base + e_l];
                    unsigned q;
                    if (e_rem < 32u) {
                        q = (block[ql_base + e_l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    } else if (e_rem < 64u) {
                        q = (block[ql_base + e_l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                    } else if (e_rem < 96u) {
                        q = (block[ql_base + e_l] >> 4) | (((qh >> 4) & 3u) << 4);
                    } else {
                        q = (block[ql_base + e_l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                    }
                    q_pack |= (int)(q & 0xffu) << (8u * k);
                }
                const int dot = __dp4a(q_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = seq_ds[b * 8u + (elem >> 5)];
                acc += x_d * d * (float)sc * (float)(dot - 32 * x_sum);
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

extern "C" __global__ void rnb_q6k_packed_q8dot_warp8(
    float* __restrict__ out,
    const signed char* __restrict__ packed_qs,
    const unsigned short* __restrict__ packed_d_super,
    const signed char* __restrict__ packed_sub_scale,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = 0.0f;
    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned packed_block = row * blocks_per_row + b;
            const signed char* q_block = packed_qs + packed_block * 256u;
            const float d_super = __half2float(__ushort_as_half(packed_d_super[packed_block]));
            const signed char* sub_scale_block = packed_sub_scale + packed_block * 16u;
            const signed char* x_block = input_qs + b * 256u;
            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const int q_pack = rnb_load_i32_aligned4(q_block + elem);
                const int x_pack = rnb_load_i32_aligned4(x_block + elem);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const float scale = d_super * (float)sub_scale_block[elem >> 4];
                const float x_d = input_ds[b * 8u + (elem >> 5)];
                acc += x_d * scale * (float)dot;
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q6k_packed_batch_q8dot_warp8(
    float* __restrict__ out,
    const signed char* __restrict__ packed_qs,
    const unsigned short* __restrict__ packed_d_super,
    const signed char* __restrict__ packed_sub_scale,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    const bool valid = row < rows && seq < seq_len;

    float acc = 0.0f;
    const signed char* seq_qs = input_qs + seq * blocks_per_row * 256u;
    const float* seq_ds = input_ds + seq * blocks_per_row * 8u;
    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned packed_block = row * blocks_per_row + b;
            const signed char* q_block = packed_qs + packed_block * 256u;
            const float d_super = __half2float(__ushort_as_half(packed_d_super[packed_block]));
            const signed char* sub_scale_block = packed_sub_scale + packed_block * 16u;
            const signed char* x_block = seq_qs + b * 256u;
            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const int q_pack = rnb_load_i32_aligned4(q_block + elem);
                const int x_pack = rnb_load_i32_aligned4(x_block + elem);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const float scale = d_super * (float)sub_scale_block[elem >> 4];
                const float x_d = seq_ds[b * 8u + (elem >> 5)];
                acc += x_d * scale * (float)dot;
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

extern "C" __global__ void rnb_q6k_packed_batch_q8dot_seq4_warp8(
    float* __restrict__ out,
    const signed char* __restrict__ packed_qs,
    const unsigned short* __restrict__ packed_d_super,
    const signed char* __restrict__ packed_sub_scale,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq_base = blockIdx.y * 4u;
    const bool row_valid = row < rows;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    if (row_valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned packed_block = row * blocks_per_row + b;
            const signed char* q_block = packed_qs + packed_block * 256u;
            const float d_super = __half2float(__ushort_as_half(packed_d_super[packed_block]));
            const signed char* sub_scale_block = packed_sub_scale + packed_block * 16u;
            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const int q_pack = rnb_load_i32_aligned4(q_block + elem);
                const float scale = d_super * (float)sub_scale_block[elem >> 4];
                const unsigned seq0 = seq_base + 0u;
                if (seq0 < seq_len) {
                    const unsigned x_base = seq0 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq0 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc0 += x_d * scale * (float)dot;
                }
                const unsigned seq1 = seq_base + 1u;
                if (seq1 < seq_len) {
                    const unsigned x_base = seq1 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq1 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc1 += x_d * scale * (float)dot;
                }
                const unsigned seq2 = seq_base + 2u;
                if (seq2 < seq_len) {
                    const unsigned x_base = seq2 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq2 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc2 += x_d * scale * (float)dot;
                }
                const unsigned seq3 = seq_base + 3u;
                if (seq3 < seq_len) {
                    const unsigned x_base = seq3 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq3 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc3 += x_d * scale * (float)dot;
                }
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (row_valid && lane == 0u) {
        const unsigned seq0 = seq_base + 0u;
        if (seq0 < seq_len) {
            out[seq0 * rows + row] = acc0;
        }
        const unsigned seq1 = seq_base + 1u;
        if (seq1 < seq_len) {
            out[seq1 * rows + row] = acc1;
        }
        const unsigned seq2 = seq_base + 2u;
        if (seq2 < seq_len) {
            out[seq2 * rows + row] = acc2;
        }
        const unsigned seq3 = seq_base + 3u;
        if (seq3 < seq_len) {
            out[seq3 * rows + row] = acc3;
        }
    }
}

extern "C" __global__ void rnb_q6k_packed_batch_q8dot_seq4_warp4(
    float* __restrict__ out,
    const signed char* __restrict__ packed_qs,
    const unsigned short* __restrict__ packed_d_super,
    const signed char* __restrict__ packed_sub_scale,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned lane = threadIdx.x;
    const unsigned warp = threadIdx.y;
    const unsigned row = blockIdx.x;
    const unsigned seq_base = blockIdx.y * 4u;
    const bool row_valid = row < rows;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    if (row_valid) {
        for (unsigned b = warp; b < blocks_per_row; b += 4u) {
            const unsigned packed_block = row * blocks_per_row + b;
            const signed char* q_block = packed_qs + packed_block * 256u;
            const float d_super = __half2float(__ushort_as_half(packed_d_super[packed_block]));
            const signed char* sub_scale_block = packed_sub_scale + packed_block * 16u;
            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const int q_pack = rnb_load_i32_aligned4(q_block + elem);
                const float scale = d_super * (float)sub_scale_block[elem >> 4];
                const unsigned seq0 = seq_base + 0u;
                if (seq0 < seq_len) {
                    const unsigned x_base = seq0 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq0 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc0 += x_d * scale * (float)dot;
                }
                const unsigned seq1 = seq_base + 1u;
                if (seq1 < seq_len) {
                    const unsigned x_base = seq1 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq1 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc1 += x_d * scale * (float)dot;
                }
                const unsigned seq2 = seq_base + 2u;
                if (seq2 < seq_len) {
                    const unsigned x_base = seq2 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq2 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc2 += x_d * scale * (float)dot;
                }
                const unsigned seq3 = seq_base + 3u;
                if (seq3 < seq_len) {
                    const unsigned x_base = seq3 * blocks_per_row * 256u + b * 256u + elem;
                    const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                    const int dot = __dp4a(q_pack, x_pack, 0);
                    const float x_d = input_ds[seq3 * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                    acc3 += x_d * scale * (float)dot;
                }
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    __shared__ float warp_sums[16];
    if (lane == 0u) {
        warp_sums[warp] = acc0;
        warp_sums[4u + warp] = acc1;
        warp_sums[8u + warp] = acc2;
        warp_sums[12u + warp] = acc3;
    }
    __syncthreads();

    if (warp == 0u) {
        float row_acc0 = lane < 4u ? warp_sums[lane] : 0.0f;
        float row_acc1 = lane < 4u ? warp_sums[4u + lane] : 0.0f;
        float row_acc2 = lane < 4u ? warp_sums[8u + lane] : 0.0f;
        float row_acc3 = lane < 4u ? warp_sums[12u + lane] : 0.0f;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            row_acc0 += __shfl_down_sync(0xffffffffu, row_acc0, offset);
            row_acc1 += __shfl_down_sync(0xffffffffu, row_acc1, offset);
            row_acc2 += __shfl_down_sync(0xffffffffu, row_acc2, offset);
            row_acc3 += __shfl_down_sync(0xffffffffu, row_acc3, offset);
        }
        if (row_valid && lane == 0u) {
            const unsigned seq0 = seq_base + 0u;
            if (seq0 < seq_len) {
                out[seq0 * rows + row] = row_acc0;
            }
            const unsigned seq1 = seq_base + 1u;
            if (seq1 < seq_len) {
                out[seq1 * rows + row] = row_acc1;
            }
            const unsigned seq2 = seq_base + 2u;
            if (seq2 < seq_len) {
                out[seq2 * rows + row] = row_acc2;
            }
            const unsigned seq3 = seq_base + 3u;
            if (seq3 < seq_len) {
                out[seq3 * rows + row] = row_acc3;
            }
        }
    }
}

extern "C" __global__ void rnb_q6k_packed_batch_q8dot_seq8_warp4(
    float* __restrict__ out,
    const signed char* __restrict__ packed_qs,
    const unsigned short* __restrict__ packed_d_super,
    const signed char* __restrict__ packed_sub_scale,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned lane = threadIdx.x;
    const unsigned warp = threadIdx.y;
    const unsigned row = blockIdx.x;
    const unsigned seq_base = blockIdx.y * 8u;
    const bool row_valid = row < rows;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (row_valid) {
        for (unsigned b = warp; b < blocks_per_row; b += 4u) {
            const unsigned packed_block = row * blocks_per_row + b;
            const signed char* q_block = packed_qs + packed_block * 256u;
            const float d_super = __half2float(__ushort_as_half(packed_d_super[packed_block]));
            const signed char* sub_scale_block = packed_sub_scale + packed_block * 16u;
            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const int q_pack = rnb_load_i32_aligned4(q_block + elem);
                const float scale = d_super * (float)sub_scale_block[elem >> 4];
#pragma unroll
                for (unsigned s = 0; s < 8u; ++s) {
                    const unsigned seq = seq_base + s;
                    if (seq < seq_len) {
                        const unsigned x_base = seq * blocks_per_row * 256u + b * 256u + elem;
                        const int x_pack = rnb_load_i32_aligned4(input_qs + x_base);
                        const int dot = __dp4a(q_pack, x_pack, 0);
                        const float x_d =
                            input_ds[seq * blocks_per_row * 8u + b * 8u + (elem >> 5)];
                        acc[s] += x_d * scale * (float)dot;
                    }
                }
            }
        }
    }

#pragma unroll
    for (unsigned s = 0; s < 8u; ++s) {
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            acc[s] += __shfl_down_sync(0xffffffffu, acc[s], offset);
        }
    }

    __shared__ float warp_sums[32];
#pragma unroll
    for (unsigned s = 0; s < 8u; ++s) {
        if (lane == 0u) {
            warp_sums[s * 4u + warp] = acc[s];
        }
    }
    __syncthreads();

    if (warp == 0u) {
#pragma unroll
        for (unsigned s = 0; s < 8u; ++s) {
            float row_acc = lane < 4u ? warp_sums[s * 4u + lane] : 0.0f;
            for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
                row_acc += __shfl_down_sync(0xffffffffu, row_acc, offset);
            }
            const unsigned seq = seq_base + s;
            if (row_valid && seq < seq_len && lane == 0u) {
                out[seq * rows + row] = row_acc;
            }
        }
    }
}

extern "C" __global__ void rnb_q6k_packed_q8dot_warp4(
    float* __restrict__ out,
    const signed char* __restrict__ packed_qs,
    const unsigned short* __restrict__ packed_d_super,
    const signed char* __restrict__ packed_sub_scale,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned row = blockIdx.x;
    const bool valid = row < rows;

    float acc = 0.0f;
    if (valid) {
        for (unsigned b = warp; b < blocks_per_row; b += 4u) {
            const unsigned packed_block = row * blocks_per_row + b;
            const signed char* q_block = packed_qs + packed_block * 256u;
            const float d_super = __half2float(__ushort_as_half(packed_d_super[packed_block]));
            const signed char* sub_scale_block = packed_sub_scale + packed_block * 16u;
            const signed char* x_block = input_qs + b * 256u;
            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned elem = chunk * 4u;
                const int q_pack = rnb_load_i32_aligned4(q_block + elem);
                const int x_pack = rnb_load_i32_aligned4(x_block + elem);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const float scale = d_super * (float)sub_scale_block[elem >> 4];
                const float x_d = input_ds[b * 8u + (elem >> 5)];
                acc += x_d * scale * (float)dot;
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    __shared__ float warp_sums[4];
    if (lane == 0u) {
        warp_sums[warp] = acc;
    }
    __syncthreads();

    if (warp == 0u) {
        float row_acc = lane < 4u ? warp_sums[lane] : 0.0f;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            row_acc += __shfl_down_sync(0xffffffffu, row_acc, offset);
        }
        if (valid && lane == 0u) {
            out[row] = row_acc;
        }
    }
}

extern "C" __global__ void rnb_q6k_gemv_argmax_warp8(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ block_values,
    unsigned* __restrict__ block_indices,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;

    float acc = -3.4028234663852886e38f;
    __shared__ float input_tile[256];
    const bool valid = row < rows;
    if (valid) {
        acc = 0.0f;
    }
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        input_tile[threadIdx.x] = input[b * 256u + threadIdx.x];
        __syncthreads();
        if (valid) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned n = tid >> 7;
                const unsigned rem = tid & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;

                unsigned q;
                int sc;
                const unsigned qh = block[qh_base + l];
                if (rem < 32u) {
                    q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is]);
                } else if (rem < 64u) {
                    q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 2u]);
                } else if (rem < 96u) {
                    q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 4u]);
                } else {
                    q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 6u]);
                }
                const float y = d * (float)sc * (float)((int)q - 32);
                acc += y * input_tile[tid];
            }
        }
        __syncthreads();
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    __shared__ float vals[8];
    __shared__ unsigned inds[8];
    if (lane == 0u) {
        vals[warp] = acc;
        inds[warp] = row;
    }
    __syncthreads();

    if (threadIdx.x < 8u) {
        float best_val = vals[threadIdx.x];
        unsigned best_idx = inds[threadIdx.x];
        for (unsigned i = threadIdx.x + 4u; i < 8u; i += 4u) {
            const float other_val = vals[i];
            const unsigned other_idx = inds[i];
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        vals[threadIdx.x] = best_val;
        inds[threadIdx.x] = best_idx;
    }
    __syncthreads();

    if (threadIdx.x < 4u) {
        const float other_val = vals[threadIdx.x + 4u];
        const unsigned other_idx = inds[threadIdx.x + 4u];
        if (other_val > vals[threadIdx.x]
            || (other_val == vals[threadIdx.x] && other_idx < inds[threadIdx.x])) {
            vals[threadIdx.x] = other_val;
            inds[threadIdx.x] = other_idx;
        }
    }
    __syncthreads();

    if (threadIdx.x < 2u) {
        const float other_val = vals[threadIdx.x + 2u];
        const unsigned other_idx = inds[threadIdx.x + 2u];
        if (other_val > vals[threadIdx.x]
            || (other_val == vals[threadIdx.x] && other_idx < inds[threadIdx.x])) {
            vals[threadIdx.x] = other_val;
            inds[threadIdx.x] = other_idx;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        const float other_val = vals[1];
        const unsigned other_idx = inds[1];
        if (other_val > vals[0] || (other_val == vals[0] && other_idx < inds[0])) {
            vals[0] = other_val;
            inds[0] = other_idx;
        }
        block_values[blockIdx.x] = vals[0];
        block_indices[blockIdx.x] = inds[0];
    }
}

extern "C" __global__ void rnb_q6k_gemv_argmax_warp8_batched(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ inputs,
    float* __restrict__ block_values,
    unsigned* __restrict__ block_indices,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned input_stride,
    unsigned token_count,
    unsigned block_count) {
    const unsigned token = blockIdx.y;
    if (token >= token_count) {
        return;
    }
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const float* input = inputs + token * input_stride;

    float acc = -3.4028234663852886e38f;
    __shared__ float input_tile[256];
    const bool valid = row < rows;
    if (valid) {
        acc = 0.0f;
    }
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        input_tile[threadIdx.x] = input[b * 256u + threadIdx.x];
        __syncthreads();
        if (valid) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned n = tid >> 7;
                const unsigned rem = tid & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;

                unsigned q;
                int sc;
                const unsigned qh = block[qh_base + l];
                if (rem < 32u) {
                    q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is]);
                } else if (rem < 64u) {
                    q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 2u]);
                } else if (rem < 96u) {
                    q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 4u]);
                } else {
                    q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 6u]);
                }
                const float y = d * (float)sc * (float)((int)q - 32);
                acc += y * input_tile[tid];
            }
        }
        __syncthreads();
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    __shared__ float vals[8];
    __shared__ unsigned inds[8];
    if (lane == 0u) {
        vals[warp] = acc;
        inds[warp] = row;
    }
    __syncthreads();

    if (threadIdx.x < 8u) {
        float best_val = vals[threadIdx.x];
        unsigned best_idx = inds[threadIdx.x];
        for (unsigned i = threadIdx.x + 4u; i < 8u; i += 4u) {
            const float other_val = vals[i];
            const unsigned other_idx = inds[i];
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        vals[threadIdx.x] = best_val;
        inds[threadIdx.x] = best_idx;
    }
    __syncthreads();

    if (threadIdx.x < 4u) {
        const float other_val = vals[threadIdx.x + 4u];
        const unsigned other_idx = inds[threadIdx.x + 4u];
        if (other_val > vals[threadIdx.x]
            || (other_val == vals[threadIdx.x] && other_idx < inds[threadIdx.x])) {
            vals[threadIdx.x] = other_val;
            inds[threadIdx.x] = other_idx;
        }
    }
    __syncthreads();

    if (threadIdx.x < 2u) {
        const float other_val = vals[threadIdx.x + 2u];
        const unsigned other_idx = inds[threadIdx.x + 2u];
        if (other_val > vals[threadIdx.x]
            || (other_val == vals[threadIdx.x] && other_idx < inds[threadIdx.x])) {
            vals[threadIdx.x] = other_val;
            inds[threadIdx.x] = other_idx;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        const float other_val = vals[1];
        const unsigned other_idx = inds[1];
        if (other_val > vals[0] || (other_val == vals[0] && other_idx < inds[0])) {
            vals[0] = other_val;
            inds[0] = other_idx;
        }
        const unsigned out = token * block_count + blockIdx.x;
        block_values[out] = vals[0];
        block_indices[out] = inds[0];
    }
}

extern "C" __global__ void rnb_q5k_gemv(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 176u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

        const unsigned j = tid >> 5;
        unsigned sc;
        unsigned mn;
        if (j < 4) {
            sc = block[4 + j] & 63u;
            mn = block[4 + j + 4] & 63u;
        } else {
            sc = (block[4 + j + 4] & 0x0fu) | ((block[4 + j - 4] >> 6) << 4);
            mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
        }

        const unsigned local = tid & 63u;
        const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
        unsigned q = block[48 + q_index];
        q = local < 32u ? (q & 0x0fu) : (q >> 4);
        const unsigned high = (block[16 + (tid & 31u)] >> (tid >> 5)) & 1u;
        q |= high << 4;

        const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
        acc += y * input[b * 256u + tid];
    }

    partial[tid] = acc;
    __syncthreads();

    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = partial[0];
    }
}

extern "C" __global__ void rnb_argmax_f32(
    const float* __restrict__ values,
    float* __restrict__ block_values,
    unsigned* __restrict__ block_indices,
    unsigned len) {
    const unsigned tid = threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;

    float best_val = -3.4028234663852886e38f;
    unsigned best_idx = 0;
    while (idx < len) {
        const float v = values[idx];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
        idx += stride;
    }

    __shared__ float vals[256];
    __shared__ unsigned inds[256];
    vals[tid] = best_val;
    inds[tid] = best_idx;
    __syncthreads();

    for (unsigned step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            const float other_val = vals[tid + step];
            const unsigned other_idx = inds[tid + step];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < inds[tid])) {
                vals[tid] = other_val;
                inds[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_values[blockIdx.x] = vals[0];
        block_indices[blockIdx.x] = inds[0];
    }
}

extern "C" __global__ void rnb_argmax_pairs_f32(
    const float* __restrict__ values,
    const unsigned* __restrict__ indices,
    float* __restrict__ block_values,
    unsigned* __restrict__ block_indices,
    unsigned len) {
    const unsigned tid = threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;

    float best_val = -3.4028234663852886e38f;
    unsigned best_idx = 0;
    while (idx < len) {
        const float v = values[idx];
        const unsigned original_idx = indices[idx];
        if (v > best_val || (v == best_val && original_idx < best_idx)) {
            best_val = v;
            best_idx = original_idx;
        }
        idx += stride;
    }

    __shared__ float vals[256];
    __shared__ unsigned inds[256];
    vals[tid] = best_val;
    inds[tid] = best_idx;
    __syncthreads();

    for (unsigned step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            const float other_val = vals[tid + step];
            const unsigned other_idx = inds[tid + step];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < inds[tid])) {
                vals[tid] = other_val;
                inds[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_values[blockIdx.x] = vals[0];
        block_indices[blockIdx.x] = inds[0];
    }
}

extern "C" __global__ void rnb_argmax_pairs_f32_batched(
    const float* __restrict__ values,
    const unsigned* __restrict__ indices,
    float* __restrict__ block_values,
    unsigned* __restrict__ block_indices,
    unsigned len,
    unsigned token_count,
    unsigned input_stride,
    unsigned output_stride) {
    const unsigned token = blockIdx.y;
    if (token >= token_count) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;
    const unsigned input_base = token * input_stride;

    float best_val = -3.4028234663852886e38f;
    unsigned best_idx = 0;
    while (idx < len) {
        const float v = values[input_base + idx];
        const unsigned original_idx = indices[input_base + idx];
        if (v > best_val || (v == best_val && original_idx < best_idx)) {
            best_val = v;
            best_idx = original_idx;
        }
        idx += stride;
    }

    __shared__ float vals[256];
    __shared__ unsigned inds[256];
    vals[tid] = best_val;
    inds[tid] = best_idx;
    __syncthreads();

    for (unsigned step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            const float other_val = vals[tid + step];
            const unsigned other_idx = inds[tid + step];
            if (other_val > vals[tid] || (other_val == vals[tid] && other_idx < inds[tid])) {
                vals[tid] = other_val;
                inds[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const unsigned out = token * output_stride + blockIdx.x;
        block_values[out] = vals[0];
        block_indices[out] = inds[0];
    }
}

extern "C" __global__ void rnb_q6k_gemv_batch(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned row = blockIdx.x;
    const unsigned seq = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || seq >= seq_len || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_row = input + seq * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned n = tid >> 7;
        const unsigned rem = tid & 127u;
        const unsigned l = rem & 31u;
        const unsigned is = l >> 4;
        const unsigned ql_base = n * 64u;
        const unsigned qh_base = 128u + n * 32u;
        const unsigned sc_base = 192u + n * 8u;

        unsigned q;
        int sc;
        const unsigned qh = block[qh_base + l];
        if (rem < 32u) {
            q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is]);
        } else if (rem < 64u) {
            q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 2u]);
        } else if (rem < 96u) {
            q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 4u]);
        } else {
            q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
            sc = (int)((signed char)block[sc_base + is + 6u]);
        }

        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float y = d * (float)sc * (float)((int)q - 32);
        acc += y * input_row[b * 256u + tid];
    }

    partial[tid] = acc;
    __syncthreads();

    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[seq * rows + row] = partial[0];
    }
}

extern "C" __global__ void rnb_q6k_gemv_batch_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    const bool valid = row < rows && seq < seq_len;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_row = input + seq * blocks_per_row * 256u;
    __shared__ float input_tile[256];

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        input_tile[threadIdx.x] = input_row[b * 256u + threadIdx.x];
        __syncthreads();
        if (valid) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned n = tid >> 7;
                const unsigned rem = tid & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;

                unsigned q;
                int sc;
                const unsigned qh = block[qh_base + l];
                if (rem < 32u) {
                    q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is]);
                } else if (rem < 64u) {
                    q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 2u]);
                } else if (rem < 96u) {
                    q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 4u]);
                } else {
                    q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 6u]);
                }
                const float y = d * (float)sc * (float)((int)q - 32);
                acc += y * input_tile[tid];
            }
        }
        __syncthreads();
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

extern "C" __global__ void rnb_q6k_gemv_batch_seq2_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input0 = input;
    const float* input1 = input + blocks_per_row * 256u;
    __shared__ float input_tile0[256];
    __shared__ float input_tile1[256];

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        input_tile0[threadIdx.x] = input0[b * 256u + threadIdx.x];
        input_tile1[threadIdx.x] = input1[b * 256u + threadIdx.x];
        __syncthreads();
        if (valid) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned n = tid >> 7;
                const unsigned rem = tid & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;

                unsigned q;
                int sc;
                const unsigned qh = block[qh_base + l];
                if (rem < 32u) {
                    q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is]);
                } else if (rem < 64u) {
                    q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 2u]);
                } else if (rem < 96u) {
                    q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 4u]);
                } else {
                    q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                    sc = (int)((signed char)block[sc_base + is + 6u]);
                }
                const float y = d * (float)sc * (float)((int)q - 32);
                acc0 += y * input_tile0[tid];
                acc1 += y * input_tile1[tid];
            }
        }
        __syncthreads();
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (valid && lane == 0u) {
        out[row] = acc0;
        out[rows + row] = acc1;
    }
}

extern "C" __global__ void rnb_q5k_gemv_batch(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned row = blockIdx.x;
    const unsigned seq = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || seq >= seq_len || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_row = input + seq * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 176u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

        const unsigned j = tid >> 5;
        unsigned sc;
        unsigned mn;
        if (j < 4) {
            sc = block[4 + j] & 63u;
            mn = block[4 + j + 4] & 63u;
        } else {
            sc = (block[4 + j + 4] & 0x0fu) | ((block[4 + j - 4] >> 6) << 4);
            mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
        }

        const unsigned local = tid & 63u;
        const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
        unsigned q = block[48 + q_index];
        q = local < 32u ? (q & 0x0fu) : (q >> 4);
        const unsigned high = (block[16 + (tid & 31u)] >> (tid >> 5)) & 1u;
        q |= high << 4;

        const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
        acc += y * input_row[b * 256u + tid];
    }

    partial[tid] = acc;
    __syncthreads();

    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[seq * rows + row] = partial[0];
    }
}

extern "C" __global__ void rnb_silu_mul_inplace(
    float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    const float x = gate[i];
    gate[i] = (x / (1.0f + expf(-x))) * up[i];
}

extern "C" __global__ void rnb_silu_mul_group4_pack_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ packed,
    const unsigned* __restrict__ group_meta,
    unsigned groups,
    unsigned blocks_per_row) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned elems_per_group = blocks_per_row * 256u * 4u;
    const unsigned group = elems_per_group == 0u ? 0u : idx / elems_per_group;
    if (group >= groups) {
        return;
    }

    const unsigned local_idx = idx - group * elems_per_group;
    const unsigned local_slot = local_idx & 3u;
    const unsigned elem_idx = local_idx >> 2;
    const unsigned b = elem_idx >> 8;
    const unsigned elem = elem_idx & 255u;
    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u || local_slot >= group_len) {
        return;
    }

    const unsigned src = (slot_start + local_slot) * blocks_per_row * 256u + b * 256u + elem;
    const float x = gate[src];
    packed[idx] = (x / (1.0f + expf(-x))) * up[src];
}

extern "C" __global__ void rnb_gelu_mul_inplace(
    float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    const float x = gate[i];
    const float x3 = x * x * x;
    const float c = 0.7978845608028654f;
    const float gelu = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
    gate[i] = gelu * up[i];
}

extern "C" __global__ void rnb_gelu_mul_q8_1(
    float* __restrict__ gate,
    const float* __restrict__ up,
    signed char* __restrict__ out_qs,
    float* __restrict__ out_ds,
    unsigned len) {
    const unsigned block = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const unsigned i = block * 32u + lane;
    __shared__ float values[32];
    __shared__ float abs_values[32];
    float value = 0.0f;
    float abs_value = 0.0f;
    if (i < len) {
        const float x = gate[i];
        const float x3 = x * x * x;
        const float c = 0.7978845608028654f;
        const float gelu = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
        value = gelu * up[i];
        abs_value = fabsf(value);
        gate[i] = value;
    }
    values[lane] = value;
    abs_values[lane] = abs_value;
    __syncthreads();
    for (unsigned stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride && abs_values[lane + stride] > abs_values[lane]) {
            abs_values[lane] = abs_values[lane + stride];
        }
        __syncthreads();
    }
    const float max_abs = abs_values[0];
    const float d = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
    if (lane == 0u) {
        out_ds[block] = d;
    }
    if (i < len) {
        int q = 0;
        if (d > 0.0f) {
            q = (int)nearbyintf(values[lane] / d);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
        }
        out_qs[i] = (signed char)q;
    }
}

extern "C" __global__ void rnb_silu_mul_q8_1(
    float* __restrict__ gate,
    const float* __restrict__ up,
    signed char* __restrict__ out_qs,
    float* __restrict__ out_ds,
    unsigned len) {
    const unsigned block = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const unsigned i = block * 32u + lane;
    __shared__ float values[32];
    __shared__ float abs_values[32];
    float value = 0.0f;
    float abs_value = 0.0f;
    if (i < len) {
        const float x = gate[i];
        value = (x / (1.0f + expf(-x))) * up[i];
        abs_value = fabsf(value);
        gate[i] = value;
    }
    values[lane] = value;
    abs_values[lane] = abs_value;
    __syncthreads();
    for (unsigned stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride && abs_values[lane + stride] > abs_values[lane]) {
            abs_values[lane] = abs_values[lane + stride];
        }
        __syncthreads();
    }
    const float max_abs = abs_values[0];
    const float d = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
    if (lane == 0u) {
        out_ds[block] = d;
    }
    if (i < len) {
        int q = 0;
        if (d > 0.0f) {
            q = (int)nearbyintf(values[lane] / d);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
        }
        out_qs[i] = (signed char)q;
    }
}

extern "C" __global__ void rnb_quantize_q8_1_by_32(
    const float* __restrict__ input,
    signed char* __restrict__ out_qs,
    float* __restrict__ out_ds,
    unsigned chunks) {
    const unsigned chunk = blockIdx.x;
    const unsigned lane = threadIdx.x;
    if (chunk >= chunks || lane >= 32u) {
        return;
    }

    __shared__ float values[32];
    __shared__ float abs_values[32];
    const unsigned i = chunk * 32u + lane;
    const float value = input[i];
    values[lane] = value;
    abs_values[lane] = fabsf(value);
    __syncthreads();

    for (unsigned stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride && abs_values[lane + stride] > abs_values[lane]) {
            abs_values[lane] = abs_values[lane + stride];
        }
        __syncthreads();
    }

    const float max_abs = abs_values[0];
    const float d = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
    if (lane == 0u) {
        out_ds[chunk] = d;
    }
    int q = 0;
    if (d > 0.0f) {
        q = (int)nearbyintf(values[lane] / d);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    out_qs[i] = (signed char)q;
}

extern "C" __global__ void rnb_rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    unsigned len,
    unsigned unit_offset) {
    const unsigned tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = unit_offset != 0u ? (1.0f + weight[i]) : weight[i];
        output[i] = input[i] * inv_rms * scale;
    }
}

extern "C" __global__ void rnb_rms_norm_add_f32_inplace(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ residual,
    float eps,
    unsigned len,
    unsigned unit_offset) {
    const unsigned tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = unit_offset != 0u ? (1.0f + weight[i]) : weight[i];
        residual[i] += input[i] * inv_rms * scale;
    }
}

extern "C" __global__ void rnb_rms_norm_add_then_rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ post_weight,
    float* __restrict__ residual,
    const float* __restrict__ pre_weight,
    float* __restrict__ output,
    float eps,
    unsigned len,
    unsigned post_unit_offset,
    unsigned pre_unit_offset) {
    const unsigned tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float post_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale =
            post_unit_offset != 0u ? (1.0f + post_weight[i]) : post_weight[i];
        residual[i] += input[i] * post_inv_rms * scale;
    }
    __syncthreads();

    float residual_sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = residual[i];
        residual_sum += v * v;
    }
    partial[tid] = residual_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float pre_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = pre_unit_offset != 0u ? (1.0f + pre_weight[i]) : pre_weight[i];
        output[i] = residual[i] * pre_inv_rms * scale;
    }
}

extern "C" __global__ void rnb_rms_norm_add_then_rms_norm_q8_1_f32(
    const float* __restrict__ input,
    const float* __restrict__ post_weight,
    float* __restrict__ residual,
    const float* __restrict__ pre_weight,
    float* __restrict__ output,
    signed char* __restrict__ out_qs,
    float* __restrict__ out_ds,
    float eps,
    unsigned len,
    unsigned post_unit_offset,
    unsigned pre_unit_offset) {
    const unsigned tid = threadIdx.x;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float post_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale =
            post_unit_offset != 0u ? (1.0f + post_weight[i]) : post_weight[i];
        residual[i] += input[i] * post_inv_rms * scale;
    }
    __syncthreads();

    float residual_sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = residual[i];
        residual_sum += v * v;
    }
    partial[tid] = residual_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float pre_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = pre_unit_offset != 0u ? (1.0f + pre_weight[i]) : pre_weight[i];
        output[i] = residual[i] * pre_inv_rms * scale;
    }
    __syncthreads();

    const unsigned chunks = len >> 5;
    for (unsigned chunk = tid; chunk < chunks; chunk += blockDim.x) {
        const unsigned base = chunk << 5;
        float max_abs = 0.0f;
        for (unsigned lane = 0u; lane < 32u; ++lane) {
            const float value = output[base + lane];
            const float abs_value = fabsf(value);
            max_abs = abs_value > max_abs ? abs_value : max_abs;
        }
        const float d = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
        out_ds[chunk] = d;
        for (unsigned lane = 0u; lane < 32u; ++lane) {
            int q = 0;
            if (d > 0.0f) {
                q = (int)nearbyintf(output[base + lane] / d);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
            }
            out_qs[base + lane] = (signed char)q;
        }
    }
}

extern "C" __global__ void rnb_rms_norm_rows_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    unsigned rows,
    unsigned len,
    unsigned unit_offset) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned base = row * len;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = unit_offset != 0u ? (1.0f + weight[i]) : weight[i];
        output[base + i] = input[base + i] * inv_rms * scale;
    }
}

extern "C" __global__ void rnb_rms_norm_rows_f32_serial(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    unsigned rows,
    unsigned len,
    unsigned unit_offset) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned base = row * len;
    __shared__ float rms_shared;
    if (tid == 0u) {
        float sum = 0.0f;
        for (unsigned i = 0; i < len; ++i) {
            const float v = input[base + i];
            sum = __fadd_rn(sum, __fmul_rn(v, v));
        }
        rms_shared = __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, (float)len), eps));
    }
    __syncthreads();
    const float rms = rms_shared;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = unit_offset != 0u ? (1.0f + weight[i]) : weight[i];
        output[base + i] = __fmul_rn(__fdiv_rn(input[base + i], rms), scale);
    }
}

extern "C" __global__ void rnb_mtp_build_eh_input_f32(
    const float* __restrict__ token_rows,
    const float* __restrict__ target_hidden_rows,
    const float* __restrict__ enorm,
    const float* __restrict__ hnorm,
    float* __restrict__ output,
    float eps,
    unsigned rows,
    unsigned hidden_dim) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned base = row * hidden_dim;
    __shared__ float token_partial[256];
    __shared__ float hidden_partial[256];
    float token_sum = 0.0f;
    float hidden_sum = 0.0f;
    for (unsigned i = tid; i < hidden_dim; i += blockDim.x) {
        const float token = token_rows[base + i];
        const float hidden = target_hidden_rows[base + i];
        token_sum += token * token;
        hidden_sum += hidden * hidden;
    }
    token_partial[tid] = token_sum;
    hidden_partial[tid] = hidden_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            token_partial[tid] += token_partial[tid + stride];
            hidden_partial[tid] += hidden_partial[tid + stride];
        }
        __syncthreads();
    }
    const float token_inv = rsqrtf(token_partial[0] / (float)hidden_dim + eps);
    const float hidden_inv = rsqrtf(hidden_partial[0] / (float)hidden_dim + eps);
    const unsigned out_base = row * hidden_dim * 2u;
    for (unsigned i = tid; i < hidden_dim; i += blockDim.x) {
        output[out_base + i] = token_rows[base + i] * token_inv * enorm[i];
        output[out_base + hidden_dim + i] =
            target_hidden_rows[base + i] * hidden_inv * hnorm[i];
    }
}

extern "C" __global__ void rnb_rms_norm_add_rows_f32_inplace(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ residual,
    float eps,
    unsigned rows,
    unsigned len,
    unsigned unit_offset) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned base = row * len;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = unit_offset != 0u ? (1.0f + weight[i]) : weight[i];
        residual[base + i] += input[base + i] * inv_rms * scale;
    }
}

extern "C" __global__ void rnb_rms_norm_add_then_rms_norm_rows_f32(
    const float* __restrict__ input,
    const float* __restrict__ post_weight,
    float* __restrict__ residual,
    const float* __restrict__ pre_weight,
    float* __restrict__ output,
    float eps,
    unsigned rows,
    unsigned len,
    unsigned post_unit_offset,
    unsigned pre_unit_offset) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned base = row * len;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float post_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale =
            post_unit_offset != 0u ? (1.0f + post_weight[i]) : post_weight[i];
        residual[base + i] += input[base + i] * post_inv_rms * scale;
    }
    __syncthreads();

    float residual_sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = residual[base + i];
        residual_sum += v * v;
    }
    partial[tid] = residual_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float pre_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = pre_unit_offset != 0u ? (1.0f + pre_weight[i]) : pre_weight[i];
        output[base + i] = residual[base + i] * pre_inv_rms * scale;
    }
}

extern "C" __global__ void rnb_rms_norm_add_then_rms_norm_rows_q8_1_f32(
    const float* __restrict__ input,
    const float* __restrict__ post_weight,
    float* __restrict__ residual,
    const float* __restrict__ pre_weight,
    float* __restrict__ output,
    signed char* __restrict__ out_qs,
    float* __restrict__ out_ds,
    float eps,
    unsigned rows,
    unsigned len,
    unsigned post_unit_offset,
    unsigned pre_unit_offset) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned base = row * len;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = input[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float post_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale =
            post_unit_offset != 0u ? (1.0f + post_weight[i]) : post_weight[i];
        residual[base + i] += input[base + i] * post_inv_rms * scale;
    }
    __syncthreads();

    float residual_sum = 0.0f;
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float v = residual[base + i];
        residual_sum += v * v;
    }
    partial[tid] = residual_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float pre_inv_rms = rsqrtf(partial[0] / (float)len + eps);
    for (unsigned i = tid; i < len; i += blockDim.x) {
        const float scale = pre_unit_offset != 0u ? (1.0f + pre_weight[i]) : pre_weight[i];
        output[base + i] = residual[base + i] * pre_inv_rms * scale;
    }
    __syncthreads();

    const unsigned chunks = len >> 5;
    for (unsigned chunk = tid; chunk < chunks; chunk += blockDim.x) {
        const unsigned chunk_base = base + (chunk << 5);
        float max_abs = 0.0f;
        for (unsigned lane = 0u; lane < 32u; ++lane) {
            const float value = output[chunk_base + lane];
            const float abs_value = fabsf(value);
            max_abs = abs_value > max_abs ? abs_value : max_abs;
        }
        const float d = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
        out_ds[row * chunks + chunk] = d;
        for (unsigned lane = 0u; lane < 32u; ++lane) {
            int q = 0;
            if (d > 0.0f) {
                q = (int)nearbyintf(output[chunk_base + lane] / d);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
            }
            out_qs[chunk_base + lane] = (signed char)q;
        }
    }
}

extern "C" __global__ void rnb_scale_rows_inplace(
    float* __restrict__ out,
    const float* __restrict__ scale,
    unsigned rows,
    unsigned row_count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned len = rows * row_count;
    if (i >= len) {
        return;
    }
    out[i] *= scale[i / rows];
}

extern "C" __global__ void rnb_add_f32_inplace(
    float* __restrict__ dst,
    const float* __restrict__ src,
    unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    dst[i] += src[i];
}

// cu44 step 20: scalar f32 multiply inplace. Gemma4 의 layer_output_scale
// (apply_layer_output_scale_inplace) 의 device 화. chain function 끝에서
// carrier 에 scalar scale apply 해서 host scratch.hidden 과 일관성 유지.
extern "C" __global__ void rnb_scale_f32_inplace(
    float* __restrict__ dst,
    float scale,
    unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    dst[i] *= scale;
}

// cu51 step 42: f32 → f16 (half) pack. K/V projection result f32 → KV cache f16.
// caller 가 input f32 + output u16 (f16 bits) device buffer 제공.
extern "C" __global__ void rnb_f32_to_f16_pack(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    __half h = __float2half_rn(src[i]);
    dst[i] = __half_as_ushort(h);
}

// =============================================================================
// cu38 Phase 1: Q4K + cuBLAS fused matmul kernel (Phase 1 naive 형태)
//
// 목적: 현재 두 kernel path (Q4K → F16 dequant → cuBLAS hgemm/sgemm) 를
// 단일 kernel 로 fused. 중간 F16 buffer 제거 + memory bandwidth 절약.
//
// Phase 1: row 단위 naive matmul. row warp parallel, 매 row 안 K loop 에서
// Q4K block on-the-fly dequant + dot product 누적. shared memory tile 없음.
// 이미 q8dot path 가 비슷한 구조 — Phase 1 은 F32 input 직접 받음 (q8dot 의
// quantize input 단계 생략).
//
// Phase 2: shared memory tile + tensor core 도입
// Phase 3: occupancy / bank conflict 최적화
//
// Input: weights (Q4K bytes), input (f32 [seq_len, K]), out (f32 [seq_len, rows])
// K = blocks_per_row * 256
//
// Layout: grid = (rows.div_ceil(8), seq_len, 1), block = (256, 1, 1)
// 8 warp per block, each warp processes 1 row.
extern "C" __global__ void rnb_q4k_sgemm_fused_naive(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;       // 0..7
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    if (row >= rows || seq >= seq_len) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_seq = input + seq * blocks_per_row * 256u;

    float acc = 0.0f;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 144u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

        // 각 lane 이 32 thread × 4 element (chunk) = 128 element 처리 (per block 128/256)
        // chunk = 0..7 (8 chunk × 32 lane × ... wait — 매 lane 0..31, 매 chunk
        // index = lane + step*32. Q4K block = 256 element. 256/32 = 8 element per lane.
        // 각 element pair (i, i+1) 의 q4 pack 으로 처리 (4 element at once via __dp4a).
        for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
            // chunk = 0..63 (64 chunk × 4 element = 256 element)
            const unsigned j = chunk >> 3;        // 0..7 (sub-block index)
            unsigned sc;
            unsigned mn;
            if (j < 4u) {
                sc = block[4u + j] & 63u;
                mn = block[4u + j + 4u] & 63u;
            } else {
                sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
            }
            const unsigned elem = (chunk & 7u) * 4u;   // 0, 4, 8, ..., 28
            const unsigned q_index = (j >> 1) * 32u + elem;
            const unsigned char* q_ptr = block + 16u + q_index;

            // q4 unpack: 4 element (q0~q3)
            unsigned char qb0, qb1;
            if ((j & 1u) == 0u) {
                qb0 = q_ptr[0] & 0x0fu;
                qb1 = q_ptr[1] & 0x0fu;
            } else {
                qb0 = q_ptr[0] >> 4;
                qb1 = q_ptr[1] >> 4;
            }
            const unsigned char qb2 = (j & 1u) == 0u ? (q_ptr[2] & 0x0fu) : (q_ptr[2] >> 4);
            const unsigned char qb3 = (j & 1u) == 0u ? (q_ptr[3] & 0x0fu) : (q_ptr[3] >> 4);

            const float weight_scale = d * (float)sc;
            const float weight_min = dmin * (float)mn;

            // input load (4 f32 element)
            const unsigned x_base = b * 256u + j * 32u + elem;
            const float x0 = input_seq[x_base + 0u];
            const float x1 = input_seq[x_base + 1u];
            const float x2 = input_seq[x_base + 2u];
            const float x3 = input_seq[x_base + 3u];

            // dot: (weight - min) * input
            acc += weight_scale * ((float)qb0 * x0 + (float)qb1 * x1
                                 + (float)qb2 * x2 + (float)qb3 * x3)
                 - weight_min * (x0 + x1 + x2 + x3);
        }
    }

    // warp reduce
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

// =============================================================================
// cu38 Phase 2: Q4K + cuBLAS fused matmul kernel — nvcuda::wmma tensor core 변형
//
// Ampere SM 80+ 의 tensor core mma.m16n8k16.f16.f16 사용. wmma fragment API.
//
// 구조:
// - 1 block = 1 warp (32 thread)
// - output tile: 16 row × 16 seq (wmma fragment 한 unit)
// - shared mem: weight tile [16, 16] F16 + input tile [16, 16] F16 (col-major
//   for matrix_b)
// - K loop 단위 16. K = blocks_per_row * 256 = 16의 배수
// - Q4K weight 매 tile mat 마다 shared mem 안 dequant (per element)
//
// grid = (rows / 16, seq_len / 16, 1), block = (32, 1, 1).
//
// limitation Phase 2:
// - Q4K dequant per element (32 thread × 8 element each) — 가능 inefficient
// - shared mem bank conflict 미고려
// - register pressure 미최적화
//
// Phase 3 후속 (occupancy / bank conflict 최적화):
// - shared mem padding (avoid 32-bank conflict)
// - 4 warp / block (32x32 output tile)
// - register tiling for K accumulation
extern "C" __global__ void rnb_q4k_sgemm_fused_wmma(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const __half* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    using namespace nvcuda;
    const unsigned tid = threadIdx.x;
    const unsigned row_base = blockIdx.x * 16u;
    const unsigned seq_base = blockIdx.y * 16u;

    if (row_base >= rows) return;
    // seq_base 가 seq_len 넘으면 일부 tile 만 valid. fragment 0 fill 처리.

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ __half weight_tile[16 * 16];
    __shared__ __half input_tile[16 * 16];

    const unsigned K = blocks_per_row * 256u;

    for (unsigned k = 0; k < K; k += 16) {
        // weight tile [16, 16] row-major. row_base + i, k + j.
        for (unsigned i = tid; i < 256u; i += 32u) {
            const unsigned tile_row = i / 16u;
            const unsigned tile_col = i % 16u;
            const unsigned actual_row = row_base + tile_row;
            const unsigned actual_col = k + tile_col;
            __half w_h = __float2half(0.0f);
            if (actual_row < rows && actual_col < K) {
                const unsigned b = actual_col / 256u;
                const unsigned within = actual_col % 256u;
                const unsigned j = within / 32u;
                const unsigned within_sub = within % 32u;
                const unsigned char* block_ptr = weights + actual_row * blocks_per_row * 144u + b * 144u;
                const unsigned raw_d = (unsigned)block_ptr[0] | ((unsigned)block_ptr[1] << 8);
                const unsigned raw_dmin = (unsigned)block_ptr[2] | ((unsigned)block_ptr[3] << 8);
                const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                unsigned sc, mn;
                if (j < 4u) {
                    sc = block_ptr[4u + j] & 63u;
                    mn = block_ptr[4u + j + 4u] & 63u;
                } else {
                    sc = (block_ptr[4u + j + 4u] & 0x0fu) | ((block_ptr[4u + j - 4u] >> 6) << 4);
                    mn = (block_ptr[4u + j + 4u] >> 4) | ((block_ptr[4u + j] >> 6) << 4);
                }
                const unsigned q_index = (j >> 1) * 32u + within_sub;
                const unsigned char q_byte = block_ptr[16u + q_index];
                const unsigned q4 = (j & 1u) == 0u ? (q_byte & 0x0fu) : (q_byte >> 4);
                const float w = d * (float)sc * (float)q4 - dmin * (float)mn;
                w_h = __float2half(w);
            }
            weight_tile[tile_row * 16u + tile_col] = w_h;
        }

        // input tile col-major for matrix_b: b_frag[k_idx, n_idx] = ptr[n * ldm + k].
        // ldm = 16. ptr = input_tile. We want b_frag[k, n] = input[seq_base + n, k_iter + k].
        // → input_tile[n * 16 + k] = input[seq_base + n, k_iter + k].
        for (unsigned i = tid; i < 256u; i += 32u) {
            const unsigned tile_seq = i / 16u;
            const unsigned tile_k = i % 16u;
            const unsigned actual_seq = seq_base + tile_seq;
            const unsigned actual_k = k + tile_k;
            __half x_h = __float2half(0.0f);
            if (actual_seq < seq_len && actual_k < K) {
                x_h = input[actual_seq * K + actual_k];
            }
            input_tile[tile_seq * 16u + tile_k] = x_h;
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, weight_tile, 16);
        wmma::load_matrix_sync(b_frag, input_tile, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    // output [seq_len, rows] row-major. out[seq][row] = c_frag[row, seq].
    // wmma store mem_col_major: ptr[j * ldm + i] = c_frag[i, j].
    // ptr = out + seq_base * rows + row_base. ldm = rows.
    //   ptr[j * rows + i] = out[(seq_base + j) * rows + (row_base + i)]. ✓
    // out-of-bounds 처리: 끝 tile 의 seq_base + 16 > seq_len 시 partial store
    // 필요. 일단 단순 full store (메모리 over-write 가능. seq_len 16의 배수
    // 일 때만 안전. Phase 2 limitation).
    if (row_base < rows && seq_base < seq_len) {
        wmma::store_matrix_sync(
            out + seq_base * rows + row_base,
            c_frag,
            rows,
            wmma::mem_col_major);
    }
}

// =============================================================================
// cu38 Phase 3: Q4K + cuBLAS fused matmul — wmma 4-warp 32x32 output tile
//
// Phase 2 (1-warp 16x16) 보다 4× output / block. occupancy 향상.
//
// 구조:
// - 1 block = 4 warps = 128 threads
// - output tile: 32 rows × 32 seqs (4 wmma fragment, 각 warp 16x16 quadrant)
// - shared mem: weight_tile [32 rows × 16 K] F16 = 1 KB, input_tile [32 seqs × 16 K] F16 = 1 KB
// - K loop step 16. mma_sync per warp per K iter.
//
// warp 0: c_frag[0..16, 0..16]  (tile_row=0, tile_seq=0)
// warp 1: c_frag[0..16, 16..32] (tile_row=0, tile_seq=16)
// warp 2: c_frag[16..32, 0..16] (tile_row=16, tile_seq=0)
// warp 3: c_frag[16..32, 16..32] (tile_row=16, tile_seq=16)
//
// grid = (rows.div_ceil(32), seq_len.div_ceil(32), 1), block = (128, 1, 1).
extern "C" __global__ void rnb_q4k_sgemm_fused_wmma_4warp(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const __half* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    using namespace nvcuda;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned tile_row = (warp >> 1) * 16u;    // 0 or 16
    const unsigned tile_seq = (warp & 1u) * 16u;    // 0 or 16
    const unsigned block_row_base = blockIdx.x * 32u;
    const unsigned block_seq_base = blockIdx.y * 32u;
    const unsigned row_base = block_row_base + tile_row;
    const unsigned seq_base = block_seq_base + tile_seq;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ __half weight_tile[32 * 16];
    __shared__ __half input_tile[32 * 16];

    const unsigned K = blocks_per_row * 256u;

    for (unsigned k = 0; k < K; k += 16) {
        // weight_tile [32, 16] row-major. 128 threads × 4 elem = 512 elem total.
        for (unsigned i = threadIdx.x; i < 512u; i += 128u) {
            const unsigned t_row = i / 16u;
            const unsigned t_col = i % 16u;
            const unsigned actual_row = block_row_base + t_row;
            const unsigned actual_col = k + t_col;
            __half w_h = __float2half(0.0f);
            if (actual_row < rows && actual_col < K) {
                const unsigned b = actual_col / 256u;
                const unsigned within = actual_col % 256u;
                const unsigned j = within / 32u;
                const unsigned within_sub = within % 32u;
                const unsigned char* block_ptr = weights + actual_row * blocks_per_row * 144u + b * 144u;
                const unsigned raw_d = (unsigned)block_ptr[0] | ((unsigned)block_ptr[1] << 8);
                const unsigned raw_dmin = (unsigned)block_ptr[2] | ((unsigned)block_ptr[3] << 8);
                const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                unsigned sc, mn;
                if (j < 4u) {
                    sc = block_ptr[4u + j] & 63u;
                    mn = block_ptr[4u + j + 4u] & 63u;
                } else {
                    sc = (block_ptr[4u + j + 4u] & 0x0fu) | ((block_ptr[4u + j - 4u] >> 6) << 4);
                    mn = (block_ptr[4u + j + 4u] >> 4) | ((block_ptr[4u + j] >> 6) << 4);
                }
                const unsigned q_index = (j >> 1) * 32u + within_sub;
                const unsigned char q_byte = block_ptr[16u + q_index];
                const unsigned q4 = (j & 1u) == 0u ? (q_byte & 0x0fu) : (q_byte >> 4);
                const float w = d * (float)sc * (float)q4 - dmin * (float)mn;
                w_h = __float2half(w);
            }
            weight_tile[t_row * 16u + t_col] = w_h;
        }

        // input_tile [32 seqs, 16 K] col_major: input_tile[seq * 16 + k] = input[seq][k].
        for (unsigned i = threadIdx.x; i < 512u; i += 128u) {
            const unsigned t_seq = i / 16u;
            const unsigned t_k = i % 16u;
            const unsigned actual_seq = block_seq_base + t_seq;
            const unsigned actual_k = k + t_k;
            __half x_h = __float2half(0.0f);
            if (actual_seq < seq_len && actual_k < K) {
                x_h = input[actual_seq * K + actual_k];
            }
            input_tile[t_seq * 16u + t_k] = x_h;
        }
        __syncthreads();

        // warp 별 다른 16x16 quadrant.
        wmma::load_matrix_sync(a_frag, weight_tile + tile_row * 16u, 16);
        wmma::load_matrix_sync(b_frag, input_tile + tile_seq * 16u, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    if (row_base < rows && seq_base < seq_len) {
        wmma::store_matrix_sync(
            out + seq_base * rows + row_base,
            c_frag,
            rows,
            wmma::mem_col_major);
    }
}

// =============================================================================
// cu39 Phase 1: Q4_K weight × Q8_1 activation → DP4A integer dot matmul.
//
// 동기: llama.cpp Gemma E4B prefill 2885 tok/s vs 우리 246 tok/s (~11.7× 격차).
// llama.cpp 의 mmq.cuh (vec_dot_q4_K_q8_1_impl_mmq) 가 Q4_K 4-bit nibble × Q8_1
// int8 activation 을 Ampere DP4A 로 직접 dot product. cuBLAS 거치지 않음.
// cu38 Phase 1-3 (naive/wmma) 가 cuBLAS hgemm 와 ε 동등에서 멈췄던 이유 =
// cuBLAS 자체도 dp4a path 보다 느림 (weight 메모리 BW F16/4.5bit ≈ 3.5×, +
// dequant launch overhead). 본 kernel 은 weight 를 dequant 안 하고 nibble 단위
// dp4a 로 직접 multiply-accumulate.
// =============================================================================

extern "C" __global__ void rnb_quantize_q8_1_with_sum_by_32(
    const float* __restrict__ input,
    signed char* __restrict__ out_qs,
    float* __restrict__ out_ds,
    float* __restrict__ out_sums,
    unsigned chunks) {
    const unsigned chunk = blockIdx.x;
    const unsigned lane = threadIdx.x;
    if (chunk >= chunks || lane >= 32u) {
        return;
    }

    __shared__ float values[32];
    __shared__ float abs_values[32];
    __shared__ float sum_values[32];
    const unsigned i = chunk * 32u + lane;
    const float value = input[i];
    values[lane] = value;
    abs_values[lane] = fabsf(value);
    sum_values[lane] = value;
    __syncthreads();

    for (unsigned stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride && abs_values[lane + stride] > abs_values[lane]) {
            abs_values[lane] = abs_values[lane + stride];
        }
        if (lane < stride) {
            sum_values[lane] += sum_values[lane + stride];
        }
        __syncthreads();
    }

    const float max_abs = abs_values[0];
    const float d = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
    if (lane == 0u) {
        out_ds[chunk] = d;
        out_sums[chunk] = sum_values[0];
    }
    int q = 0;
    if (d > 0.0f) {
        q = (int)nearbyintf(values[lane] / d);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    out_qs[i] = (signed char)q;
}

// cu39: Q4_K × Q8_1 DP4A matmul, naive 1-warp-per-row, no shared mem tile.
//
// Layout 결정:
// - grid = (rows.div_ceil(8), seq_len, 1), block = (256, 1, 1) = 8 warp.
// - 1 warp = 1 output row, K loop over blocks_per_row Q4_K macroblocks.
// - 매 warp pass = 1 Q4_K block (256 elem) 통째로 처리:
//     32 lane × 8 elem (= 2 dp4a) = 256 elem. 모든 lane busy.
//     lane → sub-block j = lane / 4, sub-block 안 offset = (lane & 3) * 8.
// - 사용 instruction: __dp4a (Ampere SM 80+ native, 1 cycle 16 int8 MACs).
//
// 수식 (per warp pass = 1 block):
//   acc += d_q4k * Σ_j (d_q8[j] * sc[j] * dp4a_sumi[j])
//        - dmin_q4k * Σ_j (sum_q8[j] * mn[j])
// where j = 0..7 sub-blocks, dp4a_sumi[j] = Σ (4 lane × 2 dp4a) = 32-elem dot.
//
// 한계 (Phase 2+ 에서 다룰 것):
// - shared mem tile staging 없음 → weight memory 매번 global load.
// - seq_len > 1 일 때 weight reuse 없음. mmq mma path (8 row × 32 seq tile) 보다 느림.
// - Phase 1 목표: correctness 확정 + DP4A baseline 확보 (cu38 phase 1 의 float
//   path 대비 점진 비교).
extern "C" __global__ void rnb_q4k_q8_1_matmul_dp4a(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ input_sums,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned seq = blockIdx.y;
    if (row >= rows || seq >= seq_len) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const signed char* input_qs_seq = input_qs + seq * blocks_per_row * 256u;
    const float* input_ds_seq = input_ds + seq * blocks_per_row * 8u;
    const float* input_sums_seq = input_sums + seq * blocks_per_row * 8u;

    const unsigned j = lane >> 2u;            // 0..7 (sub-block)
    const unsigned sub_off = (lane & 3u) * 8u; // 0, 8, 16, 24

    float acc = 0.0f;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block_ptr = row_ptr + b * 144u;
        const unsigned raw_d = (unsigned)block_ptr[0] | ((unsigned)block_ptr[1] << 8);
        const unsigned raw_dmin = (unsigned)block_ptr[2] | ((unsigned)block_ptr[3] << 8);
        const float d_q4k = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin_q4k = __half2float(__ushort_as_half((unsigned short)raw_dmin));

        // Q4_K 6-bit scales/mins unpack (same logic as our naive kernel).
        unsigned sc;
        unsigned mn;
        if (j < 4u) {
            sc = block_ptr[4u + j] & 63u;
            mn = block_ptr[4u + j + 4u] & 63u;
        } else {
            sc = (block_ptr[4u + j + 4u] & 0x0fu) | ((block_ptr[4u + j - 4u] >> 6) << 4);
            mn = (block_ptr[4u + j + 4u] >> 4) | ((block_ptr[4u + j] >> 6) << 4);
        }

        // Q4_K nibble layout: sub-block j 의 32 nibble = bytes[nibble_base..+32], shift=(j&1)*4.
        // 본 lane 의 8 elements: bytes[nibble_base + sub_off .. + sub_off + 7].
        // 4 nibble per dp4a, 2 dp4a per lane.
        const unsigned nibble_base = 16u + (j >> 1) * 32u + sub_off;
        const unsigned shift = (j & 1u) * 4u;
        const unsigned char* qbytes = block_ptr + nibble_base;

        const int w_int_a = ((int)((qbytes[3] >> shift) & 0x0Fu) << 24)
                          | ((int)((qbytes[2] >> shift) & 0x0Fu) << 16)
                          | ((int)((qbytes[1] >> shift) & 0x0Fu) << 8)
                          |  (int)((qbytes[0] >> shift) & 0x0Fu);

        const int w_int_b = ((int)((qbytes[7] >> shift) & 0x0Fu) << 24)
                          | ((int)((qbytes[6] >> shift) & 0x0Fu) << 16)
                          | ((int)((qbytes[5] >> shift) & 0x0Fu) << 8)
                          |  (int)((qbytes[4] >> shift) & 0x0Fu);

        // Q8_1 chunk: 32 int8 + (d, sum). chunk_idx = b * 8 + j.
        const unsigned q8_chunk_idx = b * 8u + j;
        const float d_q8 = input_ds_seq[q8_chunk_idx];
        const float sum_q8 = input_sums_seq[q8_chunk_idx];

        const signed char* q8_ptr = input_qs_seq + b * 256u + j * 32u + sub_off;
        const int q8_int_a = *reinterpret_cast<const int*>(q8_ptr);
        const int q8_int_b = *reinterpret_cast<const int*>(q8_ptr + 4);

        const int sumi = __dp4a(w_int_b, q8_int_b, __dp4a(w_int_a, q8_int_a, 0));

        // per-lane d-term partial (each lane covers 8 elem of sub-block j).
        float lane_d = d_q8 * (float)sc * (float)sumi;
        // per-lane m-term partial — 4 lane per sub-block 이라 1 lane 만 sum_q8 기여
        // (안 그러면 4× 중복). lane & 3 == 0 → lanes 0,4,8,...,28.
        float lane_m = ((lane & 3u) == 0u) ? sum_q8 * (float)mn : 0.0f;

        // warp reduce: sum across 32 lane → block-level partial.
        for (unsigned o = 16u; o > 0u; o >>= 1u) {
            lane_d += __shfl_down_sync(0xffffffffu, lane_d, o);
            lane_m += __shfl_down_sync(0xffffffffu, lane_m, o);
        }

        if (lane == 0u) {
            acc += d_q4k * lane_d - dmin_q4k * lane_m;
        }
    }

    if (lane == 0u) {
        out[seq * rows + row] = acc;
    }
}

// =============================================================================
// cu39 Phase 2: Q4_K × Q8_1 DP4A matmul with shared mem weight tile staging.
//
// 동기: Phase 1 (naive 1-warp-per-row) 가 cuBLAS HGEMM 와 ε 동등 (ABAB 측정).
// 같은 weight 가 매 seq position 별 CTA 마다 global memory 에서 다시 읽힘 →
// seq_len = 1115 일 때 weight memory traffic 1115× 낭비.
// Phase 2 = mmq.cuh 의 load_tiles_q4_K 패턴 port. CTA 당 8 row × 8 seq 의
// output tile 처리. weight 를 shared mem 에 한 번 로드 + 8 seq 에 재사용.
//
// Layout:
// - grid = (rows.div_ceil(8), seq_len.div_ceil(8), 1)
// - block = (32, 1, 1) = 1 warp
// - per CTA output: 8 row × 8 seq = 64 cell
// - per lane: 2 output cell (row A = lane/8, row B = lane/8 + 4; seq = lane%8)
//
// K loop: per iter = 32 K elem = 1 Q4_K sub-block. blocks_per_row × 8 iter total.
// =============================================================================
extern "C" __global__ void rnb_q4k_q8_1_matmul_dp4a_tile(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ input_sums,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned lane = threadIdx.x;
    const unsigned row_base = blockIdx.x * 8u;
    const unsigned seq_base = blockIdx.y * 8u;

    const unsigned local_row_a = lane >> 3;        // 0..3
    const unsigned local_row_b = local_row_a + 4u; // 4..7
    const unsigned local_seq = lane & 7u;          // 0..7

    const unsigned row_a = row_base + local_row_a;
    const unsigned row_b = row_base + local_row_b;
    const unsigned seq = seq_base + local_seq;

    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;
    const bool seq_valid = seq < seq_len;

    __shared__ signed char x_qs[8 * 32];
    __shared__ float x_d[8];
    __shared__ float x_dmin[8];
    __shared__ unsigned char x_sc[8];
    __shared__ unsigned char x_mn[8];
    __shared__ signed char y_qs[8 * 32];
    __shared__ float y_d[8];
    __shared__ float y_sum[8];

    const unsigned row_bytes = blocks_per_row * 144u;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        float blk_sumfd_a = 0.0f;
        float blk_sumfm_a = 0.0f;
        float blk_sumfd_b = 0.0f;
        float blk_sumfm_b = 0.0f;

        for (unsigned j = 0; j < 8u; ++j) {
            // 1. Load weight tile (8 rows × 32 nibbles).
            //    lane = (row 0..7, byte_base 0/8/16/24): row = lane/4, byte_base = (lane%4)*8.
            const unsigned load_row = lane >> 2;
            const unsigned load_byte_base = (lane & 3u) * 8u;
            const unsigned actual_row = row_base + load_row;
            if (actual_row < rows) {
                const unsigned char* block_ptr = weights + actual_row * row_bytes + b * 144u;
                if (j == 0u && load_byte_base == 0u) {
                    const unsigned raw_d = (unsigned)block_ptr[0] | ((unsigned)block_ptr[1] << 8);
                    const unsigned raw_dmin = (unsigned)block_ptr[2] | ((unsigned)block_ptr[3] << 8);
                    x_d[load_row] = __half2float(__ushort_as_half((unsigned short)raw_d));
                    x_dmin[load_row] = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                }
                if (load_byte_base == 0u) {
                    unsigned sc;
                    unsigned mn;
                    if (j < 4u) {
                        sc = block_ptr[4u + j] & 63u;
                        mn = block_ptr[4u + j + 4u] & 63u;
                    } else {
                        sc = (block_ptr[4u + j + 4u] & 0x0fu) | ((block_ptr[4u + j - 4u] >> 6) << 4);
                        mn = (block_ptr[4u + j + 4u] >> 4) | ((block_ptr[4u + j] >> 6) << 4);
                    }
                    x_sc[load_row] = (unsigned char)sc;
                    x_mn[load_row] = (unsigned char)mn;
                }
                const unsigned nibble_base = 16u + (j >> 1) * 32u + load_byte_base;
                const unsigned shift = (j & 1u) * 4u;
                const unsigned char* qb = block_ptr + nibble_base;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    x_qs[load_row * 32u + load_byte_base + e] = (signed char)((qb[e] >> shift) & 0x0Fu);
                }
            }

            // 2. Load activation tile + d_q8/sum_q8.
            const unsigned load_seq = lane >> 2;
            const unsigned load_qoff = (lane & 3u) * 8u;
            const unsigned actual_seq = seq_base + load_seq;
            if (actual_seq < seq_len) {
                const unsigned chunk_idx_global = (actual_seq * blocks_per_row + b) * 8u + j;
                if (load_qoff == 0u) {
                    y_d[load_seq] = input_ds[chunk_idx_global];
                    y_sum[load_seq] = input_sums[chunk_idx_global];
                }
                const signed char* q8_ptr = input_qs
                    + actual_seq * blocks_per_row * 256u
                    + b * 256u + j * 32u + load_qoff;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    y_qs[load_seq * 32u + load_qoff + e] = q8_ptr[e];
                }
            }

            __syncwarp();

            // 3. Compute per-lane 2 output cells.
            if (seq_valid) {
                const float d_q8 = y_d[local_seq];
                const float sum_q8 = y_sum[local_seq];

                if (row_a_valid) {
                    int sumi = 0;
                    const signed char* xr = &x_qs[local_row_a * 32u];
                    const signed char* yc = &y_qs[local_seq * 32u];
                    #pragma unroll
                    for (int k = 0; k < 32; k += 4) {
                        const int xi = *reinterpret_cast<const int*>(xr + k);
                        const int yi = *reinterpret_cast<const int*>(yc + k);
                        sumi = __dp4a(xi, yi, sumi);
                    }
                    blk_sumfd_a += d_q8 * (float)x_sc[local_row_a] * (float)sumi;
                    blk_sumfm_a += sum_q8 * (float)x_mn[local_row_a];
                }
                if (row_b_valid) {
                    int sumi = 0;
                    const signed char* xr = &x_qs[local_row_b * 32u];
                    const signed char* yc = &y_qs[local_seq * 32u];
                    #pragma unroll
                    for (int k = 0; k < 32; k += 4) {
                        const int xi = *reinterpret_cast<const int*>(xr + k);
                        const int yi = *reinterpret_cast<const int*>(yc + k);
                        sumi = __dp4a(xi, yi, sumi);
                    }
                    blk_sumfd_b += d_q8 * (float)x_sc[local_row_b] * (float)sumi;
                    blk_sumfm_b += sum_q8 * (float)x_mn[local_row_b];
                }
            }

            __syncwarp();
        }

        if (seq_valid) {
            if (row_a_valid) {
                acc_a += x_d[local_row_a] * blk_sumfd_a - x_dmin[local_row_a] * blk_sumfm_a;
            }
            if (row_b_valid) {
                acc_b += x_d[local_row_b] * blk_sumfd_b - x_dmin[local_row_b] * blk_sumfm_b;
            }
        }
    }

    if (seq_valid) {
        if (row_a_valid) {
            out[seq * rows + row_a] = acc_a;
        }
        if (row_b_valid) {
            out[seq * rows + row_b] = acc_b;
        }
    }
}

// =============================================================================
// cu39 Phase 3: Q4_K × Q8_1 mma.m16n8k32.s8.s8 tensor core matmul.
//
// 동기: Phase 1-2 (naive + tile dp4a) 모두 cuBLAS HGEMM 와 ε 동등. dp4a 는 1 cycle
// 16 int8 MAC = wmma F16 throughput 의 ~2× 만 (cuBLAS 가 이미 well-optimized).
// Ampere SM 80+ tensor core mma.m16n8k32.s8.s8 = 1 instruction 으로 16M × 32K × 8N
// = 4096 int8 MAC. 이론적으로 cuBLAS HGEMM (m16n16k16 f16 = 4096 f16 MAC) 와 같은
// throughput 이지만, Q4_K weight memory BW (4.5 bit/elem vs F16 16 bit) ~3.5× 절약
// 이 진짜 lever.
//
// Layout:
// - grid = (rows.div_ceil(16), seq_len.div_ceil(8), 1)
// - block = (32, 1, 1) = 1 warp
// - per CTA output: 16 row × 8 seq = 128 cell
// - per K-iter: 32 K elem = 1 Q4_K sub-block.
// - 1 mma.m16n8k32.s8.s8 = 1 sub-block 의 16×8 partial sum (int32)
//
// PTX layout (Ampere mma m16n8k32.row.col.s32.s8.s8.s32):
// - A: M×K = 16×32 row-major, thread t holds 4 int32 reg = 16 int8:
//     a0 = A[t/4    ][4*(t%4)+0..3]
//     a1 = A[t/4+8  ][4*(t%4)+0..3]
//     a2 = A[t/4    ][4*(t%4)+16..19]
//     a3 = A[t/4+8  ][4*(t%4)+16..19]
// - B: K×N = 32×8 col-major, thread t holds 2 int32 reg = 8 int8:
//     b0 = B[4*(t%4)+0..3   ][t/4]
//     b1 = B[4*(t%4)+16..19][t/4]
// - C/D: M×N = 16×8 row-major, thread t holds 4 int32:
//     d0 = D[t/4    ][2*(t%4)  ]
//     d1 = D[t/4    ][2*(t%4)+1]
//     d2 = D[t/4+8  ][2*(t%4)  ]
//     d3 = D[t/4+8  ][2*(t%4)+1]
// =============================================================================

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void rnb_mma_m16n8k32_s8(
    int& d0, int& d1, int& d2, int& d3,
    int a0, int a1, int a2, int a3,
    int b0, int b1,
    int c0, int c1, int c2, int c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}
#endif

extern "C" __global__ void rnb_q4k_q8_1_matmul_mma(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ input_sums,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
    // Ampere 이전 device 에서는 호출 안 됨 (host launcher 가 가드).
    (void)out; (void)weights; (void)input_qs; (void)input_ds; (void)input_sums;
    (void)rows; (void)blocks_per_row; (void)seq_len;
    return;
#else
    const unsigned lane = threadIdx.x;
    const unsigned row_base = blockIdx.x * 16u;
    const unsigned seq_base = blockIdx.y * 8u;

    // shared mem layout per K-iter:
    //   A tile: 16 rows × 32 K, row-major, int8 → 16*32 = 512 bytes
    //   B tile: 32 K × 8 N, col-major, int8 → 32*8 = 256 bytes
    //   x_d[16]: per-row d_q4k (float) = 64 bytes (loaded once per block)
    //   x_dmin[16]: per-row dmin_q4k = 64 bytes
    //   x_sc[16]: per-row sc per sub-block = 16 bytes
    //   x_mn[16]: per-row mn per sub-block = 16 bytes
    //   y_d[8]: per-seq d_q8 per sub-block = 32 bytes
    //   y_sum[8]: per-seq sum_q8 per sub-block = 32 bytes
    __shared__ signed char a_tile[16 * 32];
    __shared__ signed char b_tile[32 * 8];
    __shared__ float x_d[16];
    __shared__ float x_dmin[16];
    __shared__ unsigned char x_sc[16];
    __shared__ unsigned char x_mn[16];
    __shared__ float y_d[8];
    __shared__ float y_sum[8];

    // Per-thread output accumulator. thread holds 4 int32 → 4 f32 accumulators
    // (one per output cell). Cell layout per thread t:
    //   cell[0] = (t/4    , 2*(t%4)  )
    //   cell[1] = (t/4    , 2*(t%4)+1)
    //   cell[2] = (t/4+8  , 2*(t%4)  )
    //   cell[3] = (t/4+8  , 2*(t%4)+1)
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const unsigned t_row_a = lane >> 2;          // 0..7
    const unsigned t_row_b = t_row_a + 8u;       // 8..15
    const unsigned t_col_a = (lane & 3u) << 1;   // 0,2,4,6
    const unsigned t_col_b = t_col_a + 1u;       // 1,3,5,7

    const unsigned row_a_g = row_base + t_row_a;
    const unsigned row_b_g = row_base + t_row_b;
    const unsigned seq_a_g = seq_base + t_col_a;
    const unsigned seq_b_g = seq_base + t_col_b;

    const bool row_a_valid = row_a_g < rows;
    const bool row_b_valid = row_b_g < rows;

    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        // sub-block-accumulator: per-cell scaled int32 sum for this block
        float blk_d_a0 = 0.0f, blk_d_a1 = 0.0f, blk_d_b0 = 0.0f, blk_d_b1 = 0.0f;
        float blk_m_a0 = 0.0f, blk_m_a1 = 0.0f, blk_m_b0 = 0.0f, blk_m_b1 = 0.0f;

        for (unsigned j = 0; j < 8u; ++j) {
            // 1. Load A tile (weight 16 rows × 32 nibbles).
            //    16 row × 32 byte = 512 bytes. 32 lanes × 16 byte/lane.
            //    layout: lane → row = lane/2, byte_base = (lane%2)*16.
            //    각 lane 16 nibble unpack.
            const unsigned aload_row = lane >> 1;
            const unsigned aload_off = (lane & 1u) * 16u;
            const unsigned aload_row_g = row_base + aload_row;
            if (aload_row_g < rows) {
                const unsigned char* block_ptr = weights + aload_row_g * row_bytes + b * 144u;
                if (j == 0u && aload_off == 0u) {
                    const unsigned raw_d = (unsigned)block_ptr[0] | ((unsigned)block_ptr[1] << 8);
                    const unsigned raw_dmin = (unsigned)block_ptr[2] | ((unsigned)block_ptr[3] << 8);
                    x_d[aload_row] = __half2float(__ushort_as_half((unsigned short)raw_d));
                    x_dmin[aload_row] = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                }
                if (aload_off == 0u) {
                    unsigned sc, mn;
                    if (j < 4u) {
                        sc = block_ptr[4u + j] & 63u;
                        mn = block_ptr[4u + j + 4u] & 63u;
                    } else {
                        sc = (block_ptr[4u + j + 4u] & 0x0fu) | ((block_ptr[4u + j - 4u] >> 6) << 4);
                        mn = (block_ptr[4u + j + 4u] >> 4) | ((block_ptr[4u + j] >> 6) << 4);
                    }
                    x_sc[aload_row] = (unsigned char)sc;
                    x_mn[aload_row] = (unsigned char)mn;
                }
                const unsigned nibble_base = 16u + (j >> 1) * 32u + aload_off;
                const unsigned shift = (j & 1u) * 4u;
                const unsigned char* qb = block_ptr + nibble_base;
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    a_tile[aload_row * 32u + aload_off + e] = (signed char)((qb[e] >> shift) & 0x0Fu);
                }
            }

            // 2. Load B tile (activations 32 K × 8 N col-major).
            //    32 lanes load 256 bytes = 8 byte/lane. lane → seq = lane/4, kb = (lane%4)*8.
            //    B[k][s] = b_tile[s*32 + k].
            const unsigned bload_seq = lane >> 2;
            const unsigned bload_kb = (lane & 3u) * 8u;
            const unsigned bload_seq_g = seq_base + bload_seq;
            if (bload_seq_g < seq_len) {
                const unsigned chunk_idx_global = (bload_seq_g * blocks_per_row + b) * 8u + j;
                if (bload_kb == 0u) {
                    y_d[bload_seq] = input_ds[chunk_idx_global];
                    y_sum[bload_seq] = input_sums[chunk_idx_global];
                }
                const signed char* q8_ptr = input_qs
                    + bload_seq_g * blocks_per_row * 256u
                    + b * 256u + j * 32u + bload_kb;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    b_tile[bload_seq * 32u + bload_kb + e] = q8_ptr[e];
                }
            }

            __syncwarp();

            // 3. Pack A fragment (4 int = 16 int8) for this lane.
            //    a0..a3 layout (per PTX m16n8k32 .row .col .s8 .s8):
            //    a0 = A[t/4    ][4*(t%4)+0..3]
            //    a1 = A[t/4+8  ][4*(t%4)+0..3]
            //    a2 = A[t/4    ][4*(t%4)+16..19]
            //    a3 = A[t/4+8  ][4*(t%4)+16..19]
            const unsigned a_row_lo = lane >> 2;        // 0..7
            const unsigned a_row_hi = a_row_lo + 8u;    // 8..15
            const unsigned a_col_lo = (lane & 3u) * 4u; // 0,4,8,12
            const unsigned a_col_hi = a_col_lo + 16u;   // 16,20,24,28
            int a0 = *reinterpret_cast<const int*>(&a_tile[a_row_lo * 32u + a_col_lo]);
            int a1 = *reinterpret_cast<const int*>(&a_tile[a_row_hi * 32u + a_col_lo]);
            int a2 = *reinterpret_cast<const int*>(&a_tile[a_row_lo * 32u + a_col_hi]);
            int a3 = *reinterpret_cast<const int*>(&a_tile[a_row_hi * 32u + a_col_hi]);

            // 4. Pack B fragment (2 int = 8 int8) for this lane.
            //    b0 = B[4*(t%4)+0..3 ][t/4]
            //    b1 = B[4*(t%4)+16..19][t/4]
            const unsigned b_seq = lane >> 2;            // 0..7
            const unsigned b_row_lo = (lane & 3u) * 4u;  // 0,4,8,12
            const unsigned b_row_hi = b_row_lo + 16u;    // 16,20,24,28
            int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_row_lo]);
            int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_row_hi]);

            // 5. Issue mma: D = A * B (int32 accumulator).
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            // 6. Scale d_q8 * sc per cell, accumulate to blk_d.
            //    cell 0: (row_a, col_a) — d0
            //    cell 1: (row_a, col_b) — d1
            //    cell 2: (row_b, col_a) — d2
            //    cell 3: (row_b, col_b) — d3
            const float sc_a = (float)x_sc[t_row_a];
            const float sc_b = (float)x_sc[t_row_b];
            const float mn_a = (float)x_mn[t_row_a];
            const float mn_b = (float)x_mn[t_row_b];
            const float dq_a = y_d[t_col_a];
            const float dq_b = y_d[t_col_b];
            const float sm_a = y_sum[t_col_a];
            const float sm_b = y_sum[t_col_b];

            blk_d_a0 += dq_a * sc_a * (float)d0;
            blk_d_a1 += dq_b * sc_a * (float)d1;
            blk_d_b0 += dq_a * sc_b * (float)d2;
            blk_d_b1 += dq_b * sc_b * (float)d3;

            blk_m_a0 += sm_a * mn_a;
            blk_m_a1 += sm_b * mn_a;
            blk_m_b0 += sm_a * mn_b;
            blk_m_b1 += sm_b * mn_b;

            __syncwarp();
        }

        // 7. End-of-block: acc += d_q4k * blk_d - dmin_q4k * blk_m
        const float d_a = x_d[t_row_a];
        const float d_b = x_d[t_row_b];
        const float dmin_a = x_dmin[t_row_a];
        const float dmin_b = x_dmin[t_row_b];
        acc[0] += d_a * blk_d_a0 - dmin_a * blk_m_a0;
        acc[1] += d_a * blk_d_a1 - dmin_a * blk_m_a1;
        acc[2] += d_b * blk_d_b0 - dmin_b * blk_m_b0;
        acc[3] += d_b * blk_d_b1 - dmin_b * blk_m_b1;
    }

    // 8. Write outputs (row-major output[seq, row]).
    if (row_a_valid && seq_a_g < seq_len) {
        out[seq_a_g * rows + row_a_g] = acc[0];
    }
    if (row_a_valid && seq_b_g < seq_len) {
        out[seq_b_g * rows + row_a_g] = acc[1];
    }
    if (row_b_valid && seq_a_g < seq_len) {
        out[seq_a_g * rows + row_b_g] = acc[2];
    }
    if (row_b_valid && seq_b_g < seq_len) {
        out[seq_b_g * rows + row_b_g] = acc[3];
    }
#endif
}

// =============================================================================
// cu39 Phase 4: Q4_K × Q8_1 mma 4-warp expansion (mmq_y=64, mmq_x=8).
//
// Phase 3 (1-warp) 가 cuBLAS 와 ε — mma 1개/CTA = 4096 MAC/cycle. cuBLAS HGEMM 도
// 같은 throughput level. Phase 4 = 4 warps × 16 row sub-tile = 64 row mmq_y +
// per CTA 4 mma per K-iter. weight 메모리 BW 절약 + occupancy 증가.
//
// Layout:
// - grid = (rows.div_ceil(64), seq_len.div_ceil(8), 1)
// - block = (128, 1, 1) = 4 warps
// - per CTA output: 64 row × 8 seq = 512 cells
// - warp w (0..3) handles row sub-tile [w*16 .. w*16+15] × 8 seqs
// =============================================================================
extern "C" __global__ void rnb_q4k_q8_1_matmul_mma_4warp(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ input_sums,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
    (void)out; (void)weights; (void)input_qs; (void)input_ds; (void)input_sums;
    (void)rows; (void)blocks_per_row; (void)seq_len;
    return;
#else
    const unsigned warp = threadIdx.x >> 5;       // 0..3
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_row_off = warp * 16u;
    const unsigned cta_row_base = blockIdx.x * 64u;
    const unsigned cta_seq_base = blockIdx.y * 8u;
    const unsigned row_base = cta_row_base + warp_row_off;

    __shared__ signed char a_tile[64 * 32];
    __shared__ signed char b_tile[32 * 8];
    __shared__ float x_d[64];
    __shared__ float x_dmin[64];
    __shared__ unsigned char x_sc[64];
    __shared__ unsigned char x_mn[64];
    __shared__ float y_d[8];
    __shared__ float y_sum[8];

    const unsigned t_row_a = lane >> 2;          // 0..7 within warp's sub-tile
    const unsigned t_row_b = t_row_a + 8u;       // 8..15
    const unsigned t_col_a = (lane & 3u) << 1;   // 0,2,4,6
    const unsigned t_col_b = t_col_a + 1u;       // 1,3,5,7

    const unsigned row_a_g = row_base + t_row_a;
    const unsigned row_b_g = row_base + t_row_b;
    const unsigned seq_a_g = cta_seq_base + t_col_a;
    const unsigned seq_b_g = cta_seq_base + t_col_b;
    const bool row_a_valid = row_a_g < rows;
    const bool row_b_valid = row_b_g < rows;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        float blk_d_a0=0, blk_d_a1=0, blk_d_b0=0, blk_d_b1=0;
        float blk_m_a0=0, blk_m_a1=0, blk_m_b0=0, blk_m_b1=0;

        for (unsigned j = 0; j < 8u; ++j) {
            // 1. Cooperative weight load: 128 threads × 16 nibble = 64 rows × 32 nibble.
            //    threadIdx.x → row_idx (0..63) = idx/2, byte_off = (idx%2)*16.
            const unsigned tidx = threadIdx.x;
            const unsigned al_row = tidx >> 1;            // 0..63
            const unsigned al_off = (tidx & 1u) * 16u;    // 0 or 16
            const unsigned al_row_g = cta_row_base + al_row;
            if (al_row_g < rows) {
                const unsigned char* bp = weights + al_row_g * row_bytes + b * 144u;
                if (j == 0u && al_off == 0u) {
                    const unsigned raw_d = (unsigned)bp[0] | ((unsigned)bp[1] << 8);
                    const unsigned raw_dmin = (unsigned)bp[2] | ((unsigned)bp[3] << 8);
                    x_d[al_row] = __half2float(__ushort_as_half((unsigned short)raw_d));
                    x_dmin[al_row] = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                }
                if (al_off == 0u) {
                    unsigned sc, mn;
                    if (j < 4u) {
                        sc = bp[4u+j] & 63u;
                        mn = bp[4u+j+4u] & 63u;
                    } else {
                        sc = (bp[4u+j+4u] & 0x0fu) | ((bp[4u+j-4u] >> 6) << 4);
                        mn = (bp[4u+j+4u] >> 4) | ((bp[4u+j] >> 6) << 4);
                    }
                    x_sc[al_row] = (unsigned char)sc;
                    x_mn[al_row] = (unsigned char)mn;
                }
                const unsigned nibble_base = 16u + (j >> 1) * 32u + al_off;
                const unsigned shift = (j & 1u) * 4u;
                const unsigned char* qb = bp + nibble_base;
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    a_tile[al_row * 32u + al_off + e] = (signed char)((qb[e] >> shift) & 0x0Fu);
                }
            }

            // 2. Load B tile (act). only warp 0 (32 threads × 8 byte = 256 bytes).
            if (warp == 0u) {
                const unsigned bl_seq = lane >> 2;
                const unsigned bl_kb = (lane & 3u) * 8u;
                const unsigned bl_seq_g = cta_seq_base + bl_seq;
                if (bl_seq_g < seq_len) {
                    const unsigned chunk_idx_global = (bl_seq_g * blocks_per_row + b) * 8u + j;
                    if (bl_kb == 0u) {
                        y_d[bl_seq] = input_ds[chunk_idx_global];
                        y_sum[bl_seq] = input_sums[chunk_idx_global];
                    }
                    const signed char* q8_ptr = input_qs
                        + bl_seq_g * blocks_per_row * 256u
                        + b * 256u + j * 32u + bl_kb;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        b_tile[bl_seq * 32u + bl_kb + e] = q8_ptr[e];
                    }
                }
            }

            __syncthreads();

            // 3. Pack A frag from warp's 16-row sub-tile.
            const unsigned ac_l = (lane & 3u) * 4u;     // 0,4,8,12
            const unsigned ac_h = ac_l + 16u;
            int a0 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_a) * 32u + ac_l]);
            int a1 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_b) * 32u + ac_l]);
            int a2 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_a) * 32u + ac_h]);
            int a3 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_b) * 32u + ac_h]);

            const unsigned b_seq = lane >> 2;
            const unsigned b_rl = (lane & 3u) * 4u;
            const unsigned b_rh = b_rl + 16u;
            int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_rl]);
            int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_rh]);

            int d0=0, d1=0, d2=0, d3=0;
            rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            const float sc_a = (float)x_sc[warp_row_off + t_row_a];
            const float sc_b = (float)x_sc[warp_row_off + t_row_b];
            const float mn_a = (float)x_mn[warp_row_off + t_row_a];
            const float mn_b = (float)x_mn[warp_row_off + t_row_b];
            const float dq_a = y_d[t_col_a];
            const float dq_b = y_d[t_col_b];
            const float sm_a = y_sum[t_col_a];
            const float sm_b = y_sum[t_col_b];

            blk_d_a0 += dq_a * sc_a * (float)d0;
            blk_d_a1 += dq_b * sc_a * (float)d1;
            blk_d_b0 += dq_a * sc_b * (float)d2;
            blk_d_b1 += dq_b * sc_b * (float)d3;
            blk_m_a0 += sm_a * mn_a;
            blk_m_a1 += sm_b * mn_a;
            blk_m_b0 += sm_a * mn_b;
            blk_m_b1 += sm_b * mn_b;

            __syncthreads();
        }

        const float d_a = x_d[warp_row_off + t_row_a];
        const float d_b = x_d[warp_row_off + t_row_b];
        const float dmin_a = x_dmin[warp_row_off + t_row_a];
        const float dmin_b = x_dmin[warp_row_off + t_row_b];
        acc[0] += d_a * blk_d_a0 - dmin_a * blk_m_a0;
        acc[1] += d_a * blk_d_a1 - dmin_a * blk_m_a1;
        acc[2] += d_b * blk_d_b0 - dmin_b * blk_m_b0;
        acc[3] += d_b * blk_d_b1 - dmin_b * blk_m_b1;
    }

    if (row_a_valid && seq_a_g < seq_len) out[seq_a_g * rows + row_a_g] = acc[0];
    if (row_a_valid && seq_b_g < seq_len) out[seq_b_g * rows + row_a_g] = acc[1];
    if (row_b_valid && seq_a_g < seq_len) out[seq_a_g * rows + row_b_g] = acc[2];
    if (row_b_valid && seq_b_g < seq_len) out[seq_b_g * rows + row_b_g] = acc[3];
#endif
}

// =============================================================================
// cu39 Phase 5: mma 4-warp variant for dense.rs q4k_batch_q8dot_to_dev integration.
//
// 진짜 prefill dominant dispatcher (nsys: rnb_q4k_gemv_batch_q8dot_seq4_warp8 =
// 39.9% GPU time) 가 `dense.rs:505 q4k_batch_q8dot_to_dev` 안에 있음. 그 signature
// 가 (qs, ds) 만 받고 sums 없음 → mma 4warp 도 sum 을 inline 으로 dp4a(0x01010101, qy)
// 로 계산.
//
// 다른 점 vs mma_4warp:
// - input_sums 파라미터 없음
// - 매 K-iter 마다 sum_qy 를 b_tile 에서 4 dp4a 로 계산
// - sum_x ≈ d_q8 * sum_qy (Q8_1 symmetric quantize)
// =============================================================================
extern "C" __global__ void rnb_q4k_q8_1_matmul_mma_4warp_v2(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
    (void)out; (void)weights; (void)input_qs; (void)input_ds;
    (void)rows; (void)blocks_per_row; (void)seq_len;
    return;
#else
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_row_off = warp * 16u;
    const unsigned cta_row_base = blockIdx.x * 64u;
    const unsigned cta_seq_base = blockIdx.y * 8u;
    const unsigned row_base = cta_row_base + warp_row_off;

    __shared__ signed char a_tile[64 * 32];
    __shared__ signed char b_tile[32 * 8];
    __shared__ float x_d[64];
    __shared__ float x_dmin[64];
    __shared__ unsigned char x_sc[64];
    __shared__ unsigned char x_mn[64];
    __shared__ float y_d[8];

    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;
    const unsigned t_col_a = (lane & 3u) << 1;
    const unsigned t_col_b = t_col_a + 1u;

    const unsigned row_a_g = row_base + t_row_a;
    const unsigned row_b_g = row_base + t_row_b;
    const unsigned seq_a_g = cta_seq_base + t_col_a;
    const unsigned seq_b_g = cta_seq_base + t_col_b;
    const bool row_a_valid = row_a_g < rows;
    const bool row_b_valid = row_b_g < rows;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        float blk_d_a0=0, blk_d_a1=0, blk_d_b0=0, blk_d_b1=0;
        float blk_m_a0=0, blk_m_a1=0, blk_m_b0=0, blk_m_b1=0;

        for (unsigned j = 0; j < 8u; ++j) {
            // 1. Load A tile (64 row × 32 nibble, cooperative).
            const unsigned tidx = threadIdx.x;
            const unsigned al_row = tidx >> 1;
            const unsigned al_off = (tidx & 1u) * 16u;
            const unsigned al_row_g = cta_row_base + al_row;
            if (al_row_g < rows) {
                const unsigned char* bp = weights + al_row_g * row_bytes + b * 144u;
                if (j == 0u && al_off == 0u) {
                    const unsigned raw_d = (unsigned)bp[0] | ((unsigned)bp[1] << 8);
                    const unsigned raw_dmin = (unsigned)bp[2] | ((unsigned)bp[3] << 8);
                    x_d[al_row] = __half2float(__ushort_as_half((unsigned short)raw_d));
                    x_dmin[al_row] = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                }
                if (al_off == 0u) {
                    unsigned sc, mn;
                    if (j < 4u) {
                        sc = bp[4u+j] & 63u;
                        mn = bp[4u+j+4u] & 63u;
                    } else {
                        sc = (bp[4u+j+4u] & 0x0fu) | ((bp[4u+j-4u] >> 6) << 4);
                        mn = (bp[4u+j+4u] >> 4) | ((bp[4u+j] >> 6) << 4);
                    }
                    x_sc[al_row] = (unsigned char)sc;
                    x_mn[al_row] = (unsigned char)mn;
                }
                const unsigned nibble_base = 16u + (j >> 1) * 32u + al_off;
                const unsigned shift = (j & 1u) * 4u;
                const unsigned char* qb = bp + nibble_base;
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    a_tile[al_row * 32u + al_off + e] = (signed char)((qb[e] >> shift) & 0x0Fu);
                }
            }

            // 2. Load B tile (act 32×8 col-major) + d_q8.
            if (warp == 0u) {
                const unsigned bl_seq = lane >> 2;
                const unsigned bl_kb = (lane & 3u) * 8u;
                const unsigned bl_seq_g = cta_seq_base + bl_seq;
                if (bl_seq_g < seq_len) {
                    const unsigned chunk_idx_global = (bl_seq_g * blocks_per_row + b) * 8u + j;
                    if (bl_kb == 0u) {
                        y_d[bl_seq] = input_ds[chunk_idx_global];
                    }
                    const signed char* q8_ptr = input_qs
                        + bl_seq_g * blocks_per_row * 256u
                        + b * 256u + j * 32u + bl_kb;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        b_tile[bl_seq * 32u + bl_kb + e] = q8_ptr[e];
                    }
                }
            }

            __syncthreads();

            // 3. Pack A frag from warp's 16-row sub-tile.
            const unsigned ac_l = (lane & 3u) * 4u;
            const unsigned ac_h = ac_l + 16u;
            int a0 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_a) * 32u + ac_l]);
            int a1 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_b) * 32u + ac_l]);
            int a2 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_a) * 32u + ac_h]);
            int a3 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_b) * 32u + ac_h]);

            // 4. Pack B frag.
            const unsigned b_seq = lane >> 2;
            const unsigned b_rl = (lane & 3u) * 4u;
            const unsigned b_rh = b_rl + 16u;
            int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_rl]);
            int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_rh]);

            // 5. mma.
            int d0=0, d1=0, d2=0, d3=0;
            rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            // 6. Inline sum_qy compute per (t_col_a, t_col_b) seq.
            //    sum_qy = sum of 32 int8 in b_tile[seq * 32 + 0..31]
            //    using __dp4a(0x01010101, b_int, acc) trick.
            int sum_qy_a = 0;
            int sum_qy_b = 0;
            const bool seq_a_valid_inner = seq_a_g < seq_len;
            const bool seq_b_valid_inner = seq_b_g < seq_len;
            #pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (seq_a_valid_inner) {
                    int yi_a = *reinterpret_cast<const int*>(&b_tile[t_col_a * 32u + k]);
                    sum_qy_a = __dp4a(0x01010101, yi_a, sum_qy_a);
                }
                if (seq_b_valid_inner) {
                    int yi_b = *reinterpret_cast<const int*>(&b_tile[t_col_b * 32u + k]);
                    sum_qy_b = __dp4a(0x01010101, yi_b, sum_qy_b);
                }
            }

            // 7. Scale, accumulate.
            const float sc_a = (float)x_sc[warp_row_off + t_row_a];
            const float sc_b = (float)x_sc[warp_row_off + t_row_b];
            const float mn_a = (float)x_mn[warp_row_off + t_row_a];
            const float mn_b = (float)x_mn[warp_row_off + t_row_b];
            const float dq_a = seq_a_valid_inner ? y_d[t_col_a] : 0.0f;
            const float dq_b = seq_b_valid_inner ? y_d[t_col_b] : 0.0f;

            blk_d_a0 += dq_a * sc_a * (float)d0;
            blk_d_a1 += dq_b * sc_a * (float)d1;
            blk_d_b0 += dq_a * sc_b * (float)d2;
            blk_d_b1 += dq_b * sc_b * (float)d3;
            blk_m_a0 += dq_a * (float)sum_qy_a * mn_a;
            blk_m_a1 += dq_b * (float)sum_qy_b * mn_a;
            blk_m_b0 += dq_a * (float)sum_qy_a * mn_b;
            blk_m_b1 += dq_b * (float)sum_qy_b * mn_b;

            __syncthreads();
        }

        const float d_a = x_d[warp_row_off + t_row_a];
        const float d_b = x_d[warp_row_off + t_row_b];
        const float dmin_a = x_dmin[warp_row_off + t_row_a];
        const float dmin_b = x_dmin[warp_row_off + t_row_b];
        acc[0] += d_a * blk_d_a0 - dmin_a * blk_m_a0;
        acc[1] += d_a * blk_d_a1 - dmin_a * blk_m_a1;
        acc[2] += d_b * blk_d_b0 - dmin_b * blk_m_b0;
        acc[3] += d_b * blk_d_b1 - dmin_b * blk_m_b1;
    }

    if (row_a_valid && seq_a_g < seq_len) out[seq_a_g * rows + row_a_g] = acc[0];
    if (row_a_valid && seq_b_g < seq_len) out[seq_b_g * rows + row_a_g] = acc[1];
    if (row_b_valid && seq_a_g < seq_len) out[seq_a_g * rows + row_b_g] = acc[2];
    if (row_b_valid && seq_b_g < seq_len) out[seq_b_g * rows + row_b_g] = acc[3];
#endif
}

// =============================================================================
// cu39 Phase 6: mma 4-warp v3 — packed nibble unpack (mmq.cuh pattern).
//
// v2 weight unpack = 16-iter byte loop. v3 = 4-int load + bit shift (mmq pattern).
// Q4_K nibble layout 의 16-byte block (al_off 단위) = 4 int (4-byte aligned).
// 각 int holds 4 nibble-pairs (low + high). Mask 0x0F0F0F0F = 4 low nibbles in
// int8. Shift 4 → high nibbles.
// =============================================================================
extern "C" __global__ void rnb_q4k_q8_1_matmul_mma_4warp_v3(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
#if __CUDA_ARCH__ < 800
    (void)out; (void)weights; (void)input_qs; (void)input_ds;
    (void)rows; (void)blocks_per_row; (void)seq_len;
    return;
#else
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_row_off = warp * 16u;
    const unsigned cta_row_base = blockIdx.x * 64u;
    const unsigned cta_seq_base = blockIdx.y * 8u;
    const unsigned row_base = cta_row_base + warp_row_off;

    __shared__ signed char a_tile[64 * 32];
    __shared__ signed char b_tile[32 * 8];
    __shared__ float x_d[64];
    __shared__ float x_dmin[64];
    __shared__ unsigned char x_sc[64];
    __shared__ unsigned char x_mn[64];
    __shared__ float y_d[8];

    const unsigned t_row_a = lane >> 2;
    const unsigned t_row_b = t_row_a + 8u;
    const unsigned t_col_a = (lane & 3u) << 1;
    const unsigned t_col_b = t_col_a + 1u;

    const unsigned row_a_g = row_base + t_row_a;
    const unsigned row_b_g = row_base + t_row_b;
    const unsigned seq_a_g = cta_seq_base + t_col_a;
    const unsigned seq_b_g = cta_seq_base + t_col_b;
    const bool row_a_valid = row_a_g < rows;
    const bool row_b_valid = row_b_g < rows;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        float blk_d_a0=0, blk_d_a1=0, blk_d_b0=0, blk_d_b1=0;
        float blk_m_a0=0, blk_m_a1=0, blk_m_b0=0, blk_m_b1=0;

        for (unsigned j = 0; j < 8u; ++j) {
            // 1. Load A tile — PACKED unpack (4 int load + bit shift).
            const unsigned tidx = threadIdx.x;
            const unsigned al_row = tidx >> 1;
            const unsigned al_off = (tidx & 1u) * 16u;
            const unsigned al_row_g = cta_row_base + al_row;
            if (al_row_g < rows) {
                const unsigned char* bp = weights + al_row_g * row_bytes + b * 144u;
                if (j == 0u && al_off == 0u) {
                    const unsigned raw_d = (unsigned)bp[0] | ((unsigned)bp[1] << 8);
                    const unsigned raw_dmin = (unsigned)bp[2] | ((unsigned)bp[3] << 8);
                    x_d[al_row] = __half2float(__ushort_as_half((unsigned short)raw_d));
                    x_dmin[al_row] = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                }
                if (al_off == 0u) {
                    unsigned sc, mn;
                    if (j < 4u) {
                        sc = bp[4u+j] & 63u;
                        mn = bp[4u+j+4u] & 63u;
                    } else {
                        sc = (bp[4u+j+4u] & 0x0fu) | ((bp[4u+j-4u] >> 6) << 4);
                        mn = (bp[4u+j+4u] >> 4) | ((bp[4u+j] >> 6) << 4);
                    }
                    x_sc[al_row] = (unsigned char)sc;
                    x_mn[al_row] = (unsigned char)mn;
                }
                const unsigned nibble_base = 16u + (j >> 1) * 32u + al_off;
                const unsigned shift = (j & 1u) * 4u;
                const int* qb_int = reinterpret_cast<const int*>(bp + nibble_base);
                int* a_dst_int = reinterpret_cast<int*>(&a_tile[al_row * 32u + al_off]);
                if (shift == 0u) {
                    a_dst_int[0] = qb_int[0] & 0x0F0F0F0F;
                    a_dst_int[1] = qb_int[1] & 0x0F0F0F0F;
                    a_dst_int[2] = qb_int[2] & 0x0F0F0F0F;
                    a_dst_int[3] = qb_int[3] & 0x0F0F0F0F;
                } else {
                    a_dst_int[0] = (qb_int[0] >> 4) & 0x0F0F0F0F;
                    a_dst_int[1] = (qb_int[1] >> 4) & 0x0F0F0F0F;
                    a_dst_int[2] = (qb_int[2] >> 4) & 0x0F0F0F0F;
                    a_dst_int[3] = (qb_int[3] >> 4) & 0x0F0F0F0F;
                }
            }

            // 2. Load B tile (packed 2-int) + d_q8.
            if (warp == 0u) {
                const unsigned bl_seq = lane >> 2;
                const unsigned bl_kb = (lane & 3u) * 8u;
                const unsigned bl_seq_g = cta_seq_base + bl_seq;
                if (bl_seq_g < seq_len) {
                    const unsigned chunk_idx_global = (bl_seq_g * blocks_per_row + b) * 8u + j;
                    if (bl_kb == 0u) {
                        y_d[bl_seq] = input_ds[chunk_idx_global];
                    }
                    const signed char* q8_ptr = input_qs
                        + bl_seq_g * blocks_per_row * 256u
                        + b * 256u + j * 32u + bl_kb;
                    int* b_dst_int = reinterpret_cast<int*>(&b_tile[bl_seq * 32u + bl_kb]);
                    const int* q8_int = reinterpret_cast<const int*>(q8_ptr);
                    b_dst_int[0] = q8_int[0];
                    b_dst_int[1] = q8_int[1];
                }
            }

            __syncthreads();

            // 3. Pack A frag.
            const unsigned ac_l = (lane & 3u) * 4u;
            const unsigned ac_h = ac_l + 16u;
            int a0 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_a) * 32u + ac_l]);
            int a1 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_b) * 32u + ac_l]);
            int a2 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_a) * 32u + ac_h]);
            int a3 = *reinterpret_cast<const int*>(&a_tile[(warp_row_off + t_row_b) * 32u + ac_h]);

            const unsigned b_seq = lane >> 2;
            const unsigned b_rl = (lane & 3u) * 4u;
            const unsigned b_rh = b_rl + 16u;
            int b0 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_rl]);
            int b1 = *reinterpret_cast<const int*>(&b_tile[b_seq * 32u + b_rh]);

            int d0=0, d1=0, d2=0, d3=0;
            rnb_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            int sum_qy_a = 0;
            int sum_qy_b = 0;
            const bool seq_a_valid_inner = seq_a_g < seq_len;
            const bool seq_b_valid_inner = seq_b_g < seq_len;
            #pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (seq_a_valid_inner) {
                    int yi_a = *reinterpret_cast<const int*>(&b_tile[t_col_a * 32u + k]);
                    sum_qy_a = __dp4a(0x01010101, yi_a, sum_qy_a);
                }
                if (seq_b_valid_inner) {
                    int yi_b = *reinterpret_cast<const int*>(&b_tile[t_col_b * 32u + k]);
                    sum_qy_b = __dp4a(0x01010101, yi_b, sum_qy_b);
                }
            }

            const float sc_a = (float)x_sc[warp_row_off + t_row_a];
            const float sc_b = (float)x_sc[warp_row_off + t_row_b];
            const float mn_a = (float)x_mn[warp_row_off + t_row_a];
            const float mn_b = (float)x_mn[warp_row_off + t_row_b];
            const float dq_a = seq_a_valid_inner ? y_d[t_col_a] : 0.0f;
            const float dq_b = seq_b_valid_inner ? y_d[t_col_b] : 0.0f;

            blk_d_a0 += dq_a * sc_a * (float)d0;
            blk_d_a1 += dq_b * sc_a * (float)d1;
            blk_d_b0 += dq_a * sc_b * (float)d2;
            blk_d_b1 += dq_b * sc_b * (float)d3;
            blk_m_a0 += dq_a * (float)sum_qy_a * mn_a;
            blk_m_a1 += dq_b * (float)sum_qy_b * mn_a;
            blk_m_b0 += dq_a * (float)sum_qy_a * mn_b;
            blk_m_b1 += dq_b * (float)sum_qy_b * mn_b;

            __syncthreads();
        }

        const float d_a = x_d[warp_row_off + t_row_a];
        const float d_b = x_d[warp_row_off + t_row_b];
        const float dmin_a = x_dmin[warp_row_off + t_row_a];
        const float dmin_b = x_dmin[warp_row_off + t_row_b];
        acc[0] += d_a * blk_d_a0 - dmin_a * blk_m_a0;
        acc[1] += d_a * blk_d_a1 - dmin_a * blk_m_a1;
        acc[2] += d_b * blk_d_b0 - dmin_b * blk_m_b0;
        acc[3] += d_b * blk_d_b1 - dmin_b * blk_m_b1;
    }

    if (row_a_valid && seq_a_g < seq_len) out[seq_a_g * rows + row_a_g] = acc[0];
    if (row_a_valid && seq_b_g < seq_len) out[seq_b_g * rows + row_a_g] = acc[1];
    if (row_b_valid && seq_a_g < seq_len) out[seq_a_g * rows + row_b_g] = acc[2];
    if (row_b_valid && seq_b_g < seq_len) out[seq_b_g * rows + row_b_g] = acc[3];
#endif
}


#include "quant_megakernel.cuh"
#include "decode_device.cuh"
