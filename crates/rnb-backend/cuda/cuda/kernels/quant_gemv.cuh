__device__ __forceinline__ int rnb_load_i32_aligned4(const void* ptr) {
    return *reinterpret_cast<const int*>(ptr);
}

__device__ __forceinline__ int rnb_load_i32_unaligned(const void* ptr) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(ptr);
    return (int)((unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24));
}

__device__ __forceinline__ int rnb_q4_pack4(const unsigned char* ptr, unsigned j) {
    const unsigned raw = *reinterpret_cast<const unsigned*>(ptr);
    const unsigned unpacked = ((j & 1u) == 0u ? raw : (raw >> 4)) & 0x0f0f0f0fu;
    return (int)unpacked;
}

__device__ __forceinline__ float rnb_iq4nl_value(unsigned q) {
    switch (q & 0x0fu) {
        case 0u: return -127.0f;
        case 1u: return -104.0f;
        case 2u: return -83.0f;
        case 3u: return -65.0f;
        case 4u: return -49.0f;
        case 5u: return -35.0f;
        case 6u: return -22.0f;
        case 7u: return -10.0f;
        case 8u: return 1.0f;
        case 9u: return 13.0f;
        case 10u: return 25.0f;
        case 11u: return 38.0f;
        case 12u: return 53.0f;
        case 13u: return 69.0f;
        case 14u: return 89.0f;
        default: return 113.0f;
    }
}

extern "C" __global__ void rnb_q2k_gemv_warp8(
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
    const unsigned row_bytes = blocks_per_row * 84u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 84u;
            const unsigned raw_d = (unsigned)block[80] | ((unsigned)block[81] << 8);
            const unsigned raw_dmin = (unsigned)block[82] | ((unsigned)block[83] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned scale_idx = tid >> 4;
                const unsigned scale_min = block[scale_idx];
                const unsigned q_index = (tid >> 7) * 32u + (tid & 31u);
                const unsigned shift = ((tid & 127u) >> 5) * 2u;
                const unsigned q = (block[16u + q_index] >> shift) & 3u;
                const float scale = d * (float)(scale_min & 0x0fu);
                const float min_value = dmin * (float)(scale_min >> 4);
                const float value = scale * (float)q - min_value;
                acc += value * input[b * 256u + tid];
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

extern "C" __global__ void rnb_q3k_gemv_warp8(
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
    const unsigned row_bytes = blocks_per_row * 110u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 110u;
            const unsigned raw_d = (unsigned)block[108] | ((unsigned)block[109] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned scale_idx = tid >> 4;
                const unsigned scale_lane = scale_idx & 3u;
                const unsigned packed_high = block[96u + 8u + scale_lane];
                unsigned scale_code;
                if (scale_idx < 4u) {
                    scale_code =
                        (block[96u + scale_lane] & 0x0fu) | ((packed_high & 0x03u) << 4);
                } else if (scale_idx < 8u) {
                    scale_code =
                        (block[96u + 4u + scale_lane] & 0x0fu)
                        | (((packed_high >> 2) & 0x03u) << 4);
                } else if (scale_idx < 12u) {
                    scale_code =
                        (block[96u + scale_lane] >> 4)
                        | (((packed_high >> 4) & 0x03u) << 4);
                } else {
                    scale_code =
                        (block[96u + 4u + scale_lane] >> 4)
                        | (((packed_high >> 6) & 0x03u) << 4);
                }

                const unsigned q_index = (tid >> 7) * 32u + (tid & 31u);
                const unsigned shift = ((tid & 127u) >> 5) * 2u;
                const unsigned q = (block[32u + q_index] >> shift) & 3u;
                const unsigned high_mask = 1u << (tid >> 5);
                const int high = (block[tid & 31u] & high_mask) != 0u ? 0 : 4;
                const int signed_scale = (int)scale_code - 32;
                const float value = d * (float)signed_scale * (float)((int)q - high);
                acc += value * input[b * 256u + tid];
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

extern "C" __global__ void rnb_q4k_gemv(
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

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;

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

extern "C" __global__ void rnb_q4k_dequant_f16(
    __half* __restrict__ out,
    const unsigned char* __restrict__ weights,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }
    const unsigned cols = blocks_per_row * 256u;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    __half* out_row = out + row * cols;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 144u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

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
        unsigned q = block[16 + q_index];
        q = local < 32u ? (q & 0x0fu) : (q >> 4);

        const float scale = d * (float)sc;
        const float min_val = dmin * (float)mn;
        const float prod = scale * (float)q;
        const float y = prod - min_val;
        out_row[b * 256u + tid] = __float2half(y);
    }
}

// cu19: same as rnb_q4k_dequant_f16 but writes f32 output.
// Used by q4k_f32_gemm_batch_cached fallback so weight upload becomes
// host-bytes-only (Q4_K layout, 0.625 byte/element) instead of host-dequant
// f32 + 4-byte H2D — a 6.4x reduction in PCIe weight traffic per prefill call.
extern "C" __global__ void rnb_q4k_dequant_f32(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }
    const unsigned cols = blocks_per_row * 256u;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    float* out_row = out + row * cols;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 144u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

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
        unsigned q = block[16 + q_index];
        q = local < 32u ? (q & 0x0fu) : (q >> 4);

        const float scale = d * (float)sc;
        const float min_val = dmin * (float)mn;
        const float prod = scale * (float)q;
        out_row[b * 256u + tid] = prod - min_val;
    }
}

static __device__ __forceinline__ float rnb_q4k_block_dot_f32_lane(
    const unsigned char* __restrict__ block,
    const float* __restrict__ input,
    unsigned lane) {
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
    float acc = 0.0f;
#pragma unroll
    for (unsigned group = 0; group < 4u; ++group) {
        const unsigned j0 = group * 2u;
        const unsigned j1 = j0 + 1u;
        unsigned sc0;
        unsigned mn0;
        unsigned sc1;
        unsigned mn1;
        if (j0 < 4u) {
            sc0 = block[4u + j0] & 63u;
            mn0 = block[8u + j0] & 63u;
            sc1 = block[4u + j1] & 63u;
            mn1 = block[8u + j1] & 63u;
        } else {
            sc0 = (block[8u + j0] & 0x0fu) | ((block[j0] >> 6) << 4);
            mn0 = (block[8u + j0] >> 4) | ((block[4u + j0] >> 6) << 4);
            sc1 = (block[8u + j1] & 0x0fu) | ((block[j1] >> 6) << 4);
            mn1 = (block[8u + j1] >> 4) | ((block[4u + j1] >> 6) << 4);
        }
        const unsigned q = block[16u + group * 32u + lane];
        const float y0 = (d * (float)sc0) * (float)(q & 0x0fu) - dmin * (float)mn0;
        const float y1 = (d * (float)sc1) * (float)(q >> 4) - dmin * (float)mn1;
        acc += y0 * input[group * 64u + lane];
        acc += y1 * input[group * 64u + lane + 32u];
    }
    return acc;
}

extern "C" __global__ void rnb_q4k_gemv_warp8(
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
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 144u;
            acc += rnb_q4k_block_dot_f32_lane(block, input + b * 256u, lane);
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q4k_embedding_gather_f32(
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

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* block = weights + row * row_bytes + block_idx * 144u;
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

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
    out[token_idx * blocks_per_row * 256u + block_idx * 256u + tid] = y;
}

extern "C" __global__ void rnb_iq4_xs_gemv_warp8(
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
    const unsigned row_bytes = blocks_per_row * 136u;
    const unsigned char* row_ptr = weights + row * row_bytes;

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
                acc += dl * rnb_iq4nl_value(q) * input[b * 256u + tid];
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

extern "C" __global__ void rnb_q4k_gemv_q8dot_warp8(
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
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 144u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

            const unsigned j0 = lane >> 3;
            const unsigned j1 = j0 + 4u;
            const unsigned elem = (lane & 7u) * 4u;
            const unsigned sc0 = block[4u + j0] & 63u;
            const unsigned mn0 = block[8u + j0] & 63u;
            const unsigned sc1 =
                (block[8u + j1] & 0x0fu) | ((block[j1] >> 6) << 4);
            const unsigned mn1 =
                (block[8u + j1] >> 4) | ((block[4u + j1] >> 6) << 4);
            const unsigned char* q_ptr0 =
                block + 16u + (j0 >> 1) * 32u + elem;
            const unsigned char* q_ptr1 = q_ptr0 + 64u;
            const signed char* x_qs0 =
                input_qs + b * 256u + j0 * 32u + elem;
            const signed char* x_qs1 = x_qs0 + 128u;
            const int q_pack0 = rnb_q4_pack4(q_ptr0, j0);
            const int q_pack1 = rnb_q4_pack4(q_ptr1, j1);
            const int x_pack0 = rnb_load_i32_aligned4(x_qs0);
            const int x_pack1 = rnb_load_i32_aligned4(x_qs1);
            const int dot0 = __dp4a(q_pack0, x_pack0, 0);
            const int dot1 = __dp4a(q_pack1, x_pack1, 0);
            const int x_sum0 = __dp4a(0x01010101, x_pack0, 0);
            const int x_sum1 = __dp4a(0x01010101, x_pack1, 0);
            acc += input_ds[b * 8u + j0]
                * ((d * (float)sc0) * (float)dot0 - dmin * (float)mn0 * (float)x_sum0);
            acc += input_ds[b * 8u + j1]
                * ((d * (float)sc1) * (float)dot1 - dmin * (float)mn1 * (float)x_sum1);
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q4k_gemv_gelu_mul_warp8(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ mul,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights + row * row_bytes;

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
                acc += y * input[b * 256u + tid];
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        const float x = acc;
        const float x3 = x * x * x;
        const float c = 0.7978845608028654f;
        const float gelu = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
        out[row] = gelu * mul[row];
    }
}

extern "C" __global__ void rnb_q4k_gate_up_gemv_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* __restrict__ gate_weights,
    const unsigned char* __restrict__ up_weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row_ptr = gate_weights + row * row_bytes;
    const unsigned char* up_row_ptr = up_weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* gate_block = gate_row_ptr + b * 144u;
            const unsigned char* up_block = up_row_ptr + b * 144u;
            const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
            const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
            const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
            const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
            const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
            const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
            const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
            const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned j = tid >> 5;
                unsigned gate_sc;
                unsigned gate_mn;
                unsigned up_sc;
                unsigned up_mn;
                if (j < 4u) {
                    gate_sc = gate_block[4u + j] & 63u;
                    gate_mn = gate_block[4u + j + 4u] & 63u;
                    up_sc = up_block[4u + j] & 63u;
                    up_mn = up_block[4u + j + 4u] & 63u;
                } else {
                    gate_sc = (gate_block[4u + j + 4u] & 0x0fu) | ((gate_block[4u + j - 4u] >> 6) << 4);
                    gate_mn = (gate_block[4u + j + 4u] >> 4) | ((gate_block[4u + j] >> 6) << 4);
                    up_sc = (up_block[4u + j + 4u] & 0x0fu) | ((up_block[4u + j - 4u] >> 6) << 4);
                    up_mn = (up_block[4u + j + 4u] >> 4) | ((up_block[4u + j] >> 6) << 4);
                }

                const unsigned local = tid & 63u;
                const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
                unsigned gate_q = gate_block[16u + q_index];
                unsigned up_q = up_block[16u + q_index];
                gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
                up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);

                const float x = input[b * 256u + tid];
                gate_acc += ((gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn) * x;
                up_acc += ((up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn) * x;
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }

    if (valid && lane == 0u) {
        gate_out[row] = gate_acc;
        up_out[row] = up_acc;
    }
}

extern "C" __global__ void rnb_q4k_packed_gemv_q8dot_warp8(
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
    const unsigned row_bytes = blocks_per_row * 148u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 148u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned j = chunk >> 3;
                const unsigned elem = (chunk & 7u) * 4u;
                const unsigned q_index = (j >> 1) * 32u + elem;
                const unsigned char* q_ptr = block + 20u + q_index;
                const signed char* x_qs = input_qs + b * 256u + j * 32u + elem;
                const unsigned scmn =
                    (unsigned)block[4u + j * 2u] | ((unsigned)block[5u + j * 2u] << 8);
                const unsigned sc = scmn & 0xffu;
                const unsigned mn = scmn >> 8;
                const int q_pack = rnb_q4_pack4(q_ptr, j);
                const int x_pack = rnb_load_i32_aligned4(x_qs);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = input_ds[b * 8u + j];
                acc += x_d * ((d * (float)sc) * (float)dot - dmin * (float)mn * (float)x_sum);
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

extern "C" __global__ void rnb_q4k_packed_gemv_q8dot_warp4(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned row = blockIdx.x;
    const bool valid = row < rows;

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 148u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = warp; b < blocks_per_row; b += 4u) {
            const unsigned char* block = row_ptr + b * 148u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned j = chunk >> 3;
                const unsigned elem = (chunk & 7u) * 4u;
                const unsigned q_index = (j >> 1) * 32u + elem;
                const unsigned char* q_ptr = block + 20u + q_index;
                const signed char* x_qs = input_qs + b * 256u + j * 32u + elem;
                const unsigned scmn =
                    (unsigned)block[4u + j * 2u] | ((unsigned)block[5u + j * 2u] << 8);
                const unsigned sc = scmn & 0xffu;
                const unsigned mn = scmn >> 8;
                const int q_pack = rnb_q4_pack4(q_ptr, j);
                const int x_pack = rnb_load_i32_aligned4(x_qs);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = input_ds[b * 8u + j];
                acc += x_d * ((d * (float)sc) * (float)dot - dmin * (float)mn * (float)x_sum);
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

extern "C" __global__ void rnb_q4k_gate_up_gemv_q8dot_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* __restrict__ gate_weights,
    const unsigned char* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row_ptr = gate_weights + row * row_bytes;
    const unsigned char* up_row_ptr = up_weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* gate_block = gate_row_ptr + b * 144u;
            const unsigned char* up_block = up_row_ptr + b * 144u;
            const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
            const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
            const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
            const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
            const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
            const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
            const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
            const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned j = chunk >> 3;
                unsigned gate_sc;
                unsigned gate_mn;
                unsigned up_sc;
                unsigned up_mn;
                if (j < 4u) {
                    gate_sc = gate_block[4u + j] & 63u;
                    gate_mn = gate_block[4u + j + 4u] & 63u;
                    up_sc = up_block[4u + j] & 63u;
                    up_mn = up_block[4u + j + 4u] & 63u;
                } else {
                    gate_sc = (gate_block[4u + j + 4u] & 0x0fu) | ((gate_block[4u + j - 4u] >> 6) << 4);
                    gate_mn = (gate_block[4u + j + 4u] >> 4) | ((gate_block[4u + j] >> 6) << 4);
                    up_sc = (up_block[4u + j + 4u] & 0x0fu) | ((up_block[4u + j - 4u] >> 6) << 4);
                    up_mn = (up_block[4u + j + 4u] >> 4) | ((up_block[4u + j] >> 6) << 4);
                }

                const unsigned elem = (chunk & 7u) * 4u;
                const unsigned q_index = (j >> 1) * 32u + elem;
                const unsigned char* gate_qs = gate_block + 16u + q_index;
                const unsigned char* up_qs = up_block + 16u + q_index;
                const signed char* x_qs = input_qs + b * 256u + j * 32u + elem;
                const int gate_pack = rnb_q4_pack4(gate_qs, j);
                const int up_pack = rnb_q4_pack4(up_qs, j);
                const int x_pack = rnb_load_i32_aligned4(x_qs);
                const int gate_dot = __dp4a(gate_pack, x_pack, 0);
                const int up_dot = __dp4a(up_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = input_ds[b * 8u + j];
                gate_acc += x_d * ((gate_d * (float)gate_sc) * (float)gate_dot - gate_dmin * (float)gate_mn * (float)x_sum);
                up_acc += x_d * ((up_d * (float)up_sc) * (float)up_dot - up_dmin * (float)up_mn * (float)x_sum);
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }

    if (valid && lane == 0u) {
        gate_out[row] = gate_acc;
        up_out[row] = up_acc;
    }
}

extern "C" __global__ void rnb_q4k_gate_up_gemv_batch_seq2_q8dot_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* __restrict__ gate_weights,
    const unsigned char* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float gate_acc0 = 0.0f;
    float up_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc1 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row_ptr = gate_weights + row * row_bytes;
    const unsigned char* up_row_ptr = up_weights + row * row_bytes;
    const signed char* input_qs0 = input_qs;
    const signed char* input_qs1 = input_qs + blocks_per_row * 256u;
    const float* input_ds0 = input_ds;
    const float* input_ds1 = input_ds + blocks_per_row * 8u;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* gate_block = gate_row_ptr + b * 144u;
            const unsigned char* up_block = up_row_ptr + b * 144u;
            const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
            const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
            const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
            const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
            const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
            const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
            const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
            const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned j = chunk >> 3;
                unsigned gate_sc;
                unsigned gate_mn;
                unsigned up_sc;
                unsigned up_mn;
                if (j < 4u) {
                    gate_sc = gate_block[4u + j] & 63u;
                    gate_mn = gate_block[4u + j + 4u] & 63u;
                    up_sc = up_block[4u + j] & 63u;
                    up_mn = up_block[4u + j + 4u] & 63u;
                } else {
                    gate_sc = (gate_block[4u + j + 4u] & 0x0fu) | ((gate_block[4u + j - 4u] >> 6) << 4);
                    gate_mn = (gate_block[4u + j + 4u] >> 4) | ((gate_block[4u + j] >> 6) << 4);
                    up_sc = (up_block[4u + j + 4u] & 0x0fu) | ((up_block[4u + j - 4u] >> 6) << 4);
                    up_mn = (up_block[4u + j + 4u] >> 4) | ((up_block[4u + j] >> 6) << 4);
                }

                const unsigned elem = (chunk & 7u) * 4u;
                const unsigned q_index = (j >> 1) * 32u + elem;
                const unsigned char* gate_qs = gate_block + 16u + q_index;
                const unsigned char* up_qs = up_block + 16u + q_index;
                const int gate_pack = rnb_q4_pack4(gate_qs, j);
                const int up_pack = rnb_q4_pack4(up_qs, j);
                const int x_pack0 = rnb_load_i32_aligned4(input_qs0 + b * 256u + j * 32u + elem);
                const int x_pack1 = rnb_load_i32_aligned4(input_qs1 + b * 256u + j * 32u + elem);
                const int gate_dot0 = __dp4a(gate_pack, x_pack0, 0);
                const int up_dot0 = __dp4a(up_pack, x_pack0, 0);
                const int x_sum0 = __dp4a(0x01010101, x_pack0, 0);
                const int gate_dot1 = __dp4a(gate_pack, x_pack1, 0);
                const int up_dot1 = __dp4a(up_pack, x_pack1, 0);
                const int x_sum1 = __dp4a(0x01010101, x_pack1, 0);
                const float x_d0 = input_ds0[b * 8u + j];
                const float x_d1 = input_ds1[b * 8u + j];
                gate_acc0 += x_d0 * ((gate_d * (float)gate_sc) * (float)gate_dot0 - gate_dmin * (float)gate_mn * (float)x_sum0);
                up_acc0 += x_d0 * ((up_d * (float)up_sc) * (float)up_dot0 - up_dmin * (float)up_mn * (float)x_sum0);
                gate_acc1 += x_d1 * ((gate_d * (float)gate_sc) * (float)gate_dot1 - gate_dmin * (float)gate_mn * (float)x_sum1);
                up_acc1 += x_d1 * ((up_d * (float)up_sc) * (float)up_dot1 - up_dmin * (float)up_mn * (float)x_sum1);
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
    }

    if (valid && lane == 0u) {
        gate_out[row] = gate_acc0;
        up_out[row] = up_acc0;
        gate_out[rows + row] = gate_acc1;
        up_out[rows + row] = up_acc1;
    }
}

extern "C" __global__ void rnb_q4k_packed_gate_up_gemv_q8dot_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* __restrict__ gate_weights,
    const unsigned char* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 148u;
    const unsigned char* gate_row_ptr = gate_weights + row * row_bytes;
    const unsigned char* up_row_ptr = up_weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* gate_block = gate_row_ptr + b * 148u;
            const unsigned char* up_block = up_row_ptr + b * 148u;
            const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
            const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
            const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
            const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
            const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
            const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
            const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
            const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned j = chunk >> 3;
                const unsigned elem = (chunk & 7u) * 4u;
                const unsigned q_index = (j >> 1) * 32u + elem;
                const unsigned char* gate_q_ptr = gate_block + 20u + q_index;
                const unsigned char* up_q_ptr = up_block + 20u + q_index;
                const signed char* x_qs = input_qs + b * 256u + j * 32u + elem;
                const unsigned gate_scmn =
                    (unsigned)gate_block[4u + j * 2u] | ((unsigned)gate_block[5u + j * 2u] << 8);
                const unsigned up_scmn =
                    (unsigned)up_block[4u + j * 2u] | ((unsigned)up_block[5u + j * 2u] << 8);
                const unsigned gate_sc = gate_scmn & 0xffu;
                const unsigned gate_mn = gate_scmn >> 8;
                const unsigned up_sc = up_scmn & 0xffu;
                const unsigned up_mn = up_scmn >> 8;
                const int gate_pack = rnb_q4_pack4(gate_q_ptr, j);
                const int up_pack = rnb_q4_pack4(up_q_ptr, j);
                const int x_pack = rnb_load_i32_aligned4(x_qs);
                const int gate_dot = __dp4a(gate_pack, x_pack, 0);
                const int up_dot = __dp4a(up_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = input_ds[b * 8u + j];
                gate_acc += x_d * ((gate_d * (float)gate_sc) * (float)gate_dot - gate_dmin * (float)gate_mn * (float)x_sum);
                up_acc += x_d * ((up_d * (float)up_sc) * (float)up_dot - up_dmin * (float)up_mn * (float)x_sum);
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }

    if (valid && lane == 0u) {
        gate_out[row] = gate_acc;
        up_out[row] = up_acc;
    }
}

extern "C" __global__ void rnb_q4k_packed_gate_up_gemv_batch_seq2_q8dot_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* __restrict__ gate_weights,
    const unsigned char* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float gate_acc0 = 0.0f;
    float up_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc1 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 148u;
    const unsigned char* gate_row_ptr = gate_weights + row * row_bytes;
    const unsigned char* up_row_ptr = up_weights + row * row_bytes;
    const signed char* input_qs0 = input_qs;
    const signed char* input_qs1 = input_qs + blocks_per_row * 256u;
    const float* input_ds0 = input_ds;
    const float* input_ds1 = input_ds + blocks_per_row * 8u;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* gate_block = gate_row_ptr + b * 148u;
            const unsigned char* up_block = up_row_ptr + b * 148u;
            const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
            const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
            const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
            const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
            const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
            const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
            const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
            const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

            for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
                const unsigned j = chunk >> 3;
                const unsigned elem = (chunk & 7u) * 4u;
                const unsigned q_index = (j >> 1) * 32u + elem;
                const unsigned char* gate_q_ptr = gate_block + 20u + q_index;
                const unsigned char* up_q_ptr = up_block + 20u + q_index;
                const unsigned gate_scmn =
                    (unsigned)gate_block[4u + j * 2u] | ((unsigned)gate_block[5u + j * 2u] << 8);
                const unsigned up_scmn =
                    (unsigned)up_block[4u + j * 2u] | ((unsigned)up_block[5u + j * 2u] << 8);
                const unsigned gate_sc = gate_scmn & 0xffu;
                const unsigned gate_mn = gate_scmn >> 8;
                const unsigned up_sc = up_scmn & 0xffu;
                const unsigned up_mn = up_scmn >> 8;
                const int gate_pack = rnb_q4_pack4(gate_q_ptr, j);
                const int up_pack = rnb_q4_pack4(up_q_ptr, j);

                const int x_pack0 = rnb_load_i32_aligned4(input_qs0 + b * 256u + j * 32u + elem);
                const int gate_dot0 = __dp4a(gate_pack, x_pack0, 0);
                const int up_dot0 = __dp4a(up_pack, x_pack0, 0);
                const int x_sum0 = __dp4a(0x01010101, x_pack0, 0);
                const float x_d0 = input_ds0[b * 8u + j];
                gate_acc0 += x_d0 * ((gate_d * (float)gate_sc) * (float)gate_dot0 - gate_dmin * (float)gate_mn * (float)x_sum0);
                up_acc0 += x_d0 * ((up_d * (float)up_sc) * (float)up_dot0 - up_dmin * (float)up_mn * (float)x_sum0);

                const int x_pack1 = rnb_load_i32_aligned4(input_qs1 + b * 256u + j * 32u + elem);
                const int gate_dot1 = __dp4a(gate_pack, x_pack1, 0);
                const int up_dot1 = __dp4a(up_pack, x_pack1, 0);
                const int x_sum1 = __dp4a(0x01010101, x_pack1, 0);
                const float x_d1 = input_ds1[b * 8u + j];
                gate_acc1 += x_d1 * ((gate_d * (float)gate_sc) * (float)gate_dot1 - gate_dmin * (float)gate_mn * (float)x_sum1);
                up_acc1 += x_d1 * ((up_d * (float)up_sc) * (float)up_dot1 - up_dmin * (float)up_mn * (float)x_sum1);
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
    }

    if (valid && lane == 0u) {
        gate_out[row] = gate_acc0;
        up_out[row] = up_acc0;
        gate_out[rows + row] = gate_acc1;
        up_out[rows + row] = up_acc1;
    }
}

extern "C" __global__ void rnb_q4k_qkv_gemv_warp8(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const unsigned char* __restrict__ q_weights,
    const unsigned char* __restrict__ k_weights,
    const unsigned char* __restrict__ v_weights,
    const float* __restrict__ input,
    unsigned q_rows,
    unsigned kv_rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned logical_row = blockIdx.x * 8u + warp;
    const unsigned total_rows = q_rows + kv_rows * 2u;
    const bool valid = logical_row < total_rows;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = nullptr;
    float* out = nullptr;
    unsigned row = logical_row;
    if (valid) {
        if (logical_row < q_rows) {
            row_ptr = q_weights + logical_row * row_bytes;
            out = q_out;
        } else if (logical_row < q_rows + kv_rows) {
            row = logical_row - q_rows;
            row_ptr = k_weights + row * row_bytes;
            out = k_out;
        } else {
            row = logical_row - q_rows - kv_rows;
            row_ptr = v_weights + row * row_bytes;
            out = v_out;
        }
    }

    float acc = 0.0f;
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
                acc += ((d * (float)sc) * (float)q - dmin * (float)mn) * input[b * 256u + tid];
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

extern "C" __global__ void rnb_q4k_qkv_gemv_q8dot_warp8(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const unsigned char* __restrict__ q_weights,
    const unsigned char* __restrict__ k_weights,
    const unsigned char* __restrict__ v_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned q_rows,
    unsigned kv_rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned logical_row = blockIdx.x * 8u + warp;
    const unsigned total_rows = q_rows + kv_rows * 2u;
    const bool valid = logical_row < total_rows;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = nullptr;
    float* out = nullptr;
    unsigned row = logical_row;
    if (valid) {
        if (logical_row < q_rows) {
            row_ptr = q_weights + logical_row * row_bytes;
            out = q_out;
        } else if (logical_row < q_rows + kv_rows) {
            row = logical_row - q_rows;
            row_ptr = k_weights + row * row_bytes;
            out = k_out;
        } else {
            row = logical_row - q_rows - kv_rows;
            row_ptr = v_weights + row * row_bytes;
            out = v_out;
        }
    }

    float acc = 0.0f;
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
                const signed char* x_qs = input_qs + b * 256u + j * 32u + elem;
                const int q_pack = rnb_q4_pack4(q_ptr, j);
                const int x_pack = rnb_load_i32_aligned4(x_qs);
                const int dot = __dp4a(q_pack, x_pack, 0);
                const int x_sum = __dp4a(0x01010101, x_pack, 0);
                const float x_d = input_ds[b * 8u + j];
                acc += x_d * ((d * (float)sc) * (float)dot - dmin * (float)mn * (float)x_sum);
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

extern "C" __global__ void rnb_bf16_gemv_warp8(
    float* __restrict__ out,
    const unsigned short* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned cols) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = 0.0f;
    if (valid) {
        const unsigned short* row_ptr = weights + (unsigned long long)row * cols;
        for (unsigned col = lane; col < cols; col += 32u) {
            const unsigned raw = (unsigned)row_ptr[col] << 16;
            const float w = __uint_as_float(raw);
            acc += w * input[col];
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_f16_gemv_warp8(
    float* __restrict__ out,
    const unsigned short* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned cols) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = 0.0f;
    if (valid) {
        const unsigned short* row_ptr = weights + (unsigned long long)row * cols;
        for (unsigned col = lane; col < cols; col += 32u) {
            const float w = __half2float(__ushort_as_half(row_ptr[col]));
            acc += w * input[col];
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q5_0_gemv(
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

    const unsigned row_bytes = blocks_per_row * 22u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 22u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned qh = (unsigned)block[2] | ((unsigned)block[3] << 8) |
                            ((unsigned)block[4] << 16) | ((unsigned)block[5] << 24);
        const unsigned byte = block[6u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float y = ((float)(low | (high << 4)) - 16.0f) * d;
        acc += y * input[idx];
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

extern "C" __global__ void rnb_q5_1_gemv(
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

    const unsigned row_bytes = blocks_per_row * 24u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 24u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
        const unsigned qh = (unsigned)block[4] | ((unsigned)block[5] << 8) |
                            ((unsigned)block[6] << 16) | ((unsigned)block[7] << 24);
        const unsigned byte = block[8u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float y = (float)(low | (high << 4)) * d + m;
        acc += y * input[idx];
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

extern "C" __global__ void rnb_q5_0_gemv_batch(
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

    const unsigned row_bytes = blocks_per_row * 22u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* input_row = input + seq * cols;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 22u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned qh = (unsigned)block[2] | ((unsigned)block[3] << 8) |
                            ((unsigned)block[4] << 16) | ((unsigned)block[5] << 24);
        const unsigned byte = block[6u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float y = ((float)(low | (high << 4)) - 16.0f) * d;
        acc += y * input_row[idx];
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

extern "C" __global__ void rnb_q5_0_gemv_batch_seq32(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256 || seq_len > 32u) {
        return;
    }

    __shared__ float partial[32][256];
    float acc[32];
    for (unsigned s = 0; s < seq_len; ++s) {
        acc[s] = 0.0f;
    }

    const unsigned row_bytes = blocks_per_row * 22u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 22u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned qh = (unsigned)block[2] | ((unsigned)block[3] << 8) |
                            ((unsigned)block[4] << 16) | ((unsigned)block[5] << 24);
        const unsigned byte = block[6u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float y = ((float)(low | (high << 4)) - 16.0f) * d;
        for (unsigned s = 0; s < seq_len; ++s) {
            acc[s] += y * input[s * cols + idx];
        }
    }

    for (unsigned s = 0; s < seq_len; ++s) {
        partial[s][tid] = acc[s];
    }
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (unsigned s = 0; s < seq_len; ++s) {
                partial[s][tid] += partial[s][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (unsigned s = 0; s < seq_len; ++s) {
            out[s * rows + row] = partial[s][0];
        }
    }
}

extern "C" __global__ void rnb_q5_1_gemv_batch(
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

    const unsigned row_bytes = blocks_per_row * 24u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* input_row = input + seq * cols;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 24u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
        const unsigned qh = (unsigned)block[4] | ((unsigned)block[5] << 8) |
                            ((unsigned)block[6] << 16) | ((unsigned)block[7] << 24);
        const unsigned byte = block[8u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float y = (float)(low | (high << 4)) * d + m;
        acc += y * input_row[idx];
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

extern "C" __global__ void rnb_q8_0_gemv(
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

    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + lane];
        acc += ((float)q * d) * input[idx];
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

extern "C" __global__ void rnb_q8_0_gemv_argmax_warp8(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ block_values,
    unsigned* __restrict__ block_indices,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = valid ? 0.0f : -3.4028234663852886e38f;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 34u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const signed char q = (signed char)block[2u + lane];
            acc += ((float)q * d) * input[b * 32u + lane];
        }
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

extern "C" __global__ void rnb_q8_0_gemv_q8dot_argmax_warp8(
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    float* __restrict__ block_values,
    unsigned* __restrict__ block_indices,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float acc = valid ? 0.0f : -3.4028234663852886e38f;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        const unsigned chunks_per_row = blocks_per_row * 8u;
        for (unsigned chunk = lane; chunk < chunks_per_row; chunk += 32u) {
            const unsigned b = chunk >> 3;
            const unsigned elem = (chunk & 7u) * 4u;
            const unsigned char* block = row_ptr + b * 34u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const float w_d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const int w_pack = rnb_load_i32_unaligned(block + 2u + elem);
            const int x_pack = rnb_load_i32_aligned4(input_qs + b * 32u + elem);
            const int dot = __dp4a(w_pack, x_pack, 0);
            acc += w_d * input_ds[b] * (float)dot;
        }
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

extern "C" __global__ void rnb_q8_0_gemv_warp8(
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
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;

    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 34u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const signed char q = (signed char)block[2u + lane];
            acc += ((float)q * d) * input[b * 32u + lane];
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q8_0_gemv_relu_sqr_input(
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

    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + lane];
        const float v = input[idx];
        const float activated = v > 0.0f ? v * v : 0.0f;
        acc += ((float)q * d) * activated;
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

extern "C" __global__ void rnb_q8_0_gemv_warp4(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;

    for (unsigned idx = lane; idx < cols; idx += 32u) {
        const unsigned b = idx >> 5;
        const unsigned qlane = idx & 31u;
        const unsigned char* block = row_ptr + b * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + qlane];
        acc += ((float)q * d) * input[idx];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        out[row] = acc;
    }
}

extern "C" __global__ void rnb_q8_0_gemv_relu_sqr_input_warp4(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;

    for (unsigned idx = lane; idx < cols; idx += 32u) {
        const unsigned b = idx >> 5;
        const unsigned qlane = idx & 31u;
        const unsigned char* block = row_ptr + b * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + qlane];
        const float v = input[idx];
        const float activated = v > 0.0f ? v * v : 0.0f;
        acc += ((float)q * d) * activated;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        out[row] = acc;
    }
}



extern "C" __global__ void rnb_q8_0_gemv_batch_token2(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned seq_len) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || seq_len != 2u || tid >= 128u) {
        return;
    }

    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    __shared__ float warp_sums0[4];
    __shared__ float warp_sums1[4];
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    for (unsigned idx = tid; idx < cols; idx += 128u) {
        const unsigned block_index = idx >> 5;
        const unsigned char* block = row_ptr + block_index * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + lane];
        const float weight = (float)q * d;
        acc0 += weight * input[idx];
        acc1 += weight * input[cols + idx];
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        warp_sums0[warp] = acc0;
        warp_sums1[warp] = acc1;
    }
    __syncthreads();

    if (warp == 0u) {
        acc0 = lane < 4u ? warp_sums0[lane] : 0.0f;
        acc1 = lane < 4u ? warp_sums1[lane] : 0.0f;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        }
        if (lane == 0u) {
            out[row] = acc0;
            out[rows + row] = acc1;
        }
    }
}

extern "C" __global__ void rnb_q8_0_gemv_batch_token2_multi3(
    float* __restrict__ out0,
    const unsigned char* __restrict__ weights0,
    unsigned rows0,
    float* __restrict__ out1,
    const unsigned char* __restrict__ weights1,
    unsigned rows1,
    float* __restrict__ out2,
    const unsigned char* __restrict__ weights2,
    unsigned rows2,
    const float* __restrict__ input,
    unsigned blocks_per_row,
    unsigned seq_len) {
    unsigned flat_row = blockIdx.x;
    float* out;
    const unsigned char* weights;
    unsigned rows;
    unsigned row;
    if (flat_row < rows0) {
        out = out0;
        weights = weights0;
        rows = rows0;
        row = flat_row;
    } else if ((flat_row -= rows0) < rows1) {
        out = out1;
        weights = weights1;
        rows = rows1;
        row = flat_row;
    } else {
        flat_row -= rows1;
        if (flat_row >= rows2) {
            return;
        }
        out = out2;
        weights = weights2;
        rows = rows2;
        row = flat_row;
    }

    const unsigned tid = threadIdx.x;
    if (seq_len != 2u || tid >= 128u) {
        return;
    }
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    __shared__ float warp_sums0[4];
    __shared__ float warp_sums1[4];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    for (unsigned idx = tid; idx < cols; idx += 128u) {
        const unsigned block_index = idx >> 5;
        const unsigned char* block = row_ptr + block_index * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + lane];
        const float weight = (float)q * d;
        acc0 += weight * input[idx];
        acc1 += weight * input[cols + idx];
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        warp_sums0[warp] = acc0;
        warp_sums1[warp] = acc1;
    }
    __syncthreads();
    if (warp == 0u) {
        acc0 = lane < 4u ? warp_sums0[lane] : 0.0f;
        acc1 = lane < 4u ? warp_sums1[lane] : 0.0f;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        }
        if (lane == 0u) {
            out[row] = acc0;
            out[rows + row] = acc1;
        }
    }
}

extern "C" __global__ void rnb_f32_gemv_batch_token2_multi2(
    float* __restrict__ out0,
    const float* __restrict__ weights0,
    unsigned rows0,
    float* __restrict__ out1,
    const float* __restrict__ weights1,
    unsigned rows1,
    const float* __restrict__ input,
    unsigned cols,
    unsigned seq_len) {
    unsigned flat_row = blockIdx.x;
    float* out;
    const float* weights;
    unsigned rows;
    unsigned row;
    if (flat_row < rows0) {
        out = out0;
        weights = weights0;
        rows = rows0;
        row = flat_row;
    } else {
        flat_row -= rows0;
        if (flat_row >= rows1) {
            return;
        }
        out = out1;
        weights = weights1;
        rows = rows1;
        row = flat_row;
    }

    const unsigned tid = threadIdx.x;
    if (seq_len != 2u || tid >= 256u) {
        return;
    }
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    __shared__ float warp_sums0[8];
    __shared__ float warp_sums1[8];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const float* row_ptr = weights + row * cols;
    for (unsigned index = tid; index < cols; index += 256u) {
        const float weight = row_ptr[index];
        acc0 += weight * input[index];
        acc1 += weight * input[cols + index];
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        warp_sums0[warp] = acc0;
        warp_sums1[warp] = acc1;
    }
    __syncthreads();
    if (warp == 0u) {
        acc0 = lane < 8u ? warp_sums0[lane] : 0.0f;
        acc1 = lane < 8u ? warp_sums1[lane] : 0.0f;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        }
        if (lane == 0u) {
            out[row] = acc0;
            out[rows + row] = acc1;
        }
    }
}

extern "C" __global__ void rnb_q8_0_gemv_batch(
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

    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* input_row = input + seq * cols;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + lane];
        acc += ((float)q * d) * input_row[idx];
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




extern "C" __global__ void rnb_q8_0_head_gemv_batch(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned rows_per_head,
    unsigned blocks_per_row,
    unsigned token_count) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned token = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || token >= token_count || lane >= 32u || rows_per_head == 0u) {
        return;
    }

    const unsigned head_count = rows / rows_per_head;
    const unsigned head = row / rows_per_head;
    const unsigned cols = blocks_per_row * 32u;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights + row * row_bytes;
    const float* input_row = input + (token * head_count + head) * cols;
    float acc = 0.0f;

    for (unsigned idx = lane; idx < cols; idx += 32u) {
        const unsigned block_idx = idx >> 5;
        const unsigned quant_lane = idx & 31u;
        const unsigned char* block = row_ptr + block_idx * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + quant_lane];
        acc += ((float)q * d) * input_row[idx];
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        out[token * rows + row] = acc;
    }
}

extern "C" __global__ void rnb_q8_0_dequant_f32(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned idx = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned cols = blocks_per_row * 32u;
    if (row >= rows || idx >= cols) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned b = idx >> 5;
    const unsigned lane = idx & 31u;
    const unsigned char* block = weights + row * row_bytes + b * 34u;
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const signed char q = (signed char)block[2u + lane];
    out[row * cols + idx] = (float)q * d;
}

extern "C" __global__ void rnb_relu_sqr_inplace(
    float* __restrict__ values,
    unsigned len) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const float v = values[idx];
    values[idx] = v > 0.0f ? v * v : 0.0f;
}

extern "C" __global__ void rnb_q5_0_selected_relu_sqr_by_token(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned slot = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 22u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* input_row = input + token_ids[slot] * cols;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 22u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned qh = (unsigned)block[2] | ((unsigned)block[3] << 8) |
                            ((unsigned)block[4] << 16) | ((unsigned)block[5] << 24);
        const unsigned byte = block[6u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float y = ((float)(low | (high << 4)) - 16.0f) * d;
        acc += y * input_row[idx];
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
        const float v = partial[0];
        out[slot * rows + row] = v > 0.0f ? v * v : 0.0f;
    }
}

extern "C" __global__ void rnb_q5_1_selected_down_accum_by_token(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned slot = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 24u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* slot_input = input + slot * cols;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 24u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
        const unsigned qh = (unsigned)block[4] | ((unsigned)block[5] << 8) |
                            ((unsigned)block[6] << 16) | ((unsigned)block[7] << 24);
        const unsigned byte = block[8u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float y = (float)(low | (high << 4)) * d + m;
        acc += y * slot_input[idx];
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
        atomicAdd(out + token_ids[slot] * rows + row, partial[0] * route[slot]);
    }
}

extern "C" __global__ void rnb_q5_1_selected_down_accum_by_token_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned slot = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 24u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* slot_input = input + slot * cols;

    for (unsigned idx = lane; idx < cols; idx += 32u) {
        const unsigned b = idx >> 5;
        const unsigned qlane = idx & 31u;
        const unsigned char* block = row_ptr + b * 24u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
        const unsigned qh = (unsigned)block[4] | ((unsigned)block[5] << 8) |
                            ((unsigned)block[6] << 16) | ((unsigned)block[7] << 24);
        const unsigned byte = block[8u + (qlane & 15u)];
        const unsigned low = qlane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> qlane) & 1u;
        const float y = (float)(low | (high << 4)) * d + m;
        acc += y * slot_input[idx];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        atomicAdd(out + token_ids[slot] * rows + row, acc * route[slot]);
    }
}

extern "C" __global__ void rnb_q8_0_selected_down_accum_by_token(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned slot = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;

    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* slot_input = input + slot * cols;

    for (unsigned idx = tid; idx < cols; idx += 256u) {
        const unsigned b = idx >> 5;
        const unsigned lane = idx & 31u;
        const unsigned char* block = row_ptr + b * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + lane];
        acc += ((float)q * d) * slot_input[idx];
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
        atomicAdd(out + token_ids[slot] * rows + row, partial[0] * route[slot]);
    }
}

extern "C" __global__ void rnb_q8_0_selected_down_accum_by_token_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned slot = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const unsigned cols = blocks_per_row * 32u;
    const float* slot_input = input + slot * cols;

    for (unsigned idx = lane; idx < cols; idx += 32u) {
        const unsigned b = idx >> 5;
        const unsigned qlane = idx & 31u;
        const unsigned char* block = row_ptr + b * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = (signed char)block[2u + qlane];
        acc += ((float)q * d) * slot_input[idx];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        atomicAdd(out + token_ids[slot] * rows + row, acc * route[slot]);
    }
}

template <typename Decoder>
__device__ __forceinline__ void rnb_quant_gemv_warp8_body(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned token) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;
    const unsigned block_elems = Decoder::block_elems();
    const unsigned block_bytes = Decoder::block_bytes();
    const unsigned row_bytes = blocks_per_row * block_bytes;
    const unsigned cols = blocks_per_row * block_elems;
    const unsigned char* row_ptr = weights + (unsigned long long)row * row_bytes;
    const float* token_input = input + (unsigned long long)token * cols;

    float acc = 0.0f;
    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * block_bytes;
            for (unsigned index = lane; index < block_elems; index += 32u) {
                acc += Decoder::value(block, index) * token_input[b * block_elems + index];
            }
        }
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[(unsigned long long)token * rows + row] = acc;
    }
}

struct RnbQ2KDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 84u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[80] | ((unsigned)block[81] << 8);
        const unsigned raw_dmin = (unsigned)block[82] | ((unsigned)block[83] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
        const unsigned scale_min = block[index >> 4];
        const unsigned q_index = (index >> 7) * 32u + (index & 31u);
        const unsigned shift = ((index & 127u) >> 5) * 2u;
        const unsigned q = (block[16u + q_index] >> shift) & 3u;
        return d * (float)(scale_min & 0x0fu) * (float)q
            - dmin * (float)(scale_min >> 4);
    }
};

struct RnbQ3KDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 110u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[108] | ((unsigned)block[109] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned scale_idx = index >> 4;
        const unsigned scale_lane = scale_idx & 3u;
        const unsigned packed_high = block[104u + scale_lane];
        unsigned scale_code;
        if (scale_idx < 4u) {
            scale_code =
                (block[96u + scale_lane] & 0x0fu) | ((packed_high & 0x03u) << 4);
        } else if (scale_idx < 8u) {
            scale_code =
                (block[100u + scale_lane] & 0x0fu)
                | (((packed_high >> 2) & 0x03u) << 4);
        } else if (scale_idx < 12u) {
            scale_code =
                (block[96u + scale_lane] >> 4)
                | (((packed_high >> 4) & 0x03u) << 4);
        } else {
            scale_code =
                (block[100u + scale_lane] >> 4)
                | (((packed_high >> 6) & 0x03u) << 4);
        }
        const unsigned q_index = (index >> 7) * 32u + (index & 31u);
        const unsigned shift = ((index & 127u) >> 5) * 2u;
        const unsigned q = (block[32u + q_index] >> shift) & 3u;
        const unsigned high_mask = 1u << (index >> 5);
        const int high = (block[index & 31u] & high_mask) != 0u ? 0 : 4;
        return d * (float)((int)scale_code - 32) * (float)((int)q - high);
    }
};

struct RnbQ40Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 18u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned packed = block[2u + (index & 15u)];
        const unsigned q = index < 16u ? (packed & 0x0fu) : (packed >> 4);
        return ((float)q - 8.0f) * d;
    }
};

struct RnbQ41Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 20u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
        const unsigned packed = block[4u + (index & 15u)];
        const unsigned q = index < 16u ? (packed & 0x0fu) : (packed >> 4);
        return (float)q * d + m;
    }
};

struct RnbQ81Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 36u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        return (float)(signed char)block[4u + index] * d;
    }
};

struct RnbF16Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 64u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned offset = index * 2u;
        const unsigned raw = (unsigned)block[offset] | ((unsigned)block[offset + 1u] << 8);
        return __half2float(__ushort_as_half((unsigned short)raw));
    }
};

struct RnbBf16Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 64u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned offset = index * 2u;
        const unsigned raw = (unsigned)block[offset] | ((unsigned)block[offset + 1u] << 8);
        return __uint_as_float(raw << 16);
    }
};

extern "C" __global__ void rnb_q2k_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbQ2KDecoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_q3k_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbQ3KDecoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_q4_0_gemv_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row) {
    rnb_quant_gemv_warp8_body<RnbQ40Decoder>(
        out, weights, input, rows, blocks_per_row, 0u);
}

extern "C" __global__ void rnb_q4_0_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbQ40Decoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_q4_1_gemv_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row) {
    rnb_quant_gemv_warp8_body<RnbQ41Decoder>(
        out, weights, input, rows, blocks_per_row, 0u);
}

extern "C" __global__ void rnb_q4_1_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbQ41Decoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_q8_1_gemv_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row) {
    rnb_quant_gemv_warp8_body<RnbQ81Decoder>(
        out, weights, input, rows, blocks_per_row, 0u);
}

extern "C" __global__ void rnb_q8_1_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbQ81Decoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_f16_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbF16Decoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_bf16_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbBf16Decoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}
