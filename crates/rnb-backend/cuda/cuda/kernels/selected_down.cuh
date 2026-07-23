extern "C" __global__ void rnb_q4k_selected_down_accum(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights[expert] + row * row_bytes;
    const float* expert_input = input + expert * blocks_per_row * 256u;

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
        acc += y * expert_input[b * 256u + tid];
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
        atomicAdd(out + row, partial[0] * route[expert]);
    }
}

extern "C" __global__ void rnb_q4k_selected_down_accum_silu(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights[expert] + row * row_bytes;
    const float* gate_input = gate + expert * blocks_per_row * 256u;
    const float* up_input = up + expert * blocks_per_row * 256u;

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
        const float g = gate_input[b * 256u + tid];
        const float act = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
        acc += y * act;
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
        atomicAdd(out + row, partial[0] * route[expert]);
    }
}

extern "C" __global__ void rnb_q4k_selected_down_silu_rowreduce(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned selected,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;

    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr = weights[expert] + row * row_bytes;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
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
            const float g = gate_input[b * 256u + tid];
            const float act =
                (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += y * act;
        }
        acc += expert_acc * route[expert];
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

extern "C" __global__ void rnb_q3k_selected_down_silu_rowreduce(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned selected,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 110u;
    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr = weights[expert] + row * row_bytes;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 110u;
            const unsigned raw_d = (unsigned)block[108] | ((unsigned)block[109] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const unsigned scale_idx = tid >> 4;
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
            const unsigned q_index = (tid >> 7) * 32u + (tid & 31u);
            const unsigned shift = ((tid & 127u) >> 5) * 2u;
            const unsigned q = (block[32u + q_index] >> shift) & 3u;
            const unsigned high_mask = 1u << (tid >> 5);
            const int high = (block[tid & 31u] & high_mask) != 0u ? 0 : 4;
            const int signed_scale = (int)scale_code - 32;
            const float value = d * (float)signed_scale * (float)((int)q - high);
            const float g = gate_input[b * 256u + tid];
            const float act = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += value * act;
        }
        acc += expert_acc * route[expert];
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


extern "C" __global__ void rnb_q3k_selected_down_silu_per_slot(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 110u;
    const unsigned char* row_ptr = weights[expert] + row * row_bytes;
    const float* gate_input = gate + expert * blocks_per_row * 256u;
    const float* up_input = up + expert * blocks_per_row * 256u;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 110u;
        const unsigned raw_d = (unsigned)block[108] | ((unsigned)block[109] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned scale_idx = tid >> 4;
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
        const unsigned q_index = (tid >> 7) * 32u + (tid & 31u);
        const unsigned shift = ((tid & 127u) >> 5) * 2u;
        const unsigned q = (block[32u + q_index] >> shift) & 3u;
        const unsigned high_mask = 1u << (tid >> 5);
        const int high = (block[tid & 31u] & high_mask) != 0u ? 0 : 4;
        const int signed_scale = (int)scale_code - 32;
        const float value = d * (float)signed_scale * (float)((int)q - high);
        const float g = gate_input[b * 256u + tid];
        const float act = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
        acc += value * act;
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
        out[expert * rows + row] = partial[0];
    }
}
extern "C" __global__ void rnb_iq4_xs_selected_down_silu_rowreduce(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned selected,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 136u;

    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr = weights[expert] + row * row_bytes;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 136u;
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned scales_h = (unsigned)block[2] | ((unsigned)block[3] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const unsigned ib = tid >> 5;
            const unsigned local = tid & 31u;
            const unsigned low =
                (block[4u + (ib >> 1)] >> (4u * (ib & 1u))) & 0x0fu;
            const unsigned high = ((scales_h >> (2u * ib)) & 0x03u) << 4u;
            const float dl = d * ((float)(low | high) - 32.0f);
            const unsigned q_byte = block[8u + ib * 16u + (local & 15u)];
            const unsigned q = local < 16u ? (q_byte & 0x0fu) : (q_byte >> 4);
            const float y = dl * rnb_iq4nl_value(q);
            const float g = gate_input[b * 256u + tid];
            const float act = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += y * act;
        }
        acc += expert_acc * route[expert];
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

extern "C" __global__ void rnb_q6k_selected_down_accum(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights[expert] + row * row_bytes;
    const float* expert_input = input + expert * blocks_per_row * 256u;

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
        acc += y * expert_input[b * 256u + tid];
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
        atomicAdd(out + row, partial[0] * route[expert]);
    }
}

extern "C" __global__ void rnb_q5k_selected_down_accum(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights[expert] + row * row_bytes;
    const float* expert_input = input + expert * blocks_per_row * 256u;

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
        acc += y * expert_input[b * 256u + tid];
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
        atomicAdd(out + row, partial[0] * route[expert]);
    }
}

extern "C" __global__ void rnb_q5k_selected_down_accum_silu(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights[expert] + row * row_bytes;
    const float* gate_input = gate + expert * blocks_per_row * 256u;
    const float* up_input = up + expert * blocks_per_row * 256u;

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
        const float g = gate_input[b * 256u + tid];
        const float act = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
        acc += y * act;
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
        atomicAdd(out + row, partial[0] * route[expert]);
    }
}

extern "C" __global__ void rnb_q5k_selected_down_silu_rowreduce(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned selected,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 176u;

    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr = weights[expert] + row * row_bytes;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
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
            const float g = gate_input[b * 256u + tid];
            const float act = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += y * act;
        }
        acc += expert_acc * route[expert];
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

extern "C" __global__ void rnb_q6k_selected_down_accum_silu(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights[expert] + row * row_bytes;
    const float* gate_input = gate + expert * blocks_per_row * 256u;
    const float* up_input = up + expert * blocks_per_row * 256u;

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
        const float g = gate_input[b * 256u + tid];
        const float act = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
        acc += y * act;
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
        atomicAdd(out + row, partial[0] * route[expert]);
    }
}

extern "C" __global__ void rnb_q6k_selected_down_silu_rowreduce(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned selected,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;

    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr = weights[expert] + row * row_bytes;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
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
            const float g = gate_input[b * 256u + tid];
            const float act =
                (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += y * act;
        }
        acc += expert_acc * route[expert];
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

extern "C" __global__ void rnb_q5k_selected_down_accum_by_token(
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
    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const float* slot_input = input + slot * blocks_per_row * 256u;

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
        acc += y * slot_input[b * 256u + tid];
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

extern "C" __global__ void rnb_q4k_selected_down_accum_by_token(
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
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const float* slot_input = input + slot * blocks_per_row * 256u;

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
        acc += y * slot_input[b * 256u + tid];
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

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token(
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
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const float* slot_input = input + slot * blocks_per_row * 256u;

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
        acc += y * slot_input[b * 256u + tid];
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

static __device__ __forceinline__ float rnb_selected_q4k_block_dot_f32_lane(
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

extern "C" __global__ void rnb_q4k_selected_down_silu_rowreduce_warp8(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned selected,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;
    const unsigned row_bytes = blocks_per_row * 144u;
    __shared__ float act_tile[256];
    float acc = 0.0f;
    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr =
            valid ? weights[expert] + row * row_bytes : nullptr;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned index = b * 256u + threadIdx.x;
            const float g = gate_input[index];
            act_tile[threadIdx.x] =
                (g / (1.0f + expf(-g))) * up_input[index];
            __syncthreads();
            if (valid) {
                const unsigned char* block = row_ptr + b * 144u;
                expert_acc += rnb_selected_q4k_block_dot_f32_lane(block, act_tile, lane);
            }
            __syncthreads();
        }
        acc += expert_acc * route[expert];
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[row] = acc;
    }
}

static __device__ __forceinline__ float rnb_selected_q6k_block_dot_f32_lane(
    const unsigned char* __restrict__ block,
    const float* __restrict__ input,
    unsigned lane) {
    const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    float acc = 0.0f;
#pragma unroll
    for (unsigned n = 0; n < 2u; ++n) {
        const unsigned ql_base = n * 64u;
        const unsigned qh_base = 128u + n * 32u;
        const unsigned sc_base = 192u + n * 8u;
        const unsigned ql0 = block[ql_base + lane];
        const unsigned ql1 = block[ql_base + lane + 32u];
        const unsigned qh = block[qh_base + lane];
        const unsigned is = lane >> 4;
        const int sc0 = (int)((signed char)block[sc_base + is]);
        const int sc1 = (int)((signed char)block[sc_base + is + 2u]);
        const int sc2 = (int)((signed char)block[sc_base + is + 4u]);
        const int sc3 = (int)((signed char)block[sc_base + is + 6u]);
        const unsigned input_base = n * 128u + lane;
        acc += d * (float)sc0
            * (float)((int)((ql0 & 0x0fu) | (((qh >> 0) & 3u) << 4)) - 32)
            * input[input_base];
        acc += d * (float)sc1
            * (float)((int)((ql1 & 0x0fu) | (((qh >> 2) & 3u) << 4)) - 32)
            * input[input_base + 32u];
        acc += d * (float)sc2
            * (float)((int)((ql0 >> 4) | (((qh >> 4) & 3u) << 4)) - 32)
            * input[input_base + 64u];
        acc += d * (float)sc3
            * (float)((int)((ql1 >> 4) | (((qh >> 6) & 3u) << 4)) - 32)
            * input[input_base + 96u];
    }
    return acc;
}

extern "C" __global__ void rnb_q6k_selected_down_silu_rowreduce_warp8(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned selected,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;
    const unsigned row_bytes = blocks_per_row * 210u;
    __shared__ float act_tile[256];
    float acc = 0.0f;
    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr =
            valid ? weights[expert] + row * row_bytes : nullptr;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned index = b * 256u + threadIdx.x;
            const float g = gate_input[index];
            act_tile[threadIdx.x] =
                (g / (1.0f + expf(-g))) * up_input[index];
            __syncthreads();
            if (valid) {
                const unsigned char* block = row_ptr + b * 210u;
                expert_acc += rnb_selected_q6k_block_dot_f32_lane(block, act_tile, lane);
            }
            __syncthreads();
        }
        acc += expert_acc * route[expert];
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (valid && lane == 0u) {
        out[row] = acc;
    }
}
