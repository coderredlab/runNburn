extern "C" __global__ void rnb_q4k_selected_gemv(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
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
        out[expert * rows + row] = partial[0];
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[expert] + row * row_bytes;
    const unsigned char* up_row = up_weights[expert] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned char* up_block = up_row + b * 144u;
        const unsigned j = tid >> 5;
        const unsigned local = tid & 63u;
        const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);

        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
        unsigned gate_sc;
        unsigned gate_mn;
        if (j < 4) {
            gate_sc = gate_block[4 + j] & 63u;
            gate_mn = gate_block[4 + j + 4] & 63u;
        } else {
            gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
            gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
        }
        unsigned gate_q = gate_block[16 + q_index];
        gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
        const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;
        gate_acc += gate_y * input[b * 256u + tid];

        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
        unsigned up_sc;
        unsigned up_mn;
        if (j < 4) {
            up_sc = up_block[4 + j] & 63u;
            up_mn = up_block[4 + j + 4] & 63u;
        } else {
            up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
            up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
        }
        unsigned up_q = up_block[16 + q_index];
        up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
        const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;
        up_acc += up_y * input[b * 256u + tid];
    }

    partial_gate[tid] = gate_acc;
    partial_up[tid] = up_acc;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_gate[tid] += partial_gate[tid + stride];
            partial_up[tid] += partial_up[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const unsigned out_idx = expert * rows + row;
        gate_out[out_idx] = partial_gate[0];
        up_out[out_idx] = partial_up[0];
    }
}

extern "C" __global__ void rnb_q2k_selected_gate_up_gemv_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned expert = blockIdx.y;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (valid) {
        const unsigned row_bytes = blocks_per_row * 84u;
        const unsigned char* gate_row = gate_weights[expert] + row * row_bytes;
        const unsigned char* up_row = up_weights[expert] + row * row_bytes;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* gate_block = gate_row + b * 84u;
            const unsigned char* up_block = up_row + b * 84u;
            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned scale_idx = tid >> 4;
                const unsigned q_index = (tid >> 7) * 32u + (tid & 31u);
                const unsigned shift = ((tid & 127u) >> 5) * 2u;

                const unsigned gate_raw_d =
                    (unsigned)gate_block[80] | ((unsigned)gate_block[81] << 8);
                const unsigned gate_raw_dmin =
                    (unsigned)gate_block[82] | ((unsigned)gate_block[83] << 8);
                const float gate_d =
                    __half2float(__ushort_as_half((unsigned short)gate_raw_d));
                const float gate_dmin =
                    __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
                const unsigned gate_scale_min = gate_block[scale_idx];
                const unsigned gate_q = (gate_block[16u + q_index] >> shift) & 3u;
                const float gate_value =
                    gate_d * (float)(gate_scale_min & 0x0fu) * (float)gate_q
                    - gate_dmin * (float)(gate_scale_min >> 4);
                gate_acc += gate_value * input[b * 256u + tid];

                const unsigned up_raw_d =
                    (unsigned)up_block[80] | ((unsigned)up_block[81] << 8);
                const unsigned up_raw_dmin =
                    (unsigned)up_block[82] | ((unsigned)up_block[83] << 8);
                const float up_d =
                    __half2float(__ushort_as_half((unsigned short)up_raw_d));
                const float up_dmin =
                    __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
                const unsigned up_scale_min = up_block[scale_idx];
                const unsigned up_q = (up_block[16u + q_index] >> shift) & 3u;
                const float up_value =
                    up_d * (float)(up_scale_min & 0x0fu) * (float)up_q
                    - up_dmin * (float)(up_scale_min >> 4);
                up_acc += up_value * input[b * 256u + tid];
            }
        }
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    if (valid && lane == 0u) {
        const unsigned out_idx = expert * rows + row;
        gate_out[out_idx] = gate_acc;
        up_out[out_idx] = up_acc;
    }
}

extern "C" __global__ void rnb_iq4_xs_selected_gate_up_gemv(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned expert = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 136u;
    const unsigned char* gate_row = gate_weights[expert] + row * row_bytes;
    const unsigned char* up_row = up_weights[expert] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gate_block = gate_row + b * 136u;
        const unsigned char* up_block = up_row + b * 136u;
        const float x = input[b * 256u + tid];
        const unsigned ib = tid >> 5;
        const unsigned local = tid & 31u;

        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_scales_h =
            (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const unsigned gate_low =
            (gate_block[4u + (ib >> 1)] >> (4u * (ib & 1u))) & 0x0fu;
        const unsigned gate_high = ((gate_scales_h >> (2u * ib)) & 0x03u) << 4u;
        const float gate_dl = gate_d * ((float)(gate_low | gate_high) - 32.0f);
        const unsigned gate_q_byte = gate_block[8u + ib * 16u + (local & 15u)];
        const unsigned gate_q = local < 16u ? (gate_q_byte & 0x0fu) : (gate_q_byte >> 4);
        gate_acc += gate_dl * rnb_iq4nl_value(gate_q) * x;

        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_scales_h = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const unsigned up_low =
            (up_block[4u + (ib >> 1)] >> (4u * (ib & 1u))) & 0x0fu;
        const unsigned up_high = ((up_scales_h >> (2u * ib)) & 0x03u) << 4u;
        const float up_dl = up_d * ((float)(up_low | up_high) - 32.0f);
        const unsigned up_q_byte = up_block[8u + ib * 16u + (local & 15u)];
        const unsigned up_q = local < 16u ? (up_q_byte & 0x0fu) : (up_q_byte >> 4);
        up_acc += up_dl * rnb_iq4nl_value(up_q) * x;
    }

    partial_gate[tid] = gate_acc;
    partial_up[tid] = up_acc;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_gate[tid] += partial_gate[tid + stride];
            partial_up[tid] += partial_up[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const unsigned out_idx = expert * rows + row;
        gate_out[out_idx] = partial_gate[0];
        up_out[out_idx] = partial_up[0];
    }
}

extern "C" __global__ void rnb_q4k_selected_gemv_by_token(
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
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const float* input_row = input + token_ids[slot] * blocks_per_row * 256u;

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
        out[slot * rows + row] = partial[0];
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
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

    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot] + row * row_bytes;
    const float* input_row = input + token_ids[slot] * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned char* up_block = up_row + b * 144u;
        const float x = input_row[b * 256u + tid];
        const unsigned j = tid >> 5;
        const unsigned local = tid & 63u;
        const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);

        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
        unsigned gate_sc;
        unsigned gate_mn;
        if (j < 4) {
            gate_sc = gate_block[4 + j] & 63u;
            gate_mn = gate_block[4 + j + 4] & 63u;
        } else {
            gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
            gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
        }
        unsigned gate_q = gate_block[16 + q_index];
        gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
        const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;
        gate_acc += gate_y * x;

        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
        unsigned up_sc;
        unsigned up_mn;
        if (j < 4) {
            up_sc = up_block[4 + j] & 63u;
            up_mn = up_block[4 + j + 4] & 63u;
        } else {
            up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
            up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
        }
        unsigned up_q = up_block[16 + q_index];
        up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
        const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;
        up_acc += up_y * x;
    }

    partial_gate[tid] = gate_acc;
    partial_up[tid] = up_acc;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_gate[tid] += partial_gate[tid + stride];
            partial_up[tid] += partial_up[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const unsigned out_idx = slot * rows + row;
        gate_out[out_idx] = partial_gate[0];
        up_out[out_idx] = partial_up[0];
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token_warp_reduce(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
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

    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    __shared__ float warp_gate[8];
    __shared__ float warp_up[8];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot] + row * row_bytes;
    const float* input_row = input + token_ids[slot] * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned char* up_block = up_row + b * 144u;
        const float x = input_row[b * 256u + tid];
        const unsigned j = tid >> 5;
        const unsigned local = tid & 63u;
        const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);

        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin =
            (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
        unsigned gate_sc;
        unsigned gate_mn;
        if (j < 4) {
            gate_sc = gate_block[4 + j] & 63u;
            gate_mn = gate_block[4 + j + 4] & 63u;
        } else {
            gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
            gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
        }
        unsigned gate_q = gate_block[16 + q_index];
        gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
        const float gate_y =
            (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;
        gate_acc += gate_y * x;

        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
        unsigned up_sc;
        unsigned up_mn;
        if (j < 4) {
            up_sc = up_block[4 + j] & 63u;
            up_mn = up_block[4 + j + 4] & 63u;
        } else {
            up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
            up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
        }
        unsigned up_q = up_block[16 + q_index];
        up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
        const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;
        up_acc += up_y * x;
    }

    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    if (lane == 0u) {
        warp_gate[warp] = gate_acc;
        warp_up[warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0u) {
        gate_acc = lane < 8u ? warp_gate[lane] : 0.0f;
        up_acc = lane < 8u ? warp_up[lane] : 0.0f;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
            up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
        }
        if (lane == 0u) {
            const unsigned out_idx = slot * rows + row;
            gate_out[out_idx] = gate_acc;
            up_out[out_idx] = up_acc;
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token_pair2_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ expert_ids,
    const unsigned* __restrict__ pair_slots,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned slots_per_token,
    unsigned blocks_per_row,
    unsigned fuse_silu) {
    constexpr unsigned INVALID_SLOT = 0xffffffffu;
    constexpr unsigned SKIP_SLOT = 0xfffffffeu;
    const unsigned candidate_slot = blockIdx.y;
    const unsigned total_slots = slots_per_token * 2u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const bool valid = row < rows;
    __shared__ unsigned shared_primary_slot;
    __shared__ unsigned shared_partner_slot;
    __shared__ float input_tile0[256];
    __shared__ float input_tile1[256];

    unsigned primary_slot;
    unsigned partner_slot;
    if (pair_slots != nullptr) {
        primary_slot = candidate_slot < total_slots ? candidate_slot : INVALID_SLOT;
        partner_slot =
            primary_slot != INVALID_SLOT ? pair_slots[candidate_slot] : INVALID_SLOT;
        if (partner_slot == SKIP_SLOT) {
            primary_slot = INVALID_SLOT;
        }
    } else {
        if (threadIdx.x == 0u) {
            shared_primary_slot =
                candidate_slot < total_slots ? candidate_slot : INVALID_SLOT;
            shared_partner_slot = INVALID_SLOT;
            if (shared_primary_slot != INVALID_SLOT) {
                if (candidate_slot < slots_per_token) {
                    const unsigned expert = expert_ids[candidate_slot];
                    for (unsigned second = slots_per_token; second < total_slots; ++second) {
                        if (expert_ids[second] == expert) {
                            shared_partner_slot = second;
                            break;
                        }
                    }
                } else {
                    const unsigned expert = expert_ids[candidate_slot];
                    for (unsigned first = 0; first < slots_per_token; ++first) {
                        if (expert_ids[first] == expert) {
                            shared_primary_slot = INVALID_SLOT;
                            break;
                        }
                    }
                }
            }
        }
        __syncthreads();
        primary_slot = shared_primary_slot;
        partner_slot = shared_partner_slot;
    }
    if (primary_slot == INVALID_SLOT) {
        return;
    }

    const bool paired = partner_slot != INVALID_SLOT;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row =
        valid ? gate_weights[primary_slot] + row * row_bytes : nullptr;
    const unsigned char* up_row =
        valid ? up_weights[primary_slot] + row * row_bytes : nullptr;
    const float* input_row0 =
        input + token_ids[primary_slot] * blocks_per_row * 256u;
    const float* input_row1 = paired
        ? input + token_ids[partner_slot] * blocks_per_row * 256u
        : input_row0;
    float gate_acc0 = 0.0f;
    float up_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc1 = 0.0f;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        input_tile0[threadIdx.x] = input_row0[b * 256u + threadIdx.x];
        input_tile1[threadIdx.x] =
            paired ? input_row1[b * 256u + threadIdx.x] : 0.0f;
        __syncthreads();
        if (valid) {
            const unsigned char* gate_block = gate_row + b * 144u;
            const unsigned char* up_block = up_row + b * 144u;
            float gate_d_lane = 0.0f;
            float gate_dmin_lane = 0.0f;
            float up_d_lane = 0.0f;
            float up_dmin_lane = 0.0f;
            if (lane == 0u) {
                const unsigned gate_raw_d =
                    (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
                const unsigned gate_raw_dmin =
                    (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
                const unsigned up_raw_d =
                    (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
                const unsigned up_raw_dmin =
                    (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
                gate_d_lane = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
                gate_dmin_lane =
                    __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
                up_d_lane = __half2float(__ushort_as_half((unsigned short)up_raw_d));
                up_dmin_lane = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
            }
            const float gate_d = __shfl_sync(0xffffffffu, gate_d_lane, 0);
            const float gate_dmin = __shfl_sync(0xffffffffu, gate_dmin_lane, 0);
            const float up_d = __shfl_sync(0xffffffffu, up_d_lane, 0);
            const float up_dmin = __shfl_sync(0xffffffffu, up_dmin_lane, 0);

            unsigned gate_sc_lane = 0u;
            unsigned gate_mn_lane = 0u;
            unsigned up_sc_lane = 0u;
            unsigned up_mn_lane = 0u;
            if (lane < 8u) {
                const unsigned j = lane;
                if (j < 4u) {
                    gate_sc_lane = gate_block[4u + j] & 63u;
                    gate_mn_lane = gate_block[4u + j + 4u] & 63u;
                    up_sc_lane = up_block[4u + j] & 63u;
                    up_mn_lane = up_block[4u + j + 4u] & 63u;
                } else {
                    gate_sc_lane =
                        (gate_block[4u + j + 4u] & 0x0fu)
                        | ((gate_block[4u + j - 4u] >> 6) << 4);
                    gate_mn_lane =
                        (gate_block[4u + j + 4u] >> 4)
                        | ((gate_block[4u + j] >> 6) << 4);
                    up_sc_lane =
                        (up_block[4u + j + 4u] & 0x0fu)
                        | ((up_block[4u + j - 4u] >> 6) << 4);
                    up_mn_lane =
                        (up_block[4u + j + 4u] >> 4)
                        | ((up_block[4u + j] >> 6) << 4);
                }
            }
            for (unsigned tid = lane; tid < 256u; tid += 32u) {
                const unsigned j = tid >> 5;
                const unsigned gate_sc =
                    __shfl_sync(0xffffffffu, gate_sc_lane, j);
                const unsigned gate_mn =
                    __shfl_sync(0xffffffffu, gate_mn_lane, j);
                const unsigned up_sc =
                    __shfl_sync(0xffffffffu, up_sc_lane, j);
                const unsigned up_mn =
                    __shfl_sync(0xffffffffu, up_mn_lane, j);
                const unsigned local = tid & 63u;
                const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
                unsigned gate_q = gate_block[16u + q_index];
                unsigned up_q = up_block[16u + q_index];
                gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
                up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
                const float gate_y =
                    (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;
                const float up_y =
                    (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;
                gate_acc0 += gate_y * input_tile0[tid];
                up_acc0 += up_y * input_tile0[tid];
                if (paired) {
                    gate_acc1 += gate_y * input_tile1[tid];
                    up_acc1 += up_y * input_tile1[tid];
                }
            }
        }
        __syncthreads();
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
    }
    if (valid && lane == 0u) {
        const unsigned out0 = primary_slot * rows + row;
        if (fuse_silu != 0u) {
            gate_out[out0] = (gate_acc0 / (1.0f + expf(-gate_acc0))) * up_acc0;
        } else {
            gate_out[out0] = gate_acc0;
            up_out[out0] = up_acc0;
        }
        if (paired) {
            const unsigned out1 = partner_slot * rows + row;
            if (fuse_silu != 0u) {
                gate_out[out1] = (gate_acc1 / (1.0f + expf(-gate_acc1))) * up_acc1;
            } else {
                gate_out[out1] = gate_acc1;
                up_out[out1] = up_acc1;
            }
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned slot = blockIdx.y;
    const bool valid = row < rows;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = valid ? gate_weights[slot] + row * row_bytes : nullptr;
    const unsigned char* up_row = valid ? up_weights[slot] + row * row_bytes : nullptr;
    const float* input_row = input + token_ids[slot] * blocks_per_row * 256u;
    __shared__ float input_tile[256];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        input_tile[threadIdx.x] = input_row[b * 256u + threadIdx.x];
        __syncthreads();
        if (valid) {
            const unsigned char* gate_block = gate_row + b * 144u;
            const unsigned char* up_block = up_row + b * 144u;
            const unsigned gate_raw_d =
                (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
            const unsigned gate_raw_dmin =
                (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
            const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
            const unsigned up_raw_dmin =
                (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
            const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
            const float gate_dmin =
                __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
            const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
            const float up_dmin =
                __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
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
                    gate_sc =
                        (gate_block[4u + j + 4u] & 0x0fu) | ((gate_block[4u + j - 4u] >> 6) << 4);
                    gate_mn = (gate_block[4u + j + 4u] >> 4)
                        | ((gate_block[4u + j] >> 6) << 4);
                    up_sc =
                        (up_block[4u + j + 4u] & 0x0fu) | ((up_block[4u + j - 4u] >> 6) << 4);
                    up_mn =
                        (up_block[4u + j + 4u] >> 4) | ((up_block[4u + j] >> 6) << 4);
                }
                const unsigned local = tid & 63u;
                const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
                unsigned gate_q = gate_block[16u + q_index];
                unsigned up_q = up_block[16u + q_index];
                gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
                up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
                const float x = input_tile[tid];
                gate_acc +=
                    ((gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn) * x;
                up_acc += ((up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn) * x;
            }
        }
        __syncthreads();
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    if (valid && lane == 0u) {
        const unsigned out_idx = slot * rows + row;
        gate_out[out_idx] = gate_acc;
        up_out[out_idx] = up_acc;
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_q8dot_by_token_warp8(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned slot = blockIdx.y;
    const bool valid = row < rows;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row =
        valid ? gate_weights[slot] + row * row_bytes : nullptr;
    const unsigned char* up_row =
        valid ? up_weights[slot] + row * row_bytes : nullptr;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (valid) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* gate_block = gate_row + b * 144u;
            const unsigned char* up_block = up_row + b * 144u;
            const unsigned gate_raw_d =
                (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
            const unsigned gate_raw_dmin =
                (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
            const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
            const unsigned up_raw_dmin =
                (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
            const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
            const float gate_dmin =
                __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
            const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
            const float up_dmin =
                __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
            const unsigned j0 = lane >> 3;
            const unsigned j1 = j0 + 4u;
            const unsigned elem = (lane & 7u) * 4u;
            const unsigned gate_sc0 = gate_block[4u + j0] & 63u;
            const unsigned gate_mn0 = gate_block[8u + j0] & 63u;
            const unsigned gate_sc1 =
                (gate_block[8u + j1] & 0x0fu) | ((gate_block[j1] >> 6) << 4);
            const unsigned gate_mn1 =
                (gate_block[8u + j1] >> 4) | ((gate_block[4u + j1] >> 6) << 4);
            const unsigned up_sc0 = up_block[4u + j0] & 63u;
            const unsigned up_mn0 = up_block[8u + j0] & 63u;
            const unsigned up_sc1 =
                (up_block[8u + j1] & 0x0fu) | ((up_block[j1] >> 6) << 4);
            const unsigned up_mn1 =
                (up_block[8u + j1] >> 4) | ((up_block[4u + j1] >> 6) << 4);
            const unsigned q_index0 = (j0 >> 1) * 32u + elem;
            const unsigned q_index1 = q_index0 + 64u;
            const int gate_pack0 =
                rnb_q4_pack4(gate_block + 16u + q_index0, j0);
            const int gate_pack1 =
                rnb_q4_pack4(gate_block + 16u + q_index1, j1);
            const int up_pack0 =
                rnb_q4_pack4(up_block + 16u + q_index0, j0);
            const int up_pack1 =
                rnb_q4_pack4(up_block + 16u + q_index1, j1);
            const signed char* x_qs0 =
                input_qs + b * 256u + j0 * 32u + elem;
            const signed char* x_qs1 = x_qs0 + 128u;
            const int x_pack0 = rnb_load_i32_aligned4(x_qs0);
            const int x_pack1 = rnb_load_i32_aligned4(x_qs1);
            const int x_sum0 = __dp4a(0x01010101, x_pack0, 0);
            const int x_sum1 = __dp4a(0x01010101, x_pack1, 0);
            const float x_d0 = input_ds[b * 8u + j0];
            const float x_d1 = input_ds[b * 8u + j1];
            gate_acc += x_d0
                * ((gate_d * (float)gate_sc0)
                        * (float)__dp4a(gate_pack0, x_pack0, 0)
                    - gate_dmin * (float)gate_mn0 * (float)x_sum0);
            gate_acc += x_d1
                * ((gate_d * (float)gate_sc1)
                        * (float)__dp4a(gate_pack1, x_pack1, 0)
                    - gate_dmin * (float)gate_mn1 * (float)x_sum1);
            up_acc += x_d0
                * ((up_d * (float)up_sc0) * (float)__dp4a(up_pack0, x_pack0, 0)
                    - up_dmin * (float)up_mn0 * (float)x_sum0);
            up_acc += x_d1
                * ((up_d * (float)up_sc1) * (float)__dp4a(up_pack1, x_pack1, 0)
                    - up_dmin * (float)up_mn1 * (float)x_sum1);
        }
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    if (valid && lane == 0u) {
        const unsigned out_idx = slot * rows + row;
        gate_out[out_idx] = gate_acc;
        up_out[out_idx] = up_acc;
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token_group4(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned group = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    __shared__ float partial_gate0[256];
    __shared__ float partial_gate1[256];
    __shared__ float partial_gate2[256];
    __shared__ float partial_gate3[256];
    __shared__ float partial_up0[256];
    __shared__ float partial_up1[256];
    __shared__ float partial_up2[256];
    __shared__ float partial_up3[256];

    float gate_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float gate_acc2 = 0.0f;
    float gate_acc3 = 0.0f;
    float up_acc0 = 0.0f;
    float up_acc1 = 0.0f;
    float up_acc2 = 0.0f;
    float up_acc3 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned j = tid >> 5;
        const unsigned local = tid & 63u;
        const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
        const unsigned x_off = b * 256u + tid;

        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
        unsigned gate_sc;
        unsigned gate_mn;
        if (j < 4) {
            gate_sc = gate_block[4 + j] & 63u;
            gate_mn = gate_block[4 + j + 4] & 63u;
        } else {
            gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
            gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
        }
        unsigned gate_q = gate_block[16 + q_index];
        gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
        const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
        unsigned up_sc;
        unsigned up_mn;
        if (j < 4) {
            up_sc = up_block[4 + j] & 63u;
            up_mn = up_block[4 + j + 4] & 63u;
        } else {
            up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
            up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
        }
        unsigned up_q = up_block[16 + q_index];
        up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
        const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;

        const float* input0 = input + token_ids[slot_start + 0u] * blocks_per_row * 256u;
        gate_acc0 += gate_y * input0[x_off];
        up_acc0 += up_y * input0[x_off];
        if (group_len > 1u) {
            const float* input1 = input + token_ids[slot_start + 1u] * blocks_per_row * 256u;
            gate_acc1 += gate_y * input1[x_off];
            up_acc1 += up_y * input1[x_off];
        }
        if (group_len > 2u) {
            const float* input2 = input + token_ids[slot_start + 2u] * blocks_per_row * 256u;
            gate_acc2 += gate_y * input2[x_off];
            up_acc2 += up_y * input2[x_off];
        }
        if (group_len > 3u) {
            const float* input3 = input + token_ids[slot_start + 3u] * blocks_per_row * 256u;
            gate_acc3 += gate_y * input3[x_off];
            up_acc3 += up_y * input3[x_off];
        }
    }

    partial_gate0[tid] = gate_acc0;
    partial_gate1[tid] = gate_acc1;
    partial_gate2[tid] = gate_acc2;
    partial_gate3[tid] = gate_acc3;
    partial_up0[tid] = up_acc0;
    partial_up1[tid] = up_acc1;
    partial_up2[tid] = up_acc2;
    partial_up3[tid] = up_acc3;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_gate0[tid] += partial_gate0[tid + stride];
            partial_gate1[tid] += partial_gate1[tid + stride];
            partial_gate2[tid] += partial_gate2[tid + stride];
            partial_gate3[tid] += partial_gate3[tid + stride];
            partial_up0[tid] += partial_up0[tid + stride];
            partial_up1[tid] += partial_up1[tid + stride];
            partial_up2[tid] += partial_up2[tid + stride];
            partial_up3[tid] += partial_up3[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const unsigned out0 = slot_start * rows + row;
        gate_out[out0] = partial_gate0[0];
        up_out[out0] = partial_up0[0];
        if (group_len > 1u) {
            gate_out[out0 + rows] = partial_gate1[0];
            up_out[out0 + rows] = partial_up1[0];
        }
        if (group_len > 2u) {
            gate_out[out0 + 2u * rows] = partial_gate2[0];
            up_out[out0 + 2u * rows] = partial_up2[0];
        }
        if (group_len > 3u) {
            gate_out[out0 + 3u * rows] = partial_gate3[0];
            up_out[out0 + 3u * rows] = partial_up3[0];
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token_group4_warp4(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float gate_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float gate_acc2 = 0.0f;
    float gate_acc3 = 0.0f;
    float up_acc0 = 0.0f;
    float up_acc1 = 0.0f;
    float up_acc2 = 0.0f;
    float up_acc3 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned base_x = b * 256u;
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned gate_sc;
            unsigned gate_mn;
            if (j < 4u) {
                gate_sc = gate_block[4 + j] & 63u;
                gate_mn = gate_block[4 + j + 4] & 63u;
            } else {
                gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
                gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned gate_q = gate_block[16 + q_index];
            gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
            const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;

            unsigned up_sc;
            unsigned up_mn;
            if (j < 4u) {
                up_sc = up_block[4 + j] & 63u;
                up_mn = up_block[4 + j + 4] & 63u;
            } else {
                up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
                up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
            }
            unsigned up_q = up_block[16 + q_index];
            up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
            const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;

            const unsigned x_off = base_x + tid;
            const float* input0 = input + token_ids[slot_start + 0u] * blocks_per_row * 256u;
            gate_acc0 += gate_y * input0[x_off];
            up_acc0 += up_y * input0[x_off];
            if (group_len > 1u) {
                const float* input1 = input + token_ids[slot_start + 1u] * blocks_per_row * 256u;
                gate_acc1 += gate_y * input1[x_off];
                up_acc1 += up_y * input1[x_off];
            }
            if (group_len > 2u) {
                const float* input2 = input + token_ids[slot_start + 2u] * blocks_per_row * 256u;
                gate_acc2 += gate_y * input2[x_off];
                up_acc2 += up_y * input2[x_off];
            }
            if (group_len > 3u) {
                const float* input3 = input + token_ids[slot_start + 3u] * blocks_per_row * 256u;
                gate_acc3 += gate_y * input3[x_off];
                up_acc3 += up_y * input3[x_off];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        gate_acc2 += __shfl_down_sync(0xffffffffu, gate_acc2, offset);
        gate_acc3 += __shfl_down_sync(0xffffffffu, gate_acc3, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
        up_acc2 += __shfl_down_sync(0xffffffffu, up_acc2, offset);
        up_acc3 += __shfl_down_sync(0xffffffffu, up_acc3, offset);
    }
    if (lane == 0u) {
        const unsigned out0 = slot_start * rows + row;
        gate_out[out0] = gate_acc0;
        up_out[out0] = up_acc0;
        if (group_len > 1u) {
            gate_out[out0 + rows] = gate_acc1;
            up_out[out0 + rows] = up_acc1;
        }
        if (group_len > 2u) {
            gate_out[out0 + 2u * rows] = gate_acc2;
            up_out[out0 + 2u * rows] = up_acc2;
        }
        if (group_len > 3u) {
            gate_out[out0 + 3u * rows] = gate_acc3;
            up_out[out0 + 3u * rows] = up_acc3;
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token_group8_warp4(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 8u) {
        return;
    }

    float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
    float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
    float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
    float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned base_x = b * 256u;
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned gate_sc;
            unsigned gate_mn;
            if (j < 4u) {
                gate_sc = gate_block[4 + j] & 63u;
                gate_mn = gate_block[4 + j + 4] & 63u;
            } else {
                gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
                gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned gate_q = gate_block[16 + q_index];
            gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
            const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;

            unsigned up_sc;
            unsigned up_mn;
            if (j < 4u) {
                up_sc = up_block[4 + j] & 63u;
                up_mn = up_block[4 + j + 4] & 63u;
            } else {
                up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
                up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
            }
            unsigned up_q = up_block[16 + q_index];
            up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
            const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;

            const unsigned x_off = base_x + tid;
            const float* input0 = input + token_ids[slot_start + 0u] * blocks_per_row * 256u;
            gate_acc0 += gate_y * input0[x_off];
            up_acc0 += up_y * input0[x_off];
            if (group_len > 1u) {
                const float* input1 = input + token_ids[slot_start + 1u] * blocks_per_row * 256u;
                gate_acc1 += gate_y * input1[x_off];
                up_acc1 += up_y * input1[x_off];
            }
            if (group_len > 2u) {
                const float* input2 = input + token_ids[slot_start + 2u] * blocks_per_row * 256u;
                gate_acc2 += gate_y * input2[x_off];
                up_acc2 += up_y * input2[x_off];
            }
            if (group_len > 3u) {
                const float* input3 = input + token_ids[slot_start + 3u] * blocks_per_row * 256u;
                gate_acc3 += gate_y * input3[x_off];
                up_acc3 += up_y * input3[x_off];
            }
            if (group_len > 4u) {
                const float* input4 = input + token_ids[slot_start + 4u] * blocks_per_row * 256u;
                gate_acc4 += gate_y * input4[x_off];
                up_acc4 += up_y * input4[x_off];
            }
            if (group_len > 5u) {
                const float* input5 = input + token_ids[slot_start + 5u] * blocks_per_row * 256u;
                gate_acc5 += gate_y * input5[x_off];
                up_acc5 += up_y * input5[x_off];
            }
            if (group_len > 6u) {
                const float* input6 = input + token_ids[slot_start + 6u] * blocks_per_row * 256u;
                gate_acc6 += gate_y * input6[x_off];
                up_acc6 += up_y * input6[x_off];
            }
            if (group_len > 7u) {
                const float* input7 = input + token_ids[slot_start + 7u] * blocks_per_row * 256u;
                gate_acc7 += gate_y * input7[x_off];
                up_acc7 += up_y * input7[x_off];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        gate_acc2 += __shfl_down_sync(0xffffffffu, gate_acc2, offset);
        gate_acc3 += __shfl_down_sync(0xffffffffu, gate_acc3, offset);
        gate_acc4 += __shfl_down_sync(0xffffffffu, gate_acc4, offset);
        gate_acc5 += __shfl_down_sync(0xffffffffu, gate_acc5, offset);
        gate_acc6 += __shfl_down_sync(0xffffffffu, gate_acc6, offset);
        gate_acc7 += __shfl_down_sync(0xffffffffu, gate_acc7, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
        up_acc2 += __shfl_down_sync(0xffffffffu, up_acc2, offset);
        up_acc3 += __shfl_down_sync(0xffffffffu, up_acc3, offset);
        up_acc4 += __shfl_down_sync(0xffffffffu, up_acc4, offset);
        up_acc5 += __shfl_down_sync(0xffffffffu, up_acc5, offset);
        up_acc6 += __shfl_down_sync(0xffffffffu, up_acc6, offset);
        up_acc7 += __shfl_down_sync(0xffffffffu, up_acc7, offset);
    }
    if (lane == 0u) {
        const unsigned out0 = slot_start * rows + row;
        gate_out[out0] = gate_acc0;
        up_out[out0] = up_acc0;
        if (group_len > 1u) {
            gate_out[out0 + rows] = gate_acc1;
            up_out[out0 + rows] = up_acc1;
        }
        if (group_len > 2u) {
            gate_out[out0 + 2u * rows] = gate_acc2;
            up_out[out0 + 2u * rows] = up_acc2;
        }
        if (group_len > 3u) {
            gate_out[out0 + 3u * rows] = gate_acc3;
            up_out[out0 + 3u * rows] = up_acc3;
        }
        if (group_len > 4u) {
            gate_out[out0 + 4u * rows] = gate_acc4;
            up_out[out0 + 4u * rows] = up_acc4;
        }
        if (group_len > 5u) {
            gate_out[out0 + 5u * rows] = gate_acc5;
            up_out[out0 + 5u * rows] = up_acc5;
        }
        if (group_len > 6u) {
            gate_out[out0 + 6u * rows] = gate_acc6;
            up_out[out0 + 6u * rows] = up_acc6;
        }
        if (group_len > 7u) {
            gate_out[out0 + 7u * rows] = gate_acc7;
            up_out[out0 + 7u * rows] = up_acc7;
        }
    }
}

__device__ __forceinline__ float rnb_qwen_silu_mul_f32(const float gate, const float up) {
    return (gate / (1.0f + expf(-gate))) * up;
}

extern "C" __global__ void rnb_q4k_selected_gate_up_silu_pack4_f32_by_token_group4_warp4(
    float* __restrict__ packed_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned input_blocks_per_row,
    unsigned pack_blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || pack_blocks_per_row == 0u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float gate_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float gate_acc2 = 0.0f;
    float gate_acc3 = 0.0f;
    float up_acc0 = 0.0f;
    float up_acc1 = 0.0f;
    float up_acc2 = 0.0f;
    float up_acc3 = 0.0f;

    const unsigned row_bytes = input_blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < input_blocks_per_row; ++b) {
        const unsigned base_x = b * 256u;
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned gate_sc;
            unsigned gate_mn;
            if (j < 4u) {
                gate_sc = gate_block[4 + j] & 63u;
                gate_mn = gate_block[4 + j + 4] & 63u;
            } else {
                gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
                gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned gate_q = gate_block[16 + q_index];
            gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
            const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;

            unsigned up_sc;
            unsigned up_mn;
            if (j < 4u) {
                up_sc = up_block[4 + j] & 63u;
                up_mn = up_block[4 + j + 4] & 63u;
            } else {
                up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
                up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
            }
            unsigned up_q = up_block[16 + q_index];
            up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
            const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;

            const unsigned x_off = base_x + tid;
            const float* input0 = input + token_ids[slot_start + 0u] * input_blocks_per_row * 256u;
            gate_acc0 += gate_y * input0[x_off];
            up_acc0 += up_y * input0[x_off];
            if (group_len > 1u) {
                const float* input1 = input + token_ids[slot_start + 1u] * input_blocks_per_row * 256u;
                gate_acc1 += gate_y * input1[x_off];
                up_acc1 += up_y * input1[x_off];
            }
            if (group_len > 2u) {
                const float* input2 = input + token_ids[slot_start + 2u] * input_blocks_per_row * 256u;
                gate_acc2 += gate_y * input2[x_off];
                up_acc2 += up_y * input2[x_off];
            }
            if (group_len > 3u) {
                const float* input3 = input + token_ids[slot_start + 3u] * input_blocks_per_row * 256u;
                gate_acc3 += gate_y * input3[x_off];
                up_acc3 += up_y * input3[x_off];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        gate_acc2 += __shfl_down_sync(0xffffffffu, gate_acc2, offset);
        gate_acc3 += __shfl_down_sync(0xffffffffu, gate_acc3, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
        up_acc2 += __shfl_down_sync(0xffffffffu, up_acc2, offset);
        up_acc3 += __shfl_down_sync(0xffffffffu, up_acc3, offset);
    }

    if (lane == 0u) {
        const unsigned row_block = row >> 8;
        if (row_block >= pack_blocks_per_row) {
            return;
        }
        const unsigned elem = row & 255u;
        const unsigned out0 = ((group * pack_blocks_per_row + row_block) * 256u + elem) * 4u;
        packed_out[out0] = rnb_qwen_silu_mul_f32(gate_acc0, up_acc0);
        if (group_len > 1u) {
            packed_out[out0 + 1u] = rnb_qwen_silu_mul_f32(gate_acc1, up_acc1);
        }
        if (group_len > 2u) {
            packed_out[out0 + 2u] = rnb_qwen_silu_mul_f32(gate_acc2, up_acc2);
        }
        if (group_len > 3u) {
            packed_out[out0 + 3u] = rnb_qwen_silu_mul_f32(gate_acc3, up_acc3);
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_silu_pack4_f32_by_token_group8_warp4(
    float* __restrict__ packed_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    const unsigned* __restrict__ pack_group_offsets,
    unsigned rows,
    unsigned input_blocks_per_row,
    unsigned pack_blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || pack_blocks_per_row == 0u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 8u) {
        return;
    }

    float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
    float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
    float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
    float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;

    const unsigned row_bytes = input_blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < input_blocks_per_row; ++b) {
        const unsigned base_x = b * 256u;
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned gate_sc;
            unsigned gate_mn;
            if (j < 4u) {
                gate_sc = gate_block[4 + j] & 63u;
                gate_mn = gate_block[4 + j + 4] & 63u;
            } else {
                gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
                gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned gate_q = gate_block[16 + q_index];
            gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
            const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;

            unsigned up_sc;
            unsigned up_mn;
            if (j < 4u) {
                up_sc = up_block[4 + j] & 63u;
                up_mn = up_block[4 + j + 4] & 63u;
            } else {
                up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
                up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
            }
            unsigned up_q = up_block[16 + q_index];
            up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
            const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;

            const unsigned x_off = base_x + tid;
            const float* input0 = input + token_ids[slot_start + 0u] * input_blocks_per_row * 256u;
            gate_acc0 += gate_y * input0[x_off];
            up_acc0 += up_y * input0[x_off];
            if (group_len > 1u) {
                const float* input1 = input + token_ids[slot_start + 1u] * input_blocks_per_row * 256u;
                gate_acc1 += gate_y * input1[x_off];
                up_acc1 += up_y * input1[x_off];
            }
            if (group_len > 2u) {
                const float* input2 = input + token_ids[slot_start + 2u] * input_blocks_per_row * 256u;
                gate_acc2 += gate_y * input2[x_off];
                up_acc2 += up_y * input2[x_off];
            }
            if (group_len > 3u) {
                const float* input3 = input + token_ids[slot_start + 3u] * input_blocks_per_row * 256u;
                gate_acc3 += gate_y * input3[x_off];
                up_acc3 += up_y * input3[x_off];
            }
            if (group_len > 4u) {
                const float* input4 = input + token_ids[slot_start + 4u] * input_blocks_per_row * 256u;
                gate_acc4 += gate_y * input4[x_off];
                up_acc4 += up_y * input4[x_off];
            }
            if (group_len > 5u) {
                const float* input5 = input + token_ids[slot_start + 5u] * input_blocks_per_row * 256u;
                gate_acc5 += gate_y * input5[x_off];
                up_acc5 += up_y * input5[x_off];
            }
            if (group_len > 6u) {
                const float* input6 = input + token_ids[slot_start + 6u] * input_blocks_per_row * 256u;
                gate_acc6 += gate_y * input6[x_off];
                up_acc6 += up_y * input6[x_off];
            }
            if (group_len > 7u) {
                const float* input7 = input + token_ids[slot_start + 7u] * input_blocks_per_row * 256u;
                gate_acc7 += gate_y * input7[x_off];
                up_acc7 += up_y * input7[x_off];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        gate_acc2 += __shfl_down_sync(0xffffffffu, gate_acc2, offset);
        gate_acc3 += __shfl_down_sync(0xffffffffu, gate_acc3, offset);
        gate_acc4 += __shfl_down_sync(0xffffffffu, gate_acc4, offset);
        gate_acc5 += __shfl_down_sync(0xffffffffu, gate_acc5, offset);
        gate_acc6 += __shfl_down_sync(0xffffffffu, gate_acc6, offset);
        gate_acc7 += __shfl_down_sync(0xffffffffu, gate_acc7, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
        up_acc2 += __shfl_down_sync(0xffffffffu, up_acc2, offset);
        up_acc3 += __shfl_down_sync(0xffffffffu, up_acc3, offset);
        up_acc4 += __shfl_down_sync(0xffffffffu, up_acc4, offset);
        up_acc5 += __shfl_down_sync(0xffffffffu, up_acc5, offset);
        up_acc6 += __shfl_down_sync(0xffffffffu, up_acc6, offset);
        up_acc7 += __shfl_down_sync(0xffffffffu, up_acc7, offset);
    }

    if (lane == 0u) {
        const unsigned row_block = row >> 8;
        if (row_block >= pack_blocks_per_row) {
            return;
        }
        const unsigned elem = row & 255u;
        const unsigned pack_group0 = pack_group_offsets[group];
        const unsigned base0 = ((pack_group0 * pack_blocks_per_row + row_block) * 256u + elem) * 4u;
        packed_out[base0] = rnb_qwen_silu_mul_f32(gate_acc0, up_acc0);
        if (group_len > 1u) {
            packed_out[base0 + 1u] = rnb_qwen_silu_mul_f32(gate_acc1, up_acc1);
        }
        if (group_len > 2u) {
            packed_out[base0 + 2u] = rnb_qwen_silu_mul_f32(gate_acc2, up_acc2);
        }
        if (group_len > 3u) {
            packed_out[base0 + 3u] = rnb_qwen_silu_mul_f32(gate_acc3, up_acc3);
        }
        if (group_len > 4u) {
            const unsigned pack_group1 = pack_group0 + 1u;
            const unsigned base1 = ((pack_group1 * pack_blocks_per_row + row_block) * 256u + elem) * 4u;
            packed_out[base1] = rnb_qwen_silu_mul_f32(gate_acc4, up_acc4);
            if (group_len > 5u) {
                packed_out[base1 + 1u] = rnb_qwen_silu_mul_f32(gate_acc5, up_acc5);
            }
            if (group_len > 6u) {
                packed_out[base1 + 2u] = rnb_qwen_silu_mul_f32(gate_acc6, up_acc6);
            }
            if (group_len > 7u) {
                packed_out[base1 + 3u] = rnb_qwen_silu_mul_f32(gate_acc7, up_acc7);
            }
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_gate_up_silu_by_token_group8_warp4(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 8u) {
        return;
    }

    float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
    float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
    float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
    float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned base_x = b * 256u;
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned gate_sc;
            unsigned gate_mn;
            if (j < 4u) {
                gate_sc = gate_block[4 + j] & 63u;
                gate_mn = gate_block[4 + j + 4] & 63u;
            } else {
                gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
                gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned gate_q = gate_block[16 + q_index];
            gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
            const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;

            unsigned up_sc;
            unsigned up_mn;
            if (j < 4u) {
                up_sc = up_block[4 + j] & 63u;
                up_mn = up_block[4 + j + 4] & 63u;
            } else {
                up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
                up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
            }
            unsigned up_q = up_block[16 + q_index];
            up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
            const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;

            const unsigned x_off = base_x + tid;
            const float* input0 = input + token_ids[slot_start + 0u] * blocks_per_row * 256u;
            gate_acc0 += gate_y * input0[x_off];
            up_acc0 += up_y * input0[x_off];
            if (group_len > 1u) {
                const float* input1 = input + token_ids[slot_start + 1u] * blocks_per_row * 256u;
                gate_acc1 += gate_y * input1[x_off];
                up_acc1 += up_y * input1[x_off];
            }
            if (group_len > 2u) {
                const float* input2 = input + token_ids[slot_start + 2u] * blocks_per_row * 256u;
                gate_acc2 += gate_y * input2[x_off];
                up_acc2 += up_y * input2[x_off];
            }
            if (group_len > 3u) {
                const float* input3 = input + token_ids[slot_start + 3u] * blocks_per_row * 256u;
                gate_acc3 += gate_y * input3[x_off];
                up_acc3 += up_y * input3[x_off];
            }
            if (group_len > 4u) {
                const float* input4 = input + token_ids[slot_start + 4u] * blocks_per_row * 256u;
                gate_acc4 += gate_y * input4[x_off];
                up_acc4 += up_y * input4[x_off];
            }
            if (group_len > 5u) {
                const float* input5 = input + token_ids[slot_start + 5u] * blocks_per_row * 256u;
                gate_acc5 += gate_y * input5[x_off];
                up_acc5 += up_y * input5[x_off];
            }
            if (group_len > 6u) {
                const float* input6 = input + token_ids[slot_start + 6u] * blocks_per_row * 256u;
                gate_acc6 += gate_y * input6[x_off];
                up_acc6 += up_y * input6[x_off];
            }
            if (group_len > 7u) {
                const float* input7 = input + token_ids[slot_start + 7u] * blocks_per_row * 256u;
                gate_acc7 += gate_y * input7[x_off];
                up_acc7 += up_y * input7[x_off];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        gate_acc2 += __shfl_down_sync(0xffffffffu, gate_acc2, offset);
        gate_acc3 += __shfl_down_sync(0xffffffffu, gate_acc3, offset);
        gate_acc4 += __shfl_down_sync(0xffffffffu, gate_acc4, offset);
        gate_acc5 += __shfl_down_sync(0xffffffffu, gate_acc5, offset);
        gate_acc6 += __shfl_down_sync(0xffffffffu, gate_acc6, offset);
        gate_acc7 += __shfl_down_sync(0xffffffffu, gate_acc7, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
        up_acc2 += __shfl_down_sync(0xffffffffu, up_acc2, offset);
        up_acc3 += __shfl_down_sync(0xffffffffu, up_acc3, offset);
        up_acc4 += __shfl_down_sync(0xffffffffu, up_acc4, offset);
        up_acc5 += __shfl_down_sync(0xffffffffu, up_acc5, offset);
        up_acc6 += __shfl_down_sync(0xffffffffu, up_acc6, offset);
        up_acc7 += __shfl_down_sync(0xffffffffu, up_acc7, offset);
    }
    if (lane == 0u) {
        const unsigned out0 = slot_start * rows + row;
        gate_out[out0] = rnb_qwen_silu_mul_f32(gate_acc0, up_acc0);
        if (group_len > 1u) {
            gate_out[out0 + rows] = rnb_qwen_silu_mul_f32(gate_acc1, up_acc1);
        }
        if (group_len > 2u) {
            gate_out[out0 + 2u * rows] = rnb_qwen_silu_mul_f32(gate_acc2, up_acc2);
        }
        if (group_len > 3u) {
            gate_out[out0 + 3u * rows] = rnb_qwen_silu_mul_f32(gate_acc3, up_acc3);
        }
        if (group_len > 4u) {
            gate_out[out0 + 4u * rows] = rnb_qwen_silu_mul_f32(gate_acc4, up_acc4);
        }
        if (group_len > 5u) {
            gate_out[out0 + 5u * rows] = rnb_qwen_silu_mul_f32(gate_acc5, up_acc5);
        }
        if (group_len > 6u) {
            gate_out[out0 + 6u * rows] = rnb_qwen_silu_mul_f32(gate_acc6, up_acc6);
        }
        if (group_len > 7u) {
            gate_out[out0 + 7u * rows] = rnb_qwen_silu_mul_f32(gate_acc7, up_acc7);
        }
        (void)up_out;
    }
}

__device__ __forceinline__ void rnb_q4k_selected_gate_up_q8dot_accum(
    int gate_pack,
    int up_pack,
    float gate_scale,
    float gate_min,
    float up_scale,
    float up_min,
    const signed char* __restrict__ input_qs,
    float input_d,
    float& gate_acc,
    float& up_acc) {
    const int input_pack = rnb_load_i32_aligned4(input_qs);
    const int input_sum = __dp4a(0x01010101, input_pack, 0);
    gate_acc += input_d * (gate_scale * (float)__dp4a(gate_pack, input_pack, 0)
        - gate_min * (float)input_sum);
    up_acc += input_d * (up_scale * (float)__dp4a(up_pack, input_pack, 0)
        - up_min * (float)input_sum);
}

extern "C" __global__ void rnb_q4k_selected_gate_up_silu_q8dot_by_token_group8_warp4(
    float* __restrict__ packed_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    const unsigned* __restrict__ pack_group_offsets,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned pack_blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || pack_blocks_per_row == 0u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 8u) {
        return;
    }

    float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
    float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
    float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
    float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;
    const unsigned input_row_stride = blocks_per_row * 256u;
    const unsigned input_scale_stride = blocks_per_row * 8u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
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
                gate_sc = (gate_block[4u + j + 4u] & 0x0fu)
                    | ((gate_block[4u + j - 4u] >> 6) << 4);
                gate_mn = (gate_block[4u + j + 4u] >> 4)
                    | ((gate_block[4u + j] >> 6) << 4);
                up_sc = (up_block[4u + j + 4u] & 0x0fu)
                    | ((up_block[4u + j - 4u] >> 6) << 4);
                up_mn = (up_block[4u + j + 4u] >> 4)
                    | ((up_block[4u + j] >> 6) << 4);
            }

            const unsigned elem = (chunk & 7u) * 4u;
            const unsigned q_index = (j >> 1) * 32u + elem;
            const int gate_pack = rnb_q4_pack4(gate_block + 16u + q_index, j);
            const int up_pack = rnb_q4_pack4(up_block + 16u + q_index, j);
            const float gate_scale = gate_d * (float)gate_sc;
            const float gate_min = gate_dmin * (float)gate_mn;
            const float up_scale = up_d * (float)up_sc;
            const float up_min = up_dmin * (float)up_mn;
            const unsigned input_q_offset = b * 256u + j * 32u + elem;
            const unsigned input_d_offset = b * 8u + j;

#define RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(SLOT) \
            if (group_len > SLOT) { \
                const unsigned token = token_ids[slot_start + SLOT]; \
                rnb_q4k_selected_gate_up_q8dot_accum( \
                    gate_pack, up_pack, gate_scale, gate_min, up_scale, up_min, \
                    input_qs + token * input_row_stride + input_q_offset, \
                    input_ds[token * input_scale_stride + input_d_offset], \
                    gate_acc##SLOT, up_acc##SLOT); \
            }
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(0)
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(1)
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(2)
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(3)
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(4)
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(5)
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(6)
            RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT(7)
#undef RNB_Q4K_SELECTED_GATE_UP_Q8DOT_SLOT
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc0 += __shfl_down_sync(0xffffffffu, gate_acc0, offset);
        gate_acc1 += __shfl_down_sync(0xffffffffu, gate_acc1, offset);
        gate_acc2 += __shfl_down_sync(0xffffffffu, gate_acc2, offset);
        gate_acc3 += __shfl_down_sync(0xffffffffu, gate_acc3, offset);
        gate_acc4 += __shfl_down_sync(0xffffffffu, gate_acc4, offset);
        gate_acc5 += __shfl_down_sync(0xffffffffu, gate_acc5, offset);
        gate_acc6 += __shfl_down_sync(0xffffffffu, gate_acc6, offset);
        gate_acc7 += __shfl_down_sync(0xffffffffu, gate_acc7, offset);
        up_acc0 += __shfl_down_sync(0xffffffffu, up_acc0, offset);
        up_acc1 += __shfl_down_sync(0xffffffffu, up_acc1, offset);
        up_acc2 += __shfl_down_sync(0xffffffffu, up_acc2, offset);
        up_acc3 += __shfl_down_sync(0xffffffffu, up_acc3, offset);
        up_acc4 += __shfl_down_sync(0xffffffffu, up_acc4, offset);
        up_acc5 += __shfl_down_sync(0xffffffffu, up_acc5, offset);
        up_acc6 += __shfl_down_sync(0xffffffffu, up_acc6, offset);
        up_acc7 += __shfl_down_sync(0xffffffffu, up_acc7, offset);
    }
    if (lane == 0u) {
        if (pack_group_offsets == 0) {
            packed_out[(slot_start + 0u) * rows + row] =
                rnb_qwen_silu_mul_f32(gate_acc0, up_acc0);
            if (group_len > 1u) {
                packed_out[(slot_start + 1u) * rows + row] =
                    rnb_qwen_silu_mul_f32(gate_acc1, up_acc1);
            }
            if (group_len > 2u) {
                packed_out[(slot_start + 2u) * rows + row] =
                    rnb_qwen_silu_mul_f32(gate_acc2, up_acc2);
            }
            if (group_len > 3u) {
                packed_out[(slot_start + 3u) * rows + row] =
                    rnb_qwen_silu_mul_f32(gate_acc3, up_acc3);
            }
            if (group_len > 4u) {
                packed_out[(slot_start + 4u) * rows + row] =
                    rnb_qwen_silu_mul_f32(gate_acc4, up_acc4);
            }
            if (group_len > 5u) {
                packed_out[(slot_start + 5u) * rows + row] =
                    rnb_qwen_silu_mul_f32(gate_acc5, up_acc5);
            }
            if (group_len > 6u) {
                packed_out[(slot_start + 6u) * rows + row] =
                    rnb_qwen_silu_mul_f32(gate_acc6, up_acc6);
            }
            if (group_len > 7u) {
                packed_out[(slot_start + 7u) * rows + row] =
                    rnb_qwen_silu_mul_f32(gate_acc7, up_acc7);
            }
            return;
        }
        const unsigned row_block = row >> 8;
        if (row_block >= pack_blocks_per_row) {
            return;
        }
        const unsigned elem = row & 255u;
        const unsigned pack_group0 = pack_group_offsets[group];
        const unsigned base0 =
            ((pack_group0 * pack_blocks_per_row + row_block) * 256u + elem) * 4u;
        packed_out[base0] = rnb_qwen_silu_mul_f32(gate_acc0, up_acc0);
        if (group_len > 1u) {
            packed_out[base0 + 1u] = rnb_qwen_silu_mul_f32(gate_acc1, up_acc1);
        }
        if (group_len > 2u) {
            packed_out[base0 + 2u] = rnb_qwen_silu_mul_f32(gate_acc2, up_acc2);
        }
        if (group_len > 3u) {
            packed_out[base0 + 3u] = rnb_qwen_silu_mul_f32(gate_acc3, up_acc3);
        }
        if (group_len > 4u) {
            const unsigned pack_group1 = pack_group0 + 1u;
            const unsigned base1 =
                ((pack_group1 * pack_blocks_per_row + row_block) * 256u + elem) * 4u;
            packed_out[base1] = rnb_qwen_silu_mul_f32(gate_acc4, up_acc4);
            if (group_len > 5u) {
                packed_out[base1 + 1u] = rnb_qwen_silu_mul_f32(gate_acc5, up_acc5);
            }
            if (group_len > 6u) {
                packed_out[base1 + 2u] = rnb_qwen_silu_mul_f32(gate_acc6, up_acc6);
            }
            if (group_len > 7u) {
                packed_out[base1 + 3u] = rnb_qwen_silu_mul_f32(gate_acc7, up_acc7);
            }
        }
    }
}

__device__ __forceinline__ void rnb_q4k_selected_q8dot_store(
    float* __restrict__ output,
    const unsigned* __restrict__ pack_group_offsets,
    unsigned group,
    unsigned slot_start,
    unsigned local_slot,
    unsigned rows,
    unsigned row,
    unsigned pack_blocks_per_row,
    float value) {
    if (pack_group_offsets == 0) {
        output[(slot_start + local_slot) * rows + row] = value;
        return;
    }
    const unsigned pack_group = pack_group_offsets[group] + (local_slot >> 2);
    const unsigned row_block = row >> 8;
    const unsigned elem = row & 255u;
    const unsigned base =
        ((pack_group * pack_blocks_per_row + row_block) * 256u + elem) * 4u;
    output[base + (local_slot & 3u)] = value;
}

extern "C" __global__ void rnb_q4k_selected_gate_up_silu_q8dot_mmq_group8(
    float* __restrict__ output,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    const unsigned* __restrict__ pack_group_offsets,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned pack_blocks_per_row) {
#if __CUDA_ARCH__ < 750
    (void)output;
    (void)gate_weights;
    (void)up_weights;
    (void)input_qs;
    (void)input_ds;
    (void)token_ids;
    (void)group_meta;
    (void)pack_group_offsets;
    (void)rows;
    (void)blocks_per_row;
    (void)pack_blocks_per_row;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 32u;
    const unsigned group = blockIdx.y;
    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (warp >= 2u || group_len == 0u || group_len > 8u) {
        return;
    }

    __shared__ signed char gate_tile[32 * 32];
    __shared__ signed char up_tile[32 * 32];
    __shared__ signed char input_tile[8 * 32];
    __shared__ float gate_d[32];
    __shared__ float gate_dmin[32];
    __shared__ float up_d[32];
    __shared__ float up_dmin[32];
    __shared__ unsigned char gate_sc[32];
    __shared__ unsigned char gate_mn[32];
    __shared__ unsigned char up_sc[32];
    __shared__ unsigned char up_mn[32];
    __shared__ float activation_d[8];

    const unsigned warp_row_off = warp * 16u;
    const unsigned frag_row_a = lane >> 2;
    const unsigned frag_row_b = frag_row_a + 8u;
    const unsigned local_seq_a = (lane & 3u) << 1;
    const unsigned local_seq_b = local_seq_a + 1u;
    const unsigned row_a = row_base + warp_row_off + frag_row_a;
    const unsigned row_b = row_base + warp_row_off + frag_row_b;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;
    const unsigned row_bytes = blocks_per_row * 144u;
    float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        for (unsigned sub = 0; sub < 8u; ++sub) {
            if (tid < 32u) {
                const unsigned global_row = row_base + tid;
                if (global_row < rows) {
                    const unsigned char* gate_block =
                        gate_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned char* up_block =
                        up_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned gate_raw_d =
                        (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
                    const unsigned gate_raw_dmin =
                        (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
                    const unsigned up_raw_d =
                        (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
                    const unsigned up_raw_dmin =
                        (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
                    gate_d[tid] =
                        __half2float(__ushort_as_half((unsigned short)gate_raw_d));
                    gate_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
                    up_d[tid] = __half2float(__ushort_as_half((unsigned short)up_raw_d));
                    up_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
                    if (sub < 4u) {
                        gate_sc[tid] = gate_block[4u + sub] & 63u;
                        gate_mn[tid] = gate_block[8u + sub] & 63u;
                        up_sc[tid] = up_block[4u + sub] & 63u;
                        up_mn[tid] = up_block[8u + sub] & 63u;
                    } else {
                        gate_sc[tid] = (gate_block[8u + sub] & 0x0fu)
                            | ((gate_block[sub] >> 6) << 4);
                        gate_mn[tid] = (gate_block[8u + sub] >> 4)
                            | ((gate_block[4u + sub] >> 6) << 4);
                        up_sc[tid] = (up_block[8u + sub] & 0x0fu)
                            | ((up_block[sub] >> 6) << 4);
                        up_mn[tid] = (up_block[8u + sub] >> 4)
                            | ((up_block[4u + sub] >> 6) << 4);
                    }
                } else {
                    gate_d[tid] = 0.0f;
                    gate_dmin[tid] = 0.0f;
                    up_d[tid] = 0.0f;
                    up_dmin[tid] = 0.0f;
                    gate_sc[tid] = 0u;
                    gate_mn[tid] = 0u;
                    up_sc[tid] = 0u;
                    up_mn[tid] = 0u;
                }
            }

            for (unsigned load = tid; load < 256u; load += 64u) {
                const unsigned load_row = load >> 3;
                const unsigned load_off = (load & 7u) * 4u;
                const unsigned global_row = row_base + load_row;
                signed char* gate_dst = gate_tile + load_row * 32u + load_off;
                signed char* up_dst = up_tile + load_row * 32u + load_off;
                if (global_row < rows) {
                    const unsigned char* gate_block =
                        gate_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned char* up_block =
                        up_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned nibble_base = 16u + (sub >> 1) * 32u;
                    const unsigned gate_packed =
                        *reinterpret_cast<const unsigned*>(gate_block + nibble_base + load_off);
                    const unsigned up_packed =
                        *reinterpret_cast<const unsigned*>(up_block + nibble_base + load_off);
                    const unsigned shift = (sub & 1u) * 4u;
                    *reinterpret_cast<unsigned*>(gate_dst) =
                        (gate_packed >> shift) & 0x0f0f0f0fu;
                    *reinterpret_cast<unsigned*>(up_dst) =
                        (up_packed >> shift) & 0x0f0f0f0fu;
                } else {
                    *reinterpret_cast<unsigned*>(gate_dst) = 0u;
                    *reinterpret_cast<unsigned*>(up_dst) = 0u;
                }
            }

            const unsigned local_seq = tid >> 3;
            const unsigned input_off = (tid & 7u) * 4u;
            signed char* input_dst = input_tile + local_seq * 32u + input_off;
            if (local_seq < group_len) {
                const unsigned token = token_ids[slot_start + local_seq];
                const unsigned chunk = block * 8u + sub;
                const signed char* input_src = input_qs
                    + token * blocks_per_row * 256u + chunk * 32u + input_off;
                *reinterpret_cast<unsigned*>(input_dst) =
                    *reinterpret_cast<const unsigned*>(input_src);
                if (input_off == 0u) {
                    activation_d[local_seq] =
                        input_ds[token * blocks_per_row * 8u + chunk];
                }
            } else {
                *reinterpret_cast<unsigned*>(input_dst) = 0u;
                if (input_off == 0u) {
                    activation_d[local_seq] = 0.0f;
                }
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int gate_a0 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int gate_a1 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int gate_a2 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int gate_a3 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const int up_a0 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int up_a1 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int up_a2 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int up_a3 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const unsigned b_seq = lane >> 2;
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 =
                *reinterpret_cast<const int*>(&input_tile[b_seq * 32u + b_col_lo]);
            const int b1 =
                *reinterpret_cast<const int*>(&input_tile[b_seq * 32u + b_col_hi]);

            int gate_dot0 = 0, gate_dot1 = 0, gate_dot2 = 0, gate_dot3 = 0;
            int up_dot0 = 0, up_dot1 = 0, up_dot2 = 0, up_dot3 = 0;
            rnb_mma_m16n8k32_s8(
                gate_dot0, gate_dot1, gate_dot2, gate_dot3,
                gate_a0, gate_a1, gate_a2, gate_a3, b0, b1, 0, 0, 0, 0);
            rnb_mma_m16n8k32_s8(
                up_dot0, up_dot1, up_dot2, up_dot3,
                up_a0, up_a1, up_a2, up_a3, b0, b1, 0, 0, 0, 0);

            int sum_a = 0;
            int sum_b = 0;
#pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (local_seq_a < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_a * 32u + k]);
                    sum_a = __dp4a(0x01010101, value, sum_a);
                }
                if (local_seq_b < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_b * 32u + k]);
                    sum_b = __dp4a(0x01010101, value, sum_b);
                }
            }
            const float dy_a =
                local_seq_a < group_len ? activation_d[local_seq_a] : 0.0f;
            const float dy_b =
                local_seq_b < group_len ? activation_d[local_seq_b] : 0.0f;
            const unsigned local_row_a = warp_row_off + frag_row_a;
            const unsigned local_row_b = warp_row_off + frag_row_b;

#define RNB_Q4K_SELECTED_MMQ_ACCUM(ACC, DOT, DY, SUM, ROW, SC, MN, D, DMIN) \
            ACC += DY * (D[ROW] * (float)SC[ROW] * (float)DOT \
                - DMIN[ROW] * (float)MN[ROW] * (float)SUM)
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                gate_acc[0], gate_dot0, dy_a, sum_a, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                gate_acc[1], gate_dot1, dy_b, sum_b, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                gate_acc[2], gate_dot2, dy_a, sum_a, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                gate_acc[3], gate_dot3, dy_b, sum_b, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                up_acc[0], up_dot0, dy_a, sum_a, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                up_acc[1], up_dot1, dy_b, sum_b, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                up_acc[2], up_dot2, dy_a, sum_a, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ_ACCUM(
                up_acc[3], up_dot3, dy_b, sum_b, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
#undef RNB_Q4K_SELECTED_MMQ_ACCUM
            __syncthreads();
        }
    }

    if (row_a_valid && local_seq_a < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_a,
            rows, row_a, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[0], up_acc[0]));
    }
    if (row_a_valid && local_seq_b < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_b,
            rows, row_a, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[1], up_acc[1]));
    }
    if (row_b_valid && local_seq_a < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_a,
            rows, row_b, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[2], up_acc[2]));
    }
    if (row_b_valid && local_seq_b < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_b,
            rows, row_b, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[3], up_acc[3]));
    }
#endif
}



extern "C" __global__ void rnb_q4k_selected_gate_up_silu_q8dot_mmq_group16(
    float* __restrict__ output,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    const unsigned* __restrict__ pack_group_offsets,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned pack_blocks_per_row) {
#if __CUDA_ARCH__ < 750
    (void)output;
    (void)gate_weights;
    (void)up_weights;
    (void)input_qs;
    (void)input_ds;
    (void)token_ids;
    (void)group_meta;
    (void)pack_group_offsets;
    (void)rows;
    (void)blocks_per_row;
    (void)pack_blocks_per_row;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 32u;
    const unsigned group = blockIdx.y;
    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (warp >= 2u || group_len == 0u || group_len > 16u) {
        return;
    }

    __shared__ signed char gate_tile[32 * 32];
    __shared__ signed char up_tile[32 * 32];
    __shared__ signed char input_tile[16 * 32];
    __shared__ float gate_d[32];
    __shared__ float gate_dmin[32];
    __shared__ float up_d[32];
    __shared__ float up_dmin[32];
    __shared__ unsigned char gate_sc[32];
    __shared__ unsigned char gate_mn[32];
    __shared__ unsigned char up_sc[32];
    __shared__ unsigned char up_mn[32];
    __shared__ float activation_d[16];

    const unsigned warp_row_off = warp * 16u;
    const unsigned frag_row_a = lane >> 2;
    const unsigned frag_row_b = frag_row_a + 8u;
    const unsigned local_seq_a = (lane & 3u) << 1;
    const unsigned local_seq_b = local_seq_a + 1u;
    const unsigned local_seq_c = local_seq_a + 8u;
    const unsigned local_seq_d = local_seq_b + 8u;
    const unsigned row_a = row_base + warp_row_off + frag_row_a;
    const unsigned row_b = row_base + warp_row_off + frag_row_b;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;
    const unsigned row_bytes = blocks_per_row * 144u;
    float gate_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        for (unsigned sub = 0; sub < 8u; ++sub) {
            if (tid < 32u) {
                const unsigned global_row = row_base + tid;
                if (global_row < rows) {
                    const unsigned char* gate_block =
                        gate_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned char* up_block =
                        up_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned gate_raw_d =
                        (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
                    const unsigned gate_raw_dmin =
                        (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
                    const unsigned up_raw_d =
                        (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
                    const unsigned up_raw_dmin =
                        (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
                    gate_d[tid] =
                        __half2float(__ushort_as_half((unsigned short)gate_raw_d));
                    gate_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
                    up_d[tid] = __half2float(__ushort_as_half((unsigned short)up_raw_d));
                    up_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
                    if (sub < 4u) {
                        gate_sc[tid] = gate_block[4u + sub] & 63u;
                        gate_mn[tid] = gate_block[8u + sub] & 63u;
                        up_sc[tid] = up_block[4u + sub] & 63u;
                        up_mn[tid] = up_block[8u + sub] & 63u;
                    } else {
                        gate_sc[tid] = (gate_block[8u + sub] & 0x0fu)
                            | ((gate_block[sub] >> 6) << 4);
                        gate_mn[tid] = (gate_block[8u + sub] >> 4)
                            | ((gate_block[4u + sub] >> 6) << 4);
                        up_sc[tid] = (up_block[8u + sub] & 0x0fu)
                            | ((up_block[sub] >> 6) << 4);
                        up_mn[tid] = (up_block[8u + sub] >> 4)
                            | ((up_block[4u + sub] >> 6) << 4);
                    }
                } else {
                    gate_d[tid] = 0.0f;
                    gate_dmin[tid] = 0.0f;
                    up_d[tid] = 0.0f;
                    up_dmin[tid] = 0.0f;
                    gate_sc[tid] = 0u;
                    gate_mn[tid] = 0u;
                    up_sc[tid] = 0u;
                    up_mn[tid] = 0u;
                }
            }

            for (unsigned load = tid; load < 256u; load += 64u) {
                const unsigned load_row = load >> 3;
                const unsigned load_off = (load & 7u) * 4u;
                const unsigned global_row = row_base + load_row;
                signed char* gate_dst = gate_tile + load_row * 32u + load_off;
                signed char* up_dst = up_tile + load_row * 32u + load_off;
                if (global_row < rows) {
                    const unsigned char* gate_block =
                        gate_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned char* up_block =
                        up_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned nibble_base = 16u + (sub >> 1) * 32u;
                    const unsigned gate_packed =
                        *reinterpret_cast<const unsigned*>(gate_block + nibble_base + load_off);
                    const unsigned up_packed =
                        *reinterpret_cast<const unsigned*>(up_block + nibble_base + load_off);
                    const unsigned shift = (sub & 1u) * 4u;
                    *reinterpret_cast<unsigned*>(gate_dst) =
                        (gate_packed >> shift) & 0x0f0f0f0fu;
                    *reinterpret_cast<unsigned*>(up_dst) =
                        (up_packed >> shift) & 0x0f0f0f0fu;
                } else {
                    *reinterpret_cast<unsigned*>(gate_dst) = 0u;
                    *reinterpret_cast<unsigned*>(up_dst) = 0u;
                }
            }

            for (unsigned load = tid; load < 128u; load += 64u) {
                const unsigned local_seq = load >> 3;
                const unsigned input_off = (load & 7u) * 4u;
                signed char* input_dst = input_tile + local_seq * 32u + input_off;
                if (local_seq < group_len) {
                    const unsigned token = token_ids[slot_start + local_seq];
                    const unsigned chunk = block * 8u + sub;
                    const signed char* input_src = input_qs
                        + token * blocks_per_row * 256u + chunk * 32u + input_off;
                    *reinterpret_cast<unsigned*>(input_dst) =
                        *reinterpret_cast<const unsigned*>(input_src);
                    if (input_off == 0u) {
                        activation_d[local_seq] =
                            input_ds[token * blocks_per_row * 8u + chunk];
                    }
                } else {
                    *reinterpret_cast<unsigned*>(input_dst) = 0u;
                    if (input_off == 0u) {
                        activation_d[local_seq] = 0.0f;
                    }
                }
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int gate_a0 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int gate_a1 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int gate_a2 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int gate_a3 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const int up_a0 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int up_a1 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int up_a2 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int up_a3 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const unsigned b_seq = lane >> 2;
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 =
                *reinterpret_cast<const int*>(&input_tile[b_seq * 32u + b_col_lo]);
            const int b1 =
                *reinterpret_cast<const int*>(&input_tile[b_seq * 32u + b_col_hi]);
            const int b2 =
                *reinterpret_cast<const int*>(&input_tile[(b_seq + 8u) * 32u + b_col_lo]);
            const int b3 =
                *reinterpret_cast<const int*>(&input_tile[(b_seq + 8u) * 32u + b_col_hi]);

            int gate_dot0 = 0, gate_dot1 = 0, gate_dot2 = 0, gate_dot3 = 0;
            int gate_dot4 = 0, gate_dot5 = 0, gate_dot6 = 0, gate_dot7 = 0;
            int up_dot0 = 0, up_dot1 = 0, up_dot2 = 0, up_dot3 = 0;
            int up_dot4 = 0, up_dot5 = 0, up_dot6 = 0, up_dot7 = 0;
            rnb_mma_m16n8k32_s8(
                gate_dot0, gate_dot1, gate_dot2, gate_dot3,
                gate_a0, gate_a1, gate_a2, gate_a3, b0, b1, 0, 0, 0, 0);
            rnb_mma_m16n8k32_s8(
                gate_dot4, gate_dot5, gate_dot6, gate_dot7,
                gate_a0, gate_a1, gate_a2, gate_a3, b2, b3, 0, 0, 0, 0);
            rnb_mma_m16n8k32_s8(
                up_dot0, up_dot1, up_dot2, up_dot3,
                up_a0, up_a1, up_a2, up_a3, b0, b1, 0, 0, 0, 0);
            rnb_mma_m16n8k32_s8(
                up_dot4, up_dot5, up_dot6, up_dot7,
                up_a0, up_a1, up_a2, up_a3, b2, b3, 0, 0, 0, 0);

            int sum_a = 0;
            int sum_b = 0;
            int sum_c = 0;
            int sum_d = 0;
#pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (local_seq_a < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_a * 32u + k]);
                    sum_a = __dp4a(0x01010101, value, sum_a);
                }
                if (local_seq_b < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_b * 32u + k]);
                    sum_b = __dp4a(0x01010101, value, sum_b);
                }
                if (local_seq_c < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_c * 32u + k]);
                    sum_c = __dp4a(0x01010101, value, sum_c);
                }
                if (local_seq_d < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_d * 32u + k]);
                    sum_d = __dp4a(0x01010101, value, sum_d);
                }
            }
            const float dy_a =
                local_seq_a < group_len ? activation_d[local_seq_a] : 0.0f;
            const float dy_b =
                local_seq_b < group_len ? activation_d[local_seq_b] : 0.0f;
            const float dy_c =
                local_seq_c < group_len ? activation_d[local_seq_c] : 0.0f;
            const float dy_d =
                local_seq_d < group_len ? activation_d[local_seq_d] : 0.0f;
            const unsigned local_row_a = warp_row_off + frag_row_a;
            const unsigned local_row_b = warp_row_off + frag_row_b;

#define RNB_Q4K_SELECTED_MMQ16_ACCUM(ACC, DOT, DY, SUM, ROW, SC, MN, D, DMIN) \
            ACC += DY * (D[ROW] * (float)SC[ROW] * (float)DOT \
                - DMIN[ROW] * (float)MN[ROW] * (float)SUM)
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[0], gate_dot0, dy_a, sum_a, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[1], gate_dot1, dy_b, sum_b, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[2], gate_dot2, dy_a, sum_a, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[3], gate_dot3, dy_b, sum_b, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[4], gate_dot4, dy_c, sum_c, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[5], gate_dot5, dy_d, sum_d, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[6], gate_dot6, dy_c, sum_c, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                gate_acc[7], gate_dot7, dy_d, sum_d, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[0], up_dot0, dy_a, sum_a, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[1], up_dot1, dy_b, sum_b, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[2], up_dot2, dy_a, sum_a, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[3], up_dot3, dy_b, sum_b, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[4], up_dot4, dy_c, sum_c, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[5], up_dot5, dy_d, sum_d, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[6], up_dot6, dy_c, sum_c, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ16_ACCUM(
                up_acc[7], up_dot7, dy_d, sum_d, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
#undef RNB_Q4K_SELECTED_MMQ16_ACCUM
            __syncthreads();
        }
    }

    if (row_a_valid && local_seq_a < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_a,
            rows, row_a, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[0], up_acc[0]));
    }
    if (row_a_valid && local_seq_b < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_b,
            rows, row_a, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[1], up_acc[1]));
    }
    if (row_b_valid && local_seq_a < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_a,
            rows, row_b, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[2], up_acc[2]));
    }
    if (row_b_valid && local_seq_b < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_b,
            rows, row_b, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[3], up_acc[3]));
    }
    if (row_a_valid && local_seq_c < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_c,
            rows, row_a, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[4], up_acc[4]));
    }
    if (row_a_valid && local_seq_d < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_d,
            rows, row_a, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[5], up_acc[5]));
    }
    if (row_b_valid && local_seq_c < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_c,
            rows, row_b, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[6], up_acc[6]));
    }
    if (row_b_valid && local_seq_d < group_len) {
        rnb_q4k_selected_q8dot_store(
            output, pack_group_offsets, group, slot_start, local_seq_d,
            rows, row_b, pack_blocks_per_row,
            rnb_qwen_silu_mul_f32(gate_acc[7], up_acc[7]));
    }
#endif
}

extern "C" __global__ void rnb_q4k_selected_gate_up_silu_q8dot_mmq_group32(
    float* __restrict__ output,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    const unsigned* __restrict__ pack_group_offsets,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned pack_blocks_per_row,
    signed char* __restrict__ output_qs,
    float* __restrict__ output_ds) {
#if __CUDA_ARCH__ < 750
    (void)output;
    (void)gate_weights;
    (void)up_weights;
    (void)input_qs;
    (void)input_ds;
    (void)token_ids;
    (void)group_meta;
    (void)pack_group_offsets;
    (void)rows;
    (void)blocks_per_row;
    (void)pack_blocks_per_row;
    (void)output_qs;
    (void)output_ds;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 32u;
    const unsigned group = blockIdx.y;
    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 32u) {
        return;
    }

    __shared__ signed char gate_tile[32 * 32];
    __shared__ signed char up_tile[32 * 32];
    __shared__ signed char input_tile[32 * 32];
    __shared__ float gate_d[32];
    __shared__ float gate_dmin[32];
    __shared__ float up_d[32];
    __shared__ float up_dmin[32];
    __shared__ unsigned char gate_sc[32];
    __shared__ unsigned char gate_mn[32];
    __shared__ unsigned char up_sc[32];
    __shared__ unsigned char up_mn[32];
    __shared__ float activation_d[32];
    __shared__ float output_tile[32 * 32];

    const unsigned warp_row_off = (warp & 1u) * 16u;
    const unsigned warp_seq_off = (warp >> 1) * 8u;
    const unsigned frag_row_a = lane >> 2;
    const unsigned frag_row_b = frag_row_a + 8u;
    const unsigned local_seq_a = warp_seq_off + ((lane & 3u) << 1);
    const unsigned local_seq_b = local_seq_a + 1u;
    const unsigned row_a = row_base + warp_row_off + frag_row_a;
    const unsigned row_b = row_base + warp_row_off + frag_row_b;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;
    const unsigned row_bytes = blocks_per_row * 144u;
    float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        for (unsigned sub = 0; sub < 8u; ++sub) {
            if (tid < 32u) {
                const unsigned global_row = row_base + tid;
                if (global_row < rows) {
                    const unsigned char* gate_block =
                        gate_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned char* up_block =
                        up_weights[slot_start] + global_row * row_bytes + block * 144u;
                    const unsigned gate_raw_d =
                        (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
                    const unsigned gate_raw_dmin =
                        (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
                    const unsigned up_raw_d =
                        (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
                    const unsigned up_raw_dmin =
                        (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
                    gate_d[tid] =
                        __half2float(__ushort_as_half((unsigned short)gate_raw_d));
                    gate_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));
                    up_d[tid] = __half2float(__ushort_as_half((unsigned short)up_raw_d));
                    up_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)up_raw_dmin));
                    if (sub < 4u) {
                        gate_sc[tid] = gate_block[4u + sub] & 63u;
                        gate_mn[tid] = gate_block[8u + sub] & 63u;
                        up_sc[tid] = up_block[4u + sub] & 63u;
                        up_mn[tid] = up_block[8u + sub] & 63u;
                    } else {
                        gate_sc[tid] = (gate_block[8u + sub] & 0x0fu)
                            | ((gate_block[sub] >> 6) << 4);
                        gate_mn[tid] = (gate_block[8u + sub] >> 4)
                            | ((gate_block[4u + sub] >> 6) << 4);
                        up_sc[tid] = (up_block[8u + sub] & 0x0fu)
                            | ((up_block[sub] >> 6) << 4);
                        up_mn[tid] = (up_block[8u + sub] >> 4)
                            | ((up_block[4u + sub] >> 6) << 4);
                    }
                } else {
                    gate_d[tid] = 0.0f;
                    gate_dmin[tid] = 0.0f;
                    up_d[tid] = 0.0f;
                    up_dmin[tid] = 0.0f;
                    gate_sc[tid] = 0u;
                    gate_mn[tid] = 0u;
                    up_sc[tid] = 0u;
                    up_mn[tid] = 0u;
                }
            }

            const unsigned load_row = tid >> 3;
            const unsigned load_off = (tid & 7u) * 4u;
            const unsigned global_row = row_base + load_row;
            signed char* gate_dst = gate_tile + load_row * 32u + load_off;
            signed char* up_dst = up_tile + load_row * 32u + load_off;
            if (global_row < rows) {
                const unsigned char* gate_block =
                    gate_weights[slot_start] + global_row * row_bytes + block * 144u;
                const unsigned char* up_block =
                    up_weights[slot_start] + global_row * row_bytes + block * 144u;
                const unsigned nibble_base = 16u + (sub >> 1) * 32u;
                const unsigned gate_packed =
                    *reinterpret_cast<const unsigned*>(gate_block + nibble_base + load_off);
                const unsigned up_packed =
                    *reinterpret_cast<const unsigned*>(up_block + nibble_base + load_off);
                const unsigned shift = (sub & 1u) * 4u;
                *reinterpret_cast<unsigned*>(gate_dst) =
                    (gate_packed >> shift) & 0x0f0f0f0fu;
                *reinterpret_cast<unsigned*>(up_dst) =
                    (up_packed >> shift) & 0x0f0f0f0fu;
            } else {
                *reinterpret_cast<unsigned*>(gate_dst) = 0u;
                *reinterpret_cast<unsigned*>(up_dst) = 0u;
            }

            const unsigned local_seq = tid >> 3;
            const unsigned input_off = (tid & 7u) * 4u;
            signed char* input_dst = input_tile + local_seq * 32u + input_off;
            if (local_seq < group_len) {
                const unsigned token = token_ids[slot_start + local_seq];
                const unsigned chunk = block * 8u + sub;
                const signed char* input_src =
                    input_qs + token * blocks_per_row * 256u + chunk * 32u + input_off;
                *reinterpret_cast<unsigned*>(input_dst) =
                    *reinterpret_cast<const unsigned*>(input_src);
                if (input_off == 0u) {
                    activation_d[local_seq] =
                        input_ds[token * blocks_per_row * 8u + chunk];
                }
            } else {
                *reinterpret_cast<unsigned*>(input_dst) = 0u;
                if (input_off == 0u) {
                    activation_d[local_seq] = 0.0f;
                }
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int gate_a0 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int gate_a1 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int gate_a2 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int gate_a3 = *reinterpret_cast<const int*>(
                &gate_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const int up_a0 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int up_a1 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int up_a2 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int up_a3 = *reinterpret_cast<const int*>(
                &up_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const unsigned b_seq = warp_seq_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 =
                *reinterpret_cast<const int*>(&input_tile[b_seq * 32u + b_col_lo]);
            const int b1 =
                *reinterpret_cast<const int*>(&input_tile[b_seq * 32u + b_col_hi]);

            int gate_dot0 = 0, gate_dot1 = 0, gate_dot2 = 0, gate_dot3 = 0;
            int up_dot0 = 0, up_dot1 = 0, up_dot2 = 0, up_dot3 = 0;
            rnb_mma_m16n8k32_s8(
                gate_dot0, gate_dot1, gate_dot2, gate_dot3,
                gate_a0, gate_a1, gate_a2, gate_a3, b0, b1, 0, 0, 0, 0);
            rnb_mma_m16n8k32_s8(
                up_dot0, up_dot1, up_dot2, up_dot3,
                up_a0, up_a1, up_a2, up_a3, b0, b1, 0, 0, 0, 0);

            int sum_a = 0;
            int sum_b = 0;
#pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (local_seq_a < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_a * 32u + k]);
                    sum_a = __dp4a(0x01010101, value, sum_a);
                }
                if (local_seq_b < group_len) {
                    const int value =
                        *reinterpret_cast<const int*>(&input_tile[local_seq_b * 32u + k]);
                    sum_b = __dp4a(0x01010101, value, sum_b);
                }
            }
            const float dy_a =
                local_seq_a < group_len ? activation_d[local_seq_a] : 0.0f;
            const float dy_b =
                local_seq_b < group_len ? activation_d[local_seq_b] : 0.0f;
            const unsigned local_row_a = warp_row_off + frag_row_a;
            const unsigned local_row_b = warp_row_off + frag_row_b;

#define RNB_Q4K_SELECTED_MMQ32_ACCUM(ACC, DOT, DY, SUM, ROW, SC, MN, D, DMIN) \
            ACC += DY * (D[ROW] * (float)SC[ROW] * (float)DOT \
                - DMIN[ROW] * (float)MN[ROW] * (float)SUM)
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                gate_acc[0], gate_dot0, dy_a, sum_a, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                gate_acc[1], gate_dot1, dy_b, sum_b, local_row_a,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                gate_acc[2], gate_dot2, dy_a, sum_a, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                gate_acc[3], gate_dot3, dy_b, sum_b, local_row_b,
                gate_sc, gate_mn, gate_d, gate_dmin);
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                up_acc[0], up_dot0, dy_a, sum_a, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                up_acc[1], up_dot1, dy_b, sum_b, local_row_a,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                up_acc[2], up_dot2, dy_a, sum_a, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
            RNB_Q4K_SELECTED_MMQ32_ACCUM(
                up_acc[3], up_dot3, dy_b, sum_b, local_row_b,
                up_sc, up_mn, up_d, up_dmin);
#undef RNB_Q4K_SELECTED_MMQ32_ACCUM
            __syncthreads();
        }
    }

    const bool q8_output = output_qs != nullptr && output_ds != nullptr;
    const unsigned local_row_a = warp_row_off + frag_row_a;
    const unsigned local_row_b = warp_row_off + frag_row_b;
#define RNB_Q4K_SELECTED_MMQ32_STORE(VALID, LOCAL_SEQ, ROW, LOCAL_ROW, VALUE) \
    if ((VALID) && (LOCAL_SEQ) < group_len) { \
        if (q8_output) { \
            output_tile[(LOCAL_SEQ) * 32u + (LOCAL_ROW)] = (VALUE); \
        } else { \
            rnb_q4k_selected_q8dot_store( \
                output, pack_group_offsets, group, slot_start, (LOCAL_SEQ), \
                rows, (ROW), pack_blocks_per_row, (VALUE)); \
        } \
    }
    RNB_Q4K_SELECTED_MMQ32_STORE(
        row_a_valid, local_seq_a, row_a, local_row_a,
        rnb_qwen_silu_mul_f32(gate_acc[0], up_acc[0]));
    RNB_Q4K_SELECTED_MMQ32_STORE(
        row_a_valid, local_seq_b, row_a, local_row_a,
        rnb_qwen_silu_mul_f32(gate_acc[1], up_acc[1]));
    RNB_Q4K_SELECTED_MMQ32_STORE(
        row_b_valid, local_seq_a, row_b, local_row_b,
        rnb_qwen_silu_mul_f32(gate_acc[2], up_acc[2]));
    RNB_Q4K_SELECTED_MMQ32_STORE(
        row_b_valid, local_seq_b, row_b, local_row_b,
        rnb_qwen_silu_mul_f32(gate_acc[3], up_acc[3]));
#undef RNB_Q4K_SELECTED_MMQ32_STORE
    __syncthreads();

    if (q8_output) {
        const unsigned q8_blocks_per_slot = rows / 32u;
        for (unsigned local_seq = warp; local_seq < group_len; local_seq += 8u) {
            const unsigned row = row_base + lane;
            const float value = row < rows ? output_tile[local_seq * 32u + lane] : 0.0f;
            float max_abs = fabsf(value);
            for (unsigned stride = 16u; stride > 0u; stride >>= 1u) {
                const float other = __shfl_down_sync(0xffffffffu, max_abs, stride);
                if (lane < stride && other > max_abs) {
                    max_abs = other;
                }
            }
            const float block_max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
            const float d = block_max_abs > 0.0f ? block_max_abs / 127.0f : 0.0f;
            const unsigned slot = slot_start + local_seq;
            if (lane == 0u) {
                output_ds[slot * q8_blocks_per_slot + blockIdx.x] = d;
            }
            if (row < rows) {
                int q = 0;
                if (d > 0.0f) {
                    q = (int)nearbyintf(value / d);
                    q = q < -127 ? -127 : (q > 127 ? 127 : q);
                }
                output_qs[slot * rows + row] = (signed char)q;
            }
        }
    }
#endif
}

extern "C" __global__ void rnb_q4k_selected_gate_up_gemv_by_token_group16_warp4(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const unsigned char* const* __restrict__ gate_weights,
    const unsigned char* const* __restrict__ up_weights,
    const float* __restrict__ input,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 16u) {
        return;
    }

    float gate_acc[16];
    float up_acc[16];
    #pragma unroll
    for (unsigned i = 0; i < 16u; ++i) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }

    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned base_x = b * 256u;
        const unsigned char* gate_block = gate_row + b * 144u;
        const unsigned gate_raw_d = (unsigned)gate_block[0] | ((unsigned)gate_block[1] << 8);
        const unsigned gate_raw_dmin = (unsigned)gate_block[2] | ((unsigned)gate_block[3] << 8);
        const float gate_d = __half2float(__ushort_as_half((unsigned short)gate_raw_d));
        const float gate_dmin = __half2float(__ushort_as_half((unsigned short)gate_raw_dmin));

        const unsigned char* up_block = up_row + b * 144u;
        const unsigned up_raw_d = (unsigned)up_block[0] | ((unsigned)up_block[1] << 8);
        const unsigned up_raw_dmin = (unsigned)up_block[2] | ((unsigned)up_block[3] << 8);
        const float up_d = __half2float(__ushort_as_half((unsigned short)up_raw_d));
        const float up_dmin = __half2float(__ushort_as_half((unsigned short)up_raw_dmin));

        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned gate_sc;
            unsigned gate_mn;
            if (j < 4u) {
                gate_sc = gate_block[4 + j] & 63u;
                gate_mn = gate_block[4 + j + 4] & 63u;
            } else {
                gate_sc = (gate_block[4 + j + 4] & 0x0fu) | ((gate_block[4 + j - 4] >> 6) << 4);
                gate_mn = (gate_block[4 + j + 4] >> 4) | ((gate_block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned gate_q = gate_block[16 + q_index];
            gate_q = local < 32u ? (gate_q & 0x0fu) : (gate_q >> 4);
            const float gate_y = (gate_d * (float)gate_sc) * (float)gate_q - gate_dmin * (float)gate_mn;

            unsigned up_sc;
            unsigned up_mn;
            if (j < 4u) {
                up_sc = up_block[4 + j] & 63u;
                up_mn = up_block[4 + j + 4] & 63u;
            } else {
                up_sc = (up_block[4 + j + 4] & 0x0fu) | ((up_block[4 + j - 4] >> 6) << 4);
                up_mn = (up_block[4 + j + 4] >> 4) | ((up_block[4 + j] >> 6) << 4);
            }
            unsigned up_q = up_block[16 + q_index];
            up_q = local < 32u ? (up_q & 0x0fu) : (up_q >> 4);
            const float up_y = (up_d * (float)up_sc) * (float)up_q - up_dmin * (float)up_mn;

            const unsigned x_off = base_x + tid;
            #pragma unroll
            for (unsigned i = 0; i < 16u; ++i) {
                if (i < group_len) {
                    const float* input_i = input + token_ids[slot_start + i] * blocks_per_row * 256u;
                    gate_acc[i] += gate_y * input_i[x_off];
                    up_acc[i] += up_y * input_i[x_off];
                }
            }
        }
    }

    #pragma unroll
    for (unsigned i = 0; i < 16u; ++i) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc[i] += __shfl_down_sync(0xffffffffu, gate_acc[i], offset);
            up_acc[i] += __shfl_down_sync(0xffffffffu, up_acc[i], offset);
        }
    }
    if (lane == 0u) {
        const unsigned out0 = slot_start * rows + row;
        #pragma unroll
        for (unsigned i = 0; i < 16u; ++i) {
            if (i < group_len) {
                gate_out[out0 + i * rows] = gate_acc[i];
                up_out[out0 + i * rows] = up_acc[i];
            }
        }
    }
}
