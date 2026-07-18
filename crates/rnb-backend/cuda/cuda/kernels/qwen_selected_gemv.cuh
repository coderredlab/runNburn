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
