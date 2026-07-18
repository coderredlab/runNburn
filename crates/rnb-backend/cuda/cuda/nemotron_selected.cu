#include <cuda_fp16.h>

__device__ __forceinline__ float rnb_q5_0_basic_value_at(
    const unsigned char* __restrict__ block,
    unsigned lane) {
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const unsigned qh = (unsigned)block[2] | ((unsigned)block[3] << 8) |
                        ((unsigned)block[4] << 16) | ((unsigned)block[5] << 24);
    const unsigned byte = block[6u + (lane & 15u)];
    const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
    const unsigned high = (qh >> lane) & 1u;
    return ((float)(low | (high << 4)) - 16.0f) * d;
}

__device__ __forceinline__ float rnb_q5_1_basic_value_at(
    const unsigned char* __restrict__ block,
    unsigned lane) {
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
    const unsigned qh = (unsigned)block[4] | ((unsigned)block[5] << 8) |
                        ((unsigned)block[6] << 16) | ((unsigned)block[7] << 24);
    const unsigned byte = block[8u + (lane & 15u)];
    const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
    const unsigned high = (qh >> lane) & 1u;
    return (float)(low | (high << 4)) * d + m;
}

__device__ __forceinline__ float rnb_q8_0_basic_value_at(
    const unsigned char* __restrict__ block,
    unsigned lane) {
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const signed char q = (signed char)block[2u + lane];
    return (float)q * d;
}

extern "C" __global__ void rnb_q5_0_selected_relu_sqr_by_token_group4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
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

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 22u;
    const unsigned cols = blocks_per_row * 32u;
    const unsigned char* row_ptr = weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_idx = b * 32u + lane;
        const float y = rnb_q5_0_basic_value_at(row_ptr + b * 22u, lane);
        acc0 += y * input[token_ids[slot_start] * cols + input_idx];
        if (group_len > 1u) {
            acc1 += y * input[token_ids[slot_start + 1u] * cols + input_idx];
        }
        if (group_len > 2u) {
            acc2 += y * input[token_ids[slot_start + 2u] * cols + input_idx];
        }
        if (group_len > 3u) {
            acc3 += y * input[token_ids[slot_start + 3u] * cols + input_idx];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (lane == 0u) {
        const float v0 = acc0;
        out[slot_start * rows + row] = v0 > 0.0f ? v0 * v0 : 0.0f;
        if (group_len > 1u) {
            const float v1 = acc1;
            out[(slot_start + 1u) * rows + row] = v1 > 0.0f ? v1 * v1 : 0.0f;
        }
        if (group_len > 2u) {
            const float v2 = acc2;
            out[(slot_start + 2u) * rows + row] = v2 > 0.0f ? v2 * v2 : 0.0f;
        }
        if (group_len > 3u) {
            const float v3 = acc3;
            out[(slot_start + 3u) * rows + row] = v3 > 0.0f ? v3 * v3 : 0.0f;
        }
    }
}

extern "C" __global__ void rnb_q5_1_selected_down_accum_by_token_group4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
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

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 24u;
    const unsigned cols = blocks_per_row * 32u;
    const unsigned char* row_ptr = weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_idx = b * 32u + lane;
        const float y = rnb_q5_1_basic_value_at(row_ptr + b * 24u, lane);
        acc0 += y * input[slot_start * cols + input_idx];
        if (group_len > 1u) {
            acc1 += y * input[(slot_start + 1u) * cols + input_idx];
        }
        if (group_len > 2u) {
            acc2 += y * input[(slot_start + 2u) * cols + input_idx];
        }
        if (group_len > 3u) {
            acc3 += y * input[(slot_start + 3u) * cols + input_idx];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q8_0_selected_down_accum_by_token_group4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
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

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 34u;
    const unsigned cols = blocks_per_row * 32u;
    const unsigned char* row_ptr = weights[slot_start] + row * row_bytes;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_idx = b * 32u + lane;
        const float y = rnb_q8_0_basic_value_at(row_ptr + b * 34u, lane);
        acc0 += y * input[slot_start * cols + input_idx];
        if (group_len > 1u) {
            acc1 += y * input[(slot_start + 1u) * cols + input_idx];
        }
        if (group_len > 2u) {
            acc2 += y * input[(slot_start + 2u) * cols + input_idx];
        }
        if (group_len > 3u) {
            acc3 += y * input[(slot_start + 3u) * cols + input_idx];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}
