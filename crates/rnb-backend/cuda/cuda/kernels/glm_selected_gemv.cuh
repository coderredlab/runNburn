#pragma once

#include "iq_tables.cuh"
#include "iq2s_table.cuh"

__device__ __forceinline__ unsigned rnb_iq_load_u16(const unsigned char* ptr) {
    return (unsigned)ptr[0] | ((unsigned)ptr[1] << 8);
}

__device__ __forceinline__ unsigned rnb_iq_load_u32(const unsigned char* ptr) {
    return (unsigned)ptr[0]
        | ((unsigned)ptr[1] << 8)
        | ((unsigned)ptr[2] << 16)
        | ((unsigned)ptr[3] << 24);
}

__device__ __forceinline__ float rnb_iq2_xxs_value(
    const unsigned char* block,
    unsigned index) {
    const unsigned ib32 = index >> 5;
    const unsigned local = index & 31u;
    const unsigned group = local >> 3;
    const unsigned lane = local & 7u;
    const unsigned char* packed = block + 2u + ib32 * 8u;
    const unsigned scales_and_signs = rnb_iq_load_u32(packed + 4u);
    const unsigned sign_bits = rnb_ksigns_iq2xs[
        (scales_and_signs >> (7u * group)) & 127u];
    const unsigned long long grid = rnb_iq2xxs_grid[packed[group]];
    const float d = __half2float(__ushort_as_half((unsigned short)rnb_iq_load_u16(block)));
    const float scale = d * (0.5f + (float)(scales_and_signs >> 28)) * 0.25f;
    const float magnitude = (float)((grid >> (8u * lane)) & 0xffu);
    return (sign_bits & (1u << lane)) != 0u ? -scale * magnitude : scale * magnitude;
}

__device__ __forceinline__ float rnb_iq2_s_value(
    const unsigned char* block,
    unsigned index) {
    const unsigned ib32 = index >> 5;
    const unsigned local = index & 31u;
    const unsigned group = local >> 3;
    const unsigned lane = local & 7u;
    const unsigned qh = block[66u + ib32];
    const unsigned scale_byte = block[74u + ib32];
    const unsigned scale_nibble = group < 2u
        ? scale_byte & 0x0fu
        : scale_byte >> 4u;
    const unsigned high = (qh << (8u - 2u * group)) & 0x300u;
    const unsigned grid_index = (unsigned)block[2u + ib32 * 4u + group] | high;
    const unsigned long long grid = rnb_iq2s_grid[grid_index];
    const unsigned sign_bits = block[34u + ib32 * 4u + group];
    const float d = __half2float(__ushort_as_half((unsigned short)rnb_iq_load_u16(block)));
    const float scale = d * (0.5f + (float)scale_nibble) * 0.25f;
    const float magnitude = (float)((grid >> (8u * lane)) & 0xffu);
    return (sign_bits & (1u << lane)) != 0u ? -scale * magnitude : scale * magnitude;
}

__device__ __forceinline__ float rnb_glm_iq4_xs_value(
    const unsigned char* block,
    unsigned index) {
    const unsigned ib = index >> 5;
    const unsigned local = index & 31u;
    const unsigned scales_h = rnb_iq_load_u16(block + 2u);
    const unsigned low = (block[4u + (ib >> 1)] >> (4u * (ib & 1u))) & 0x0fu;
    const unsigned high = ((scales_h >> (2u * ib)) & 0x03u) << 4u;
    const unsigned q_byte = block[8u + ib * 16u + (local & 15u)];
    const unsigned q = local < 16u ? q_byte & 0x0fu : q_byte >> 4u;
    const float d = __half2float(__ushort_as_half((unsigned short)rnb_iq_load_u16(block)));
    return d * ((float)(low | high) - 32.0f) * rnb_iq4nl_value(q);
}


__device__ __forceinline__ float rnb_iq3_xxs_value(
    const unsigned char* block,
    unsigned index) {
    const unsigned ib32 = index >> 5;
    const unsigned local = index & 31u;
    const unsigned group = local >> 3;
    const unsigned lane = local & 7u;
    const unsigned packed = rnb_iq_load_u32(block + 66u + ib32 * 4u);
    const unsigned sign_bits = rnb_ksigns_iq2xs[(packed >> (7u * group)) & 127u];
    const unsigned grid_index = block[2u + ib32 * 8u + group * 2u + (lane >> 2)];
    const unsigned grid = rnb_iq3xxs_grid[grid_index];
    const float d = __half2float(__ushort_as_half((unsigned short)rnb_iq_load_u16(block)));
    const float scale = d * (0.5f + (float)(packed >> 28)) * 0.5f;
    const float magnitude = (float)((grid >> (8u * (lane & 3u))) & 0xffu);
    return (sign_bits & (1u << lane)) != 0u ? -scale * magnitude : scale * magnitude;
}

extern "C" __global__ void rnb_iq2_xxs_selected_gate_up_gemv(
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
    if (row >= rows || tid >= 256u) {
        return;
    }

    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 66u;
    const unsigned char* gate_row = gate_weights[expert] + row * row_bytes;
    const unsigned char* up_row = up_weights[expert] + row * row_bytes;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const float x = input[b * 256u + tid];
        gate_acc += rnb_iq2_xxs_value(gate_row + b * 66u, tid) * x;
        up_acc += rnb_iq2_xxs_value(up_row + b * 66u, tid) * x;
    }

    partial_gate[tid] = gate_acc;
    partial_up[tid] = up_acc;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_gate[tid] += partial_gate[tid + stride];
            partial_up[tid] += partial_up[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const unsigned out_idx = expert * rows + row;
        gate_out[out_idx] = partial_gate[0];
        up_out[out_idx] = partial_up[0];
    }
}


extern "C" __global__ void rnb_iq2_xxs_selected_gate_up_gemv_by_token(
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
    if (row >= rows || tid >= 256u) {
        return;
    }

    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned token = token_ids[slot];
    const float* token_input = input + token * blocks_per_row * 256u;
    const unsigned row_bytes = blocks_per_row * 66u;
    const unsigned char* gate_row = gate_weights[slot] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot] + row * row_bytes;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const float x = token_input[b * 256u + tid];
        gate_acc += rnb_iq2_xxs_value(gate_row + b * 66u, tid) * x;
        up_acc += rnb_iq2_xxs_value(up_row + b * 66u, tid) * x;
    }

    partial_gate[tid] = gate_acc;
    partial_up[tid] = up_acc;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_gate[tid] += partial_gate[tid + stride];
            partial_up[tid] += partial_up[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const unsigned out_idx = slot * rows + row;
        gate_out[out_idx] = partial_gate[0];
        up_out[out_idx] = partial_up[0];
    }
}

extern "C" __global__ void rnb_iq3_xxs_selected_down_silu_rowreduce(
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
    if (row >= rows || tid >= 256u) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 98u;
    for (unsigned expert = 0; expert < selected; ++expert) {
        const unsigned char* row_ptr = weights[expert] + row * row_bytes;
        const float* gate_input = gate + expert * blocks_per_row * 256u;
        const float* up_input = up + expert * blocks_per_row * 256u;
        float expert_acc = 0.0f;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const float g = gate_input[b * 256u + tid];
            const float activation = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += rnb_iq3_xxs_value(row_ptr + b * 98u, tid) * activation;
        }
        acc += expert_acc * route[expert];
    }

    partial[tid] = acc;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        out[row] = partial[0];
    }
}

extern "C" __global__ void rnb_iq3_xxs_selected_down_silu_rowreduce_by_token(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned slots_per_token,
    unsigned token_count,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned token = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || token >= token_count || tid >= 256u) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 98u;
    const unsigned first_slot = token * slots_per_token;
    for (unsigned local_slot = 0; local_slot < slots_per_token; ++local_slot) {
        const unsigned slot = first_slot + local_slot;
        const unsigned char* row_ptr = weights[slot] + row * row_bytes;
        const float* gate_input = gate + slot * blocks_per_row * 256u;
        const float* up_input = up + slot * blocks_per_row * 256u;
        float expert_acc = 0.0f;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const float g = gate_input[b * 256u + tid];
            const float activation = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += rnb_iq3_xxs_value(row_ptr + b * 98u, tid) * activation;
        }
        acc += expert_acc * route[slot];
    }

    partial[tid] = acc;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        out[token * rows + row] = partial[0];
    }
}

extern "C" __global__ void rnb_iq2_xxs_selected_gate_up_gemv_by_token_grouped_warp4(
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

    const unsigned slot_start = group_meta[group * 2u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 66u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;
    const float* input_rows[4];
#pragma unroll
    for (unsigned slot = 0; slot < 4u; ++slot) {
        input_rows[slot] = slot < group_len
            ? input + token_ids[slot_start + slot] * blocks_per_row * 256u
            : nullptr;
    }

    float gate_acc[4] = {};
    float up_acc[4] = {};
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_base = b * 256u;
        const unsigned char* gate_block = gate_row + b * 66u;
        const unsigned char* up_block = up_row + b * 66u;
        for (unsigned index = lane; index < 256u; index += 32u) {
            const float gate_weight = rnb_iq2_xxs_value(gate_block, index);
            const float up_weight = rnb_iq2_xxs_value(up_block, index);
            const unsigned input_index = input_base + index;
#pragma unroll
            for (unsigned slot = 0; slot < 4u; ++slot) {
                if (slot < group_len) {
                    gate_acc[slot] += gate_weight * input_rows[slot][input_index];
                    up_acc[slot] += up_weight * input_rows[slot][input_index];
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            gate_acc[slot] += __shfl_down_sync(0xffffffffu, gate_acc[slot], offset);
            up_acc[slot] += __shfl_down_sync(0xffffffffu, up_acc[slot], offset);
        }
    }
    if (lane == 0u) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            if (slot < group_len) {
                const unsigned output = (slot_start + slot) * rows + row;
                gate_out[output] = gate_acc[slot];
                up_out[output] = up_acc[slot];
            }
        }
    }
}


extern "C" __global__ void rnb_iq3_xxs_selected_down_accum_by_token_grouped_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ activation,
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

    const unsigned slot_start = group_meta[group * 2u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 98u;
    const unsigned char* row_ptr = weights[slot_start] + row * row_bytes;
    const float* input_rows[4];
#pragma unroll
    for (unsigned slot = 0; slot < 4u; ++slot) {
        input_rows[slot] = slot < group_len
            ? activation + (slot_start + slot) * blocks_per_row * 256u
            : nullptr;
    }

    float acc[4] = {};
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_base = b * 256u;
        const unsigned char* block = row_ptr + b * 98u;
        for (unsigned index = lane; index < 256u; index += 32u) {
            const float weight = rnb_iq3_xxs_value(block, index);
            const unsigned input_index = input_base + index;
#pragma unroll
            for (unsigned slot = 0; slot < 4u; ++slot) {
                if (slot < group_len) {
                    acc[slot] += weight * input_rows[slot][input_index];
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            acc[slot] += __shfl_down_sync(0xffffffffu, acc[slot], offset);
        }
    }
    if (lane == 0u) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            if (slot < group_len) {
                atomicAdd(
                    out + token_ids[slot_start + slot] * rows + row,
                    acc[slot] * route[slot_start + slot]);
            }
        }
    }
}

extern "C" __global__ void rnb_iq2_s_selected_gate_up_gemv_by_token(
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
    if (row >= rows || tid >= 256u) {
        return;
    }

    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const unsigned token = token_ids[slot];
    const float* token_input = input + token * blocks_per_row * 256u;
    const unsigned row_bytes = blocks_per_row * 82u;
    const unsigned char* gate_row = gate_weights[slot] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot] + row * row_bytes;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const float x = token_input[b * 256u + tid];
        gate_acc += rnb_iq2_s_value(gate_row + b * 82u, tid) * x;
        up_acc += rnb_iq2_s_value(up_row + b * 82u, tid) * x;
    }

    partial_gate[tid] = gate_acc;
    partial_up[tid] = up_acc;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_gate[tid] += partial_gate[tid + stride];
            partial_up[tid] += partial_up[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const unsigned out_idx = slot * rows + row;
        gate_out[out_idx] = partial_gate[0];
        up_out[out_idx] = partial_up[0];
    }
}

extern "C" __global__ void rnb_iq4_xs_selected_down_silu_rowreduce_by_token(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route,
    unsigned rows,
    unsigned slots_per_token,
    unsigned token_count,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned token = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || token >= token_count || tid >= 256u) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 136u;
    const unsigned first_slot = token * slots_per_token;
    for (unsigned local_slot = 0; local_slot < slots_per_token; ++local_slot) {
        const unsigned slot = first_slot + local_slot;
        const unsigned char* row_ptr = weights[slot] + row * row_bytes;
        const float* gate_input = gate + slot * blocks_per_row * 256u;
        const float* up_input = up + slot * blocks_per_row * 256u;
        float expert_acc = 0.0f;
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const float g = gate_input[b * 256u + tid];
            const float activation = (g / (1.0f + expf(-g))) * up_input[b * 256u + tid];
            expert_acc += rnb_glm_iq4_xs_value(row_ptr + b * 136u, tid) * activation;
        }
        acc += expert_acc * route[slot];
    }

    partial[tid] = acc;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        out[token * rows + row] = partial[0];
    }
}

extern "C" __global__ void rnb_iq2_s_selected_gate_up_gemv_by_token_grouped_warp4(
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

    const unsigned slot_start = group_meta[group * 2u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 82u;
    const unsigned char* gate_row = gate_weights[slot_start] + row * row_bytes;
    const unsigned char* up_row = up_weights[slot_start] + row * row_bytes;
    const float* input_rows[4];
#pragma unroll
    for (unsigned slot = 0; slot < 4u; ++slot) {
        input_rows[slot] = slot < group_len
            ? input + token_ids[slot_start + slot] * blocks_per_row * 256u
            : nullptr;
    }

    float gate_acc[4] = {};
    float up_acc[4] = {};
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_base = b * 256u;
        const unsigned char* gate_block = gate_row + b * 82u;
        const unsigned char* up_block = up_row + b * 82u;
        for (unsigned index = lane; index < 256u; index += 32u) {
            const float gate_weight = rnb_iq2_s_value(gate_block, index);
            const float up_weight = rnb_iq2_s_value(up_block, index);
            const unsigned input_index = input_base + index;
#pragma unroll
            for (unsigned slot = 0; slot < 4u; ++slot) {
                if (slot < group_len) {
                    gate_acc[slot] += gate_weight * input_rows[slot][input_index];
                    up_acc[slot] += up_weight * input_rows[slot][input_index];
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            gate_acc[slot] += __shfl_down_sync(0xffffffffu, gate_acc[slot], offset);
            up_acc[slot] += __shfl_down_sync(0xffffffffu, up_acc[slot], offset);
        }
    }
    if (lane == 0u) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            if (slot < group_len) {
                const unsigned output = (slot_start + slot) * rows + row;
                gate_out[output] = gate_acc[slot];
                up_out[output] = up_acc[slot];
            }
        }
    }
}

extern "C" __global__ void rnb_iq4_xs_selected_down_accum_by_token_grouped_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ activation,
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

    const unsigned slot_start = group_meta[group * 2u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    const unsigned row_bytes = blocks_per_row * 136u;
    const unsigned char* row_ptr = weights[slot_start] + row * row_bytes;
    const float* input_rows[4];
#pragma unroll
    for (unsigned slot = 0; slot < 4u; ++slot) {
        input_rows[slot] = slot < group_len
            ? activation + (slot_start + slot) * blocks_per_row * 256u
            : nullptr;
    }

    float acc[4] = {};
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_base = b * 256u;
        const unsigned char* block = row_ptr + b * 136u;
        for (unsigned index = lane; index < 256u; index += 32u) {
            const float weight = rnb_glm_iq4_xs_value(block, index);
            const unsigned input_index = input_base + index;
#pragma unroll
            for (unsigned slot = 0; slot < 4u; ++slot) {
                if (slot < group_len) {
                    acc[slot] += weight * input_rows[slot][input_index];
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            acc[slot] += __shfl_down_sync(0xffffffffu, acc[slot], offset);
        }
    }
    if (lane == 0u) {
#pragma unroll
        for (unsigned slot = 0; slot < 4u; ++slot) {
            if (slot < group_len) {
                atomicAdd(
                    out + token_ids[slot_start + slot] * rows + row,
                    acc[slot] * route[slot_start + slot]);
            }
        }
    }
}

struct RnbIq2XxsDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 66u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_iq2_xxs_value(block, index);
    }
};

struct RnbIq2SDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 82u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_iq2_s_value(block, index);
    }
};

struct RnbIq3XxsDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 98u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_iq3_xxs_value(block, index);
    }
};

extern "C" __global__ void rnb_iq2_xxs_gemv_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row) {
    rnb_quant_gemv_warp8_body<RnbIq2XxsDecoder>(
        out, weights, input, rows, blocks_per_row, 0u);
}

extern "C" __global__ void rnb_iq2_xxs_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbIq2XxsDecoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_iq2_s_gemv_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row) {
    rnb_quant_gemv_warp8_body<RnbIq2SDecoder>(
        out, weights, input, rows, blocks_per_row, 0u);
}

extern "C" __global__ void rnb_iq2_s_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbIq2SDecoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}

extern "C" __global__ void rnb_iq3_xxs_gemv_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row) {
    rnb_quant_gemv_warp8_body<RnbIq3XxsDecoder>(
        out, weights, input, rows, blocks_per_row, 0u);
}

extern "C" __global__ void rnb_iq3_xxs_gemv_batch_warp8(
    float* out, const unsigned char* weights, const float* input,
    unsigned rows, unsigned blocks_per_row, unsigned seq_len) {
    if (blockIdx.y < seq_len) {
        rnb_quant_gemv_warp8_body<RnbIq3XxsDecoder>(
            out, weights, input, rows, blocks_per_row, blockIdx.y);
    }
}
