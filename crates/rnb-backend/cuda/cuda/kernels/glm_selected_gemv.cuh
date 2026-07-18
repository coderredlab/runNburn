#pragma once

#include "iq_tables.cuh"

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

extern "C" __global__ void rnb_iq2_xxs_selected_gate_up_gemv_by_token_group4_warp4(
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
    const float* input0 = input + token_ids[slot_start] * blocks_per_row * 256u;
    const float* input1 = group_len > 1u
        ? input + token_ids[slot_start + 1u] * blocks_per_row * 256u
        : nullptr;
    const float* input2 = group_len > 2u
        ? input + token_ids[slot_start + 2u] * blocks_per_row * 256u
        : nullptr;
    const float* input3 = group_len > 3u
        ? input + token_ids[slot_start + 3u] * blocks_per_row * 256u
        : nullptr;

    float gate_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float gate_acc2 = 0.0f;
    float gate_acc3 = 0.0f;
    float up_acc0 = 0.0f;
    float up_acc1 = 0.0f;
    float up_acc2 = 0.0f;
    float up_acc3 = 0.0f;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_base = b * 256u;
        const unsigned char* gate_block = gate_row + b * 66u;
        const unsigned char* up_block = up_row + b * 66u;
        for (unsigned index = lane; index < 256u; index += 32u) {
            const float gate_weight = rnb_iq2_xxs_value(gate_block, index);
            const float up_weight = rnb_iq2_xxs_value(up_block, index);
            const unsigned input_index = input_base + index;
            gate_acc0 += gate_weight * input0[input_index];
            up_acc0 += up_weight * input0[input_index];
            if (group_len > 1u) {
                gate_acc1 += gate_weight * input1[input_index];
                up_acc1 += up_weight * input1[input_index];
            }
            if (group_len > 2u) {
                gate_acc2 += gate_weight * input2[input_index];
                up_acc2 += up_weight * input2[input_index];
            }
            if (group_len > 3u) {
                gate_acc3 += gate_weight * input3[input_index];
                up_acc3 += up_weight * input3[input_index];
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
        const unsigned output_start = slot_start * rows + row;
        gate_out[output_start] = gate_acc0;
        up_out[output_start] = up_acc0;
        if (group_len > 1u) {
            gate_out[output_start + rows] = gate_acc1;
            up_out[output_start + rows] = up_acc1;
        }
        if (group_len > 2u) {
            gate_out[output_start + 2u * rows] = gate_acc2;
            up_out[output_start + 2u * rows] = up_acc2;
        }
        if (group_len > 3u) {
            gate_out[output_start + 3u * rows] = gate_acc3;
            up_out[output_start + 3u * rows] = up_acc3;
        }
    }
}

extern "C" __global__ void rnb_iq3_xxs_selected_down_accum_by_token_group4_warp4(
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
    const float* input0 = activation + slot_start * blocks_per_row * 256u;
    const float* input1 = group_len > 1u
        ? activation + (slot_start + 1u) * blocks_per_row * 256u
        : nullptr;
    const float* input2 = group_len > 2u
        ? activation + (slot_start + 2u) * blocks_per_row * 256u
        : nullptr;
    const float* input3 = group_len > 3u
        ? activation + (slot_start + 3u) * blocks_per_row * 256u
        : nullptr;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned input_base = b * 256u;
        const unsigned char* block = row_ptr + b * 98u;
        for (unsigned index = lane; index < 256u; index += 32u) {
            const float weight = rnb_iq3_xxs_value(block, index);
            const unsigned input_index = input_base + index;
            acc0 += weight * input0[input_index];
            if (group_len > 1u) {
                acc1 += weight * input1[input_index];
            }
            if (group_len > 2u) {
                acc2 += weight * input2[input_index];
            }
            if (group_len > 3u) {
                acc3 += weight * input3[input_index];
            }
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
            atomicAdd(
                out + token_ids[slot_start + 1u] * rows + row,
                acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(
                out + token_ids[slot_start + 2u] * rows + row,
                acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(
                out + token_ids[slot_start + 3u] * rows + row,
                acc3 * route[slot_start + 3u]);
        }
    }
}
