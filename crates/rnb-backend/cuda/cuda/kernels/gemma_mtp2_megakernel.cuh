#pragma once

#include <cooperative_groups.h>
#include "q4k_block_dot.cuh"

namespace cg = cooperative_groups;

struct GemmaMtp2SelectedSparseParams {
    unsigned long long normalized;
    unsigned long long normalized_qs;
    unsigned long long normalized_ds;
    unsigned long long expert_ids;
    unsigned long long route_weights;
    unsigned long long gate_up_weights;
    unsigned long long down_weights;
    unsigned long long down_scale;
    unsigned long long gate_up_scratch;
    unsigned long long rank_output;
    unsigned long long output;
    unsigned long long residual;
    unsigned long long shared_raw;
    unsigned long long post_norm_1;
    unsigned long long post_norm_2;
    unsigned long long common_post_norm;
    unsigned tokens;
    unsigned hidden_dim;
    unsigned n_ff;
    unsigned n_expert;
    unsigned top_k;
    unsigned down_quant;
    unsigned q8dot_wide;
    float norm_eps;
    unsigned finalize_enabled;
    unsigned unit_offset;
};

static __device__ __forceinline__ float rnb_warp_sum(float value) {
#pragma unroll
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}


static __device__ __forceinline__ float rnb_q4k_row_dot(
    const unsigned char* __restrict__ row,
    const float* __restrict__ input,
    unsigned blocks_per_row,
    unsigned lane) {
    float acc = 0.0f;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row + (size_t)b * 144u;
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
                sc = (block[4u + j + 4u] & 0x0fu) |
                    ((block[4u + j - 4u] >> 6) << 4);
                mn = (block[4u + j + 4u] >> 4) |
                    ((block[4u + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned q = block[16u + q_index];
            q = local < 32u ? (q & 0x0fu) : (q >> 4);
            const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
            acc += y * input[(size_t)b * 256u + tid];
        }
    }
    return rnb_warp_sum(acc);
}

static __device__ __forceinline__ float rnb_q4k_row_dot_q8_narrow(
    const unsigned char* __restrict__ row,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned blocks_per_row,
    unsigned lane) {
    float acc = 0.0f;
    for (unsigned block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const unsigned char* block = row + (size_t)block_idx * 144u;
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
                sc = (block[4u + j + 4u] & 0x0fu) |
                    ((block[4u + j - 4u] >> 6) << 4);
                mn = (block[4u + j + 4u] >> 4) |
                    ((block[4u + j] >> 6) << 4);
            }
            const unsigned elem = (chunk & 7u) * 4u;
            const unsigned q_index = (j >> 1) * 32u + elem;
            const unsigned char* q_ptr = block + 16u + q_index;
            const unsigned shift = (j & 1u) * 4u;
            const unsigned q_raw = *reinterpret_cast<const unsigned*>(q_ptr);
            const int q_pack = (int)((q_raw >> shift) & 0x0f0f0f0fu);
            const int x_pack = *reinterpret_cast<const int*>(
                input_qs + (size_t)block_idx * 256u + j * 32u + elem);
            const int dot = __dp4a(q_pack, x_pack, 0);
            const int x_sum = __dp4a(0x01010101, x_pack, 0);
            const float x_d = input_ds[block_idx * 8u + j];
            acc += x_d *
                ((d * (float)sc) * (float)dot - dmin * (float)mn * (float)x_sum);
        }
    }
    return rnb_warp_sum(acc);
}

static __device__ __forceinline__ float rnb_q4k_row_dot_q8_wide(
    const unsigned char* __restrict__ row,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    unsigned blocks_per_row,
    unsigned lane) {
    float acc = 0.0f;
    const unsigned j = lane >> 2u;
    const unsigned elem = (lane & 3u) * 8u;
    for (unsigned block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const RnbMtp2Q4WideLane w =
            rnb_mtp2_q4k_wide_lane_decode(row + (size_t)block_idx * 144u, j, elem);
        const int2 x_raw = *reinterpret_cast<const int2*>(
            input_qs + (size_t)block_idx * 256u + j * 32u + elem);
        const float x_d = input_ds[block_idx * 8u + j];
        const int dot0 = __dp4a(w.q_pack0, x_raw.x, 0);
        const int x_sum0 = __dp4a(0x01010101, x_raw.x, 0);
        acc += x_d *
            ((w.d * w.sc) * (float)dot0 - w.dmin * w.mn * (float)x_sum0);
        const int dot1 = __dp4a(w.q_pack1, x_raw.y, 0);
        const int x_sum1 = __dp4a(0x01010101, x_raw.y, 0);
        acc += x_d *
            ((w.d * w.sc) * (float)dot1 - w.dmin * w.mn * (float)x_sum1);
    }
    return rnb_warp_sum(acc);
}

static __device__ __forceinline__ float rnb_q5_1_row_dot(
    const unsigned char* __restrict__ row,
    const float* __restrict__ input,
    unsigned blocks_per_row,
    unsigned lane) {
    float acc = 0.0f;
    for (unsigned block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const unsigned char* block = row + (size_t)block_idx * 24u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
        const unsigned qh = (unsigned)block[4] | ((unsigned)block[5] << 8) |
                            ((unsigned)block[6] << 16) | ((unsigned)block[7] << 24);
        const unsigned byte = block[8u + (lane & 15u)];
        const unsigned low = lane < 16u ? (byte & 0x0fu) : (byte >> 4);
        const unsigned high = (qh >> lane) & 1u;
        const float value = (float)(low | (high << 4)) * d + m;
        acc += value * input[(size_t)block_idx * 32u + lane];
    }
    return rnb_warp_sum(acc);
}

static __device__ __forceinline__ float rnb_q8_0_row_dot(
    const unsigned char* __restrict__ row,
    const float* __restrict__ input,
    unsigned blocks_per_row,
    unsigned lane) {
    float acc = 0.0f;
    for (unsigned block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const unsigned char* block = row + (size_t)block_idx * 34u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const signed char q = reinterpret_cast<const signed char*>(block + 2u)[lane];
        acc += (d * (float)q) * input[(size_t)block_idx * 32u + lane];
    }
    return rnb_warp_sum(acc);
}

extern "C" __global__ __launch_bounds__(256, 4) void rnb_gemma_mtp2_selected_sparse_sm86(
    GemmaMtp2SelectedSparseParams params) {
    cg::grid_group grid = cg::this_grid();
    const unsigned warp = threadIdx.x >> 5u;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned global_warp = blockIdx.x * 8u + warp;
    const unsigned warp_stride = gridDim.x * 8u;

    const float* normalized = reinterpret_cast<const float*>(params.normalized);
    signed char* normalized_qs = reinterpret_cast<signed char*>(params.normalized_qs);
    float* normalized_ds = reinterpret_cast<float*>(params.normalized_ds);
    const unsigned* expert_ids = reinterpret_cast<const unsigned*>(params.expert_ids);
    const float* route_weights = reinterpret_cast<const float*>(params.route_weights);
    const unsigned char* gate_up_weights =
        reinterpret_cast<const unsigned char*>(params.gate_up_weights);
    const unsigned char* down_weights =
        reinterpret_cast<const unsigned char*>(params.down_weights);
    const float* down_scale = reinterpret_cast<const float*>(params.down_scale);
    float* gate_up = reinterpret_cast<float*>(params.gate_up_scratch);
    float* rank_output = reinterpret_cast<float*>(params.rank_output);
    float* output = reinterpret_cast<float*>(params.output);
    float* residual = reinterpret_cast<float*>(params.residual);
    const float* shared_raw = reinterpret_cast<const float*>(params.shared_raw);
    const float* post_norm_1 = reinterpret_cast<const float*>(params.post_norm_1);
    const float* post_norm_2 = reinterpret_cast<const float*>(params.post_norm_2);
    const float* common_post_norm =
        reinterpret_cast<const float*>(params.common_post_norm);
    __shared__ float down_partial[256];

    const unsigned slots = params.tokens * params.top_k;
    const unsigned gate_up_rows = params.n_ff * 2u;
    const unsigned q4_blocks = params.hidden_dim / 256u;
    const size_t q4_row_bytes = (size_t)q4_blocks * 144u;
    const size_t gate_up_expert_bytes = (size_t)gate_up_rows * q4_row_bytes;
    const unsigned gate_up_total = slots * gate_up_rows;

    const unsigned q8_chunks = params.tokens * params.hidden_dim / 32u;
    for (unsigned chunk = global_warp; chunk < q8_chunks; chunk += warp_stride) {
        const unsigned index = chunk * 32u + lane;
        const float value = normalized[index];
        float max_abs = fabsf(value);
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            const float other = __shfl_down_sync(0xffffffffu, max_abs, offset);
            if (other > max_abs) {
                max_abs = other;
            }
        }
        const float d = __shfl_sync(0xffffffffu, max_abs / 127.0f, 0u);
        if (lane == 0u) {
            normalized_ds[chunk] = max_abs > 0.0f ? d : 0.0f;
        }
        int q = 0;
        if (d > 0.0f) {
            q = (int)nearbyintf(value / d);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
        }
        normalized_qs[index] = (signed char)q;
    }
    grid.sync();

    for (unsigned linear = global_warp; linear < gate_up_total; linear += warp_stride) {
        const unsigned slot = linear / gate_up_rows;
        const unsigned row = linear - slot * gate_up_rows;
        const unsigned token = slot / params.top_k;
        const unsigned expert = expert_ids[slot];
        const unsigned char* gate_up_expert =
            gate_up_weights + (size_t)expert * gate_up_expert_bytes;
        const unsigned char* weight_row =
            gate_up_expert + (size_t)row * q4_row_bytes;
        bool duplicate = false;
        for (unsigned other = 0; other < slots; ++other) {
            duplicate |= other != slot && expert_ids[other] == expert;
        }
        float value;
        if (duplicate) {
            const signed char* token_qs =
                normalized_qs + (size_t)token * params.hidden_dim;
            const float* token_ds =
                normalized_ds + (size_t)token * q4_blocks * 8u;
            value = params.q8dot_wide != 0u
                ? rnb_q4k_row_dot_q8_wide(weight_row, token_qs, token_ds, q4_blocks, lane)
                : rnb_q4k_row_dot_q8_narrow(weight_row, token_qs, token_ds, q4_blocks, lane);
        } else {
            value = rnb_q4k_row_dot(
                weight_row,
                normalized + (size_t)token * params.hidden_dim,
                q4_blocks,
                lane);
        }
        if (lane == 0u) {
            gate_up[linear] = value;
        }
    }
    grid.sync();

    const unsigned activation_total = slots * params.n_ff;
    const unsigned global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned thread_stride = gridDim.x * blockDim.x;
    for (unsigned linear = global_thread; linear < activation_total; linear += thread_stride) {
        const unsigned slot = linear / params.n_ff;
        const unsigned row = linear - slot * params.n_ff;
        const size_t base = (size_t)slot * gate_up_rows;
        const float gate = gate_up[base + row];
        const float up = gate_up[base + params.n_ff + row];
        const float gate3 = gate * gate * gate;
        const float gelu = 0.5f * gate *
            (1.0f + tanhf(0.7978845608028654f * (gate + 0.044715f * gate3)));
        gate_up[base + row] = gelu * up;
    }
    grid.sync();

    const unsigned down_blocks = params.n_ff / 32u;
    const size_t down_block_bytes = params.down_quant == 7u ? 24u : 34u;
    const size_t down_row_bytes = (size_t)down_blocks * down_block_bytes;
    const size_t down_expert_bytes = (size_t)params.hidden_dim * down_row_bytes;
    const unsigned down_total = slots * params.hidden_dim;
    for (unsigned linear = global_warp; linear < down_total; linear += warp_stride) {
        const unsigned slot = linear / params.hidden_dim;
        const unsigned row = linear - slot * params.hidden_dim;
        const unsigned expert = expert_ids[slot];
        const unsigned char* down_expert =
            down_weights + (size_t)expert * down_expert_bytes;
        const unsigned char* weight_row =
            down_expert + (size_t)row * down_row_bytes;
        const float* activation = gate_up + (size_t)slot * gate_up_rows;
        float partial[8];
#pragma unroll
        for (unsigned group = 0; group < 8u; ++group) {
            float acc = 0.0f;
            for (unsigned index = lane + group * 32u;
                 index < params.n_ff;
                 index += 256u) {
                const unsigned block_idx = index >> 5u;
                const unsigned char* block =
                    weight_row + (size_t)block_idx * down_block_bytes;
                float value;
                if (params.down_quant == 7u) {  // Q5_1
                    const unsigned d_m =
                        *reinterpret_cast<const unsigned*>(block);
                    const unsigned raw_d = d_m & 0xffffu;
                    const unsigned raw_m = d_m >> 16;
                    const float d =
                        __half2float(__ushort_as_half((unsigned short)raw_d));
                    const float m =
                        __half2float(__ushort_as_half((unsigned short)raw_m));
                    const unsigned qh =
                        *reinterpret_cast<const unsigned*>(block + 4u);
                    const unsigned byte = block[8u + (lane & 15u)];
                    const unsigned low = lane < 16u
                        ? (byte & 0x0fu)
                        : (byte >> 4);
                    const int q =
                        (int)(low | (((qh >> lane) & 1u) << 4));
                    value = d * (float)q + m;
                } else {  // Q8_0
                    const unsigned short raw_d =
                        *reinterpret_cast<const unsigned short*>(block);
                    const float d =
                        __half2float(__ushort_as_half(raw_d));
                    const signed char q = (signed char)block[2u + lane];
                    value = d * (float)q;
                }
                acc += value * activation[index];
            }
            partial[group] = acc;
        }
        partial[0] += partial[4];
        partial[1] += partial[5];
        partial[2] += partial[6];
        partial[3] += partial[7];
        partial[0] += partial[2];
        partial[1] += partial[3];
        partial[0] += partial[1];
        const float value = rnb_warp_sum(partial[0]);
        if (lane == 0u) {
            rank_output[linear] =
                value * (route_weights[slot] * down_scale[expert]);
        }
    }
    grid.sync();

    const unsigned output_total = params.tokens * params.hidden_dim;
    for (unsigned linear = global_thread; linear < output_total; linear += thread_stride) {
        const unsigned token = linear / params.hidden_dim;
        const unsigned row = linear - token * params.hidden_dim;
        const size_t token_base = (size_t)token * params.top_k * params.hidden_dim;
        float sum = 0.0f;
        for (unsigned rank = 0; rank < params.top_k; ++rank) {
            sum += rank_output[
                token_base + (size_t)rank * params.hidden_dim + row];
        }
        output[linear] = sum;
    }

    grid.sync();
    if (params.finalize_enabled != 0u && blockIdx.x < params.tokens) {
        const unsigned token = blockIdx.x;
        const size_t base = (size_t)token * params.hidden_dim;

        float shared_sum_sq = 0.0f;
        for (unsigned row = threadIdx.x; row < params.hidden_dim; row += blockDim.x) {
            const float value = shared_raw[base + row];
            shared_sum_sq += value * value;
        }
        down_partial[threadIdx.x] = shared_sum_sq;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) {
                down_partial[threadIdx.x] += down_partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            down_partial[0] = rsqrtf(
                down_partial[0] / (float)params.hidden_dim + params.norm_eps);
        }
        __syncthreads();
        const float shared_rrms = down_partial[0];

        float sparse_sum_sq = 0.0f;
        for (unsigned row = threadIdx.x; row < params.hidden_dim; row += blockDim.x) {
            const float value = output[base + row];
            sparse_sum_sq += value * value;
        }
        down_partial[threadIdx.x] = sparse_sum_sq;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) {
                down_partial[threadIdx.x] += down_partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            down_partial[0] = rsqrtf(
                down_partial[0] / (float)params.hidden_dim + params.norm_eps);
        }
        __syncthreads();
        const float sparse_rrms = down_partial[0];

        for (unsigned row = threadIdx.x; row < params.hidden_dim; row += blockDim.x) {
            const float shared_weight = params.unit_offset != 0u
                ? 1.0f + post_norm_1[row]
                : post_norm_1[row];
            const float sparse_weight = params.unit_offset != 0u
                ? 1.0f + post_norm_2[row]
                : post_norm_2[row];
            output[base + row] =
                shared_raw[base + row] * shared_rrms * shared_weight +
                output[base + row] * sparse_rrms * sparse_weight;
        }
        __syncthreads();

        float combined_sum_sq = 0.0f;
        for (unsigned row = threadIdx.x; row < params.hidden_dim; row += blockDim.x) {
            const float value = output[base + row];
            combined_sum_sq += value * value;
        }
        down_partial[threadIdx.x] = combined_sum_sq;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) {
                down_partial[threadIdx.x] += down_partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            down_partial[0] = rsqrtf(
                down_partial[0] / (float)params.hidden_dim + params.norm_eps);
        }
        __syncthreads();
        const float combined_rrms = down_partial[0];

        for (unsigned row = threadIdx.x; row < params.hidden_dim; row += blockDim.x) {
            const float common_weight = params.unit_offset != 0u
                ? 1.0f + common_post_norm[row]
                : common_post_norm[row];
            residual[base + row] += output[base + row] * combined_rrms * common_weight;
        }
    }
}
