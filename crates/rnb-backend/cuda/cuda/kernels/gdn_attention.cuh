extern "C" __global__ void rnb_gdn_gated_norm_silu(
    float* __restrict__ out,
    const float* __restrict__ delta_out,
    const float* __restrict__ z,
    const float* __restrict__ norm_weight,
    unsigned rows,
    unsigned head_dim,
    float eps) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256u) {
        return;
    }

    __shared__ float partial[256];
    float sum = 0.0f;
    for (unsigned i = tid; i < head_dim; i += 256u) {
        const float v = delta_out[row * head_dim + i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();

    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
    for (unsigned i = tid; i < head_dim; i += 256u) {
        const unsigned idx = row * head_dim + i;
        const float zv = z[idx];
        out[idx] = delta_out[idx] * inv_rms * norm_weight[i] * (zv / (1.0f + expf(-zv)));
    }
}

extern "C" __global__ void rnb_zero_f32(float* __restrict__ out, unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        out[i] = 0.0f;
    }
}

extern "C" __global__ void rnb_qwen35_router_topk_softmax_f32(
    const float* __restrict__ logits,
    unsigned* __restrict__ expert_ids,
    float* __restrict__ route_weights,
    unsigned* __restrict__ token_ids,
    unsigned window_tokens,
    unsigned n_expert,
    unsigned n_expert_used) {
    const unsigned token = blockIdx.x;
    const unsigned lane = threadIdx.x;
    if (token >= window_tokens || lane >= 32u) {
        return;
    }

    __shared__ unsigned selected_ids[32];
    __shared__ float selected_logits[32];
    const unsigned selected_count = n_expert_used < 32u ? n_expert_used : 32u;
    const float* token_logits = logits + token * n_expert;
    for (unsigned slot = 0; slot < selected_count; ++slot) {
        float best = -3.4028234663852886e38f;
        unsigned best_id = 0xffffffffu;
        for (unsigned expert = lane; expert < n_expert; expert += 32u) {
            bool already_selected = false;
#pragma unroll
            for (unsigned prev = 0; prev < slot; ++prev) {
                already_selected = already_selected || selected_ids[prev] == expert;
            }
            if (!already_selected) {
                const float value = token_logits[expert];
                if (value > best || (value == best && expert < best_id)) {
                    best = value;
                    best_id = expert;
                }
            }
        }
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            const float other = __shfl_down_sync(0xffffffffu, best, offset);
            const unsigned other_id = __shfl_down_sync(0xffffffffu, best_id, offset);
            if (other > best || (other == best && other_id < best_id)) {
                best = other;
                best_id = other_id;
            }
        }
        if (lane == 0u) {
            selected_ids[slot] = best_id;
            selected_logits[slot] = best;
        }
        __syncwarp();
    }

    if (lane == 0u) {
        float max_selected = selected_logits[0];
        for (unsigned slot = 1; slot < selected_count; ++slot) {
            max_selected = fmaxf(max_selected, selected_logits[slot]);
        }
        float selected_sum = 0.0f;
        for (unsigned slot = 0; slot < selected_count; ++slot) {
            selected_sum += expf(selected_logits[slot] - max_selected);
        }
        const unsigned out_base = token * n_expert_used;
        for (unsigned slot = 0; slot < selected_count; ++slot) {
            expert_ids[out_base + slot] = selected_ids[slot];
            route_weights[out_base + slot] =
                expf(selected_logits[slot] - max_selected) / selected_sum;
            token_ids[out_base + slot] = token;
        }
    }
}

extern "C" __global__ void rnb_qwen35_router_topk_logits_f32(
    const float* __restrict__ logits,
    unsigned* __restrict__ expert_ids,
    float* __restrict__ route_logits,
    unsigned* __restrict__ token_ids,
    unsigned window_tokens,
    unsigned n_expert,
    unsigned n_expert_used) {
    const unsigned token = blockIdx.x;
    const unsigned lane = threadIdx.x;
    if (token >= window_tokens || lane >= 32u) {
        return;
    }

    __shared__ unsigned selected_ids[32];
    __shared__ float selected_logits[32];
    const unsigned selected_count = n_expert_used < 32u ? n_expert_used : 32u;
    const float* token_logits = logits + token * n_expert;
    for (unsigned slot = 0; slot < selected_count; ++slot) {
        float best = -3.4028234663852886e38f;
        unsigned best_id = 0xffffffffu;
        for (unsigned expert = lane; expert < n_expert; expert += 32u) {
            bool already_selected = false;
#pragma unroll
            for (unsigned prev = 0; prev < slot; ++prev) {
                already_selected = already_selected || selected_ids[prev] == expert;
            }
            if (!already_selected) {
                const float value = token_logits[expert];
                if (value > best || (value == best && expert < best_id)) {
                    best = value;
                    best_id = expert;
                }
            }
        }
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            const float other = __shfl_down_sync(0xffffffffu, best, offset);
            const unsigned other_id = __shfl_down_sync(0xffffffffu, best_id, offset);
            if (other > best || (other == best && other_id < best_id)) {
                best = other;
                best_id = other_id;
            }
        }
        if (lane == 0u) {
            selected_ids[slot] = best_id;
            selected_logits[slot] = best;
        }
        __syncwarp();
    }

    if (lane == 0u) {
        const unsigned out_base = token * n_expert_used;
        for (unsigned slot = 0; slot < selected_count; ++slot) {
            expert_ids[out_base + slot] = selected_ids[slot];
            route_logits[out_base + slot] = selected_logits[slot];
            token_ids[out_base + slot] = token;
        }
    }
}

extern "C" __global__ void rnb_nemotron_sigmoid_topk_route_f32(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    unsigned* __restrict__ expert_ids,
    float* __restrict__ route_weights,
    unsigned* __restrict__ token_ids,
    unsigned window_tokens,
    unsigned n_expert,
    unsigned n_expert_used,
    float expert_weight_scale,
    unsigned has_bias) {
    const unsigned token = blockIdx.x;
    if (token >= window_tokens || threadIdx.x != 0u) {
        return;
    }

    unsigned selected_ids[32];
    float selected_scores[32];
    float selected_weights[32];
    const unsigned selected_count = n_expert_used < 32u ? n_expert_used : 32u;
    const float* token_logits = logits + token * n_expert;
    for (unsigned slot = 0; slot < selected_count; ++slot) {
        float best_score = -3.4028234663852886e38f;
        float best_weight = 0.0f;
        unsigned best_id = 0u;
        for (unsigned expert = 0; expert < n_expert; ++expert) {
            bool already_selected = false;
            for (unsigned prev = 0; prev < slot; ++prev) {
                already_selected = already_selected || selected_ids[prev] == expert;
            }
            if (already_selected) {
                continue;
            }
            const float logit = token_logits[expert];
            const float weight = 1.0f / (1.0f + expf(-logit));
            const float score = weight + (has_bias ? bias[expert] : 0.0f);
            if (score > best_score || (score == best_score && expert < best_id)) {
                best_score = score;
                best_weight = weight;
                best_id = expert;
            }
        }
        selected_ids[slot] = best_id;
        selected_scores[slot] = best_score;
        selected_weights[slot] = best_weight;
    }

    float selected_sum = 0.0f;
    for (unsigned slot = 0; slot < selected_count; ++slot) {
        selected_sum += selected_weights[slot];
    }
    const unsigned out_base = token * n_expert_used;
    for (unsigned slot = 0; slot < selected_count; ++slot) {
        expert_ids[out_base + slot] = selected_ids[slot];
        route_weights[out_base + slot] =
            selected_sum > 0.0f ? (selected_weights[slot] / selected_sum) * expert_weight_scale : 0.0f;
        token_ids[out_base + slot] = token;
    }
}

extern "C" __global__ void rnb_nemotron_reorder_route_slots(
    const unsigned* __restrict__ expert_ids_in,
    const float* __restrict__ route_weights_in,
    const unsigned* __restrict__ token_ids_in,
    const unsigned* __restrict__ order_indices,
    unsigned* __restrict__ expert_ids_out,
    float* __restrict__ route_weights_out,
    unsigned* __restrict__ token_ids_out,
    unsigned slots) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) {
        return;
    }
    const unsigned src = order_indices[slot];
    expert_ids_out[slot] = expert_ids_in[src];
    route_weights_out[slot] = route_weights_in[src];
    token_ids_out[slot] = token_ids_in[src];
}

extern "C" __global__ void rnb_qwen35_build_q4k_full_layer_slot_ptrs(
    unsigned long long* __restrict__ gate_ptrs,
    unsigned long long* __restrict__ up_ptrs,
    unsigned long long* __restrict__ down_ptrs,
    const unsigned* __restrict__ expert_ids,
    unsigned* __restrict__ pair_slots,
    unsigned long long gate_base,
    unsigned long long up_base,
    unsigned long long down_base,
    unsigned gate_expert_bytes,
    unsigned down_expert_bytes,
    unsigned slots_per_token,
    unsigned slots) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) {
        return;
    }
    const unsigned expert = expert_ids[slot];
    gate_ptrs[slot] = gate_base + (unsigned long long)expert * (unsigned long long)gate_expert_bytes;
    up_ptrs[slot] = up_base + (unsigned long long)expert * (unsigned long long)gate_expert_bytes;
    down_ptrs[slot] = down_base + (unsigned long long)expert * (unsigned long long)down_expert_bytes;
    if (pair_slots != nullptr
        && slots_per_token != 0u
        && slots == slots_per_token * 2u) {
        constexpr unsigned INVALID_SLOT = 0xffffffffu;
        constexpr unsigned SKIP_SLOT = 0xfffffffeu;
        unsigned pair_slot = INVALID_SLOT;
        if (slot < slots_per_token) {
            for (unsigned second = slots_per_token; second < slots; ++second) {
                if (expert_ids[second] == expert) {
                    pair_slot = second;
                    break;
                }
            }
        } else {
            for (unsigned first = 0; first < slots_per_token; ++first) {
                if (expert_ids[first] == expert) {
                    pair_slot = SKIP_SLOT;
                    break;
                }
            }
        }
        pair_slots[slot] = pair_slot;
    }
}

extern "C" __global__ void rnb_qwen35_build_q4k_compact_slot_ptrs(
    unsigned long long* __restrict__ gate_ptrs,
    unsigned long long* __restrict__ up_ptrs,
    unsigned long long* __restrict__ down_ptrs,
    const unsigned* __restrict__ expert_ids,
    const unsigned* __restrict__ expert_slab_indices,
    unsigned long long gate_base,
    unsigned long long up_base,
    unsigned long long down_base,
    unsigned gate_expert_bytes,
    unsigned up_expert_bytes,
    unsigned down_expert_bytes,
    unsigned slots) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) {
        return;
    }
    const unsigned expert = expert_ids[slot];
    const unsigned compact = expert_slab_indices[expert];
    gate_ptrs[slot] = gate_base + (unsigned long long)compact * (unsigned long long)gate_expert_bytes;
    up_ptrs[slot] = up_base + (unsigned long long)compact * (unsigned long long)up_expert_bytes;
    down_ptrs[slot] = down_base + (unsigned long long)compact * (unsigned long long)down_expert_bytes;
}

extern "C" __global__ void rnb_qwen35_build_q4k_mixed_slot_ptrs(
    unsigned long long* __restrict__ gate_ptrs,
    unsigned long long* __restrict__ up_ptrs,
    unsigned long long* __restrict__ down_ptrs,
    const unsigned* __restrict__ expert_ids,
    const unsigned long long* __restrict__ gate_expert_ptrs,
    const unsigned long long* __restrict__ up_expert_ptrs,
    const unsigned long long* __restrict__ down_expert_ptrs,
    unsigned slots) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) {
        return;
    }
    const unsigned expert = expert_ids[slot];
    gate_ptrs[slot] = gate_expert_ptrs[expert];
    up_ptrs[slot] = up_expert_ptrs[expert];
    down_ptrs[slot] = down_expert_ptrs[expert];
}

extern "C" __global__ void rnb_qwen35_shared_route_sigmoid_f32(
    float* __restrict__ route,
    const float* __restrict__ input,
    const float* __restrict__ scale,
    unsigned rows,
    unsigned hidden_dim) {
    const unsigned row = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256u) {
        return;
    }

    __shared__ float partial[256];
    float acc = 0.0f;
    const float* input_row = input + row * hidden_dim;
    for (unsigned i = tid; i < hidden_dim; i += 256u) {
        acc += input_row[i] * scale[i];
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
        route[row] = 1.0f / (1.0f + expf(-partial[0]));
    }
}

extern "C" __global__ void rnb_nemotron_mamba2_split_projection(
    const float* __restrict__ projected,
    const float* __restrict__ dt_bias,
    float* __restrict__ z_out,
    float* __restrict__ conv_out,
    float* __restrict__ dt_out,
    unsigned seq_len,
    unsigned d_inner,
    unsigned conv_channels,
    unsigned num_heads) {
    const unsigned rows = d_inner + conv_channels + num_heads;
    const unsigned total = seq_len * rows;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const unsigned token = i / rows;
    const unsigned col = i - token * rows;
    const float value = projected[i];
    if (col < d_inner) {
        z_out[token * d_inner + col] = value;
    } else if (col < d_inner + conv_channels) {
        const unsigned c = col - d_inner;
        conv_out[token * conv_channels + c] = value;
    } else {
        const unsigned h = col - d_inner - conv_channels;
        dt_out[token * num_heads + h] = value + dt_bias[h];
    }
}

extern "C" __global__ void rnb_gdn_build_conv_input_f32(
    float* __restrict__ out,
    const float* __restrict__ conv_state,
    const float* __restrict__ qkv_rows,
    unsigned window_tokens,
    unsigned channels,
    unsigned kernel_size) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned state_rows = kernel_size - 1u;
    const unsigned len = (window_tokens + state_rows) * channels;
    if (i >= len) {
        return;
    }
    const unsigned row = i / channels;
    const unsigned channel = i - row * channels;
    if (row < state_rows) {
        out[i] = conv_state[i];
    } else {
        out[i] = qkv_rows[(row - state_rows) * channels + channel];
    }
}

extern "C" __global__ void rnb_ssm_conv1d_silu(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    unsigned seq_len,
    unsigned channels,
    unsigned kernel_size) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned len = seq_len * channels;
    if (i >= len) {
        return;
    }
    const unsigned t = i / channels;
    const unsigned c = i - t * channels;
    float sum = 0.0f;
    for (unsigned k = 0; k < kernel_size; ++k) {
        sum += input[(t + k) * channels + c] * kernel[k * channels + c];
    }
    out[i] = sum / (1.0f + expf(-sum));
}

extern "C" __global__ void rnb_nemotron_mamba2_conv1d_bias_silu(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    const float* __restrict__ bias,
    unsigned seq_len,
    unsigned channels,
    unsigned kernel_size) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned len = seq_len * channels;
    if (i >= len) {
        return;
    }
    const unsigned t = i / channels;
    const unsigned c = i - t * channels;
    float sum = bias[c];
    for (unsigned k = 0; k < kernel_size; ++k) {
        sum += input[(t + k) * channels + c] * kernel[k * channels + c];
    }
    out[i] = sum / (1.0f + expf(-sum));
}

extern "C" __global__ void rnb_nemotron_mamba2_add_residual(
    float* __restrict__ out,
    const float* __restrict__ proj,
    const float* __restrict__ residual,
    unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        out[i] = proj[i] + residual[i];
    }
}

extern "C" __global__ void rnb_gdn_prepare_delta_qkv_f32(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ conv_out,
    unsigned window_tokens,
    unsigned conv_channels,
    unsigned num_k_heads,
    unsigned num_v_heads,
    unsigned head_k_dim,
    unsigned head_v_dim,
    float eps,
    float q_scale) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned rows = window_tokens * num_v_heads;
    if (row >= rows) {
        return;
    }

    const unsigned token = row / num_v_heads;
    const unsigned v_head = row - token * num_v_heads;
    const unsigned k_head = v_head % num_k_heads;
    const unsigned q_dim = num_k_heads * head_k_dim;
    const unsigned k_dim = q_dim;
    const unsigned conv_base = token * conv_channels;
    const unsigned q_base = conv_base + k_head * head_k_dim;
    const unsigned k_base = conv_base + q_dim + k_head * head_k_dim;
    const unsigned v_base = conv_base + q_dim + k_dim + v_head * head_v_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (unsigned d = 0; d < head_k_dim; ++d) {
        const float q = conv_out[q_base + d];
        const float k = conv_out[k_base + d];
        q_sum = __fadd_rn(q_sum, __fmul_rn(q, q));
        k_sum = __fadd_rn(k_sum, __fmul_rn(k, k));
    }
    const float q_inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(q_sum, eps)));
    const float k_inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(k_sum, eps)));
    for (unsigned d = 0; d < head_k_dim; ++d) {
        q_out[row * head_k_dim + d] =
            __fmul_rn(__fmul_rn(conv_out[q_base + d], q_inv), q_scale);
        k_out[row * head_k_dim + d] = __fmul_rn(conv_out[k_base + d], k_inv);
    }
    for (unsigned d = 0; d < head_v_dim; ++d) {
        v_out[row * head_v_dim + d] = conv_out[v_base + d];
    }
}

extern "C" __global__ void rnb_gdn_prepare_delta_qkv_f32_warp(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ conv_out,
    unsigned window_tokens,
    unsigned conv_channels,
    unsigned num_k_heads,
    unsigned num_v_heads,
    unsigned head_k_dim,
    unsigned head_v_dim,
    float eps,
    float q_scale) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * 8u + warp;
    const unsigned rows = window_tokens * num_v_heads;
    if (row >= rows) {
        return;
    }

    const unsigned token = row / num_v_heads;
    const unsigned v_head = row - token * num_v_heads;
    const unsigned k_head = v_head % num_k_heads;
    const unsigned q_dim = num_k_heads * head_k_dim;
    const unsigned conv_base = token * conv_channels;
    const unsigned q_base = conv_base + k_head * head_k_dim;
    const unsigned k_base = conv_base + q_dim + k_head * head_k_dim;
    const unsigned v_base = conv_base + 2u * q_dim + v_head * head_v_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (unsigned d = lane; d < head_k_dim; d += 32u) {
        const float q = conv_out[q_base + d];
        const float k = conv_out[k_base + d];
        q_sum = __fadd_rn(q_sum, __fmul_rn(q, q));
        k_sum = __fadd_rn(k_sum, __fmul_rn(k, k));
    }
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
    const float q_inv =
        __fdiv_rn(1.0f, sqrtf(__fadd_rn(__shfl_sync(0xffffffffu, q_sum, 0), eps)));
    const float k_inv =
        __fdiv_rn(1.0f, sqrtf(__fadd_rn(__shfl_sync(0xffffffffu, k_sum, 0), eps)));
    for (unsigned d = lane; d < head_k_dim; d += 32u) {
        q_out[row * head_k_dim + d] =
            __fmul_rn(__fmul_rn(conv_out[q_base + d], q_inv), q_scale);
        k_out[row * head_k_dim + d] = __fmul_rn(conv_out[k_base + d], k_inv);
    }
    for (unsigned d = lane; d < head_v_dim; d += 32u) {
        v_out[row * head_v_dim + d] = conv_out[v_base + d];
    }
}

extern "C" __global__ void rnb_gdn_prepare_delta_gate_beta_f32(
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    unsigned len,
    unsigned num_heads) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    const unsigned head = i % num_heads;
    const float biased = alpha[i] + dt_bias[head];
    gate_out[i] = logf(1.0f + expf(biased)) * ssm_a[head];
    beta_out[i] = 1.0f / (1.0f + expf(-beta[i]));
}

extern "C" __global__ void rnb_attention_prefill_flash_hd256(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned head_dim,
    float scale,
    unsigned window,
    float softcap) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (i >= seq_len || h >= num_heads || num_kv_heads == 0u || head_dim == 0u) {
        return;
    }

    __shared__ float partial[8];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned start =
        window > 0u && global_pos + 1u > window ? global_pos + 1u - window : 0u;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    for (unsigned chunk = 0u; chunk < head_dim; chunk += 256u) {
        const unsigned dim = chunk + tid;
        const bool active = dim < head_dim;
        float row_max = -3.4028234663852886e38f;
        float row_sum = 0.0f;
        float acc = 0.0f;

        for (unsigned j = start; j <= global_pos; ++j) {
            float dot = 0.0f;
            for (unsigned d = tid; d < head_dim; d += 256u) {
                dot += q[i * num_heads * head_dim + h * head_dim + d]
                    * k[j * num_kv_heads * head_dim + kv_h * head_dim + d];
            }
            for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            }
            if (lane == 0u) {
                partial[warp] = dot;
            }
            __syncthreads();
            if (tid < 8u) {
                float sum = partial[tid];
                for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                    sum += __shfl_down_sync(0x000000ffu, sum, offset);
                }
                if (tid == 0u) {
                    partial[0] = sum;
                }
            }
            __syncthreads();
            float score = partial[0] * scale;
            if (softcap > 0.0f) {
                score = softcap * tanhf(score / softcap);
            }
            const float new_max = fmaxf(row_max, score);
            const float old_scale = row_max == -3.4028234663852886e38f
                ? 0.0f
                : expf(row_max - new_max);
            const float p = expf(score - new_max);
            const float vv = active
                ? v[j * num_kv_heads * head_dim + kv_h * head_dim + dim]
                : 0.0f;
            acc = acc * old_scale + p * vv;
            row_sum = row_sum * old_scale + p;
            row_max = new_max;
            __syncthreads();
        }

        if (active) {
            out[i * num_heads * head_dim + h * head_dim + dim] =
                row_sum > 0.0f ? acc / row_sum : 0.0f;
        }
        __syncthreads();
    }
}

extern "C" __global__ void rnb_attention_prefill_flash_hd256_window(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 256u || i >= seq_len || h >= num_heads || num_kv_heads == 0u || window == 0u) {
        return;
    }

    __shared__ float partial[8];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned start = global_pos + 1u > window ? global_pos + 1u - window : 0u;
    const float qv = q[i * num_heads * 256u + h * 256u + tid];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = start; j <= global_pos; ++j) {
        const float kv = k[j * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot = qv * kv;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 8u) {
            float sum = partial[tid];
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_down_sync(0x000000ffu, sum, offset);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = v[j * num_kv_heads * 256u + kv_h * 256u + tid];
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[i * num_heads * 256u + h * 256u + tid] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

extern "C" __global__ void rnb_attention_prefill_flash_hd512(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 512u || i >= seq_len || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[16];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const float qv = q[i * num_heads * 512u + h * 512u + tid];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = 0; j <= global_pos; ++j) {
        const float kv = k[j * num_kv_heads * 512u + kv_h * 512u + tid];
        float dot = qv * kv;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 16u) {
            float sum = partial[tid];
            for (unsigned offset = 8u; offset > 0u; offset >>= 1u) {
                sum += __shfl_down_sync(0x0000ffffu, sum, offset);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = v[j * num_kv_heads * 512u + kv_h * 512u + tid];
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[i * num_heads * 512u + h * 512u + tid] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

extern "C" __global__ void rnb_attention_prefill_flash_hd512_w256(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 256u || i >= seq_len || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[8];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned q_base = i * num_heads * 512u + h * 512u;
    const float q0 = q[q_base + tid];
    const float q1 = q[q_base + tid + 256u];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned j = 0; j <= global_pos; ++j) {
        const unsigned kv_base = j * num_kv_heads * 512u + kv_h * 512u;
        const float k0 = k[kv_base + tid];
        const float k1 = k[kv_base + tid + 256u];
        float dot = q0 * k0 + q1 * k1;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 8u) {
            float sum = partial[tid];
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_down_sync(0x000000ffu, sum, offset);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float v0 = v[kv_base + tid];
        const float v1 = v[kv_base + tid + 256u];
        acc0 = acc0 * old_scale + p * v0;
        acc1 = acc1 * old_scale + p * v1;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    const unsigned out_base = i * num_heads * 512u + h * 512u;
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    out[out_base + tid] = acc0 * inv_sum;
    out[out_base + tid + 256u] = acc1 * inv_sum;
}

extern "C" __global__ void rnb_attention_prefill_flash_hd512_window_w256(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 256u || i >= seq_len || h >= num_heads || num_kv_heads == 0u || window == 0u) {
        return;
    }

    __shared__ float partial[8];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned start = global_pos + 1u > window ? global_pos + 1u - window : 0u;
    const unsigned q_base = i * num_heads * 512u + h * 512u;
    const float q0 = q[q_base + tid];
    const float q1 = q[q_base + tid + 256u];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned j = start; j <= global_pos; ++j) {
        const unsigned kv_base = j * num_kv_heads * 512u + kv_h * 512u;
        const float k0 = k[kv_base + tid];
        const float k1 = k[kv_base + tid + 256u];
        float dot = q0 * k0 + q1 * k1;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 8u) {
            float sum = partial[tid];
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_down_sync(0x000000ffu, sum, offset);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float v0 = v[kv_base + tid];
        const float v1 = v[kv_base + tid + 256u];
        acc0 = acc0 * old_scale + p * v0;
        acc1 = acc1 * old_scale + p * v1;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    const unsigned out_base = i * num_heads * 512u + h * 512u;
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    out[out_base + tid] = acc0 * inv_sum;
    out[out_base + tid + 256u] = acc1 * inv_sum;
}

extern "C" __global__ void rnb_attention_prefill_flash_hd128(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 128u || i >= seq_len || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[4];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const float qv = q[i * num_heads * 128u + h * 128u + tid];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = 0; j <= global_pos; ++j) {
        const float kv = k[j * num_kv_heads * 128u + kv_h * 128u + tid];
        float dot = qv * kv;
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 4u) {
            float sum = partial[tid];
            for (unsigned offset = 2u; offset > 0u; offset >>= 1u) {
                sum += __shfl_down_sync(0x0000000fu, sum, offset);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = v[j * num_kv_heads * 128u + kv_h * 128u + tid];
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[i * num_heads * 128u + h * 128u + tid] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

extern "C" __global__ void rnb_attention_decode_hd256(
    float* __restrict__ out,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    if (lane >= 256u || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[256];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const float qv = q[h * 256u + lane];

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = 0; j < kv_len; ++j) {
        const unsigned kv_base = j * num_kv_heads * 256u + kv_h * 256u;
        const float kv = __half2float(__ushort_as_half(k[kv_base + lane]));
        partial[lane] = qv * kv;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            __syncthreads();
        }
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = __half2float(__ushort_as_half(v[kv_base + lane]));
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[h * 256u + lane] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}
extern "C" __global__ void rnb_attention_decode_hd256_split_partials(
    float* __restrict__ partial_acc,
    float* __restrict__ partial_meta,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned chunk_size) {
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    const unsigned chunk = blockIdx.y;
    if (lane >= 256u || h >= num_heads || num_kv_heads == 0u || chunk_size == 0u) {
        return;
    }

    __shared__ float partial[256];
    const unsigned num_chunks = (kv_len + chunk_size - 1u) / chunk_size;
    const unsigned start = chunk * chunk_size;
    unsigned end = start + chunk_size;
    if (end > kv_len) {
        end = kv_len;
    }
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const float qv = q[h * 256u + lane];

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = start; j < end; ++j) {
        const unsigned kv_base = j * num_kv_heads * 256u + kv_h * 256u;
        const float kv = __half2float(__ushort_as_half(k[kv_base + lane]));
        partial[lane] = qv * kv;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            __syncthreads();
        }
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = __half2float(__ushort_as_half(v[kv_base + lane]));
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    const unsigned chunk_base = h * num_chunks + chunk;
    partial_acc[chunk_base * 256u + lane] = acc;
    if (lane == 0u) {
        partial_meta[chunk_base * 2u] = row_max;
        partial_meta[chunk_base * 2u + 1u] = row_sum;
    }
}

extern "C" __global__ void rnb_attention_decode_hd256_split_reduce(
    float* __restrict__ out,
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_meta,
    unsigned num_heads,
    unsigned num_chunks) {
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    if (lane >= 256u || h >= num_heads || num_chunks == 0u) {
        return;
    }

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned chunk = 0; chunk < num_chunks; ++chunk) {
        const unsigned chunk_base = h * num_chunks + chunk;
        const float chunk_max = partial_meta[chunk_base * 2u];
        const float chunk_sum = partial_meta[chunk_base * 2u + 1u];
        const float chunk_acc = partial_acc[chunk_base * 256u + lane];
        if (chunk_sum <= 0.0f) {
            continue;
        }
        const float new_max = fmaxf(row_max, chunk_max);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float chunk_scale = expf(chunk_max - new_max);
        acc = acc * old_scale + chunk_acc * chunk_scale;
        row_sum = row_sum * old_scale + chunk_sum * chunk_scale;
        row_max = new_max;
    }

    out[h * 256u + lane] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}



extern "C" __global__ void rnb_attention_decode_hd512(
    float* __restrict__ out,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    if (lane >= 512u || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[512];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const float qv = q[h * 512u + lane];

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = 0; j < kv_len; ++j) {
        const unsigned kv_base = j * num_kv_heads * 512u + kv_h * 512u;
        const float kv = __half2float(__ushort_as_half(k[kv_base + lane]));
        partial[lane] = qv * kv;
        __syncthreads();
        for (unsigned stride = 256u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            __syncthreads();
        }
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = __half2float(__ushort_as_half(v[kv_base + lane]));
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[h * 512u + lane] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

extern "C" __global__ void rnb_attention_decode_hd512_len_device(
    float* __restrict__ out,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    const unsigned* __restrict__ kv_len_dev,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned kv_len = *kv_len_dev;
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    if (lane >= 512u || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[512];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const float qv = q[h * 512u + lane];

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = 0; j < kv_len; ++j) {
        const unsigned kv_base = j * num_kv_heads * 512u + kv_h * 512u;
        const float kv = __half2float(__ushort_as_half(k[kv_base + lane]));
        partial[lane] = qv * kv;
        __syncthreads();
        for (unsigned stride = 256u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            __syncthreads();
        }
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = __half2float(__ushort_as_half(v[kv_base + lane]));
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[h * 512u + lane] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

extern "C" __global__ void rnb_attention_decode_hd512_split_partials(
    float* __restrict__ partial_acc,
    float* __restrict__ partial_meta,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned chunk_size) {
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    const unsigned chunk = blockIdx.y;
    if (lane >= 512u || h >= num_heads || num_kv_heads == 0u || chunk_size == 0u) {
        return;
    }

    __shared__ float partial[512];
    const unsigned num_chunks = (kv_len + chunk_size - 1u) / chunk_size;
    const unsigned start = chunk * chunk_size;
    unsigned end = start + chunk_size;
    if (end > kv_len) {
        end = kv_len;
    }
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const float qv = q[h * 512u + lane];

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = start; j < end; ++j) {
        const unsigned kv_base = j * num_kv_heads * 512u + kv_h * 512u;
        const float kv = __half2float(__ushort_as_half(k[kv_base + lane]));
        partial[lane] = qv * kv;
        __syncthreads();
        for (unsigned stride = 256u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            __syncthreads();
        }
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = __half2float(__ushort_as_half(v[kv_base + lane]));
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    const unsigned chunk_base = h * num_chunks + chunk;
    partial_acc[chunk_base * 512u + lane] = acc;
    if (lane == 0u) {
        partial_meta[chunk_base * 2u] = row_max;
        partial_meta[chunk_base * 2u + 1u] = row_sum;
    }
}

extern "C" __global__ void rnb_attention_decode_hd512_split_reduce(
    float* __restrict__ out,
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_meta,
    unsigned num_heads,
    unsigned num_chunks) {
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    if (lane >= 512u || h >= num_heads || num_chunks == 0u) {
        return;
    }

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned chunk = 0; chunk < num_chunks; ++chunk) {
        const unsigned chunk_base = h * num_chunks + chunk;
        const float chunk_max = partial_meta[chunk_base * 2u];
        const float chunk_sum = partial_meta[chunk_base * 2u + 1u];
        const float chunk_acc = partial_acc[chunk_base * 512u + lane];
        if (chunk_sum <= 0.0f) {
            continue;
        }
        const float new_max = fmaxf(row_max, chunk_max);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float chunk_scale = expf(chunk_max - new_max);
        acc = acc * old_scale + chunk_acc * chunk_scale;
        row_sum = row_sum * old_scale + chunk_sum * chunk_scale;
        row_max = new_max;
    }

    out[h * 512u + lane] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

extern "C" __global__ void rnb_attention_decode_hd128(
    float* __restrict__ out,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned lane = threadIdx.x;
    const unsigned h = blockIdx.x;
    if (lane >= 128u || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[128];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const float qv = q[h * 128u + lane];

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (unsigned j = 0; j < kv_len; ++j) {
        const unsigned kv_base = j * num_kv_heads * 128u + kv_h * 128u;
        const float kv = __half2float(__ushort_as_half(k[kv_base + lane]));
        partial[lane] = qv * kv;
        __syncthreads();
        for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            __syncthreads();
        }
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = __half2float(__ushort_as_half(v[kv_base + lane]));
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[h * 128u + lane] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

// =============================================================================
// cu38 Phase 4: attention prefill flash hd256 window — j batch=4 변형
//
// 원본 (line 483 hd256_window) 의 inner loop = 매 j 마다 1 token 처리. 매번
// warp+block reduce + 2 syncthreads. heavy overhead.
//
// j batch=4: 4 token K 동시 load + 4 dot product partial reduce. syncthreads
// 1번 per 4-j chunk. dot reduce 도 batch.
//
// register: 4 acc (acc0~3), 4 row_max, 4 row_sum, 4 k value. block size 256.
// shared mem partial[8*4] = 32 float.
//
// grid (seq_len, num_heads, 1), block (256, 1, 1) — 원본 동일.
extern "C" __global__ void rnb_attention_prefill_flash_hd256_window_jbatch4(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 256u || i >= seq_len || h >= num_heads || num_kv_heads == 0u || window == 0u) {
        return;
    }

    __shared__ float partial[8 * 4];   // 4 j × 8 warp = 32 float
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned start = global_pos + 1u > window ? global_pos + 1u - window : 0u;
    const float qv = q[i * num_heads * 256u + h * 256u + tid];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    // 4-j batch loop
    unsigned j = start;
    for (; j + 4u <= global_pos + 1u; j += 4u) {
        // 4 k load + 4 dot
        float dot0 = qv * k[(j + 0u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot1 = qv * k[(j + 1u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot2 = qv * k[(j + 2u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot3 = qv * k[(j + 3u) * num_kv_heads * 256u + kv_h * 256u + tid];
        // warp reduce 4 dot
        for (unsigned off = 16u; off > 0u; off >>= 1u) {
            dot0 += __shfl_down_sync(0xffffffffu, dot0, off);
            dot1 += __shfl_down_sync(0xffffffffu, dot1, off);
            dot2 += __shfl_down_sync(0xffffffffu, dot2, off);
            dot3 += __shfl_down_sync(0xffffffffu, dot3, off);
        }
        if (lane == 0u) {
            partial[warp * 4 + 0] = dot0;
            partial[warp * 4 + 1] = dot1;
            partial[warp * 4 + 2] = dot2;
            partial[warp * 4 + 3] = dot3;
        }
        __syncthreads();
        // 8 warp reduce (per j)
        if (tid < 32u) {
            const unsigned slot = tid >> 3;       // 0..3 (j index)
            const unsigned warp_id = tid & 7u;    // 0..7
            float sum = partial[warp_id * 4 + slot];
            // reduce 8 warps via shfl within first 8 lanes of slot group
            for (unsigned off = 4u; off > 0u; off >>= 1u) {
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            if (warp_id == 0u) {
                partial[slot] = sum;
            }
        }
        __syncthreads();
        const float s0 = partial[0] * scale;
        const float s1 = partial[1] * scale;
        const float s2 = partial[2] * scale;
        const float s3 = partial[3] * scale;
        __syncthreads();

        // online softmax update 4 step
        #pragma unroll
        for (unsigned sj = 0; sj < 4u; ++sj) {
            const float score = (sj == 0) ? s0 : (sj == 1) ? s1 : (sj == 2) ? s2 : s3;
            const float new_max = fmaxf(row_max, score);
            const float old_scale = row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
            const float p = expf(score - new_max);
            const float vv = v[(j + sj) * num_kv_heads * 256u + kv_h * 256u + tid];
            acc = acc * old_scale + p * vv;
            row_sum = row_sum * old_scale + p;
            row_max = new_max;
        }
    }
    // 남은 j (tail) 처리 — 원본 단일 j loop
    for (; j <= global_pos; ++j) {
        const float kv = k[j * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot = qv * kv;
        for (unsigned off = 16u; off > 0u; off >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 8u) {
            float sum = partial[tid];
            for (unsigned off = 4u; off > 0u; off >>= 1u) {
                sum += __shfl_down_sync(0x000000ffu, sum, off);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv = v[j * num_kv_heads * 256u + kv_h * 256u + tid];
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    out[i * num_heads * 256u + h * 256u + tid] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}


extern "C" __global__ void rnb_attention_prefill_flash_hd256_window_split_partials_jbatch8(
    float* __restrict__ partial_acc,
    float* __restrict__ partial_meta,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window,
    unsigned chunk_size,
    unsigned num_chunks) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned chunk = blockIdx.z;
    if (tid >= 256u || i >= seq_len || h >= num_heads || chunk >= num_chunks
        || num_kv_heads == 0u || window == 0u || chunk_size == 0u || kv_len < seq_len) {
        return;
    }

    __shared__ float partial[8 * 8];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned window_start =
        global_pos + 1u > window ? global_pos + 1u - window : 0u;
    const unsigned start = window_start + chunk * chunk_size;
    unsigned end = start + chunk_size;
    if (end > global_pos + 1u) {
        end = global_pos + 1u;
    }
    const float qv = q[i * num_heads * 256u + h * 256u + tid];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    unsigned j = start;
    for (; j + 8u <= end; j += 8u) {
        float dot0 = qv * k[(j + 0u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot1 = qv * k[(j + 1u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot2 = qv * k[(j + 2u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot3 = qv * k[(j + 3u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot4 = qv * k[(j + 4u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot5 = qv * k[(j + 5u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot6 = qv * k[(j + 6u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot7 = qv * k[(j + 7u) * num_kv_heads * 256u + kv_h * 256u + tid];
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot0 += __shfl_down_sync(0xffffffffu, dot0, offset);
            dot1 += __shfl_down_sync(0xffffffffu, dot1, offset);
            dot2 += __shfl_down_sync(0xffffffffu, dot2, offset);
            dot3 += __shfl_down_sync(0xffffffffu, dot3, offset);
            dot4 += __shfl_down_sync(0xffffffffu, dot4, offset);
            dot5 += __shfl_down_sync(0xffffffffu, dot5, offset);
            dot6 += __shfl_down_sync(0xffffffffu, dot6, offset);
            dot7 += __shfl_down_sync(0xffffffffu, dot7, offset);
        }
        if (lane == 0u) {
            partial[warp * 8u + 0u] = dot0;
            partial[warp * 8u + 1u] = dot1;
            partial[warp * 8u + 2u] = dot2;
            partial[warp * 8u + 3u] = dot3;
            partial[warp * 8u + 4u] = dot4;
            partial[warp * 8u + 5u] = dot5;
            partial[warp * 8u + 6u] = dot6;
            partial[warp * 8u + 7u] = dot7;
        }
        __syncthreads();
        if (tid < 64u) {
            const unsigned slot = tid >> 3;
            const unsigned warp_id = tid & 7u;
            float sum = partial[warp_id * 8u + slot];
#pragma unroll
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            if (warp_id == 0u) {
                partial[slot] = sum;
            }
        }
        __syncthreads();
        const float score0 = partial[0] * scale;
        const float score1 = partial[1] * scale;
        const float score2 = partial[2] * scale;
        const float score3 = partial[3] * scale;
        const float score4 = partial[4] * scale;
        const float score5 = partial[5] * scale;
        const float score6 = partial[6] * scale;
        const float score7 = partial[7] * scale;
        __syncthreads();

#pragma unroll
        for (unsigned batch = 0u; batch < 8u; ++batch) {
            const float score = batch == 0u   ? score0
                : batch == 1u                ? score1
                : batch == 2u                ? score2
                : batch == 3u                ? score3
                : batch == 4u                ? score4
                : batch == 5u                ? score5
                : batch == 6u                ? score6
                                             : score7;
            const float new_max = fmaxf(row_max, score);
            const float old_scale =
                row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
            const float probability = expf(score - new_max);
            const float value =
                v[(j + batch) * num_kv_heads * 256u + kv_h * 256u + tid];
            acc = acc * old_scale + probability * value;
            row_sum = row_sum * old_scale + probability;
            row_max = new_max;
        }
    }
    for (; j < end; ++j) {
        const float kv = k[j * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot = qv * kv;
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 8u) {
            float sum = partial[tid];
#pragma unroll
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_down_sync(0x000000ffu, sum, offset);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale =
            row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
        const float probability = expf(score - new_max);
        const float value = v[j * num_kv_heads * 256u + kv_h * 256u + tid];
        acc = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = new_max;
        __syncthreads();
    }

    const unsigned partial_row = (i * num_heads + h) * num_chunks + chunk;
    partial_acc[partial_row * 256u + tid] = acc;
    if (tid == 0u) {
        partial_meta[partial_row * 2u] = row_max;
        partial_meta[partial_row * 2u + 1u] = row_sum;
    }
}


// Four causal queries share each F32 K/V load while preserving the per-query
// online-softmax and ordered split reduction arithmetic.
extern "C" __global__ void rnb_attention_prefill_flash_hd256_window_split_partials_qtile4_jbatch4(
    float* __restrict__ partial_acc,
    float* __restrict__ partial_meta,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window,
    unsigned chunk_size,
    unsigned num_chunks) {
    constexpr unsigned query_tile = 4u;
    constexpr unsigned key_batch = 4u;
    const unsigned tid = threadIdx.x;
    const unsigned query_base = blockIdx.x * query_tile;
    const unsigned h = blockIdx.y;
    const unsigned chunk = blockIdx.z;
    if (tid >= 256u || query_base >= seq_len || h >= num_heads || chunk >= num_chunks
        || num_kv_heads == 0u || window < kv_len || chunk_size == 0u || kv_len < seq_len) {
        return;
    }

    __shared__ float partial[8 * query_tile * key_batch];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned last_query =
        query_base + query_tile < seq_len ? query_base + query_tile - 1u : seq_len - 1u;
    const unsigned last_global_pos = kv_len - seq_len + last_query;
    const unsigned start = chunk * chunk_size;
    unsigned end = start + chunk_size;
    if (end > last_global_pos + 1u) {
        end = last_global_pos + 1u;
    }
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float qv[query_tile];
    float row_max[query_tile];
    float row_sum[query_tile];
    float acc[query_tile];
#pragma unroll
    for (unsigned query = 0u; query < query_tile; ++query) {
        const unsigned i = query_base + query;
        qv[query] =
            i < seq_len ? q[i * num_heads * 256u + h * 256u + tid] : 0.0f;
        row_max[query] = -3.4028234663852886e38f;
        row_sum[query] = 0.0f;
        acc[query] = 0.0f;
    }

    unsigned j = start;
    for (; j + key_batch <= end; j += key_batch) {
        float dot[query_tile][key_batch];
#pragma unroll
        for (unsigned key = 0u; key < key_batch; ++key) {
            const float kv =
                k[(j + key) * num_kv_heads * 256u + kv_h * 256u + tid];
#pragma unroll
            for (unsigned query = 0u; query < query_tile; ++query) {
                dot[query][key] = qv[query] * kv;
            }
        }
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
#pragma unroll
            for (unsigned query = 0u; query < query_tile; ++query) {
#pragma unroll
                for (unsigned key = 0u; key < key_batch; ++key) {
                    dot[query][key] +=
                        __shfl_down_sync(0xffffffffu, dot[query][key], offset);
                }
            }
        }
        if (lane == 0u) {
#pragma unroll
            for (unsigned query = 0u; query < query_tile; ++query) {
#pragma unroll
                for (unsigned key = 0u; key < key_batch; ++key) {
                    partial[(warp * query_tile + query) * key_batch + key] =
                        dot[query][key];
                }
            }
        }
        __syncthreads();
        if (tid < 8u * query_tile * key_batch) {
            const unsigned slot = tid >> 3;
            const unsigned warp_id = tid & 7u;
            float sum = partial[(warp_id * query_tile * key_batch) + slot];
#pragma unroll
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            if (warp_id == 0u) {
                partial[slot] = sum;
            }
        }
        __syncthreads();

#pragma unroll
        for (unsigned key = 0u; key < key_batch; ++key) {
            const float value =
                v[(j + key) * num_kv_heads * 256u + kv_h * 256u + tid];
#pragma unroll
            for (unsigned query = 0u; query < query_tile; ++query) {
                const unsigned i = query_base + query;
                const unsigned global_pos = kv_len - seq_len + i;
                if (i >= seq_len || j + key > global_pos) {
                    continue;
                }
                const float score = partial[query * key_batch + key] * scale;
                const float new_max = fmaxf(row_max[query], score);
                const float old_scale = row_max[query] == -3.4028234663852886e38f
                    ? 0.0f
                    : expf(row_max[query] - new_max);
                const float probability = expf(score - new_max);
                acc[query] = acc[query] * old_scale + probability * value;
                row_sum[query] =
                    row_sum[query] * old_scale + probability;
                row_max[query] = new_max;
            }
        }
        __syncthreads();
    }
    for (; j < end; ++j) {
        const float kv = k[j * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot[query_tile];
#pragma unroll
        for (unsigned query = 0u; query < query_tile; ++query) {
            dot[query] = qv[query] * kv;
        }
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
#pragma unroll
            for (unsigned query = 0u; query < query_tile; ++query) {
                dot[query] +=
                    __shfl_down_sync(0xffffffffu, dot[query], offset);
            }
        }
        if (lane == 0u) {
#pragma unroll
            for (unsigned query = 0u; query < query_tile; ++query) {
                partial[warp * query_tile + query] = dot[query];
            }
        }
        __syncthreads();
        if (tid < 8u * query_tile) {
            const unsigned query = tid >> 3;
            const unsigned warp_id = tid & 7u;
            float sum = partial[warp_id * query_tile + query];
#pragma unroll
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            if (warp_id == 0u) {
                partial[query] = sum;
            }
        }
        __syncthreads();
        const float value = v[j * num_kv_heads * 256u + kv_h * 256u + tid];
#pragma unroll
        for (unsigned query = 0u; query < query_tile; ++query) {
            const unsigned i = query_base + query;
            const unsigned global_pos = kv_len - seq_len + i;
            if (i >= seq_len || j > global_pos) {
                continue;
            }
            const float score = partial[query] * scale;
            const float new_max = fmaxf(row_max[query], score);
            const float old_scale = row_max[query] == -3.4028234663852886e38f
                ? 0.0f
                : expf(row_max[query] - new_max);
            const float probability = expf(score - new_max);
            acc[query] = acc[query] * old_scale + probability * value;
            row_sum[query] = row_sum[query] * old_scale + probability;
            row_max[query] = new_max;
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned query = 0u; query < query_tile; ++query) {
        const unsigned i = query_base + query;
        if (i >= seq_len) {
            continue;
        }
        const unsigned partial_row = (i * num_heads + h) * num_chunks + chunk;
        partial_acc[partial_row * 256u + tid] = acc[query];
        if (tid == 0u) {
            partial_meta[partial_row * 2u] = row_max[query];
            partial_meta[partial_row * 2u + 1u] = row_sum[query];
        }
    }
}

extern "C" __global__ void rnb_attention_prefill_flash_hd256_window_split_reduce(
    float* __restrict__ out,
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_meta,
    unsigned seq_len,
    unsigned num_heads,
    unsigned num_chunks) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 256u || i >= seq_len || h >= num_heads || num_chunks == 0u) {
        return;
    }

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;
    for (unsigned chunk = 0u; chunk < num_chunks; ++chunk) {
        const unsigned partial_row = (i * num_heads + h) * num_chunks + chunk;
        const float chunk_max = partial_meta[partial_row * 2u];
        const float chunk_sum = partial_meta[partial_row * 2u + 1u];
        if (chunk_sum <= 0.0f) {
            continue;
        }
        const float chunk_acc = partial_acc[partial_row * 256u + tid];
        const float new_max = fmaxf(row_max, chunk_max);
        const float old_scale =
            row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
        const float chunk_scale = expf(chunk_max - new_max);
        acc = acc * old_scale + chunk_acc * chunk_scale;
        row_sum = row_sum * old_scale + chunk_sum * chunk_scale;
        row_max = new_max;
    }
    out[i * num_heads * 256u + h * 256u + tid] =
        row_sum > 0.0f ? acc / row_sum : 0.0f;
}
extern "C" __global__ void rnb_attention_prefill_flash_hd256_window_jbatch8(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale,
    unsigned window) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 256u || i >= seq_len || h >= num_heads || num_kv_heads == 0u || window == 0u) {
        return;
    }

    __shared__ float partial[8 * 8];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned start = global_pos + 1u > window ? global_pos + 1u - window : 0u;
    const float qv = q[i * num_heads * 256u + h * 256u + tid];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    unsigned j = start;
    for (; j + 8u <= global_pos + 1u; j += 8u) {
        float dot0 = qv * k[(j + 0u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot1 = qv * k[(j + 1u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot2 = qv * k[(j + 2u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot3 = qv * k[(j + 3u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot4 = qv * k[(j + 4u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot5 = qv * k[(j + 5u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot6 = qv * k[(j + 6u) * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot7 = qv * k[(j + 7u) * num_kv_heads * 256u + kv_h * 256u + tid];
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot0 += __shfl_down_sync(0xffffffffu, dot0, offset);
            dot1 += __shfl_down_sync(0xffffffffu, dot1, offset);
            dot2 += __shfl_down_sync(0xffffffffu, dot2, offset);
            dot3 += __shfl_down_sync(0xffffffffu, dot3, offset);
            dot4 += __shfl_down_sync(0xffffffffu, dot4, offset);
            dot5 += __shfl_down_sync(0xffffffffu, dot5, offset);
            dot6 += __shfl_down_sync(0xffffffffu, dot6, offset);
            dot7 += __shfl_down_sync(0xffffffffu, dot7, offset);
        }
        if (lane == 0u) {
            partial[warp * 8u + 0u] = dot0;
            partial[warp * 8u + 1u] = dot1;
            partial[warp * 8u + 2u] = dot2;
            partial[warp * 8u + 3u] = dot3;
            partial[warp * 8u + 4u] = dot4;
            partial[warp * 8u + 5u] = dot5;
            partial[warp * 8u + 6u] = dot6;
            partial[warp * 8u + 7u] = dot7;
        }
        __syncthreads();
        if (tid < 64u) {
            const unsigned slot = tid >> 3;
            const unsigned warp_id = tid & 7u;
            float sum = partial[warp_id * 8u + slot];
#pragma unroll
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            if (warp_id == 0u) {
                partial[slot] = sum;
            }
        }
        __syncthreads();
        const float score0 = partial[0] * scale;
        const float score1 = partial[1] * scale;
        const float score2 = partial[2] * scale;
        const float score3 = partial[3] * scale;
        const float score4 = partial[4] * scale;
        const float score5 = partial[5] * scale;
        const float score6 = partial[6] * scale;
        const float score7 = partial[7] * scale;
        __syncthreads();

#pragma unroll
        for (unsigned batch = 0u; batch < 8u; ++batch) {
            const float score = batch == 0u   ? score0
                : batch == 1u                ? score1
                : batch == 2u                ? score2
                : batch == 3u                ? score3
                : batch == 4u                ? score4
                : batch == 5u                ? score5
                : batch == 6u                ? score6
                                             : score7;
            const float new_max = fmaxf(row_max, score);
            const float old_scale =
                row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
            const float probability = expf(score - new_max);
            const float value =
                v[(j + batch) * num_kv_heads * 256u + kv_h * 256u + tid];
            acc = acc * old_scale + probability * value;
            row_sum = row_sum * old_scale + probability;
            row_max = new_max;
        }
    }
    for (; j <= global_pos; ++j) {
        const float kv = k[j * num_kv_heads * 256u + kv_h * 256u + tid];
        float dot = qv * kv;
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 8u) {
            float sum = partial[tid];
#pragma unroll
            for (unsigned offset = 4u; offset > 0u; offset >>= 1u) {
                sum += __shfl_down_sync(0x000000ffu, sum, offset);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale =
            row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
        const float probability = expf(score - new_max);
        const float value = v[j * num_kv_heads * 256u + kv_h * 256u + tid];
        acc = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = new_max;
        __syncthreads();
    }

    out[i * num_heads * 256u + h * 256u + tid] = row_sum > 0.0f ? acc / row_sum : 0.0f;
}

// cu38 Phase 4: attention prefill flash hd512_w256 j-batch=4 변형 (Gemma E2B full
// attention layer). hd512 의 q0/q1 + k0/k1 + v0/v1 split + 4 j 동시 처리.
extern "C" __global__ void rnb_attention_prefill_flash_hd512_w256_jbatch4(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned seq_len,
    unsigned kv_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    float scale) {
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x;
    const unsigned h = blockIdx.y;
    if (tid >= 256u || i >= seq_len || h >= num_heads || num_kv_heads == 0u) {
        return;
    }

    __shared__ float partial[8 * 4];
    const unsigned heads_per_group = num_heads / num_kv_heads;
    const unsigned kv_h = h / heads_per_group;
    const unsigned global_pos = kv_len - seq_len + i;
    const unsigned q_base = i * num_heads * 512u + h * 512u;
    const float q0 = q[q_base + tid];
    const float q1 = q[q_base + tid + 256u];
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    unsigned j = 0u;
    for (; j + 4u <= global_pos + 1u; j += 4u) {
        float dot_arr[4];
        #pragma unroll
        for (unsigned sj = 0; sj < 4u; ++sj) {
            const unsigned kvb = (j + sj) * num_kv_heads * 512u + kv_h * 512u;
            dot_arr[sj] = q0 * k[kvb + tid] + q1 * k[kvb + tid + 256u];
            for (unsigned off = 16u; off > 0u; off >>= 1u) {
                dot_arr[sj] += __shfl_down_sync(0xffffffffu, dot_arr[sj], off);
            }
        }
        if (lane == 0u) {
            partial[warp * 4 + 0] = dot_arr[0];
            partial[warp * 4 + 1] = dot_arr[1];
            partial[warp * 4 + 2] = dot_arr[2];
            partial[warp * 4 + 3] = dot_arr[3];
        }
        __syncthreads();
        if (tid < 32u) {
            const unsigned slot = tid >> 3;
            const unsigned warp_id = tid & 7u;
            float sum = partial[warp_id * 4 + slot];
            for (unsigned off = 4u; off > 0u; off >>= 1u) {
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            if (warp_id == 0u) {
                partial[slot] = sum;
            }
        }
        __syncthreads();
        const float s0 = partial[0] * scale;
        const float s1 = partial[1] * scale;
        const float s2 = partial[2] * scale;
        const float s3 = partial[3] * scale;
        __syncthreads();
        #pragma unroll
        for (unsigned sj = 0; sj < 4u; ++sj) {
            const float score = (sj == 0) ? s0 : (sj == 1) ? s1 : (sj == 2) ? s2 : s3;
            const float new_max = fmaxf(row_max, score);
            const float old_scale = row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
            const float p = expf(score - new_max);
            const unsigned kvb = (j + sj) * num_kv_heads * 512u + kv_h * 512u;
            const float v0 = v[kvb + tid];
            const float v1 = v[kvb + tid + 256u];
            acc0 = acc0 * old_scale + p * v0;
            acc1 = acc1 * old_scale + p * v1;
            row_sum = row_sum * old_scale + p;
            row_max = new_max;
        }
    }
    // tail
    for (; j <= global_pos; ++j) {
        const unsigned kvb = j * num_kv_heads * 512u + kv_h * 512u;
        const float k0 = k[kvb + tid];
        const float k1 = k[kvb + tid + 256u];
        float dot = q0 * k0 + q1 * k1;
        for (unsigned off = 16u; off > 0u; off >>= 1u) {
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        }
        if (lane == 0u) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (tid < 8u) {
            float sum = partial[tid];
            for (unsigned off = 4u; off > 0u; off >>= 1u) {
                sum += __shfl_down_sync(0x000000ffu, sum, off);
            }
            if (tid == 0u) {
                partial[0] = sum;
            }
        }
        __syncthreads();
        const float score = partial[0] * scale;
        const float new_max = fmaxf(row_max, score);
        const float old_scale = row_max == -3.4028234663852886e38f ? 0.0f : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float v0 = v[kvb + tid];
        const float v1 = v[kvb + tid + 256u];
        acc0 = acc0 * old_scale + p * v0;
        acc1 = acc1 * old_scale + p * v1;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    const unsigned out_base = i * num_heads * 512u + h * 512u;
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    out[out_base + tid] = acc0 * inv_sum;
    out[out_base + tid + 256u] = acc1 * inv_sum;
}
