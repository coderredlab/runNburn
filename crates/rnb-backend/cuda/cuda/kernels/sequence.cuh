extern "C" __global__ void rnb_delta_net_decode(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    unsigned num_heads,
    unsigned head_k_dim,
    unsigned head_v_dim) {
    const unsigned vi = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (h >= num_heads || vi >= head_v_dim || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    const unsigned state_size = head_k_dim * head_v_dim;
    const unsigned state_off = h * state_size + vi * head_k_dim;
    const unsigned qk_off = h * head_k_dim;
    const unsigned v_off = h * head_v_dim + vi;
    const float decay = expf(gate[h]);

    float sk = 0.0f;
    if (tid < head_k_dim) {
        sk = decay * state[state_off + tid] * k[qk_off + tid];
    }
    partial[tid] = sk;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float d = (v[v_off] - partial[0]) * beta[h];
    float updated = 0.0f;
    if (tid < head_k_dim) {
        const unsigned idx = state_off + tid;
        updated = decay * state[idx] + k[qk_off + tid] * d;
        state[idx] = updated;
    }
    __syncthreads();

    float sq = 0.0f;
    if (tid < head_k_dim) {
        sq = updated * q[qk_off + tid];
    }
    partial[tid] = sq;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[v_off] = partial[0];
    }
}

extern "C" __global__ void rnb_nemotron_mamba2_decode_scan(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ b,
    const float* __restrict__ c,
    const float* __restrict__ dt,
    const float* __restrict__ a,
    const float* __restrict__ d,
    unsigned num_heads,
    unsigned head_dim,
    unsigned state_dim,
    unsigned n_group) {
    const unsigned p = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned s = threadIdx.x;
    if (h >= num_heads || p >= head_dim || s >= 256u) {
        return;
    }

    __shared__ float partial[256];
    const unsigned x_idx = h * head_dim + p;
    const unsigned heads_per_group = num_heads / n_group;
    const unsigned group = h / heads_per_group;
    const unsigned bc_off = group * state_dim;
    const unsigned state_off = h * head_dim * state_dim + p * state_dim;
    const float xv = x[x_idx];
    const float dtv = dt[h];
    const float decay = expf(dtv * a[h]);
    const float x_dt = xv * dtv;

    float contrib = 0.0f;
    if (s < state_dim) {
        const unsigned idx = state_off + s;
        const float updated = state[idx] * decay + b[bc_off + s] * x_dt;
        state[idx] = updated;
        contrib = updated * c[bc_off + s];
    }
    partial[s] = contrib;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (s < stride) {
            partial[s] += partial[s + stride];
        }
        __syncthreads();
    }
    if (s == 0) {
        output[x_idx] = d[h] * xv + partial[0];
    }
}

extern "C" __global__ void rnb_nemotron_mamba2_prefill_scan(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ conv,
    const float* __restrict__ dt_data,
    const float* __restrict__ a,
    const float* __restrict__ d,
    unsigned seq_len,
    unsigned conv_channels,
    unsigned bc_dim,
    unsigned num_heads,
    unsigned head_dim,
    unsigned state_dim,
    unsigned n_group) {
    const unsigned p = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned s = threadIdx.x;
    if (h >= num_heads || p >= head_dim || s >= 256u) {
        return;
    }

    __shared__ float partial[256];
    const unsigned d_inner = num_heads * head_dim;
    const unsigned x_idx = h * head_dim + p;
    const unsigned heads_per_group = num_heads / n_group;
    const unsigned group = h / heads_per_group;
    const unsigned bc_off = group * state_dim;
    const unsigned state_off = h * head_dim * state_dim + p * state_dim;
    const float av = a[h];
    const float dv = d[h];

    for (unsigned t = 0; t < seq_len; ++t) {
        const float* token = conv + t * conv_channels;
        const float xv = token[x_idx];
        const float dt_raw = dt_data[t * num_heads + h];
        const float dtv = dt_raw > 20.0f ? dt_raw : logf(1.0f + expf(dt_raw));
        const float decay = expf(dtv * av);
        const float x_dt = xv * dtv;
        float contrib = 0.0f;
        if (s < state_dim) {
            const unsigned idx = state_off + s;
            const float updated = state[idx] * decay + token[d_inner + bc_off + s] * x_dt;
            state[idx] = updated;
            contrib = updated * token[d_inner + bc_dim + bc_off + s];
        }
        partial[s] = contrib;
        __syncthreads();
        for (unsigned stride = 128; stride > 0; stride >>= 1) {
            if (s < stride) {
                partial[s] += partial[s + stride];
            }
            __syncthreads();
        }
        if (s == 0u) {
            output[t * d_inner + x_idx] = dv * xv + partial[0];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void rnb_delta_snapshot_store_multi(
    const unsigned long long* __restrict__ snapshot_states,
    const unsigned* __restrict__ snapshot_after_tokens,
    unsigned snapshot_count,
    unsigned token_index,
    unsigned state_index,
    float value) {
    if (snapshot_states == nullptr || snapshot_after_tokens == nullptr) {
        return;
    }
    for (unsigned s = 0; s < snapshot_count; ++s) {
        if (snapshot_after_tokens[s] == token_index) {
            float* snapshot_state = reinterpret_cast<float*>(snapshot_states[s]);
            snapshot_state[state_index] = value;
        }
    }
}

extern "C" __global__ void rnb_delta_net_prefill(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ snapshot_state,
    unsigned seq_len,
    unsigned num_heads,
    unsigned head_k_dim,
    unsigned head_v_dim,
    unsigned snapshot_after_tokens) {
    const unsigned vi = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (h >= num_heads || vi >= head_v_dim || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    const unsigned state_size = head_k_dim * head_v_dim;
    const unsigned state_off = h * state_size + vi * head_k_dim;

    for (unsigned t = 0; t < seq_len; ++t) {
        const unsigned qk_off = (t * num_heads + h) * head_k_dim;
        const unsigned v_off = (t * num_heads + h) * head_v_dim + vi;
        const unsigned gate_off = t * num_heads + h;
        const float decay = expf(gate[gate_off]);

        float sk = 0.0f;
        if (tid < head_k_dim) {
            sk = decay * state[state_off + tid] * k[qk_off + tid];
        }
        partial[tid] = sk;
        __syncthreads();
        for (unsigned stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        const float d = (v[v_off] - partial[0]) * beta[gate_off];
        float updated = 0.0f;
        if (tid < head_k_dim) {
            const unsigned idx = state_off + tid;
            updated = decay * state[idx] + k[qk_off + tid] * d;
            state[idx] = updated;
            if (snapshot_state != nullptr && snapshot_after_tokens == t + 1u) {
                snapshot_state[idx] = updated;
            }
        }
        __syncthreads();

        float sq = 0.0f;
        if (tid < head_k_dim) {
            sq = updated * q[qk_off + tid];
        }
        partial[tid] = sq;
        __syncthreads();
        for (unsigned stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[v_off] = partial[0];
        }
        __syncthreads();
    }
}

extern "C" __global__ void rnb_delta_net_prefill_multi_snapshot(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const unsigned long long* __restrict__ snapshot_states,
    const unsigned* __restrict__ snapshot_after_tokens,
    unsigned snapshot_count,
    unsigned seq_len,
    unsigned num_heads,
    unsigned head_k_dim,
    unsigned head_v_dim) {
    const unsigned vi = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (h >= num_heads || vi >= head_v_dim || tid >= 256) {
        return;
    }

    __shared__ float partial[256];
    const unsigned state_size = head_k_dim * head_v_dim;
    const unsigned state_off = h * state_size + vi * head_k_dim;

    for (unsigned t = 0; t < seq_len; ++t) {
        const unsigned qk_off = (t * num_heads + h) * head_k_dim;
        const unsigned v_off = (t * num_heads + h) * head_v_dim + vi;
        const unsigned gate_off = t * num_heads + h;
        const float decay = expf(gate[gate_off]);

        float sk = 0.0f;
        if (tid < head_k_dim) {
            sk = decay * state[state_off + tid] * k[qk_off + tid];
        }
        partial[tid] = sk;
        __syncthreads();
        for (unsigned stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        const float d = (v[v_off] - partial[0]) * beta[gate_off];
        float updated = 0.0f;
        if (tid < head_k_dim) {
            const unsigned idx = state_off + tid;
            updated = decay * state[idx] + k[qk_off + tid] * d;
            state[idx] = updated;
            rnb_delta_snapshot_store_multi(
                snapshot_states,
                snapshot_after_tokens,
                snapshot_count,
                t + 1u,
                idx,
                updated);
        }
        __syncthreads();

        float sq = 0.0f;
        if (tid < head_k_dim) {
            sq = updated * q[qk_off + tid];
        }
        partial[tid] = sq;
        __syncthreads();
        for (unsigned stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[v_off] = partial[0];
        }
        __syncthreads();
    }
}

extern "C" __global__ void rnb_delta_net_prefill_k128(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ snapshot_state,
    unsigned seq_len,
    unsigned num_heads,
    unsigned head_v_dim,
    unsigned snapshot_after_tokens) {
    const unsigned vi = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (h >= num_heads || vi >= head_v_dim || tid >= 128) {
        return;
    }

    __shared__ float partial[128];
    const unsigned head_k_dim = 128u;
    const unsigned state_size = head_k_dim * head_v_dim;
    const unsigned state_off = h * state_size + vi * head_k_dim;

    for (unsigned t = 0; t < seq_len; ++t) {
        const unsigned qk_off = (t * num_heads + h) * head_k_dim;
        const unsigned v_off = (t * num_heads + h) * head_v_dim + vi;
        const unsigned gate_off = t * num_heads + h;
        const float decay = expf(gate[gate_off]);

        partial[tid] = decay * state[state_off + tid] * k[qk_off + tid];
        __syncthreads();
        for (unsigned stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        const float d = (v[v_off] - partial[0]) * beta[gate_off];
        const unsigned idx = state_off + tid;
        const float updated = decay * state[idx] + k[qk_off + tid] * d;
        state[idx] = updated;
        if (snapshot_state != nullptr && snapshot_after_tokens == t + 1u) {
            snapshot_state[idx] = updated;
        }
        __syncthreads();

        partial[tid] = updated * q[qk_off + tid];
        __syncthreads();
        for (unsigned stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[v_off] = partial[0];
        }
        __syncthreads();
    }
}

extern "C" __global__ void rnb_delta_net_prefill_k128_multi_snapshot(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const unsigned long long* __restrict__ snapshot_states,
    const unsigned* __restrict__ snapshot_after_tokens,
    unsigned snapshot_count,
    unsigned seq_len,
    unsigned num_heads,
    unsigned head_v_dim) {
    const unsigned vi = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (h >= num_heads || vi >= head_v_dim || tid >= 128) {
        return;
    }

    __shared__ float partial[128];
    const unsigned head_k_dim = 128u;
    const unsigned state_size = head_k_dim * head_v_dim;
    const unsigned state_off = h * state_size + vi * head_k_dim;

    for (unsigned t = 0; t < seq_len; ++t) {
        const unsigned qk_off = (t * num_heads + h) * head_k_dim;
        const unsigned v_off = (t * num_heads + h) * head_v_dim + vi;
        const unsigned gate_off = t * num_heads + h;
        const float decay = expf(gate[gate_off]);

        partial[tid] = decay * state[state_off + tid] * k[qk_off + tid];
        __syncthreads();
        for (unsigned stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        const float d = (v[v_off] - partial[0]) * beta[gate_off];
        const unsigned idx = state_off + tid;
        const float updated = decay * state[idx] + k[qk_off + tid] * d;
        state[idx] = updated;
        rnb_delta_snapshot_store_multi(
            snapshot_states,
            snapshot_after_tokens,
            snapshot_count,
            t + 1u,
            idx,
            updated);
        __syncthreads();

        partial[tid] = updated * q[qk_off + tid];
        __syncthreads();
        for (unsigned stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[v_off] = partial[0];
        }
        __syncthreads();
    }
}

extern "C" __global__ void rnb_delta_net_prefill_k128_warp4(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ snapshot_state,
    unsigned seq_len,
    unsigned num_heads,
    unsigned head_v_dim,
    unsigned snapshot_after_tokens) {
    const unsigned lane = threadIdx.x;
    const unsigned col = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned h = blockIdx.y;
    if (h >= num_heads || col >= head_v_dim || lane >= 32u) {
        return;
    }

    const unsigned head_k_dim = 128u;
    const unsigned state_off = h * head_v_dim * head_k_dim + col * head_k_dim;
    float s0 = state[state_off + lane + 0u];
    float s1 = state[state_off + lane + 32u];
    float s2 = state[state_off + lane + 64u];
    float s3 = state[state_off + lane + 96u];

    for (unsigned t = 0; t < seq_len; ++t) {
        const unsigned qk_off = (t * num_heads + h) * head_k_dim;
        const unsigned v_off = (t * num_heads + h) * head_v_dim + col;
        const unsigned gate_off = t * num_heads + h;

        const float k0 = k[qk_off + lane + 0u];
        const float k1 = k[qk_off + lane + 32u];
        const float k2 = k[qk_off + lane + 64u];
        const float k3 = k[qk_off + lane + 96u];
        const float decay = expf(gate[gate_off]);

        float kv = decay * (s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
        for (int offset = 16; offset > 0; offset >>= 1) {
            kv += __shfl_down_sync(0xffffffffu, kv, offset);
        }
        kv = __shfl_sync(0xffffffffu, kv, 0);

        const float delta = (v[v_off] - kv) * beta[gate_off];
        s0 = decay * s0 + k0 * delta;
        s1 = decay * s1 + k1 * delta;
        s2 = decay * s2 + k2 * delta;
        s3 = decay * s3 + k3 * delta;
        if (snapshot_state != nullptr && snapshot_after_tokens == t + 1u) {
            snapshot_state[state_off + lane + 0u] = s0;
            snapshot_state[state_off + lane + 32u] = s1;
            snapshot_state[state_off + lane + 64u] = s2;
            snapshot_state[state_off + lane + 96u] = s3;
        }

        const float q0 = q[qk_off + lane + 0u];
        const float q1 = q[qk_off + lane + 32u];
        const float q2 = q[qk_off + lane + 64u];
        const float q3 = q[qk_off + lane + 96u];

        float attn = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        for (int offset = 16; offset > 0; offset >>= 1) {
            attn += __shfl_down_sync(0xffffffffu, attn, offset);
        }
        if (lane == 0u) {
            output[v_off] = attn;
        }
    }

    state[state_off + lane + 0u] = s0;
    state[state_off + lane + 32u] = s1;
    state[state_off + lane + 64u] = s2;
    state[state_off + lane + 96u] = s3;
}

extern "C" __global__ void rnb_delta_net_prefill_k128_warp4_multi_snapshot(
    float* __restrict__ output,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const unsigned long long* __restrict__ snapshot_states,
    const unsigned* __restrict__ snapshot_after_tokens,
    unsigned snapshot_count,
    unsigned seq_len,
    unsigned num_heads,
    unsigned head_v_dim) {
    const unsigned lane = threadIdx.x;
    const unsigned col = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned h = blockIdx.y;
    if (h >= num_heads || col >= head_v_dim || lane >= 32u) {
        return;
    }

    const unsigned head_k_dim = 128u;
    const unsigned state_off = h * head_v_dim * head_k_dim + col * head_k_dim;
    float s0 = state[state_off + lane + 0u];
    float s1 = state[state_off + lane + 32u];
    float s2 = state[state_off + lane + 64u];
    float s3 = state[state_off + lane + 96u];

    for (unsigned t = 0; t < seq_len; ++t) {
        const unsigned qk_off = (t * num_heads + h) * head_k_dim;
        const unsigned v_off = (t * num_heads + h) * head_v_dim + col;
        const unsigned gate_off = t * num_heads + h;

        const float k0 = k[qk_off + lane + 0u];
        const float k1 = k[qk_off + lane + 32u];
        const float k2 = k[qk_off + lane + 64u];
        const float k3 = k[qk_off + lane + 96u];
        const float decay = expf(gate[gate_off]);

        float kv = decay * (s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3);
        for (int offset = 16; offset > 0; offset >>= 1) {
            kv += __shfl_down_sync(0xffffffffu, kv, offset);
        }
        kv = __shfl_sync(0xffffffffu, kv, 0);

        const float delta = (v[v_off] - kv) * beta[gate_off];
        s0 = decay * s0 + k0 * delta;
        s1 = decay * s1 + k1 * delta;
        s2 = decay * s2 + k2 * delta;
        s3 = decay * s3 + k3 * delta;
        rnb_delta_snapshot_store_multi(
            snapshot_states,
            snapshot_after_tokens,
            snapshot_count,
            t + 1u,
            state_off + lane + 0u,
            s0);
        rnb_delta_snapshot_store_multi(
            snapshot_states,
            snapshot_after_tokens,
            snapshot_count,
            t + 1u,
            state_off + lane + 32u,
            s1);
        rnb_delta_snapshot_store_multi(
            snapshot_states,
            snapshot_after_tokens,
            snapshot_count,
            t + 1u,
            state_off + lane + 64u,
            s2);
        rnb_delta_snapshot_store_multi(
            snapshot_states,
            snapshot_after_tokens,
            snapshot_count,
            t + 1u,
            state_off + lane + 96u,
            s3);

        const float q0 = q[qk_off + lane + 0u];
        const float q1 = q[qk_off + lane + 32u];
        const float q2 = q[qk_off + lane + 64u];
        const float q3 = q[qk_off + lane + 96u];

        float attn = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        for (int offset = 16; offset > 0; offset >>= 1) {
            attn += __shfl_down_sync(0xffffffffu, attn, offset);
        }
        if (lane == 0u) {
            output[v_off] = attn;
        }
    }

    state[state_off + lane + 0u] = s0;
    state[state_off + lane + 32u] = s1;
    state[state_off + lane + 64u] = s2;
    state[state_off + lane + 96u] = s3;
}
