extern "C" __global__ void rnb_qk_norm_rope_neox_hd512_f16kv(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    float* __restrict__ q_out,
    unsigned short* __restrict__ k_out,
    unsigned short* __restrict__ v_out,
    float eps,
    unsigned seq_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned pos_start,
    unsigned q_unit_offset,
    unsigned k_unit_offset,
    unsigned v_no_scale_norm) {
    const unsigned tid = threadIdx.x;
    const unsigned token = blockIdx.x;
    const unsigned head_slot = blockIdx.y;
    if (tid >= 256u || token >= seq_len || head_slot >= num_heads + num_kv_heads) {
        return;
    }

    __shared__ float partial[256];
    constexpr unsigned head_dim = 512u;
    constexpr unsigned half = 256u;

    if (head_slot < num_heads) {
        const unsigned base = (token * num_heads + head_slot) * head_dim;
        const float x0 = q_in[base + tid];
        const float x1 = q_in[base + half + tid];
        partial[tid] = x0 * x0 + x1 * x1;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        const float s0 = q_unit_offset != 0u ? 1.0f + q_norm[tid] : q_norm[tid];
        const float s1 = q_unit_offset != 0u ? 1.0f + q_norm[half + tid] : q_norm[half + tid];
        const unsigned rope_idx = token * half + tid;
        const float sin_a = rope_sin[rope_idx];
        const float cos_a = rope_cos[rope_idx];
        const float y0 = x0 * inv_rms * s0;
        const float y1 = x1 * inv_rms * s1;
        q_out[base + tid] = y0 * cos_a - y1 * sin_a;
        q_out[base + half + tid] = y0 * sin_a + y1 * cos_a;
        return;
    }

    const unsigned kv_head = head_slot - num_heads;
    const unsigned base = (token * num_kv_heads + kv_head) * head_dim;
    const float k0 = k_in[base + tid];
    const float k1 = k_in[base + half + tid];
    partial[tid] = k0 * k0 + k1 * k1;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float k_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
    const float ks0 = k_unit_offset != 0u ? 1.0f + k_norm[tid] : k_norm[tid];
    const float ks1 = k_unit_offset != 0u ? 1.0f + k_norm[half + tid] : k_norm[half + tid];
    const unsigned rope_idx = token * half + tid;
    const float sin_a = rope_sin[rope_idx];
    const float cos_a = rope_cos[rope_idx];
    const float ky0 = k0 * k_inv_rms * ks0;
    const float ky1 = k1 * k_inv_rms * ks1;
    k_out[base + tid] = __half_as_ushort(__float2half_rn(ky0 * cos_a - ky1 * sin_a));
    k_out[base + half + tid] = __half_as_ushort(__float2half_rn(ky0 * sin_a + ky1 * cos_a));

    const float v0 = v_in[base + tid];
    const float v1 = v_in[base + half + tid];
    if (v_no_scale_norm != 0u) {
        partial[tid] = v0 * v0 + v1 * v1;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float v_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0 * v_inv_rms));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1 * v_inv_rms));
    } else {
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1));
    }
}

extern "C" __global__ void rnb_q_norm_rope_neox_hd512(
    const float* __restrict__ q_in,
    const float* __restrict__ q_norm,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    float* __restrict__ q_out,
    float eps,
    unsigned seq_len,
    unsigned num_heads,
    unsigned pos_start,
    unsigned q_unit_offset) {
    const unsigned tid = threadIdx.x;
    const unsigned token = blockIdx.x;
    const unsigned head = blockIdx.y;
    if (tid >= 256u || token >= seq_len || head >= num_heads) {
        return;
    }

    __shared__ float partial[256];
    constexpr unsigned head_dim = 512u;
    constexpr unsigned half = 256u;

    const unsigned base = (token * num_heads + head) * head_dim;
    const float x0 = q_in[base + tid];
    const float x1 = q_in[base + half + tid];
    partial[tid] = x0 * x0 + x1 * x1;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
    const float s0 = q_unit_offset != 0u ? 1.0f + q_norm[tid] : q_norm[tid];
    const float s1 = q_unit_offset != 0u ? 1.0f + q_norm[half + tid] : q_norm[half + tid];
    const unsigned rope_idx = token * half + tid;
    const float sin_a = rope_sin[rope_idx];
    const float cos_a = rope_cos[rope_idx];
    const float y0 = x0 * inv_rms * s0;
    const float y1 = x1 * inv_rms * s1;
    q_out[base + tid] = y0 * cos_a - y1 * sin_a;
    q_out[base + half + tid] = y0 * sin_a + y1 * cos_a;
}

extern "C" __global__ void rnb_q_norm_rope_neox_hd256(
    const float* __restrict__ q_in,
    const float* __restrict__ q_norm,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    float* __restrict__ q_out,
    float eps,
    unsigned seq_len,
    unsigned num_heads,
    unsigned pos_start,
    unsigned q_unit_offset) {
    const unsigned tid = threadIdx.x;
    const unsigned token = blockIdx.x;
    const unsigned head = blockIdx.y;
    if (tid >= 128u || token >= seq_len || head >= num_heads) {
        return;
    }

    __shared__ float partial[128];
    constexpr unsigned head_dim = 256u;
    constexpr unsigned half = 128u;

    const unsigned base = (token * num_heads + head) * head_dim;
    const float x0 = q_in[base + tid];
    const float x1 = q_in[base + half + tid];
    partial[tid] = x0 * x0 + x1 * x1;
    __syncthreads();
    for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
    const float s0 = q_unit_offset != 0u ? 1.0f + q_norm[tid] : q_norm[tid];
    const float s1 = q_unit_offset != 0u ? 1.0f + q_norm[half + tid] : q_norm[half + tid];
    const unsigned rope_idx = token * half + tid;
    const float sin_a = rope_sin[rope_idx];
    const float cos_a = rope_cos[rope_idx];
    const float y0 = x0 * inv_rms * s0;
    const float y1 = x1 * inv_rms * s1;
    q_out[base + tid] = y0 * cos_a - y1 * sin_a;
    q_out[base + half + tid] = y0 * sin_a + y1 * cos_a;
}

extern "C" __global__ void rnb_split_gated_attention_q_f32(
    const float* __restrict__ q_full,
    float* __restrict__ q_out,
    float* __restrict__ gate_out,
    unsigned seq_len,
    unsigned num_heads,
    unsigned head_dim) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned total = seq_len * num_heads * head_dim;
    if (i >= total) {
        return;
    }
    const unsigned d = i % head_dim;
    const unsigned h = (i / head_dim) % num_heads;
    const unsigned t = i / (num_heads * head_dim);
    const unsigned src = (t * num_heads + h) * head_dim * 2u + d;
    q_out[i] = q_full[src];
    gate_out[i] = q_full[src + head_dim];
}

extern "C" __global__ void rnb_sigmoid_mul_inplace(
    float* __restrict__ values,
    const float* __restrict__ gate,
    unsigned len) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    values[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

extern "C" __global__ void rnb_qk_norm_rope_neox_hd256_f16kv(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    float* __restrict__ q_out,
    unsigned short* __restrict__ k_out,
    unsigned short* __restrict__ v_out,
    float eps,
    unsigned seq_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned pos_start,
    unsigned q_unit_offset,
    unsigned k_unit_offset,
    unsigned v_no_scale_norm) {
    const unsigned tid = threadIdx.x;
    const unsigned token = blockIdx.x;
    const unsigned head_slot = blockIdx.y;
    if (tid >= 128u || token >= seq_len || head_slot >= num_heads + num_kv_heads) {
        return;
    }

    __shared__ float partial[128];
    constexpr unsigned head_dim = 256u;
    constexpr unsigned half = 128u;

    if (head_slot < num_heads) {
        const unsigned base = (token * num_heads + head_slot) * head_dim;
        const float x0 = q_in[base + tid];
        const float x1 = q_in[base + half + tid];
        partial[tid] = x0 * x0 + x1 * x1;
        __syncthreads();
        for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        const float s0 = q_unit_offset != 0u ? 1.0f + q_norm[tid] : q_norm[tid];
        const float s1 = q_unit_offset != 0u ? 1.0f + q_norm[half + tid] : q_norm[half + tid];
        const unsigned rope_idx = token * half + tid;
        const float sin_a = rope_sin[rope_idx];
        const float cos_a = rope_cos[rope_idx];
        const float y0 = x0 * inv_rms * s0;
        const float y1 = x1 * inv_rms * s1;
        q_out[base + tid] = y0 * cos_a - y1 * sin_a;
        q_out[base + half + tid] = y0 * sin_a + y1 * cos_a;
        return;
    }

    const unsigned kv_head = head_slot - num_heads;
    const unsigned base = (token * num_kv_heads + kv_head) * head_dim;
    const float k0 = k_in[base + tid];
    const float k1 = k_in[base + half + tid];
    partial[tid] = k0 * k0 + k1 * k1;
    __syncthreads();
    for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float k_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
    const float ks0 = k_unit_offset != 0u ? 1.0f + k_norm[tid] : k_norm[tid];
    const float ks1 = k_unit_offset != 0u ? 1.0f + k_norm[half + tid] : k_norm[half + tid];
    const unsigned rope_idx = token * half + tid;
    const float sin_a = rope_sin[rope_idx];
    const float cos_a = rope_cos[rope_idx];
    const float ky0 = k0 * k_inv_rms * ks0;
    const float ky1 = k1 * k_inv_rms * ks1;
    k_out[base + tid] = __half_as_ushort(__float2half_rn(ky0 * cos_a - ky1 * sin_a));
    k_out[base + half + tid] = __half_as_ushort(__float2half_rn(ky0 * sin_a + ky1 * cos_a));

    const float v0 = v_in[base + tid];
    const float v1 = v_in[base + half + tid];
    if (v_no_scale_norm != 0u) {
        partial[tid] = v0 * v0 + v1 * v1;
        __syncthreads();
        for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float v_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0 * v_inv_rms));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1 * v_inv_rms));
    } else {
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1));
    }
}

__device__ __forceinline__ float rnb_qk_norm_rope_apply_dim(
    const float* __restrict__ input,
    const float* __restrict__ norm,
    float inv_rms,
    unsigned dim,
    unsigned head_dim,
    unsigned rope_dim,
    unsigned rope_neox,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    unsigned token,
    unsigned unit_offset) {
    if (rope_dim == 0u) {
        rope_dim = head_dim;
    }
    rope_dim = rope_dim > head_dim ? head_dim : rope_dim;
    if (dim >= rope_dim) {
        const float scale = unit_offset != 0u ? 1.0f + norm[dim] : norm[dim];
        return input[dim] * inv_rms * scale;
    }

    unsigned first_dim;
    unsigned second_dim;
    unsigned pair_idx;
    bool first_output;
    if (rope_neox != 0u) {
        const unsigned half_rot = rope_dim >> 1u;
        if (dim < half_rot) {
            first_dim = dim;
            second_dim = half_rot + dim;
            pair_idx = dim;
            first_output = true;
        } else {
            pair_idx = dim - half_rot;
            first_dim = pair_idx;
            second_dim = dim;
            first_output = false;
        }
    } else {
        first_dim = dim & ~1u;
        second_dim = first_dim + 1u;
        pair_idx = first_dim >> 1u;
        first_output = (dim == first_dim);
    }

    const float scale0 = unit_offset != 0u ? 1.0f + norm[first_dim] : norm[first_dim];
    const float scale1 = unit_offset != 0u ? 1.0f + norm[second_dim] : norm[second_dim];
    const float x0 = input[first_dim] * inv_rms * scale0;
    const float x1 = input[second_dim] * inv_rms * scale1;
    const unsigned rope_idx = token * (rope_dim >> 1u) + pair_idx;
    const float sin_a = rope_sin[rope_idx];
    const float cos_a = rope_cos[rope_idx];
    return first_output ? (x0 * cos_a - x1 * sin_a) : (x0 * sin_a + x1 * cos_a);
}

extern "C" __global__ void rnb_qk_norm_rope_select_hd256_f16kv(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    float* __restrict__ q_out,
    unsigned short* __restrict__ k_out,
    unsigned short* __restrict__ v_out,
    float eps,
    unsigned seq_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned pos_start,
    unsigned q_unit_offset,
    unsigned k_unit_offset,
    unsigned v_no_scale_norm,
    unsigned rope_dim,
    unsigned rope_neox) {
    const unsigned tid = threadIdx.x;
    const unsigned token = blockIdx.x;
    const unsigned head_slot = blockIdx.y;
    if (tid >= 128u || token >= seq_len || head_slot >= num_heads + num_kv_heads) {
        return;
    }

    __shared__ float partial[128];
    constexpr unsigned head_dim = 256u;
    constexpr unsigned half = 128u;

    if (head_slot < num_heads) {
        const unsigned base = (token * num_heads + head_slot) * head_dim;
        const float x0 = q_in[base + tid];
        const float x1 = q_in[base + half + tid];
        partial[tid] = x0 * x0 + x1 * x1;
        __syncthreads();
        for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        const float* head = q_in + base;
        q_out[base + tid] = rnb_qk_norm_rope_apply_dim(
            head, q_norm, inv_rms, tid, head_dim, rope_dim, rope_neox, rope_sin, rope_cos,
            token, q_unit_offset);
        q_out[base + half + tid] = rnb_qk_norm_rope_apply_dim(
            head, q_norm, inv_rms, half + tid, head_dim, rope_dim, rope_neox, rope_sin,
            rope_cos, token, q_unit_offset);
        return;
    }

    const unsigned kv_head = head_slot - num_heads;
    const unsigned base = (token * num_kv_heads + kv_head) * head_dim;
    const float k0 = k_in[base + tid];
    const float k1 = k_in[base + half + tid];
    partial[tid] = k0 * k0 + k1 * k1;
    __syncthreads();
    for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float k_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
    const float* k_head = k_in + base;
    const float ky0 = rnb_qk_norm_rope_apply_dim(
        k_head, k_norm, k_inv_rms, tid, head_dim, rope_dim, rope_neox, rope_sin, rope_cos,
        token, k_unit_offset);
    const float ky1 = rnb_qk_norm_rope_apply_dim(
        k_head, k_norm, k_inv_rms, half + tid, head_dim, rope_dim, rope_neox, rope_sin,
        rope_cos, token, k_unit_offset);
    k_out[base + tid] = __half_as_ushort(__float2half_rn(ky0));
    k_out[base + half + tid] = __half_as_ushort(__float2half_rn(ky1));

    const float v0 = v_in[base + tid];
    const float v1 = v_in[base + half + tid];
    if (v_no_scale_norm != 0u) {
        partial[tid] = v0 * v0 + v1 * v1;
        __syncthreads();
        for (unsigned stride = 64u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float v_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0 * v_inv_rms));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1 * v_inv_rms));
    } else {
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1));
    }
}

extern "C" __global__ void rnb_qk_norm_rope_select_hd512_f16kv(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    float* __restrict__ q_out,
    unsigned short* __restrict__ k_out,
    unsigned short* __restrict__ v_out,
    float eps,
    unsigned seq_len,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned pos_start,
    unsigned q_unit_offset,
    unsigned k_unit_offset,
    unsigned v_no_scale_norm,
    unsigned rope_dim,
    unsigned rope_neox) {
    const unsigned tid = threadIdx.x;
    const unsigned token = blockIdx.x;
    const unsigned head_slot = blockIdx.y;
    if (tid >= 256u || token >= seq_len || head_slot >= num_heads + num_kv_heads) {
        return;
    }

    __shared__ float partial[256];
    constexpr unsigned head_dim = 512u;
    constexpr unsigned half = 256u;

    if (head_slot < num_heads) {
        const unsigned base = (token * num_heads + head_slot) * head_dim;
        const float x0 = q_in[base + tid];
        const float x1 = q_in[base + half + tid];
        partial[tid] = x0 * x0 + x1 * x1;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        const float* head = q_in + base;
        q_out[base + tid] = rnb_qk_norm_rope_apply_dim(
            head, q_norm, inv_rms, tid, head_dim, rope_dim, rope_neox, rope_sin, rope_cos,
            token, q_unit_offset);
        q_out[base + half + tid] = rnb_qk_norm_rope_apply_dim(
            head, q_norm, inv_rms, half + tid, head_dim, rope_dim, rope_neox, rope_sin,
            rope_cos, token, q_unit_offset);
        return;
    }

    const unsigned kv_head = head_slot - num_heads;
    const unsigned base = (token * num_kv_heads + kv_head) * head_dim;
    const float k0 = k_in[base + tid];
    const float k1 = k_in[base + half + tid];
    partial[tid] = k0 * k0 + k1 * k1;
    __syncthreads();
    for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float k_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
    const float* k_head = k_in + base;
    const float ky0 = rnb_qk_norm_rope_apply_dim(
        k_head, k_norm, k_inv_rms, tid, head_dim, rope_dim, rope_neox, rope_sin, rope_cos,
        token, k_unit_offset);
    const float ky1 = rnb_qk_norm_rope_apply_dim(
        k_head, k_norm, k_inv_rms, half + tid, head_dim, rope_dim, rope_neox, rope_sin,
        rope_cos, token, k_unit_offset);
    k_out[base + tid] = __half_as_ushort(__float2half_rn(ky0));
    k_out[base + half + tid] = __half_as_ushort(__float2half_rn(ky1));

    const float v0 = v_in[base + tid];
    const float v1 = v_in[base + half + tid];
    if (v_no_scale_norm != 0u) {
        partial[tid] = v0 * v0 + v1 * v1;
        __syncthreads();
        for (unsigned stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        const float v_inv_rms = rsqrtf(partial[0] / (float)head_dim + eps);
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0 * v_inv_rms));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1 * v_inv_rms));
    } else {
        v_out[base + tid] = __half_as_ushort(__float2half_rn(v0));
        v_out[base + half + tid] = __half_as_ushort(__float2half_rn(v1));
    }
}

// cu28: Llama / Mistral hd=128 path 용 RoPE-only QKV kernel. qk-norm 없이
// rope_neox만 Q에 적용, K에 적용 + f16 pack, V는 f16 pack only. Q는 attention
// input 으로 f32 그대로, K/V는 KV cache 들어가니까 f16 bits.
// grid = (seq_len, num_heads + num_kv_heads, 1), block = (64, 1, 1).
extern "C" __global__ void rnb_qk_rope_neox_hd128_f16kv(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ rope_sin,
    const float* __restrict__ rope_cos,
    float* __restrict__ q_out,
    unsigned short* __restrict__ k_out,
    unsigned short* __restrict__ v_out,
    unsigned seq_len,
    unsigned num_heads,
    unsigned num_kv_heads) {
    const unsigned tid = threadIdx.x;
    const unsigned token = blockIdx.x;
    const unsigned head_slot = blockIdx.y;
    if (tid >= 64u || token >= seq_len || head_slot >= num_heads + num_kv_heads) {
        return;
    }

    constexpr unsigned head_dim = 128u;
    constexpr unsigned half = 64u;
    // cu29: Llama 는 non-neox (GPT-J style) — adjacent pair (2i, 2i+1). sin/cos
    // table layout 은 rope_table_ptrs ([seq_len, half]) 그대로 사용 (pair_idx = tid).
    const unsigned rope_idx = token * half + tid;
    const float sin_a = rope_sin[rope_idx];
    const float cos_a = rope_cos[rope_idx];
    const unsigned pair0 = tid * 2u;        // 0, 2, 4, ..., 126
    const unsigned pair1 = pair0 + 1u;      // 1, 3, 5, ..., 127

    if (head_slot < num_heads) {
        const unsigned base = (token * num_heads + head_slot) * head_dim;
        const float x0 = q_in[base + pair0];
        const float x1 = q_in[base + pair1];
        q_out[base + pair0] = x0 * cos_a - x1 * sin_a;
        q_out[base + pair1] = x0 * sin_a + x1 * cos_a;
        return;
    }

    const unsigned kv_head = head_slot - num_heads;
    const unsigned base = (token * num_kv_heads + kv_head) * head_dim;
    const float k0 = k_in[base + pair0];
    const float k1 = k_in[base + pair1];
    k_out[base + pair0] = __half_as_ushort(__float2half_rn(k0 * cos_a - k1 * sin_a));
    k_out[base + pair1] = __half_as_ushort(__float2half_rn(k0 * sin_a + k1 * cos_a));

    v_out[base + pair0] = __half_as_ushort(__float2half_rn(v_in[base + pair0]));
    v_out[base + pair1] = __half_as_ushort(__float2half_rn(v_in[base + pair1]));
}
