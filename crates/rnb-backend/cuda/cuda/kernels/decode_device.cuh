// cu63: Device-resident decode kernels.
// Generic RoPE decode + f32→f16 KV cache write.

// RoPE NeoX for decode (single token, any head_dim).
// Applies rotary position encoding in-place to Q and K.
// grid = (num_heads + num_kv_heads, 1, 1)
// block = (256, 1, 1)
// Each block handles one head (head_dim dims, threads handle half pairs).
extern "C" __global__ void rnb_rope_neox_decode(
    float* __restrict__ q,
    float* __restrict__ k,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned head_dim,
    float theta_base,
    unsigned pos) {

    const unsigned head_idx = blockIdx.x;
    const bool is_k = (head_idx >= num_heads);
    float* data = is_k ? (k + (head_idx - num_heads) * head_dim) : (q + head_idx * head_dim);

    const unsigned half = head_dim / 2;
    for (unsigned i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        float r = data[i];
        float im = data[i + half];
        data[i]        = r * cos_val - im * sin_val;
        data[i + half] = r * sin_val + im * cos_val;
    }
}

// RoPE NeoX variant for reusable CUDA graphs.
// Reads pos from device memory so each replay can update the position without recapture.
extern "C" __global__ void rnb_rope_neox_decode_pos_dev(
    float* __restrict__ q,
    float* __restrict__ k,
    unsigned num_heads,
    unsigned num_kv_heads,
    unsigned head_dim,
    float theta_base,
    const unsigned* __restrict__ pos_dev) {

    const unsigned pos = *pos_dev;
    const unsigned head_idx = blockIdx.x;
    const bool is_k = (head_idx >= num_heads);
    float* data = is_k ? (k + (head_idx - num_heads) * head_dim) : (q + head_idx * head_dim);

    const unsigned half = head_dim / 2;
    for (unsigned i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        float r = data[i];
        float im = data[i + half];
        data[i]        = r * cos_val - im * sin_val;
        data[i + half] = r * sin_val + im * cos_val;
    }
}

// f32 → f16 conversion + KV cache write (single token).
// Converts dim f32 values to f16 and writes them at position pos in the cache.
// grid = (1, 1, 1), block = (256, 1, 1)
extern "C" __global__ void rnb_f32_to_f16_kv_write(
    __half* __restrict__ kv_cache,
    const float* __restrict__ src,
    unsigned dim,
    unsigned pos,
    unsigned max_seq) {
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        kv_cache[(unsigned long long)pos * dim + i] = __float2half_rn(src[i]);
    }
}
