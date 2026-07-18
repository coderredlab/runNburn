#include <metal_stdlib>
using namespace metal;

// pm48 device-resident attention chain 선행 부품:
//   prefill(seq_len>1) q/k 를 device 에 머문 채 per-head RMSNorm(qk_norm) → text M-RoPE
//   순서로 적용. CPU ground-truth 순서(projection.rs: qk_norm 먼저, 그다음 forward/rope.rs
//   rope_mrope_text)와 1:1.
//
// 1 threadgroup = (token t, head h). grid = seq_len * num_heads (linear).
//   tg_lin = t * num_heads + h, base = tg_lin * head_dim.
// 256 lane tree reduction 으로 head 별 mean-square(RMSNorm) — 기존 qk_norm.metal 패턴.
//   reduction 후 normed 값을 threadgroup buffer 에 적재 → lane 0 이 angle 누적 순서대로
//   rope rotation(rope_mrope.metal 패턴, NeoX split-half (chunk[i], chunk[half+i])).
//   norm 은 f32 accumulator (CPU f64 와 head_dim≤256 에서 rel<1e-3 수렴).
//   rope 은 head 내 페어 순차(angle *= theta_scale) → CPU 와 동일 순서.
//
// per-token pos = pos_start + t (causal prefill positions).
kernel void prefill_rope_qk_norm(
    device const float* in        [[buffer(0)]], // [seq_len * num_heads * head_dim]
    device const float* weight    [[buffer(1)]], // [head_dim] (head 공유 norm weight)
    device float*       out       [[buffer(2)]], // [seq_len * num_heads * head_dim]
    constant uint&      num_heads [[buffer(3)]],
    constant uint&      head_dim  [[buffer(4)]],
    constant uint&      n_rot     [[buffer(5)]],
    constant float&     theta     [[buffer(6)]],
    constant float&     eps       [[buffer(7)]],
    constant uint&      pos_start [[buffer(8)]],
    uint group   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint t = group / num_heads;          // token index
    uint base = group * head_dim;        // group = t*num_heads + h, contiguous head slice

    threadgroup float partial[256];
    threadgroup float normed[256];       // head_dim <= 256

    // --- per-head RMSNorm (qk_norm) ---
    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = in[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // CPU rms_norm_into: rms = sqrt(mean_sq + eps); out = (x / rms) * weight.
    float inv_rms = rsqrt(partial[0] / (float)head_dim + eps);
    for (uint i = tid; i < head_dim; i += tg_size) {
        normed[i] = in[base + i] * inv_rms * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- partial RoPE (adjacent-pair, applied on normed q/k) ---
    // lane 0 only: sequential angle accumulation matches CPU rope_partial_inplace order.
    // CPU rope_partial: pair (chunk[i], chunk[i+1]) for i in 0,2,...,n_rot-1 (adjacent),
    // theta_scale = base^(-2/n_rot), angle starts at pos. 27B Qwen3.6 default(non-iMRoPE) path.
    if (tid == 0u && n_rot != 0u) {
        uint nr = min(n_rot, head_dim);
        float theta_scale = pow(theta, -2.0f / (float)nr);
        float angle = (float)(pos_start + t);
        uint i = 0u;
        while (i < nr) {
            float cos_a = cos(angle);
            float sin_a = sin(angle);
            float x0 = normed[i];
            float x1 = normed[i + 1u];
            normed[i]      = x0 * cos_a - x1 * sin_a;
            normed[i + 1u] = x0 * sin_a + x1 * cos_a;
            angle *= theta_scale;
            i += 2u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // store (rotated prefix + passthrough tail).
    for (uint i = tid; i < head_dim; i += tg_size) {
        out[base + i] = normed[i];
    }
}
kernel void prefill_rope_only(
    device const float* in           [[buffer(0)]],
    device float*       out          [[buffer(1)]],
    device const float2* rope_cos_sin [[buffer(2)]],
    constant uint&      num_heads    [[buffer(3)]],
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      n_rot        [[buffer(5)]],
    uint group   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint token = group / num_heads;
    uint base = group * head_dim;
    threadgroup float values[256];

    for (uint col = tid; col < head_dim; col += tg_size) {
        values[col] = in[base + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u && n_rot != 0u) {
        uint nr = min(n_rot, head_dim);
        uint pair_base = token * (n_rot / 2u);
        uint col = 0u;
        while (col < nr) {
            float2 cos_sin = rope_cos_sin[pair_base + col / 2u];
            float cos_a = cos_sin.x;
            float sin_a = cos_sin.y;
            float x0 = values[col];
            float x1 = values[col + 1u];
            volatile float x0_cos = x0 * cos_a;
            volatile float x1_sin = x1 * sin_a;
            volatile float x0_sin = x0 * sin_a;
            volatile float x1_cos = x1 * cos_a;
            values[col] = x0_cos - x1_sin;
            values[col + 1u] = x0_sin + x1_cos;
            col += 2u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint col = tid; col < head_dim; col += tg_size) {
        out[base + col] = values[col];
    }
}
