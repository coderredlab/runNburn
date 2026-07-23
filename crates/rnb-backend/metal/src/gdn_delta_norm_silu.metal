#include <metal_stdlib>
using namespace metal;

// pm127: GDN decode tail fused kernel — delta_net_step -> qk_norm(ssm_norm RMSNorm)
//   -> silu_mul(z) 를 v-head 단위 threadgroup 하나로 묶는다. 원본 3커널
//   (delta_net_step.metal + qk_norm.metal + silu_mul.metal)을 threadgroup shared 로
//   이어붙인 것이라 연산·순서가 동일(bit-identical). delta_out 을 device 버퍼 대신
//   dout_sh 에 유지, qk_norm 은 head_v_dim RMSNorm(tg=256 tree reduction), silu 는
//   z[zi] = silu(z[zi]) * gated 를 in-place.
//
// 매핑: threadgroup 1개 = v-head 1개 (grid = num_v_heads), tg_size = 256.
//   - delta: thread vi = state row vi (vi < head_v_dim). k/q 는 threadgroup 협력 로드.
//     GQA: v-head h 는 k-head kh = h % num_k_heads 의 q/k 공유.
//   - qk_norm: dout_sh[0..head_v_dim] RMSNorm (tg=256 고정 tree reduction, head_v_dim
//     요소만 non-zero → 원본 qk_norm(tg=256)과 같은 덧셈 순서).
//   - silu: z[h*head_v_dim+vi] = silu(z[...]) * (dout * inv_rms * ssm_norm_w[vi]).
//
// 전제: head_k_dim <= 256, head_v_dim <= 256 (caller gate 로 보장). Qwen3.6 GDN 은
//   head_k_dim = head_v_dim = 128.
kernel void gdn_delta_norm_silu(
    device const float* q           [[buffer(0)]],  // q_norm [num_k_heads*head_k_dim]
    device const float* k           [[buffer(1)]],  // k_norm [num_k_heads*head_k_dim]
    device const float* v           [[buffer(2)]],  // conv_out at v offset [num_v_heads*head_v_dim]
    device const float* gate        [[buffer(3)]],  // alpha [num_v_heads]
    device const float* beta        [[buffer(4)]],  // [num_v_heads]
    device float*       state       [[buffer(5)]],  // delta_state in-place [num_v_heads*head_v_dim*head_k_dim]
    device const float* ssm_norm_w  [[buffer(6)]],  // [head_v_dim]
    device float*       z           [[buffer(7)]],  // silu target in-place [num_v_heads*head_v_dim]
    constant uint&      head_k_dim  [[buffer(8)]],
    constant uint&      head_v_dim  [[buffer(9)]],
    constant uint&      num_k_heads [[buffer(10)]],
    constant float&     eps         [[buffer(11)]],
    uint h       [[threadgroup_position_in_grid]],  // v-head
    uint vi      [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float k_sh[256];
    threadgroup float q_sh[256];
    threadgroup float dout_sh[256];
    threadgroup float partial[256];

    // --- delta_net_step (delta_net_step.metal 과 동일) ---
    uint k_base = (h % num_k_heads) * head_k_dim;  // GQA: v-head h -> k-head kh
    for (uint i = vi; i < head_k_dim; i += tg_size) {
        k_sh[i] = k[k_base + i];
        q_sh[i] = q[k_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float dout = 0.0f;
    if (vi < head_v_dim) {
        float decay = exp(gate[h]);
        float b = beta[h];
        uint s_off = (h * head_v_dim + vi) * head_k_dim;
        // 1. d = (v - sum_ki decay*state[vi,ki]*k[ki]) * beta
        float sk = 0.0f;
        for (uint ki = 0; ki < head_k_dim; ki++) {
            sk += decay * state[s_off + ki] * k_sh[ki];
        }
        float d = (v[h * head_v_dim + vi] - sk) * b;
        // 2+3 fused: state = decay*state + k*d; out = sum new_state*q
        float sq = 0.0f;
        for (uint ki = 0; ki < head_k_dim; ki++) {
            float ns = decay * state[s_off + ki] + k_sh[ki] * d;
            state[s_off + ki] = ns;
            sq += ns * q_sh[ki];
        }
        dout = sq;
    }
    dout_sh[vi] = dout;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- qk_norm: per-head RMSNorm over head_v_dim (qk_norm.metal 과 동일) ---
    float sum = 0.0f;
    for (uint i = vi; i < head_v_dim; i += tg_size) {
        float val = dout_sh[i];
        sum += val * val;
    }
    partial[vi] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1u; stride > 0u; stride >>= 1u) {
        if (vi < stride) {
            partial[vi] += partial[vi + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partial[0] / (float)head_v_dim + eps);

    // --- silu_mul: z[zi] = silu(z[zi]) * gated (silu_mul.metal 과 동일) ---
    if (vi < head_v_dim) {
        float gated = dout_sh[vi] * inv_rms * ssm_norm_w[vi];
        uint zi = h * head_v_dim + vi;
        float g = z[zi];
        z[zi] = (g / (1.0f + exp(-g))) * gated;
    }
}
