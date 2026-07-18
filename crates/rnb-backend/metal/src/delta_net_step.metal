#include <metal_stdlib>
using namespace metal;

// GDN delta_net recurrent scan, decode 1-step (seq_len=1).
// rnb-cpu kernels/delta_net.rs delta_net_scan_into (seq_len=1) 와 1:1.
// head 별 독립 (threadgroup 1개 = v-head 1개, grid = num_v_heads).
// thread vi 1개 = state row vi 1개 (head_v_dim threads). state[h,vi,*] 는 thread vi
// 전용이라 race 없음. k/q 는 threadgroup shared 협력 로드(head_k_dim<=256).
// GQA: q/k 는 k-head(num_k_heads)만, v-head h 는 kh=h%num_k_heads 의 q/k 를 공유
//   (CPU gdn_q_rep/gdn_k_rep repeat 을 커널 내재화 → repeat copy 불필요).
//   non-GQA 는 num_k_heads==num_v_heads → kh==h.
//   decay = exp(gate[h]); b = beta[h]
//   d[vi]  = (v[vi] - sum_ki decay*state[vi,ki]*k[ki]) * b
//   state[vi,ki] = decay*state[vi,ki] + k[ki]*d[vi]   (in-place)
//   out[vi] = sum_ki state_new[vi,ki]*q[ki]
kernel void delta_net_step(
    device const float* q           [[buffer(0)]],  // [num_k_heads*head_k_dim]
    device const float* k           [[buffer(1)]],  // [num_k_heads*head_k_dim]
    device const float* v           [[buffer(2)]],  // [num_v_heads*head_v_dim]
    device const float* gate        [[buffer(3)]],  // [num_v_heads]
    device const float* beta        [[buffer(4)]],  // [num_v_heads]
    device float*       state       [[buffer(5)]],  // [num_v_heads*head_v_dim*head_k_dim] in-place
    device float*       out         [[buffer(6)]],  // [num_v_heads*head_v_dim]
    constant uint&      head_k_dim  [[buffer(7)]],
    constant uint&      head_v_dim  [[buffer(8)]],
    constant uint&      num_k_heads [[buffer(9)]],
    uint h       [[threadgroup_position_in_grid]],  // v-head
    uint vi      [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float k_sh[256];
    threadgroup float q_sh[256];
    uint k_base = (h % num_k_heads) * head_k_dim; // GQA: v-head h → k-head kh
    for (uint i = vi; i < head_k_dim; i += tg_size) {
        k_sh[i] = k[k_base + i];
        q_sh[i] = q[k_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (vi >= head_v_dim) return;

    float decay = exp(gate[h]);
    float b = beta[h];
    uint s_off = (h * head_v_dim + vi) * head_k_dim;

    // 1. d = (v - sum_ki decay*state[vi,ki]*k[ki]) * beta
    float sk = 0.0f;
    for (uint ki = 0; ki < head_k_dim; ki++) {
        sk += decay * state[s_off + ki] * k_sh[ki];
    }
    float d = (v[h * head_v_dim + vi] - sk) * b;

    // 2+3 fused: state[vi,ki] = decay*state[vi,ki] + k[ki]*d; out = sum new_state*q
    float sq = 0.0f;
    for (uint ki = 0; ki < head_k_dim; ki++) {
        float ns = decay * state[s_off + ki] + k_sh[ki] * d;
        state[s_off + ki] = ns;
        sq += ns * q_sh[ki];
    }
    out[h * head_v_dim + vi] = sq;
}
