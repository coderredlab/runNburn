#include <metal_stdlib>
using namespace metal;

// GDN delta_net prefill scan, autoregressive row-parallel form.
//
// Mapping mirrors llama.cpp's Qwen3Next gated_delta_net Metal kernel:
//   threadgroup = one head and NSG output rows
//   thread x    = 32-lane SIMD over K
//   thread y    = output/state row within the NSG row pack
//
// Unlike delta_net_scan_chunk, this does not build chunk-local WY intermediates.
// It runs the same recurrence as delta_net_step for all tokens while keeping one
// state row in registers.
template<ushort NSG>
kernel void delta_net_scan_ar_impl(
    device const float* q          [[buffer(0)]],  // [seq*num_k_heads*head_k_dim]
    device const float* k          [[buffer(1)]],  // [seq*num_k_heads*head_k_dim]
    device const float* v          [[buffer(2)]],  // [seq*num_heads*head_v_dim]
    device const float* gate       [[buffer(3)]],  // [seq*num_heads]
    device const float* beta       [[buffer(4)]],  // [seq*num_heads]
    device float*       state      [[buffer(5)]],  // [num_heads*head_v_dim*head_k_dim] in-place
    device float*       out        [[buffer(6)]],  // [seq*num_heads*head_v_dim]
    constant uint&      seq_len    [[buffer(7)]],
    constant uint&      head_k_dim [[buffer(8)]],
    constant uint&      head_v_dim [[buffer(9)]],
    constant uint&      num_heads  [[buffer(10)]],
    constant uint&      num_k_heads [[buffer(11)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint row = tg_pos.x * NSG + ty;
    const uint h = tg_pos.y;
    if (h >= num_heads || row >= head_v_dim) {
        return;
    }

    const uint state_base = (h * head_v_dim + row) * head_k_dim;
    float state_row[NSG];

    for (ushort j = 0; j < NSG; j++) {
        const uint ki = tx * NSG + j;
        state_row[j] = state[state_base + ki];
    }

    for (uint t = 0; t < seq_len; t++) {
        const uint th = t * num_heads + h;
        const uint kh = h % num_k_heads;
        const uint qk_base = (t * num_k_heads + kh) * head_k_dim;
        const uint v_base = th * head_v_dim;
        const float decay = exp(gate[th]);

        float sk = 0.0f;
        for (ushort j = 0; j < NSG; j++) {
            const uint ki = tx * NSG + j;
            const float s = state_row[j] * decay;
            state_row[j] = s;
            sk += s * k[qk_base + ki];
        }
        sk = simd_sum(sk);

        const float d = (v[v_base + row] - sk) * beta[th];

        float sq = 0.0f;
        for (ushort j = 0; j < NSG; j++) {
            const uint ki = tx * NSG + j;
            const float s = state_row[j] + k[qk_base + ki] * d;
            state_row[j] = s;
            sq += s * q[qk_base + ki];
        }
        sq = simd_sum(sq);

        if (tx == 0) {
            out[v_base + row] = sq;
        }
    }

    for (ushort j = 0; j < NSG; j++) {
        const uint ki = tx * NSG + j;
        state[state_base + ki] = state_row[j];
    }
}

typedef decltype(delta_net_scan_ar_impl<4>) delta_net_scan_ar_t;

template [[host_name("delta_net_scan_ar1")]] kernel delta_net_scan_ar_t delta_net_scan_ar_impl<1>;
template [[host_name("delta_net_scan_ar2")]] kernel delta_net_scan_ar_t delta_net_scan_ar_impl<2>;
template [[host_name("delta_net_scan_ar4")]] kernel delta_net_scan_ar_t delta_net_scan_ar_impl<4>;
template [[host_name("delta_net_scan_ar8")]] kernel delta_net_scan_ar_t delta_net_scan_ar_impl<8>;
