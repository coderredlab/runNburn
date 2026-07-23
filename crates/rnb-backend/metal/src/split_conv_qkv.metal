#include <metal_stdlib>
using namespace metal;

// GDN prefill conv1d_silu 출력(per-token [q_dim | k_dim | v_dim | ...잔여] 인터리브,
// stride = conv_channels)을 연속 q/k/v 버퍼로 분리. host `gdn_prefill.rs:75 split_conv_qkv` 1:1.
// 단순 copy (산술 없음 → lane 간 독립 → bit-identical). flat 1D grid: 각 thread = 1 element.
// q_dim+k_dim+v_dim <= conv_channels (gate 등 잔여 채널은 어디로도 안 감 = skip).
kernel void split_conv_qkv(
    device const float* conv_data     [[buffer(0)]], // [seq_len * conv_channels] read-only
    device float*       q_out         [[buffer(1)]], // [seq_len * q_dim] write
    device float*       k_out         [[buffer(2)]], // [seq_len * k_dim] write
    device float*       v_out         [[buffer(3)]], // [seq_len * v_dim] write
    constant uint&      seq_len       [[buffer(4)]],
    constant uint&      conv_channels [[buffer(5)]],
    constant uint&      q_dim         [[buffer(6)]],
    constant uint&      k_dim         [[buffer(7)]],
    constant uint&      v_dim         [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = seq_len * conv_channels;
    if (gid >= total) return;
    uint t = gid / conv_channels;
    uint c = gid % conv_channels;
    float val = conv_data[gid];
    if (c < q_dim) {
        q_out[t * q_dim + c] = val;
    } else if (c < q_dim + k_dim) {
        k_out[t * k_dim + (c - q_dim)] = val;
    } else if (c < q_dim + k_dim + v_dim) {
        v_out[t * v_dim + (c - q_dim - k_dim)] = val;
    }
}
