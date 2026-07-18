#include <metal_stdlib>
using namespace metal;

// Decode(seq_len=1) depthwise causal conv1d + SiLU.
//   out[c] = silu( sum_{k=0..kernel_size} in[k*channels + c] * weight[k*channels + c] )
//   silu(x) = x / (1 + exp(-x))
// rnb-cpu kernels/conv.rs ssm_conv1d_silu_into (seq_len=1) 와 1:1. channel 독립
// (depthwise) — thread 1개 = channel 1개 (grid = channels, thread_position_in_grid).
// GDN layer carrier 의 conv 부품 (host readback 제거 → device chain 연속).
kernel void ssm_conv1d_silu(
    device const float* in          [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      channels    [[buffer(3)]],
    constant uint&      kernel_size [[buffer(4)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= channels) return;
    float sum = 0.0f;
    for (uint k = 0; k < kernel_size; k++) {
        sum += in[k * channels + c] * weight[k * channels + c];
    }
    out[c] = sum / (1.0f + exp(-sum)); // SiLU

}

// Prefill batch (seq_len > 1) depthwise causal conv1d + SiLU.
//   out[t*channels + c] = silu( sum_{k} in[(t+k)*channels + c] * weight[k*channels + c] ), t in 0..seq_len.
//   in 은 [(seq_len + kernel_size - 1) * channels] (causal padding, caller 준비),
//   out 은 [seq_len * channels]. rnb-cpu kernels/conv.rs ssm_conv1d_silu_into(seq_len) 와 1:1.
//   depthwise(channel 독립) — thread 1개 = (token t, channel c) 1개. grid = seq_len * channels.
kernel void ssm_conv1d_silu_batch(
    device const float* in          [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      channels    [[buffer(3)]],
    constant uint&      kernel_size [[buffer(4)]],
    constant uint&      seq_len     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= seq_len * channels) return;
    uint t = gid / channels;
    uint c = gid % channels;
    float sum = 0.0f;
    for (uint k = 0; k < kernel_size; k++) {
        sum += in[(t + k) * channels + c] * weight[k * channels + c];
    }
    out[t * channels + c] = sum / (1.0f + exp(-sum)); // SiLU
}
