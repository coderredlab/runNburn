#include <metal_stdlib>
using namespace metal;

kernel void prefill_split_q_gate(
    device const float* q_full [[buffer(0)]],
    device float*       q_out  [[buffer(1)]],
    device float*       gate   [[buffer(2)]],
    constant uint&      seq    [[buffer(3)]],
    constant uint&      num_heads [[buffer(4)]],
    constant uint&      head_dim  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint q_dim = num_heads * head_dim;
    uint total = seq * q_dim;
    if (gid >= total) return;

    uint t = gid / q_dim;
    uint rem = gid - t * q_dim;
    uint h = rem / head_dim;
    uint d = rem - h * head_dim;
    uint src = t * (q_dim * 2u) + h * (head_dim * 2u) + d;
    q_out[gid] = q_full[src];
    gate[gid] = q_full[src + head_dim];
}
