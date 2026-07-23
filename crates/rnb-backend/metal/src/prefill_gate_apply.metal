#include <metal_stdlib>
using namespace metal;

kernel void prefill_gate_apply(
    device const float* attn_out [[buffer(0)]],
    device const float* gate     [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      elems    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= elems) return;
    float g = gate[gid];
    float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = attn_out[gid] * sig;
}
