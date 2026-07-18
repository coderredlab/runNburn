#include <metal_stdlib>
using namespace metal;

// gated attention: attn_out[i] *= sigmoid(gate[i]) = attn_out[i] / (1 + exp(-gate[i])).
// host `sigmoid_inplace`(activation.rs) + `mul_inplace`(decode.rs:751-757) 1:1.
// elementwise, 1 thread = 1 elem (silu_mul 패턴).
kernel void gate_apply(
    device float*       attn_out [[buffer(0)]], // [n] in-place
    device const float* gate     [[buffer(1)]], // [n]
    constant uint&      n        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float s = 1.0f / (1.0f + exp(-gate[gid]));
    attn_out[gid] = attn_out[gid] * s;
}
