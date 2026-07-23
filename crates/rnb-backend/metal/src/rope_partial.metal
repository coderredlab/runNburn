#include <metal_stdlib>
using namespace metal;

// Partial RoPE (인접페어, decode 1 token, in-place). host `rope_partial_inplace`
// (rope.rs:387) 1:1. 첫 n_rot 차원만 회전, 인접페어 (chunk[i], chunk[i+1]).
// theta_scale 은 host f32 precompute (theta.powf(-2/n_rot), clamp 후) — 누적
// ULP drift 방지. fp 순서 = host: cos 먼저, sin, x0/x1 read, i 먼저 i+1 write,
// angle *= theta_scale. 1 thread = 1 head (grid = dim/head_dim).
kernel void rope_partial(
    device float*   data        [[buffer(0)]], // [dim] in-place
    constant uint&  head_dim    [[buffer(1)]],
    constant uint&  dim         [[buffer(2)]],
    constant uint&  n_rot       [[buffer(3)]],
    constant float& theta_scale [[buffer(4)]],
    constant uint&  pos         [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]])
{
    uint num_heads = dim / head_dim;
    if (head >= num_heads || n_rot == 0u) return;
    uint nr = min(n_rot, head_dim);
    uint base = head * head_dim;
    float angle = (float)pos;
    for (uint i = 0u; i < nr; i += 2u) {
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float x0 = data[base + i];
        float x1 = data[base + i + 1u];
        data[base + i]      = x0 * cos_a - x1 * sin_a;
        data[base + i + 1u] = x0 * sin_a + x1 * cos_a;
        angle *= theta_scale;
    }
}
