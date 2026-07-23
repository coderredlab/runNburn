#include <metal_stdlib>
using namespace metal;

// Text M-RoPE (split-half rotation), decode(seq_len=1) in-place.
//
// CPU `rope_mrope_text_inplace` 1:1: head 별로 half=n_rot/2 페어
//   (chunk[i], chunk[half+i]) 를 angle 로 회전, angle 은 페어마다 *= theta_scale.
//   theta_scale = theta^(-2/n_rot), angle 시작 = pos.
// 1 thread = 1 head(grid=num_heads = dim/head_dim). head 안 페어는 순차
// (angle 누적이 CPU 와 동일 순서 → token-identical).
kernel void rope_mrope(
    device float*   data     [[buffer(0)]], // [dim] in-place
    constant uint&  head_dim [[buffer(1)]],
    constant uint&  dim      [[buffer(2)]],
    constant uint&  n_rot    [[buffer(3)]],
    constant float& theta    [[buffer(4)]],
    constant uint&  pos       [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]])
{
    uint num_heads = dim / head_dim;
    if (head >= num_heads) return;
    if (n_rot == 0u) return;

    uint nr = min(n_rot, head_dim);
    uint half_n = nr / 2u;
    float theta_scale = pow(theta, -2.0f / (float)nr);

    uint base = head * head_dim;
    float angle = (float)pos;
    for (uint i = 0u; i < half_n; i++) {
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float x0 = data[base + i];
        float x1 = data[base + half_n + i];
        data[base + i]          = x0 * cos_a - x1 * sin_a;
        data[base + half_n + i] = x0 * sin_a + x1 * cos_a;
        angle *= theta_scale;
    }
}
