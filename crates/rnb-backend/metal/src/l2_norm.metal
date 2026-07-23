#include <metal_stdlib>
using namespace metal;

// Per-row L2 normalize (+ optional scale): out = in/sqrt(sum(in_row^2)+eps) * scale.
// rnb-cpu kernels/norm.rs l2_norm_into 1:1 — weight 없음, /dim 없음(RMS 와 다름).
// scale: GDN q 는 1/sqrt(head_k_dim)(l2_norm 후 scaling), k 는 1.0.
// GDN q/k 의 head(=row) 별 normalize. threadgroup 1개 = row 1개(grid = n_rows),
// tg=256 tree reduction(grid-stride 라 dim 임의).
kernel void l2_norm(
    device const float* in    [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant uint&      dim   [[buffer(2)]],
    constant float&     eps   [[buffer(3)]],
    constant float&     scale [[buffer(4)]],
    uint row     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    uint base = row * dim;
    float sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = in[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_norm = rsqrt(partial[0] + eps) * scale; // L2: /dim 없음, scale 적용
    for (uint i = tid; i < dim; i += tg_size) {
        out[base + i] = in[base + i] * inv_norm;
    }
}
