#include <metal_stdlib>
using namespace metal;

// residual in-place: hidden[i] += down[i]
// rnb-cpu kernels/elementwise.rs add_inplace (동일 len) 1:1.
kernel void residual_add(
    device float*       hidden [[buffer(0)]],
    device const float* down   [[buffer(1)]],
    constant uint&      dim    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    hidden[gid] += down[gid];
}
