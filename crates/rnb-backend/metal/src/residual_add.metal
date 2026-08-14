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

// Gemma layer output scale: hidden = (hidden + down) * scale.
kernel void residual_add_scaled(
    device float*       hidden [[buffer(0)]],
    device const float* down   [[buffer(1)]],
    constant uint&      dim    [[buffer(2)]],
    constant float&     scale  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    hidden[gid] = (hidden[gid] + down[gid]) * scale;
}

// Target-layer hidden capture for Muse DFlash. Input and output are token-major;
// feature_index selects one hidden-width slice inside each token's concatenated
// target feature row.
kernel void muse_capture_feature(
    device const float* hidden       [[buffer(0)]],
    device float*       features     [[buffer(1)]],
    constant uint&      hidden_dim   [[buffer(2)]],
    constant uint&      feature_dim  [[buffer(3)]],
    constant uint&      feature_base [[buffer(4)]],
    constant uint&      total        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    uint token = gid / hidden_dim;
    uint column = gid - token * hidden_dim;
    features[token * feature_dim + feature_base + column] = hidden[gid];
}
