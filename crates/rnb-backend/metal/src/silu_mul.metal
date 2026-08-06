#include <metal_stdlib>
using namespace metal;

// SwiGLU in-place: gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// rnb-cpu kernels/activation.rs fused_silu_mul_inplace 1:1.
kernel void silu_mul(
    device float*       gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint&      dim  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float g = gate[gid];
    gate[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// GeGLU in-place: exact tanh approximation used by rnb-cpu fused_gelu_mul_inplace.
kernel void gelu_mul(
    device float*       gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint&      dim  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float g = gate[gid];
    float gelu = g >= 10.0f
        ? g
        : (g <= -10.0f
            ? 0.0f
            : 0.5f * g * (1.0f + tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))));
    gate[gid] = gelu * up[gid];
}

// DeepSeek4 SwiGLU: routed/shared slots may use different clamp limits.
kernel void silu_mul_clamped_slots(
    device float*       gate        [[buffer(0)]],
    device const float* up          [[buffer(1)]],
    device const float* slot_limits [[buffer(2)]],
    constant uint&      n_ff        [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float limit = slot_limits[gid / n_ff];
    float g = min(gate[gid], limit);
    float u = clamp(up[gid], -limit, limit);
    gate[gid] = (g / (1.0f + exp(-g))) * u;
}

// Prefill tensorops down GEMM consumes half activation. This is bit-equivalent to
// silu_mul(gate, up) followed by cast_f32_to_f16, but avoids the intermediate f32
// activation write/read.
kernel void silu_mul_to_f16(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device half*        out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float g = gate[gid];
    out[gid] = (half)((g / (1.0f + exp(-g))) * up[gid]);
}

kernel void gelu_mul_to_f16(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device half*        out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float g = gate[gid];
    float gelu = g >= 10.0f
        ? g
        : (g <= -10.0f
            ? 0.0f
            : 0.5f * g * (1.0f + tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))));
    out[gid] = (half)(gelu * up[gid]);
}

kernel void silu_mul_half_to_f16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     dim  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    float g = (float)gate[gid];
    out[gid] = (half)((g / (1.0f + exp(-g))) * (float)up[gid]);
}
