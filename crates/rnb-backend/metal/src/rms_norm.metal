#include <metal_stdlib>
using namespace metal;

// RMSNorm: out[i] = (in[i] / sqrt(mean(in^2) + eps)) * weight[i]
// rnb-cpu kernels/norm.rs rms_norm_into_scalar 1:1.
// 단일 threadgroup(grid=1) 으로 실행. tg_size 는 256 고정(2의 거듭제곱 →
// tree reduction 정확, SIMD width 32 의 배수). grid-stride 라 dim 임의 OK.
kernel void rms_norm(
    device const float* in     [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      dim    [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = in[i];
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

    float inv_rms = rsqrt(partial[0] / (float)dim + eps);
    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

// pm43: GDN prefill gated RMSNorm + SiLU gate (batch, rows>1).
//   gated[row*cols + i] = (out_in[row,i] / sqrt(mean(out_in[row]^2) + eps)) * weight[i] * silu(z[row,i]).
//   silu(z) = z / (1 + exp(-z)). per-row rms(cols) + elementwise silu·z. weight[cols]=ssm_norm 공유.
//   threadgroup 1개 = row 1개(grid=rows), tg=256. CPU gdn_forward gated_norm+silu 경로와 1:1.
kernel void gated_rmsnorm_silu_batch(
    device const float* out_in [[buffer(0)]],  // [rows*cols] scan 출력
    device const float* z      [[buffer(1)]],  // [rows*cols] gate 입력
    device const float* weight [[buffer(2)]],  // [cols] ssm_norm
    device float*       gated  [[buffer(3)]],  // [rows*cols]
    constant uint&      cols   [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    uint row     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    uint base = row * cols;
    float sum = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        float v = out_in[base + i];
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
    float inv_rms = rsqrt(partial[0] / (float)cols + eps);
    for (uint i = tid; i < cols; i += tg_size) {
        float normed = out_in[base + i] * inv_rms * weight[i];
        float zz = z[base + i];
        float silu_z = zz / (1.0f + exp(-zz));
        gated[base + i] = normed * silu_z;
    }
}

// pm49: post-attn RMSNorm batch. threadgroup 1개 = row 1개.
// out[row, i] = in[row, i] / sqrt(mean(row^2) + eps) * weight[i].
kernel void rms_norm_batch(
    device const float* in     [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      cols   [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint row     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    uint base = row * cols;
    float sum = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
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

    float inv_rms = rsqrt(partial[0] / (float)cols + eps);
    for (uint i = tid; i < cols; i += tg_size) {
        out[base + i] = in[base + i] * inv_rms * weight[i];
    }
}
