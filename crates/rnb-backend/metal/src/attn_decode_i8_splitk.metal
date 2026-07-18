#include <metal_stdlib>
using namespace metal;

// pm54: split-K int8 KV decode attention.
//
// part:   one threadgroup = one (query head, KV split). It computes a local
//         online-softmax numerator, max, and denominator for that KV range.
// reduce: one threadgroup = one query head. It combines split-local softmax
//         states with log-sum-exp rescaling and writes the final output.
//
// This keeps the same input layout as attn_decode_i8, but raises long-context
// decode parallelism from num_heads threadgroups to num_heads*num_splits.

kernel void attn_decode_i8_splitk_part(
    device const float* q            [[buffer(0)]],
    device const char*  k_cache      [[buffer(1)]],
    device const char*  v_cache      [[buffer(2)]],
    device const float* k_scale      [[buffer(3)]],
    device const float* v_scale      [[buffer(4)]],
    device float*       partial_acc  [[buffer(5)]],
    device float*       partial_m    [[buffer(6)]],
    device float*       partial_s    [[buffer(7)]],
    constant uint&      num_heads    [[buffer(8)]],
    constant uint&      num_kv_heads [[buffer(9)]],
    constant uint&      head_dim     [[buffer(10)]],
    constant uint&      kv_len       [[buffer(11)]],
    constant float&     scale        [[buffer(12)]],
    constant uint&      num_splits   [[buffer(13)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint h = gid.x;
    uint split = gid.y;
    if (h >= num_heads || split >= num_splits) return;

    uint row = split * num_heads + h;
    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_h = h / heads_per_group;
    uint kv_dim = num_kv_heads * head_dim;
    uint q_off = h * head_dim;

    uint chunk = (kv_len + num_splits - 1u) / num_splits;
    uint start = split * chunk;
    uint end = min(kv_len, start + chunk);

    float qf[8];
    float acc[8];
    uint nloc = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        qf[nloc] = (float)(half)q[q_off + d];
        acc[nloc] = 0.0f;
        nloc++;
    }

    float m = -INFINITY;
    float s = 0.0f;
    for (uint j = start; j < end; j++) {
        uint kv_off = j * kv_dim + kv_h * head_dim;
        uint sidx = j * num_kv_heads + kv_h;
        float ksc = k_scale[sidx];
        float vsc = v_scale[sidx];

        float partial = 0.0f;
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            partial += qf[idx] * (float)k_cache[kv_off + d];
            idx++;
        }
        float x = simd_sum(partial) * scale * ksc;

        if (x > m) {
            bool rescale = (m > -INFINITY);
            float alpha = rescale ? exp(m - x) : 1.0f;
            if (rescale) s *= alpha;
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = acc[idx];
                if (rescale) a *= alpha;
                float vv = (float)v_cache[kv_off + d] * vsc;
                acc[idx] = a + vv;
                idx++;
            }
            s += 1.0f;
            m = x;
        } else {
            float p = exp(x - m);
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float vv = (float)v_cache[kv_off + d] * vsc;
                acc[idx] += vv * p;
                idx++;
            }
            s += p;
        }
    }

    if (lane == 0u) {
        partial_m[row] = m;
        partial_s[row] = s;
    }
    uint idx = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        partial_acc[row * head_dim + d] = acc[idx];
        idx++;
    }
}

kernel void attn_decode_i8_splitk_reduce(
    device const float* partial_acc [[buffer(0)]],
    device const float* partial_m   [[buffer(1)]],
    device const float* partial_s   [[buffer(2)]],
    device float*       out         [[buffer(3)]],
    constant uint&      num_heads   [[buffer(4)]],
    constant uint&      head_dim    [[buffer(5)]],
    constant uint&      num_splits  [[buffer(6)]],
    uint h    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (h >= num_heads) return;

    float m = -INFINITY;
    for (uint split = 0u; split < num_splits; split++) {
        uint row = split * num_heads + h;
        m = max(m, partial_m[row]);
    }

    float denom = 0.0f;
    float acc[8];
    uint nloc = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        acc[nloc] = 0.0f;
        nloc++;
    }

    for (uint split = 0u; split < num_splits; split++) {
        uint row = split * num_heads + h;
        float s = partial_s[row];
        float factor = (s > 0.0f) ? exp(partial_m[row] - m) : 0.0f;
        denom += s * factor;
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            acc[idx] += partial_acc[row * head_dim + d] * factor;
            idx++;
        }
    }

    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    uint out_off = h * head_dim;
    uint idx = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        out[out_off + d] = acc[idx] * inv;
        idx++;
    }
}
