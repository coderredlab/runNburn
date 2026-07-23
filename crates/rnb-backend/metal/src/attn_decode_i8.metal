#include <metal_stdlib>
using namespace metal;

// int8 KV decode attention(pm22). f16 attn_decode 와 동일 branched online softmax,
// K·V 만 per-slot int8(quantize_slot_i8_ref: q = round(x/scale), scale = max_abs/127).
//
// 정확성 핵심 2가지:
//   - K dequant: K 의 per-slot scale(ksc)을 simd_sum 밖에서 곱한다(per-slot 분배법칙).
//     dot 자체는 int8 K 값으로 누적하고, x = simd_sum(partial) * scale * ksc.
//   - V dequant: vv = v_i8 * vsc. v_scale 은 softmax 확률 p 가 아니라 V 값에 붙는다.
//     rescale 분기(x>m, p=1)와 else 분기 양쪽 모두 vv = v_i8 * vsc 로 dequant.
//
// q 는 f16-round 유지((float)(half)q)로 f16 attn_decode 와 같은 Q precision.
// int8 엔 read_mask 미적용(spec normal-only). 1 threadgroup = 1 query head,
// 1 SIMD-group(32 lane). lane 이 head_dim 을 stride 32 분할. head_dim <= 256.
kernel void attn_decode_i8(
    device const float* q            [[buffer(0)]],
    device const char*  k_cache      [[buffer(1)]],
    device const char*  v_cache      [[buffer(2)]],
    device const float* k_scale      [[buffer(3)]],
    device const float* v_scale      [[buffer(4)]],
    device float*       out          [[buffer(5)]],
    constant uint&      num_heads    [[buffer(6)]],
    constant uint&      num_kv_heads [[buffer(7)]],
    constant uint&      head_dim     [[buffer(8)]],
    constant uint&      kv_len       [[buffer(9)]],
    constant float&     scale        [[buffer(10)]],
    uint h    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (h >= num_heads) return;
    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_h = h / heads_per_group;
    uint kv_dim = num_kv_heads * head_dim;
    uint q_off = h * head_dim;

    // 이 lane 이 담당하는 head_dim index(d = lane, lane+32, ...).
    float qf[8]; half acc[8]; uint nloc = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        qf[nloc] = (float)(half)q[q_off + d]; // Q -> f16 round -> f32
        acc[nloc] = (half)0.0f;
        nloc++;
    }

    float m = -INFINITY, s = 0.0f;
    for (uint j = 0u; j < kv_len; j++) {
        uint kv_off = j * kv_dim + kv_h * head_dim;
        uint sidx = j * num_kv_heads + kv_h; // per-slot scale index
        float ksc = k_scale[sidx], vsc = v_scale[sidx];

        // QK^T (lane partial dot + simd reduce). K 는 int8 값으로 누적 후
        // ksc 를 simd_sum 밖에서 곱한다(per-slot 분배법칙).
        float partial = 0.0f; uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            partial += qf[idx] * (float)k_cache[kv_off + d];
            idx++;
        }
        float x = simd_sum(partial) * scale * ksc;

        // branched online softmax + f16 V accumulate. V 는 vv = v_i8 * vsc 로 dequant.
        if (x > m) {
            bool rescale = (m > -INFINITY);
            float alpha = rescale ? exp(m - x) : 1.0f;
            if (rescale) s *= alpha;
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = (float)acc[idx];
                if (rescale) a *= alpha;
                float vv = (float)v_cache[kv_off + d] * vsc;
                acc[idx] = (half)(a + vv); // p = 1
                idx++;
            }
            s += 1.0f; m = x;
        } else {
            float p = exp(x - m); idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = (float)acc[idx];
                float vv = (float)v_cache[kv_off + d] * vsc;
                acc[idx] = (half)(a + vv * p);
                idx++;
            }
            s += p;
        }
    }

    // Final f16 -> f32 normalize.
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    uint out_off = h * head_dim; uint idx = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        out[out_off + d] = (float)acc[idx] * inv_s;
        idx++;
    }
}
