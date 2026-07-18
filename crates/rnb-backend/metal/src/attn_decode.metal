#include <metal_stdlib>
using namespace metal;

// Single-token decode attention (QK^T -> online softmax -> AV).
//
// CPU `attention_decode_flash` 의 production default path(`process_head_f16_acc`,
// has_fp16=false branch) 와 token-identical 하게 emulate:
//   - Q 를 f16 으로 round 후 dot (q_to_vec_dot 매칭)
//   - V accumulator 를 f16 으로 보관, 매 step f16 round-trip (ggml VKQ16)
//   - branched online softmax: x>m 이면 acc/s 를 alpha=exp(m-x) 로 rescale 후
//     p=1 로 V 누적, 아니면 p=exp(x-m) 로 누적
//
// 1 threadgroup = 1 query head(grid=num_heads), 1 SIMD-group(32 lane).
// lane 이 head_dim 을 stride 32 로 분할. dot 은 simd_sum 으로 합산, running
// (m,s) 는 모든 lane 이 동일 x 를 받아 redundant 계산(동기화 불필요),
// acc 는 lane 별 담당 d 만 갱신(race 없음).
//
// head_dim <= 256 가정(lane 당 최대 8개). dispatch 측에서 assert.
kernel void attn_decode(
    device const float*  q            [[buffer(0)]],
    device const ushort* k_cache      [[buffer(1)]],
    device const ushort* v_cache      [[buffer(2)]],
    device float*        out          [[buffer(3)]],
    constant uint&       num_heads    [[buffer(4)]],
    constant uint&       num_kv_heads [[buffer(5)]],
    constant uint&       head_dim     [[buffer(6)]],
    constant uint&       kv_len       [[buffer(7)]],
    constant float&      scale        [[buffer(8)]],
    constant uint&       read_mask    [[buffer(9)]],
    uint h    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (h >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_h = h / heads_per_group;
    uint kv_dim = num_kv_heads * head_dim;
    uint q_off = h * head_dim;

    // 이 lane 이 담당하는 head_dim index 들(d = lane, lane+32, ...).
    float qf[8];   // f16-rounded Q
    half  acc[8];  // f16 V accumulator
    uint nloc = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        qf[nloc] = (float)(half)q[q_off + d]; // Q -> f16 round -> f32
        acc[nloc] = (half)0.0f;
        nloc++;
    }

    float m = -INFINITY;
    float s = 0.0f;

    for (uint j = 0u; j < kv_len; j++) {
        // 측정 게이트(pm22): read_mask=0xFFFFFFFF면 normal(j 그대로). 작은 window-1(예 63)이면
        // K·V read 주소를 window 슬롯으로 wrap → read traffic 만 cap, compute(simd_sum/exp/acc)는
        // kv_len 전체 그대로. normal vs capped GPU time 차 = K·V read traffic 순비용(int8 effect 추정).
        uint kv_off = (j & read_mask) * kv_dim + kv_h * head_dim;

        // QK^T (lane partial dot + simd reduce) — 모든 lane 이 동일 x.
        float partial = 0.0f;
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            float kf = (float)as_type<half>(k_cache[kv_off + d]);
            partial += qf[idx] * kf;
            idx++;
        }
        float x = simd_sum(partial) * scale;

        // branched online softmax + f16 V accumulate.
        if (x > m) {
            bool rescale = (m > -INFINITY);
            float alpha = rescale ? exp(m - x) : 1.0f;
            if (rescale) {
                s *= alpha;
            }
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = (float)acc[idx];
                if (rescale) {
                    a *= alpha;
                }
                float vv = (float)as_type<half>(v_cache[kv_off + d]);
                acc[idx] = (half)(a + vv); // p = 1
                idx++;
            }
            s += 1.0f;
            m = x;
        } else {
            float p = exp(x - m);
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = (float)acc[idx];
                float vv = (float)as_type<half>(v_cache[kv_off + d]);
                acc[idx] = (half)(a + vv * p);
                idx++;
            }
            s += p;
        }
    }

    // Final f16 -> f32 normalize.
    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    uint out_off = h * head_dim;
    uint idx = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        out[out_off + d] = (float)acc[idx] * inv_s;
        idx++;
    }
}
