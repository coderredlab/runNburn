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

// Muse sliding-window variant. KV stays in absolute slots; only the score/value loop
// starts at max(0, kv_len-window). Arithmetic within the active range matches attn_decode.
kernel void attn_decode_window(
    device const float*  q            [[buffer(0)]],
    device const ushort* k_cache      [[buffer(1)]],
    device const ushort* v_cache      [[buffer(2)]],
    device float*        out          [[buffer(3)]],
    constant uint&       num_heads    [[buffer(4)]],
    constant uint&       num_kv_heads [[buffer(5)]],
    constant uint&       head_dim     [[buffer(6)]],
    constant uint&       kv_len       [[buffer(7)]],
    constant float&      scale        [[buffer(8)]],
    constant uint&       window       [[buffer(9)]],
    uint h    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (h >= num_heads) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_h = h / heads_per_group;
    uint kv_dim = num_kv_heads * head_dim;
    uint q_off = h * head_dim;

    float qf[8];
    half acc[8];
    uint nloc = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        qf[nloc] = (float)(half)q[q_off + d];
        acc[nloc] = (half)0.0f;
        nloc++;
    }

    float m = -INFINITY;
    float s = 0.0f;
    uint start = window > 0u && kv_len > window ? kv_len - window : 0u;
    for (uint j = start; j < kv_len; j++) {
        uint kv_off = j * kv_dim + kv_h * head_dim;
        float partial = 0.0f;
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            float kf = (float)as_type<half>(k_cache[kv_off + d]);
            partial += qf[idx] * kf;
            idx++;
        }
        float x = simd_sum(partial) * scale;
        if (x > m) {
            bool rescale = (m > -INFINITY);
            float alpha = rescale ? exp(m - x) : 1.0f;
            if (rescale) s *= alpha;
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = (float)acc[idx];
                if (rescale) a *= alpha;
                float vv = (float)as_type<half>(v_cache[kv_off + d]);
                acc[idx] = (half)(a + vv);
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

    float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    uint out_off = h * head_dim;
    uint idx = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        out[out_off + d] = (float)acc[idx] * inv_s;
        idx++;
    }
}
// Muse decode vector attention. Sixteen SIMD-groups split the KV tiles for one
// query head and reduce their online-softmax states inside the threadgroup.
kernel void attn_decode_f16_gqa16(
    device const float*  q            [[buffer(0)]],
    device ushort*       k_cache      [[buffer(1)]],
    device ushort*       v_cache      [[buffer(2)]],
    device float*        out          [[buffer(3)]],
    constant uint&       num_heads    [[buffer(4)]],
    constant uint&       num_kv_heads [[buffer(5)]],
    constant uint&       head_dim     [[buffer(6)]],
    constant uint&       kv_len       [[buffer(7)]],
    constant float&      scale        [[buffer(8)]],
    constant uint&       window       [[buffer(9)]],
    device const float*  gate         [[buffer(10)]],
    constant uint&       pos          [[buffer(11)]],
    device const float*  k_current    [[buffer(12)]],
    device const float*  v_current    [[buffer(13)]],
    uint h      [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint HD = 128u;
    constexpr uint KV_TILE = 32u;
    constexpr uint SPLITS = 16u;
    if (h >= num_heads || head_dim != HD || num_heads != num_kv_heads * 16u) return;

    uint kv_h = h / 16u;
    uint kv_dim = num_kv_heads * HD;
    uint q_off = h * HD;
    uint lane_group = lane >> 3u;
    uint lane_in_group = lane & 7u;

    float4 q4[4];
    for (uint c = 0u; c < 4u; c++) {
        uint vector_index = lane_in_group + c * 8u;
        q4[c] = float4(half4(((device const float4*)(q + q_off))[vector_index]));
    }
    float4 acc = 0.0f;
    float m = -INFINITY;
    float s = 0.0f;
    threadgroup float scores[SPLITS][KV_TILE];
    threadgroup float4 split_acc[SPLITS][HD / 4u];
    threadgroup float split_m[SPLITS];
    threadgroup float split_s[SPLITS];
    threadgroup float merged_m;
    threadgroup float merged_s;
    if (sg == 0u && lane < 4u && (h & 15u) == 0u) {
        uint vector_index = lane;
        uint cache_off = pos * kv_dim + kv_h * HD + vector_index * 32u;
        uint current_off = kv_h * HD + vector_index * 32u;
        ((device half4*)(k_cache + cache_off))[0] =
            half4(((device const float4*)(k_current + current_off))[0]);
        ((device half4*)(k_cache + cache_off))[1] =
            half4(((device const float4*)(k_current + current_off))[1]);
        ((device half4*)(k_cache + cache_off))[2] =
            half4(((device const float4*)(k_current + current_off))[2]);
        ((device half4*)(k_cache + cache_off))[3] =
            half4(((device const float4*)(k_current + current_off))[3]);
        ((device half4*)(k_cache + cache_off))[4] =
            half4(((device const float4*)(k_current + current_off))[4]);
        ((device half4*)(k_cache + cache_off))[5] =
            half4(((device const float4*)(k_current + current_off))[5]);
        ((device half4*)(k_cache + cache_off))[6] =
            half4(((device const float4*)(k_current + current_off))[6]);
        ((device half4*)(k_cache + cache_off))[7] =
            half4(((device const float4*)(k_current + current_off))[7]);
        ((device half4*)(v_cache + cache_off))[0] =
            half4(((device const float4*)(v_current + current_off))[0]);
        ((device half4*)(v_cache + cache_off))[1] =
            half4(((device const float4*)(v_current + current_off))[1]);
        ((device half4*)(v_cache + cache_off))[2] =
            half4(((device const float4*)(v_current + current_off))[2]);
        ((device half4*)(v_cache + cache_off))[3] =
            half4(((device const float4*)(v_current + current_off))[3]);
        ((device half4*)(v_cache + cache_off))[4] =
            half4(((device const float4*)(v_current + current_off))[4]);
        ((device half4*)(v_cache + cache_off))[5] =
            half4(((device const float4*)(v_current + current_off))[5]);
        ((device half4*)(v_cache + cache_off))[6] =
            half4(((device const float4*)(v_current + current_off))[6]);
        ((device half4*)(v_cache + cache_off))[7] =
            half4(((device const float4*)(v_current + current_off))[7]);
    }
    uint start = window > 0u && kv_len > window ? kv_len - window : 0u;

    for (uint tile_start = start + sg * KV_TILE; tile_start < kv_len;
         tile_start += SPLITS * KV_TILE) {
        uint tile_count = min(KV_TILE, kv_len - tile_start);
        float partial[8];
        for (uint i = 0u; i < 8u; i++) {
            uint j = lane_group + i * 4u;
            float score = 0.0f;
            if (j < tile_count) {
                uint kv_off = (tile_start + j) * kv_dim + kv_h * HD;
                for (uint c = 0u; c < 4u; c++) {
                    uint vector_index = lane_in_group + c * 8u;
                    half4 k_half = tile_start + j == pos
                        ? half4(((device const float4*)(k_current + kv_h * HD))[vector_index])
                        : as_type<half4>(
                            ((device const ushort4*)(k_cache + kv_off))[vector_index]);
                    score += dot(q4[c], float4(k_half));
                }
            } else {
                score = -INFINITY;
            }
            partial[i] = score;
        }
        for (uint i = 0u; i < 8u; i++) {
            float score = partial[i];
            score += simd_shuffle_down(score, 4u);
            score += simd_shuffle_down(score, 2u);
            score += simd_shuffle_down(score, 1u);
            if (lane_in_group == 0u) {
                scores[sg][lane_group + i * 4u] = score * scale;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float tile_max = lane < tile_count ? scores[sg][lane] : -INFINITY;
        tile_max = simd_max(tile_max);
        float new_m = max(m, tile_max);
        float old_factor = m > -INFINITY ? exp(m - new_m) : 0.0f;
        acc *= old_factor;
        s *= old_factor;
        for (uint j = 0u; j < tile_count; j++) {
            float p = exp(scores[sg][j] - new_m);
            uint kv_off = (tile_start + j) * kv_dim + kv_h * HD;
            float4 v4;
            if (tile_start + j == pos) {
                v4 = float4(half4(((device const float4*)(v_current + kv_h * HD))[lane]));
            } else {
                v4 = float4(
                    as_type<half4>(((device const ushort4*)(v_cache + kv_off))[lane]));
            }
            acc += v4 * p;
            s += p;
        }
        m = new_m;
    }

    split_acc[sg][lane] = acc;
    if (lane == 0u) {
        split_m[sg] = m;
        split_s[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u && lane == 0u) {
        float global_m = -INFINITY;
        for (uint split = 0u; split < SPLITS; split++) {
            global_m = max(global_m, split_m[split]);
        }
        float denom = 0.0f;
        for (uint split = 0u; split < SPLITS; split++) {
            float factor = split_s[split] > 0.0f ? exp(split_m[split] - global_m) : 0.0f;
            denom += split_s[split] * factor;
        }
        merged_m = global_m;
        merged_s = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float4 merged = 0.0f;
        for (uint split = 0u; split < SPLITS; split++) {
            float factor =
                split_s[split] > 0.0f ? exp(split_m[split] - merged_m) : 0.0f;
            merged += split_acc[split][lane] * factor;
        }
        float4 gate4 = ((device const float4*)(gate + q_off))[lane];
        float4 sigmoid_gate = 1.0f / (1.0f + exp(-gate4));
        ((device float4*)(out + q_off))[lane] =
            (merged_s > 0.0f ? merged / merged_s : 0.0f) * sigmoid_gate;
    }
}
