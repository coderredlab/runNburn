#include <metal_stdlib>
using namespace metal;

// pm132: GQA-grouped decode attention.
//
// 1 threadgroup = 1 KV head (not 1 query head).
// threadgroup 안에서 heads_per_group 개의 query head를 순차 처리.
// KV data는 KV entry당 1번만 global read → query head 수만큼 재사용.
// → KV memory traffic을 heads_per_group배 감소 (Qwen3.6-35B: 8x).
//
// split-K와 결합: grid = (num_kv_heads, num_splits).
// 각 threadgroup은 KV range의 chunk를 처리하고, query head별로
// partial (acc, m, s)를 쓴다. reduce가 합친다.
//
// 1 SIMD-group(32 lane). lane이 head_dim을 stride 32로 분할.
// heads_per_group만큼 루프 → register에 per-head softmax state 유지.

kernel void attn_decode_gqa_splitk_part(
    device const float*  q            [[buffer(0)]],
    device const ushort* k_cache      [[buffer(1)]],
    device const ushort* v_cache      [[buffer(2)]],
    device float*        partial_acc  [[buffer(3)]],  // [splits * num_heads * head_dim]
    device float*        partial_m    [[buffer(4)]],  // [splits * num_heads]
    device float*        partial_s    [[buffer(5)]],  // [splits * num_heads]
    constant uint&       num_heads    [[buffer(6)]],
    constant uint&       num_kv_heads [[buffer(7)]],
    constant uint&       head_dim     [[buffer(8)]],
    constant uint&       kv_len       [[buffer(9)]],
    constant float&      scale        [[buffer(10)]],
    constant uint&       num_splits   [[buffer(11)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint kv_h = gid.x;
    uint split = gid.y;
    if (kv_h >= num_kv_heads || split >= num_splits) return;

    uint heads_per_group = num_heads / num_kv_heads;
    uint kv_dim = num_kv_heads * head_dim;

    uint chunk = (kv_len + num_splits - 1u) / num_splits;
    uint start = split * chunk;
    uint end = min(kv_len, start + chunk);

    // per-query-head softmax state (register)
    // heads_per_group 최대 16 가정 (register 압박)
    float m_arr[16];
    float s_arr[16];
    float acc_arr[16][8]; // [head][lane-local dim]

    // Q load for all query heads in this group
    float qf[16][8];
    uint nloc = 0u;
    for (uint d = lane; d < head_dim; d += 32u) {
        for (uint qh = 0u; qh < heads_per_group; qh++) {
            uint h = kv_h * heads_per_group + qh;
            qf[qh][nloc] = (float)(half)q[h * head_dim + d];
        }
        nloc++;
    }
    for (uint qh = 0u; qh < heads_per_group; qh++) {
        m_arr[qh] = -INFINITY;
        s_arr[qh] = 0.0f;
        for (uint i = 0u; i < 8u; i++) acc_arr[qh][i] = 0.0f;
    }

    // KV loop — KV data 1번 read, 모든 query head에 재사용
    for (uint j = start; j < end; j++) {
        uint kv_off = j * kv_dim + kv_h * head_dim;

        // K read (1번)
        float kf[8];
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            kf[idx] = (float)as_type<half>(k_cache[kv_off + d]);
            idx++;
        }
        // V read (1번)
        float vf[8];
        idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            vf[idx] = (float)as_type<half>(v_cache[kv_off + d]);
            idx++;
        }

        // 각 query head 처리
        for (uint qh = 0u; qh < heads_per_group; qh++) {
            // QK^T
            float partial = 0.0f;
            for (uint i = 0u; i < nloc; i++) {
                partial += qf[qh][i] * kf[i];
            }
            float x = simd_sum(partial) * scale;

            // online softmax + V accumulate
            float m_q = m_arr[qh];
            float s_q = s_arr[qh];
            if (x > m_q) {
                bool rescale = (m_q > -INFINITY);
                float alpha = rescale ? exp(m_q - x) : 1.0f;
                if (rescale) s_q *= alpha;
                for (uint i = 0u; i < nloc; i++) {
                    float a = acc_arr[qh][i];
                    if (rescale) a *= alpha;
                    acc_arr[qh][i] = a + vf[i];
                }
                s_q += 1.0f;
                m_q = x;
            } else {
                float p = exp(x - m_q);
                for (uint i = 0u; i < nloc; i++) {
                    acc_arr[qh][i] += vf[i] * p;
                }
                s_q += p;
            }
            m_arr[qh] = m_q;
            s_arr[qh] = s_q;
        }
    }

    // partial 결과 쓰기 (query head별)
    for (uint qh = 0u; qh < heads_per_group; qh++) {
        uint h = kv_h * heads_per_group + qh;
        uint row = split * num_heads + h;
        if (lane == 0u) {
            partial_m[row] = m_arr[qh];
            partial_s[row] = s_arr[qh];
        }
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            partial_acc[row * head_dim + d] = acc_arr[qh][idx];
            idx++;
        }
    }
}

// reduce: 기존 attn_decode_splitk_reduce와 동일.
// grid = num_heads, 1 SIMD-group.
kernel void attn_decode_gqa_splitk_reduce(
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
