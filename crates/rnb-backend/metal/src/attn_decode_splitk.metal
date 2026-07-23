#include <metal_stdlib>
using namespace metal;

// pm132: split-K f16 KV decode attention.
//
// part:   grid = (num_heads, num_splits). 각 threadgroup은 KV range의 chunk를
//         online-softmax로 처리하고 partial (acc, m, s)를 쓴다.
//         1 SIMD-group(32 lane), lane이 head_dim을 stride 32로 분할.
// reduce: grid = num_heads. split-local softmax 상태를 log-sum-exp rescaling으로
//         합치고 최종 output을 쓴다.
//
// attn_decode.metal(non-split-K)과 동일한 online softmax 산식.
// 긴 context에서 num_heads만으로는 GPU 점유율 부족 → split으로 병렬화.

kernel void attn_decode_splitk_part(
    device const float*  q            [[buffer(0)]],
    device const ushort* k_cache      [[buffer(1)]],  // f16 KV
    device const ushort* v_cache      [[buffer(2)]],
    device float*        partial_acc  [[buffer(3)]],
    device float*        partial_m    [[buffer(4)]],
    device float*        partial_s    [[buffer(5)]],
    constant uint&       num_heads    [[buffer(6)]],
    constant uint&       num_kv_heads [[buffer(7)]],
    constant uint&       head_dim     [[buffer(8)]],
    constant uint&       kv_len       [[buffer(9)]],
    constant float&      scale        [[buffer(10)]],
    constant uint&       num_splits   [[buffer(11)]],
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

    // Q load (f16 round → f32, attn_decode.metal과 동일)
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

        // QK^T
        float partial = 0.0f;
        uint idx = 0u;
        for (uint d = lane; d < head_dim; d += 32u) {
            float kf = (float)as_type<half>(k_cache[kv_off + d]);
            partial += qf[idx] * kf;
            idx++;
        }
        float x = simd_sum(partial) * scale;

        // branched online softmax (attn_decode.metal과 동일 산식)
        if (x > m) {
            bool rescale = (m > -INFINITY);
            float alpha = rescale ? exp(m - x) : 1.0f;
            if (rescale) s *= alpha;
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float a = acc[idx];
                if (rescale) a *= alpha;
                float vv = (float)as_type<half>(v_cache[kv_off + d]);
                acc[idx] = a + vv;
                idx++;
            }
            s += 1.0f;
            m = x;
        } else {
            float p = exp(x - m);
            idx = 0u;
            for (uint d = lane; d < head_dim; d += 32u) {
                float vv = (float)as_type<half>(v_cache[kv_off + d]);
                acc[idx] += vv * p;
                idx++;
            }
            s += p;
        }
    }

    // partial 결과 쓰기
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

kernel void attn_decode_splitk_reduce(
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

    // global max
    float m = -INFINITY;
    for (uint split = 0u; split < num_splits; split++) {
        uint row = split * num_heads + h;
        m = max(m, partial_m[row]);
    }

    // log-sum-exp rescaling + accumulate
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
