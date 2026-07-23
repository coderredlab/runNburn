#include <metal_stdlib>
using namespace metal;

// GLM MLA prefill attention (absorbed form) — pm116.
//
// slot = token * head_count + head. 각 slot 이 causal 범위 [0, pos_start+token]
// 의 compressed cache row (f16 bits, kv_width = kv_rank + rope_dim) 에 대해:
//   score_j = (q_absorbed[slot]·row[j][..kv_rank] + q_pe[slot]·row[j][kv_rank..]) * scale
//   softmax 후 latent_out[slot][d] = Σ_j p_j * row[j][d]  (d < kv_rank)
//
// CPU 참조(glm_dsa::prefill_layer 스칼라 루프)는 2-pass softmax, 여기는
// attn_decode 와 같은 branched online softmax — 수치 순서만 다르고 수학은 동일.
// 64 head 가 같은 cache row 를 읽는 MQA 구조라 인접 slot 의 read 는 L2 공유.
//
// 1 threadgroup = 1 SIMD-group(32 lane) = 1 slot. lane 이 dim 을 stride 32 분할.
// kv_rank <= 512, rope_dim <= 64 가정 (dispatch 측에서 검사).
kernel void glm_mla_prefill_attn(
    device const float*  q_absorbed [[buffer(0)]], // slots × kv_rank
    device const float*  q_pe       [[buffer(1)]], // slots × rope_dim
    device const ushort* cache      [[buffer(2)]], // cache_len × kv_width (f16 bits)
    device float*        latent_out [[buffer(3)]], // slots × kv_rank
    constant uint&       head_count [[buffer(4)]],
    constant uint&       kv_rank    [[buffer(5)]],
    constant uint&       rope_dim   [[buffer(6)]],
    constant uint&       pos_start  [[buffer(7)]],
    constant float&      scale      [[buffer(8)]],
    uint slot [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint kv_width = kv_rank + rope_dim;
    const uint token = slot / head_count;
    const uint attend_len = pos_start + token + 1u;

    // lane 담당 q 값 preload (dot 은 kv_width 전체를 stride 32 로 분할).
    float qf[18]; // ceil(576/32)
    uint nq = 0u;
    for (uint d = lane; d < kv_width; d += 32u) {
        qf[nq++] = d < kv_rank ? q_absorbed[(ulong)slot * kv_rank + d]
                               : q_pe[(ulong)slot * rope_dim + (d - kv_rank)];
    }
    // lane 담당 latent 누적 (kv_rank 만, stride 32).
    float acc[16]; // ceil(512/32)
    for (uint i = 0u; i < 16u; i++) acc[i] = 0.0f;

    float m = -INFINITY;
    float s = 0.0f;
    for (uint j = 0u; j < attend_len; j++) {
        device const ushort* row = cache + (ulong)j * kv_width;
        float partial = 0.0f;
        uint i = 0u;
        for (uint d = lane; d < kv_width; d += 32u, i++) {
            partial += qf[i] * (float)as_type<half>(row[d]);
        }
        float x = simd_sum(partial) * scale;
        // branched online softmax — 모든 lane 이 동일 x (동기화 불필요).
        if (x > m) {
            float alpha = exp(m - x);
            s = s * alpha + 1.0f;
            i = 0u;
            for (uint d = lane; d < kv_rank; d += 32u, i++) {
                acc[i] = acc[i] * alpha + (float)as_type<half>(row[d]);
            }
            m = x;
        } else {
            float p = exp(x - m);
            s += p;
            i = 0u;
            for (uint d = lane; d < kv_rank; d += 32u, i++) {
                acc[i] += p * (float)as_type<half>(row[d]);
            }
        }
    }
    float inv_s = 1.0f / s;
    uint i = 0u;
    for (uint d = lane; d < kv_rank; d += 32u, i++) {
        latent_out[(ulong)slot * kv_rank + d] = acc[i] * inv_s;
    }
}
