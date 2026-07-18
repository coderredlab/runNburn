#include <metal_stdlib>
using namespace metal;
// per-slot symmetric int8 KV append (pm22). grid = num_kv_heads tg, tg = 32 lane.
// 각 tg = 1 kv_head 의 head_dim 벡터 → max-abs(simd_max) → scale → quantize(rint=RNE).
// CPU quantize_slot_i8_ref 와 1:1 일치. max-abs 는 항상 ≥0 이라 head_dim<32 일 때
// 미사용 lane 의 초기값 0.0 이 simd_max 결과를 오염시키지 않는다.
kernel void kv_append_i8(
    device const float* k_f32        [[buffer(0)]],
    device const float* v_f32        [[buffer(1)]],
    device char*        k_cache      [[buffer(2)]],
    device char*        v_cache      [[buffer(3)]],
    device float*       k_scale      [[buffer(4)]],
    device float*       v_scale      [[buffer(5)]],
    constant uint&      head_dim     [[buffer(6)]],
    constant uint&      num_kv_heads [[buffer(7)]],
    constant uint&      pos          [[buffer(8)]],
    uint kv_h [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (kv_h >= num_kv_heads) return;
    uint kv_dim = num_kv_heads * head_dim;
    uint base = kv_h * head_dim;          // f32 입력은 단일 토큰([kv_dim]) 기준 head offset
    uint slot = pos * kv_dim + base;      // device cache 의 (pos, kv_h) 슬롯
    uint sidx = pos * num_kv_heads + kv_h; // per-slot scale 인덱스

    // 1) lane-strided max-abs 누적 → SIMD-group reduce
    float kmax = 0.0f, vmax = 0.0f;
    for (uint d = lane; d < head_dim; d += 32u) {
        kmax = max(kmax, fabs(k_f32[base + d]));
        vmax = max(vmax, fabs(v_f32[base + d]));
    }
    kmax = simd_max(kmax);
    vmax = simd_max(vmax);

    // 2) scale = max|v|/127 (max==0 → scale 0 = zero slot, CPU ref 와 동일)
    float ks = (kmax > 0.0f) ? (kmax / 127.0f) : 0.0f;
    float vs = (vmax > 0.0f) ? (vmax / 127.0f) : 0.0f;
    if (lane == 0u) {
        k_scale[sidx] = ks;
        v_scale[sidx] = vs;
    }

    // 3) quantize: q = clamp(rint(v/scale), -127, 127). rint=RNE → CPU round_ties_even 일치.
    float kinv = (ks > 0.0f) ? (1.0f / ks) : 0.0f;
    float vinv = (vs > 0.0f) ? (1.0f / vs) : 0.0f;
    for (uint d = lane; d < head_dim; d += 32u) {
        k_cache[slot + d] = (char)clamp(rint(k_f32[base + d] * kinv), -127.0f, 127.0f);
        v_cache[slot + d] = (char)clamp(rint(v_f32[base + d] * vinv), -127.0f, 127.0f);
    }
}
