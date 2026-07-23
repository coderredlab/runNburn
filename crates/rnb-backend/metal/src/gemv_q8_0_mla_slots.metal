#include <metal_stdlib>
using namespace metal;

// pm113: GLM MLA prefill slot-batch GEMV. slot(token*HEADS + head)이 grid y 축.
// weight 는 head 별로 다르고 (head = slot % HEADS), input/out 은 slot 별 연속 배치.
// row 처리 구조는 gemv_q8_0_coalesced (NR0=2 multi-row + activation reuse) 그대로.
//
// Q8_0 block layout (34 bytes): 0-1 d(half) / 2-33 qs[32](i8). num_blocks = K/32.
kernel void gemv_q8_0_mla_slots(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device float*       out          [[buffer(2)]],
    constant uint&      N            [[buffer(3)]],  // rows per head
    constant uint&      K            [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      HEADS        [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const uint slot = tg.y;
    const uint head = slot % HEADS;
    const uint first_row = tg.x * 2u;
    if (first_row >= N) return;
    const bool has_row1 = (first_row + 1u) < N;

    constexpr ushort NQ = 8u;
    const uint nb = K / 32u;

    const ushort ix = lane / 4u;   // 0..7  (block index within simdgroup stride)
    const ushort il = lane % 4u;   // 0..3  (NQ-chunk within a 32-elem block)

    device const uchar* x0 = weight_bytes + weight_byte_offset
        + (head * N + first_row) * (nb * 34u);
    device const uchar* x1 = x0 + nb * 34u;

    device const float* yb = input + slot * K + (uint)ix * 32u + (uint)il * NQ;

    float yl[NQ];
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (uint ib = ix; ib < nb; ib += 8u) {
        for (ushort i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        // row 0
        {
            device const uchar* blk = x0 + ib * 34u;
            ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
            float d = (float)as_type<half>(d_bits);
            device const char* qs = (device const char*)(blk + 2u) + (uint)il * NQ;
            float sumq = 0.f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * yl[i];
            }
            sumf0 += sumq * d;
        }

        // row 1 (있을 때만 — weight OOB 방지)
        if (has_row1) {
            device const uchar* blk = x1 + ib * 34u;
            ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
            float d = (float)as_type<half>(d_bits);
            device const char* qs = (device const char*)(blk + 2u) + (uint)il * NQ;
            float sumq = 0.f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += (float)qs[i] * yl[i];
            }
            sumf1 += sumq * d;
        }

        yb += 8u * 32u;
    }

    float t0 = simd_sum(sumf0);
    if (lane == 0u) out[slot * N + first_row] = t0;
    if (has_row1) {
        float t1 = simd_sum(sumf1);
        if (lane == 0u) out[slot * N + first_row + 1u] = t1;
    }
}

// pm118 연장: q_b 출력 [seq × heads×qk_dim] 에서 slot(token×heads+head)별
// 앞 q_nope_dim 만 연속으로 뽑아 kb slots 입력 [slots × q_nope_dim] 을 만든다.
// (CPU q_nope packing 의 GPU 이식 — front fused chain 중간 단계.)
kernel void glm_mla_qnope_pack(
    device const float* q          [[buffer(0)]], // seq × (heads*qk_dim)
    device float*       out        [[buffer(1)]], // (seq*heads) × q_nope_dim
    constant uint&      heads      [[buffer(2)]],
    constant uint&      qk_dim     [[buffer(3)]],
    constant uint&      q_nope_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) // x = dim, y = slot
{
    const uint d = gid.x;
    const uint slot = gid.y;
    if (d >= q_nope_dim) {
        return;
    }
    const uint token = slot / heads;
    const uint head = slot % heads;
    out[(ulong)slot * q_nope_dim + d] =
        q[(ulong)token * heads * qk_dim + head * qk_dim + d];
}

// pm119: kv_raw 의 rms(kv_rank 구간) + rope(rope 구간) 를 f16 으로 변환해
// cache buffer 의 (pos_start+token) row 에 직접 쓴다 — CPU rope 스테이지의
// GPU 이식 (rms 는 rms_norm_batch 구조, rope 는 rope_partial 수식 1:1:
// host precompute theta_scale, 인접페어, angle 누적곱).
kernel void glm_mla_kv_rms_rope_f16(
    device const float* kv_raw      [[buffer(0)]], // seq × kv_width
    device const float* norm_w      [[buffer(1)]], // kv_rank
    device ushort*      cache       [[buffer(2)]], // (pos_start+seq) × kv_width
    constant uint&      kv_rank     [[buffer(3)]],
    constant uint&      kv_width    [[buffer(4)]],
    constant uint&      pos_start   [[buffer(5)]],
    constant float&     eps         [[buffer(6)]],
    constant float&     theta_scale [[buffer(7)]],
    uint token   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    const uint in_base = token * kv_width;
    const ulong out_base = (ulong)(pos_start + token) * kv_width;
    float sum = 0.0f;
    for (uint i = tid; i < kv_rank; i += tg_size) {
        float v = kv_raw[in_base + i];
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
    float inv_rms = rsqrt(partial[0] / (float)kv_rank + eps);
    for (uint i = tid; i < kv_rank; i += tg_size) {
        half h = (half)(kv_raw[in_base + i] * inv_rms * norm_w[i]);
        cache[out_base + i] = as_type<ushort>(h);
    }
    if (tid == 0u) {
        float angle = (float)(pos_start + token);
        for (uint i = kv_rank; i < kv_width; i += 2u) {
            float cos_a = cos(angle);
            float sin_a = sin(angle);
            float x0 = kv_raw[in_base + i];
            float x1 = kv_raw[in_base + i + 1u];
            cache[out_base + i] = as_type<ushort>((half)(x0 * cos_a - x1 * sin_a));
            cache[out_base + i + 1u] = as_type<ushort>((half)(x0 * sin_a + x1 * cos_a));
            angle *= theta_scale;
        }
    }
}

// pm119: q_b 출력에서 slot 별 [q_nope_dim, qk_dim) 구간을 뽑아 rope 를 적용해
// q_pe (f32) 를 만든다 — CPU q_pe pack+rope 의 GPU 이식 (rope_partial 수식).
// 1 thread = 1 slot (rope_dim 인접페어 직렬 누적곱).
kernel void glm_mla_qpe_rope(
    device const float* q           [[buffer(0)]], // seq × heads×qk_dim
    device float*       q_pe        [[buffer(1)]], // slots × rope_dim
    constant uint&      heads       [[buffer(2)]],
    constant uint&      qk_dim      [[buffer(3)]],
    constant uint&      q_nope_dim  [[buffer(4)]],
    constant uint&      rope_dim    [[buffer(5)]],
    constant uint&      pos_start   [[buffer(6)]],
    constant float&     theta_scale [[buffer(7)]],
    constant uint&      slot_count  [[buffer(8)]],
    uint slot [[thread_position_in_grid]])
{
    if (slot >= slot_count) {
        return;
    }
    const uint token = slot / heads;
    const uint head = slot % heads;
    const ulong in_base = (ulong)token * heads * qk_dim + head * qk_dim + q_nope_dim;
    const ulong out_base = (ulong)slot * rope_dim;
    float angle = (float)(pos_start + token);
    for (uint i = 0u; i < rope_dim; i += 2u) {
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float x0 = q[in_base + i];
        float x1 = q[in_base + i + 1u];
        q_pe[out_base + i] = x0 * cos_a - x1 * sin_a;
        q_pe[out_base + i + 1u] = x0 * sin_a + x1 * cos_a;
        angle *= theta_scale;
    }
}
