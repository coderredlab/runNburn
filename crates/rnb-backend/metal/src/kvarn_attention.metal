#include <metal_stdlib>
using namespace metal;

struct KvarnAttentionParams {
    uint kv_len;
    uint tail_start;
    uint sink_len;
    uint tail_len;
    uint num_blocks;
    uint num_heads;
    uint num_kv_heads;
    uint group;
    uint value_bits;
    uint block_bytes;
    uint key_packed_offset;
    uint key_scale_offset;
    uint key_zero_offset;
    uint key_token_scale_offset;
    uint value_packed_offset;
    uint value_channel_scale_offset;
    uint value_token_scale_offset;
    uint value_zero_offset;
    uint window_start;
    uint head_dim;
    float scale;
    float softcap;
};

inline void kvarn_hadamard(
    threadgroup float* values,
    uint lane,
    uint head_dim
) {
    for (uint span = 1u; span < head_dim; span <<= 1u) {
        if (lane < head_dim / 2u) {
            const uint base = (lane / span) * (span << 1u);
            const uint offset = lane % span;
            const float left = values[base + offset];
            const float right = values[base + span + offset];
            values[base + offset] = left + right;
            values[base + span + offset] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    values[lane] *= rsqrt((float)head_dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float kvarn_dot_reduce(
    float value,
    threadgroup float* partial,
    uint lane,
    uint head_dim
) {
    partial[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = head_dim / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partial[0];
}

kernel void kvarn_attention_decode(
    device float* out [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const uchar* packed_blocks [[buffer(2)]],
    device const half* sink_key [[buffer(3)]],
    device const half* sink_value [[buffer(4)]],
    device const half* tail_key [[buffer(5)]],
    device const half* tail_value [[buffer(6)]],
    constant KvarnAttentionParams& p [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    threadgroup float transform[512];
    threadgroup float partial[512];

    const uint heads_per_group = p.num_heads / p.num_kv_heads;
    const uint kv_head = head / heads_per_group;
    const uint row_width = p.num_kv_heads * p.head_dim;
    const float q_original = query[head * p.head_dim + lane];
    transform[lane] = q_original;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kvarn_hadamard(transform, lane, p.head_dim);
    const float q_rotated = transform[lane];

    float sink_max = -FLT_MAX;
    float sink_sum = 0.0f;
    float sink_acc = 0.0f;
    for (uint token = 0u; token < p.sink_len; ++token) {
        if (token < p.window_start || token >= p.kv_len) {
            continue;
        }
        const uint base = token * row_width + kv_head * p.head_dim;
        float score = kvarn_dot_reduce(
            q_original * (float)sink_key[base + lane],
            partial,
            lane,
            p.head_dim
        ) * p.scale;
        if (p.softcap > 0.0f) {
            score = p.softcap * tanh(score / p.softcap);
        }
        const float next_max = max(sink_max, score);
        const float old_scale = sink_max == -FLT_MAX ? 0.0f : exp(sink_max - next_max);
        const float probability = exp(score - next_max);
        sink_acc = sink_acc * old_scale + probability * (float)sink_value[base + lane];
        sink_sum = sink_sum * old_scale + probability;
        sink_max = next_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float quantized_max = -FLT_MAX;
    float quantized_sum = 0.0f;
    float quantized_acc = 0.0f;
    const uint key_packed_head = p.head_dim * (p.group / 2u);
    const uint value_pack = 8u / p.value_bits;
    const uint value_packed_row = p.head_dim / value_pack;
    const uint value_packed_head = p.group * value_packed_row;
    for (uint block = 0u; block < p.num_blocks; ++block) {
        device const uchar* record = packed_blocks + (ulong)block * p.block_bytes;
        device const half* key_scale = (device const half*)(record + p.key_scale_offset);
        device const half* key_zero = (device const half*)(record + p.key_zero_offset);
        device const half* key_token_scale =
            (device const half*)(record + p.key_token_scale_offset);
        device const half* value_channel_scale =
            (device const half*)(record + p.value_channel_scale_offset);
        device const half* value_token_scale =
            (device const half*)(record + p.value_token_scale_offset);
        device const half* value_zero = (device const half*)(record + p.value_zero_offset);
        for (uint token = 0u; token < p.group; ++token) {
            const uint global_token = p.sink_len + block * p.group + token;
            if (global_token < p.window_start || global_token >= p.kv_len) {
                continue;
            }
            const uint key_byte_index =
                kv_head * key_packed_head + lane * (p.group / 2u) + token / 2u;
            const uint key_byte = record[p.key_packed_offset + key_byte_index];
            const uint key_quant = (token & 1u) != 0u ? key_byte >> 4u : key_byte & 0x0fu;
            const uint key_channel = kv_head * p.head_dim + lane;
            const float key =
                ((float)key_quant * (float)key_scale[key_channel] + (float)key_zero[key_channel])
                * (float)key_token_scale[kv_head * p.group + token];
            float score = kvarn_dot_reduce(
                q_rotated * key,
                partial,
                lane,
                p.head_dim
            ) * p.scale;
            if (p.softcap > 0.0f) {
                score = p.softcap * tanh(score / p.softcap);
            }
            const float next_max = max(quantized_max, score);
            const float old_scale = quantized_max == -FLT_MAX
                ? 0.0f
                : exp(quantized_max - next_max);
            const float probability = exp(score - next_max);

            const uint value_byte_index = kv_head * value_packed_head
                + token * value_packed_row
                + lane / value_pack;
            const uint value_byte = record[p.value_packed_offset + value_byte_index];
            const uint value_mask = (1u << p.value_bits) - 1u;
            const uint value_quant =
                (value_byte >> ((lane % value_pack) * p.value_bits)) & value_mask;
            const float value =
                ((float)value_quant * (float)value_token_scale[kv_head * p.group + token]
                    + (float)value_zero[kv_head * p.group + token])
                * (float)value_channel_scale[kv_head * p.head_dim + lane];
            quantized_acc = quantized_acc * old_scale + probability * value;
            quantized_sum = quantized_sum * old_scale + probability;
            quantized_max = next_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    transform[lane] = quantized_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kvarn_hadamard(transform, lane, p.head_dim);
    const float quantized_acc_original = transform[lane];

    float tail_max = -FLT_MAX;
    float tail_sum = 0.0f;
    float tail_acc = 0.0f;
    for (uint token = 0u; token < p.tail_len; ++token) {
        const uint global_token = p.tail_start + token;
        if (global_token < p.window_start || global_token >= p.kv_len) {
            continue;
        }
        const uint base = token * row_width + kv_head * p.head_dim;
        float score = kvarn_dot_reduce(
            q_original * (float)tail_key[base + lane],
            partial,
            lane,
            p.head_dim
        ) * p.scale;
        if (p.softcap > 0.0f) {
            score = p.softcap * tanh(score / p.softcap);
        }
        const float next_max = max(tail_max, score);
        const float old_scale = tail_max == -FLT_MAX ? 0.0f : exp(tail_max - next_max);
        const float probability = exp(score - next_max);
        tail_acc = tail_acc * old_scale + probability * (float)tail_value[base + lane];
        tail_sum = tail_sum * old_scale + probability;
        tail_max = next_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float first_max = max(sink_max, quantized_max);
    const float sink_weight = sink_sum > 0.0f ? exp(sink_max - first_max) : 0.0f;
    const float quantized_weight = quantized_sum > 0.0f
        ? exp(quantized_max - first_max)
        : 0.0f;
    float merged_acc = sink_acc * sink_weight + quantized_acc_original * quantized_weight;
    float merged_sum = sink_sum * sink_weight + quantized_sum * quantized_weight;
    float merged_max = first_max;
    if (tail_sum > 0.0f) {
        const float next_max = merged_sum > 0.0f ? max(merged_max, tail_max) : tail_max;
        const float old_weight = merged_sum > 0.0f ? exp(merged_max - next_max) : 0.0f;
        const float tail_weight = exp(tail_max - next_max);
        merged_acc = merged_acc * old_weight + tail_acc * tail_weight;
        merged_sum = merged_sum * old_weight + tail_sum * tail_weight;
    }
    out[head * p.head_dim + lane] = merged_sum > 0.0f ? merged_acc / merged_sum : 0.0f;
}

// ── split-K KVarN decode attention (compute-bound dequant → high occupancy) ──
// KVarn attention 은 dequant(bit extract+scale+Hadamard) 이 compute-bound 이라
// single-kernel(head 당 1 threadgroup=16개)은 GPU 저점유 → decode chain serial cb 에서
// GDN 과 직렬화되며 idle. block(quantized) 을 KV 축으로 split 해 num_heads*num_splits
// threadgroup 으로 GPU 를 채운다(int8 attn_decode_i8_splitk 패턴). sink/tail 은 작아서
// reduce 에서 처리. block acc 는 Hadamard rotated 공간(linear)이라 partial 을 combine 후
// 한 번만 un-Hadamard.

inline void kvarn_hadamard_sg(threadgroup float* values, uint lane, uint head_dim) {
    for (uint span = 1u; span < head_dim; span <<= 1u) {
        for (uint b = lane; b < head_dim / 2u; b += 32u) {
            const uint base = (b / span) * (span << 1u);
            const uint offset = b % span;
            const float left = values[base + offset];
            const float right = values[base + span + offset];
            values[base + offset] = left + right;
            values[base + span + offset] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = rsqrt((float)head_dim);
    for (uint d = lane; d < head_dim; d += 32u) {
        values[d] *= norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// part: 1 threadgroup = (query head, block split), 32 lane. 담당 block chunk 의
// rotated-space online-softmax partial (m, s, acc) 을 row=split*num_heads+h 에 기록.
kernel void kvarn_attention_decode_splitk_part(
    device float* partial_acc [[buffer(0)]],
    device float* partial_m [[buffer(1)]],
    device float* partial_s [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const uchar* packed_blocks [[buffer(4)]],
    constant KvarnAttentionParams& p [[buffer(5)]],
    constant uint& num_splits [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint h = gid.x;
    const uint split = gid.y;
    if (h >= p.num_heads || split >= num_splits) {
        return;
    }
    const uint hd = p.head_dim;
    const uint heads_per_group = p.num_heads / p.num_kv_heads;
    const uint kv_head = h / heads_per_group;
    const uint kv_base = kv_head * hd;
    const uint row = split * p.num_heads + h;

    threadgroup float hbuf[512];
    for (uint d = lane; d < hd; d += 32u) {
        hbuf[d] = query[h * hd + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kvarn_hadamard_sg(hbuf, lane, hd);
    float q_rot[16];
    uint nloc = 0u;
    for (uint d = lane; d < hd; d += 32u) {
        q_rot[nloc++] = hbuf[d];
    }

    const uint block_chunk = (p.num_blocks + num_splits - 1u) / num_splits;
    const uint block_start = split * block_chunk;
    const uint block_end = min(p.num_blocks, block_start + block_chunk);

    const uint key_packed_head = hd * (p.group / 2u);
    const uint value_pack = 8u / p.value_bits;
    const uint value_packed_row = hd / value_pack;
    const uint value_packed_head = p.group * value_packed_row;
    const uint value_mask = (1u << p.value_bits) - 1u;

    float m = -FLT_MAX;
    float s = 0.0f;
    float acc[16];
    for (uint i = 0u; i < nloc; ++i) acc[i] = 0.0f;
    for (uint block = block_start; block < block_end; ++block) {
        device const uchar* record = packed_blocks + (ulong)block * p.block_bytes;
        device const half* key_scale = (device const half*)(record + p.key_scale_offset);
        device const half* key_zero = (device const half*)(record + p.key_zero_offset);
        device const half* key_token_scale =
            (device const half*)(record + p.key_token_scale_offset);
        device const half* value_channel_scale =
            (device const half*)(record + p.value_channel_scale_offset);
        device const half* value_token_scale =
            (device const half*)(record + p.value_token_scale_offset);
        device const half* value_zero = (device const half*)(record + p.value_zero_offset);
        for (uint token = 0u; token < p.group; ++token) {
            const uint global_token = p.sink_len + block * p.group + token;
            if (global_token < p.window_start || global_token >= p.kv_len) {
                continue;
            }
            const float k_ts = (float)key_token_scale[kv_head * p.group + token];
            float partial = 0.0f;
            uint i = 0u;
            for (uint d = lane; d < hd; d += 32u) {
                const uint kbi = kv_head * key_packed_head + d * (p.group / 2u) + token / 2u;
                const uint kb = record[p.key_packed_offset + kbi];
                const uint kq = (token & 1u) != 0u ? kb >> 4u : kb & 0x0fu;
                const uint kc = kv_base + d;
                const float key =
                    ((float)kq * (float)key_scale[kc] + (float)key_zero[kc]) * k_ts;
                partial += q_rot[i] * key;
                ++i;
            }
            float score = simd_sum(partial) * p.scale;
            if (p.softcap > 0.0f) {
                score = p.softcap * tanh(score / p.softcap);
            }
            const float next_max = max(m, score);
            const float old_scale = m == -FLT_MAX ? 0.0f : exp(m - next_max);
            const float prob = exp(score - next_max);
            const float v_ts = (float)value_token_scale[kv_head * p.group + token];
            const float v_z = (float)value_zero[kv_head * p.group + token];
            i = 0u;
            for (uint d = lane; d < hd; d += 32u) {
                const uint vbi =
                    kv_head * value_packed_head + token * value_packed_row + d / value_pack;
                const uint vb = record[p.value_packed_offset + vbi];
                const uint vq = (vb >> ((d % value_pack) * p.value_bits)) & value_mask;
                const float value =
                    ((float)vq * v_ts + v_z) * (float)value_channel_scale[kv_base + d];
                acc[i] = acc[i] * old_scale + prob * value;
                ++i;
            }
            s = s * old_scale + prob;
            m = next_max;
        }
    }
    if (lane == 0u) {
        partial_m[row] = m;
        partial_s[row] = s;
    }
    uint i = 0u;
    for (uint d = lane; d < hd; d += 32u) {
        partial_acc[row * hd + d] = acc[i];
        ++i;
    }
}

// reduce: 1 threadgroup = query head, 32 lane. block partial(rotated) combine →
// un-Hadamard → sink/tail(원 공간) 합산 → merge → out.
kernel void kvarn_attention_decode_splitk_reduce(
    device float* out [[buffer(0)]],
    device const float* partial_acc [[buffer(1)]],
    device const float* partial_m [[buffer(2)]],
    device const float* partial_s [[buffer(3)]],
    device const float* query [[buffer(4)]],
    device const half* sink_key [[buffer(5)]],
    device const half* sink_value [[buffer(6)]],
    device const half* tail_key [[buffer(7)]],
    device const half* tail_value [[buffer(8)]],
    constant KvarnAttentionParams& p [[buffer(9)]],
    constant uint& num_splits [[buffer(10)]],
    uint h [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (h >= p.num_heads) {
        return;
    }
    const uint hd = p.head_dim;
    const uint heads_per_group = p.num_heads / p.num_kv_heads;
    const uint kv_head = h / heads_per_group;
    const uint kv_base = kv_head * hd;
    const uint row_width = p.num_kv_heads * hd;

    threadgroup float hbuf[512];
    float q_orig[16];
    uint nloc = 0u;
    for (uint d = lane; d < hd; d += 32u) {
        q_orig[nloc++] = query[h * hd + d];
    }

    // block partial combine (rotated space)
    float m_blk = -FLT_MAX;
    for (uint split = 0u; split < num_splits; ++split) {
        m_blk = max(m_blk, partial_m[split * p.num_heads + h]);
    }
    float s_blk = 0.0f;
    float acc_blk[16];
    for (uint i = 0u; i < nloc; ++i) acc_blk[i] = 0.0f;
    for (uint split = 0u; split < num_splits; ++split) {
        const uint prow = split * p.num_heads + h;
        const float sp = partial_s[prow];
        const float factor = sp > 0.0f ? exp(partial_m[prow] - m_blk) : 0.0f;
        s_blk += sp * factor;
        uint i = 0u;
        for (uint d = lane; d < hd; d += 32u) {
            acc_blk[i] += partial_acc[prow * hd + d] * factor;
            ++i;
        }
    }
    // un-Hadamard combined block acc (rotated → original)
    for (uint d = lane; d < hd; d += 32u) {
        hbuf[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        uint i = 0u;
        for (uint d = lane; d < hd; d += 32u) { hbuf[d] = acc_blk[i]; ++i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kvarn_hadamard_sg(hbuf, lane, hd);
    {
        uint i = 0u;
        for (uint d = lane; d < hd; d += 32u) { acc_blk[i] = hbuf[d]; ++i; }
    }

    // sink (raw f16, original space)
    float m_sink = -FLT_MAX;
    float s_sink = 0.0f;
    float acc_sink[16];
    for (uint i = 0u; i < nloc; ++i) acc_sink[i] = 0.0f;
    for (uint token = 0u; token < p.sink_len; ++token) {
        if (token < p.window_start || token >= p.kv_len) {
            continue;
        }
        const uint base = token * row_width + kv_base;
        float partial = 0.0f;
        uint i = 0u;
        for (uint d = lane; d < hd; d += 32u) {
            partial += q_orig[i] * (float)sink_key[base + d];
            ++i;
        }
        float score = simd_sum(partial) * p.scale;
        if (p.softcap > 0.0f) {
            score = p.softcap * tanh(score / p.softcap);
        }
        const float next_max = max(m_sink, score);
        const float old_scale = m_sink == -FLT_MAX ? 0.0f : exp(m_sink - next_max);
        const float prob = exp(score - next_max);
        i = 0u;
        for (uint d = lane; d < hd; d += 32u) {
            acc_sink[i] = acc_sink[i] * old_scale + prob * (float)sink_value[base + d];
            ++i;
        }
        s_sink = s_sink * old_scale + prob;
        m_sink = next_max;
    }

    // tail (raw f16, original space)
    float m_tail = -FLT_MAX;
    float s_tail = 0.0f;
    float acc_tail[16];
    for (uint i = 0u; i < nloc; ++i) acc_tail[i] = 0.0f;
    for (uint token = 0u; token < p.tail_len; ++token) {
        const uint global_token = p.tail_start + token;
        if (global_token < p.window_start || global_token >= p.kv_len) {
            continue;
        }
        const uint base = token * row_width + kv_base;
        float partial = 0.0f;
        uint i = 0u;
        for (uint d = lane; d < hd; d += 32u) {
            partial += q_orig[i] * (float)tail_key[base + d];
            ++i;
        }
        float score = simd_sum(partial) * p.scale;
        if (p.softcap > 0.0f) {
            score = p.softcap * tanh(score / p.softcap);
        }
        const float next_max = max(m_tail, score);
        const float old_scale = m_tail == -FLT_MAX ? 0.0f : exp(m_tail - next_max);
        const float prob = exp(score - next_max);
        i = 0u;
        for (uint d = lane; d < hd; d += 32u) {
            acc_tail[i] = acc_tail[i] * old_scale + prob * (float)tail_value[base + d];
            ++i;
        }
        s_tail = s_tail * old_scale + prob;
        m_tail = next_max;
    }

    // merge sink + block + tail
    const float first_max = max(m_sink, m_blk);
    const float sink_weight = s_sink > 0.0f ? exp(m_sink - first_max) : 0.0f;
    const float blk_weight = s_blk > 0.0f ? exp(m_blk - first_max) : 0.0f;
    float merged_sum = s_sink * sink_weight + s_blk * blk_weight;
    float merged_max = first_max;
    float merged_acc[16];
    for (uint i = 0u; i < nloc; ++i) {
        merged_acc[i] = acc_sink[i] * sink_weight + acc_blk[i] * blk_weight;
    }
    if (s_tail > 0.0f) {
        const float next_max = merged_sum > 0.0f ? max(merged_max, m_tail) : m_tail;
        const float old_weight = merged_sum > 0.0f ? exp(merged_max - next_max) : 0.0f;
        const float tail_weight = exp(m_tail - next_max);
        for (uint i = 0u; i < nloc; ++i) {
            merged_acc[i] = merged_acc[i] * old_weight + acc_tail[i] * tail_weight;
        }
        merged_sum = merged_sum * old_weight + s_tail * tail_weight;
    }
    const float inv = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    uint i = 0u;
    for (uint d = lane; d < hd; d += 32u) {
        out[h * hd + d] = merged_acc[i] * inv;
        ++i;
    }
}
