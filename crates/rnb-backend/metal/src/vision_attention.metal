#include <metal_stdlib>
using namespace metal;

constant constexpr uint QUERIES_PER_GROUP = 4u;

kernel void vision_full_attention(
    device const float *qkv    [[buffer(0)]],
    device float       *output [[buffer(1)]],
    constant uint      &embedding_length [[buffer(2)]],
    constant uint      &head_count       [[buffer(3)]],
    constant uint      &sequence_length  [[buffer(4)]],
    constant float     &scale            [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    uint query_blocks = (sequence_length + QUERIES_PER_GROUP - 1u) / QUERIES_PER_GROUP;
    uint head = group / query_blocks;
    uint query = (group % query_blocks) * QUERIES_PER_GROUP + simdgroup;
    uint head_dim = embedding_length / head_count;
    uint qkv_width = embedding_length * 3u;
    bool active = query < sequence_length && head < head_count;

    float accumulator[3] = {0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY;
    float denominator = 0.0f;

    for (uint key = 0u; key < sequence_length; key++) {
        float partial = 0.0f;
        if (active) {
            for (uint dimension = lane; dimension < head_dim; dimension += 32u) {
                float q = qkv[(ulong)query * qkv_width + head * head_dim + dimension];
                float k = qkv[(ulong)key * qkv_width + embedding_length + head * head_dim + dimension];
                partial += q * k;
            }
        }
        float score = simd_sum(partial) * scale;
        float next_maximum = max(maximum, score);
        float previous_scale = exp(maximum - next_maximum);
        float probability = exp(score - next_maximum);
        denominator = denominator * previous_scale + probability;
        if (active) {
            uint slot = 0u;
            for (uint dimension = lane; dimension < head_dim; dimension += 32u) {
                float value = qkv[(ulong)key * qkv_width + 2u * embedding_length + head * head_dim + dimension];
                accumulator[slot] = accumulator[slot] * previous_scale + probability * value;
                slot++;
            }
        }
        maximum = next_maximum;
    }

    if (active) {
        float inverse = 1.0f / denominator;
        uint slot = 0u;
        for (uint dimension = lane; dimension < head_dim; dimension += 32u) {
            output[(ulong)query * embedding_length + head * head_dim + dimension] =
                accumulator[slot] * inverse;
            slot++;
        }
    }
}
