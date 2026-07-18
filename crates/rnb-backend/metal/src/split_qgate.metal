#include <metal_stdlib>
using namespace metal;

// gated attention 의 q projection 출력(head 별 [query hd | gate hd] 인터리브)을
// 연속 query/gate 버퍼로 분리. host `decode_attention_post_qkv.rs:34-41` 1:1.
// 단순 copy (산술 없음 → 순서 무관, bit-identical). threadgroup=head, thread=차원 d
// (tg width = hd) 로 head_dim 을 lane 병렬화 (이전 tg=1 직렬 루프 대비 점유율 ↑).
kernel void split_qgate(
    device const float* q_full [[buffer(0)]], // [num_heads * hd * 2] read-only
    device float*       query  [[buffer(1)]], // [num_heads * hd] write
    device float*       gate   [[buffer(2)]], // [num_heads * hd] write
    constant uint&      hd     [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]])
{
    if (d >= hd) return;
    uint src = head * hd * 2u;
    uint dst = head * hd;
    query[dst + d] = q_full[src + d];
    gate[dst + d]  = q_full[src + hd + d];
}
