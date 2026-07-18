#include <metal_stdlib>
using namespace metal;

// GDN GQA: l2_norm 된 q/k(num_k_heads)를 num_v_heads 로 순환 브로드캐스트(repeat).
// host `gdn_prefill.rs:99 repeat_qk_for_value_heads` 1:1 (ggml_repeat_4d = 교차 아닌 순환).
// 단순 gather copy (산술 없음 → bit-identical). q/k 같은 패턴이라 한 커널에서 둘 다.
// num_v_heads == num_k_heads 면 src==gid → identity copy (특수 분기 불요).
// 출력 dst = gid = (t*num_v_heads + vh)*head_k_dim + d. flat 1D grid (1 thread = 1 element).
kernel void repeat_qk(
    device const float* q_in       [[buffer(0)]], // [seq_len * num_k_heads * head_k_dim] read-only
    device const float* k_in       [[buffer(1)]], // [seq_len * num_k_heads * head_k_dim] read-only
    device float*       q_out      [[buffer(2)]], // [seq_len * num_v_heads * head_k_dim] write
    device float*       k_out      [[buffer(3)]], // [seq_len * num_v_heads * head_k_dim] write
    constant uint&      seq_len    [[buffer(4)]],
    constant uint&      num_k_heads[[buffer(5)]],
    constant uint&      num_v_heads[[buffer(6)]],
    constant uint&      head_k_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = seq_len * num_v_heads * head_k_dim;
    if (gid >= total) return;
    uint d = gid % head_k_dim;
    uint tmp = gid / head_k_dim;       // = t * num_v_heads + vh
    uint vh = tmp % num_v_heads;
    uint t = tmp / num_v_heads;
    uint kh = vh % num_k_heads;
    uint src = (t * num_k_heads + kh) * head_k_dim + d;
    q_out[gid] = q_in[src];
    k_out[gid] = k_in[src];
}
