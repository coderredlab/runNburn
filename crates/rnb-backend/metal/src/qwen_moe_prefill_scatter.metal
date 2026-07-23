#include <metal_stdlib>
using namespace metal;

// Gather routed token rows from the full token-major normalized activation into
// the per-expert-group FFN input. This avoids host-side top-k activation packing.
kernel void qwen_moe_prefill_gather_normed(
    device const float* norm_all    [[buffer(0)]],
    device float*       group_in    [[buffer(1)]],
    device const uint*  token_ids   [[buffer(2)]],
    constant uint&      hidden_dim  [[buffer(3)]],
    constant uint&      group_start [[buffer(4)]],
    constant uint&      total_elems [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_elems) return;
    uint local = gid / hidden_dim;
    uint col = gid - local * hidden_dim;
    uint token = token_ids[group_start + local];
    group_in[gid] = norm_all[token * hidden_dim + col];
}

kernel void qwen_moe_prefill_gather_normed_f16(
    device const float* norm_all    [[buffer(0)]],
    device half*        group_in    [[buffer(1)]],
    device const uint*  token_ids   [[buffer(2)]],
    constant uint&      hidden_dim  [[buffer(3)]],
    constant uint&      group_start [[buffer(4)]],
    constant uint&      total_elems [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_elems) return;
    uint local = gid / hidden_dim;
    uint col = gid - local * hidden_dim;
    uint token = token_ids[group_start + local];
    group_in[gid] = (half)norm_all[token * hidden_dim + col];
}

// Accumulate one routed expert-group FFN output into token-major MoE output.
// Command buffers are submitted per expert group, so writes for the same token
// across different experts are ordered and do not need atomics.
kernel void qwen_moe_prefill_scatter_accum(
    device const float* down          [[buffer(0)]],
    device float*       out           [[buffer(1)]],
    device const uint*  token_ids     [[buffer(2)]],
    device const float* route_weights [[buffer(3)]],
    constant uint&      hidden_dim    [[buffer(4)]],
    constant uint&      group_start   [[buffer(5)]],
    constant uint&      total_elems   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_elems) return;
    uint local = gid / hidden_dim;
    uint col = gid - local * hidden_dim;
    uint slot = group_start + local;
    uint token = token_ids[slot];
    out[token * hidden_dim + col] += route_weights[slot] * down[gid];
}

kernel void qwen_moe_prefill_topk_from_logits(
    device const float* logits        [[buffer(0)]],
    device uint*        expert_ids    [[buffer(1)]],
    device float*       route_weights [[buffer(2)]],
    device uint*        token_ids     [[buffer(3)]],
    constant uint&      n_expert      [[buffer(4)]],
    constant uint&      n_used        [[buffer(5)]],
    constant uint&      seq_len       [[buffer(6)]],
    uint token [[thread_position_in_grid]])
{
    if (token >= seq_len || n_used == 0 || n_used > 32) return;

    float best_vals[32];
    uint best_ids[32];
    for (uint i = 0; i < n_used; i++) {
        best_vals[i] = -INFINITY;
        best_ids[i] = 0;
    }

    device const float* row = logits + token * n_expert;
    for (uint expert = 0; expert < n_expert; expert++) {
        float value = row[expert];
        for (uint rank = 0; rank < n_used; rank++) {
            if (value > best_vals[rank] ||
                (value == best_vals[rank] && expert < best_ids[rank])) {
                for (uint shift = n_used - 1; shift > rank; shift--) {
                    best_vals[shift] = best_vals[shift - 1];
                    best_ids[shift] = best_ids[shift - 1];
                }
                best_vals[rank] = value;
                best_ids[rank] = expert;
                break;
            }
        }
    }

    float selected_max = best_vals[0];
    float selected_sum = 0.0f;
    for (uint rank = 0; rank < n_used; rank++) {
        selected_sum += exp(best_vals[rank] - selected_max);
    }

    uint base = token * n_used;
    for (uint rank = 0; rank < n_used; rank++) {
        expert_ids[base + rank] = best_ids[rank];
        route_weights[base + rank] =
            selected_sum != 0.0f ? exp(best_vals[rank] - selected_max) / selected_sum : 0.0f;
        token_ids[base + rank] = token;
    }
}
