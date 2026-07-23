// Persistent decode kernel for Gemma4 E2B (RTX 3080 / Ampere sm_86).
//
// Single cooperative launch processes all 35 layers in one GPU invocation.
// CPU dispatch overhead between layers is eliminated; weight pointers are
// pre-populated in a device-side table and read inside the kernel.
//
// Layout assumption (Gemma4 E2B Q4_K_M):
//   - 35 layers, hidden=1536, n_ff=6144, head_dim=512(FULL)/256(SWA)
//   - num_heads=8, num_kv_heads=1, ple_dim=256
//   - shared_kv_layers=20 (layers 15-34 reuse KV from anchor)
//   - sliding pattern [SWA,SWA,SWA,SWA,FULL] x7
//
// The kernel uses cooperative_groups::grid_group::sync() to fence between
// phases. All blocks are persistent across all layers and all phases.

#pragma once

#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// One entry per transformer layer.  Pointers (u64) point to weight bytes that
// have already been uploaded to device memory by the resident cache layer.
// All shapes/flags are pre-resolved on the host; the kernel never branches on
// host-side state.
struct PersistentLayerParams {
    // Weight pointers (Q4_K packed bytes unless noted otherwise).
    unsigned long long q_weight;          // [q_dim x hidden]
    unsigned long long k_weight;          // [kv_dim x hidden]
    unsigned long long v_weight;          // [kv_dim x hidden]
    unsigned long long o_weight;          // [hidden x q_dim]
    unsigned long long gate_weight;       // [n_ff x hidden]
    unsigned long long up_weight;         // [n_ff x hidden]
    unsigned long long down_weight;       // [hidden x n_ff]

    // Norm weights (f32).
    unsigned long long attn_norm;         // [hidden]
    unsigned long long post_attn_norm;    // [hidden] or null
    unsigned long long ffn_norm;          // [hidden]
    unsigned long long post_ffn_norm;     // [hidden] or null
    unsigned long long q_norm;            // [head_dim] or null
    unsigned long long k_norm;            // [head_dim] or null

    // PLE (Per-Layer Embedding) f32 weights.
    unsigned long long ple_gate;          // [ple_dim x hidden] f32
    unsigned long long ple_proj;          // [hidden x ple_dim] f32
    unsigned long long ple_post_norm;     // [hidden]
    unsigned long long ple_input;         // [ple_dim] (computed from token embed lookup)

    // KV cache pointers (the layer's KV memory, written and read).
    unsigned long long k_cache;           // [max_ctx x kv_dim] f16
    unsigned long long v_cache;           // [max_ctx x kv_dim] f16

    // Dimensions / branch flags packed into u32 fields.
    unsigned int head_dim;                // 256 (SWA) or 512 (FULL)
    unsigned int q_dim;                   // num_heads * head_dim
    unsigned int kv_dim;                  // num_kv_heads * head_dim
    unsigned int n_ff;
    unsigned int sliding_window;          // 0 = full attention
    unsigned int kv_source_layer;         // == layer_idx if owns_kv, else anchor idx
    float layer_output_scale;             // 1.0 if not Gemma E2B
    unsigned int flags;                   // bit0=has_gated_attn, bit1=ple_kind_f32,
                                          // bit2=reuse_q_only, bit3=attn_rotation
};

// Top-level decode parameters.
struct PersistentDecodeParams {
    // Per-layer table (length = num_layers).
    PersistentLayerParams* layers;
    unsigned int num_layers;
    unsigned int hidden_dim;
    unsigned int vocab_size;
    float norm_eps;

    // Per-token dynamic state (updated by the host before each decode step).
    unsigned int rope_pos;                // current token position
    unsigned int kv_len;                  // total context length (incl. current token)

    // Scratch device buffers (caller-owned, sized once).
    float* hidden;                        // [hidden_dim] - carried across layers
    float* normed;                        // [hidden_dim]
    float* attn_out;                      // [q_dim_max]
    float* q_buf;                         // [q_dim_max]
    float* k_buf;                         // [kv_dim_max]
    float* v_buf;                         // [kv_dim_max]
    float* gate_buf;                      // [n_ff_max]
    float* up_buf;                        // [n_ff_max]
    float* ple_gate_buf;                  // [ple_dim]

    // Output logits.
    unsigned long long output_weight;     // Q8_0 [vocab x hidden]
    float* logits;                        // [vocab]
    int* argmax_out;                      // [1]

    // cu76 diagnostic: hidden state probe — when non-null, kernel writes the
    // hidden buffer to this device pointer right BEFORE the output projection
    // (i.e. after the last active layer's full pipeline including PLE +
    // residual + out_scale).  Host reads back and compares with eager hidden
    // at the same layer cap to find the first divergence layer.
    float* hidden_probe;                  // [hidden_dim] or null

    // cu76 phase probes (layer-0 only).  Each, when non-null, captures a
    // snapshot at the named phase of layer 0:
    //   normed_after_attn_norm_probe   — params.normed right after p0 attn_norm
    //   hidden_after_attn_probe        — params.hidden after O proj + residual
    //   hidden_after_ffn_probe         — params.hidden after FFN + residual
    // Used together with hidden_probe (after PLE+out_scale) to bisect the
    // first divergence phase inside layer 0.
    float* normed_after_attn_norm_probe;  // [hidden_dim] or null
    float* hidden_after_attn_probe;       // [hidden_dim] or null
    float* hidden_after_ffn_probe;        // [hidden_dim] or null

    // cu77: Gemma4 FULL-attention RoPE freq_factors (length = head_dim/2 = 256).
    // SWA layers skip this (kernel checks sliding_window > 0).  Null when the
    // model has no freq_factors tensor.  Placed AT THE END to match the host
    // struct ordering — any new field must extend both at the tail.
    unsigned long long rope_freqs;

    // cu78 fine-grained phase probes (layer-0 only, head-0/lane-0..head_dim of
    // each buffer is written by kernel when probe ptr is non-null).
    float* attn_out_probe;                // [head_dim] head-0 attention output
    float* q_proj_probe;                  // [head_dim] head-0 Q after Q proj (pre-RoPE)
    float* k_proj_probe;                  // [head_dim] K after K proj (pre-RoPE)
    float* v_proj_probe;                  // [head_dim] V after V proj
    float* attn_scores_probe;             // [kv_len] head-0 score per token (Q·K before softmax)
    float* attn_v_probe;                  // [head_dim] head-0 V cache value at j=0
    float* attn_acc_probe;                // [head_dim] head-0 final acc (before /row_sum)
    float* attn_row_sum_probe;            // [1] head-0 lane-0 row_sum at end
    unsigned int probe_layer_idx;         // cu86: which layer to probe (default 0)
    float* hidden_after_ffn_only_probe;   // cu87: [hidden_dim] hidden after FFN residual (before PLE)
    float* ffn_gate_probe;                // cu88: [n_ff_first_8] head-0 first 8 of gate_buf after gate proj
    float* ffn_gated_probe;               // cu88: first 8 of gate_buf after gelu(gate)*up
    float* ffn_down_probe;                // cu88: [hidden_dim] down output (= normed before residual)
    float* layer_hidden_trace;            // cu90: [num_layers] max_abs of hidden after each layer
    unsigned long long output_norm;       // cu91: [hidden_dim] f32 output_norm weight (Gemma4)
    unsigned int nan_trace;               // cu93: 1 = print [cu93-NaN] when hidden contains NaN/Inf at layer entry
    unsigned int gemma_v_norm;            // cu97: 1 = apply per-head no-scale RMS norm to V projection output (Gemma4 default)
    // cu100 Milestone 2 — batch prefill scaffold. `seq_len` = number of tokens
    // processed in this dispatch. Decode = 1 (current behavior). Batch prefill
    // = N (cu101+ wires the kernel body to per-phase token loops + causal
    // attention). Adding the field now so callers can lift to seq_len > 1
    // without struct churn at the next commit.
    unsigned int seq_len;
    // cu101 M3 batch attention: token-slot stride for q_buf / attn_out so the
    // kernel can split the layer body into per-token pre/attention/post phases
    // (q_buf[__t * q_dim_max], attn_out[__t * q_dim_max]). q_dim_max = max over
    // layers of q_dim (the host sizes both buffers as hidden_slots * q_dim_max).
    // Decode (seq_len=1) uses stride 0 → slot 0, unchanged behavior.
    unsigned int q_dim_max;
    // cu102 M4 batch FFN: batch-slot FFN buffers + n_ff stride. ffn_normed /
    // ffn_down are hidden_dim-wide per-token slots (ffn_norm output / down
    // output); gate_buf / up_buf are walked at n_ff_max stride so the batch FFN
    // phase processes all tokens' gate/up/down with shared-memory weight reuse.
    // Decode (seq_len=1) uses slot 0. Tail-appended (any new field extends BOTH
    // structs at the tail — see cu77 ABI rule).
    float* ffn_normed;
    float* ffn_down;
    unsigned int n_ff_max;
    // cu105 QKV batch-tiling: phase A is split A1 (per-token attn_norm →
    // attn_normed batch slots, hidden_dim stride) / A2 (batch Q/K/V projection
    // through the shared-memory tiled GEMM, input = attn_normed[*]) / A3
    // (per-token QK/V norm + RoPE + KV cache write). k_buf / v_buf become batch
    // slots (kv_dim_max stride) so A3 can read each token's K/V after the batch
    // projection. Decode (seq_len=1) → stride 0 → slot 0, behavior unchanged.
    // Tail-appended per cu77 ABI rule (any new field extends BOTH structs at the
    // tail). kv_dim_max (u32) before attn_normed (ptr) avoids a pad slot.
    unsigned int kv_dim_max;
    float* attn_normed;
};

// Flag bit helpers.
#define PERSISTENT_FLAG_GATED_ATTN  (1u << 0)
#define PERSISTENT_FLAG_PLE_F32     (1u << 1)
#define PERSISTENT_FLAG_REUSE_Q     (1u << 2)
#define PERSISTENT_FLAG_ATTN_ROT    (1u << 3)
// Quant kind flags: when set the corresponding projection weight is Q6_K
// instead of the default Q4_K (e.g. Gemma4 E2B's V weight and down weight).
#define PERSISTENT_FLAG_V_Q6K       (1u << 4)
#define PERSISTENT_FLAG_DOWN_Q6K    (1u << 5)
#define PERSISTENT_FLAG_O_Q6K       (1u << 6)
#define PERSISTENT_FLAG_K_Q6K       (1u << 7)
