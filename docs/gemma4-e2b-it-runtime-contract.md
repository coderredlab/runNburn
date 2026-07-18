# Gemma 4 E2B-it Runtime Contract

Source of truth for rebuilding runNburn's Gemma 4 E2B-it runtime path **from GGUF metadata + tensor graph + llama.cpp reference forward**, not from Gemma 3 analogies.

Sources:
- `gemma-4-E2B-it-Q4_K_M.gguf` inspected via `debug21`/`debug22`/`debug24`/`debug25` on Flip4, 2026-04-18.
- llama.cpp reference: `src/models/gemma4-iswa.cpp`, `src/llama-model.cpp` (ARCH_GEMMA4 block), `src/llama-hparams.cpp`.

## 1. Architecture identifier

- `general.architecture = "gemma4"` (distinct string from `"gemma"`, `"gemma2"`, `"gemma3"`, `"gemma3n"`)
- `tokenizer.ggml.model = "gemma4"`
- Loader MUST dispatch on `"gemma4"` as a dedicated arch. Current loader folds everything into `Architecture::Gemma` — that's the first split.

## 2. Model shape metadata (from GGUF)

| Key | Value (E2B-it) | Notes |
|---|---|---|
| `gemma4.block_count` | 35 | transformer layers |
| `gemma4.embedding_length` | 1536 | `n_embd` (hidden) |
| `gemma4.attention.head_count` | 8 | Q heads |
| `gemma4.attention.head_count_kv` | 1 | MQA (1 KV head shared by 8 Q heads) |
| `gemma4.attention.shared_kv_layers` | 20 | **last 20 layers reuse KV** from earlier layers — see §5 |
| `gemma4.attention.sliding_window` | (n_swa) | window size for SWA (read via `LLM_KV_ATTENTION_SLIDING_WINDOW`) |
| `gemma4.attention.sliding_window_pattern` | array[35] | per-layer is-SWA flag; full-attn where `false` |
| `gemma4.attention.key_length_swa` | (k-dim for SWA) | SWA layers have different head K dim |
| `gemma4.attention.value_length_swa` | (v-dim for SWA) | SWA layers have different head V dim |
| `gemma4.embedding_length_per_layer_input` | 256 | `n_embd_per_layer` (PLE dim) |
| `gemma4.feed_forward_length` | array[35] | layer 0–14: `6144`; layer 15–34: `12288` |
| `gemma4.rope.freq_base` | (standard) | full-attn RoPE base |
| `gemma4.rope.freq_base_swa` | (separate) | SWA RoPE base |
| `gemma4.attention.layer_norm_rms_epsilon` | — | RMSNorm eps |
| `gemma4.final_logit_softcapping` | — | optional; if present apply `softcap * tanh(logits / softcap)` |
| `gemma4.expert_feed_forward_length` | (optional) | MoE only, not relevant for E2B-it |

**Constants set in hparams for Gemma4 (not in GGUF):**
- `f_attention_scale = 1.0f` — **no pre-attention Q scaling**. Gemma4 uses `self.scaling = 1.0`. (Gemma1-3 used `1/sqrt(head_dim)`.)
- `n_layer_kv_from_start = n_layer - shared_kv_layers = 35 - 20 = 15` — only the first 15 layers compute fresh KV.

## 3. Attention layer pattern (from debug22 + `swa_layers` metadata)

Two interleaved layer types (ISWA = Interleaved Sliding-Window Attention):

| Role | `head_dim` | Q | K | V | O | Layer indices (0-based) |
|---|---|---|---|---|---|---|
| **SWA (sliding window)** | 256 | `[2048, 1536]` | `[256, 1536]` | `[256, 1536]` | `[1536, 2048]` | all others (28 layers) |
| **Full attention (global)** | 512 | `[4096, 1536]` | `[512, 1536]` | `[512, 1536]` | `[1536, 4096]` | **4, 9, 14, 19, 24, 29, 34** (7 layers) |

- Dims: `q_rows = n_head × head_dim` (8×256 or 8×512); `k_rows = v_rows = n_kv_heads × head_dim` (1 × head_dim, MQA).
- `n_embd_head_k_swa` / `n_embd_head_v_swa` from GGUF fix the SWA head dim (256). Full-attn head dim comes from the tensor shape itself.
- Pattern: every 5th layer (`il % 5 == 4`) is full-attention. Mask is full-causal for global layers, sliding-window-causal for SWA layers.
- **Layer 34 is the last layer AND a full-attention layer**. Session 46's "layer 34 explodes" was downstream of head-dim or attention-mask mis-handling at one or more of layers {4, 9, 14, 19, 24, 29, 34}; layer 34 is just where the accumulated error hits the final norm + output head.

### Per-layer attention weights (always present)

- `blk.i.attn_norm.weight` `[1536]` F32 — pre-attention RMSNorm weight
- `blk.i.attn_q_norm.weight` `[head_dim]` F32 — RMSNorm applied to Q **after reshape to heads, before RoPE**
- `blk.i.attn_k_norm.weight` `[head_dim]` F32 — RMSNorm applied to K after reshape, before RoPE (only on layers with `has_kv`)
- `blk.i.attn_post_norm.weight` `[1536]` F32 — **post-attention RMSNorm on the attn output**, applied before the residual add
- `blk.i.attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight` — see shape table above (K/V only required when `has_kv(il)` is true)

## 4. Per-layer embedding (PLE) — the actual forward

PLE has **two stages**: a setup stage before the block loop, then a per-block fusion inside each block.

### 4a. Setup (once per decode step, before the loop)

Inputs: hidden embedding `inpL ∈ R^{1536 × n_tokens}` and token ids `t ∈ Z^{n_tokens}`.

```
inp_per_layer = per_layer_token_embd[t]            # [8960, n_tokens], Q5_K table
inp_per_layer = reshape(inp_per_layer, [256, 35, n_tokens])
inp_per_layer = inp_per_layer * sqrt(256)          # = × 16

per_layer_proj = per_layer_model_proj @ inpL       # [8960, n_tokens], BF16 matmul
per_layer_proj = per_layer_proj * (1 / sqrt(1536)) # ≈ × 0.02552
per_layer_proj = reshape(per_layer_proj, [256, 35, n_tokens])
per_layer_proj = rmsnorm(per_layer_proj, per_layer_proj_norm)   # 256-wide weight

inp_per_layer = (per_layer_proj + inp_per_layer) * (1 / sqrt(2))
inp_per_layer = permute(inp_per_layer, [0, 2, 1])  # [256, n_tokens, 35]
```

Note: `per_layer_model_proj` is BF16 `[8960, 1536]`. This is the global path; it runs **once per token batch**, not per layer.

### 4b. Per-block fusion (inside the layer loop, after FFN + post-FFN norm + residual)

Inputs: block output `cur ∈ R^{1536 × n_tokens}`, and `inp_this_layer = inp_per_layer[:,:,il] ∈ R^{256 × n_tokens}`.

```
pe_in = cur                                                 # save residual

cur = inp_gate_i @ cur                                      # [256, n_tokens], `blk.i.inp_gate`
cur = gelu(cur)                                             # GeLU activation (NOT identity/sigmoid)
cur = cur * inp_this_layer                                  # elementwise [256, n_tokens]

cur = per_layer_proj_i @ cur                                # [1536, n_tokens], `blk.i.proj`
cur = rmsnorm(cur, per_layer_post_norm_i)                   # `blk.i.post_norm`

cur = pe_in + cur                                           # residual
```

### 4c. Key corrections vs earlier guesswork

- The per-block gate uses **GeLU**, not identity and not sigmoid.
- The PLE slice is **pre-fused** (global path × per-token table × RMSNorm × √2 scale) BEFORE the block loop — per-block does not re-normalize it.
- `per_layer_model_proj` is **global path**, runs once. Not per-layer.
- `per_layer_proj_norm` is a **256-dim** weight applied in the setup stage (not in every block).
- The per-block RMSNorm uses `blk.i.post_norm` (1536-dim), applied to the PLE branch output (before the pe_in residual).

## 5. KV sharing semantics (shared_kv_layers = 20)

Derived: `n_layer_kv_from_start = 35 - 20 = 15`.

- **Layers 0–14** (15 layers): `has_kv(il) == true`. Compute fresh K and V from weights, store in cache.
- **Layers 15–34** (20 layers): `has_kv(il) == false`. **Reuse** KV from an earlier layer. No K/V projection, no K/V cache write.

For each reused layer `il ≥ 15`, the source layer index is:

```
src = n_layer_kv_from_start - (is_swa(il) ? 2 : 1)
    = 15 - (is_swa ? 2 : 1)
    = 13 (if SWA)
    = 14 (if full attention)
```

So every reused layer pulls from either layer 13 (SWA source) or layer 14 (full-attn source). Layer 14 is itself a full-attention layer (`14 % 5 == 4`), so it's the natural full-attn source at the boundary; layer 13 is SWA and serves as the SWA source.

| Layer range | `has_kv` | Attn type | K/V source |
|---|---|---|---|
| 0–14 | true | mix (full at 4, 9, 14; SWA elsewhere) | own projection |
| 15–34 (SWA layers among these) | false | SWA | layer 13 K/V |
| 15–34 (full-attn layers: 19, 24, 29, 34) | false | full | layer 14 K/V |

## 6. Per-layer block forward (full path in order)

For each layer `il ∈ [0, 35)`:

```
// 1. Pre-attention norm
h_norm = rmsnorm(h_in, attn_norm_il)

// 2. Q projection + Q-norm + RoPE
Q = attn_q_il @ h_norm                          # [n_head × head_dim(il), n_tokens]
Q = reshape(Q, [head_dim(il), n_head, n_tokens])
Q = rmsnorm(Q, attn_q_norm_il)                  # per-head RMSNorm over head_dim
Q = rope(Q, positions, freq_base(il), freq_scale(il), n_rot(il),
         freq_factors=is_swa(il) ? null : rope_freqs_il)

// 3. K/V: project if has_kv(il), else reuse from cache
if has_kv(il):
    K = attn_k_il @ h_norm                      # [n_kv_head × head_dim(il), n_tokens]
    V = attn_v_il @ h_norm
    K = reshape(K, [head_dim, n_kv_head, n_tokens])
    V = reshape(V, [head_dim, n_kv_head, n_tokens])
    K = rmsnorm(K, attn_k_norm_il)              # weighted RMSNorm
    V = rms_norm(V, eps=f_norm_rms_eps)         # UNWEIGHTED RMSNorm (no weight tensor)
    K = rope(K, positions, freq_base(il), freq_scale(il), n_rot(il),
             freq_factors=is_swa(il) ? null : rope_freqs_il)
    kv_cache_write(il, K, V)
else:
    K, V = kv_cache_read(src_layer_of(il))      // src = 13 or 14; see §5

// 4. Attention (scale = 1.0, NOT 1/sqrt(head_dim))
mask = is_swa(il) ? sliding_window_causal_mask : full_causal_mask
attn_out = attention(Q, K, V, scale=1.0, mask)   # [n_embd, n_tokens]
attn_out = attn_output_il @ attn_out             # back to n_embd

// 5. Post-attn norm + residual
attn_out = rmsnorm(attn_out, attn_post_norm_il)
attn_out = h_in + attn_out                       # residual into the pre-norm input

// 6. FFN (GeLU-gated, parallel gate/up form)
ffn_in = rmsnorm(attn_out, ffn_norm_il)
gate   = gelu(ffn_gate_il @ ffn_in)              # [ff_len(il), n_tokens]
up     = ffn_up_il  @ ffn_in
hidden = ffn_down_il @ (gate * up)               # [n_embd, n_tokens]

// 7. Post-FFN norm + residual
hidden = rmsnorm(hidden, ffn_post_norm_il)
hidden = attn_out + hidden                       # residual

// 8. Per-layer embedding fusion (§4b)
pe_in  = hidden
gate   = gelu(inp_gate_il @ hidden)              # [256, n_tokens]
branch = gate * inp_per_layer[:,:,il]            # elementwise with pre-fused PLE
delta  = proj_il @ branch                        # back to [1536, n_tokens]
delta  = rmsnorm(delta, post_norm_il)
hidden = pe_in + delta                           # residual

// 9. Optional per-layer out_scale (tensor `blk.i.out_scale` if present)
if out_scale_il exists:
    hidden = hidden * out_scale_il

h_out = hidden
```

### Final stage (after the loop)

```
h_final = rmsnorm(h_out, output_norm)
logits = output_head @ h_final                   # `output.weight`, typically tied or Q6_K

if final_logit_softcapping != 0:
    logits = softcap * tanh(logits / softcap)
```

## 7. Embedding input scale

At the very top, before layer 0:

```
inpL = tok_embd[token_ids]                       # Q5_K/Q6_K table lookup
if input is token (not raw multimodal embedding):
    inpL = inpL * sqrt(n_embd) = × sqrt(1536) ≈ × 39.19
```

This scale is required — it's the classic Gemma convention. Multimodal/raw-embedding inputs skip the scale (E2B-it only does text, so always scale).

## 8. What this contract replaces

Everything in `engine.rs` currently gated on Gemma + `RNB_GEMMA_*` is provisional and should be removed or isolated:

- All `RNB_GEMMA_PLE_*` env seams (token_scale, projected_scale, skip_post_norm, replace_hidden, model_only, raw_gate_mul, branch_scale, layer_min/max/offset, etc.) — **replaced by §4** as executable code without env overrides.
- `gemma_ple_layer34_fix` / `gemma_ple_layer34_hard_fix_applies` — **replaced by correct §3 (head_dim per layer) + §5 (KV reuse)**. Layer 34's symptom is explained by ISWA + boundary KV reuse, not a layer-34-specific pathology.
- `detect_gemma_runtime_flavor` heuristic (num_layers=35 + hidden=1536) — **replaced by `arch == "gemma4"` dispatch**.
- Gemma-wide pre-attention Q scaling — **removed for Gemma4** (scale=1.0).
- `RNB_DISABLE_ATTN_LAYER=*` and late-attention-disable experiments — no longer needed; attention runs as specified per §6.

## 9. Open items left for implementation (no longer "open questions")

All five original Open Questions are resolved above. Implementation-level TODOs:

1. **Loader split**: `Architecture::Gemma4` distinct from `Gemma`. Read per-layer arrays (`sliding_window_pattern`, `feed_forward_length`) and SWA variants (`key_length_swa`, `value_length_swa`, `rope.freq_base_swa`). Emit per-layer `head_dim`, `is_swa`, `has_kv`, `kv_reuse_src` into the graph / model description.
2. **Forward path**: implement §6 in a new `forward_gemma4_*` in `engine.rs`. No env vars on the default path.
3. **PLE**: implement §4a (once per batch) + §4b (per-block). Cache `inp_per_layer` across the layer loop.
4. **KV cache reuse**: extend the existing KV cache interface to allow "read from layer `src`" without a write, for layers 15–34 (§5).
5. **Instrumentation**: per-layer hidden L2, top-K logit drift, and rank of `{서울, 수도, 대한민국}` tokens vs llama.cpp same-device. Used only to validate §6 against reference, not to gate behavior.
6. **Qwen / Llama / Gemma (old) regression**: keep the existing Gemma path intact while adding Gemma4. `cargo test --workspace` must stay green and Qwen3.5 0.8B ADB decode must stay at baseline (≈22 tok/s 4T).

## 10. Loader-level tensor inventory for E2B-it

Per-block tensors (35 layers, indices 0–34):

- Attention: `attn_norm` F32[1536], `attn_q` Q4_K[2048×1536 or 4096×1536], `attn_k`/`attn_v` Q4_K[256×1536 or 512×1536] (only when `has_kv`), `attn_output` Q4_K[1536×2048 or 1536×4096], `attn_q_norm` F32[head_dim], `attn_k_norm` F32[head_dim] (only when `has_kv`), `attn_post_norm` F32[1536]
- FFN: `ffn_norm` F32[1536], `ffn_gate`/`ffn_up` Q4_K[ff_len(il)×1536], `ffn_down` Q4_K[1536×ff_len(il)], `ffn_post_norm` F32[1536]
- PLE per-block: `inp_gate` F32[256×1536], `proj` F32[1536×256], `post_norm` F32[1536]
- Optional: `out_scale` (per-layer scalar/vector) — presence to be confirmed by probe

Global tensors:

- `token_embd.weight` (Q6_K typical) — main token embedding
- `output.weight` (Q6_K typical, often tied) — LM head
- `output_norm.weight` F32[1536]
- `per_layer_token_embd.weight` Q5_K logical[262144 × 8960]
- `per_layer_model_proj.weight` BF16[8960 × 1536]
- `per_layer_proj_norm.weight` F32[256]
- Optional per-full-attn-layer `rope_freqs` tensor (used only when `!is_swa(il)`)
