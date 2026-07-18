//! Drafter backbone forward (mt84 Stage γ).
//!
//! mt83 의 4-layer custom cross-attention 폐기. 본 모듈은 transformers
//! `Gemma4TextModel.forward` 의 verbatim reuse — drafter 의 4 layer 가 모두
//! `is_kv_shared_layer = True` 라 K/V projection 없이 `SharedKvStates` 에서
//! K/V 를 가져옴. layer_scalar 위치 정정 (decoder_layer 의 마지막
//! multiplication), Q-only RoPE (K 는 dict 에 이미 RoPE 적용된 상태).
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`
//! §1-§5.
//!
//! ## Per-step pipeline (spec §1)
//!
//! 1. **pre_projection** — `[5120] → [256]` matvec
//!    (input = `cat([target_token_embd[id_curr], target_last_hidden])`).
//! 2. **Gemma4TextModel backbone** — 4 layer, 각 layer 는
//!    [`decoder_layer_forward`] (spec §3 verbatim). 모든 layer 가
//!    `is_kv_shared_layer=True` 라 attention 부분이 `shared_kv_states[layer_type]`
//!    에서 K/V 를 가져옴.
//! 3. **output_norm** — drafter 의 final RMSNorm.
//! 4. **post_projection** — `[256] → [2560]` matvec. 다음 drafter step 의
//!    `inputs_embeds` 의 last_hidden 자리.
//! 5. **vq_masked_embedding** — mt83 `vq_head.rs` verbatim 보존
//!    ([`vq_head_forward`] + [`vocab_logits_in_top_k_clusters`]).
//!
//! ## K layout 가정 (spec §4 Critical attention semantics)
//!
//! `SharedKvLayer.k` / `.v` 는 row-major `[n_kv_heads, seq_len, head_dim]`
//! flatten f32. Stage β 의 `transpose_to_head_major` 가 이 layout 으로 변환.
//! index 식 = `K[h * seq_len * head_dim + t * head_dim + d]`. mt83 의 옛
//! `t * kv_dim + kv_h * head_dim` 식과는 axis 다름 — 본 모듈의 attention 은
//! head-major index 만 사용한다.
//!
//! Q 만 RoPE 적용 (`position_id` = last validated token position, constant
//! across drafting loop). K 는 target prefill 시 이미 RoPE 적용된 상태로
//! `shared_kv_states` 에 저장됨 — 추가 RoPE 적용 금지.

use super::dequant::dequant_to_f32;
use super::types::{Drafter, DrafterLayer};
use super::vq_head::{vocab_logits_in_top_k_clusters, vq_head_forward, ClusterTokenTable};
use super::{SharedKvLayer, SharedKvStates};

/// SWA layer 의 RoPE frequency base. Gemma 4 standard (`rope.freq_base_swa`).
const ROPE_BASE_SWA: f32 = 10_000.0;
/// Full attention layer 의 RoPE frequency base. Gemma 4 standard (`rope.freq_base`).
const ROPE_BASE_FULL: f32 = 1_000_000.0;
/// RMSNorm epsilon. Gemma 4 표준.
const RMS_EPS: f32 = 1e-6;

/// Drafter forward output (mt84 Stage γ).
///
/// `logits` — `[vocab_size]` (Gemma 4 assistant = 262144) full vector with
/// `f32::NEG_INFINITY` 채워진 non-top-K cluster token. tied-lm_head 의
/// `x_norm · drafter.token_embd[tok]` (drafter hidden=256 공간).
///
/// `projected_hidden` — `[backbone_hidden]` (Gemma 4 = 2560). drafter 의
/// `post_projection` 결과. candidate generator loop 에서 다음 drafter step
/// 의 `inputs_embeds` 의 last_hidden 자리에 들어감 (spec §7).
#[derive(Debug, Clone)]
pub struct DrafterForwardOutput {
    /// `[vocab_size]` full vector. non-selected token = `NEG_INFINITY`.
    pub logits: Vec<f32>,
    /// `[backbone_hidden]` = drafter post_projection 결과.
    pub projected_hidden: Vec<f32>,
}

/// Drafter backbone forward (mt84 Stage γ).
///
/// 입력:
/// - `drafter` — Stage A 에서 로드된 drafter weight.
/// - `inputs_embeds` — `[5120]` = `cat([target_token_embd[id_curr],
///   target_last_hidden])` (concatenation order verbatim spec §7).
/// - `shared_kv_states` — Stage β 의 `Engine::shared_kv_states_for_drafter()`
///   결과. sliding / full 두 layer_type 의 K/V dict.
/// - `position_id` — drafting loop 의 constant position
///   (= `input_ids.shape[1] - 1`, last validated token's position).
///
/// 출력: `DrafterForwardOutput { logits, projected_hidden }`.
pub fn drafter_forward(
    drafter: &Drafter,
    inputs_embeds: &[f32],
    shared_kv_states: &SharedKvStates,
    position_id: u32,
) -> DrafterForwardOutput {
    assert_eq!(
        inputs_embeds.len(),
        2 * drafter.backbone_hidden,
        "drafter_forward: inputs_embeds len {} != 2 * backbone_hidden {}",
        inputs_embeds.len(),
        2 * drafter.backbone_hidden
    );

    // ----- Step 1: pre_projection (5120 → 256) ----------------------------------
    assert_eq!(
        drafter.pre_projection.shape.len(),
        2,
        "pre_projection shape rank != 2: {:?}",
        drafter.pre_projection.shape
    );
    assert_eq!(
        drafter.pre_projection.shape[0], drafter.hidden,
        "pre_projection rows {} != hidden {}",
        drafter.pre_projection.shape[0], drafter.hidden
    );
    assert_eq!(
        drafter.pre_projection.shape[1],
        2 * drafter.backbone_hidden,
        "pre_projection cols {} != 2*backbone_hidden {}",
        drafter.pre_projection.shape[1],
        2 * drafter.backbone_hidden
    );
    let mut hidden = vec![0.0f32; drafter.hidden];
    let pre_cuda_ok = rnb_runtime::policy::drafter_cuda_enabled()
        && super::cuda::drafter_projection_cuda(
            &drafter.pre_projection,
            inputs_embeds,
            &mut hidden,
            drafter.hidden,
            2 * drafter.backbone_hidden,
        )
        .unwrap_or(false);
    if !pre_cuda_ok {
        let pre_proj_w = dequant_to_f32(&drafter.pre_projection);
        matvec(
            &pre_proj_w,
            drafter.hidden,
            2 * drafter.backbone_hidden,
            inputs_embeds,
            &mut hidden,
        );
    }

    // cu66 Phase 4 step 2: drafter K/V f16 conversion cache (forward-scoped).
    // shared_kv 의 host f32 K/V 를 entry 에 한 번 f16 변환. 매 layer 의
    // cross_attention cuda 호출 의 매번 변환 (cu63 saturation cause) 제거.
    let kv_cache_f16 = if rnb_runtime::policy::drafter_cuda_enabled() {
        let sliding_head_dim = drafter
            .layers
            .iter()
            .find(|l| l.is_sliding_window)
            .map(|l| l.head_dim)
            .unwrap_or(0);
        let full_head_dim = drafter
            .layers
            .iter()
            .find(|l| !l.is_sliding_window)
            .map(|l| l.head_dim)
            .unwrap_or(0);
        let sliding_f16 = if sliding_head_dim > 0 {
            Some(super::cuda::drafter_kv_f16_convert(
                &shared_kv_states.sliding_attention.k,
                &shared_kv_states.sliding_attention.v,
                shared_kv_states.sliding_attention.seq_len,
                shared_kv_states.sliding_attention.n_kv_heads,
                sliding_head_dim,
            ))
        } else {
            None
        };
        let full_f16 = if full_head_dim > 0 {
            Some(super::cuda::drafter_kv_f16_convert(
                &shared_kv_states.full_attention.k,
                &shared_kv_states.full_attention.v,
                shared_kv_states.full_attention.seq_len,
                shared_kv_states.full_attention.n_kv_heads,
                full_head_dim,
            ))
        } else {
            None
        };
        Some((sliding_f16, full_f16))
    } else {
        None
    };

    // ----- Step 2: Gemma4TextModel backbone (4 layer) ---------------------------
    for layer in drafter.layers.iter() {
        let shared_kv = if layer.is_sliding_window {
            &shared_kv_states.sliding_attention
        } else {
            &shared_kv_states.full_attention
        };
        let rope_base = if layer.is_sliding_window {
            ROPE_BASE_SWA
        } else {
            ROPE_BASE_FULL
        };
        let kv_f16 = kv_cache_f16.as_ref().and_then(|(s, f)| {
            if layer.is_sliding_window {
                s.as_ref().map(|(k, v)| (k.as_slice(), v.as_slice()))
            } else {
                f.as_ref().map(|(k, v)| (k.as_slice(), v.as_slice()))
            }
        });
        decoder_layer_forward(
            layer,
            drafter.hidden,
            &mut hidden,
            shared_kv,
            kv_f16,
            position_id,
            rope_base,
        );
    }

    // ----- Step 3: output_norm --------------------------------------------------
    rms_norm_inplace(&mut hidden, &drafter.output_norm, RMS_EPS);
    let last_hidden = hidden; // [hidden=256]

    // ----- Step 4: post_projection (256 → 2560) ---------------------------------
    assert_eq!(
        drafter.post_projection.shape.len(),
        2,
        "post_projection shape rank != 2: {:?}",
        drafter.post_projection.shape
    );
    assert_eq!(
        drafter.post_projection.shape[0], drafter.backbone_hidden,
        "post_projection rows {} != backbone_hidden {}",
        drafter.post_projection.shape[0], drafter.backbone_hidden
    );
    assert_eq!(
        drafter.post_projection.shape[1], drafter.hidden,
        "post_projection cols {} != hidden {}",
        drafter.post_projection.shape[1], drafter.hidden
    );
    let mut projected_hidden = vec![0.0f32; drafter.backbone_hidden];
    let post_cuda_ok = rnb_runtime::policy::drafter_cuda_enabled()
        && super::cuda::drafter_projection_cuda(
            &drafter.post_projection,
            &last_hidden,
            &mut projected_hidden,
            drafter.backbone_hidden,
            drafter.hidden,
        )
        .unwrap_or(false);
    if !post_cuda_ok {
        let post_proj_w = dequant_to_f32(&drafter.post_projection);
        matvec(
            &post_proj_w,
            drafter.backbone_hidden,
            drafter.hidden,
            &last_hidden,
            &mut projected_hidden,
        );
    }

    // ----- Step 5: VQ masked embedding (mt83 vq_head.rs verbatim) ---------------
    //
    // Spec §5 의 `Gemma4AssistantMaskedEmbedder.forward` 는 lm_head 가
    // `model.embed_tokens.weight` 와 tied — drafter 의 `token_embd.weight` Q6_K
    // [262144, 256] 그대로 사용. dim = drafter.hidden (256), not
    // backbone_hidden (2560). mt83 `vq_head.rs` 의 `vq_head_forward` +
    // `vocab_logits_in_top_k_clusters` 가 이 architecture 와 1:1 매칭.
    let vq = vq_head_forward(drafter, &last_hidden);
    let cluster_table = ClusterTokenTable::permutation(
        drafter.token_ordering.clone(),
        drafter.n_centroids as usize,
    );
    let logits = vocab_logits_in_top_k_clusters(
        drafter,
        &vq.cluster_logits,
        &last_hidden,
        drafter.centroid_top_k as usize,
        &cluster_table,
    );

    DrafterForwardOutput {
        logits,
        projected_hidden,
    }
}

/// Per-layer forward — `Gemma4TextDecoderLayer.forward` verbatim (spec §3).
///
/// drafter 의 4 layer 다 `hidden_size_per_layer_input = 0` (PLE 없음),
/// `enable_moe_block = false` 라 PLE / MoE 분기 모두 skip.
///
/// **Critical (spec §3, mt83 정정)**: `layer_scalar` 는 layer 전체 output 의
/// 마지막 multiplication 으로 적용. mt83 cross_attention.rs:222-225 의
/// attention output 측 scale 은 verbatim 위반.
fn decoder_layer_forward(
    layer: &DrafterLayer,
    hidden_size: usize,
    hidden: &mut [f32],
    shared_kv: &SharedKvLayer,
    kv_f16: Option<(&[u16], &[u16])>,
    position_id: u32,
    rope_base: f32,
) {
    assert_eq!(
        hidden.len(),
        hidden_size,
        "decoder_layer_forward: hidden len {} != hidden_size {}",
        hidden.len(),
        hidden_size
    );

    // ----- 1. Attention block ---------------------------------------------------
    let residual = hidden.to_vec();
    rms_norm_inplace(hidden, &layer.attn_norm, RMS_EPS);

    // Q proj + per-head q_norm + RoPE (Q only)
    let q_dim = layer.n_heads * layer.head_dim;
    assert_eq!(
        layer.attn_q.shape[0], q_dim,
        "attn_q rows {} != n_heads*head_dim {}",
        layer.attn_q.shape[0], q_dim
    );
    assert_eq!(
        layer.attn_q.shape[1], hidden_size,
        "attn_q cols {} != hidden {}",
        layer.attn_q.shape[1], hidden_size
    );
    // cu59 Phase 3 step 1: drafter q_proj cuda port (env opt-in).
    let mut q = vec![0.0f32; q_dim];
    let q_cuda_ok = rnb_runtime::policy::drafter_cuda_enabled()
        && super::cuda::drafter_attn_q_cuda(&layer.attn_q, hidden, &mut q, q_dim, hidden_size)
            .unwrap_or(false);
    if !q_cuda_ok {
        let attn_q_w = dequant_to_f32(&layer.attn_q);
        matvec(&attn_q_w, q_dim, hidden_size, hidden, &mut q);
    }

    // Per-head q_norm (RMSNorm with attn_q_norm weight applied per head_dim slice).
    for h in 0..layer.n_heads {
        let slice = &mut q[h * layer.head_dim..(h + 1) * layer.head_dim];
        rms_norm_inplace(slice, &layer.attn_q_norm, RMS_EPS);
    }

    // RoPE on Q only — K 는 shared_kv 에 이미 RoPE 적용된 상태 (spec §4 Critical).
    // position_id 는 drafting loop 의 constant (last validated token's position).
    apply_rope_q(
        &mut q,
        layer.n_heads,
        layer.head_dim,
        position_id as usize,
        rope_base,
    );

    // Cross-attention with shared K/V (GQA).
    assert_eq!(
        shared_kv.head_dim, layer.head_dim,
        "shared_kv head_dim {} != layer.head_dim {} \
         (sliding={}: target dict head_dim mismatch)",
        shared_kv.head_dim, layer.head_dim, layer.is_sliding_window
    );
    let mut attn_out = vec![0.0f32; q_dim];
    // cu66 Phase 4 step 2: cross_attention cuda 의 K/V f16 cache 활용.
    // forward-scoped cache 가 있으면 변환 안 함 (cu63 의 saturation 해소).
    let attn_cuda_ok = if let Some((k_f16, v_f16)) = kv_f16 {
        rnb_runtime::policy::drafter_cuda_enabled()
            && super::cuda::drafter_cross_attention_cuda_cached(
                &q,
                k_f16,
                v_f16,
                shared_kv.seq_len,
                layer.n_heads,
                shared_kv.n_kv_heads,
                layer.head_dim,
                &mut attn_out,
            )
            .unwrap_or(false)
    } else {
        rnb_runtime::policy::drafter_cuda_enabled()
            && super::cuda::drafter_cross_attention_cuda(
                &q,
                &shared_kv.k,
                &shared_kv.v,
                shared_kv.seq_len,
                layer.n_heads,
                shared_kv.n_kv_heads,
                layer.head_dim,
                &mut attn_out,
            )
            .unwrap_or(false)
    };
    if !attn_cuda_ok {
        cross_attention_gqa(
            &q,
            &shared_kv.k,
            &shared_kv.v,
            shared_kv.seq_len,
            layer.n_heads,
            shared_kv.n_kv_heads,
            layer.head_dim,
            &mut attn_out,
        );
    }

    // o_proj
    assert_eq!(
        layer.attn_output.shape[0], hidden_size,
        "attn_output rows {} != hidden {}",
        layer.attn_output.shape[0], hidden_size
    );
    assert_eq!(
        layer.attn_output.shape[1], q_dim,
        "attn_output cols {} != n_heads*head_dim {}",
        layer.attn_output.shape[1], q_dim
    );
    // cu60 Phase 3 step 2: drafter o_proj cuda port (env opt-in).
    let mut attn_proj = vec![0.0f32; hidden_size];
    let o_cuda_ok = rnb_runtime::policy::drafter_cuda_enabled()
        && super::cuda::drafter_attn_o_cuda(
            &layer.attn_output,
            &attn_out,
            &mut attn_proj,
            hidden_size,
            q_dim,
        )
        .unwrap_or(false);
    if !o_cuda_ok {
        let out_proj_w = dequant_to_f32(&layer.attn_output);
        matvec(&out_proj_w, hidden_size, q_dim, &attn_out, &mut attn_proj);
    }

    // post_attention_norm + residual.
    // spec §3: hidden = residual + post_attention_layernorm(attn_out)
    rms_norm_inplace(&mut attn_proj, &layer.post_attention_norm, RMS_EPS);
    for i in 0..hidden_size {
        hidden[i] = residual[i] + attn_proj[i];
    }

    // ----- 2. FFN block (PLE / MoE 분기 skip — drafter 는 둘 다 비활성) ---------
    let residual = hidden.to_vec();
    rms_norm_inplace(hidden, &layer.ffn_norm, RMS_EPS);

    let ffn_dim = layer.ffn_gate.shape[0];
    assert_eq!(
        layer.ffn_gate.shape[1], hidden_size,
        "ffn_gate cols {} != hidden {}",
        layer.ffn_gate.shape[1], hidden_size
    );
    assert_eq!(
        layer.ffn_up.shape[0], ffn_dim,
        "ffn_up rows {} != ffn_gate rows {}",
        layer.ffn_up.shape[0], ffn_dim
    );
    assert_eq!(
        layer.ffn_up.shape[1], hidden_size,
        "ffn_up cols {} != hidden {}",
        layer.ffn_up.shape[1], hidden_size
    );
    assert_eq!(
        layer.ffn_down.shape[0], hidden_size,
        "ffn_down rows {} != hidden {}",
        layer.ffn_down.shape[0], hidden_size
    );
    assert_eq!(
        layer.ffn_down.shape[1], ffn_dim,
        "ffn_down cols {} != ffn_dim {}",
        layer.ffn_down.shape[1], ffn_dim
    );

    let mut ffn_out = vec![0.0f32; hidden_size];

    // cu46 Phase 1 step 2: env opt-in (RNB_MTP_DRAFTER_CUDA=1) + cuda feature 시
    // drafter FFN forward 를 GPU 로 (dense_q4k_silu_ffn_batch seq_len=1). 호환
    // 안 됨 (다른 quant) 또는 cuda 비활성 시 host fallback.
    let cuda_ok = rnb_runtime::policy::drafter_cuda_enabled()
        && crate::drafter::cuda::drafter_ffn_cuda(
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
            &hidden[..hidden_size],
            &mut ffn_out,
            hidden_size,
            ffn_dim,
        )
        .unwrap_or(false);

    if !cuda_ok {
        let gate_w = dequant_to_f32(&layer.ffn_gate);
        let up_w = dequant_to_f32(&layer.ffn_up);
        let down_w = dequant_to_f32(&layer.ffn_down);

        let mut gate = vec![0.0f32; ffn_dim];
        let mut up = vec![0.0f32; ffn_dim];
        matvec(&gate_w, ffn_dim, hidden_size, hidden, &mut gate);
        matvec(&up_w, ffn_dim, hidden_size, hidden, &mut up);

        // SwiGLU: silu(gate) * up
        for j in 0..ffn_dim {
            let g = gate[j];
            let silu = g / (1.0 + (-g).exp());
            gate[j] = silu * up[j];
        }

        matvec(&down_w, hidden_size, ffn_dim, &gate, &mut ffn_out);
    }

    // post_feedforward_layernorm + residual
    rms_norm_inplace(&mut ffn_out, &layer.post_ffw_norm, RMS_EPS);
    for i in 0..hidden_size {
        hidden[i] = residual[i] + ffn_out[i];
    }

    // ----- 3. layer_scalar (spec §3, mt83 정정) ---------------------------------
    // `hidden_states *= self.layer_scalar` — layer 전체 output 의 마지막
    // multiplication. mt83 cross_attention.rs 의 attention output 측 scale 은
    // verbatim 위반이라 본 구현에서는 layer 끝 (FFN residual 합산 이후) 으로
    // 정정.
    let scale = layer.layer_output_scale;
    for v in hidden.iter_mut() {
        *v *= scale;
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// In-place RMSNorm. `x[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
///
/// Gemma4 GGUF 는 `convert_hf_to_gguf.py` 의 `Gemma4Model.norm_shift = 0.0`
/// (Gemma3 의 1.0 과 정반대) 이라 unit-offset 미적용 형태로 저장되지만,
/// 실측 (mt86 Stage A) 결과 Llama-style `normed * weight` 가 target 과
/// 분포 일치 (embed_scale 적용 후 top1 = 23.44% 였고, unit-offset `(1+weight)`
/// 추가 시 1.56% 로 악화). 즉 Gemma4RMSNorm 자체가 Llama-style 이거나
/// 우리 target Engine 측의 norm 정책과 일치.
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(weight.len(), n, "rms_norm weight length mismatch");
    let mut acc = 0.0f64;
    for &v in x.iter() {
        acc += (v as f64) * (v as f64);
    }
    let mean_sq = acc / n as f64;
    let inv = ((mean_sq + eps as f64).sqrt() as f32).recip();
    for i in 0..n {
        x[i] = x[i] * inv * weight[i];
    }
}

/// Scalar f32 matvec — `out[r] = sum_c weight[r * cols + c] * x[c]`.
///
/// row-major `weight: [rows, cols]`, `x: [cols]`, `out: [rows]`.
fn matvec(weight: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    assert_eq!(weight.len(), rows * cols, "matvec shape mismatch");
    assert_eq!(x.len(), cols, "matvec x length mismatch");
    assert_eq!(out.len(), rows, "matvec out length mismatch");
    for r in 0..rows {
        let base = r * cols;
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += weight[base + c] * x[c];
        }
        out[r] = acc;
    }
}

/// Standard RoPE on Q only. Pairs `(i, i + head_dim/2)` rotated by
/// `theta_i = pos * base^(-2i / head_dim)`.
///
/// Spec §4 Critical: drafter 의 모든 layer 가 같은 `position_id`
/// (= last validated token position) 로 단일 회전. K 는 dict 에 이미 적용된
/// 상태라 추가 회전 금지.
fn apply_rope_q(q: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, base: f32) {
    assert!(
        head_dim % 2 == 0,
        "apply_rope_q: head_dim {head_dim} must be even"
    );
    let half = head_dim / 2;
    let pos_f = pos as f32;
    for h in 0..n_heads {
        let head = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let exponent = -(2.0 * i as f32) / (head_dim as f32);
            let theta = pos_f * base.powf(exponent);
            let (sin, cos) = theta.sin_cos();
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

/// Cross-attention with GQA.
///
/// **K/V layout (spec §4 Critical)**: row-major
/// `[n_kv_heads, seq_len, head_dim]` (Stage β 의 `transpose_to_head_major`
/// 가 변환). index 식:
/// - `K[h * seq_len * head_dim + t * head_dim + d]`
/// - `V[h * seq_len * head_dim + t * head_dim + d]`
///
/// mt83 의 `t * kv_dim + kv_h * head_dim` 식 사용 금지 (axis 다름).
///
/// Q layout: `[n_heads, head_dim]` row-major.
/// out layout: `[n_heads, head_dim]` row-major (overwritten).
///
/// SWA mask: drafter 의 `q_len = 1` 이라 SWA mask 가 trivial — `position_id`
/// 가 last validated token 이고 모든 K position 이 그 이전이므로 모든 K 에
/// attend 함 (causal mask trivial, SWA mask 도 sliding window 안에 들어옴).
/// 따라서 본 구현은 mask 없이 전체 seq_len attend.
fn cross_attention_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), n_heads * head_dim, "q shape mismatch");
    assert_eq!(out.len(), n_heads * head_dim, "out shape mismatch");
    assert_eq!(
        k.len(),
        n_kv_heads * seq_len * head_dim,
        "k length {} != n_kv_heads {} * seq_len {} * head_dim {}",
        k.len(),
        n_kv_heads,
        seq_len,
        head_dim
    );
    assert_eq!(
        v.len(),
        n_kv_heads * seq_len * head_dim,
        "v length {} != n_kv_heads {} * seq_len {} * head_dim {}",
        v.len(),
        n_kv_heads,
        seq_len,
        head_dim
    );
    assert!(
        n_heads % n_kv_heads == 0,
        "GQA: n_heads {n_heads} must be multiple of n_kv_heads {n_kv_heads}"
    );

    if seq_len == 0 {
        out.fill(0.0);
        return;
    }

    // Gemma 4 의 attention scaling = 1.0 (verbatim: modeling_gemma4.py:1178 의
    // `self.scaling = 1.0`, modeling_gemma4.py:1267 의 `attention_interface(scaling=1.0)`).
    // q_norm 이 magnitude 정규화를 담당하므로 표준 `1/sqrt(head_dim)` scaling 안 함.
    // mt84 spec §4 의 "Cross-attention (GQA)" 의사코드 line 133 의 `scaling=1.0` 명시.
    let scale: f32 = 1.0;
    let queries_per_kv = n_heads / n_kv_heads;
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..n_heads {
        let kv_h = h / queries_per_kv;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let kv_head_base = kv_h * seq_len * head_dim;

        // Score = Q · K_t scaled. K layout = [n_kv_heads, seq_len, head_dim].
        let mut max_s = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_off = kv_head_base + t * head_dim;
            let k_t = &k[k_off..k_off + head_dim];
            let mut s = 0.0f32;
            for j in 0..head_dim {
                s += q_h[j] * k_t[j];
            }
            let s = s * scale;
            scores[t] = s;
            if s > max_s {
                max_s = s;
            }
        }

        // Softmax (stable)
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            let e = (*s - max_s).exp();
            *s = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            for s in scores.iter_mut() {
                *s *= inv;
            }
        }

        // Attention output: sum_t softmax_t * V_t[kv_h]
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        out_h.fill(0.0);
        for t in 0..seq_len {
            let v_off = kv_head_base + t * head_dim;
            let v_t = &v[v_off..v_off + head_dim];
            let w = scores[t];
            for j in 0..head_dim {
                out_h[j] += w * v_t[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_inplace_unit_weight() {
        let mut x = vec![3.0f32, 4.0, 0.0];
        let w = vec![1.0f32, 1.0, 1.0];
        rms_norm_inplace(&mut x, &w, 0.0);
        // mean_sq = (9 + 16 + 0) / 3 = 25/3
        let inv = (25.0_f32 / 3.0).sqrt().recip();
        assert!((x[0] - 3.0 * inv).abs() < 1e-6);
        assert!((x[1] - 4.0 * inv).abs() < 1e-6);
        assert!((x[2] - 0.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn matvec_basic() {
        // weight = [[1, 2], [3, 4]], x = [5, 6], out = [17, 39]
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![5.0f32, 6.0];
        let mut out = vec![0.0f32; 2];
        matvec(&w, 2, 2, &x, &mut out);
        assert_eq!(out, vec![17.0, 39.0]);
    }

    #[test]
    fn cross_attention_gqa_zero_seq_returns_zero() {
        let q = vec![0.5f32; 4 * 8]; // 4 heads * 8 head_dim
        let k = vec![0.0f32; 0];
        let v = vec![0.0f32; 0];
        let mut out = vec![1.0f32; 4 * 8];
        cross_attention_gqa(&q, &k, &v, 0, 4, 2, 8, &mut out);
        for &x in out.iter() {
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn cross_attention_gqa_uniform_kv_produces_uniform_attention() {
        // seq_len=3, n_heads=4, n_kv_heads=2, head_dim=4 — uniform K/V means
        // softmax produces uniform weights, attention output = V_first_row.
        let head_dim = 4;
        let n_heads = 4;
        let n_kv_heads = 2;
        let seq_len = 3;
        let q = vec![1.0f32; n_heads * head_dim];
        // K = all ones. V[kv_h, t, d] = (kv_h + 1) * (t + 1).
        let k = vec![1.0f32; n_kv_heads * seq_len * head_dim];
        let mut v = vec![0.0f32; n_kv_heads * seq_len * head_dim];
        for h in 0..n_kv_heads {
            for t in 0..seq_len {
                for d in 0..head_dim {
                    v[h * seq_len * head_dim + t * head_dim + d] =
                        (h as f32 + 1.0) * (t as f32 + 1.0);
                }
            }
        }
        let mut out = vec![0.0f32; n_heads * head_dim];
        cross_attention_gqa(&q, &k, &v, seq_len, n_heads, n_kv_heads, head_dim, &mut out);
        // Uniform softmax: each weight = 1/3. Output per head =
        //   sum_t (1/3) * v[kv_h, t, d] = (kv_h+1) * (1+2+3)/3 = (kv_h+1) * 2.
        let queries_per_kv = n_heads / n_kv_heads;
        for h in 0..n_heads {
            let kv_h = h / queries_per_kv;
            let expected = (kv_h as f32 + 1.0) * 2.0;
            for d in 0..head_dim {
                let got = out[h * head_dim + d];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "h={h} d={d}: got {got}, expected {expected}"
                );
            }
        }
    }
}
