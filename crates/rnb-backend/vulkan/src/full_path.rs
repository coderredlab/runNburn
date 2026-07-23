//! Full-path GPU prefill / decode command-buffer builder (mv22 Task 8+).
//!
//! Owns the high-level "single-submit per step" orchestration that mv1-19's
//! window helpers don't expose. Existing `VulkanLayerGemv::embed_lookup`,
//! `kv_append`, `logit_argmax`, etc. each open + submit + wait their own
//! command buffer. This module's `run_prefill` records all per-layer Attention
//! dispatches into chunked command buffers and submits them in groups bounded
//! by `MAX_BATCH_OUTPUTS`.
//!
//! ## Design decisions (Task 8d)
//!
//! ### Cmd-buffer lifecycle — Option A (adopted)
//!
//! Two new public methods on `VulkanLayerGemv`:
//! - `begin_recording()` — resets + begins `self.command_buffer`; returns the
//!   raw handle for `record_*` calls.
//! - `submit_and_wait()` — ends + submits + waits for the fence.
//!
//! Option B (exposing `command_buffer` as `pub(crate)`) was rejected because
//! it breaks the invariant that only `VulkanLayerGemv` manages the fence
//! lifecycle and pipeline-set allocation.
//!
//! ### Weight wiring — stub interface (Task 10's responsibility)
//!
//! `FullPathPrefillInput::layer_weights` and `FullPathDecodeStepInput::layer_weights`
//! both accept `Option<&[LayerWeightHandles]>`.  When `None` the layer loop is
//! skipped entirely (only embed + argmax run).  Task 10 wires real GPU-resident
//! handles and passes `Some(handles)`.  This avoids the footgun of calling
//! `record_attention_layer_fullpath` with dangling or zero-length weight buffers.
//!
//! ### Staging buffer allocation — startup-hoisted (mv26-task10b-3)
//!
//! After the mv26-task10b-1 retro-fix, `AttentionLayerFullpathInput` carries
//! eight strided staging buffers covering every per-layer intermediate (no
//! more per-token GpuBuffer slices):
//! - `norm_attn_strided_buf` (`seq_len * hidden * 4`)
//! - `q_strided_buf` (`seq_len * num_heads * head_dim * 4`)
//! - `o_proj_input_strided_buf` (= attention output, `seq_len * num_heads * head_dim * 4`)
//! - `o_proj_out_strided_buf` (`seq_len * hidden * 4`)
//! - `norm_ffn_strided_buf` (`seq_len * hidden * 4`)
//! - `gate_strided_buf` (`seq_len * ffn_inner * 4`)
//! - `up_strided_buf` (`seq_len * ffn_inner * 4`)
//! - `down_strided_buf` (`seq_len * max(hidden, num_heads * head_dim, ffn_inner) * 4`)
//!
//! mv26-task10b-3 hoisted these onto `VulkanLayerGemv` itself
//! (`fullpath_staging_*` fields) — `run_prefill` / `run_decode_step` now call
//! `ensure_fullpath_staging` once and reuse the resident buffers across every
//! subsequent call. The capacity helper grows on demand and is a no-op when the
//! current shape already fits.
//!
//! ### Hidden ping-pong — startup-hoisted (mv26-task10b-3)
//!
//! Two device-local ping-pong buffers (`hidden_ping_a` / `hidden_ping_b`)
//! replace the old host `Vec<f32> hidden_buffer` round-trip for the
//! `layer_weights = Some(...)` path. Layer N reads from one and writes to the
//! other; the alternation is driven by `layer_idx % 2` inside
//! `run_layer_loop`. This guarantees the `hidden_in != hidden_out` aliasing
//! check inside `record_attention_layer_fullpath` (mv26-task10b-1 cleanup) sees
//! genuinely distinct buffers.
//!
//! The `layer_weights = None` smoke-test path is unchanged — it still uses the
//! host `hidden_buffer` directly between `embed_lookup` and `logit_argmax`.
//! Wiring real GPU forward (embed → ping_a copy, last layer → host download
//! before logit_argmax) is mv26-task10b-4's responsibility.
//!
//! ### GDN layers — recorded fullpath
//!
//! `ModelLayerKind::Recurrent` layers are recorded through
//! `record_gdn_layer_fullpath`: norm, fused QKV/conv, delta-state update,
//! gated norm, SSM out, residual, and FFN all stay in Vulkan command buffers.
//! The state buffers are host-visible allocations today, but the production
//! path does not materialize intermediate tensors on the CPU. Any future CPU
//! tensor roundtrip in the production path is rejected by
//! `ensure_no_forbidden_fullpath_cpu_escape`.
//!
//! ### MoE layers — Err (out of scope for dense PoC)
//!
//! `ModelLayerKind::MoE` returns `Err` immediately.
//!
//! ### run_layer_loop reuse (Task 9)
//!
//! `run_prefill` and `run_decode_step` share the same `run_layer_loop` helper
//! via `LayerLoopParams`.  The only decode-specific differences are
//! `seq_len = 1` and `pos_start = kv_cursor` (non-zero).  Chunking is retained
//! for decode even though seq_len=1 makes overflow unlikely — correctness on
//! 24-layer models hasn't been validated in a single submit for all pipelines.
//!
//! ## Roadmap
//!
//! - **Task 8d** (this file): 24-layer loop architecture, chunking,
//!   `begin_recording`/`submit_and_wait` API.
//! - **Task 9** (this file): `run_decode_step` — seq_len=1, pos_start=kv_cursor.
//! - **Task 10**: engine wiring — real GPU-resident weight handles.
//! - **Task 11**: ADB smoke test on Lenovo TB373FU.

use crate::context::GpuBuffer;
use crate::ffi::types::*;
use crate::kv_resident::KvResidentLayout;
use crate::layer_gemv::{
    descriptor_window_limit, embed_lookup_row_bytes, logit_argmax_row_bytes,
    AttentionLayerFullpathInput, AttentionLayerSetsConsumed, GdnLayerFullpathInput,
    RuntimeCounters, VulkanLayerGemv,
};
#[cfg(test)]
use crate::layer_gemv::{LEGACY_DESCRIPTOR_WINDOW, MAX_BATCH_OUTPUTS};
use crate::staging::StagingPolicy;
use crate::weight_cache::QuantType;
use rnb_loader::ModelLayerKind;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Weight-handle stub (Task 10 fills in real GPU-resident buffers)
// ---------------------------------------------------------------------------

/// Weight buffers for a single transformer layer.
///
/// All `&GpuBuffer` references must remain valid for the duration of
/// `run_prefill`.  Task 10 uploads these once at engine init and stores them
/// in the engine struct; this type is a view into that storage.
///
/// Fields mirror the weight fields of `AttentionLayerFullpathInput` so that
/// `run_prefill` can construct the full input without knowing the inner layout
/// of the weight cache.
#[allow(dead_code)] // constructed by Task 10's engine wiring
pub struct LayerWeightHandles<'a> {
    // Attention norm
    pub attn_norm_buf: &'a GpuBuffer,
    pub attn_norm_size: u64,
    // Q/K/V projections
    pub q_weight_buf: &'a GpuBuffer,
    pub q_weight_size: u64,
    pub q_rows: usize,
    pub q_cols: usize,
    pub q_quant: QuantType,
    /// Per-layer attention head dimension. Some architectures use different
    /// dimensions for local and global attention layers.
    pub head_dim: usize,
    pub q_bias_buf: Option<&'a GpuBuffer>,
    pub q_bias_size: u64,
    pub q_norm_buf: Option<&'a GpuBuffer>,
    pub q_norm_size: u64,
    /// One buffer per KV head.
    pub k_weight_bufs: &'a [&'a GpuBuffer],
    pub k_weight_size: u64,
    pub k_quant: QuantType,
    pub k_bias_buf: Option<&'a GpuBuffer>,
    pub k_bias_size: u64,
    pub k_norm_buf: Option<&'a GpuBuffer>,
    pub k_norm_size: u64,
    /// One buffer per KV head.
    pub v_weight_bufs: &'a [&'a GpuBuffer],
    pub v_weight_size: u64,
    pub v_quant: QuantType,
    pub v_bias_buf: Option<&'a GpuBuffer>,
    pub v_bias_size: u64,
    // Output projection
    pub o_weight_buf: &'a GpuBuffer,
    pub o_weight_size: u64,
    pub o_quant: QuantType,
    // FFN norm
    pub ffn_norm_buf: &'a GpuBuffer,
    pub ffn_norm_size: u64,
    // FFN gate/up/down
    pub gate_weight_buf: &'a GpuBuffer,
    pub gate_weight_size: u64,
    pub gate_quant: QuantType,
    pub up_weight_buf: &'a GpuBuffer,
    pub up_weight_size: u64,
    pub up_quant: QuantType,
    pub down_weight_buf: &'a GpuBuffer,
    pub down_weight_size: u64,
    pub down_quant: QuantType,
    // mv26-task10b-1: per-token intermediate GpuBuffer slices removed.
    // Every per-layer intermediate (norm_attn / q / attn / o_proj_out /
    // norm_ffn / gate / up / down) now lives in a contiguous strided staging
    // buffer owned by the caller (`run_layer_loop` allocates them per-call;
    // mv26-task10b-3 will hoist them to startup-alloc). The strided buffers
    // are passed via `AttentionLayerFullpathInput`'s `*_strided_buf` fields,
    // not via `LayerWeightHandles`.
}

/// Weight buffers for a single hybrid GatedDeltaNet (GDN) layer.
///
/// 대응 관계:
/// - `LayerWeightHandles<'a>` — pure-attention 레이어용 (Q/K/V/O + FFN)
/// - `GdnLayerWeightHandles<'a>` — GDN 레이어용 (fused QKV + SSM 파라미터 +
///   z gate + FFN)
///
/// Field source of truth: `GdnLayerWeights` in
/// `crates/rnb-llm/src/engine/layer_weights.rs:47-67`. Fixed f32 tensors
/// (`attn_norm`, `post_attn_norm`, `ssm_a`, `ssm_conv1d`, `ssm_dt_bias`,
/// `ssm_norm`) use `GpuWeightMode::Raw32`; projection tensors, including
/// quantized `ssm_alpha` / `ssm_beta`, use the `WeightCache` SoA path.
///
/// All `&GpuBuffer` references must remain valid for the duration of
/// `run_prefill` / `run_decode_step`. mv28 task 10b-5a defines the type only;
/// 5b/5c/5d wire the rnb-runtime / rnb-llm extraction and the
/// `record_gdn_layer_fullpath` consumer.
///
/// 필드 순서: norm → fused QKV → z gate → SSM α/β → SSM raw 묶음 (a /
/// conv1d / dt_bias / norm) → SSM out → post-attn norm → FFN gate/up/down.
///
/// Out of scope (5a):
/// - `ffn_gate_up_fused` (Qwen3.5 0.8B 미사용; 필요 모델 등장 시 추가)
/// - `moe_qwen` (Qwen3.5 35B-A3B 전용; dense PoC 범위 밖)
#[allow(dead_code)] // constructed by Task 10b-5b's rnb-runtime wrapper
pub struct GdnLayerWeightHandles<'a> {
    // Attention norm (Raw32)
    pub attn_norm_buf: &'a GpuBuffer,
    pub attn_norm_size: u64,
    // Fused QKV projection — [conv_channels, hidden]
    pub qkv_weight_buf: &'a GpuBuffer,
    pub qkv_weight_size: u64,
    pub qkv_rows: usize,
    pub qkv_cols: usize,
    pub qkv_quant: QuantType,
    // z gate (SSM input gating) — [d_inner, hidden]
    pub gate_weight_buf: &'a GpuBuffer,
    pub gate_weight_size: u64,
    pub gate_rows: usize,
    pub gate_cols: usize,
    pub gate_quant: QuantType,
    // SSM α / β — [num_heads, hidden]. None quant means F32 Raw32.
    pub ssm_alpha_buf: &'a GpuBuffer,
    pub ssm_alpha_size: u64,
    pub ssm_alpha_rows: usize,
    pub ssm_alpha_cols: usize,
    pub ssm_alpha_quant: Option<QuantType>,
    pub ssm_beta_buf: &'a GpuBuffer,
    pub ssm_beta_size: u64,
    pub ssm_beta_rows: usize,
    pub ssm_beta_cols: usize,
    pub ssm_beta_quant: Option<QuantType>,
    // SSM raw 묶음 (Raw32)
    /// `A_log` per head — [num_heads]
    pub ssm_a_buf: &'a GpuBuffer,
    pub ssm_a_size: u64,
    /// conv1d kernel — [conv_kernel, conv_channels]
    pub ssm_conv1d_buf: &'a GpuBuffer,
    pub ssm_conv1d_size: u64,
    /// Δt bias per head — [num_heads]
    pub ssm_dt_bias_buf: &'a GpuBuffer,
    pub ssm_dt_bias_size: u64,
    /// per-head-dim RMS norm — [head_v_dim]
    pub ssm_norm_buf: &'a GpuBuffer,
    pub ssm_norm_size: u64,
    /// GDN key/query group count (`metadata.ssm_n_group`), independent from
    /// attention KV head count.
    pub num_k_heads: usize,
    /// GDN key/query state width (`metadata.ssm_d_state`).
    pub head_k_dim: usize,
    // SSM out projection — [hidden, d_inner]
    pub ssm_out_buf: &'a GpuBuffer,
    pub ssm_out_size: u64,
    pub ssm_out_rows: usize,
    pub ssm_out_cols: usize,
    pub ssm_out_quant: QuantType,
    // Post-attention norm (Raw32) — attention path 의 ffn_norm 과 같은 자리
    pub post_attn_norm_buf: &'a GpuBuffer,
    pub post_attn_norm_size: u64,
    // FFN gate/up/down (mirror LayerWeightHandles 마지막 블록)
    pub ffn_gate_weight_buf: &'a GpuBuffer,
    pub ffn_gate_weight_size: u64,
    pub ffn_gate_rows: usize,
    pub ffn_gate_cols: usize,
    pub ffn_gate_quant: QuantType,
    pub ffn_up_weight_buf: &'a GpuBuffer,
    pub ffn_up_weight_size: u64,
    pub ffn_up_rows: usize,
    pub ffn_up_cols: usize,
    pub ffn_up_quant: QuantType,
    pub ffn_down_weight_buf: &'a GpuBuffer,
    pub ffn_down_weight_size: u64,
    pub ffn_down_rows: usize,
    pub ffn_down_cols: usize,
    pub ffn_down_quant: QuantType,
}

/// Per-layer fullpath weight handle.
///
/// The backend owns this enum so higher layers do not need to leak their
/// wrapper-internal storage types across crate boundaries. Attention and GDN
/// handles can live in one slice whose order matches `layer_kinds`.
#[allow(dead_code)] // consumed by the 5d mixed-layer dispatch path
pub enum LayerHandle<'a> {
    Attention(LayerWeightHandles<'a>),
    Gdn(GdnLayerWeightHandles<'a>),
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// Inputs for a full prefill pass.
///
/// **Weight wiring is Task 10's responsibility** — pass `layer_weights: None`
/// until real GPU-resident buffers are available.  When `None` only the
/// `embed_lookup` + `logit_argmax` path executes (useful for plumbing smoke
/// tests); the layer loop is skipped entirely.
#[allow(dead_code)] // constructed by Task 10's engine wiring
pub struct FullPathPrefillInput<'a> {
    /// Tokenized prompt (input ids).
    pub prompt_token_ids: &'a [u32],
    /// Number of transformer layers in the model.
    pub num_layers: usize,
    /// Hidden dimension (= embedding dim = model dim).
    pub hidden: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of KV heads (GQA: num_kv_heads <= num_heads).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// FFN intermediate (inner) dimension.
    pub ffn_inner: usize,
    /// RMS norm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency (rope_theta from model metadata).
    pub base_freq: f32,
    /// Number of per-head dimensions to rotate. `0` means full `head_dim`.
    pub rope_dim: usize,
    /// True for NEOX/Gemma-style split-half pairing; false for adjacent pairs.
    pub rope_neox: bool,
    /// Vocab size — also the row count of the output (lm head) projection.
    pub vocab: usize,
    /// KV cache layout shared across the whole step.
    pub kv_layout: KvResidentLayout,
    /// Staging buffer policy (used by Task 10 weight upload path).
    #[allow(dead_code)]
    pub staging: StagingPolicy,
    /// Quantized `output.weight` (lm head). Required for the final argmax.
    pub output_table_q6k: &'a [u8],
    pub output_quant: QuantType,
    /// Final model RMSNorm weight (`output_norm.weight`).
    pub output_norm: &'a [f32],
    /// Quantized `token_embd.weight`. Required for the embedding lookup.
    pub embed_table_q6k: &'a [u8],
    pub embed_quant: QuantType,
    /// Per-layer weight handles.  `None` → layer loop is skipped; only
    /// embed_lookup + logit_argmax run (plumbing smoke-test mode).
    /// `Some(handles)` → full 24-layer forward pass; `handles.len()` must
    /// equal `num_layers`.
    ///
    /// Weight wiring is Task 10's responsibility.  Caller must provide
    /// GPU-resident buffers that remain valid for the duration of this call.
    pub layer_weights: Option<&'a [LayerHandle<'a>]>,
    /// Per-layer kind (Attention / Recurrent / MoE).  Must have length
    /// `num_layers` when `layer_weights.is_some()`; ignored otherwise.
    ///
    /// If empty (and `layer_weights` is `Some`), all layers are treated as
    /// Attention — no GDN/MoE branches are taken.  This is safe for callers
    /// that have not yet wired per-layer metadata, but produces incorrect
    /// results for models with non-Attention layers.
    pub layer_kinds: &'a [ModelLayerKind],
}

/// Result of a full prefill pass.
#[derive(Debug, Clone, Copy)]
pub struct FullPathPrefillOutput {
    /// argmax of logits at the last position — first decoded token id.
    pub last_token_id: u32,
    /// KV cursor after this step (= prompt_len for prefill).
    pub kv_cursor_after: usize,
    /// Aggregated counters for this step.
    pub counters: RuntimeCounters,
}

// ---------------------------------------------------------------------------
// Decode-step input / output types (Task 9)
// ---------------------------------------------------------------------------

/// Inputs for a single-token decode step.
///
/// **Weight wiring is Task 10's responsibility** — pass `layer_weights: None`
/// until real GPU-resident buffers are available.  When `None` only
/// `embed_lookup` + `logit_argmax` run (plumbing smoke-test mode).
#[allow(dead_code)] // constructed by Task 10's engine wiring
pub struct FullPathDecodeStepInput<'a> {
    /// The single token to decode (output of the previous step / prefill).
    pub token_id: u32,
    /// Current KV cache cursor — the number of tokens already in the cache
    /// (= seq_len after prefill + number of decode steps taken so far).
    /// This is the absolute position of `token_id` in the sequence.
    pub kv_cursor: usize,
    /// Number of transformer layers in the model.
    pub num_layers: usize,
    /// Hidden dimension (= embedding dim = model dim).
    pub hidden: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of KV heads (GQA: num_kv_heads <= num_heads).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// FFN intermediate (inner) dimension.
    pub ffn_inner: usize,
    /// RMS norm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency (rope_theta from model metadata).
    pub base_freq: f32,
    /// Number of per-head dimensions to rotate. `0` means full `head_dim`.
    pub rope_dim: usize,
    /// True for NEOX/Gemma-style split-half pairing; false for adjacent pairs.
    pub rope_neox: bool,
    /// Vocab size — also the row count of the output (lm head) projection.
    pub vocab: usize,
    /// KV cache layout shared across the whole step.
    pub kv_layout: KvResidentLayout,
    /// Staging buffer policy (used by Task 10 weight upload path).
    #[allow(dead_code)]
    pub staging: StagingPolicy,
    /// Quantized `output.weight` (lm head). Required for the final argmax.
    pub output_table_q6k: &'a [u8],
    pub output_quant: QuantType,
    /// Final model RMSNorm weight (`output_norm.weight`).
    pub output_norm: &'a [f32],
    /// Quantized `token_embd.weight`. Required for the embedding lookup.
    pub embed_table_q6k: &'a [u8],
    pub embed_quant: QuantType,
    /// Per-layer kind (Attention / Recurrent / MoE).  Must have length
    /// `num_layers` when `layer_weights.is_some()`; ignored otherwise.
    pub layer_kinds: &'a [ModelLayerKind],
    /// Per-layer weight handles.  `None` → layer loop is skipped; only
    /// embed_lookup + logit_argmax run (plumbing smoke-test mode).
    /// `Some(handles)` → full forward pass; `handles.len()` must equal
    /// `num_layers`.
    ///
    /// Weight wiring is Task 10's responsibility.  Caller must provide
    /// GPU-resident buffers that remain valid for the duration of this call.
    pub layer_weights: Option<&'a [LayerHandle<'a>]>,
}

/// Result of a single-token decode step.
#[derive(Debug, Clone, Copy)]
pub struct FullPathDecodeStepOutput {
    /// argmax of logits — the predicted next token id.
    pub next_token_id: u32,
    /// KV cursor after this step (= kv_cursor + 1).
    pub kv_cursor_after: usize,
    /// Aggregated counters for this step.
    pub counters: RuntimeCounters,
}

fn rms_norm_slice_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), String> {
    if input.len() != weight.len() || input.len() != output.len() {
        return Err(format!(
            "rms_norm_slice_into: input/weight/output len mismatch ({}/{}/{})",
            input.len(),
            weight.len(),
            output.len()
        ));
    }
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_sq + eps).powf(-0.5);
    for ((dst, src), wt) in output.iter_mut().zip(input.iter()).zip(weight.iter()) {
        *dst = *src * inv_rms * *wt;
    }
    Ok(())
}

fn fullpath_layer_trace_enabled() -> bool {
    std::env::var_os("RNB_DEBUG_FULLPATH_LAYER_TRACE").is_some()
}

fn gdn_stage_trace_enabled(layer_idx: usize) -> bool {
    std::env::var_os("RNB_DEBUG_GDN_STAGE_TRACE").is_some() && layer_idx == 0
}

fn attention_stage_trace_enabled(layer_idx: usize) -> bool {
    std::env::var_os("RNB_DEBUG_ATTN_STAGE_TRACE").is_some() && layer_idx == 3
}

fn fullpath_layer_profile_flush_enabled() -> bool {
    std::env::var_os("RNB_VULKAN_FULLPATH_PROFILE_LAYER_FLUSH").is_some()
}

fn fullpath_counters_log_enabled(profile: &FullpathProfile) -> bool {
    profile.enabled || std::env::var_os("RNB_VULKAN_FULLPATH_COUNTERS").is_some()
}

#[derive(Default)]
struct FullpathProfile {
    enabled: bool,
    kind: &'static str,
    staging_us: u128,
    embed_us: u128,
    layer_loop_us: u128,
    attention_record_us: u128,
    gdn_record_us: u128,
    submit_wait_us: u128,
    output_us: u128,
    total_us: u128,
}

impl FullpathProfile {
    fn new(kind: &'static str) -> Self {
        Self {
            enabled: std::env::var_os("RNB_VULKAN_FULLPATH_PROFILE").is_some(),
            kind,
            ..Self::default()
        }
    }

    #[cfg(test)]
    fn enabled_for_test(kind: &'static str) -> Self {
        Self {
            enabled: true,
            kind,
            ..Self::default()
        }
    }

    fn timer(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn elapsed_us(start: Option<Instant>) -> u128 {
        start.map_or(0, |t| duration_us(t.elapsed()))
    }

    fn finish(&mut self, start: Option<Instant>) {
        self.total_us = Self::elapsed_us(start);
    }

    fn summary_line(
        &self,
        counters: &RuntimeCounters,
        attention_chunks: usize,
        gdn_layers: usize,
    ) -> String {
        format!(
            "[fullpath:profile] kind={} total_ms={:.3} staging_ms={:.3} embed_ms={:.3} \
             layer_loop_ms={:.3} attention_record_ms={:.3} gdn_record_ms={:.3} \
             submit_wait_ms={:.3} output_ms={:.3} submits={} upload_bytes={} download_bytes={} \
             materializations={} host_tensor_roundtrip_bytes={} attention_chunks={} gdn_layers={}",
            self.kind,
            us_to_ms(self.total_us),
            us_to_ms(self.staging_us),
            us_to_ms(self.embed_us),
            us_to_ms(self.layer_loop_us),
            us_to_ms(self.attention_record_us),
            us_to_ms(self.gdn_record_us),
            us_to_ms(self.submit_wait_us),
            us_to_ms(self.output_us),
            counters.submits,
            counters.upload_bytes,
            counters.download_bytes,
            counters.materializations,
            counters.host_tensor_roundtrip_bytes,
            attention_chunks,
            gdn_layers
        )
    }
}

fn duration_us(duration: Duration) -> u128 {
    duration.as_micros()
}

fn us_to_ms(us: u128) -> f64 {
    us as f64 / 1000.0
}

fn emit_fullpath_buffer_stage_trace(
    gemv: &mut VulkanLayerGemv,
    tag: &str,
    layer_idx: usize,
    buf: &GpuBuffer,
    seq_len: usize,
    width: usize,
) -> Result<(), String> {
    emit_fullpath_buffer_stage_trace_at(gemv, tag, layer_idx, buf, 0, seq_len, width)
}

fn emit_fullpath_buffer_stage_trace_at(
    gemv: &mut VulkanLayerGemv,
    tag: &str,
    layer_idx: usize,
    buf: &GpuBuffer,
    byte_offset: u64,
    seq_len: usize,
    width: usize,
) -> Result<(), String> {
    let f32_len = seq_len
        .checked_mul(width)
        .ok_or("emit_fullpath_buffer_stage_trace: seq_len*width overflow")?;
    let data = gemv.debug_download_buffer_f32_range(
        buf,
        byte_offset,
        (f32_len * std::mem::size_of::<f32>()) as u64,
    )?;
    emit_fullpath_slice_stage_trace(tag, layer_idx, &data, seq_len, width);
    Ok(())
}

fn emit_fullpath_slice_stage_trace(
    tag: &str,
    layer_idx: usize,
    data: &[f32],
    seq_len: usize,
    width: usize,
) {
    if seq_len == 0 || width == 0 || data.len() < seq_len * width {
        eprintln!(
            "[gdn-stage-trace][{}] layer={} invalid len={} seq_len={} width={}",
            tag,
            layer_idx,
            data.len(),
            seq_len,
            width
        );
        return;
    }
    let start = (seq_len - 1) * width;
    let row = &data[start..start + width];
    let n = row.len().max(1) as f32;
    let mean = row.iter().sum::<f32>() / n;
    let l2 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    let first = &data[..width];
    let mid_start = (seq_len / 2) * width;
    let mid = &data[mid_start..mid_start + width];
    let prev_start = seq_len.saturating_sub(2) * width;
    let prev = &data[prev_start..prev_start + width];
    let first_head = first.iter().take(4).copied().collect::<Vec<_>>();
    let mid_head = mid.iter().take(4).copied().collect::<Vec<_>>();
    let prev_head = prev.iter().take(4).copied().collect::<Vec<_>>();
    let last_head = row.iter().take(4).copied().collect::<Vec<_>>();
    eprintln!(
        "[gdn-stage-trace][{}] layer={} mean={:.6} l2={:.6} first={:?} mid={:?} prev={:?} last={:?}",
        tag, layer_idx, mean, l2, first_head, mid_head, prev_head, last_head
    );
}

fn emit_fullpath_layer_trace(
    gemv: &mut VulkanLayerGemv,
    tag: &str,
    layer_idx: usize,
    buf: &GpuBuffer,
    seq_len: usize,
    hidden: usize,
) -> Result<(), String> {
    emit_fullpath_layer_trace_at(gemv, tag, layer_idx, buf, 0, seq_len, hidden)
}

fn emit_fullpath_layer_trace_at(
    gemv: &mut VulkanLayerGemv,
    tag: &str,
    layer_idx: usize,
    buf: &GpuBuffer,
    byte_offset: u64,
    seq_len: usize,
    hidden: usize,
) -> Result<(), String> {
    let f32_len = seq_len
        .checked_mul(hidden)
        .ok_or("emit_fullpath_layer_trace: seq_len*hidden overflow")?;
    let data = gemv.debug_download_buffer_f32_range(
        buf,
        byte_offset,
        (f32_len * std::mem::size_of::<f32>()) as u64,
    )?;
    let last_start = (seq_len - 1) * hidden;
    let row = &data[last_start..last_start + hidden];
    let n = row.len().max(1) as f32;
    let mean = row.iter().sum::<f32>() / n;
    let l2 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    let head = row.iter().take(4).copied().collect::<Vec<_>>();
    eprintln!(
        "[fullpath-layer-trace][{}] layer={} mean={:.6} l2={:.6} head={:?}",
        tag, layer_idx, mean, l2, head
    );
    Ok(())
}

fn ensure_no_forbidden_fullpath_cpu_escape(
    caller: &str,
    counters: &RuntimeCounters,
) -> Result<(), String> {
    if counters.has_forbidden_fullpath_cpu_escape() {
        return Err(format!(
            "{caller}: production fullpath attempted a forbidden CPU tensor escape \
             (host_tensor_roundtrip_bytes={} materializations={})",
            counters.host_tensor_roundtrip_bytes, counters.materializations
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// run_decode_step (Task 9)
// ---------------------------------------------------------------------------

/// Validate a `FullPathDecodeStepInput` before any GPU work is started.
///
/// Extracted from `run_decode_step` so that unit tests can exercise the
/// validation rules without needing a Vulkan device, and so the body of
/// `run_decode_step` reads as a sequence of high-level steps.
///
/// Errors returned here are formatted with the `full_path::run_decode_step:`
/// prefix so the call site of the failure remains identifiable.
#[allow(dead_code)] // exercised by tests; Task 10 wiring also calls indirectly via run_decode_step
pub(crate) fn validate_decode_step_input(
    input: &FullPathDecodeStepInput<'_>,
) -> Result<(), String> {
    embed_lookup_row_bytes(input.hidden as u32, input.embed_quant)
        .map_err(|err| format!("full_path::run_decode_step: {err}"))?;
    logit_argmax_row_bytes(input.hidden as u32, input.output_quant)
        .map_err(|err| format!("full_path::run_decode_step: {err}"))?;
    if input.vocab == 0 {
        return Err("full_path::run_decode_step: vocab must be > 0".into());
    }
    if input.num_layers == 0 {
        return Err("full_path::run_decode_step: num_layers must be > 0".into());
    }
    if input.output_norm.len() != input.hidden {
        return Err(format!(
            "full_path::run_decode_step: output_norm.len() {} != hidden {}",
            input.output_norm.len(),
            input.hidden
        ));
    }
    if let Some(wh) = input.layer_weights {
        if wh.len() != input.num_layers {
            return Err(format!(
                "full_path::run_decode_step: layer_weights.len() {} != num_layers {}",
                wh.len(),
                input.num_layers
            ));
        }
        if !input.layer_kinds.is_empty() && input.layer_kinds.len() != input.num_layers {
            return Err(format!(
                "full_path::run_decode_step: layer_kinds.len() {} != num_layers {}",
                input.layer_kinds.len(),
                input.num_layers
            ));
        }
    }
    Ok(())
}

/// Drive a single-token decode step on the GPU and return the argmax of the
/// next-token logits.
///
/// **Weight wiring is Task 10's responsibility** — see
/// `FullPathDecodeStepInput`.  When `input.layer_weights` is `None` only
/// `embed_lookup` + `logit_argmax` run (plumbing smoke-test path).  When
/// `Some(handles)` the full layer loop runs with `seq_len = 1` and
/// `pos_start = kv_cursor`.
///
/// Decode reuses the same `run_layer_loop` helper as `run_prefill`.  Chunking
/// is retained even though `seq_len = 1` makes overflow unlikely — a 24-layer
/// single-submit has not been validated against all Vulkan implementations.
/// Chunking overhead with seq_len=1 is negligible.
///
/// **Per-call staging alloc**: the four strided buffers are allocated fresh
/// each call and freed on exit (both success and error paths).  TODO(Task 10):
/// hoist to engine startup alongside the prefill buffers.
///
/// **GDN layers**: recorded via `record_gdn_layer_fullpath`; no CPU tensor
/// roundtrip is allowed in production fullpath.
///
/// **MoE layers**: returns `Err` immediately (out of scope for dense PoC).
#[allow(dead_code)] // called by Task 10's engine wiring
pub fn run_decode_step(
    gemv: &mut VulkanLayerGemv,
    input: FullPathDecodeStepInput<'_>,
) -> Result<FullPathDecodeStepOutput, String> {
    // ----- basic validation -----
    validate_decode_step_input(&input)?;
    let mut profile = FullpathProfile::new("decode");
    let total_timer = profile.timer();

    // kv_cursor is cast to u32 below as `pos_start`.  64-bit usize > u32::MAX
    // would silently wrap; trip a debug assertion instead so it surfaces in
    // tests/dev builds.  Real models cap kv_cursor at max_ctx (≤ 1M) so this
    // only protects against caller bugs that pass an out-of-range cursor.
    debug_assert!(
        input.kv_cursor <= u32::MAX as usize,
        "kv_cursor {} exceeds u32::MAX",
        input.kv_cursor
    );

    // seq_len = 1 for decode.
    let seq_len: usize = 1;
    let mut counters = RuntimeCounters::default();
    let production_fullpath = input.layer_weights.is_some();

    let mut attention_chunks: usize = 0;
    let mut gdn_layers: usize = 0;
    let next_token_id = if let Some(layer_weights) = input.layer_weights {
        // mv26-task10b-3: staging + hidden ping-pong are owned by
        // VulkanLayerGemv. `ensure_*` is idempotent — the first call allocates,
        // subsequent calls within capacity are no-ops. seq_len=1 for decode so
        // the alloc sizes here are tiny but we use the same code path to keep
        // run_prefill / run_decode_step symmetric.
        let stage_timer = profile.timer();
        let staging_inner = max_fullpath_staging_inner(
            layer_weights,
            input.hidden,
            input.num_kv_heads,
            input.ffn_inner,
        )?;
        gemv.ensure_fullpath_staging(
            seq_len as u32,
            input.hidden as u32,
            input.num_heads as u32,
            input.head_dim as u32,
            staging_inner as u32,
        )?;
        let hidden_cap_bytes = (seq_len * input.hidden * std::mem::size_of::<f32>()) as u64;
        gemv.ensure_hidden_ping_pong(hidden_cap_bytes)?;
        profile.staging_us += FullpathProfile::elapsed_us(stage_timer);

        let ping_a_view = {
            let buf = gemv
                .hidden_ping_a
                .as_ref()
                .ok_or_else(|| "run_decode_step: hidden_ping_a missing".to_string())?;
            GpuBuffer {
                buffer: buf.buffer,
                memory: 0,
                size: buf.size,
            }
        };
        let stage_timer = profile.timer();
        gemv.embed_lookup_compact_rows_bound_to_buffer(
            &[input.token_id],
            input.hidden as u32,
            input.vocab as u32,
            input.embed_quant,
            input.embed_table_q6k,
            &ping_a_view,
            0,
            hidden_cap_bytes,
        )?;
        counters.submits += 1;
        let embed_row_bytes = embed_lookup_row_bytes(input.hidden as u32, input.embed_quant)?;
        let embed_table_upload = (embed_row_bytes + 3) & !3;
        counters.upload_bytes += (std::mem::size_of::<u32>() + embed_table_upload) as u64;
        profile.embed_us += FullpathProfile::elapsed_us(stage_timer);

        let loop_params = LayerLoopParams {
            seq_len,
            hidden_token_start: 0,
            trace_tag: "decode",
            pos_start: input.kv_cursor as u32, // decode: absolute position = kv_cursor
            num_layers: input.num_layers,
            num_heads: input.num_heads,
            num_kv_heads: input.num_kv_heads,
            ffn_inner: input.ffn_inner,
            norm_eps: input.norm_eps,
            base_freq: input.base_freq,
            rope_dim: input.rope_dim,
            rope_neox: input.rope_neox,
            layer_kinds: input.layer_kinds,
            hidden: input.hidden,
            descriptor_limit: descriptor_window_limit(),
        };

        let stage_timer = profile.timer();
        run_layer_loop(
            gemv,
            &loop_params,
            layer_weights,
            &mut counters,
            &mut attention_chunks,
            &mut gdn_layers,
            &mut profile,
        )?;
        profile.layer_loop_us += FullpathProfile::elapsed_us(stage_timer);
        let final_ping = final_hidden_ping_after_layers(input.num_layers);
        let final_buf = {
            let opt = if matches!(final_ping, HiddenPing::A) {
                gemv.hidden_ping_a.as_ref()
            } else {
                gemv.hidden_ping_b.as_ref()
            };
            let buf =
                opt.ok_or_else(|| "run_decode_step: final hidden ping missing".to_string())?;
            GpuBuffer {
                buffer: buf.buffer,
                memory: 0,
                size: buf.size,
            }
        };
        let stage_timer = profile.timer();
        gemv.ensure_output_table_bound(
            input.output_table_q6k,
            input.output_quant,
            input.vocab as u32,
            input.hidden as u32,
        )?;
        let token = gemv.logit_argmax_bound_from_buffer_with_norm(
            &final_buf,
            0,
            input.output_norm,
            input.norm_eps,
            input.output_quant,
            input.vocab as u32,
            input.hidden as u32,
        )?;
        counters.submits += 1;
        counters.upload_bytes += (input.output_norm.len() * std::mem::size_of::<f32>()) as u64;
        counters.download_bytes += 4;
        profile.output_us += FullpathProfile::elapsed_us(stage_timer);
        token
    } else {
        // ----- Smoke-test mode: embed_lookup → host hidden → logit_argmax -----
        let mut hidden_buffer = vec![0.0f32; input.hidden];
        let stage_timer = profile.timer();
        gemv.embed_lookup(
            &[input.token_id],
            input.hidden as u32,
            input.vocab as u32,
            input.embed_quant,
            input.embed_table_q6k,
            &mut hidden_buffer,
        )?;
        counters.submits += 1;
        counters.upload_bytes += std::mem::size_of::<u32>() as u64;
        counters.upload_bytes += input.embed_table_q6k.len() as u64;
        let hidden_download_bytes = (input.hidden * std::mem::size_of::<f32>()) as u64;
        profile.embed_us += FullpathProfile::elapsed_us(stage_timer);
        eprintln!(
            "[fullpath:run_decode_step] layer_weights=None — skipping layer loop \
             (smoke-test mode). Weight wiring is Task 10's responsibility; \
             caller must provide resident GPU buffers."
        );
        let mut normed = vec![0.0f32; input.hidden];
        rms_norm_slice_into(
            &hidden_buffer,
            input.output_norm,
            input.norm_eps,
            &mut normed,
        )?;
        let stage_timer = profile.timer();
        let token = gemv.logit_argmax(
            &normed,
            input.output_table_q6k,
            input.output_quant,
            input.vocab as u32,
            input.hidden as u32,
        )?;
        counters.submits += 1;
        let norm_upload_bytes = (normed.len() * std::mem::size_of::<f32>()) as u64;
        counters.record_host_tensor_roundtrip(hidden_download_bytes, norm_upload_bytes);
        counters.upload_bytes += input.output_table_q6k.len() as u64;
        counters.download_bytes += 4;
        profile.output_us += FullpathProfile::elapsed_us(stage_timer);
        token
    };

    profile.finish(total_timer);
    if fullpath_counters_log_enabled(&profile) {
        eprintln!(
            "[fullpath:decode:counters] submits={} upload_bytes={} download_bytes={} \
             materializations={} host_tensor_roundtrip_bytes={} kv_cursor={} hidden={} num_layers={} \
             attention_chunks={} gdn_layers={} kv_layers={}",
            counters.submits,
            counters.upload_bytes,
            counters.download_bytes,
            counters.materializations,
            counters.host_tensor_roundtrip_bytes,
            input.kv_cursor,
            input.hidden,
            input.num_layers,
            attention_chunks,
            gdn_layers,
            input.kv_layout.num_layers,
        );
    }
    if profile.enabled {
        eprintln!(
            "{}",
            profile.summary_line(&counters, attention_chunks, gdn_layers)
        );
    }
    if production_fullpath {
        ensure_no_forbidden_fullpath_cpu_escape("full_path::run_decode_step", &counters)?;
    }

    Ok(FullPathDecodeStepOutput {
        next_token_id,
        kv_cursor_after: input.kv_cursor + 1,
        counters,
    })
}

// ---------------------------------------------------------------------------
// Strided staging buffer allocation
// ---------------------------------------------------------------------------
//
// mv26-task10b-3: the per-call `alloc_fullpath_staging` helper that used to
// live here was hoisted onto `VulkanLayerGemv` (`ensure_fullpath_staging`).
// Callers now invoke `gemv.ensure_fullpath_staging(...)` once at the top of
// `run_prefill` / `run_decode_step` and read the resident buffers via the
// `fullpath_staging_*` fields. Hidden ping-pong (`hidden_ping_a` /
// `hidden_ping_b`) follows the same pattern via `ensure_hidden_ping_pong`.

// ---------------------------------------------------------------------------
// Chunking helper
// ---------------------------------------------------------------------------

/// Descriptor-set index tracker for the layer loop.
///
/// Each cached pipeline owns `MAX_BATCH_OUTPUTS` descriptor-set slots.  The
/// per-layer consumption from `AttentionLayerSetsConsumed::max_per_pipeline()`
/// accumulates here.  When adding the next layer would overflow the budget a
/// mid-loop flush is triggered: the current command buffer is submitted,
/// all counters are reset to 0, and recording restarts.
#[allow(dead_code)] // used in tests + Task 10
#[derive(Default)]
struct SetIdxCounters {
    norm: usize,
    q: usize,
    k: usize,
    v: usize,
    attn: usize,
    o: usize,
    silu: usize,
    add: usize,
    bias: usize,
    gate: usize,
    up: usize,
    down: usize,
    /// Shared rope_apply pipeline pool: rope_q + rope_k combined.
    rope: usize,
}

impl SetIdxCounters {
    /// Would adding `consumed` overflow any pipeline's pool?
    fn would_overflow(
        &self,
        consumed: &AttentionLayerSetsConsumed,
        descriptor_limit: usize,
    ) -> bool {
        self.norm + consumed.norm > descriptor_limit
            || self.q + consumed.q > descriptor_limit
            || self.k + consumed.k > descriptor_limit
            || self.v + consumed.v > descriptor_limit
            || self.attn + consumed.attn > descriptor_limit
            || self.o + consumed.o > descriptor_limit
            || self.silu + consumed.silu > descriptor_limit
            || self.add + consumed.add > descriptor_limit
            || self.bias + consumed.bias > descriptor_limit
            || self.gate + consumed.gate > descriptor_limit
            || self.up + consumed.up > descriptor_limit
            || self.down + consumed.down > descriptor_limit
            || self.rope + consumed.rope_q + consumed.rope_k > descriptor_limit
    }

    fn advance(&mut self, consumed: &AttentionLayerSetsConsumed) {
        self.norm += consumed.norm;
        self.q += consumed.q;
        self.k += consumed.k;
        self.v += consumed.v;
        self.attn += consumed.attn;
        self.o += consumed.o;
        self.silu += consumed.silu;
        self.add += consumed.add;
        self.bias += consumed.bias;
        self.gate += consumed.gate;
        self.up += consumed.up;
        self.down += consumed.down;
        self.rope += consumed.rope_q + consumed.rope_k;
    }

    fn reset(&mut self) {
        *self = Default::default();
    }
}

// ---------------------------------------------------------------------------
// run_prefill
// ---------------------------------------------------------------------------

/// Drive a full-path prefill on the GPU and return the argmax of logits at
/// the final prompt position.
///
/// **Weight wiring is Task 10's responsibility** — see `FullPathPrefillInput`.
///
/// When `input.layer_weights` is `None` only `embed_lookup` + `logit_argmax`
/// run (plumbing smoke-test path).  When `Some(handles)` the full 24-layer
/// loop runs: Attention layers are recorded into chunked command buffers
/// (flushed when `MAX_BATCH_OUTPUTS` is approached), GDN layers are recorded
/// into the same fullpath command-buffer flow, and MoE layers return `Err`
/// immediately.
///
/// **Counter semantics**: `upload_bytes` in smoke-test mode includes both the
/// per-step token-id transfer and the weight table bytes.  Once Task 10 wires
/// resident GPU weights the weight bytes will no longer appear here.
#[allow(dead_code)] // called by Task 10's engine wiring
pub fn run_prefill(
    gemv: &mut VulkanLayerGemv,
    input: FullPathPrefillInput<'_>,
) -> Result<FullPathPrefillOutput, String> {
    // ----- basic validation -----
    let mut profile = FullpathProfile::new("prefill");
    let total_timer = profile.timer();
    if input.prompt_token_ids.is_empty() {
        return Err("full_path::run_prefill: prompt_token_ids must not be empty".into());
    }
    embed_lookup_row_bytes(input.hidden as u32, input.embed_quant)
        .map_err(|err| format!("full_path::run_prefill: {err}"))?;
    logit_argmax_row_bytes(input.hidden as u32, input.output_quant)
        .map_err(|err| format!("full_path::run_prefill: {err}"))?;
    if input.vocab == 0 {
        return Err("full_path::run_prefill: vocab must be > 0".into());
    }
    if input.num_layers == 0 {
        return Err("full_path::run_prefill: num_layers must be > 0".into());
    }
    if input.output_norm.len() != input.hidden {
        return Err(format!(
            "full_path::run_prefill: output_norm.len() {} != hidden {}",
            input.output_norm.len(),
            input.hidden
        ));
    }
    if let Some(wh) = input.layer_weights {
        if wh.len() != input.num_layers {
            return Err(format!(
                "full_path::run_prefill: layer_weights.len() {} != num_layers {}",
                wh.len(),
                input.num_layers
            ));
        }
        if !input.layer_kinds.is_empty() && input.layer_kinds.len() != input.num_layers {
            return Err(format!(
                "full_path::run_prefill: layer_kinds.len() {} != num_layers {}",
                input.layer_kinds.len(),
                input.num_layers
            ));
        }
    }

    let seq_len = input.prompt_token_ids.len();
    let mut counters = RuntimeCounters::default();
    let production_fullpath = input.layer_weights.is_some();

    let mut attention_chunks: usize = 0;
    let mut gdn_layers: usize = 0;
    let last_token_id = if let Some(layer_weights) = input.layer_weights {
        // mv26-task10b-3: 8 strided staging + 2 hidden ping-pong are owned by
        // VulkanLayerGemv. ensure_* is idempotent — the first call allocates,
        // subsequent calls within capacity are no-ops. The previous per-call
        // alloc/free helper (`alloc_fullpath_staging`) is gone.
        let stage_timer = profile.timer();
        let staging_inner = max_fullpath_staging_inner(
            layer_weights,
            input.hidden,
            input.num_kv_heads,
            input.ffn_inner,
        )?;
        gemv.ensure_fullpath_staging(
            seq_len as u32,
            input.hidden as u32,
            input.num_heads as u32,
            input.head_dim as u32,
            staging_inner as u32,
        )?;
        let hidden_cap_bytes = (seq_len * input.hidden * std::mem::size_of::<f32>()) as u64;
        gemv.ensure_hidden_ping_pong(hidden_cap_bytes)?;
        profile.staging_us += FullpathProfile::elapsed_us(stage_timer);

        let ping_a_view = {
            let buf = gemv
                .hidden_ping_a
                .as_ref()
                .ok_or_else(|| "run_prefill: hidden_ping_a missing".to_string())?;
            GpuBuffer {
                buffer: buf.buffer,
                memory: 0,
                size: buf.size,
            }
        };
        let stage_timer = profile.timer();
        gemv.embed_lookup_compact_rows_bound_to_buffer(
            input.prompt_token_ids,
            input.hidden as u32,
            input.vocab as u32,
            input.embed_quant,
            input.embed_table_q6k,
            &ping_a_view,
            0,
            hidden_cap_bytes,
        )?;
        counters.submits += 1;
        let embed_row_bytes = embed_lookup_row_bytes(input.hidden as u32, input.embed_quant)?;
        let embed_table_upload = (input.prompt_token_ids.len() * embed_row_bytes + 3) & !3;
        counters.upload_bytes +=
            (input.prompt_token_ids.len() * std::mem::size_of::<u32>() + embed_table_upload) as u64;
        profile.embed_us += FullpathProfile::elapsed_us(stage_timer);
        if fullpath_layer_trace_enabled() {
            emit_fullpath_layer_trace(
                gemv,
                "prefill-input",
                usize::MAX,
                &ping_a_view,
                seq_len,
                input.hidden,
            )?;
        }
        if gdn_stage_trace_enabled(0) {
            emit_fullpath_buffer_stage_trace(
                gemv,
                "fullpath-embed",
                0,
                &ping_a_view,
                seq_len,
                input.hidden,
            )?;
        }

        let mut loop_params = LayerLoopParams {
            seq_len,
            pos_start: 0,
            hidden_token_start: 0,
            trace_tag: "prefill",
            num_layers: input.num_layers,
            num_heads: input.num_heads,
            num_kv_heads: input.num_kv_heads,
            ffn_inner: input.ffn_inner,
            norm_eps: input.norm_eps,
            base_freq: input.base_freq,
            rope_dim: input.rope_dim,
            rope_neox: input.rope_neox,
            layer_kinds: input.layer_kinds,
            hidden: input.hidden,
            descriptor_limit: descriptor_window_limit(),
        };
        let descriptor_chunk_len = descriptor_safe_prefill_chunk_len(&loop_params, layer_weights)?;
        let stage_timer = profile.timer();
        for hidden_token_start in (0..seq_len).step_by(descriptor_chunk_len) {
            loop_params.seq_len = descriptor_chunk_len.min(seq_len - hidden_token_start);
            loop_params.pos_start = u32::try_from(hidden_token_start)
                .map_err(|_| "run_prefill: token position exceeds u32".to_string())?;
            loop_params.hidden_token_start = hidden_token_start;
            run_layer_loop(
                gemv,
                &loop_params,
                layer_weights,
                &mut counters,
                &mut attention_chunks,
                &mut gdn_layers,
                &mut profile,
            )?;
        }
        profile.layer_loop_us += FullpathProfile::elapsed_us(stage_timer);
        let final_ping = final_hidden_ping_after_layers(input.num_layers);
        let final_buf = {
            let opt = if matches!(final_ping, HiddenPing::A) {
                gemv.hidden_ping_a.as_ref()
            } else {
                gemv.hidden_ping_b.as_ref()
            };
            let buf = opt.ok_or_else(|| "run_prefill: final hidden ping missing".to_string())?;
            GpuBuffer {
                buffer: buf.buffer,
                memory: 0,
                size: buf.size,
            }
        };
        let stage_timer = profile.timer();
        gemv.ensure_output_table_bound(
            input.output_table_q6k,
            input.output_quant,
            input.vocab as u32,
            input.hidden as u32,
        )?;
        let last_hidden_offset = ((seq_len - 1) * input.hidden * std::mem::size_of::<f32>()) as u64;
        let token = gemv.logit_argmax_bound_from_buffer_with_norm(
            &final_buf,
            last_hidden_offset,
            input.output_norm,
            input.norm_eps,
            input.output_quant,
            input.vocab as u32,
            input.hidden as u32,
        )?;
        counters.submits += 1;
        counters.upload_bytes += (input.output_norm.len() * std::mem::size_of::<f32>()) as u64;
        counters.download_bytes += 4;
        profile.output_us += FullpathProfile::elapsed_us(stage_timer);
        token
    } else {
        // ----- Smoke-test mode: embed_lookup → host hidden → logit_argmax -----
        let mut hidden_buffer = vec![0.0f32; seq_len * input.hidden];
        let stage_timer = profile.timer();
        gemv.embed_lookup(
            input.prompt_token_ids,
            input.hidden as u32,
            input.vocab as u32,
            input.embed_quant,
            input.embed_table_q6k,
            &mut hidden_buffer,
        )?;
        counters.submits += 1;
        counters.upload_bytes += (input.prompt_token_ids.len() * std::mem::size_of::<u32>()) as u64;
        counters.upload_bytes += input.embed_table_q6k.len() as u64;
        let hidden_download_bytes = (hidden_buffer.len() * std::mem::size_of::<f32>()) as u64;
        profile.embed_us += FullpathProfile::elapsed_us(stage_timer);
        eprintln!(
            "[fullpath:run_prefill] layer_weights=None — skipping layer loop \
             (smoke-test mode). Weight wiring is Task 10's responsibility; \
             caller must provide resident GPU buffers."
        );
        let last_offset = (seq_len - 1) * input.hidden;
        let last_hidden = &hidden_buffer[last_offset..last_offset + input.hidden];
        let mut normed = vec![0.0f32; input.hidden];
        rms_norm_slice_into(last_hidden, input.output_norm, input.norm_eps, &mut normed)?;
        let stage_timer = profile.timer();
        let token = gemv.logit_argmax(
            &normed,
            input.output_table_q6k,
            input.output_quant,
            input.vocab as u32,
            input.hidden as u32,
        )?;
        counters.submits += 1;
        let norm_upload_bytes = (normed.len() * std::mem::size_of::<f32>()) as u64;
        counters.record_host_tensor_roundtrip(hidden_download_bytes, norm_upload_bytes);
        counters.upload_bytes += input.output_table_q6k.len() as u64;
        counters.download_bytes += 4;
        profile.output_us += FullpathProfile::elapsed_us(stage_timer);
        token
    };

    profile.finish(total_timer);
    if fullpath_counters_log_enabled(&profile) {
        eprintln!(
            "[fullpath:counters] submits={} upload_bytes={} download_bytes={} \
             materializations={} host_tensor_roundtrip_bytes={} prompt_len={} hidden={} num_layers={} \
             attention_chunks={} gdn_layers={} kv_layers={}",
            counters.submits,
            counters.upload_bytes,
            counters.download_bytes,
            counters.materializations,
            counters.host_tensor_roundtrip_bytes,
            seq_len,
            input.hidden,
            input.num_layers,
            attention_chunks,
            gdn_layers,
            input.kv_layout.num_layers,
        );
    }
    if profile.enabled {
        eprintln!(
            "{}",
            profile.summary_line(&counters, attention_chunks, gdn_layers)
        );
    }
    if production_fullpath {
        ensure_no_forbidden_fullpath_cpu_escape("full_path::run_prefill", &counters)?;
    }

    Ok(FullPathPrefillOutput {
        last_token_id,
        kv_cursor_after: seq_len,
        counters,
    })
}

// ---------------------------------------------------------------------------
// Layer loop shared params (used by both run_prefill and run_decode_step)
// ---------------------------------------------------------------------------

/// Model-shape and positional parameters that `run_layer_loop` needs from
/// both `run_prefill` and `run_decode_step`.
///
/// Extracting these avoids duplicating the loop body: callers fill this from
/// their own input struct and pass it in.
#[derive(Clone, Copy)]
struct LayerLoopParams<'a> {
    /// Number of tokens in this descriptor-safe chunk.
    seq_len: usize,
    /// Absolute model position of the first token in this chunk.
    pos_start: u32,
    /// First hidden-row index in the full prompt ping-pong buffers.
    hidden_token_start: usize,
    trace_tag: &'static str,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    ffn_inner: usize,
    norm_eps: f32,
    base_freq: f32,
    rope_dim: usize,
    rope_neox: bool,
    layer_kinds: &'a [ModelLayerKind],
    hidden: usize,
    descriptor_limit: usize,
}

fn estimated_layer_descriptor_sets(
    params: &LayerLoopParams<'_>,
    layer_weights: &[LayerHandle<'_>],
    layer_idx: usize,
    seq_len: usize,
) -> Result<AttentionLayerSetsConsumed, String> {
    let kind = if params.layer_kinds.len() == params.num_layers {
        params.layer_kinds[layer_idx]
    } else {
        ModelLayerKind::Attention
    };
    match (kind, &layer_weights[layer_idx]) {
        (ModelLayerKind::Attention, LayerHandle::Attention(handle)) => {
            AttentionLayerSetsConsumed::attention_fullpath(
                seq_len,
                params.num_heads,
                params.num_kv_heads,
                handle.q_norm_buf.is_some(),
                handle.k_norm_buf.is_some(),
                handle.q_bias_buf.is_some(),
                handle.k_bias_buf.is_some(),
                handle.v_bias_buf.is_some(),
            )
            .ok_or_else(|| {
                format!("run_layer_loop: attention layer {layer_idx} descriptor-set count overflow")
            })
        }
        (ModelLayerKind::Recurrent, LayerHandle::Gdn(handle)) => {
            AttentionLayerSetsConsumed::gdn_fullpath(
                seq_len,
                handle.ssm_alpha_quant.is_some(),
                handle.ssm_beta_quant.is_some(),
            )
            .ok_or_else(|| {
                format!("run_layer_loop: GDN layer {layer_idx} descriptor-set count overflow")
            })
        }
        (ModelLayerKind::MoE, _) => Ok(AttentionLayerSetsConsumed::default()),
        (ModelLayerKind::Attention, LayerHandle::Gdn(_)) => Err(format!(
            "run_layer_loop: layer {layer_idx} kind=Attention but handle=Gdn"
        )),
        (ModelLayerKind::Recurrent, LayerHandle::Attention(_)) => Err(format!(
            "run_layer_loop: layer {layer_idx} kind=Recurrent but handle=Attention"
        )),
    }
}

fn largest_safe_descriptor_chunk_len<F>(
    requested_seq_len: usize,
    descriptor_limit: usize,
    mut fits: F,
) -> Result<usize, String>
where
    F: FnMut(usize) -> Result<bool, String>,
{
    if requested_seq_len == 0 {
        return Err("descriptor chunk length requires at least one token".to_string());
    }
    if !fits(1)? {
        return Err(format!(
            "one token exceeds descriptor window {descriptor_limit}"
        ));
    }

    let mut low = 1;
    let mut high = requested_seq_len;
    while low < high {
        let mid = low + (high - low + 1) / 2;
        if fits(mid)? {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(low)
}

fn descriptor_safe_prefill_chunk_len(
    params: &LayerLoopParams<'_>,
    layer_weights: &[LayerHandle<'_>],
) -> Result<usize, String> {
    largest_safe_descriptor_chunk_len(
        params.seq_len,
        params.descriptor_limit,
        |candidate_seq_len| {
            for layer_idx in 0..params.num_layers {
                let consumed = estimated_layer_descriptor_sets(
                    params,
                    layer_weights,
                    layer_idx,
                    candidate_seq_len,
                )?;
                if consumed.max_per_pipeline() > params.descriptor_limit {
                    return Ok(false);
                }
            }
            Ok(true)
        },
    )
}

// ---------------------------------------------------------------------------
// Layer loop (extracted to keep run_prefill / run_decode_step readable)
// ---------------------------------------------------------------------------

/// Stable raw pointers to the startup-hoisted staging + ping-pong buffers
/// owned by `VulkanLayerGemv`. mv26-task10b-3.
///
/// `run_layer_loop` snapshots all ten buffer references once at entry, then
/// reborrows them per layer to construct each `AttentionLayerFullpathInput`.
/// This is the only place in the file that needs `unsafe`, because Rust's
/// borrow checker rejects holding `&GpuBuffer` (immutable) into `gemv` while
/// also calling `gemv.record_*` (`&mut self`) inside the loop.
///
/// SAFETY contract (validated by `ensure_*` callers + the loop body):
/// 1. `ensure_fullpath_staging` and `ensure_hidden_ping_pong` succeed before
///    `snapshot()` runs, so every `Option<GpuBuffer>` is `Some`.
/// 2. The loop never calls any method that frees or re-allocates these
///    buffers (no `ensure_*` regrow, no `take()`).  The owning `VulkanLayerGemv`
///    outlives the loop, so the underlying `GpuBuffer`s do too.
/// 3. The only mutation happening on `gemv` during the loop is descriptor-set
///    bookkeeping inside `record_*` — none of those methods touch the staging
///    or ping-pong slots.
struct FullpathBufferRefs {
    norm_attn: *const GpuBuffer,
    q: *const GpuBuffer,
    o_proj_input: *const GpuBuffer,
    o_proj_out: *const GpuBuffer,
    norm_ffn: *const GpuBuffer,
    gate: *const GpuBuffer,
    up: *const GpuBuffer,
    down: *const GpuBuffer,
    ping_a: *const GpuBuffer,
    ping_b: *const GpuBuffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HiddenPing {
    A,
    B,
}

fn final_hidden_ping_after_layers(num_layers: usize) -> HiddenPing {
    debug_assert!(num_layers > 0);
    if num_layers % 2 == 0 {
        HiddenPing::A
    } else {
        HiddenPing::B
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GdnFullpathDims {
    hidden: usize,
    conv_channels: usize,
    d_inner: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_v_dim: usize,
    head_k_dim: usize,
    conv_kernel: usize,
}

fn raw32_elems(size_bytes: u64, label: &str) -> Result<usize, String> {
    if size_bytes == 0 || size_bytes % 4 != 0 {
        return Err(format!(
            "derive_gdn_fullpath_dims: {label} size {size_bytes} must be a positive f32 byte size",
        ));
    }
    Ok((size_bytes / 4) as usize)
}

fn derive_gdn_fullpath_dims(
    handle: &GdnLayerWeightHandles<'_>,
    hidden: usize,
) -> Result<GdnFullpathDims, String> {
    if hidden == 0 || handle.num_k_heads == 0 || handle.head_k_dim == 0 {
        return Err(format!(
            "derive_gdn_fullpath_dims: hidden ({hidden}), num_k_heads ({}) and head_k_dim ({}) must be > 0",
            handle.num_k_heads, handle.head_k_dim,
        ));
    }
    if handle.qkv_cols != hidden {
        return Err(format!(
            "derive_gdn_fullpath_dims: qkv_cols {} != hidden {}",
            handle.qkv_cols, hidden
        ));
    }
    if handle.gate_cols != hidden {
        return Err(format!(
            "derive_gdn_fullpath_dims: gate_cols {} != hidden {}",
            handle.gate_cols, hidden
        ));
    }
    if handle.ssm_out_rows != hidden {
        return Err(format!(
            "derive_gdn_fullpath_dims: ssm_out_rows {} != hidden {}",
            handle.ssm_out_rows, hidden
        ));
    }
    if handle.ffn_gate_cols != hidden
        || handle.ffn_up_cols != hidden
        || handle.ffn_down_rows != hidden
    {
        return Err(format!(
            "derive_gdn_fullpath_dims: FFN hidden shape mismatch \
             (gate_cols={} up_cols={} down_rows={} hidden={})",
            handle.ffn_gate_cols, handle.ffn_up_cols, handle.ffn_down_rows, hidden
        ));
    }
    if handle.ffn_gate_rows != handle.ffn_up_rows || handle.ffn_down_cols != handle.ffn_gate_rows {
        return Err(format!(
            "derive_gdn_fullpath_dims: FFN inner shape mismatch \
             (gate_rows={} up_rows={} down_cols={})",
            handle.ffn_gate_rows, handle.ffn_up_rows, handle.ffn_down_cols
        ));
    }

    let conv_channels = handle.qkv_rows;
    let d_inner = handle.gate_rows;
    if conv_channels <= d_inner {
        return Err(format!(
            "derive_gdn_fullpath_dims: qkv_rows/conv_channels {} must be greater than d_inner {}",
            conv_channels, d_inner
        ));
    }
    if handle.ssm_out_cols != d_inner {
        return Err(format!(
            "derive_gdn_fullpath_dims: ssm_out_cols {} != d_inner {}",
            handle.ssm_out_cols, d_inner
        ));
    }

    let head_v_dim = raw32_elems(handle.ssm_norm_size, "ssm_norm")?;
    if d_inner % head_v_dim != 0 {
        return Err(format!(
            "derive_gdn_fullpath_dims: d_inner {} not divisible by head_v_dim {}",
            d_inner, head_v_dim
        ));
    }
    let num_v_heads = d_inner / head_v_dim;
    if handle.ssm_alpha_rows != num_v_heads || handle.ssm_beta_rows != num_v_heads {
        return Err(format!(
            "derive_gdn_fullpath_dims: alpha/beta rows ({}/{}) != num_v_heads {}",
            handle.ssm_alpha_rows, handle.ssm_beta_rows, num_v_heads
        ));
    }
    if handle.ssm_alpha_cols != hidden || handle.ssm_beta_cols != hidden {
        return Err(format!(
            "derive_gdn_fullpath_dims: alpha/beta cols ({}/{}) != hidden {}",
            handle.ssm_alpha_cols, handle.ssm_beta_cols, hidden
        ));
    }
    if raw32_elems(handle.ssm_a_size, "ssm_a")? != num_v_heads {
        return Err(format!(
            "derive_gdn_fullpath_dims: ssm_a len != num_v_heads {}",
            num_v_heads
        ));
    }
    if raw32_elems(handle.ssm_dt_bias_size, "ssm_dt_bias")? != num_v_heads {
        return Err(format!(
            "derive_gdn_fullpath_dims: ssm_dt_bias len != num_v_heads {}",
            num_v_heads
        ));
    }

    let qk_total = conv_channels - d_inner;
    if qk_total % 2 != 0 {
        return Err(format!(
            "derive_gdn_fullpath_dims: conv_channels - d_inner must split evenly into q/k dims \
             (conv_channels={} d_inner={})",
            conv_channels, d_inner
        ));
    }
    let q_or_k_dim = qk_total / 2;
    let expected_q_or_k_dim = handle
        .num_k_heads
        .checked_mul(handle.head_k_dim)
        .ok_or("derive_gdn_fullpath_dims: num_k_heads*head_k_dim overflow")?;
    if q_or_k_dim != expected_q_or_k_dim {
        return Err(format!(
            "derive_gdn_fullpath_dims: q/k dim {} != num_k_heads*head_k_dim {}*{}={}",
            q_or_k_dim, handle.num_k_heads, handle.head_k_dim, expected_q_or_k_dim
        ));
    }

    let conv_kernel_elems = raw32_elems(handle.ssm_conv1d_size, "ssm_conv1d")?;
    if conv_kernel_elems % conv_channels != 0 {
        return Err(format!(
            "derive_gdn_fullpath_dims: ssm_conv1d elems {} not divisible by conv_channels {}",
            conv_kernel_elems, conv_channels
        ));
    }
    let conv_kernel = conv_kernel_elems / conv_channels;
    if conv_kernel < 2 {
        return Err(format!(
            "derive_gdn_fullpath_dims: conv_kernel {} must be >= 2",
            conv_kernel
        ));
    }

    Ok(GdnFullpathDims {
        hidden,
        conv_channels,
        d_inner,
        num_v_heads,
        num_k_heads: handle.num_k_heads,
        head_v_dim,
        head_k_dim: handle.head_k_dim,
        conv_kernel,
    })
}

fn max_fullpath_staging_inner(
    layer_weights: &[LayerHandle<'_>],
    hidden: usize,
    _num_kv_heads: usize,
    ffn_inner: usize,
) -> Result<usize, String> {
    let mut inner = ffn_inner;
    for handle in layer_weights {
        match handle {
            LayerHandle::Attention(attn) => {
                inner = inner.max(attn.q_rows);
            }
            LayerHandle::Gdn(gdn) => {
                let dims = derive_gdn_fullpath_dims(gdn, hidden)?;
                inner = inner.max(dims.conv_channels).max(dims.d_inner);
            }
        }
    }
    Ok(inner)
}

impl FullpathBufferRefs {
    fn snapshot(gemv: &VulkanLayerGemv) -> Result<Self, String> {
        let get = |opt: Option<&GpuBuffer>, name: &str| -> Result<*const GpuBuffer, String> {
            opt.map(|r| r as *const GpuBuffer)
                .ok_or_else(|| format!("run_layer_loop: {} staging/ping-pong missing", name))
        };
        Ok(Self {
            norm_attn: get(gemv.fullpath_staging_norm_attn.as_ref(), "norm_attn")?,
            q: get(gemv.fullpath_staging_q.as_ref(), "q")?,
            o_proj_input: get(gemv.fullpath_staging_o_proj_input.as_ref(), "o_proj_input")?,
            o_proj_out: get(gemv.fullpath_staging_o_proj_out.as_ref(), "o_proj_out")?,
            norm_ffn: get(gemv.fullpath_staging_norm_ffn.as_ref(), "norm_ffn")?,
            gate: get(gemv.fullpath_staging_gate.as_ref(), "gate")?,
            up: get(gemv.fullpath_staging_up.as_ref(), "up")?,
            down: get(gemv.fullpath_staging_down.as_ref(), "down")?,
            ping_a: get(gemv.hidden_ping_a.as_ref(), "hidden_ping_a")?,
            ping_b: get(gemv.hidden_ping_b.as_ref(), "hidden_ping_b")?,
        })
    }

    /// Reborrow a raw pointer back into `&GpuBuffer` for one layer's
    /// `AttentionLayerFullpathInput`. SAFETY: see struct doc — the buffer lives
    /// in `VulkanLayerGemv` for the entire `run_layer_loop` call.
    #[inline]
    unsafe fn r<'a>(p: *const GpuBuffer) -> &'a GpuBuffer {
        unsafe { &*p }
    }
}

fn submit_layer_loop_chunk(
    gemv: &mut VulkanLayerGemv,
    counters: &mut RuntimeCounters,
    chunks: &mut usize,
    profile: &mut FullpathProfile,
) -> Result<(), String> {
    let timer = profile.timer();
    gemv.submit_and_wait()?;
    profile.submit_wait_us += FullpathProfile::elapsed_us(timer);
    counters.submits += 1;
    *chunks += 1;
    Ok(())
}

#[allow(clippy::too_many_arguments, dead_code)]
fn run_layer_loop(
    gemv: &mut VulkanLayerGemv,
    params: &LayerLoopParams<'_>,
    layer_weights: &[LayerHandle<'_>],
    counters: &mut RuntimeCounters,
    attention_chunks: &mut usize,
    gdn_layers: &mut usize,
    profile: &mut FullpathProfile,
) -> Result<(), String> {
    // mv26-task10b-3: snapshot the startup-hoisted buffer references on
    // VulkanLayerGemv. ensure_fullpath_staging / ensure_hidden_ping_pong run
    // in run_prefill / run_decode_step before us, so all ten Options are Some.
    let bufs = FullpathBufferRefs::snapshot(gemv)?;

    let seq_len = params.seq_len;
    let hidden_offset_bytes = params
        .hidden_token_start
        .checked_mul(params.hidden)
        .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or("run_layer_loop: hidden chunk byte offset overflow")?;

    // Descriptor-set indices accumulate across the layer loop.  When a layer
    // would overflow the pool we flush and restart from 0.
    let mut set_idx = SetIdxCounters::default();

    // True while a command buffer is open (between begin_recording and
    // submit_and_wait).  Starts false; set to true after the first
    // begin_recording call.
    let mut cmd_in_progress = false;
    // Raw cmdbuf handle returned by begin_recording.  Valid only while
    // cmd_in_progress == true.
    let mut cmdbuf: VkCommandBuffer = std::ptr::null_mut();
    let trace_layers = fullpath_layer_trace_enabled();

    // Default layer kind for models that don't supply layer_kinds.
    let default_attention = [ModelLayerKind::Attention];
    let layer_kinds: &[ModelLayerKind] = if params.layer_kinds.len() == params.num_layers {
        params.layer_kinds
    } else {
        &default_attention
    };
    let trace_tag = params.trace_tag;
    let layer_profile_flush = fullpath_layer_profile_flush_enabled();

    for layer_idx in 0..params.num_layers {
        let kind = if layer_kinds.len() == 1 {
            // single-element slice = "all attention" fallback
            ModelLayerKind::Attention
        } else {
            layer_kinds[layer_idx]
        };

        match kind {
            ModelLayerKind::MoE => {
                // Flush any open cmd buffer before returning an error.
                if cmd_in_progress {
                    submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
                }
                return Err(format!(
                    "full_path::run_layer_loop: MoE layer {} encountered — \
                     MoE is out of scope for the dense PoC (Task 8d). \
                     Run decode via the CPU or partial-offload path instead.",
                    layer_idx
                ));
            }

            ModelLayerKind::Recurrent => {
                let gdn_handle = match &layer_weights[layer_idx] {
                    LayerHandle::Gdn(handle) => handle,
                    LayerHandle::Attention(_) => {
                        return Err(format!(
                            "run_layer_loop: layer {layer_idx} kind=Recurrent but handle=Attention",
                        ));
                    }
                };

                let dims = derive_gdn_fullpath_dims(gdn_handle, params.hidden)?;
                let n = seq_len;
                let estimated = AttentionLayerSetsConsumed::gdn_fullpath(
                    n,
                    gdn_handle.ssm_alpha_quant.is_some(),
                    gdn_handle.ssm_beta_quant.is_some(),
                )
                .ok_or("run_layer_loop: GDN descriptor-set count overflow")?;
                if cmd_in_progress && set_idx.would_overflow(&estimated, params.descriptor_limit) {
                    submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
                    cmd_in_progress = false;
                    set_idx.reset();
                }
                if !cmd_in_progress {
                    cmdbuf = gemv.begin_recording()?;
                    cmd_in_progress = true;
                }

                let (hidden_in_ptr, hidden_out_ptr) = if layer_idx % 2 == 0 {
                    (bufs.ping_a, bufs.ping_b)
                } else {
                    (bufs.ping_b, bufs.ping_a)
                };
                let (
                    hidden_in_buf,
                    hidden_out_buf,
                    norm_attn_strided_buf,
                    alpha_strided_buf,
                    beta_strided_buf,
                    delta_strided_buf,
                    ssm_out_strided_buf,
                    norm_ffn_strided_buf,
                    conv_gated_strided_buf,
                    z_strided_buf,
                    down_strided_buf,
                ) = unsafe {
                    (
                        FullpathBufferRefs::r(hidden_in_ptr),
                        FullpathBufferRefs::r(hidden_out_ptr),
                        FullpathBufferRefs::r(bufs.norm_attn),
                        FullpathBufferRefs::r(bufs.q),
                        FullpathBufferRefs::r(bufs.o_proj_input),
                        FullpathBufferRefs::r(bufs.down),
                        FullpathBufferRefs::r(bufs.o_proj_out),
                        FullpathBufferRefs::r(bufs.norm_ffn),
                        FullpathBufferRefs::r(bufs.gate),
                        FullpathBufferRefs::r(bufs.up),
                        FullpathBufferRefs::r(bufs.down),
                    )
                };

                let fullpath_input = GdnLayerFullpathInput {
                    layer_idx,
                    hidden_in_buf,
                    hidden_in_offset: hidden_offset_bytes,
                    hidden_out_buf,
                    hidden_out_offset: hidden_offset_bytes,
                    norm_attn_strided_buf,
                    alpha_strided_buf,
                    beta_strided_buf,
                    delta_strided_buf,
                    ssm_out_strided_buf,
                    norm_ffn_strided_buf,
                    conv_gated_strided_buf,
                    z_strided_buf,
                    down_strided_buf,
                    attn_norm_buf: gdn_handle.attn_norm_buf,
                    attn_norm_size: gdn_handle.attn_norm_size,
                    qkv_weight_buf: gdn_handle.qkv_weight_buf,
                    qkv_weight_size: gdn_handle.qkv_weight_size,
                    qkv_quant: gdn_handle.qkv_quant,
                    gate_weight_buf: gdn_handle.gate_weight_buf,
                    gate_weight_size: gdn_handle.gate_weight_size,
                    gate_quant: gdn_handle.gate_quant,
                    ssm_alpha_buf: gdn_handle.ssm_alpha_buf,
                    ssm_alpha_size: gdn_handle.ssm_alpha_size,
                    ssm_alpha_quant: gdn_handle.ssm_alpha_quant,
                    ssm_beta_buf: gdn_handle.ssm_beta_buf,
                    ssm_beta_size: gdn_handle.ssm_beta_size,
                    ssm_beta_quant: gdn_handle.ssm_beta_quant,
                    ssm_a_buf: gdn_handle.ssm_a_buf,
                    ssm_a_size: gdn_handle.ssm_a_size,
                    ssm_conv1d_buf: gdn_handle.ssm_conv1d_buf,
                    ssm_conv1d_size: gdn_handle.ssm_conv1d_size,
                    ssm_dt_bias_buf: gdn_handle.ssm_dt_bias_buf,
                    ssm_dt_bias_size: gdn_handle.ssm_dt_bias_size,
                    ssm_norm_buf: gdn_handle.ssm_norm_buf,
                    ssm_norm_size: gdn_handle.ssm_norm_size,
                    ssm_out_weight_buf: gdn_handle.ssm_out_buf,
                    ssm_out_weight_size: gdn_handle.ssm_out_size,
                    ssm_out_quant: gdn_handle.ssm_out_quant,
                    ffn_norm_buf: gdn_handle.post_attn_norm_buf,
                    ffn_norm_size: gdn_handle.post_attn_norm_size,
                    ffn_gate_weight_buf: gdn_handle.ffn_gate_weight_buf,
                    ffn_gate_weight_size: gdn_handle.ffn_gate_weight_size,
                    ffn_gate_quant: gdn_handle.ffn_gate_quant,
                    ffn_up_weight_buf: gdn_handle.ffn_up_weight_buf,
                    ffn_up_weight_size: gdn_handle.ffn_up_weight_size,
                    ffn_up_quant: gdn_handle.ffn_up_quant,
                    ffn_down_weight_buf: gdn_handle.ffn_down_weight_buf,
                    ffn_down_weight_size: gdn_handle.ffn_down_weight_size,
                    ffn_down_quant: gdn_handle.ffn_down_quant,
                    seq_len: seq_len as u32,
                    hidden_dim: dims.hidden as u32,
                    conv_channels: dims.conv_channels as u32,
                    d_inner: dims.d_inner as u32,
                    num_k_heads: dims.num_k_heads as u32,
                    num_v_heads: dims.num_v_heads as u32,
                    head_k_dim: dims.head_k_dim as u32,
                    head_v_dim: dims.head_v_dim as u32,
                    conv_kernel: dims.conv_kernel as u32,
                    ffn_inner: params.ffn_inner as u32,
                    norm_eps: params.norm_eps,
                    set_idx_norm_base: set_idx.norm,
                    set_idx_q_base: set_idx.q,
                    set_idx_k_base: set_idx.k,
                    set_idx_v_base: set_idx.v,
                    set_idx_attn_base: set_idx.attn,
                    set_idx_o_base: set_idx.o,
                    set_idx_silu_base: set_idx.silu,
                    set_idx_add_base: set_idx.add,
                    set_idx_gate_base: set_idx.gate,
                    set_idx_up_base: set_idx.up,
                    set_idx_down_base: set_idx.down,
                };
                let record_timer = profile.timer();
                let consumed = gemv.record_gdn_layer_fullpath(cmdbuf, &fullpath_input)?;
                let record_us = FullpathProfile::elapsed_us(record_timer);
                profile.gdn_record_us += record_us;
                set_idx.advance(&consumed);
                if layer_profile_flush {
                    let submit_before = profile.submit_wait_us;
                    submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
                    let submit_us = profile.submit_wait_us - submit_before;
                    eprintln!(
                        "[fullpath:layer-profile] phase={} layer={} kind=gdn record_ms={:.3} submit_wait_ms={:.3}",
                        trace_tag,
                        layer_idx,
                        us_to_ms(record_us),
                        us_to_ms(submit_us)
                    );
                    cmd_in_progress = false;
                    set_idx.reset();
                }
                let trace_gdn_stage = gdn_stage_trace_enabled(layer_idx);
                if trace_layers || trace_gdn_stage {
                    let trace_scratch = if trace_gdn_stage {
                        Some(gemv.debug_fullpath_scratch_view()?)
                    } else {
                        None
                    };
                    let trace_conv_len = seq_len
                        .checked_mul(dims.conv_channels)
                        .ok_or("run_layer_loop: trace conv len overflow")?;
                    let trace_delta_len = seq_len
                        .checked_mul(dims.d_inner)
                        .ok_or("run_layer_loop: trace delta len overflow")?;
                    let trace_snapshot_f32_len = trace_conv_len
                        .checked_add(trace_delta_len)
                        .and_then(|x| x.checked_add(trace_conv_len))
                        .and_then(|x| x.checked_add(trace_delta_len))
                        .ok_or("run_layer_loop: trace snapshot len overflow")?;
                    let trace_hidden_out = GpuBuffer {
                        buffer: hidden_out_buf.buffer,
                        memory: 0,
                        size: hidden_out_buf.size,
                    };
                    let trace_norm_attn = GpuBuffer {
                        buffer: norm_attn_strided_buf.buffer,
                        memory: 0,
                        size: norm_attn_strided_buf.size,
                    };
                    let trace_alpha = GpuBuffer {
                        buffer: alpha_strided_buf.buffer,
                        memory: 0,
                        size: alpha_strided_buf.size,
                    };
                    let trace_beta = GpuBuffer {
                        buffer: beta_strided_buf.buffer,
                        memory: 0,
                        size: beta_strided_buf.size,
                    };
                    let trace_ssm_out = GpuBuffer {
                        buffer: ssm_out_strided_buf.buffer,
                        memory: 0,
                        size: ssm_out_strided_buf.size,
                    };
                    let trace_norm_ffn = GpuBuffer {
                        buffer: norm_ffn_strided_buf.buffer,
                        memory: 0,
                        size: norm_ffn_strided_buf.size,
                    };
                    let trace_ffn_down = GpuBuffer {
                        buffer: down_strided_buf.buffer,
                        memory: 0,
                        size: down_strided_buf.size,
                    };
                    submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
                    cmd_in_progress = false;
                    set_idx.reset();
                    if trace_gdn_stage {
                        if let Some(ref scratch) = trace_scratch {
                            let trace_snapshot_byte_offset = gdn_handle
                                .ssm_conv1d_size
                                .checked_add(
                                    ((dims.conv_kernel - 1)
                                        .checked_mul(dims.conv_channels)
                                        .ok_or("run_layer_loop: trace conv state len overflow")?
                                        * std::mem::size_of::<f32>())
                                        as u64,
                                )
                                .and_then(|x| {
                                    x.checked_add(
                                        (trace_conv_len * std::mem::size_of::<f32>()) as u64,
                                    )
                                })
                                .ok_or("run_layer_loop: trace snapshot byte offset overflow")?;
                            let snapshot = gemv.debug_download_buffer_f32_range(
                                scratch,
                                trace_snapshot_byte_offset,
                                (trace_snapshot_f32_len * std::mem::size_of::<f32>()) as u64,
                            )?;
                            let qkv_end = trace_conv_len;
                            let conv_end = qkv_end + trace_conv_len;
                            let delta_end = conv_end + trace_delta_len;
                            let gated_end = delta_end + trace_delta_len;
                            emit_fullpath_slice_stage_trace(
                                "fullpath-qkv-raw",
                                layer_idx,
                                &snapshot[..qkv_end],
                                seq_len,
                                dims.conv_channels,
                            );
                            emit_fullpath_slice_stage_trace(
                                "fullpath-conv",
                                layer_idx,
                                &snapshot[qkv_end..conv_end],
                                seq_len,
                                dims.conv_channels,
                            );
                            emit_fullpath_slice_stage_trace(
                                "fullpath-delta",
                                layer_idx,
                                &snapshot[conv_end..delta_end],
                                seq_len,
                                dims.d_inner,
                            );
                            emit_fullpath_slice_stage_trace(
                                "fullpath-gated",
                                layer_idx,
                                &snapshot[delta_end..gated_end],
                                seq_len,
                                dims.d_inner,
                            );
                        }
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "fullpath-conv-kernel",
                            layer_idx,
                            gdn_handle.ssm_conv1d_buf,
                            dims.conv_kernel,
                            dims.conv_channels,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "fullpath-norm-attn",
                            layer_idx,
                            &trace_norm_attn,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "fullpath-alpha-raw",
                            layer_idx,
                            &trace_alpha,
                            seq_len,
                            dims.num_v_heads,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "fullpath-beta-raw",
                            layer_idx,
                            &trace_beta,
                            seq_len,
                            dims.num_v_heads,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "fullpath-ssm-out",
                            layer_idx,
                            &trace_ssm_out,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "fullpath-norm-ffn",
                            layer_idx,
                            &trace_norm_ffn,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "fullpath-ffn-down",
                            layer_idx,
                            &trace_ffn_down,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace_at(
                            gemv,
                            "fullpath-final",
                            layer_idx,
                            &trace_hidden_out,
                            hidden_offset_bytes,
                            seq_len,
                            params.hidden,
                        )?;
                    }
                    if trace_layers {
                        emit_fullpath_layer_trace_at(
                            gemv,
                            trace_tag,
                            layer_idx,
                            &trace_hidden_out,
                            hidden_offset_bytes,
                            seq_len,
                            params.hidden,
                        )?;
                    }
                }
                *gdn_layers += 1;
            }

            ModelLayerKind::Attention => {
                let wh = match &layer_weights[layer_idx] {
                    LayerHandle::Attention(handle) => handle,
                    LayerHandle::Gdn(_) => {
                        return Err(format!(
                            "run_layer_loop: layer {layer_idx} kind=Attention but handle=Gdn",
                        ));
                    }
                };

                let n = seq_len;
                let estimated =
                    estimated_layer_descriptor_sets(params, layer_weights, layer_idx, n)?;

                gemv.ensure_attention_cache_layer_buffers_for_fullpath(
                    layer_idx,
                    params.num_kv_heads,
                    params.pos_start as usize + seq_len,
                    wh.head_dim,
                )?;

                // Flush if adding this layer would overflow any pool.
                if cmd_in_progress && set_idx.would_overflow(&estimated, params.descriptor_limit) {
                    submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
                    cmd_in_progress = false;
                    set_idx.reset();
                }

                // Begin a new command buffer if one isn't already open.
                if !cmd_in_progress {
                    cmdbuf = gemv.begin_recording()?;
                    cmd_in_progress = true;
                }

                // mv26-task10b-3: hidden ping-pong by `layer_idx % 2`.
                // Even layers read from `ping_a`, write to `ping_b`; odd layers
                // swap. Both buffers are device-local distinct allocations, so
                // the `hidden_in_buf != hidden_out_buf` check inside
                // `record_attention_layer_fullpath` (mv26-task10b-1 cleanup)
                // passes by construction.
                //
                // Wiring of the initial embed → ping_a copy and the final
                // ping_X → host download for logit_argmax is mv26-task10b-4's
                // responsibility (this code path only runs when
                // `layer_weights = Some(...)`, which task 10b-4 first
                // exercises).
                let (hidden_in_ptr, hidden_out_ptr) = if layer_idx % 2 == 0 {
                    (bufs.ping_a, bufs.ping_b)
                } else {
                    (bufs.ping_b, bufs.ping_a)
                };

                // SAFETY: see FullpathBufferRefs doc. The startup-hoisted
                // staging + ping-pong buffers live for the full duration of
                // run_layer_loop, and nothing in this loop frees or reallocates
                // them. The reborrow is necessary because `record_*` takes
                // `&mut gemv` while we still need shared `&GpuBuffer`s for
                // the input struct.
                let (
                    hidden_in_buf,
                    hidden_out_buf,
                    norm_attn_strided_buf,
                    q_strided_buf,
                    o_proj_input_strided_buf,
                    o_proj_out_strided_buf,
                    norm_ffn_strided_buf,
                    gate_strided_buf,
                    up_strided_buf,
                    down_strided_buf,
                ) = unsafe {
                    (
                        FullpathBufferRefs::r(hidden_in_ptr),
                        FullpathBufferRefs::r(hidden_out_ptr),
                        FullpathBufferRefs::r(bufs.norm_attn),
                        FullpathBufferRefs::r(bufs.q),
                        FullpathBufferRefs::r(bufs.o_proj_input),
                        FullpathBufferRefs::r(bufs.o_proj_out),
                        FullpathBufferRefs::r(bufs.norm_ffn),
                        FullpathBufferRefs::r(bufs.gate),
                        FullpathBufferRefs::r(bufs.up),
                        FullpathBufferRefs::r(bufs.down),
                    )
                };

                // Build the fullpath input for this layer. Every per-layer
                // intermediate slot points at a startup-hoisted device-local
                // staging or ping-pong buffer — no per-call allocations.
                let fullpath_input = AttentionLayerFullpathInput {
                    layer_idx,
                    hidden_in_buf,
                    hidden_in_offset: hidden_offset_bytes,
                    hidden_out_buf,
                    hidden_out_offset: hidden_offset_bytes,
                    norm_attn_strided_buf,
                    q_strided_buf,
                    o_proj_input_strided_buf,
                    o_proj_out_strided_buf,
                    norm_ffn_strided_buf,
                    gate_strided_buf,
                    up_strided_buf,
                    down_strided_buf,
                    attn_norm_buf: wh.attn_norm_buf,
                    attn_norm_size: wh.attn_norm_size,
                    q_weight_buf: wh.q_weight_buf,
                    q_weight_size: wh.q_weight_size,
                    q_rows: wh.q_rows as u32,
                    q_cols: wh.q_cols as u32,
                    q_quant: wh.q_quant,
                    q_bias_buf: wh.q_bias_buf,
                    q_bias_size: wh.q_bias_size,
                    q_norm_buf: wh.q_norm_buf,
                    q_norm_size: wh.q_norm_size,
                    k_weight_bufs: wh.k_weight_bufs,
                    k_weight_size: wh.k_weight_size,
                    k_quant: wh.k_quant,
                    k_bias_buf: wh.k_bias_buf,
                    k_bias_size: wh.k_bias_size,
                    k_norm_buf: wh.k_norm_buf,
                    k_norm_size: wh.k_norm_size,
                    v_weight_bufs: wh.v_weight_bufs,
                    v_weight_size: wh.v_weight_size,
                    v_quant: wh.v_quant,
                    v_bias_buf: wh.v_bias_buf,
                    v_bias_size: wh.v_bias_size,
                    o_weight_buf: wh.o_weight_buf,
                    o_weight_size: wh.o_weight_size,
                    o_quant: wh.o_quant,
                    ffn_norm_buf: wh.ffn_norm_buf,
                    ffn_norm_size: wh.ffn_norm_size,
                    gate_weight_buf: wh.gate_weight_buf,
                    gate_weight_size: wh.gate_weight_size,
                    gate_quant: wh.gate_quant,
                    up_weight_buf: wh.up_weight_buf,
                    up_weight_size: wh.up_weight_size,
                    up_quant: wh.up_quant,
                    down_weight_buf: wh.down_weight_buf,
                    down_weight_size: wh.down_weight_size,
                    down_quant: wh.down_quant,
                    seq_len: seq_len as u32,
                    hidden_dim: params.hidden as u32,
                    num_heads: params.num_heads as u32,
                    num_kv_heads: params.num_kv_heads as u32,
                    head_dim: wh.head_dim as u32,
                    ffn_inner: params.ffn_inner as u32,
                    norm_eps: params.norm_eps,
                    pos_start: params.pos_start,
                    base_freq: params.base_freq,
                    rope_dim: params.rope_dim.min(wh.head_dim) as u32,
                    rope_neox: params.rope_neox,
                    // Descriptor-set bases from current accumulator state.
                    set_idx_norm_base: set_idx.norm,
                    set_idx_q_base: set_idx.q,
                    set_idx_k_base: set_idx.k,
                    set_idx_v_base: set_idx.v,
                    set_idx_attn_base: set_idx.attn,
                    set_idx_o_base: set_idx.o,
                    set_idx_silu_base: set_idx.silu,
                    set_idx_add_base: set_idx.add,
                    set_idx_bias_base: set_idx.bias,
                    set_idx_gate_base: set_idx.gate,
                    set_idx_up_base: set_idx.up,
                    set_idx_down_base: set_idx.down,
                    set_idx_rope_q_base: set_idx.rope,
                    set_idx_rope_k_base: set_idx.rope + estimated.rope_q,
                };

                // Use the actual consumed counts returned by the callee as the
                // single source of truth for advance.  This prevents a mismatch
                // between the manually-computed `estimated` and the real counts
                // (e.g. norm: 2*n in the callee vs an older n here).
                let record_timer = profile.timer();
                let consumed = gemv.record_attention_layer_fullpath(cmdbuf, &fullpath_input)?;
                let record_us = FullpathProfile::elapsed_us(record_timer);
                profile.attention_record_us += record_us;
                set_idx.advance(&consumed);
                if layer_profile_flush {
                    let submit_before = profile.submit_wait_us;
                    submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
                    let submit_us = profile.submit_wait_us - submit_before;
                    eprintln!(
                        "[fullpath:layer-profile] phase={} layer={} kind=attention record_ms={:.3} submit_wait_ms={:.3}",
                        trace_tag,
                        layer_idx,
                        us_to_ms(record_us),
                        us_to_ms(submit_us)
                    );
                    cmd_in_progress = false;
                    set_idx.reset();
                }
                let trace_attention_stage = attention_stage_trace_enabled(layer_idx);
                if trace_layers || trace_attention_stage {
                    let trace_scratch = if trace_attention_stage {
                        Some(gemv.debug_fullpath_scratch_view()?)
                    } else {
                        None
                    };
                    let trace_buf = GpuBuffer {
                        buffer: hidden_out_buf.buffer,
                        memory: 0,
                        size: hidden_out_buf.size,
                    };
                    let trace_norm_attn = GpuBuffer {
                        buffer: norm_attn_strided_buf.buffer,
                        memory: 0,
                        size: norm_attn_strided_buf.size,
                    };
                    let trace_hidden_in = GpuBuffer {
                        buffer: hidden_in_buf.buffer,
                        memory: 0,
                        size: hidden_in_buf.size,
                    };
                    let trace_q = GpuBuffer {
                        buffer: q_strided_buf.buffer,
                        memory: 0,
                        size: q_strided_buf.size,
                    };
                    let trace_o_input = GpuBuffer {
                        buffer: o_proj_input_strided_buf.buffer,
                        memory: 0,
                        size: o_proj_input_strided_buf.size,
                    };
                    let trace_o_out = GpuBuffer {
                        buffer: o_proj_out_strided_buf.buffer,
                        memory: 0,
                        size: o_proj_out_strided_buf.size,
                    };
                    let trace_norm_ffn = GpuBuffer {
                        buffer: norm_ffn_strided_buf.buffer,
                        memory: 0,
                        size: norm_ffn_strided_buf.size,
                    };
                    let trace_down = GpuBuffer {
                        buffer: down_strided_buf.buffer,
                        memory: 0,
                        size: down_strided_buf.size,
                    };
                    submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
                    cmd_in_progress = false;
                    set_idx.reset();
                    if trace_attention_stage {
                        let q_dim_usize = params
                            .num_heads
                            .checked_mul(wh.head_dim)
                            .ok_or("run_layer_loop: attention trace q_dim overflow")?;
                        if let Some(ref scratch) = trace_scratch {
                            let q_full_len = seq_len
                                .checked_mul(wh.q_rows)
                                .ok_or("run_layer_loop: attention q-full trace len overflow")?;
                            let q_len = seq_len
                                .checked_mul(q_dim_usize)
                                .ok_or("run_layer_loop: attention q trace len overflow")?;
                            let q_split_start = q_full_len;
                            let q_norm_start = q_split_start + q_len;
                            let q_rope_start = q_norm_start + q_len;
                            let q_trace_len = q_rope_start + q_len;
                            let snapshot = gemv.debug_download_buffer_f32(
                                scratch,
                                (q_trace_len * std::mem::size_of::<f32>()) as u64,
                            )?;
                            emit_fullpath_slice_stage_trace(
                                "attn-q-full",
                                layer_idx,
                                &snapshot[..q_full_len],
                                seq_len,
                                wh.q_rows,
                            );
                            emit_fullpath_slice_stage_trace(
                                "attn-q-split",
                                layer_idx,
                                &snapshot[q_split_start..q_norm_start],
                                seq_len,
                                q_dim_usize,
                            );
                            emit_fullpath_slice_stage_trace(
                                "attn-q-norm",
                                layer_idx,
                                &snapshot[q_norm_start..q_rope_start],
                                seq_len,
                                q_dim_usize,
                            );
                            emit_fullpath_slice_stage_trace(
                                "attn-q-rope-snapshot",
                                layer_idx,
                                &snapshot[q_rope_start..q_trace_len],
                                seq_len,
                                q_dim_usize,
                            );
                        }
                        emit_fullpath_buffer_stage_trace_at(
                            gemv,
                            "attn-hidden-in",
                            layer_idx,
                            &trace_hidden_in,
                            hidden_offset_bytes,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "attn-norm-attn",
                            layer_idx,
                            &trace_norm_attn,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "attn-q-rope",
                            layer_idx,
                            &trace_q,
                            seq_len,
                            q_dim_usize,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "attn-o-input",
                            layer_idx,
                            &trace_o_input,
                            seq_len,
                            q_dim_usize,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "attn-o-out",
                            layer_idx,
                            &trace_o_out,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "attn-norm-ffn",
                            layer_idx,
                            &trace_norm_ffn,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace(
                            gemv,
                            "attn-ffn-down",
                            layer_idx,
                            &trace_down,
                            seq_len,
                            params.hidden,
                        )?;
                        emit_fullpath_buffer_stage_trace_at(
                            gemv,
                            "attn-final",
                            layer_idx,
                            &trace_buf,
                            hidden_offset_bytes,
                            seq_len,
                            params.hidden,
                        )?;
                    }
                    emit_fullpath_layer_trace_at(
                        gemv,
                        trace_tag,
                        layer_idx,
                        &trace_buf,
                        hidden_offset_bytes,
                        seq_len,
                        params.hidden,
                    )?;
                }
            }
        }
    }

    // Flush any remaining open command buffer.
    if cmd_in_progress {
        submit_layer_loop_chunk(gemv, counters, attention_chunks, profile)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (no Vulkan device required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer_gemv::gdn_fullpath_gemv_dispatches;

    #[test]
    fn gdn_gemv_budget_includes_quantized_alpha_and_beta() {
        assert_eq!(gdn_fullpath_gemv_dispatches(32, false, false), Some(192));
        assert_eq!(gdn_fullpath_gemv_dispatches(32, true, false), Some(224));
        assert_eq!(gdn_fullpath_gemv_dispatches(32, false, true), Some(224));
        assert_eq!(gdn_fullpath_gemv_dispatches(32, true, true), Some(256));
    }

    #[test]
    fn qwen35_attention_prefill_derives_descriptor_safe_chunks() {
        let whole = AttentionLayerSetsConsumed::attention_fullpath(
            42, 16, 4, true, true, false, false, false,
        )
        .unwrap();
        assert_eq!(whole.norm, 924);
        assert!(whole.max_per_pipeline() > LEGACY_DESCRIPTOR_WINDOW);

        let legacy_chunk =
            largest_safe_descriptor_chunk_len(42, LEGACY_DESCRIPTOR_WINDOW, |seq_len| {
                Ok(AttentionLayerSetsConsumed::attention_fullpath(
                    seq_len, 16, 4, true, true, false, false, false,
                )
                .unwrap()
                .max_per_pipeline()
                    <= LEGACY_DESCRIPTOR_WINDOW)
            })
            .unwrap();
        assert_eq!(legacy_chunk, 23);

        let expanded_limit = 4096;
        let whole_chunk = largest_safe_descriptor_chunk_len(42, expanded_limit, |seq_len| {
            Ok(AttentionLayerSetsConsumed::attention_fullpath(
                seq_len, 16, 4, true, true, false, false, false,
            )
            .unwrap()
            .max_per_pipeline()
                <= expanded_limit)
        })
        .unwrap();
        assert_eq!(whole_chunk, 42);

        let long_chunk = largest_safe_descriptor_chunk_len(1140, expanded_limit, |seq_len| {
            Ok(AttentionLayerSetsConsumed::attention_fullpath(
                seq_len, 16, 4, true, true, false, false, false,
            )
            .unwrap()
            .max_per_pipeline()
                <= expanded_limit)
        })
        .unwrap();
        assert_eq!(long_chunk, 186);
    }

    #[test]
    fn set_idx_counters_would_overflow_detects_norm_overflow() {
        let mut s = SetIdxCounters::default();
        s.norm = MAX_BATCH_OUTPUTS - 1;
        let consumed = AttentionLayerSetsConsumed {
            norm: 2, // (MAX_BATCH_OUTPUTS - 1) + 2 > MAX_BATCH_OUTPUTS
            q: 0,
            k: 0,
            v: 0,
            attn: 0,
            o: 0,
            gate: 0,
            up: 0,
            down: 0,
            silu: 0,
            add: 0,
            bias: 0,
            rope_q: 0,
            rope_k: 0,
        };
        assert!(
            s.would_overflow(&consumed, MAX_BATCH_OUTPUTS),
            "should detect norm overflow"
        );
    }

    #[test]
    fn set_idx_counters_advance_and_reset() {
        let mut s = SetIdxCounters::default();
        let consumed = AttentionLayerSetsConsumed {
            norm: 4,
            q: 4,
            k: 8,
            v: 8,
            attn: 16,
            o: 4,
            gate: 4,
            up: 4,
            down: 4,
            silu: 4,
            add: 8,
            bias: 5,
            rope_q: 1,
            rope_k: 2,
        };
        s.advance(&consumed);
        assert_eq!(s.norm, 4);
        assert_eq!(s.rope, 3); // rope_q + rope_k = 1 + 2
        assert_eq!(s.bias, 5);
        s.reset();
        assert_eq!(s.norm, 0);
        assert_eq!(s.rope, 0);
    }

    #[test]
    fn set_idx_counters_no_overflow_within_budget() {
        let mut s = SetIdxCounters::default();
        let consumed = AttentionLayerSetsConsumed {
            norm: 4,
            q: 4,
            k: 4,
            v: 4,
            attn: 4,
            o: 4,
            gate: 4,
            up: 4,
            down: 4,
            silu: 4,
            add: 8,
            bias: 0,
            rope_q: 1,
            rope_k: 2,
        };
        // Max consumed per pipeline per layer = 8 (add slots), so the cap is
        // `MAX_BATCH_OUTPUTS / 8` layers without overflow.
        let cap_layers = MAX_BATCH_OUTPUTS / 8;
        for _ in 0..(cap_layers - 1) {
            assert!(!s.would_overflow(&consumed, MAX_BATCH_OUTPUTS));
            s.advance(&consumed);
        }
        // Final boundary layer fits exactly: (cap-1)*8 + 8 == MAX_BATCH_OUTPUTS.
        assert!(!s.would_overflow(&consumed, MAX_BATCH_OUTPUTS));
        s.advance(&consumed);
        // One more layer would overflow: cap*8 + 8 > MAX_BATCH_OUTPUTS.
        assert!(s.would_overflow(&consumed, MAX_BATCH_OUTPUTS));
    }

    // -----------------------------------------------------------------------
    // FullPathDecodeStepInput validation tests (Task 9)
    // These run without a Vulkan device — they only exercise the validation
    // guard clauses at the top of run_decode_step.
    // -----------------------------------------------------------------------

    /// Builds a minimal FullPathDecodeStepInput with valid-looking fields so
    /// that we can poke one field at a time to trigger validation errors.
    fn make_valid_decode_input<'a>(
        embed: &'a [u8],
        output: &'a [u8],
        layer_kinds: &'a [ModelLayerKind],
        kv_layout: KvResidentLayout,
    ) -> FullPathDecodeStepInput<'a> {
        static OUTPUT_NORM: [f32; 256] = [1.0; 256];
        FullPathDecodeStepInput {
            token_id: 1,
            kv_cursor: 5,
            num_layers: 2,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            ffn_inner: 512,
            norm_eps: 1e-5,
            base_freq: 500_000.0,
            rope_dim: 64,
            rope_neox: false,
            vocab: 32000,
            kv_layout,
            staging: StagingPolicy::default(),
            output_table_q6k: output,
            output_quant: crate::weight_cache::QuantType::Q6K,
            output_norm: &OUTPUT_NORM,
            embed_table_q6k: embed,
            embed_quant: crate::weight_cache::QuantType::Q6K,
            layer_kinds,
            layer_weights: None, // smoke-test mode: skips layer loop
        }
    }

    #[test]
    fn decode_step_input_validation_hidden_zero() {
        // hidden=0 should be rejected before any GPU call.
        let embed = [];
        let output = [];
        let kv_layout = KvResidentLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 64,
            max_ctx: 512,
        };
        let mut input = make_valid_decode_input(&embed, &output, &[], kv_layout);
        input.hidden = 0;

        let err = validate_decode_step_input(&input);
        assert!(err.is_err(), "hidden=0 should fail validation");
        assert!(
            err.unwrap_err().contains("hidden"),
            "error should mention 'hidden'"
        );
    }

    #[test]
    fn decode_step_input_validation_hidden_not_multiple_of_256() {
        let embed = [];
        let output = [];
        let kv_layout = KvResidentLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 64,
            max_ctx: 512,
        };
        let mut input = make_valid_decode_input(&embed, &output, &[], kv_layout);
        input.hidden = 128; // valid size but not multiple of 256

        let err = validate_decode_step_input(&input);
        assert!(err.is_err(), "hidden=128 should fail validation");
    }

    #[test]
    fn decode_step_input_validation_output_norm_len_mismatch() {
        let embed = [];
        let output = [];
        let output_norm = [1.0f32; 255];
        let kv_layout = KvResidentLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 64,
            max_ctx: 512,
        };
        let input = FullPathDecodeStepInput {
            token_id: 1,
            kv_cursor: 5,
            num_layers: 2,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            ffn_inner: 512,
            norm_eps: 1e-5,
            base_freq: 500_000.0,
            rope_dim: 64,
            rope_neox: false,
            vocab: 32000,
            kv_layout,
            staging: StagingPolicy::default(),
            output_table_q6k: &output,
            output_quant: crate::weight_cache::QuantType::Q6K,
            output_norm: &output_norm,
            embed_table_q6k: &embed,
            embed_quant: crate::weight_cache::QuantType::Q6K,
            layer_kinds: &[],
            layer_weights: None,
        };

        let err = validate_decode_step_input(&input);
        assert!(
            err.is_err(),
            "output_norm len mismatch should fail validation"
        );
        assert!(
            err.unwrap_err().contains("output_norm"),
            "error should mention output_norm"
        );
    }

    #[test]
    fn decode_step_input_validation_layer_weights_len_mismatch() {
        // layer_weights.len() != num_layers should be caught.
        let embed = [];
        let output = [];
        let kv_layout = KvResidentLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 64,
            max_ctx: 512,
        };
        // Build a &[LayerHandle] of length 1, but num_layers = 2.
        let empty_bufs: &[&GpuBuffer] = &[];
        let dummy_buf = GpuBuffer {
            buffer: 0, // VkBuffer = u64, VK_NULL_HANDLE = 0
            memory: 0, // VkDeviceMemory = u64, VK_NULL_HANDLE = 0
            size: 0,
        };
        let wh = LayerWeightHandles {
            attn_norm_buf: &dummy_buf,
            attn_norm_size: 0,
            q_weight_buf: &dummy_buf,
            q_weight_size: 0,
            q_rows: 256,
            q_cols: 256,
            q_quant: crate::weight_cache::QuantType::Q4K,
            head_dim: 64,
            q_bias_buf: None,
            q_bias_size: 0,
            q_norm_buf: None,
            q_norm_size: 0,
            k_weight_bufs: empty_bufs,
            k_weight_size: 0,
            k_quant: crate::weight_cache::QuantType::Q4K,
            k_bias_buf: None,
            k_bias_size: 0,
            k_norm_buf: None,
            k_norm_size: 0,
            v_weight_bufs: empty_bufs,
            v_weight_size: 0,
            v_quant: crate::weight_cache::QuantType::Q4K,
            v_bias_buf: None,
            v_bias_size: 0,
            o_weight_buf: &dummy_buf,
            o_weight_size: 0,
            o_quant: crate::weight_cache::QuantType::Q4K,
            ffn_norm_buf: &dummy_buf,
            ffn_norm_size: 0,
            gate_weight_buf: &dummy_buf,
            gate_weight_size: 0,
            gate_quant: crate::weight_cache::QuantType::Q4K,
            up_weight_buf: &dummy_buf,
            up_weight_size: 0,
            up_quant: crate::weight_cache::QuantType::Q4K,
            down_weight_buf: &dummy_buf,
            down_weight_size: 0,
            down_quant: crate::weight_cache::QuantType::Q4K,
        };
        let handles = [LayerHandle::Attention(wh)]; // length 1, but num_layers = 2
        let output_norm = [1.0f32; 256];

        let input = FullPathDecodeStepInput {
            token_id: 1,
            kv_cursor: 0,
            num_layers: 2, // mismatch with handles.len() = 1
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            ffn_inner: 512,
            norm_eps: 1e-5,
            base_freq: 500_000.0,
            rope_dim: 64,
            rope_neox: false,
            vocab: 32000,
            kv_layout,
            staging: StagingPolicy::default(),
            output_table_q6k: &output,
            output_quant: crate::weight_cache::QuantType::Q6K,
            output_norm: &output_norm,
            embed_table_q6k: &embed,
            embed_quant: crate::weight_cache::QuantType::Q6K,
            layer_kinds: &[],
            layer_weights: Some(&handles),
        };

        let err = validate_decode_step_input(&input);
        assert!(err.is_err(), "len-mismatch should fail validation");
        assert!(
            err.unwrap_err().contains("layer_weights.len()"),
            "error should name the mismatch"
        );
    }

    /// mv28 task 10b-5a smoke: `GdnLayerWeightHandles<'a>` constructs from
    /// dummy `GpuBuffer` references. Confirms the field set / lifetimes /
    /// quant tags compile. No consumer yet (5d wires `run_layer_loop`'s
    /// Recurrent branch); this just locks the struct shape so 5b/5c can
    /// build against it without surprises.
    #[test]
    fn gdn_layer_weight_handles_constructs_from_dummy_buffers() {
        let dummy_buf = GpuBuffer {
            buffer: 0, // VK_NULL_HANDLE
            memory: 0, // VK_NULL_HANDLE
            size: 0,
        };
        let q = crate::weight_cache::QuantType::Q4K;
        let handles = GdnLayerWeightHandles {
            attn_norm_buf: &dummy_buf,
            attn_norm_size: 0,
            qkv_weight_buf: &dummy_buf,
            qkv_weight_size: 0,
            qkv_rows: 768,
            qkv_cols: 256,
            qkv_quant: q,
            gate_weight_buf: &dummy_buf,
            gate_weight_size: 0,
            gate_rows: 512,
            gate_cols: 256,
            gate_quant: q,
            ssm_alpha_buf: &dummy_buf,
            ssm_alpha_size: 0,
            ssm_alpha_rows: 1,
            ssm_alpha_cols: 256,
            ssm_alpha_quant: None,
            ssm_beta_buf: &dummy_buf,
            ssm_beta_size: 0,
            ssm_beta_rows: 1,
            ssm_beta_cols: 256,
            ssm_beta_quant: None,
            ssm_a_buf: &dummy_buf,
            ssm_a_size: 0,
            ssm_conv1d_buf: &dummy_buf,
            ssm_conv1d_size: 0,
            ssm_dt_bias_buf: &dummy_buf,
            ssm_dt_bias_size: 0,
            ssm_norm_buf: &dummy_buf,
            ssm_norm_size: 0,
            num_k_heads: 1,
            head_k_dim: 128,
            ssm_out_buf: &dummy_buf,
            ssm_out_size: 0,
            ssm_out_rows: 256,
            ssm_out_cols: 512,
            ssm_out_quant: q,
            post_attn_norm_buf: &dummy_buf,
            post_attn_norm_size: 0,
            ffn_gate_weight_buf: &dummy_buf,
            ffn_gate_weight_size: 0,
            ffn_gate_rows: 1024,
            ffn_gate_cols: 256,
            ffn_gate_quant: q,
            ffn_up_weight_buf: &dummy_buf,
            ffn_up_weight_size: 0,
            ffn_up_rows: 1024,
            ffn_up_cols: 256,
            ffn_up_quant: q,
            ffn_down_weight_buf: &dummy_buf,
            ffn_down_weight_size: 0,
            ffn_down_rows: 256,
            ffn_down_cols: 1024,
            ffn_down_quant: q,
        };
        // Sanity-check that f32-raw and quantized field groups came through.
        // (raw32: norms + ssm_alpha/beta + ssm_a/conv1d/dt_bias/norm)
        // (quant: qkv + gate + ssm_out + ffn_{gate,up,down})
        assert_eq!(handles.attn_norm_size, 0);
        assert_eq!(handles.post_attn_norm_size, 0);
        assert_eq!(handles.ssm_alpha_rows, 1);
        assert_eq!(handles.ssm_alpha_cols, 256);
        assert_eq!(handles.qkv_rows, 768);
        assert_eq!(handles.gate_rows, 512);
        assert_eq!(handles.ssm_out_cols, 512);
        assert_eq!(handles.ffn_down_cols, 1024);
        assert_eq!(handles.ssm_a_size, 0);
        assert_eq!(handles.ssm_conv1d_size, 0);
        assert_eq!(handles.ssm_dt_bias_size, 0);
        assert_eq!(handles.ssm_norm_size, 0);
        assert!(matches!(
            handles.qkv_quant,
            crate::weight_cache::QuantType::Q4K
        ));
        assert!(matches!(
            handles.ffn_down_quant,
            crate::weight_cache::QuantType::Q4K
        ));
    }

    #[test]
    fn prefill_input_accepts_mixed_attention_and_gdn_handles() {
        let dummy_buf = GpuBuffer {
            buffer: 0,
            memory: 0,
            size: 0,
        };
        let kv_bufs = [&dummy_buf];
        let q = crate::weight_cache::QuantType::Q4K;
        let attention = LayerWeightHandles {
            attn_norm_buf: &dummy_buf,
            attn_norm_size: 0,
            q_weight_buf: &dummy_buf,
            q_weight_size: 0,
            q_rows: 256,
            q_cols: 256,
            q_quant: q,
            head_dim: 64,
            q_bias_buf: None,
            q_bias_size: 0,
            q_norm_buf: None,
            q_norm_size: 0,
            k_weight_bufs: &kv_bufs,
            k_weight_size: 0,
            k_quant: q,
            k_bias_buf: None,
            k_bias_size: 0,
            k_norm_buf: None,
            k_norm_size: 0,
            v_weight_bufs: &kv_bufs,
            v_weight_size: 0,
            v_quant: q,
            v_bias_buf: None,
            v_bias_size: 0,
            o_weight_buf: &dummy_buf,
            o_weight_size: 0,
            o_quant: q,
            ffn_norm_buf: &dummy_buf,
            ffn_norm_size: 0,
            gate_weight_buf: &dummy_buf,
            gate_weight_size: 0,
            gate_quant: q,
            up_weight_buf: &dummy_buf,
            up_weight_size: 0,
            up_quant: q,
            down_weight_buf: &dummy_buf,
            down_weight_size: 0,
            down_quant: q,
        };
        let gdn = GdnLayerWeightHandles {
            attn_norm_buf: &dummy_buf,
            attn_norm_size: 0,
            qkv_weight_buf: &dummy_buf,
            qkv_weight_size: 0,
            qkv_rows: 768,
            qkv_cols: 256,
            qkv_quant: q,
            gate_weight_buf: &dummy_buf,
            gate_weight_size: 0,
            gate_rows: 512,
            gate_cols: 256,
            gate_quant: q,
            ssm_alpha_buf: &dummy_buf,
            ssm_alpha_size: 0,
            ssm_alpha_rows: 1,
            ssm_alpha_cols: 256,
            ssm_alpha_quant: None,
            ssm_beta_buf: &dummy_buf,
            ssm_beta_size: 0,
            ssm_beta_rows: 1,
            ssm_beta_cols: 256,
            ssm_beta_quant: None,
            ssm_a_buf: &dummy_buf,
            ssm_a_size: 0,
            ssm_conv1d_buf: &dummy_buf,
            ssm_conv1d_size: 0,
            ssm_dt_bias_buf: &dummy_buf,
            ssm_dt_bias_size: 0,
            ssm_norm_buf: &dummy_buf,
            ssm_norm_size: 0,
            num_k_heads: 1,
            head_k_dim: 128,
            ssm_out_buf: &dummy_buf,
            ssm_out_size: 0,
            ssm_out_rows: 256,
            ssm_out_cols: 512,
            ssm_out_quant: q,
            post_attn_norm_buf: &dummy_buf,
            post_attn_norm_size: 0,
            ffn_gate_weight_buf: &dummy_buf,
            ffn_gate_weight_size: 0,
            ffn_gate_rows: 1024,
            ffn_gate_cols: 256,
            ffn_gate_quant: q,
            ffn_up_weight_buf: &dummy_buf,
            ffn_up_weight_size: 0,
            ffn_up_rows: 1024,
            ffn_up_cols: 256,
            ffn_up_quant: q,
            ffn_down_weight_buf: &dummy_buf,
            ffn_down_weight_size: 0,
            ffn_down_rows: 256,
            ffn_down_cols: 1024,
            ffn_down_quant: q,
        };
        let handles = [LayerHandle::Attention(attention), LayerHandle::Gdn(gdn)];
        let layer_kinds = [ModelLayerKind::Attention, ModelLayerKind::Recurrent];
        let output_norm = [1.0f32; 256];
        let input = FullPathPrefillInput {
            prompt_token_ids: &[1, 2],
            num_layers: 2,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 64,
            ffn_inner: 512,
            norm_eps: 1e-5,
            base_freq: 500_000.0,
            rope_dim: 64,
            rope_neox: false,
            vocab: 32000,
            kv_layout: KvResidentLayout::compute(2, 512, 1, 64),
            staging: StagingPolicy::default(),
            output_table_q6k: &[],
            output_quant: crate::weight_cache::QuantType::Q6K,
            output_norm: &output_norm,
            embed_table_q6k: &[],
            embed_quant: crate::weight_cache::QuantType::Q6K,
            layer_weights: Some(&handles),
            layer_kinds: &layer_kinds,
        };

        assert!(input.layer_weights.is_some());
        assert_eq!(input.layer_weights.unwrap().len(), 2);
    }

    #[test]
    fn final_hidden_ping_alternates_by_layer_count() {
        assert_eq!(final_hidden_ping_after_layers(1), HiddenPing::B);
        assert_eq!(final_hidden_ping_after_layers(2), HiddenPing::A);
        assert_eq!(final_hidden_ping_after_layers(3), HiddenPing::B);
        assert_eq!(final_hidden_ping_after_layers(24), HiddenPing::A);
    }

    #[test]
    fn runtime_counters_allow_token_only_download_in_fullpath() {
        let mut counters = RuntimeCounters::default();
        counters.download_bytes += 4;

        assert_eq!(counters.host_tensor_roundtrip_bytes, 0);
        assert!(!counters.has_forbidden_fullpath_cpu_escape());
    }

    #[test]
    fn runtime_counters_mark_hidden_roundtrip_as_fullpath_cpu_escape() {
        let mut counters = RuntimeCounters::default();
        counters.record_host_tensor_roundtrip(1024, 512);

        assert_eq!(counters.host_tensor_roundtrip_bytes, 1536);
        assert!(counters.has_forbidden_fullpath_cpu_escape());
    }

    #[test]
    fn fullpath_cpu_escape_guard_rejects_tensor_roundtrip() {
        let mut counters = RuntimeCounters::default();
        counters.record_host_tensor_roundtrip(256, 256);

        let err = ensure_no_forbidden_fullpath_cpu_escape("test_guard", &counters)
            .expect_err("host tensor roundtrip must fail production fullpath");

        assert!(err.contains("test_guard"));
        assert!(err.contains("host_tensor_roundtrip_bytes=512"));
    }

    #[test]
    fn fullpath_profile_summary_includes_stage_times_and_cpu_escape_counter() {
        let mut profile = FullpathProfile::enabled_for_test("decode");
        profile.staging_us = 1_000;
        profile.embed_us = 2_000;
        profile.layer_loop_us = 3_000;
        profile.output_us = 4_000;
        profile.attention_record_us = 5_000;
        profile.gdn_record_us = 6_000;
        profile.submit_wait_us = 7_000;
        profile.total_us = 10_000;
        let mut counters = RuntimeCounters::default();
        counters.record_host_tensor_roundtrip(128, 128);

        let line = profile.summary_line(&counters, 2, 3);

        assert!(line.contains("kind=decode"));
        assert!(line.contains("staging_ms=1.000"));
        assert!(line.contains("embed_ms=2.000"));
        assert!(line.contains("layer_loop_ms=3.000"));
        assert!(line.contains("output_ms=4.000"));
        assert!(line.contains("attention_record_ms=5.000"));
        assert!(line.contains("gdn_record_ms=6.000"));
        assert!(line.contains("submit_wait_ms=7.000"));
        assert!(line.contains("host_tensor_roundtrip_bytes=256"));
        assert!(line.contains("attention_chunks=2"));
        assert!(line.contains("gdn_layers=3"));
    }

    #[test]
    fn fullpath_counters_log_is_profile_or_env_gated() {
        std::env::remove_var("RNB_VULKAN_FULLPATH_COUNTERS");
        let disabled = FullpathProfile::default();
        assert!(!fullpath_counters_log_enabled(&disabled));

        let profile_enabled = FullpathProfile::enabled_for_test("decode");
        assert!(fullpath_counters_log_enabled(&profile_enabled));

        std::env::set_var("RNB_VULKAN_FULLPATH_COUNTERS", "1");
        assert!(fullpath_counters_log_enabled(&disabled));
        std::env::remove_var("RNB_VULKAN_FULLPATH_COUNTERS");
    }

    fn qwen_like_gdn_handles<'a>(
        buf: &'a GpuBuffer,
        q: crate::weight_cache::QuantType,
    ) -> GdnLayerWeightHandles<'a> {
        GdnLayerWeightHandles {
            attn_norm_buf: buf,
            attn_norm_size: 1024 * 4,
            qkv_weight_buf: buf,
            qkv_weight_size: 0,
            qkv_rows: 3072,
            qkv_cols: 1024,
            qkv_quant: q,
            gate_weight_buf: buf,
            gate_weight_size: 0,
            gate_rows: 2048,
            gate_cols: 1024,
            gate_quant: q,
            ssm_alpha_buf: buf,
            ssm_alpha_size: 8 * 1024 * 4,
            ssm_alpha_rows: 8,
            ssm_alpha_cols: 1024,
            ssm_alpha_quant: Some(crate::weight_cache::QuantType::Q8_0),
            ssm_beta_buf: buf,
            ssm_beta_size: 8 * 1024 * 4,
            ssm_beta_rows: 8,
            ssm_beta_cols: 1024,
            ssm_beta_quant: Some(crate::weight_cache::QuantType::Q8_0),
            ssm_a_buf: buf,
            ssm_a_size: 8 * 4,
            ssm_conv1d_buf: buf,
            ssm_conv1d_size: 4 * 3072 * 4,
            ssm_dt_bias_buf: buf,
            ssm_dt_bias_size: 8 * 4,
            ssm_norm_buf: buf,
            ssm_norm_size: 256 * 4,
            num_k_heads: 4,
            head_k_dim: 128,
            ssm_out_buf: buf,
            ssm_out_size: 0,
            ssm_out_rows: 1024,
            ssm_out_cols: 2048,
            ssm_out_quant: q,
            post_attn_norm_buf: buf,
            post_attn_norm_size: 1024 * 4,
            ffn_gate_weight_buf: buf,
            ffn_gate_weight_size: 0,
            ffn_gate_rows: 3584,
            ffn_gate_cols: 1024,
            ffn_gate_quant: q,
            ffn_up_weight_buf: buf,
            ffn_up_weight_size: 0,
            ffn_up_rows: 3584,
            ffn_up_cols: 1024,
            ffn_up_quant: q,
            ffn_down_weight_buf: buf,
            ffn_down_weight_size: 0,
            ffn_down_rows: 1024,
            ffn_down_cols: 3584,
            ffn_down_quant: q,
        }
    }

    #[test]
    fn gdn_fullpath_dims_derive_from_weight_shapes() {
        let dummy_buf = GpuBuffer {
            buffer: 0,
            memory: 0,
            size: 0,
        };
        let handles = qwen_like_gdn_handles(&dummy_buf, crate::weight_cache::QuantType::Q4K);

        let dims = derive_gdn_fullpath_dims(&handles, 1024).unwrap();

        assert_eq!(dims.hidden, 1024);
        assert_eq!(dims.conv_channels, 3072);
        assert_eq!(dims.d_inner, 2048);
        assert_eq!(dims.num_v_heads, 8);
        assert_eq!(dims.num_k_heads, 4);
        assert_eq!(dims.head_v_dim, 256);
        assert_eq!(dims.head_k_dim, 128);
        assert_eq!(dims.conv_kernel, 4);
    }

    #[test]
    fn gdn_fullpath_dims_use_gdn_heads_not_attention_kv_heads() {
        let dummy_buf = GpuBuffer {
            buffer: 0,
            memory: 0,
            size: 0,
        };
        let mut handles = qwen_like_gdn_handles(&dummy_buf, crate::weight_cache::QuantType::Q4K);
        handles.qkv_rows = 6144;
        handles.gate_rows = 2048;
        handles.ssm_alpha_size = 16 * 1024 * 4;
        handles.ssm_alpha_rows = 16;
        handles.ssm_alpha_cols = 1024;
        handles.ssm_beta_size = 16 * 1024 * 4;
        handles.ssm_beta_rows = 16;
        handles.ssm_beta_cols = 1024;
        handles.ssm_a_size = 16 * 4;
        handles.ssm_conv1d_size = 4 * 6144 * 4;
        handles.ssm_dt_bias_size = 16 * 4;
        handles.ssm_norm_size = 128 * 4;
        handles.num_k_heads = 16;
        handles.head_k_dim = 128;
        handles.ssm_out_cols = 2048;

        let dims = derive_gdn_fullpath_dims(&handles, 1024).unwrap();

        assert_eq!(dims.num_v_heads, 16);
        assert_eq!(dims.head_v_dim, 128);
        assert_eq!(dims.num_k_heads, 16);
        assert_eq!(dims.head_k_dim, 128);
    }

    #[test]
    fn fullpath_staging_inner_includes_gdn_conv_channels() {
        let dummy_buf = GpuBuffer {
            buffer: 0,
            memory: 0,
            size: 0,
        };
        let mut gdn = qwen_like_gdn_handles(&dummy_buf, crate::weight_cache::QuantType::Q4K);
        gdn.qkv_rows = 6144;
        gdn.ssm_conv1d_size = 4 * 6144 * 4;
        gdn.num_k_heads = 16;
        let handles = [LayerHandle::Gdn(gdn)];

        let staging_inner = max_fullpath_staging_inner(&handles, 1024, 4, 3584).unwrap();

        assert_eq!(staging_inner, 6144);
    }

    #[test]
    fn gdn_fullpath_dims_reject_qkv_hidden_mismatch() {
        let dummy_buf = GpuBuffer {
            buffer: 0,
            memory: 0,
            size: 0,
        };
        let mut handles = qwen_like_gdn_handles(&dummy_buf, crate::weight_cache::QuantType::Q4K);
        handles.qkv_cols = 2048;

        let err = derive_gdn_fullpath_dims(&handles, 1024).unwrap_err();

        assert!(err.contains("qkv_cols"));
    }
}
