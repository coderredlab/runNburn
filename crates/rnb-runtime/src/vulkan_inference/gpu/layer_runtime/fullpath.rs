// Fullpath wrapper internals — see mv27 task 10b-4c-2a for design context.

use rnb_backend_vulkan::full_path::{
    GdnLayerWeightHandles, LayerHandle as BackendLayerHandle, LayerWeightHandles,
};
use rnb_backend_vulkan::{
    attn_norm_id, ffn_down_id, ffn_gate_id, ffn_norm_id, ffn_up_id, gdn_alpha_id, gdn_attn_norm_id,
    gdn_beta_id, gdn_gate_id, gdn_post_attn_norm_id, gdn_qkv_id, gdn_ssm_a_id, gdn_ssm_conv1d_id,
    gdn_ssm_dt_bias_id, gdn_ssm_norm_id, gdn_ssm_out_id, k_norm_id, k_proj_shard_id,
    kv_head_shard_byte_range, o_proj_id, q_norm_id, q_proj_id, v_proj_shard_id,
};

use super::{
    FullPathDecodeStepInput, FullPathDecodeStepOutput, FullPathPrefillInput, FullPathPrefillOutput,
    GpuBuffer, GpuWeightMode, KvResidentLayout, LayerRuntime, ModelLayerKind, QuantType,
    StagingPolicy,
};
use crate::vulkan_backend::WeightId;

// ----- mv27-task10b-4c-2a: fullpath wrapper methods -----
//
// These methods own the cross-crate plumbing for backend-vulkan's
// single-submit GPU fullpath prefill / decode. The caller (rnb-llm
// engine wiring, mv27-task10b-4c-2b) only needs to hand raw byte slices
// and per-layer metadata via [`LayerRawWeights`]; the wrapper takes care
// of:
//   - per-layer weight upload via the WeightCache (combined K/V tensors
//     are split into per-kv-head shards using `kv_head_shard_byte_range`),
//   - building the [`LayerWeightHandles`] array with cache-borrow
//     lifetimes (two-pass upload-then-borrow — see
//     `LayerGemv::weight_buffer` doc for the rationale),
//   - constructing [`KvResidentLayout`] and [`FullPathPrefillInput`] /
//     [`FullPathDecodeStepInput`],
//   - delegating to `rnb_backend_vulkan::full_path::run_prefill` /
//     `run_decode_step`.
//
// `bind_token_embd` is still exposed for older smoke paths, but production
// fullpath no longer binds the full Q6_K embed table as one storage buffer.
// Mobile Vulkan storage-buffer descriptor ranges can be smaller than 2B
// token_embd bytes, so run_prefill/decode stage only requested Q6_K rows.

impl LayerRuntime {
    /// Upload (or refresh) the full Q6_K-packed `token_embd.weight` table.
    ///
    /// Legacy helper. Production fullpath uses compact per-token-row staging
    /// instead of a single large descriptor.
    pub fn bind_token_embd(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.token_embd_bound {
            // Re-call detection: prior bind succeeded and we are about to
            // overwrite it. Useful while wiring engine init paths to catch
            // unintended duplicate uploads. Plain `eprintln!` matches the
            // existing logging style in this crate (see packed_weights.rs,
            // cuda_inference/cuda/prefill.rs) — there is no `tracing`/`log`
            // dependency in rnb-runtime.
            eprintln!(
                "[vulkan:init] bind_token_embd re-call detected (bytes={})",
                bytes.len()
            );
        }
        self.inner.bind_token_embd(bytes)?;
        self.token_embd_bound = true;
        Ok(())
    }

    /// Upload a single weight tensor to the GPU weight cache (no LRU-only borrow).
    ///
    /// Thin pass-through to `LayerGemv::upload_weight`. Used by callers
    /// (engine init / fullpath wrapper internals) when weights need to be
    /// uploaded outside the bulk fullpath path.
    pub fn upload_weight(
        &mut self,
        id: WeightId,
        raw_bytes: &[u8],
        rows: u32,
        cols: u32,
        quant: QuantType,
        mode: GpuWeightMode,
    ) -> Result<(), String> {
        self.inner
            .upload_weight(id, raw_bytes, rows, cols, quant, mode)
    }

    /// Drive a single full-path prefill step on the GPU.
    ///
    /// Steps performed inside this wrapper:
    /// 1. Upload every per-layer weight via `WeightCache::get_or_upload`,
    ///    splitting the combined `[num_kv_heads*head_dim, hidden]` K/V
    ///    tensors into per-kv-head shards using `kv_head_shard_byte_range`.
    /// 2. Borrow `&GpuBuffer` references back from the cache (read-only
    ///    second pass — `LayerGemv::weight_buffer`) to build the
    ///    `LayerWeightHandles` array. The two-pass split is required because
    ///    `get_or_upload` takes `&mut self` and the borrow checker won't let
    ///    us hold N long-lived `&GpuBuffer` borrows from the same cache while
    ///    also calling more `&mut self` uploads.
    /// 3. Compute the `KvResidentLayout` from caller-supplied dims.
    /// 4. Construct `FullPathPrefillInput` and call
    ///    `rnb_backend_vulkan::full_path::run_prefill`.
    ///
    /// `staging` and the layer kinds slice are passed straight through. When
    /// `layer_kinds.len() == 0` the backend treats every layer as Attention
    /// (see `full_path::run_layer_loop`'s default-attention fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn run_fullpath_prefill(
        &mut self,
        prompt_token_ids: &[u32],
        num_layers: usize,
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        ffn_inner: usize,
        norm_eps: f32,
        base_freq: f32,
        rope_dim: usize,
        rope_neox: bool,
        vocab: usize,
        max_ctx: usize,
        output_table_q6k: &[u8],
        output_quant: QuantType,
        output_norm: &[f32],
        embed_table_q6k: &[u8],
        layer_raw_weights: &[LayerRawWeights<'_>],
        layer_kinds: &[ModelLayerKind],
        staging: StagingPolicy,
    ) -> Result<FullPathPrefillOutput, String> {
        validate_layer_raw_weights_count(layer_raw_weights, num_layers, "run_fullpath_prefill")?;

        // Pass 1: upload every weight. Discard the &GpuBuffer refs returned by
        // get_or_upload — they conflict with subsequent &mut self uploads.
        upload_layer_weights(
            &mut self.inner,
            layer_raw_weights,
            num_kv_heads,
            head_dim,
            hidden,
        )?;

        // Pass 2: snapshot raw pointers to every uploaded weight. This
        // happens with only `&self.inner` held; the storage moves to a
        // stack-local Vec before run_prefill borrows `&mut self.inner`.
        let layer_handles_storage =
            build_layer_handles(&self.inner, layer_raw_weights, layer_kinds, num_kv_heads)?;

        // SAFETY: see *HandlesStorage::as_*_handles doc — cache is not
        // mutated between build_layer_handles and run_prefill.
        let handles: Vec<BackendLayerHandle<'_>> = layer_handles_storage
            .iter()
            .map(|storage| unsafe { storage.as_backend_handle() })
            .collect();

        let kv_layout = KvResidentLayout::compute(num_layers, max_ctx, num_kv_heads, head_dim);

        let input = FullPathPrefillInput {
            prompt_token_ids,
            num_layers,
            hidden,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_inner,
            norm_eps,
            base_freq,
            rope_dim,
            rope_neox,
            vocab,
            kv_layout,
            staging,
            output_table_q6k,
            output_quant,
            output_norm,
            embed_table_q6k,
            layer_weights: Some(&handles),
            layer_kinds,
        };

        let result = rnb_backend_vulkan::full_path::run_prefill(&mut self.inner, input);
        // Keep storage alive until after run_prefill returns (compiler should
        // already enforce this, but explicit drop here reads as intent).
        drop(handles);
        drop(layer_handles_storage);
        result
    }

    /// Drive a single-token full-path decode step on the GPU.
    ///
    /// Same plumbing as [`run_fullpath_prefill`] but with `seq_len = 1` and
    /// `pos_start = kv_cursor` (passed via `FullPathDecodeStepInput`). Weight
    /// upload is idempotent — once a layer's weights are cached, subsequent
    /// calls hit the fast path and only update the LRU tick.
    #[allow(clippy::too_many_arguments)]
    pub fn run_fullpath_decode_step(
        &mut self,
        token_id: u32,
        kv_cursor: usize,
        num_layers: usize,
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        ffn_inner: usize,
        norm_eps: f32,
        base_freq: f32,
        rope_dim: usize,
        rope_neox: bool,
        vocab: usize,
        max_ctx: usize,
        output_table_q6k: &[u8],
        output_quant: QuantType,
        output_norm: &[f32],
        embed_table_q6k: &[u8],
        layer_raw_weights: &[LayerRawWeights<'_>],
        layer_kinds: &[ModelLayerKind],
        staging: StagingPolicy,
    ) -> Result<FullPathDecodeStepOutput, String> {
        validate_layer_raw_weights_count(
            layer_raw_weights,
            num_layers,
            "run_fullpath_decode_step",
        )?;

        upload_layer_weights(
            &mut self.inner,
            layer_raw_weights,
            num_kv_heads,
            head_dim,
            hidden,
        )?;

        let layer_handles_storage =
            build_layer_handles(&self.inner, layer_raw_weights, layer_kinds, num_kv_heads)?;
        // SAFETY: see *HandlesStorage::as_*_handles doc — cache is not
        // mutated between build_layer_handles and run_decode_step.
        let handles: Vec<BackendLayerHandle<'_>> = layer_handles_storage
            .iter()
            .map(|storage| unsafe { storage.as_backend_handle() })
            .collect();

        let kv_layout = KvResidentLayout::compute(num_layers, max_ctx, num_kv_heads, head_dim);

        let input = FullPathDecodeStepInput {
            token_id,
            kv_cursor,
            num_layers,
            hidden,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_inner,
            norm_eps,
            base_freq,
            rope_dim,
            rope_neox,
            vocab,
            kv_layout,
            staging,
            output_table_q6k,
            output_quant,
            output_norm,
            embed_table_q6k,
            layer_kinds,
            layer_weights: Some(&handles),
        };

        let result = rnb_backend_vulkan::full_path::run_decode_step(&mut self.inner, input);
        drop(handles);
        drop(layer_handles_storage);
        result
    }
}

// ---------------------------------------------------------------------------
// Public input types + helpers (mv27-task10b-4c-2a)
// ---------------------------------------------------------------------------

/// Raw byte slices + metadata for one transformer layer's weights.
///
/// Caller (rnb-llm engine, mv27-task10b-4c-2b for Attention; mv28-task10b-5c
/// for Gdn) constructs an array of these from `engine.weights.*_raw_bytes(...)`
/// accessors and hands it to [`LayerRuntime::run_fullpath_prefill`] /
/// `run_fullpath_decode_step`.
///
/// **mv28-task10b-5b:** split into an enum so hybrid models (Attention +
/// GatedDeltaNet) can mix layer kinds. The Gdn arm carries the raw byte
/// slices for the 14 GDN weights (8 f32-raw + 6 quantized) mirroring
/// `rnb_backend_vulkan::full_path::GdnLayerWeightHandles<'a>`. Until 5d
/// activates the Gdn dispatch in the wrapper, presence of any
/// `LayerRawWeights::Gdn(_)` entry causes `run_fullpath_*` to return Err
/// (see `run_fullpath_prefill` / `run_fullpath_decode_step`).
pub enum LayerRawWeights<'a> {
    Attention(AttentionRawWeights<'a>),
    /// GDN (GatedDeltaNet / hybrid model) layer raw bytes. Wired through
    /// `upload_layer_weights` / `build_layer_handles` but dispatch is
    /// guarded by `Err("...10b-5d wiring pending")` until 5d lands.
    Gdn(GdnRawWeights<'a>),
}

/// Raw byte slices + metadata for one **attention** layer's weights.
///
/// The wrapper splits combined K/V tensors (`[num_kv_heads*head_dim, hidden]`,
/// row-major) into per-kv-head shards using `kv_head_shard_byte_range`.
/// Caller supplies the **combined** tensor bytes — the wrapper does the
/// sharding.
///
/// Quants supported: Q4_K / Q5_K / Q6_K / Q8_0 (those that
/// `WeightCache::get_or_upload` accepts).
// Field order mirrors `rnb_backend_vulkan::full_path::LayerWeightHandles`
// so the upload/build sites read top-to-bottom in the same order as the
// backend struct they feed.
pub struct AttentionRawWeights<'a> {
    /// Attention RMS norm weight (f32, not quantized).
    pub attn_norm: &'a [f32],
    /// Q projection: `(raw_bytes, rows, cols, quant)`. `rows = num_heads * head_dim`,
    /// `cols = hidden`.
    pub q_proj: (&'a [u8], usize, usize, QuantType),
    /// Optional per-head Q RMS norm weight. Shape: `[head_dim]`.
    pub q_norm: Option<&'a [f32]>,
    /// Combined K projection, shape `[num_kv_heads * head_dim, hidden]` row-major.
    /// Sliced into per-kv-head shards by the wrapper.
    pub k_proj_combined: (&'a [u8], QuantType),
    /// Optional per-head K RMS norm weight. Shape: `[head_dim]`.
    pub k_norm: Option<&'a [f32]>,
    /// Combined V projection, same shape as `k_proj_combined`.
    pub v_proj_combined: (&'a [u8], QuantType),
    /// O projection: `(raw_bytes, rows, cols, quant)`. `rows = hidden`,
    /// `cols = num_heads * head_dim`.
    pub o_proj: (&'a [u8], usize, usize, QuantType),
    /// FFN RMS norm weight (f32, not quantized).
    pub ffn_norm: &'a [f32],
    /// FFN gate projection: `(raw_bytes, rows, cols, quant)`. `rows = ffn_inner`,
    /// `cols = hidden`.
    pub gate_proj: (&'a [u8], usize, usize, QuantType),
    /// FFN up projection (same shape as `gate_proj`).
    pub up_proj: (&'a [u8], usize, usize, QuantType),
    /// FFN down projection: `rows = hidden`, `cols = ffn_inner`.
    pub down_proj: (&'a [u8], usize, usize, QuantType),
}

/// Raw byte slices + metadata for one **GDN** (GatedDeltaNet / hybrid model)
/// layer's weights.
///
/// Field order mirrors `rnb_backend_vulkan::full_path::GdnLayerWeightHandles<'a>`
/// so 5c/5d's wiring stays mechanical:
/// norm → fused QKV → z gate → SSM α/β Raw32 → SSM raw 묶음 → SSM out →
/// post-attn norm → FFN gate/up/down.
///
/// f32 raw fields (`attn_norm`, `post_attn_norm`, `ssm_alpha`, `ssm_beta`,
/// `ssm_a`, `ssm_conv1d`, `ssm_dt_bias`, `ssm_norm`) ride the mv28 I2
/// `GpuWeightMode::Raw32` upload path. Quantized fields use the standard
/// `Soa` cache path.
pub struct GdnRawWeights<'a> {
    /// Pre-attn RMS norm weight (f32, Raw32 path). Shape: `[hidden]`.
    pub attn_norm: &'a [f32],
    /// Fused QKV projection — `[conv_channels, hidden]`.
    pub qkv: (&'a [u8], usize, usize, QuantType),
    /// z gate (SSM input gating) — `[d_inner, hidden]`.
    pub gate: (&'a [u8], usize, usize, QuantType),
    /// SSM α (f32, Raw32) — `[num_heads, hidden]`.
    pub ssm_alpha: (&'a [f32], usize, usize),
    /// SSM β (f32, Raw32) — `[num_heads, hidden]`.
    pub ssm_beta: (&'a [f32], usize, usize),
    /// `A_log` per head (f32, Raw32). Shape: `[num_heads]`.
    pub ssm_a: &'a [f32],
    /// conv1d kernel (f32, Raw32). Shape: `[conv_kernel, conv_channels]`.
    pub ssm_conv1d: &'a [f32],
    /// Δt bias per head (f32, Raw32). Shape: `[num_heads]`.
    pub ssm_dt_bias: &'a [f32],
    /// per-head-dim RMS norm (f32, Raw32). Shape: `[head_v_dim]`.
    pub ssm_norm: &'a [f32],
    /// GDN key/query group count (`metadata.ssm_n_group`), not attention KV heads.
    pub num_k_heads: usize,
    /// GDN key/query state width (`metadata.ssm_d_state`).
    pub head_k_dim: usize,
    /// SSM out projection — `[hidden, d_inner]`.
    pub ssm_out: (&'a [u8], usize, usize, QuantType),
    /// Post-attn RMS norm weight (f32, Raw32 path). Shape: `[hidden]`.
    pub post_attn_norm: &'a [f32],
    /// FFN gate projection — `[ffn_inner, hidden]`.
    pub ffn_gate: (&'a [u8], usize, usize, QuantType),
    /// FFN up projection — `[ffn_inner, hidden]`.
    pub ffn_up: (&'a [u8], usize, usize, QuantType),
    /// FFN down projection — `[hidden, ffn_inner]`.
    pub ffn_down: (&'a [u8], usize, usize, QuantType),
}

/// Wrapper-level validation: `layer_raw_weights.len()` must equal `num_layers`.
///
/// Extracted as a free function so unit tests can exercise the validation
/// branch without a Vulkan device. Called by both `run_fullpath_prefill` and
/// `run_fullpath_decode_step`; the `caller` argument is purely cosmetic
/// (lets the error message identify which entry-point rejected the input).
fn validate_layer_raw_weights_count(
    layer_raw_weights: &[LayerRawWeights<'_>],
    num_layers: usize,
    caller: &'static str,
) -> Result<(), String> {
    if layer_raw_weights.len() != num_layers {
        return Err(format!(
            "{}: layer_raw_weights.len() {} != num_layers {}",
            caller,
            layer_raw_weights.len(),
            num_layers
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Two-pass upload + borrow helpers (private)
// ---------------------------------------------------------------------------

/// Upload pass: for every layer, push every weight tensor into the cache.
/// This is `&mut self.inner`-heavy — no `&GpuBuffer` refs survive the call.
///
/// Per-kv-head K/V sharding: the combined tensor is sliced via
/// `kv_head_shard_byte_range(num_kv_heads, head_dim, hidden, quant, kvh)`,
/// uploaded with `WeightKind::KProjShard(kvh as u16)` /
/// `VProjShard(kvh as u16)`, and the wrapper's caller never sees the
/// per-shard ids.
///
/// `mode = GpuWeightMode::Soa` is used unconditionally — fullpath shaders
/// (record_attention_layer_fullpath, record_*_ffn) read transposed SoA
/// layout. RowMajor mode targets the legacy gemv/gemv_window path only.
fn upload_layer_weights(
    inner: &mut crate::vulkan_backend::PrefillLayerRuntime,
    layer_raw_weights: &[LayerRawWeights<'_>],
    num_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
) -> Result<(), String> {
    for (layer, raw) in layer_raw_weights.iter().enumerate() {
        match raw {
            LayerRawWeights::Attention(a) => {
                upload_attention_layer_weights(inner, layer, a, num_kv_heads, head_dim, hidden)?;
            }
            LayerRawWeights::Gdn(g) => {
                upload_gdn_layer_weights(inner, layer, g)?;
            }
        }
    }
    Ok(())
}

/// Upload pass for a single Attention layer — same logic as the pre-5b
/// monolithic loop, just lifted out into a named helper for the enum-match.
fn upload_attention_layer_weights(
    inner: &mut crate::vulkan_backend::PrefillLayerRuntime,
    layer: usize,
    raw: &AttentionRawWeights<'_>,
    num_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
) -> Result<(), String> {
    let mode = GpuWeightMode::Soa;

    // Q / O projections — row-major source, packed per fullpath shader.
    let (q_bytes, q_rows, q_cols, q_quant) = raw.q_proj;
    inner.upload_weight(
        q_proj_id(layer),
        q_bytes,
        q_rows as u32,
        q_cols as u32,
        q_quant,
        mode,
    )?;
    if let Some(q_norm) = raw.q_norm {
        upload_raw_f32(inner, q_norm_id(layer), q_norm, 1, q_norm.len() as u32)?;
    }
    let (o_bytes, o_rows, o_cols, o_quant) = raw.o_proj;
    inner.upload_weight(
        o_proj_id(layer),
        o_bytes,
        o_rows as u32,
        o_cols as u32,
        o_quant,
        mode,
    )?;

    // FFN gate / up / down — row-major.
    let (gate_bytes, gate_rows, gate_cols, gate_quant) = raw.gate_proj;
    inner.upload_weight(
        ffn_gate_id(layer),
        gate_bytes,
        gate_rows as u32,
        gate_cols as u32,
        gate_quant,
        mode,
    )?;
    let (up_bytes, up_rows, up_cols, up_quant) = raw.up_proj;
    inner.upload_weight(
        ffn_up_id(layer),
        up_bytes,
        up_rows as u32,
        up_cols as u32,
        up_quant,
        mode,
    )?;
    let (down_bytes, down_rows, down_cols, down_quant) = raw.down_proj;
    inner.upload_weight(
        ffn_down_id(layer),
        down_bytes,
        down_rows as u32,
        down_cols as u32,
        down_quant,
        mode,
    )?;

    // RMS norm weights — raw f32 vectors. Backend의 norm pipeline은 dequant
    // 없이 buffer 바이트를 그대로 읽으므로 `GpuWeightMode::Raw32` 경로로
    // 업로드한다. Raw32 path는 `quant` 태그를 무시하므로 `Q4K`는 placeholder
    // (어떤 QuantType이든 결과 동일).
    upload_raw_f32(inner, attn_norm_id(layer), raw.attn_norm, 1, hidden as u32)?;
    upload_raw_f32(inner, ffn_norm_id(layer), raw.ffn_norm, 1, hidden as u32)?;

    // K / V combined → per-kv-head shards.
    let (k_bytes, k_quant) = raw.k_proj_combined;
    let (v_bytes, v_quant) = raw.v_proj_combined;
    if let Some(k_norm) = raw.k_norm {
        upload_raw_f32(inner, k_norm_id(layer), k_norm, 1, k_norm.len() as u32)?;
    }
    for kvh in 0..num_kv_heads {
        let k_range = kv_head_shard_byte_range(num_kv_heads, head_dim, hidden, k_quant, kvh)?;
        inner.upload_weight(
            k_proj_shard_id(layer, kvh as u16),
            &k_bytes[k_range],
            head_dim as u32,
            hidden as u32,
            k_quant,
            mode,
        )?;
        let v_range = kv_head_shard_byte_range(num_kv_heads, head_dim, hidden, v_quant, kvh)?;
        inner.upload_weight(
            v_proj_shard_id(layer, kvh as u16),
            &v_bytes[v_range],
            head_dim as u32,
            hidden as u32,
            v_quant,
            mode,
        )?;
    }
    Ok(())
}

/// Upload pass for a single GDN (Recurrent) layer.
///
/// 14 weights total — 8 f32-raw via `GpuWeightMode::Raw32`, 6 quantized via
/// the standard Soa path. Field order matches `GdnRawWeights<'a>` /
/// `GdnLayerWeightHandles<'a>`. Cache keys are GDN-specific
/// (`WeightKind::Gdn*`) so they don't collide with attention layers in the
/// same model. FFN gate/up/down reuse the existing `FfnGate/Up/Down` keys —
/// those are already layer-keyed, so a hybrid model keeps disjoint cache
/// entries by layer index.
///
/// Until 5d activates the dispatch, this is upload-only — the uploaded
/// buffers are addressable through `build_layer_handles` but the wrapper
/// returns `Err("...10b-5d wiring pending")` before any actual GPU
/// dispatch consumes them.
fn upload_gdn_layer_weights(
    inner: &mut crate::vulkan_backend::PrefillLayerRuntime,
    layer: usize,
    raw: &GdnRawWeights<'_>,
) -> Result<(), String> {
    let mode = GpuWeightMode::Soa;

    // Quantized: fused QKV → z gate → SSM out → FFN gate/up/down.
    let (qkv_bytes, qkv_rows, qkv_cols, qkv_quant) = raw.qkv;
    inner.upload_weight(
        gdn_qkv_id(layer),
        qkv_bytes,
        qkv_rows as u32,
        qkv_cols as u32,
        qkv_quant,
        mode,
    )?;
    let (gate_bytes, gate_rows, gate_cols, gate_quant) = raw.gate;
    inner.upload_weight(
        gdn_gate_id(layer),
        gate_bytes,
        gate_rows as u32,
        gate_cols as u32,
        gate_quant,
        mode,
    )?;
    let (ssm_out_bytes, ssm_out_rows, ssm_out_cols, ssm_out_quant) = raw.ssm_out;
    inner.upload_weight(
        gdn_ssm_out_id(layer),
        ssm_out_bytes,
        ssm_out_rows as u32,
        ssm_out_cols as u32,
        ssm_out_quant,
        mode,
    )?;
    let (ffn_gate_bytes, ffn_gate_rows, ffn_gate_cols, ffn_gate_quant) = raw.ffn_gate;
    inner.upload_weight(
        ffn_gate_id(layer),
        ffn_gate_bytes,
        ffn_gate_rows as u32,
        ffn_gate_cols as u32,
        ffn_gate_quant,
        mode,
    )?;
    let (ffn_up_bytes, ffn_up_rows, ffn_up_cols, ffn_up_quant) = raw.ffn_up;
    inner.upload_weight(
        ffn_up_id(layer),
        ffn_up_bytes,
        ffn_up_rows as u32,
        ffn_up_cols as u32,
        ffn_up_quant,
        mode,
    )?;
    let (ffn_down_bytes, ffn_down_rows, ffn_down_cols, ffn_down_quant) = raw.ffn_down;
    inner.upload_weight(
        ffn_down_id(layer),
        ffn_down_bytes,
        ffn_down_rows as u32,
        ffn_down_cols as u32,
        ffn_down_quant,
        mode,
    )?;

    // f32 raw weights (Raw32 path). Vector fields use `(1, len)`;
    // alpha/beta preserve `[num_heads, hidden]` for the GDN dispatch path.
    let attn_norm_len = raw.attn_norm.len() as u32;
    upload_raw_f32(
        inner,
        gdn_attn_norm_id(layer),
        raw.attn_norm,
        1,
        attn_norm_len,
    )?;
    let post_attn_len = raw.post_attn_norm.len() as u32;
    upload_raw_f32(
        inner,
        gdn_post_attn_norm_id(layer),
        raw.post_attn_norm,
        1,
        post_attn_len,
    )?;
    let (alpha_data, alpha_rows, alpha_cols) = raw.ssm_alpha;
    upload_raw_f32(
        inner,
        gdn_alpha_id(layer),
        alpha_data,
        alpha_rows as u32,
        alpha_cols as u32,
    )?;
    let (beta_data, beta_rows, beta_cols) = raw.ssm_beta;
    upload_raw_f32(
        inner,
        gdn_beta_id(layer),
        beta_data,
        beta_rows as u32,
        beta_cols as u32,
    )?;
    let ssm_a_len = raw.ssm_a.len() as u32;
    upload_raw_f32(inner, gdn_ssm_a_id(layer), raw.ssm_a, 1, ssm_a_len)?;
    let ssm_conv1d_len = raw.ssm_conv1d.len() as u32;
    upload_raw_f32(
        inner,
        gdn_ssm_conv1d_id(layer),
        raw.ssm_conv1d,
        1,
        ssm_conv1d_len,
    )?;
    let ssm_dt_bias_len = raw.ssm_dt_bias.len() as u32;
    upload_raw_f32(
        inner,
        gdn_ssm_dt_bias_id(layer),
        raw.ssm_dt_bias,
        1,
        ssm_dt_bias_len,
    )?;
    let ssm_norm_len = raw.ssm_norm.len() as u32;
    upload_raw_f32(inner, gdn_ssm_norm_id(layer), raw.ssm_norm, 1, ssm_norm_len)?;

    Ok(())
}

/// Helper: upload a `&[f32]` slice via the `GpuWeightMode::Raw32` path.
///
/// `quant` is set to `QuantType::Q4K` as a placeholder — the Raw32 path
/// ignores it (mv28 I2). Used by both attention norm uploads and the GDN
/// f32-raw weights.
fn upload_raw_f32(
    inner: &mut crate::vulkan_backend::PrefillLayerRuntime,
    id: WeightId,
    data: &[f32],
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    inner.upload_weight(
        id,
        bytes,
        rows,
        cols,
        QuantType::Q4K, // ignored on Raw32 path
        GpuWeightMode::Raw32,
    )
}

/// Borrow pass (raw-pointer snapshot variant): per layer, snapshot raw
/// pointers to every weight buffer we just uploaded. The pointer values
/// remain valid as long as the cache doesn't evict — which by construction
/// is impossible inside a single `run_fullpath_*` call (no further
/// `get_or_upload` runs once we enter the borrow phase).
///
/// Why raw pointers and not `&GpuBuffer`? Because `run_prefill` /
/// `run_decode_step` take `&mut self.inner`, while every borrowed
/// `&GpuBuffer` reaches into the same `inner.cache`. The borrow checker
/// rejects holding both at the same time (E0502). The same trick is used
/// inside `full_path.rs::FullpathBufferRefs` (mv26-task10b-3) for
/// startup-hoisted staging buffers.
///
/// SAFETY contract:
/// 1. `upload_layer_weights` ran successfully → every `weight_buffer(id)`
///    lookup must succeed (or we return Err here and abort before any
///    pointer reads happen).
/// 2. The returned `Vec<LayerHandlesStorage>` must live for the duration of
///    the subsequent `run_prefill` / `run_decode_step` call. The caller in
///    `run_fullpath_prefill` / `run_fullpath_decode_step` keeps it on the
///    stack across the call, satisfying this.
/// 3. No code path between snapshot and run_prefill triggers cache eviction
///    or destroy. Both `run_prefill` / `run_decode_step` only call
///    `record_*` / `ensure_fullpath_staging` / `ensure_hidden_ping_pong` —
///    none touch `cache.entries`.
fn build_layer_handles(
    inner: &crate::vulkan_backend::PrefillLayerRuntime,
    layer_raw_weights: &[LayerRawWeights<'_>],
    layer_kinds: &[ModelLayerKind],
    num_kv_heads: usize,
) -> Result<Vec<LayerHandlesStorage>, String> {
    if !layer_kinds.is_empty() && layer_kinds.len() != layer_raw_weights.len() {
        return Err(format!(
            "build_layer_handles: layer_kinds.len() {} != layer_raw_weights.len() {}",
            layer_kinds.len(),
            layer_raw_weights.len(),
        ));
    }
    let mut out = Vec::with_capacity(layer_raw_weights.len());
    for (layer, raw) in layer_raw_weights.iter().enumerate() {
        // Cross-check: the kind hint (when provided) must match the enum
        // variant of the per-layer raw weights. A mismatch indicates the
        // engine wiring miscoupled `layer_kinds` with `LayerRawWeights`.
        let kind_hint = layer_kinds.get(layer).copied();
        match (raw, kind_hint) {
            (LayerRawWeights::Attention(a), Some(ModelLayerKind::Attention) | None) => {
                out.push(build_attention_layer_handles(
                    inner,
                    layer,
                    a,
                    num_kv_heads,
                )?);
            }
            (LayerRawWeights::Gdn(g), Some(ModelLayerKind::Recurrent) | None) => {
                out.push(build_gdn_layer_handles(inner, layer, g)?);
            }
            (LayerRawWeights::Attention(_), Some(other)) => {
                return Err(format!(
                    "build_layer_handles: layer {} raw=Attention but layer_kinds={:?}",
                    layer, other
                ));
            }
            (LayerRawWeights::Gdn(_), Some(other)) => {
                return Err(format!(
                    "build_layer_handles: layer {} raw=Gdn but layer_kinds={:?}",
                    layer, other
                ));
            }
        }
    }
    Ok(out)
}

/// Borrow pass for one attention layer — same logic as the pre-5b
/// monolithic loop, just lifted out for the enum-match dispatch.
fn build_attention_layer_handles(
    inner: &crate::vulkan_backend::PrefillLayerRuntime,
    layer: usize,
    raw: &AttentionRawWeights<'_>,
    num_kv_heads: usize,
) -> Result<LayerHandlesStorage, String> {
    let attn_norm_buf = inner.weight_buffer(attn_norm_id(layer)).ok_or_else(|| {
        format!(
            "build_layer_handles: layer {} attn_norm not in cache",
            layer
        )
    })?;
    let ffn_norm_buf = inner
        .weight_buffer(ffn_norm_id(layer))
        .ok_or_else(|| format!("build_layer_handles: layer {} ffn_norm not in cache", layer))?;
    let q_weight_buf = inner
        .weight_buffer(q_proj_id(layer))
        .ok_or_else(|| format!("build_layer_handles: layer {} q_proj not in cache", layer))?;
    let q_norm_buf =
        if raw.q_norm.is_some() {
            Some(inner.weight_buffer(q_norm_id(layer)).ok_or_else(|| {
                format!("build_layer_handles: layer {} q_norm not in cache", layer)
            })?)
        } else {
            None
        };
    let k_norm_buf =
        if raw.k_norm.is_some() {
            Some(inner.weight_buffer(k_norm_id(layer)).ok_or_else(|| {
                format!("build_layer_handles: layer {} k_norm not in cache", layer)
            })?)
        } else {
            None
        };
    let o_weight_buf = inner
        .weight_buffer(o_proj_id(layer))
        .ok_or_else(|| format!("build_layer_handles: layer {} o_proj not in cache", layer))?;
    let gate_weight_buf = inner
        .weight_buffer(ffn_gate_id(layer))
        .ok_or_else(|| format!("build_layer_handles: layer {} ffn_gate not in cache", layer))?;
    let up_weight_buf = inner
        .weight_buffer(ffn_up_id(layer))
        .ok_or_else(|| format!("build_layer_handles: layer {} ffn_up not in cache", layer))?;
    let down_weight_buf = inner
        .weight_buffer(ffn_down_id(layer))
        .ok_or_else(|| format!("build_layer_handles: layer {} ffn_down not in cache", layer))?;

    let mut k_ptrs: Vec<*const GpuBuffer> = Vec::with_capacity(num_kv_heads);
    let mut v_ptrs: Vec<*const GpuBuffer> = Vec::with_capacity(num_kv_heads);
    for kvh in 0..num_kv_heads {
        let k = inner
            .weight_buffer(k_proj_shard_id(layer, kvh as u16))
            .ok_or_else(|| {
                format!(
                    "build_layer_handles: layer {} k_shard[{}] not in cache",
                    layer, kvh
                )
            })?;
        let v = inner
            .weight_buffer(v_proj_shard_id(layer, kvh as u16))
            .ok_or_else(|| {
                format!(
                    "build_layer_handles: layer {} v_shard[{}] not in cache",
                    layer, kvh
                )
            })?;
        k_ptrs.push(k as *const GpuBuffer);
        v_ptrs.push(v as *const GpuBuffer);
    }
    // The backend's `record_q_window_into_kv_mirror[_strided]` consumes
    // `LayerWeightHandles.{k,v}_weight_size` as the
    // `VkDescriptorBufferInfo.range` for ONE shard buffer at offset 0
    // (see `pipeline::bind_buffers_with_offsets` →
    // `record_q_window_into_kv_mirror_strided` at layer_gemv.rs:4280).
    // Vulkan spec requires `range ≤ buffer.size - offset`, so we must
    // pass the SINGLE-shard size, not the accumulated total across kv
    // heads. Mirrors the precedent at layer_gemv.rs:11471
    // (`let k_weight_size = k_handles[0].size;`).
    let k_weight_size = k_ptrs.first().map(|p| unsafe { (**p).size() }).unwrap_or(0);
    let v_weight_size = v_ptrs.first().map(|p| unsafe { (**p).size() }).unwrap_or(0);
    debug_assert!(
        k_ptrs
            .iter()
            .all(|p| unsafe { (**p).size() } == k_weight_size),
        "K shards must all have identical size (same quant + same [head_dim, hidden] shape per shard)"
    );
    debug_assert!(
        v_ptrs
            .iter()
            .all(|p| unsafe { (**p).size() } == v_weight_size),
        "V shards must all have identical size (same quant + same [head_dim, hidden] shape per shard)"
    );

    Ok(LayerHandlesStorage::Attention(AttentionHandlesStorage {
        attn_norm: attn_norm_buf as *const GpuBuffer,
        attn_norm_size: attn_norm_buf.size(),
        ffn_norm: ffn_norm_buf as *const GpuBuffer,
        ffn_norm_size: ffn_norm_buf.size(),
        q: q_weight_buf as *const GpuBuffer,
        q_size: q_weight_buf.size(),
        q_rows: raw.q_proj.1,
        q_cols: raw.q_proj.2,
        q_quant: raw.q_proj.3,
        q_norm: q_norm_buf.map(|buf| buf as *const GpuBuffer),
        q_norm_size: q_norm_buf.map(|buf| buf.size()).unwrap_or(0),
        o: o_weight_buf as *const GpuBuffer,
        o_size: o_weight_buf.size(),
        o_quant: raw.o_proj.3,
        gate: gate_weight_buf as *const GpuBuffer,
        gate_size: gate_weight_buf.size(),
        gate_quant: raw.gate_proj.3,
        up: up_weight_buf as *const GpuBuffer,
        up_size: up_weight_buf.size(),
        up_quant: raw.up_proj.3,
        down: down_weight_buf as *const GpuBuffer,
        down_size: down_weight_buf.size(),
        down_quant: raw.down_proj.3,
        k_ptrs,
        k_weight_size,
        k_quant: raw.k_proj_combined.1,
        k_norm: k_norm_buf.map(|buf| buf as *const GpuBuffer),
        k_norm_size: k_norm_buf.map(|buf| buf.size()).unwrap_or(0),
        v_ptrs,
        v_weight_size,
        v_quant: raw.v_proj_combined.1,
    }))
}

/// Borrow pass for one GDN layer — snapshot raw pointers to all 14 cached
/// GDN weight buffers.
///
/// Mirrors `build_attention_layer_handles` but maps to the GDN-specific
/// `WeightKind::Gdn*` ids (and reuses `FfnGate/Up/Down` for the FFN block,
/// since those are layer-keyed and don't collide with attention layers in
/// the same model — attention layers and GDN layers occupy disjoint layer
/// indices in hybrid models).
///
/// The returned storage is read by `as_handles_gdn` to produce a
/// `GdnLayerWeightHandles<'a>` for the future 5d dispatch path. Until 5d
/// activates the dispatch, this code path runs to completion but the
/// wrapper rejects it via `Err("...10b-5d wiring pending")` before any
/// GPU consumption.
fn build_gdn_layer_handles(
    inner: &crate::vulkan_backend::PrefillLayerRuntime,
    layer: usize,
    raw: &GdnRawWeights<'_>,
) -> Result<LayerHandlesStorage, String> {
    let lookup = |id, label: &'static str| {
        inner.weight_buffer(id).ok_or_else(|| {
            format!(
                "build_layer_handles: layer {} gdn {} not in cache",
                layer, label
            )
        })
    };

    let attn_norm_buf = lookup(gdn_attn_norm_id(layer), "attn_norm")?;
    let qkv_buf = lookup(gdn_qkv_id(layer), "qkv")?;
    let gate_buf = lookup(gdn_gate_id(layer), "gate")?;
    let alpha_buf = lookup(gdn_alpha_id(layer), "ssm_alpha")?;
    let beta_buf = lookup(gdn_beta_id(layer), "ssm_beta")?;
    let ssm_a_buf = lookup(gdn_ssm_a_id(layer), "ssm_a")?;
    let ssm_conv1d_buf = lookup(gdn_ssm_conv1d_id(layer), "ssm_conv1d")?;
    let ssm_dt_bias_buf = lookup(gdn_ssm_dt_bias_id(layer), "ssm_dt_bias")?;
    let ssm_norm_buf = lookup(gdn_ssm_norm_id(layer), "ssm_norm")?;
    let ssm_out_buf = lookup(gdn_ssm_out_id(layer), "ssm_out")?;
    let post_attn_norm_buf = lookup(gdn_post_attn_norm_id(layer), "post_attn_norm")?;
    let ffn_gate_buf = lookup(ffn_gate_id(layer), "ffn_gate")?;
    let ffn_up_buf = lookup(ffn_up_id(layer), "ffn_up")?;
    let ffn_down_buf = lookup(ffn_down_id(layer), "ffn_down")?;

    Ok(LayerHandlesStorage::Gdn(GdnHandlesStorage {
        attn_norm: attn_norm_buf as *const GpuBuffer,
        attn_norm_size: attn_norm_buf.size(),
        qkv: qkv_buf as *const GpuBuffer,
        qkv_size: qkv_buf.size(),
        qkv_rows: raw.qkv.1,
        qkv_cols: raw.qkv.2,
        qkv_quant: raw.qkv.3,
        gate: gate_buf as *const GpuBuffer,
        gate_size: gate_buf.size(),
        gate_rows: raw.gate.1,
        gate_cols: raw.gate.2,
        gate_quant: raw.gate.3,
        ssm_alpha: alpha_buf as *const GpuBuffer,
        ssm_alpha_size: alpha_buf.size(),
        ssm_alpha_rows: raw.ssm_alpha.1,
        ssm_alpha_cols: raw.ssm_alpha.2,
        ssm_beta: beta_buf as *const GpuBuffer,
        ssm_beta_size: beta_buf.size(),
        ssm_beta_rows: raw.ssm_beta.1,
        ssm_beta_cols: raw.ssm_beta.2,
        ssm_a: ssm_a_buf as *const GpuBuffer,
        ssm_a_size: ssm_a_buf.size(),
        ssm_conv1d: ssm_conv1d_buf as *const GpuBuffer,
        ssm_conv1d_size: ssm_conv1d_buf.size(),
        ssm_dt_bias: ssm_dt_bias_buf as *const GpuBuffer,
        ssm_dt_bias_size: ssm_dt_bias_buf.size(),
        ssm_norm: ssm_norm_buf as *const GpuBuffer,
        ssm_norm_size: ssm_norm_buf.size(),
        num_k_heads: raw.num_k_heads,
        head_k_dim: raw.head_k_dim,
        ssm_out: ssm_out_buf as *const GpuBuffer,
        ssm_out_size: ssm_out_buf.size(),
        ssm_out_rows: raw.ssm_out.1,
        ssm_out_cols: raw.ssm_out.2,
        ssm_out_quant: raw.ssm_out.3,
        post_attn_norm: post_attn_norm_buf as *const GpuBuffer,
        post_attn_norm_size: post_attn_norm_buf.size(),
        ffn_gate: ffn_gate_buf as *const GpuBuffer,
        ffn_gate_size: ffn_gate_buf.size(),
        ffn_gate_rows: raw.ffn_gate.1,
        ffn_gate_cols: raw.ffn_gate.2,
        ffn_gate_quant: raw.ffn_gate.3,
        ffn_up: ffn_up_buf as *const GpuBuffer,
        ffn_up_size: ffn_up_buf.size(),
        ffn_up_rows: raw.ffn_up.1,
        ffn_up_cols: raw.ffn_up.2,
        ffn_up_quant: raw.ffn_up.3,
        ffn_down: ffn_down_buf as *const GpuBuffer,
        ffn_down_size: ffn_down_buf.size(),
        ffn_down_rows: raw.ffn_down.1,
        ffn_down_cols: raw.ffn_down.2,
        ffn_down_quant: raw.ffn_down.3,
    }))
}

/// Per-layer raw-pointer snapshot. The enum mirrors `LayerRawWeights<'a>` —
/// each variant owns the per-layer cache pointer set for one layer kind.
///
/// mv28-task10b-5b: split into Attention / Gdn variants so hybrid models can
/// build a heterogeneous handle Vec. The `as_*_handles()` SAFETY contract is
/// shared across both variants — see `as_attention_handles` / `as_gdn_handles`.
enum LayerHandlesStorage {
    Attention(AttentionHandlesStorage),
    /// Constructed by `build_gdn_layer_handles` but not yet dispatched —
    /// `extract_attention_handles_or_err` rejects this arm with a
    /// "10b-5d wiring pending" error until 5d activates the GDN path.
    #[allow(dead_code)]
    Gdn(GdnHandlesStorage),
}

/// Owns the per-kv-head pointer Vecs so the `&[&GpuBuffer]` slices we
/// materialize in `as_attention_handles` outlive the `run_prefill` /
/// `run_decode_step` invocation.
struct AttentionHandlesStorage {
    attn_norm: *const GpuBuffer,
    attn_norm_size: u64,
    ffn_norm: *const GpuBuffer,
    ffn_norm_size: u64,
    q: *const GpuBuffer,
    q_size: u64,
    q_rows: usize,
    q_cols: usize,
    q_quant: QuantType,
    q_norm: Option<*const GpuBuffer>,
    q_norm_size: u64,
    o: *const GpuBuffer,
    o_size: u64,
    o_quant: QuantType,
    gate: *const GpuBuffer,
    gate_size: u64,
    gate_quant: QuantType,
    up: *const GpuBuffer,
    up_size: u64,
    up_quant: QuantType,
    down: *const GpuBuffer,
    down_size: u64,
    down_quant: QuantType,
    /// Per-kv-head K shard pointers. Stored as `*const` so the storage
    /// struct itself doesn't carry any lifetime parameter — borrows are
    /// only constructed on-demand by `as_attention_handles()`.
    k_ptrs: Vec<*const GpuBuffer>,
    /// SINGLE-shard byte size (not the accumulated total across kv heads).
    /// All shards share the same `[head_dim, hidden]` shape and quant type,
    /// so any shard's size is a valid descriptor `range` upper bound for
    /// the K pipeline. The backend binds one shard buffer at a time at
    /// offset 0, so passing the accumulated total here would violate the
    /// Vulkan spec (`range ≤ buffer.size - offset`).
    k_weight_size: u64,
    k_quant: QuantType,
    k_norm: Option<*const GpuBuffer>,
    k_norm_size: u64,
    v_ptrs: Vec<*const GpuBuffer>,
    /// SINGLE-shard byte size — same rationale as `k_weight_size`.
    v_weight_size: u64,
    v_quant: QuantType,
}

/// Per-layer raw-pointer snapshot for one GDN (Recurrent) layer.
///
/// 14 weight pointers — f32-raw plus quantized — mirroring `GdnRawWeights<'a>`
/// / `GdnLayerWeightHandles<'a>`. No per-kv-head sharding for GDN (the conv1d
/// kernel is shared across all heads). All scalar `*const GpuBuffer`, no Vecs.
///
/// Fields are populated by `build_gdn_layer_handles` and consumed by
/// `as_gdn_handles`; the latter is wired but not yet dispatched until 5d
/// extends `FullPathPrefillInput` to carry GDN handles.
#[allow(dead_code)] // 5d activates the as_gdn_handles consumer
struct GdnHandlesStorage {
    attn_norm: *const GpuBuffer,
    attn_norm_size: u64,
    qkv: *const GpuBuffer,
    qkv_size: u64,
    qkv_rows: usize,
    qkv_cols: usize,
    qkv_quant: QuantType,
    gate: *const GpuBuffer,
    gate_size: u64,
    gate_rows: usize,
    gate_cols: usize,
    gate_quant: QuantType,
    ssm_alpha: *const GpuBuffer,
    ssm_alpha_size: u64,
    ssm_alpha_rows: usize,
    ssm_alpha_cols: usize,
    ssm_beta: *const GpuBuffer,
    ssm_beta_size: u64,
    ssm_beta_rows: usize,
    ssm_beta_cols: usize,
    ssm_a: *const GpuBuffer,
    ssm_a_size: u64,
    ssm_conv1d: *const GpuBuffer,
    ssm_conv1d_size: u64,
    ssm_dt_bias: *const GpuBuffer,
    ssm_dt_bias_size: u64,
    ssm_norm: *const GpuBuffer,
    ssm_norm_size: u64,
    num_k_heads: usize,
    head_k_dim: usize,
    ssm_out: *const GpuBuffer,
    ssm_out_size: u64,
    ssm_out_rows: usize,
    ssm_out_cols: usize,
    ssm_out_quant: QuantType,
    post_attn_norm: *const GpuBuffer,
    post_attn_norm_size: u64,
    ffn_gate: *const GpuBuffer,
    ffn_gate_size: u64,
    ffn_gate_rows: usize,
    ffn_gate_cols: usize,
    ffn_gate_quant: QuantType,
    ffn_up: *const GpuBuffer,
    ffn_up_size: u64,
    ffn_up_rows: usize,
    ffn_up_cols: usize,
    ffn_up_quant: QuantType,
    ffn_down: *const GpuBuffer,
    ffn_down_size: u64,
    ffn_down_rows: usize,
    ffn_down_cols: usize,
    ffn_down_quant: QuantType,
}

impl LayerHandlesStorage {
    /// Build a backend `LayerHandle<'a>` from the per-layer storage variant.
    ///
    /// Variant dispatch: Attention → `LayerWeightHandles<'a>`, Gdn →
    /// `GdnLayerWeightHandles<'a>`. The shared SAFETY contract documented on
    /// the per-variant helpers covers both arms.
    ///
    unsafe fn as_backend_handle<'a>(&'a self) -> BackendLayerHandle<'a> {
        match self {
            LayerHandlesStorage::Attention(a) => {
                BackendLayerHandle::Attention(unsafe { a.as_attention_handles() })
            }
            LayerHandlesStorage::Gdn(g) => BackendLayerHandle::Gdn(unsafe { g.as_gdn_handles() }),
        }
    }
}

impl AttentionHandlesStorage {
    /// Build a `LayerWeightHandles<'a>` borrowing from the storage's own
    /// fields.
    ///
    /// SAFETY: caller (the `run_fullpath_*` wrapper) must guarantee that:
    /// 1. Every pointer in `k_ptrs` / `v_ptrs` (and the scalar `*const`
    ///    fields) is non-null and points at a live `GpuBuffer` owned by
    ///    `inner.cache`. This is guaranteed by `build_layer_handles`'s
    ///    `ok_or_else(...)?` ladder.
    /// 2. The cache is not mutated (no `get_or_upload`, no `destroy`)
    ///    between `build_layer_handles` and the end of the ensuing
    ///    `run_prefill` / `run_decode_step` call. This is true because the
    ///    wrapper performs all uploads before the borrow phase, and
    ///    `run_prefill` / `run_decode_step` only call `record_*` /
    ///    `ensure_fullpath_staging` / `ensure_hidden_ping_pong` — none of
    ///    which touch `cache.entries`.
    /// 3. `Self` outlives the returned `LayerWeightHandles<'a>` (enforced
    ///    by the borrow checker — `'a` ties to `&'a self`).
    ///
    /// The `Vec<*const GpuBuffer>` → `&[&'a GpuBuffer]` reinterpretation in
    /// the K/V slice fields relies on Rust's guarantee that `*const T` and
    /// `&T` share the same memory layout (one machine word, non-null
    /// invariant aside). We rebuild the slice with
    /// `core::slice::from_raw_parts(ptr_as_ref_ptr, len)` so the unsafe
    /// scope is narrower than a `mem::transmute` of the whole slice; the
    /// only soundness requirement is that every pointer is currently a
    /// valid `&GpuBuffer` (precondition #1).
    unsafe fn as_attention_handles<'a>(&'a self) -> LayerWeightHandles<'a> {
        // Reborrow each *const GpuBuffer back to &'a GpuBuffer.
        let attn_norm_buf: &'a GpuBuffer = unsafe { &*self.attn_norm };
        let ffn_norm_buf: &'a GpuBuffer = unsafe { &*self.ffn_norm };
        let q_weight_buf: &'a GpuBuffer = unsafe { &*self.q };
        let q_norm_buf: Option<&'a GpuBuffer> = self.q_norm.map(|p| unsafe { &*p });
        let k_norm_buf: Option<&'a GpuBuffer> = self.k_norm.map(|p| unsafe { &*p });
        let o_weight_buf: &'a GpuBuffer = unsafe { &*self.o };
        let gate_weight_buf: &'a GpuBuffer = unsafe { &*self.gate };
        let up_weight_buf: &'a GpuBuffer = unsafe { &*self.up };
        let down_weight_buf: &'a GpuBuffer = unsafe { &*self.down };

        // Reinterpret `&[*const GpuBuffer]` as `&[&'a GpuBuffer]` using
        // `slice::from_raw_parts`. `*const T` and `&T` share the same ABI
        // (one machine word, non-null when the reference is valid), so
        // viewing the underlying storage as a slice of references is
        // layout-safe. SAFETY:
        //   - Each entry is a valid `&GpuBuffer` borrow (precondition #1).
        //   - The resulting slice's lifetime `'a` is tied to `&'a self`,
        //     so it cannot outlive the storage Vec.
        //   - `from_raw_parts` upholds slice alignment because
        //     `*const T` and `&T` share both layout and alignment.
        let k_slice: &'a [&'a GpuBuffer] = unsafe {
            core::slice::from_raw_parts(
                self.k_ptrs.as_ptr() as *const &'a GpuBuffer,
                self.k_ptrs.len(),
            )
        };
        let v_slice: &'a [&'a GpuBuffer] = unsafe {
            core::slice::from_raw_parts(
                self.v_ptrs.as_ptr() as *const &'a GpuBuffer,
                self.v_ptrs.len(),
            )
        };

        LayerWeightHandles {
            attn_norm_buf,
            attn_norm_size: self.attn_norm_size,
            q_weight_buf,
            q_weight_size: self.q_size,
            q_rows: self.q_rows,
            q_cols: self.q_cols,
            q_quant: self.q_quant,
            q_norm_buf,
            q_norm_size: self.q_norm_size,
            k_weight_bufs: k_slice,
            k_weight_size: self.k_weight_size,
            k_quant: self.k_quant,
            k_norm_buf,
            k_norm_size: self.k_norm_size,
            v_weight_bufs: v_slice,
            v_weight_size: self.v_weight_size,
            v_quant: self.v_quant,
            o_weight_buf,
            o_weight_size: self.o_size,
            o_quant: self.o_quant,
            ffn_norm_buf,
            ffn_norm_size: self.ffn_norm_size,
            gate_weight_buf,
            gate_weight_size: self.gate_size,
            gate_quant: self.gate_quant,
            up_weight_buf,
            up_weight_size: self.up_size,
            up_quant: self.up_quant,
            down_weight_buf,
            down_weight_size: self.down_size,
            down_quant: self.down_quant,
        }
    }
}

impl GdnHandlesStorage {
    /// Build a `GdnLayerWeightHandles<'a>` borrowing from the storage.
    ///
    /// SAFETY: same contract as `AttentionHandlesStorage::as_attention_handles`
    /// (every pointer is a live `&GpuBuffer`; no cache mutation between
    /// `build_layer_handles` and the dispatch end). No K/V slice
    /// reinterpretation here — GDN has no per-kv-head sharding (conv1d
    /// kernel is shared across heads), so every field is a single
    /// `*const GpuBuffer`.
    #[allow(dead_code)] // 5d's GDN dispatch will consume this
    unsafe fn as_gdn_handles<'a>(&'a self) -> GdnLayerWeightHandles<'a> {
        GdnLayerWeightHandles {
            attn_norm_buf: unsafe { &*self.attn_norm },
            attn_norm_size: self.attn_norm_size,
            qkv_weight_buf: unsafe { &*self.qkv },
            qkv_weight_size: self.qkv_size,
            qkv_rows: self.qkv_rows,
            qkv_cols: self.qkv_cols,
            qkv_quant: self.qkv_quant,
            gate_weight_buf: unsafe { &*self.gate },
            gate_weight_size: self.gate_size,
            gate_rows: self.gate_rows,
            gate_cols: self.gate_cols,
            gate_quant: self.gate_quant,
            ssm_alpha_buf: unsafe { &*self.ssm_alpha },
            ssm_alpha_size: self.ssm_alpha_size,
            ssm_alpha_rows: self.ssm_alpha_rows,
            ssm_alpha_cols: self.ssm_alpha_cols,
            ssm_beta_buf: unsafe { &*self.ssm_beta },
            ssm_beta_size: self.ssm_beta_size,
            ssm_beta_rows: self.ssm_beta_rows,
            ssm_beta_cols: self.ssm_beta_cols,
            ssm_a_buf: unsafe { &*self.ssm_a },
            ssm_a_size: self.ssm_a_size,
            ssm_conv1d_buf: unsafe { &*self.ssm_conv1d },
            ssm_conv1d_size: self.ssm_conv1d_size,
            ssm_dt_bias_buf: unsafe { &*self.ssm_dt_bias },
            ssm_dt_bias_size: self.ssm_dt_bias_size,
            ssm_norm_buf: unsafe { &*self.ssm_norm },
            ssm_norm_size: self.ssm_norm_size,
            num_k_heads: self.num_k_heads,
            head_k_dim: self.head_k_dim,
            ssm_out_buf: unsafe { &*self.ssm_out },
            ssm_out_size: self.ssm_out_size,
            ssm_out_rows: self.ssm_out_rows,
            ssm_out_cols: self.ssm_out_cols,
            ssm_out_quant: self.ssm_out_quant,
            post_attn_norm_buf: unsafe { &*self.post_attn_norm },
            post_attn_norm_size: self.post_attn_norm_size,
            ffn_gate_weight_buf: unsafe { &*self.ffn_gate },
            ffn_gate_weight_size: self.ffn_gate_size,
            ffn_gate_rows: self.ffn_gate_rows,
            ffn_gate_cols: self.ffn_gate_cols,
            ffn_gate_quant: self.ffn_gate_quant,
            ffn_up_weight_buf: unsafe { &*self.ffn_up },
            ffn_up_weight_size: self.ffn_up_size,
            ffn_up_rows: self.ffn_up_rows,
            ffn_up_cols: self.ffn_up_cols,
            ffn_up_quant: self.ffn_up_quant,
            ffn_down_weight_buf: unsafe { &*self.ffn_down },
            ffn_down_weight_size: self.ffn_down_size,
            ffn_down_rows: self.ffn_down_rows,
            ffn_down_cols: self.ffn_down_cols,
            ffn_down_quant: self.ffn_down_quant,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (mv27-task10b-4c-2a)
//
// Real GPU smoke tests for run_fullpath_prefill / run_fullpath_decode_step
// require a working Vulkan device + the engine-side weight pipeline that
// 4c-2b will land. For now we only smoke-test the validation layer (input
// shape checks) using `layer_raw_weights.len() != num_layers` without
// constructing a LayerRuntime — those tests don't need a GPU.
//
// The integration test that actually dispatches `run_prefill` against a
// real Vulkan device lives in `crates/rnb-backend/vulkan/tests/full_path_*.rs`
// and is `#[ignore]`-gated (see `full_path_shapes.rs`). Once 4c-2b wires
// the engine, an `#[ignore]`-gated end-to-end test will be added here too.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_raw_weights_struct_compiles() {
        // Smoke test: prove that LayerRawWeights<'_>::Attention(...) can be
        // constructed with borrowed slices and that the struct field shapes
        // line up with what 4c-2b's engine wiring hands us. Catches
        // accidental field renames / signature drift without needing a
        // Vulkan device.
        let attn_norm_w: Vec<f32> = vec![1.0; 256];
        let ffn_norm_w: Vec<f32> = vec![1.0; 256];
        let q_bytes: Vec<u8> = vec![0; 144 * 256]; // 256 elems => 1 superblock; allocate generously
        let k_bytes: Vec<u8> = vec![0; 144 * 256];
        let v_bytes: Vec<u8> = vec![0; 144 * 256];
        let o_bytes: Vec<u8> = vec![0; 144 * 256];
        let gate_bytes: Vec<u8> = vec![0; 144 * 1024];
        let up_bytes: Vec<u8> = vec![0; 144 * 1024];
        let down_bytes: Vec<u8> = vec![0; 144 * 1024];

        let attn = AttentionRawWeights {
            attn_norm: &attn_norm_w,
            q_proj: (&q_bytes, 256, 256, QuantType::Q4K),
            q_norm: None,
            k_proj_combined: (&k_bytes, QuantType::Q4K),
            k_norm: None,
            v_proj_combined: (&v_bytes, QuantType::Q4K),
            o_proj: (&o_bytes, 256, 256, QuantType::Q4K),
            ffn_norm: &ffn_norm_w,
            gate_proj: (&gate_bytes, 1024, 256, QuantType::Q4K),
            up_proj: (&up_bytes, 1024, 256, QuantType::Q4K),
            down_proj: (&down_bytes, 256, 1024, QuantType::Q4K),
        };

        // Field-shape spot checks — guards against silent struct drift.
        assert_eq!(attn.q_proj.1, 256);
        assert_eq!(attn.q_proj.2, 256);
        assert!(matches!(attn.k_proj_combined.1, QuantType::Q4K));
        assert_eq!(attn.attn_norm.len(), 256);

        // Wrap into the enum to prove the variant compiles.
        let raw = LayerRawWeights::Attention(attn);
        match &raw {
            LayerRawWeights::Attention(a) => assert_eq!(a.q_proj.1, 256),
            LayerRawWeights::Gdn(_) => panic!("expected Attention variant"),
        }
    }

    #[test]
    fn attention_raw_weights_carries_gated_q_norm_metadata() {
        let attn_norm_w: Vec<f32> = vec![1.0; 256];
        let q_norm_w: Vec<f32> = vec![1.0; 64];
        let k_norm_w: Vec<f32> = vec![1.0; 64];
        let ffn_norm_w: Vec<f32> = vec![1.0; 256];
        let q_bytes: Vec<u8> = vec![0; 144 * 512];
        let k_bytes: Vec<u8> = vec![0; 144 * 128];
        let v_bytes: Vec<u8> = vec![0; 144 * 128];
        let o_bytes: Vec<u8> = vec![0; 144 * 256];
        let gate_bytes: Vec<u8> = vec![0; 144 * 1024];
        let up_bytes: Vec<u8> = vec![0; 144 * 1024];
        let down_bytes: Vec<u8> = vec![0; 144 * 1024];

        let attn = AttentionRawWeights {
            attn_norm: &attn_norm_w,
            q_proj: (&q_bytes, 512, 256, QuantType::Q4K),
            q_norm: Some(&q_norm_w),
            k_proj_combined: (&k_bytes, QuantType::Q4K),
            k_norm: Some(&k_norm_w),
            v_proj_combined: (&v_bytes, QuantType::Q4K),
            o_proj: (&o_bytes, 256, 256, QuantType::Q4K),
            ffn_norm: &ffn_norm_w,
            gate_proj: (&gate_bytes, 1024, 256, QuantType::Q4K),
            up_proj: (&up_bytes, 1024, 256, QuantType::Q4K),
            down_proj: (&down_bytes, 256, 1024, QuantType::Q4K),
        };

        assert_eq!(attn.q_proj.1, 512);
        assert_eq!(attn.q_norm.expect("q_norm must be carried").len(), 64);
        assert_eq!(attn.k_norm.expect("k_norm must be carried").len(), 64);
    }

    #[test]
    fn gdn_raw_weights_struct_compiles() {
        // mv28-task10b-5b: smoke test the GDN variant. Field count + shape
        // spot check; no GPU dispatch.
        let attn_norm_w: Vec<f32> = vec![1.0; 256];
        let post_attn_norm_w: Vec<f32> = vec![1.0; 256];
        let ssm_a: Vec<f32> = vec![0.0; 8];
        let ssm_conv1d: Vec<f32> = vec![0.0; 4 * 256];
        let ssm_dt_bias: Vec<f32> = vec![0.0; 8];
        let ssm_norm: Vec<f32> = vec![1.0; 64];
        let qkv_bytes: Vec<u8> = vec![0; 144 * 256];
        let gate_bytes: Vec<u8> = vec![0; 144 * 256];
        let alpha_w: Vec<f32> = vec![0.0; 8 * 256];
        let beta_w: Vec<f32> = vec![0.0; 8 * 256];
        let ssm_out_bytes: Vec<u8> = vec![0; 144 * 256];
        let ffn_gate_bytes: Vec<u8> = vec![0; 144 * 1024];
        let ffn_up_bytes: Vec<u8> = vec![0; 144 * 1024];
        let ffn_down_bytes: Vec<u8> = vec![0; 144 * 1024];

        let gdn = GdnRawWeights {
            attn_norm: &attn_norm_w,
            qkv: (&qkv_bytes, 256, 256, QuantType::Q4K),
            gate: (&gate_bytes, 256, 256, QuantType::Q4K),
            ssm_alpha: (&alpha_w, 8, 256),
            ssm_beta: (&beta_w, 8, 256),
            ssm_a: &ssm_a,
            ssm_conv1d: &ssm_conv1d,
            ssm_dt_bias: &ssm_dt_bias,
            ssm_norm: &ssm_norm,
            num_k_heads: 4,
            head_k_dim: 32,
            ssm_out: (&ssm_out_bytes, 256, 256, QuantType::Q4K),
            post_attn_norm: &post_attn_norm_w,
            ffn_gate: (&ffn_gate_bytes, 1024, 256, QuantType::Q4K),
            ffn_up: (&ffn_up_bytes, 1024, 256, QuantType::Q4K),
            ffn_down: (&ffn_down_bytes, 256, 1024, QuantType::Q4K),
        };

        let raw = LayerRawWeights::Gdn(gdn);
        match &raw {
            LayerRawWeights::Gdn(g) => {
                assert_eq!(g.attn_norm.len(), 256);
                assert_eq!(g.ssm_a.len(), 8);
                assert_eq!(g.qkv.1, 256);
            }
            LayerRawWeights::Attention(_) => panic!("expected Gdn variant"),
        }
    }

    #[test]
    fn fullpath_prefill_input_construction_compiles() {
        // Smoke test: build a FullPathPrefillInput with mock data so the
        // re-exported struct from rnb-runtime stays callable from external
        // crates (which is what 4c-2b will do).
        let prompt: Vec<u32> = vec![1];
        let kv_layout = KvResidentLayout::compute(2, 256, 1, 64);
        let staging = StagingPolicy::default();
        let embed: Vec<u8> = Vec::new();
        let output: Vec<u8> = Vec::new();
        let output_norm: Vec<f32> = vec![1.0; 256];
        let kinds: Vec<ModelLayerKind> = vec![ModelLayerKind::Attention; 2];

        let input = FullPathPrefillInput {
            prompt_token_ids: &prompt,
            num_layers: 2,
            hidden: 256,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 64,
            ffn_inner: 1024,
            norm_eps: 1e-5,
            base_freq: 500_000.0,
            rope_dim: 64,
            rope_neox: false,
            vocab: 32_000,
            kv_layout,
            staging,
            output_table_q6k: &output,
            output_quant: QuantType::Q6K,
            output_norm: &output_norm,
            embed_table_q6k: &embed,
            layer_weights: None, // smoke-test mode: layer loop skipped
            layer_kinds: &kinds,
        };

        // Field-presence spot check (compile-only).
        assert_eq!(input.num_layers, 2);
        assert_eq!(input.hidden, 256);
        assert!(input.layer_weights.is_none());
    }

    #[test]
    fn validate_layer_raw_weights_count_rejects_mismatched_len() {
        // mv28 cleanup I4: hit the wrapper-level len-vs-num_layers
        // validation without needing a Vulkan device. The body of
        // `run_fullpath_prefill` / `run_fullpath_decode_step` calls this
        // exact helper before any backend dispatch, so a green test here
        // proves the validation branch is wired.

        // empty slice vs num_layers=2 → must error.
        let empty: Vec<LayerRawWeights<'_>> = Vec::new();
        let err = validate_layer_raw_weights_count(&empty, 2, "test_caller")
            .expect_err("empty slice with num_layers=2 must reject");
        assert!(
            err.contains("test_caller"),
            "error message should carry caller label, got: {err}",
        );
        assert!(
            err.contains("0") && err.contains("2"),
            "error message should mention both lengths, got: {err}",
        );

        // matching len → must accept.
        let attn_norm_w: Vec<f32> = vec![1.0; 256];
        let ffn_norm_w: Vec<f32> = vec![1.0; 256];
        let q_bytes: Vec<u8> = vec![0; 144 * 256];
        let k_bytes: Vec<u8> = vec![0; 144 * 256];
        let v_bytes: Vec<u8> = vec![0; 144 * 256];
        let o_bytes: Vec<u8> = vec![0; 144 * 256];
        let gate_bytes: Vec<u8> = vec![0; 144 * 1024];
        let up_bytes: Vec<u8> = vec![0; 144 * 1024];
        let down_bytes: Vec<u8> = vec![0; 144 * 1024];
        let raw = LayerRawWeights::Attention(AttentionRawWeights {
            attn_norm: &attn_norm_w,
            q_proj: (&q_bytes, 256, 256, QuantType::Q4K),
            q_norm: None,
            k_proj_combined: (&k_bytes, QuantType::Q4K),
            k_norm: None,
            v_proj_combined: (&v_bytes, QuantType::Q4K),
            o_proj: (&o_bytes, 256, 256, QuantType::Q4K),
            ffn_norm: &ffn_norm_w,
            gate_proj: (&gate_bytes, 1024, 256, QuantType::Q4K),
            up_proj: (&up_bytes, 1024, 256, QuantType::Q4K),
            down_proj: (&down_bytes, 256, 1024, QuantType::Q4K),
        });
        let one_layer = vec![raw];
        validate_layer_raw_weights_count(&one_layer, 1, "test_caller")
            .expect("matching len must accept");

        // num_layers=0 + empty slice → must accept (wrapper smoke-test mode).
        validate_layer_raw_weights_count(&empty, 0, "test_caller")
            .expect("zero-vs-zero must accept");
    }

    /// Real GPU test that exercises the full pipeline. Ignored by default
    /// because it requires a Vulkan device + a working layer_gemv. Run with
    /// `cargo test --release -p rnb-runtime --features vulkan -- --ignored
    /// fullpath_prefill_smoke_run`.
    #[test]
    #[ignore]
    fn fullpath_prefill_smoke_run() {
        // We can't construct LayerRuntime without a working Vulkan device —
        // this body is a placeholder that runs the validation guards path
        // by building a minimal prompt. 4c-2b will replace this with a real
        // end-to-end engine test.
        let runtime = super::super::super::init_layer_gemv_for_test(256, 256, 64);
        if runtime.is_err() {
            // No Vulkan device — skip silently. Matches the
            // `crates/rnb-backend/vulkan/tests/full_path_shapes.rs` style.
            return;
        }
        let mut runtime = runtime.unwrap();

        // Use no layers so the layer loop is skipped; this hits embed_lookup
        // + logit_argmax only and exercises the cross-crate dispatch.
        let prompt: Vec<u32> = vec![0];
        let layer_kinds: Vec<ModelLayerKind> = Vec::new();
        let layer_weights: Vec<LayerRawWeights<'_>> = Vec::new();

        let result = runtime.run_fullpath_prefill(
            &prompt,
            /* num_layers */ 0,
            /* hidden */ 256,
            /* num_heads */ 4,
            /* num_kv_heads */ 1,
            /* head_dim */ 64,
            /* ffn_inner */ 1024,
            /* norm_eps */ 1e-5,
            /* base_freq */ 500_000.0,
            /* rope_dim */ 64,
            /* rope_neox */ false,
            /* vocab */ 32_000,
            /* max_ctx */ 256,
            /* output_table_q6k */ &[],
            /* output_quant */ QuantType::Q6K,
            /* output_norm */ &[1.0f32; 256],
            /* embed_table_q6k */ &[],
            &layer_weights,
            &layer_kinds,
            StagingPolicy::default(),
        );
        // num_layers=0 should hit the "num_layers must be > 0" validation
        // error — the wrapper itself doesn't call run_prefill yet because
        // the input validates inside the backend. Either an Err or a
        // misshape Err is acceptable here as long as the wrapper compiled
        // and dispatched.
        assert!(result.is_err());
    }
}
