// Engine-side dispatcher for the persistent cooperative decode kernel
// (Gemma4 E2B).  This is the cu74 product wire entry point.
//
// Eligibility (single-token decode only, Gemma4 E2B, env opt-in):
//   - architecture == Gemma4 && hidden_dim == 1536 && ple_dim == 256
//   - RNB_CUDA_PERSISTENT_DECODE=1
//   - all 35 layers are LayerType::Attention (no GDN/Mamba2/MoE)
//
// On eligible paths, `try_persistent_decode_dispatch` collects every
// per-layer host weight slice, KV cache bytes, PLE inputs, and the input
// hidden state into a `PersistentDecodeRequest` and invokes
// `cuda_runtime::dispatch_persistent_decode`.  On success it returns the
// `argmax` token id; on any setup failure (eligibility, missing weight,
// dispatch error) it returns `Ok(None)` so the caller falls back to the
// eager decode path.

use super::layer_weights::{AttentionLayerWeights, LayerType, ModelWeights};
use super::types::ModelMetadata;
use crate::engine::compute_runtime::{
    PersistentDecodeLayerInput, PersistentDecodeRequest, PERSISTENT_DECODE_FLAG_DOWN_Q6K,
    PERSISTENT_DECODE_FLAG_K_Q6K, PERSISTENT_DECODE_FLAG_O_Q6K, PERSISTENT_DECODE_FLAG_PLE_F32,
    PERSISTENT_DECODE_FLAG_REUSE_Q, PERSISTENT_DECODE_FLAG_V_Q6K,
};
use crate::engine::cuda_runtime;
use crate::engine::models::gemma::{
    active_sliding_window, gemma_per_layer_enabled_for_model, prepare_gemma_per_layer_base,
    shared_kv_source_layer,
};
use crate::kv_cache::KVCache;
use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;
use rnb_loader::GGMLType;

/// Returns `Ok(Some(token_id))` once cu74+ has wired the full pointer
/// collection.  Returns `Ok(None)` to fall back to the eager dense-chain
/// path on any eligibility miss (the eager path is the production default
/// until token-by-token correctness is verified).
pub(super) fn try_persistent_decode_dispatch(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    kv_cache: &mut KVCache,
    input_hidden: &[f32],
    rope_pos: usize,
    input_token: u32,
    out_logits: Option<&mut [f32]>,
) -> Result<Option<i32>, String> {
    // cu101 wrapper — single-token decode forwards as a batch dispatch with
    // seq_len=1 / batch_tokens=&[input_token]. Allows reuse of the same body
    // for both decode and batch prefill callers.
    try_persistent_decode_dispatch_batch(
        metadata,
        architecture,
        weights,
        kv_cache,
        input_hidden,
        rope_pos,
        std::slice::from_ref(&input_token),
        out_logits,
    )
}

/// cu101 Milestone 2 — batch prefill entry. `input_tokens.len()` == seq_len.
/// `input_hidden.len()` must equal `seq_len * hidden_dim` (caller packs the
/// per-token embeddings consecutively after applying embedding_scale). The
/// underlying body branches the per-layer `ple_input` slice between single-
/// token and batch (seq_len * ple_dim) layouts.
pub(super) fn try_persistent_decode_dispatch_batch(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
    kv_cache: &mut KVCache,
    input_hidden: &[f32],
    rope_pos: usize,
    input_tokens: &[u32],
    out_logits: Option<&mut [f32]>,
) -> Result<Option<i32>, String> {
    let input_token = *input_tokens
        .first()
        .ok_or_else(|| "try_persistent_decode_dispatch_batch: input_tokens empty".to_string())?;
    let _ = input_token; // unused in batch body
                         // cu101 diag: trace eligibility/check failures so the batch caller can see
                         // why dispatch returns None.
    let _trace_eligible =
        crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_PREFILL_BATCH_TRACE").is_some();
    macro_rules! pp_trace {
        ($($arg:tt)*) => {
            if _trace_eligible {
                eprintln!("[cu101-batch-disp] {}", format!($($arg)*));
            }
        };
    }
    pp_trace!(
        "entry seq_len={} rope_pos={} input_hidden.len()={} hidden_dim={}",
        input_tokens.len(),
        rope_pos,
        input_hidden.len(),
        metadata.hidden_dim,
    );
    if !is_persistent_decode_eligible(metadata, architecture, weights) {
        pp_trace!("not eligible: persistent_decode_enabled={} arch={:?} hidden_dim={} ple_dim={} layers={}",
            cuda_runtime::persistent_decode_enabled(),
            architecture,
            metadata.hidden_dim,
            metadata.embedding_length_per_layer_input,
            weights.layers.len());
        return Ok(None);
    }
    if input_hidden.len() != metadata.hidden_dim * input_tokens.len() {
        pp_trace!(
            "input_hidden length mismatch: got {}, expected {} * {} = {}",
            input_hidden.len(),
            metadata.hidden_dim,
            input_tokens.len(),
            metadata.hidden_dim * input_tokens.len()
        );
        return Ok(None);
    }
    // cu75: compute per-token PLE base (matches eager path at decode_inference.rs:337).
    // Without this, persistent kernel skips PLE entirely and decodes diverge silently.
    // cu101: batch mode — when input_tokens.len() > 1, prepare_gemma_per_layer_base
    // produces `base.mixed` of length `seq_len * num_layers * ple_dim` (per-token,
    // per-layer). The per-layer slice extraction below picks the right slab.
    let seq_len_batch = input_tokens.len();
    let ple_dim = metadata.embedding_length_per_layer_input;
    let gemma_ple_base = if gemma_per_layer_enabled_for_model(weights, metadata, architecture) {
        let hidden_tensor = Tensor::from_slice(input_hidden, &[seq_len_batch, metadata.hidden_dim]);
        prepare_gemma_per_layer_base(
            weights,
            &hidden_tensor,
            input_tokens,
            metadata,
            architecture,
            metadata.norm_eps,
        )
        .map_err(|e| format!("PLE base prepare failed: {e}"))?
    } else {
        None
    };
    let num_layers = metadata.num_layers;
    let attention_layers: Vec<&AttentionLayerWeights> = weights
        .layers
        .iter()
        .filter_map(|l| match l {
            LayerType::Attention(a) => Some(a),
            _ => None,
        })
        .collect();
    if attention_layers.len() != num_layers {
        return Ok(None);
    }
    let max_seq_len = kv_cache.max_seq_len as u32;
    if max_seq_len == 0 {
        return Ok(None);
    }
    // cu101: kv_len_for_step now covers the last batch token's slot.
    let kv_len_for_step = (rope_pos as u32) + (seq_len_batch as u32);
    if kv_len_for_step > max_seq_len {
        return Ok(None);
    }

    // cu101: pre-build per-layer batch PLE input buffers when seq_len > 1.
    // base.mixed layout is [seq_len, num_layers, ple_dim]; we slice out
    // layer-major batch chunks of length seq_len * ple_dim per layer so the
    // kernel's token-loop can stride through them with `__t * ple_dim`.
    let total_ple_d = num_layers * ple_dim;
    let batch_ple_per_layer: Option<Vec<Vec<f32>>> = if seq_len_batch > 1 {
        gemma_ple_base.as_ref().map(|base| {
            (0..num_layers)
                .map(|layer_idx| {
                    let mut buf = Vec::with_capacity(seq_len_batch * ple_dim);
                    for t in 0..seq_len_batch {
                        let off = t * total_ple_d + layer_idx * ple_dim;
                        buf.extend_from_slice(&base.mixed[off..off + ple_dim]);
                    }
                    buf
                })
                .collect()
        })
    } else {
        None
    };
    // cu103 diag: trace batch PLE layer-0 token-0 vs base.mixed layer-0
    // to check whether the per-layer split layout matches what the kernel reads.
    if _trace_eligible {
        if let Some(ref bufs) = batch_ple_per_layer {
            let nan_count = bufs[0].iter().filter(|v| v.is_nan()).count();
            let inf_count = bufs[0].iter().filter(|v| v.is_infinite()).count();
            let mut max_abs: f32 = 0.0;
            for v in &bufs[0] {
                if !v.is_nan() && !v.is_infinite() && v.abs() > max_abs {
                    max_abs = v.abs();
                }
            }
            eprintln!(
                "[cu103-batch-disp] batch_ple L0 buf.len={} nan={nan_count} inf={inf_count} max_abs={max_abs:.4} first8={:?}",
                bufs[0].len(),
                &bufs[0][..8.min(bufs[0].len())]
            );
        }
        if let Some(ref base) = gemma_ple_base {
            eprintln!(
                "[cu103-batch-disp] base.mixed.len={} first8={:?}",
                base.mixed.len(),
                &base.mixed[..8.min(base.mixed.len())]
            );
        }
        // Also check input_hidden for NaN at token 0
        let mut hidden_nan = 0usize;
        let mut hidden_max: f32 = 0.0;
        for &v in &input_hidden[..metadata.hidden_dim] {
            if v.is_nan() {
                hidden_nan += 1;
            } else if v.abs() > hidden_max {
                hidden_max = v.abs();
            }
        }
        eprintln!("[cu103-batch-disp] input_hidden t0 nan={hidden_nan} max_abs={hidden_max:.4}");
    }
    let mut layer_inputs: Vec<PersistentDecodeLayerInput> = Vec::with_capacity(num_layers);
    let mut q_dim_max: u32 = 0;
    let mut kv_dim_max: u32 = 0;
    let mut n_ff_max: u32 = 0;

    let trace_enabled = crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_DECODE_TRACE")
        .as_deref()
        == Some("1");

    for (idx, layer) in attention_layers.iter().enumerate() {
        let Some(q_bytes) = layer.q_weight.data.as_bytes() else {
            return Ok(None);
        };
        let Some(k_bytes) = layer.k_weight.data.as_bytes() else {
            return Ok(None);
        };
        let Some(v_bytes) = layer.v_weight.data.as_bytes() else {
            return Ok(None);
        };
        let Some(o_bytes) = layer.o_weight.data.as_bytes() else {
            return Ok(None);
        };
        let Some(gate_bytes) = layer.ffn_gate_weight.data.as_bytes() else {
            return Ok(None);
        };
        let Some(up_bytes) = layer.ffn_up_weight.data.as_bytes() else {
            return Ok(None);
        };
        let Some(down_bytes) = layer.ffn_down_weight.data.as_bytes() else {
            return Ok(None);
        };
        let Some(attn_norm_bytes) = layer.attn_norm.as_bytes() else {
            return Ok(None);
        };
        if idx == 0 && trace_enabled {
            let n = attn_norm_bytes.len().min(16);
            let mut probe_f32 = [0.0f32; 4];
            for i in 0..(n / 4).min(4) {
                let b = [
                    attn_norm_bytes[i * 4],
                    attn_norm_bytes[i * 4 + 1],
                    attn_norm_bytes[i * 4 + 2],
                    attn_norm_bytes[i * 4 + 3],
                ];
                probe_f32[i] = f32::from_le_bytes(b);
            }
            // also interpret as bf16 (2 bytes / element) to detect type confusion
            let mut probe_bf16 = [0.0f32; 4];
            for i in 0..(n / 2).min(4) {
                let bits = u16::from_le_bytes([attn_norm_bytes[i * 2], attn_norm_bytes[i * 2 + 1]]);
                // bf16 -> f32: shift left 16
                probe_bf16[i] = f32::from_bits((bits as u32) << 16);
            }
            eprintln!(
                "[cu76 attn_norm probe] layer 0 dtype={:?} len_bytes={} as_f32={:?} as_bf16={:?}",
                layer.attn_norm.dtype(),
                attn_norm_bytes.len(),
                probe_f32,
                probe_bf16,
            );
        }
        let Some(ffn_norm_bytes) = layer.ffn_norm.as_bytes() else {
            return Ok(None);
        };
        let post_attn_norm = layer.post_attn_norm.as_ref().and_then(|t| t.as_bytes());
        let post_ffn_norm = layer.post_ffw_norm.as_ref().and_then(|t| t.as_bytes());
        let q_norm = layer.q_norm.as_ref().and_then(|t| t.as_bytes());
        let k_norm = layer.k_norm.as_ref().and_then(|t| t.as_bytes());

        let (ple_gate, ple_proj, ple_post_norm, ple_is_f32) = collect_ple_for_layer(weights, idx);
        // cu101: select per-layer batch buffer (length seq_len * ple_dim) when
        // we're in batch mode, otherwise fall back to the single-token base
        // slice (length ple_dim).
        let ple_input_slice: Option<&[f32]> = if let Some(ref bufs) = batch_ple_per_layer {
            bufs.get(idx).map(|b| b.as_slice())
        } else {
            gemma_ple_base.as_ref().and_then(|base| {
                let off = idx * ple_dim;
                base.mixed.get(off..off + ple_dim)
            })
        };

        let k_cache_slice = kv_cache.get_key(idx);
        let v_cache_slice = kv_cache.get_value(idx);
        if idx == 0 && trace_enabled && rope_pos < 12 {
            let k_first = &k_cache_slice[..16.min(k_cache_slice.len())];
            let v_first = &v_cache_slice[..16.min(v_cache_slice.len())];
            let mut nonzero_k = 0usize;
            for v in k_cache_slice.iter() {
                if *v != 0 {
                    nonzero_k += 1;
                }
            }
            eprintln!(
                "[cu77 kv layer0 rope_pos={rope_pos}] k_first={:?} v_first={:?} k_len={} nonzero_k={nonzero_k}",
                k_first,
                v_first,
                k_cache_slice.len(),
            );
        }
        let (k_cache_ptr, k_cache_len) = u16_slice_as_bytes(k_cache_slice);
        let (v_cache_ptr, v_cache_len) = u16_slice_as_bytes(v_cache_slice);

        let head_dim: u32 = if active_sliding_window(metadata, architecture, idx).is_some() {
            256
        } else {
            512
        };
        let kv_dim = kv_cache.layer_kv_dim(idx) as u32;
        let q_heads = metadata.num_heads as u32;
        let q_dim = q_heads * head_dim;
        let n_ff = layer.ffn_gate_weight.rows as u32;
        q_dim_max = q_dim_max.max(q_dim);
        kv_dim_max = kv_dim_max.max(kv_dim);
        n_ff_max = n_ff_max.max(n_ff);

        let kv_source = shared_kv_source_layer(metadata, architecture, idx).unwrap_or(idx);
        let mut flags: u32 = 0;
        if kv_source != idx {
            flags |= PERSISTENT_DECODE_FLAG_REUSE_Q;
        }
        if ple_is_f32 {
            flags |= PERSISTENT_DECODE_FLAG_PLE_F32;
        }
        // Quant-kind flags: Gemma4 E2B mixes Q4_K (Q/K/gate/up/output) and
        // Q6_K (V, down, sometimes O) projections.  Signal Q6_K to the
        // cooperative kernel so it routes through the Q6_K GEMV device fn.
        if matches!(layer.v_weight.ggml_type, GGMLType::Q6_K) {
            flags |= PERSISTENT_DECODE_FLAG_V_Q6K;
        }
        if matches!(layer.ffn_down_weight.ggml_type, GGMLType::Q6_K) {
            flags |= PERSISTENT_DECODE_FLAG_DOWN_Q6K;
        }
        if matches!(layer.o_weight.ggml_type, GGMLType::Q6_K) {
            flags |= PERSISTENT_DECODE_FLAG_O_Q6K;
        }
        if matches!(layer.k_weight.ggml_type, GGMLType::Q6_K) {
            flags |= PERSISTENT_DECODE_FLAG_K_Q6K;
        }

        let layer_output_scale = layer
            .out_scale
            .as_ref()
            .and_then(|t| t.as_bytes())
            .and_then(|b| {
                if b.len() >= 4 {
                    let mut buf = [0u8; 4];
                    buf.copy_from_slice(&b[..4]);
                    Some(f32::from_le_bytes(buf))
                } else {
                    None
                }
            })
            .unwrap_or(1.0);
        if idx < 3 && trace_enabled {
            eprintln!(
                "[cu76 layer{idx} out_scale]={layer_output_scale} has_out_scale={}",
                layer.out_scale.is_some(),
            );
            eprintln!(
                "[cu77 layer{idx} types] q={:?} k={:?} v={:?} o={:?} gate={:?} up={:?} down={:?}",
                layer.q_weight.ggml_type,
                layer.k_weight.ggml_type,
                layer.v_weight.ggml_type,
                layer.o_weight.ggml_type,
                layer.ffn_gate_weight.ggml_type,
                layer.ffn_up_weight.ggml_type,
                layer.ffn_down_weight.ggml_type,
            );
            eprintln!(
                "[cu77 layer{idx} dims] q_w={}x{} k_w={}x{} v_w={}x{} o_w={}x{} gate_w={}x{} dispatch q_dim={} kv_dim={} head_dim={} sliding={}",
                layer.q_weight.rows, layer.q_weight.cols,
                layer.k_weight.rows, layer.k_weight.cols,
                layer.v_weight.rows, layer.v_weight.cols,
                layer.o_weight.rows, layer.o_weight.cols,
                layer.ffn_gate_weight.rows, layer.ffn_gate_weight.cols,
                q_dim, kv_dim, head_dim,
                active_sliding_window(metadata, architecture, idx).map(|w| w as u32).unwrap_or(0),
            );
        }

        let sliding_window = active_sliding_window(metadata, architecture, idx)
            .map(|w| w as u32)
            .unwrap_or(0);

        layer_inputs.push(PersistentDecodeLayerInput {
            q_weight_bytes: q_bytes.as_ptr(),
            q_weight_len: q_bytes.len(),
            k_weight_bytes: k_bytes.as_ptr(),
            k_weight_len: k_bytes.len(),
            v_weight_bytes: v_bytes.as_ptr(),
            v_weight_len: v_bytes.len(),
            o_weight_bytes: o_bytes.as_ptr(),
            o_weight_len: o_bytes.len(),
            gate_weight_bytes: gate_bytes.as_ptr(),
            gate_weight_len: gate_bytes.len(),
            up_weight_bytes: up_bytes.as_ptr(),
            up_weight_len: up_bytes.len(),
            down_weight_bytes: down_bytes.as_ptr(),
            down_weight_len: down_bytes.len(),
            attn_norm_bytes: attn_norm_bytes.as_ptr(),
            attn_norm_len: attn_norm_bytes.len(),
            post_attn_norm_bytes: post_attn_norm
                .map(|s| s.as_ptr())
                .unwrap_or(std::ptr::null()),
            post_attn_norm_len: post_attn_norm.map(|s| s.len()).unwrap_or(0),
            ffn_norm_bytes: ffn_norm_bytes.as_ptr(),
            ffn_norm_len: ffn_norm_bytes.len(),
            post_ffn_norm_bytes: post_ffn_norm
                .map(|s| s.as_ptr())
                .unwrap_or(std::ptr::null()),
            post_ffn_norm_len: post_ffn_norm.map(|s| s.len()).unwrap_or(0),
            q_norm_bytes: q_norm.map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
            q_norm_len: q_norm.map(|s| s.len()).unwrap_or(0),
            k_norm_bytes: k_norm.map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
            k_norm_len: k_norm.map(|s| s.len()).unwrap_or(0),
            ple_gate_bytes: ple_gate.map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
            ple_gate_len: ple_gate.map(|s| s.len()).unwrap_or(0),
            ple_proj_bytes: ple_proj.map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
            ple_proj_len: ple_proj.map(|s| s.len()).unwrap_or(0),
            ple_post_norm_bytes: ple_post_norm
                .map(|s| s.as_ptr())
                .unwrap_or(std::ptr::null()),
            ple_post_norm_len: ple_post_norm.map(|s| s.len()).unwrap_or(0),
            // cu75: per-layer PLE input slice from host-computed `mixed` base.
            // bytes view (4 bytes/f32 × ple_dim).
            ple_input_bytes: ple_input_slice
                .map(|s| s.as_ptr() as *const u8)
                .unwrap_or(std::ptr::null()),
            ple_input_len: ple_input_slice.map(|s| s.len() * 4).unwrap_or(0),
            k_cache_bytes: k_cache_ptr,
            k_cache_len,
            v_cache_bytes: v_cache_ptr,
            v_cache_len,
            head_dim,
            q_dim,
            kv_dim,
            n_ff,
            sliding_window,
            kv_source_layer: kv_source as u32,
            layer_output_scale,
            flags,
        });
    }

    if q_dim_max == 0 || kv_dim_max == 0 || n_ff_max == 0 {
        return Ok(None);
    }

    let output_bytes = weights.output.data.as_bytes();
    let (output_ptr, output_len) = match output_bytes {
        Some(b) => (b.as_ptr(), b.len()),
        None => (std::ptr::null(), 0),
    };

    let vocab_size = metadata.vocab_size as u32;
    let mut output_logits = vec![0.0f32; vocab_size as usize];
    let mut argmax_out: i32 = -1;

    // cu76 diag: when RNB_CUDA_PERSISTENT_DECODE_PROBE=1, allocate hidden_probe
    // host buffer.  Combined with RNB_CUDA_PERSISTENT_DECODE_LAYERS=N this lets
    // us snapshot the hidden state right after the last active layer.
    let probe_enabled = crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_DECODE_PROBE")
        .as_deref()
        == Some("1");
    let mut probe_buf: Vec<f32> = if probe_enabled {
        vec![0.0f32; metadata.hidden_dim]
    } else {
        Vec::new()
    };
    // cu76 phase probes (layer-0 only).
    let phase_probe_enabled =
        crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_DECODE_PHASE_PROBE").as_deref()
            == Some("1");
    let mut normed_after_attn_norm_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.hidden_dim]
    } else {
        Vec::new()
    };
    let mut hidden_after_attn_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.hidden_dim]
    } else {
        Vec::new()
    };
    let mut hidden_after_ffn_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.hidden_dim]
    } else {
        Vec::new()
    };
    // cu78 attn-out / Q / K head-0 head_dim buffers (use metadata.head_dim).
    let mut attn_out_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.head_dim]
    } else {
        Vec::new()
    };
    let mut q_proj_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.head_dim]
    } else {
        Vec::new()
    };
    let mut k_proj_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.head_dim]
    } else {
        Vec::new()
    };
    let mut v_proj_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.head_dim]
    } else {
        Vec::new()
    };
    let mut attn_scores_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; kv_len_for_step as usize]
    } else {
        Vec::new()
    };
    let mut attn_v_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.head_dim]
    } else {
        Vec::new()
    };
    let mut attn_acc_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.head_dim]
    } else {
        Vec::new()
    };
    let mut attn_row_sum_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; 1]
    } else {
        Vec::new()
    };
    let mut ffn_gate_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; 1024]
    } else {
        Vec::new()
    };
    let mut ffn_gated_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; 1024]
    } else {
        Vec::new()
    };
    let mut ffn_down_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.hidden_dim]
    } else {
        Vec::new()
    };
    let mut layer_hidden_trace_buf: Vec<f32> = if phase_probe_enabled {
        vec![0.0f32; metadata.num_layers]
    } else {
        Vec::new()
    };

    // cu74 NaN-isolation: optional env to limit number of layers processed,
    // so we can tell whether the kernel produces NaN on a single layer or
    // whether the divergence is an accumulation across all 35 layers.
    let layer_cap = crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_DECODE_LAYERS")
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| (n as usize) <= layer_inputs.len())
        .unwrap_or(num_layers as u32);
    let active_layer_inputs = &layer_inputs[..layer_cap as usize];

    // cu77: Gemma4 RoPE freq_factors (FULL-attention only). bytes view (f32).
    let rope_freqs_info = weights.rope_freqs.as_ref().map(|t| {
        (
            t.dtype(),
            t.numel(),
            t.as_bytes().map(|s| (s.as_ptr(), s.len())),
        )
    });
    if crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_DECODE_TRACE").as_deref() == Some("1")
    {
        eprintln!("[cu77 rope_freqs info] {:?}", rope_freqs_info);
    }
    let rope_freqs_enabled =
        crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_ROPE_FREQS").as_deref() != Some("0");
    let (output_norm_bytes, output_norm_len) = {
        let bytes = rnb_core::tensor::Tensor::as_bytes(&weights.output_norm);
        bytes
            .map(|s| (s.as_ptr(), s.len()))
            .unwrap_or((std::ptr::null(), 0))
    };
    let (rope_freqs_bytes, rope_freqs_len) = if rope_freqs_enabled {
        weights
            .rope_freqs
            .as_ref()
            .filter(|t| t.dtype() == rnb_core::tensor::DType::F32)
            .and_then(|t| t.as_bytes())
            .filter(|s| s.len() >= (metadata.head_dim / 2) * 4)
            .map(|s| (s.as_ptr(), s.len()))
            .unwrap_or((std::ptr::null(), 0))
    } else {
        (std::ptr::null(), 0)
    };

    let mut request = PersistentDecodeRequest {
        num_layers: layer_cap,
        hidden_dim: metadata.hidden_dim as u32,
        vocab_size,
        norm_eps: metadata.norm_eps,
        rope_pos: rope_pos as u32,
        kv_len: kv_len_for_step,
        max_seq_len,
        q_dim_max,
        kv_dim_max,
        n_ff_max,
        ple_dim: metadata.embedding_length_per_layer_input as u32,
        layers: active_layer_inputs,
        output_weight_bytes: output_ptr,
        output_weight_len: output_len,
        // cu101: pass batch seq_len through to the kernel. Decode = 1.
        seq_len: seq_len_batch as u32,
        input_hidden,
        output_logits: &mut output_logits,
        argmax_out: &mut argmax_out,
        hidden_probe: if probe_enabled {
            Some(probe_buf.as_mut_slice())
        } else {
            None
        },
        normed_after_attn_norm_probe: if phase_probe_enabled {
            Some(normed_after_attn_norm_buf.as_mut_slice())
        } else {
            None
        },
        hidden_after_attn_probe: if phase_probe_enabled {
            Some(hidden_after_attn_buf.as_mut_slice())
        } else {
            None
        },
        hidden_after_ffn_probe: if phase_probe_enabled {
            Some(hidden_after_ffn_buf.as_mut_slice())
        } else {
            None
        },
        rope_freqs_bytes,
        rope_freqs_len,
        attn_out_probe: if phase_probe_enabled {
            Some(attn_out_buf.as_mut_slice())
        } else {
            None
        },
        q_proj_probe: if phase_probe_enabled {
            Some(q_proj_buf.as_mut_slice())
        } else {
            None
        },
        k_proj_probe: if phase_probe_enabled {
            Some(k_proj_buf.as_mut_slice())
        } else {
            None
        },
        v_proj_probe: if phase_probe_enabled {
            Some(v_proj_buf.as_mut_slice())
        } else {
            None
        },
        attn_scores_probe: if phase_probe_enabled {
            Some(attn_scores_buf.as_mut_slice())
        } else {
            None
        },
        attn_v_probe: if phase_probe_enabled {
            Some(attn_v_buf.as_mut_slice())
        } else {
            None
        },
        attn_acc_probe: if phase_probe_enabled {
            Some(attn_acc_buf.as_mut_slice())
        } else {
            None
        },
        attn_row_sum_probe: if phase_probe_enabled {
            Some(attn_row_sum_buf.as_mut_slice())
        } else {
            None
        },
        hidden_after_ffn_only_probe: None,
        ffn_gate_probe: if phase_probe_enabled {
            Some(ffn_gate_buf.as_mut_slice())
        } else {
            None
        },
        ffn_gated_probe: if phase_probe_enabled {
            Some(ffn_gated_buf.as_mut_slice())
        } else {
            None
        },
        ffn_down_probe: if phase_probe_enabled {
            Some(ffn_down_buf.as_mut_slice())
        } else {
            None
        },
        layer_hidden_trace: if phase_probe_enabled {
            Some(layer_hidden_trace_buf.as_mut_slice())
        } else {
            None
        },
        output_norm_bytes,
        output_norm_len,
    };

    cuda_runtime::dispatch_persistent_decode(&mut request)?;

    if probe_enabled {
        let nan_count = probe_buf.iter().filter(|v| v.is_nan()).count();
        let max_abs = probe_buf
            .iter()
            .filter(|v| !v.is_nan())
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let mean =
            probe_buf.iter().filter(|v| !v.is_nan()).sum::<f32>() / probe_buf.len().max(1) as f32;
        eprintln!(
            "[cu76 probe] persistent hidden after layer_cap={} rope_pos={} mean={:.6} max_abs={:.6} nan={} first8={:?}",
            layer_cap,
            rope_pos,
            mean,
            max_abs,
            nan_count,
            &probe_buf[..8.min(probe_buf.len())],
        );
        // write to env-specified file for offline diff with eager snapshot
        if let Some(path) =
            crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_DECODE_PROBE_OUT")
        {
            let bytes: Vec<u8> = probe_buf.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("[cu76 probe] failed to write {path}: {e}");
            }
        }
    }

    if phase_probe_enabled {
        // cu83-84: find max idx in each probe to detect alignment artifacts.
        let max_of = |buf: &[f32]| -> Option<(usize, f32)> {
            buf.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .map(|(i, v)| (i, *v))
        };
        eprintln!(
            "[cu84 max_idx] attn_out={:?} attn_acc={:?} normed={:?} hidden_after_attn={:?} hidden_after_ffn={:?} hidden_final={:?}",
            max_of(&attn_out_buf),
            max_of(&attn_acc_buf),
            max_of(&normed_after_attn_norm_buf),
            max_of(&hidden_after_attn_buf),
            max_of(&hidden_after_ffn_buf),
            max_of(&probe_buf),
        );
        let probes = [
            ("normed_after_attn_norm", &normed_after_attn_norm_buf),
            ("hidden_after_attn", &hidden_after_attn_buf),
            ("hidden_after_ffn", &hidden_after_ffn_buf),
            ("attn_out", &attn_out_buf),
            ("q_proj", &q_proj_buf),
            ("k_proj", &k_proj_buf),
            ("v_proj", &v_proj_buf),
            ("attn_scores", &attn_scores_buf),
            ("attn_v_at_j0", &attn_v_buf),
            ("attn_acc", &attn_acc_buf),
            ("attn_row_sum", &attn_row_sum_buf),
            ("ffn_gate", &ffn_gate_buf),
            ("ffn_gated", &ffn_gated_buf),
            ("ffn_down", &ffn_down_buf),
        ];
        // cu90: print per-layer hidden trace separately (compact format).
        if !layer_hidden_trace_buf.is_empty() {
            let trace_str: Vec<String> = layer_hidden_trace_buf
                .iter()
                .enumerate()
                .map(|(i, v)| format!("L{i}={v:.2}"))
                .collect();
            eprintln!("[cu90 layer_trace] {}", trace_str.join(" "));
        }
        for (name, buf) in probes {
            if buf.is_empty() {
                continue;
            }
            let mean = buf.iter().sum::<f32>() / buf.len() as f32;
            let max_abs = buf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "[cu76 phase] {name} mean={:.6} max_abs={:.6} first8={:?}",
                mean,
                max_abs,
                &buf[..8.min(buf.len())],
            );
            if let Some(prefix) =
                crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_DECODE_PHASE_OUT_PREFIX")
            {
                let path = format!("{prefix}_{name}.bin");
                let bytes: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
                if let Err(e) = std::fs::write(&path, bytes) {
                    eprintln!("[cu76 phase] failed to write {path}: {e}");
                }
            }
        }
    }

    if trace_enabled {
        let mut sample_max = f32::NEG_INFINITY;
        let mut sample_min = f32::INFINITY;
        let mut sample_nan = 0usize;
        let mut sample_neginf = 0usize;
        for &v in output_logits.iter() {
            if v.is_nan() {
                sample_nan += 1;
            } else if v == f32::NEG_INFINITY {
                sample_neginf += 1;
            } else {
                if v > sample_max {
                    sample_max = v;
                }
                if v < sample_min {
                    sample_min = v;
                }
            }
        }
        let in_sumsq: f32 = input_hidden.iter().map(|v| v * v).sum();
        eprintln!(
            "[cu74 persistent-decode] dispatched rope_pos={rope_pos} kv_len={kv_len_for_step} argmax={argmax_out} \
             logits_range=[{sample_min:.4}, {sample_max:.4}] nan={sample_nan} neginf={sample_neginf} \
             output_weight_len={output_len} input_hidden_range=[{:.4}, {:.4}] \
             input_first8={:?} input_sumsq={:.6}",
            input_hidden.iter().fold(f32::INFINITY, |a, b| a.min(*b)),
            input_hidden.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b)),
            &input_hidden[..8.min(input_hidden.len())],
            in_sumsq,
        );
    }

    if argmax_out < 0 {
        if crate::engine::policy::env_string("RNB_CUDA_PERSISTENT_PREFILL_BATCH_TRACE").is_some() {
            eprintln!(
                "[cu101-batch-disp] argmax_out negative ({argmax_out}) after kernel dispatch — kernel didn't write argmax_dev"
            );
        }
        return Ok(None);
    }
    // cu94 Milestone 0: write logits back to caller's buffer so non-argmax-only
    // callers (e.g. token-by-token prefill loop, sampler chain) see real logits
    // instead of a stale all-zero scratch slot.
    if let Some(dst) = out_logits {
        let n = dst.len().min(output_logits.len());
        dst[..n].copy_from_slice(&output_logits[..n]);
    }
    Ok(Some(argmax_out))
}

fn is_persistent_decode_eligible(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    weights: &ModelWeights,
) -> bool {
    if !cuda_runtime::persistent_decode_enabled() {
        return false;
    }
    if !matches!(architecture, ModelArchitecture::Gemma4) {
        return false;
    }
    if metadata.hidden_dim != 1536 {
        return false;
    }
    if metadata.embedding_length_per_layer_input != 256 {
        return false;
    }
    if weights.layers.len() != metadata.num_layers {
        return false;
    }
    weights
        .layers
        .iter()
        .all(|l| matches!(l, LayerType::Attention(_)))
}

fn u16_slice_as_bytes(s: &[u16]) -> (*const u8, usize) {
    // cu95 fix: do NOT return null on empty slice. The persistent decode
    // runtime needs the *backing buffer's* address to perform the D2H
    // writeback for the new token. KVCache.get_key returns &key[..current_len],
    // and for `current_len == 0` the slice is empty but the underlying Vec is
    // already allocated to `max_seq_len * stride`. Rust guarantees that an
    // empty slice taken from `&vec[..0]` carries the Vec's base pointer, so
    // returning it as the writeback target lands the new token's K/V at
    // offset 0 of the real allocation. The length is still 0 (bookkeeping),
    // but the runtime writeback code now sees a non-null ptr.
    (
        s.as_ptr() as *const u8,
        s.len() * std::mem::size_of::<u16>(),
    )
}

fn collect_ple_for_layer(
    weights: &ModelWeights,
    layer_idx: usize,
) -> (Option<&[u8]>, Option<&[u8]>, Option<&[u8]>, bool) {
    let Some(gemma) = weights.gemma_per_layer.as_ref() else {
        return (None, None, None, false);
    };
    let Some(layer) = gemma.layers.get(layer_idx) else {
        return (None, None, None, false);
    };
    let ple_gate = layer.inp_gate.data.as_bytes();
    let ple_proj = layer.proj.data.as_bytes();
    let ple_post_norm = layer.post_norm.as_bytes();
    let ple_is_f32 = matches!(layer.inp_gate.ggml_type, GGMLType::F32);
    (ple_gate, ple_proj, ple_post_norm, ple_is_f32)
}
