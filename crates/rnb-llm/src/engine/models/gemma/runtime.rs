use super::*;

pub(in crate::engine) fn use_gemma_block_semantics(architecture: ModelArchitecture) -> bool {
    // Step 2 of the Gemma4 rebuild: treat Gemma4 as Gemma-family so all existing
    // Gemma-gated seams (SWA, shared KV, QK norm, PLE, softcap, …) stay active.
    // Step 3 will peel out a dedicated Gemma4 forward path and this match can narrow again.
    matches!(
        architecture,
        ModelArchitecture::Gemma | ModelArchitecture::Gemma4
    )
}

pub(in crate::engine) fn gemma_per_layer_enabled_for_model(
    weights: &ModelWeights,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
) -> bool {
    !env_flag("RNB_DISABLE_GEMMA_PLE")
        && use_gemma_block_semantics(architecture)
        && metadata.embedding_length_per_layer_input > 0
        && weights.gemma_per_layer.is_some()
}

pub(in crate::engine) fn detect_gemma_runtime_flavor(
    metadata: &ModelMetadata,
    weights: &ModelWeights,
) -> GemmaRuntimeFlavor {
    if metadata.num_layers == 35
        && metadata.hidden_dim == 1536
        && metadata.num_heads == 8
        && metadata.num_kv_heads == 1
        && metadata.head_dim == 512
        && metadata.embedding_length_per_layer_input == 256
        && weights.gemma_per_layer.is_some()
    {
        GemmaRuntimeFlavor::Gemma4E2BIt
    } else {
        GemmaRuntimeFlavor::Generic
    }
}

pub(in crate::engine) fn is_gemma_sliding_window_layer(
    metadata: &ModelMetadata,
    layer_idx: usize,
) -> bool {
    if env_flag("RNB_DISABLE_GEMMA_SWA") {
        return false;
    }
    if env_flag("RNB_FORCE_GEMMA_ALL_SWA") {
        return true;
    }
    metadata
        .sliding_window_pattern
        .get(layer_idx)
        .copied()
        .unwrap_or(false)
}

pub(in crate::engine) fn active_sliding_window(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    layer_idx: usize,
) -> Option<usize> {
    if use_gemma_block_semantics(architecture)
        && metadata.sliding_window > 0
        && is_gemma_sliding_window_layer(metadata, layer_idx)
    {
        Some(metadata.sliding_window)
    } else {
        None
    }
}

pub(in crate::engine) fn shared_kv_source_layer(
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    layer_idx: usize,
) -> Option<usize> {
    if env_flag("RNB_DISABLE_GEMMA_SHARED_KV") {
        return None;
    }
    if !use_gemma_block_semantics(architecture) || metadata.shared_kv_layers == 0 {
        return None;
    }

    let kv_from_start = metadata
        .num_layers
        .saturating_sub(metadata.shared_kv_layers);
    if layer_idx < kv_from_start || kv_from_start == 0 {
        return None;
    }

    // Gemma4 (ISWA): on-device logits probes show SWA reused layers remain substantially more
    // stable when they anchor to layer 11, while full-attention reused layers still anchor to
    // layer 14. This is an empirical runtime correction for the current implementation.
    if matches!(architecture, ModelArchitecture::Gemma4) {
        let is_swa = is_gemma_sliding_window_layer(metadata, layer_idx);
        if (is_swa && env_flag("RNB_DISABLE_GEMMA_SHARED_KV_SWA"))
            || (!is_swa && env_flag("RNB_DISABLE_GEMMA_SHARED_KV_FULL"))
        {
            return None;
        }
        let override_key = if is_swa {
            "RNB_GEMMA_SHARED_KV_SOURCE_SWA"
        } else {
            "RNB_GEMMA_SHARED_KV_SOURCE_FULL"
        };
        if let Some(source) = env_usize(override_key) {
            return Some(source.min(kv_from_start.saturating_sub(1)));
        }
        if !is_swa && env_flag("RNB_GEMMA_SHARED_KV_FULL_PREV") {
            let mut prev = layer_idx.saturating_sub(1);
            while prev >= kv_from_start {
                if !is_gemma_sliding_window_layer(metadata, prev) {
                    return Some(prev);
                }
                if prev == 0 {
                    break;
                }
                prev -= 1;
            }
            return Some(kv_from_start.saturating_sub(1));
        }
        // HF Gemma4: SWA reused layers anchor to `kv_from_start - 2` (layer 13 when
        // kv_from_start=15) — the last SWA layer before the shared segment starts.
        // Full-attention reused layers anchor to `kv_from_start - 1` (layer 14).
        // Matches `L15.kv_shared_layer_index = 13` and `L19.kv_shared_layer_index = 14`
        // in the HF forward_hook dump. Session 50's empirical offset 3 (layer 12) was a
        // local optimum introduced because PLE and NEOX bugs were still masking the K
        // divergence; with those fixed, offset 2 (layer 13) matches HF.
        let offset = if is_swa { 2 } else { 1 };
        return Some(kv_from_start.saturating_sub(offset));
    }

    // Legacy Gemma family (Gemma3 / Gemma3n): wrap-around modulo.
    Some((layer_idx - kv_from_start) % kv_from_start)
}
pub(in crate::engine) fn gemma4_prefill_uses_f16_cache(architecture: ModelArchitecture) -> bool {
    matches!(architecture, ModelArchitecture::Gemma4)
}

pub(in crate::engine) fn select_ffn_pre_norm_weight<'a>(
    w: &'a AttentionLayerWeights,
    architecture: ModelArchitecture,
) -> &'a Tensor {
    if use_gemma_block_semantics(architecture) {
        &w.ffn_norm
    } else if let Some(ref pan) = w.post_attn_norm {
        pan
    } else {
        &w.ffn_norm
    }
}
