use super::layer_weights::AttentionLayerWeights;
use super::policy;
use super::types::ModelMetadata;

#[derive(Debug, Clone, Copy)]
pub(super) struct AttentionLayout {
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) q_dim: usize,
    pub(super) kv_dim: usize,
    pub(super) has_gated_attn: bool,
}

pub(super) fn resolve_attention_layout(
    metadata: &ModelMetadata,
    w: &AttentionLayerWeights,
    num_kv_heads_override: Option<usize>,
) -> crate::error::Result<AttentionLayout> {
    let num_heads = metadata.num_heads;
    let num_kv_heads = num_kv_heads_override.unwrap_or(metadata.num_kv_heads);
    let q_out_dim = w.q_weight.rows;
    let k_out_dim = w.k_weight.rows;
    let v_out_dim = w.v_weight.rows;

    let q_norm_dim = w.q_norm.as_ref().map(|t| t.numel());
    let k_norm_dim = w.k_norm.as_ref().map(|t| t.numel());

    let inferred_head_dim = q_norm_dim
        .or(k_norm_dim)
        .or_else(|| {
            if num_kv_heads > 0 && k_out_dim % num_kv_heads == 0 {
                Some(k_out_dim / num_kv_heads)
            } else {
                None
            }
        })
        .or_else(|| {
            if num_heads > 0 && q_out_dim % num_heads == 0 {
                Some(q_out_dim / num_heads)
            } else {
                None
            }
        })
        .unwrap_or(metadata.head_dim);

    let has_gated_attn = q_out_dim == num_heads * inferred_head_dim * 2;
    let q_dim = if has_gated_attn {
        q_out_dim / 2
    } else {
        q_out_dim
    };
    let kv_dim = k_out_dim;

    if num_heads == 0 || num_kv_heads == 0 {
        return Err(crate::error::LlmError::Forward(
            "attention layout: zero head count".into(),
        ));
    }
    if q_dim % num_heads != 0 {
        return Err(crate::error::LlmError::Forward(format!(
            "attention layout: q_dim {} not divisible by num_heads {}",
            q_dim, num_heads
        )));
    }
    if kv_dim % num_kv_heads != 0 {
        return Err(crate::error::LlmError::Forward(format!(
            "attention layout: kv_dim {} not divisible by num_kv_heads {}",
            kv_dim, num_kv_heads
        )));
    }

    let q_head_dim = q_dim / num_heads;
    let kv_head_dim = kv_dim / num_kv_heads;
    if policy::debug_gemma_layout_enabled() {
        eprintln!(
            "[layout] q_out={} k_out={} v_out={} q_norm={:?} k_norm={:?} inferred={} q_head={} kv_head={} gated={}",
            q_out_dim,
            k_out_dim,
            v_out_dim,
            q_norm_dim,
            k_norm_dim,
            inferred_head_dim,
            q_head_dim,
            kv_head_dim,
            has_gated_attn
        );
    }
    if q_head_dim != inferred_head_dim || kv_head_dim != inferred_head_dim {
        return Err(crate::error::LlmError::Forward(format!(
            "attention layout mismatch: inferred head_dim={}, q_head_dim={}, kv_head_dim={}",
            inferred_head_dim, q_head_dim, kv_head_dim
        )));
    }
    if k_out_dim != v_out_dim {
        return Err(crate::error::LlmError::Forward(format!(
            "attention layout mismatch: k_out_dim {} != v_out_dim {}",
            k_out_dim, v_out_dim
        )));
    }

    Ok(AttentionLayout {
        num_heads,
        num_kv_heads,
        head_dim: inferred_head_dim,
        q_dim,
        kv_dim,
        has_gated_attn,
    })
}

pub(super) fn resolve_attention_layout_gemma4_reuse(
    metadata: &ModelMetadata,
    w: &AttentionLayerWeights,
    num_kv_heads_override: Option<usize>,
) -> crate::error::Result<AttentionLayout> {
    let num_heads = metadata.num_heads;
    let num_kv_heads = num_kv_heads_override.unwrap_or(metadata.num_kv_heads);
    let q_out_dim = w.q_weight.rows;
    let q_norm_dim = w.q_norm.as_ref().map(|t| t.numel());
    let inferred_head_dim = q_norm_dim
        .or_else(|| {
            if num_heads > 0 && q_out_dim % num_heads == 0 {
                Some(q_out_dim / num_heads)
            } else {
                None
            }
        })
        .unwrap_or(metadata.head_dim);

    let has_gated_attn = q_out_dim == num_heads * inferred_head_dim * 2;
    let q_dim = if has_gated_attn {
        q_out_dim / 2
    } else {
        q_out_dim
    };
    let kv_dim = num_kv_heads * inferred_head_dim;

    if num_heads == 0 || num_kv_heads == 0 {
        return Err(crate::error::LlmError::Forward(
            "attention layout: zero head count".into(),
        ));
    }
    if q_dim % num_heads != 0 || kv_dim % num_kv_heads != 0 {
        return Err(crate::error::LlmError::Forward(
            "gemma4 reused attention layout mismatch".into(),
        ));
    }

    Ok(AttentionLayout {
        num_heads,
        num_kv_heads,
        head_dim: inferred_head_dim,
        q_dim,
        kv_dim,
        has_gated_attn,
    })
}
