pub mod deepseek4;
pub mod gemma;
pub mod llama;
pub mod muse_glimmer;
pub mod phi;

use crate::error::LoaderError;
use crate::gguf::metadata::{
    get_bool_array, get_bool_opt, get_f32, get_f32_array, get_f32_opt, get_string,
    get_string_array, get_u32, get_u32_array, get_u32_opt,
};
use crate::gguf::types::GGUFValue;
use crate::TokenizerData;
use rnb_core::ir::graph::Graph;
use rnb_model_nemotron::{
    classify_layer_from_tensor_names, decode_hybrid_pattern, NemotronLayerKind,
};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    LLaMA,
    Gemma,
    Gemma4,
    Gemma4Assistant,
    Phi,
    Qwen2,
    Qwen35,
    Qwen35MoE,
    NemotronHMoE,
    Hy3,
    GlmDsa,
    DeepSeek4,
    MuseGlimmer,
    DFlash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLayerKind {
    Attention,
    Recurrent,
    MoE,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpMetadata {
    pub total_block_count: usize,
    pub trunk_layers: usize,
    pub first_mtp_layer: usize,
    pub nextn_predict_layers: usize,
}

/// Gemma4 assistant (drafter) 모델 전용 metadata.
///
/// `general.architecture = "gemma4_assistant"` 인 GGUF 의 `gemma4_assistant.*` 키들을
/// 모은 묶음. 대상 (target) 모델 metadata 는 일반 `ModelMetadata` 로 따로 들고,
/// drafter 가 verify 와 KV 를 공유할 때 이 구조로 호환성과 잠재 토큰 수,
/// shared-KV layer 경계 등을 확인한다.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMetadata {
    pub n_centroids: u32,
    pub centroid_top_k: u32,
    pub n_embd_backbone: u32,
    pub use_ordered_embeddings: bool,
    pub requires_target_arch: String,
    pub shared_kv_layers: u32,
    pub sliding_window_pattern: Vec<bool>,
    pub key_length_full: u32,
    pub key_length_swa: u32,
    pub rope_freq_base_full: f32,
    pub rope_freq_base_swa: f32,
    /// SWA window size in tokens (`gemma4_assistant.attention.sliding_window` = 512).
    /// drafter cross-attention 의 sliding window 크기로 사용한다.
    pub sliding_window: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpLayerTensors {
    pub layer_index: usize,
    pub eh_proj_weight: String,
    pub enorm_weight: String,
    pub hnorm_weight: String,
    pub shared_head_norm_weight: String,
    pub embed_tokens_weight: Option<String>,
    pub shared_head_head_weight: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub architecture: Architecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rope_theta_swa: f32,
    pub rope_dim: usize, // rope.dimension_count (0 = head_dim)
    pub rope_dim_swa: usize,
    pub rope_sections: [usize; 4], // rope.dimension_sections (MRoPE)
    pub norm_eps: f32,
    pub final_logit_softcapping: f32,
    pub query_pre_attn_scalar: f32,
    /// RMS epsilon for post-attention and post-FFN normalization.
    pub post_norm_eps: f32,
    /// Positive multiplier applied after the output projection and before softcapping.
    pub logit_scale: f32,
    pub sliding_window: usize,
    pub shared_kv_layers: usize,
    pub sliding_window_pattern: Vec<bool>,
    /// Full attention layer 의 key dim (= GGUF `<arch>.attention.key_length`). Gemma4 에서 SWA vs full
    /// 의 head_dim 분기에 사용. 0 = head_dim 그대로 사용.
    pub key_length_full: usize,
    pub key_length_swa: usize,
    pub value_length_swa: usize,
    pub embedding_length_per_layer_input: usize,
    /// MoE expert 개수 (gemma4 26B-A4B = 128). 0 = dense.
    pub expert_count: usize,
    /// 매 token 활성화되는 top-k expert (gemma4 26B-A4B = 8).
    pub expert_used_count: usize,
    /// Shared expert count. Hy3 uses one always-active shared expert per MoE layer.
    pub expert_shared_count: usize,
    /// Number of leading dense transformer blocks before sparse MoE begins.
    pub leading_dense_block_count: usize,
    /// GGUF expert gating function identifier (`2` = sigmoid).
    pub expert_gating_func: u32,
    /// Whether selected routed-expert weights are normalized before scaling.
    pub expert_weights_norm: bool,
    /// Routed expert weights multiplier after top-k normalization.
    pub expert_weights_scale: f32,
    /// expert 별 FFN intermediate length (gemma4 26B-A4B = 704).
    pub expert_feed_forward_length: usize,
    /// per-layer KV head count. Some 면 layer 별로 가변, None 이면 num_kv_heads 단일값 사용.
    pub head_count_kv_per_layer: Option<Vec<usize>>,
    pub tokenizer: TokenizerData,
    // SSM/Delta Net fields (Qwen3.5 etc.)
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv_kernel: usize,
    pub full_attention_interval: usize,
    pub layer_kinds: Vec<ModelLayerKind>,
    pub mtp: Option<MtpMetadata>,
    /// Gemma4 assistant 모델 전용 메타. `architecture == Gemma4Assistant` 일 때만 Some.
    pub assistant: Option<AssistantMetadata>,
    /// pm119: GLM DSA lightning indexer 메타 (`<arch>.attention.indexer.*`).
    /// GlmDsa 에서 세 키가 모두 있을 때만 Some.
    pub glm_indexer: Option<GlmIndexerMetadata>,
    pub deepseek4: Option<DeepSeek4Metadata>,
}

/// pm119: DSA lightning indexer 하이퍼파라미터 (GLM-5.2: heads=32, key=128, top_k=2048).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmIndexerMetadata {
    pub head_count: usize,
    pub key_length: usize,
    pub top_k: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeek4Metadata {
    pub q_lora_rank: usize,
    pub indexer: GlmIndexerMetadata,
    pub output_group_count: usize,
    pub output_lora_rank: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f32,
    pub hyper_connection_count: usize,
    pub sinkhorn_iterations: usize,
    pub hyper_connection_eps: f32,
    pub hash_layer_count: usize,
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shared: Vec<f32>,
    pub rope_scaling_factor: f32,
    pub rope_original_context_length: usize,
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    /// DFlash/DSpark sidecar-only metadata. `target_layers` uses the GGUF
    /// layer-input numbering, so `num_layers` denotes the final trunk output.
    pub dspark_block_size: Option<usize>,
    pub dspark_target_layers: Vec<usize>,
    pub dspark_mask_token_id: Option<u32>,
}

pub fn detect_architecture(metadata: &[(String, GGUFValue)]) -> Result<Architecture, LoaderError> {
    let arch = get_string(metadata, "general.architecture")?;
    match arch {
        "llama" => Ok(Architecture::LLaMA),
        "gemma" | "gemma2" | "gemma3" | "gemma3n" => Ok(Architecture::Gemma),
        "gemma4" => Ok(Architecture::Gemma4),
        "gemma4_assistant" | "gemma4-assistant" => Ok(Architecture::Gemma4Assistant),
        "phi" => Ok(Architecture::Phi),
        "qwen2" => Ok(Architecture::Qwen2),
        "qwen35" => Ok(Architecture::Qwen35),
        "qwen35moe" | "qwen35moe_mtp" => Ok(Architecture::Qwen35MoE),
        "nemotron_h_moe" => Ok(Architecture::NemotronHMoE),
        "hy_v3" => Ok(Architecture::Hy3),
        "glm-dsa" => Ok(Architecture::GlmDsa),
        "deepseek4" => Ok(Architecture::DeepSeek4),
        "muse-glimmer" => Ok(Architecture::MuseGlimmer),
        "dflash" => Ok(Architecture::DFlash),
        other => Err(LoaderError::UnsupportedArchitecture(other.to_string())),
    }
}

fn deepseek4_layer_f32(
    metadata: &[(String, GGUFValue)],
    key: &str,
    num_layers: usize,
) -> Result<Vec<f32>, LoaderError> {
    let mut values = match get_f32_array(metadata, key) {
        Ok(values) => values,
        Err(LoaderError::TypeMismatch { .. }) => vec![get_f32(metadata, key)?; num_layers],
        Err(error) => return Err(error),
    };
    if values.len() < num_layers {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!(
                "{key} has {} entries, expected at least {num_layers}",
                values.len()
            ),
        });
    }
    values.truncate(num_layers);
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!("{key} values must be finite and positive"),
        });
    }
    Ok(values)
}

fn validate_deepseek4_compression_ratios(
    prefix: &str,
    ratios: &[usize],
) -> Result<(), LoaderError> {
    if let Some(ratio) = ratios
        .iter()
        .copied()
        .find(|ratio| !matches!(ratio, 0 | 4 | 128))
    {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!(
                "{prefix}.attention.compress_ratios contains unsupported ratio {ratio}; expected 0, 4, or 128"
            ),
        });
    }
    Ok(())
}

fn optional_metadata<T>(value: Result<T, LoaderError>) -> Result<Option<T>, LoaderError> {
    match value {
        Ok(value) => Ok(Some(value)),
        Err(LoaderError::MissingKey(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

fn muse_glimmer_sliding_window_pattern(
    metadata: &[(String, GGUFValue)],
    key: &str,
    num_layers: usize,
) -> Result<Vec<bool>, LoaderError> {
    let value = metadata
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value)
        .ok_or_else(|| LoaderError::MissingKey(key.to_string()))?;
    if matches!(value, GGUFValue::Array(_)) {
        return get_bool_array(metadata, key);
    }

    let period = get_u32(metadata, key)? as usize;
    if period == 0 {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!("{key} scalar period must be greater than zero"),
        });
    }
    Ok((0..num_layers)
        .map(|layer| layer % period < period - 1)
        .collect())
}

pub fn extract_metadata(metadata: &[(String, GGUFValue)]) -> Result<ModelMetadata, LoaderError> {
    let arch = detect_architecture(metadata)?;

    // 아키텍처별 메타데이터 키 prefix
    let arch_str = get_string(metadata, "general.architecture").unwrap_or("llama");
    let known_prefix = match arch {
        Architecture::LLaMA => "llama",
        Architecture::Gemma => "gemma",
        Architecture::Gemma4 => "gemma4",
        Architecture::Gemma4Assistant => "gemma4_assistant",
        Architecture::Phi => "phi",
        Architecture::Qwen2 => "qwen2",
        Architecture::Qwen35 => "qwen35",
        Architecture::Qwen35MoE => "qwen35moe",
        Architecture::NemotronHMoE => "nemotron_h_moe",
        Architecture::Hy3 => "hy_v3",
        Architecture::GlmDsa => "glm-dsa",
        Architecture::DeepSeek4 => "deepseek4",
        Architecture::MuseGlimmer => "muse-glimmer",
        Architecture::DFlash => "dflash",
    };
    let prefix = if get_u32(metadata, &format!("{arch_str}.embedding_length")).is_ok() {
        arch_str
    } else {
        known_prefix
    };

    let vocab_size = get_u32_opt(metadata, &format!("{prefix}.vocab_size"))?
        .map(|value| value as usize)
        .unwrap_or(32000);

    let hidden_size = get_u32(metadata, &format!("{prefix}.embedding_length"))? as usize;
    let total_block_count = get_u32(metadata, &format!("{prefix}.block_count"))? as usize;
    let declared_nextn_predict_layers =
        get_u32_opt(metadata, &format!("{prefix}.nextn_predict_layers"))?.unwrap_or(0) as usize;
    // External Gemma4 assistant GGUFs use `nextn_predict_layers` to describe
    // their own decoder depth. Unlike an in-model MTP tail, those blocks are
    // all executable drafter layers and must not be subtracted from block_count.
    let nextn_predict_layers = if arch == Architecture::Gemma4Assistant {
        0
    } else {
        declared_nextn_predict_layers
    };
    if nextn_predict_layers > total_block_count {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!(
                "{prefix}.nextn_predict_layers ({nextn_predict_layers}) exceeds block_count ({total_block_count})"
            ),
        });
    }
    let num_layers = total_block_count - nextn_predict_layers;
    let mtp = (nextn_predict_layers > 0).then_some(MtpMetadata {
        total_block_count,
        trunk_layers: num_layers,
        first_mtp_layer: total_block_count - nextn_predict_layers,
        nextn_predict_layers,
    });
    let num_heads = get_u32(metadata, &format!("{prefix}.attention.head_count"))? as usize;
    if num_heads == 0 {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!("{prefix}.attention.head_count must be positive"),
        });
    }
    // attention.head_count_kv accepts either a scalar or a per-layer array.
    let head_count_kv_key = format!("{prefix}.attention.head_count_kv");
    let (head_count_kv_per_layer, num_kv_heads) = match get_u32_array(metadata, &head_count_kv_key)
    {
        Ok(values) => {
            let values = values
                .into_iter()
                .map(|value| value as usize)
                .collect::<Vec<_>>();
            let max = values.iter().copied().max().unwrap_or(num_heads);
            (Some(values), max)
        }
        Err(LoaderError::TypeMismatch { .. }) => {
            (None, get_u32(metadata, &head_count_kv_key)? as usize)
        }
        Err(LoaderError::MissingKey(_)) => (None, num_heads),
        Err(error) => return Err(error),
    };
    let feed_forward_key = format!("{prefix}.feed_forward_length");
    let intermediate_size = match get_u32(metadata, &feed_forward_key) {
        Ok(value) => value as usize,
        Err(LoaderError::TypeMismatch { .. }) => get_u32_array(metadata, &feed_forward_key)?
            .into_iter()
            .map(|value| value as usize)
            .max()
            .unwrap_or(0),
        Err(LoaderError::MissingKey(_)) => {
            // MoE-only models (e.g. qwen35moe) have no dense FFN.
            get_u32(metadata, &format!("{prefix}.expert_feed_forward_length"))? as usize
        }
        Err(error) => return Err(error),
    };
    let declared_max_seq_len = get_u32_opt(metadata, &format!("{prefix}.context_length"))?
        .map(|value| value as usize)
        .unwrap_or(4096);
    // The current GLM DSA path evaluates dense attention and is equivalent to
    // the model's sparse top-k attention only while every cached token fits in
    // the selected set. Keep the advertised 1M context disabled until the
    // IndexShare selector is wired.
    let max_seq_len = if arch == Architecture::GlmDsa {
        get_u32_opt(metadata, &format!("{prefix}.attention.indexer.top_k"))?
            .map(|top_k| declared_max_seq_len.min(top_k as usize))
            .unwrap_or(declared_max_seq_len)
    } else {
        declared_max_seq_len
    };
    // pm119: DSA lightning indexer 메타 (GlmDsa 한정, 세 키 모두 있을 때만).
    let glm_indexer = if arch == Architecture::GlmDsa {
        match (
            get_u32_opt(metadata, &format!("{prefix}.attention.indexer.head_count"))?,
            get_u32_opt(metadata, &format!("{prefix}.attention.indexer.key_length"))?,
            get_u32_opt(metadata, &format!("{prefix}.attention.indexer.top_k"))?,
        ) {
            (Some(head_count), Some(key_length), Some(top_k)) => Some(GlmIndexerMetadata {
                head_count: head_count as usize,
                key_length: key_length as usize,
                top_k: top_k as usize,
            }),
            _ => None,
        }
    } else {
        None
    };
    let deepseek4 = if matches!(arch, Architecture::DeepSeek4 | Architecture::DFlash) {
        let compress_ratios =
            get_u32_array(metadata, &format!("{prefix}.attention.compress_ratios"))?
                .into_iter()
                .take(num_layers)
                .map(|ratio| ratio as usize)
                .collect::<Vec<_>>();
        if compress_ratios.len() != num_layers {
            return Err(LoaderError::ParseError {
                offset: 0,
                msg: format!(
                    "{prefix}.attention.compress_ratios has {} trunk entries, expected {num_layers}",
                    compress_ratios.len()
                ),
            });
        }
        validate_deepseek4_compression_ratios(prefix, &compress_ratios)?;
        let swiglu_clamp_exp =
            deepseek4_layer_f32(metadata, &format!("{prefix}.swiglu_clamp_exp"), num_layers)?;
        let swiglu_clamp_shared = match deepseek4_layer_f32(
            metadata,
            &format!("{prefix}.swiglu_clamp_shexp"),
            num_layers,
        ) {
            Ok(values) => values,
            Err(LoaderError::MissingKey(_)) => swiglu_clamp_exp.clone(),
            Err(error) => return Err(error),
        };
        Some(DeepSeek4Metadata {
            q_lora_rank: get_u32(metadata, &format!("{prefix}.attention.q_lora_rank"))? as usize,
            indexer: GlmIndexerMetadata {
                head_count: get_u32(metadata, &format!("{prefix}.attention.indexer.head_count"))?
                    as usize,
                key_length: get_u32(metadata, &format!("{prefix}.attention.indexer.key_length"))?
                    as usize,
                top_k: get_u32(metadata, &format!("{prefix}.attention.indexer.top_k"))? as usize,
            },
            output_group_count: get_u32(
                metadata,
                &format!("{prefix}.attention.output_group_count"),
            )? as usize,
            output_lora_rank: get_u32(metadata, &format!("{prefix}.attention.output_lora_rank"))?
                as usize,
            compress_ratios,
            compress_rope_theta: get_f32(
                metadata,
                &format!("{prefix}.attention.compress_rope_freq_base"),
            )?,
            hyper_connection_count: get_u32(metadata, &format!("{prefix}.hyper_connection.count"))?
                as usize,
            sinkhorn_iterations: get_u32(
                metadata,
                &format!("{prefix}.hyper_connection.sinkhorn_iterations"),
            )? as usize,
            hyper_connection_eps: get_f32(metadata, &format!("{prefix}.hyper_connection.epsilon"))?,
            hash_layer_count: get_u32(metadata, &format!("{prefix}.hash_layer_count"))? as usize,
            swiglu_clamp_exp,
            swiglu_clamp_shared,
            rope_scaling_factor: get_f32(metadata, &format!("{prefix}.rope.scaling.factor"))?,
            rope_original_context_length: get_u32(
                metadata,
                &format!("{prefix}.rope.scaling.original_context_length"),
            )? as usize,
            rope_yarn_beta_fast: get_f32(
                metadata,
                &format!("{prefix}.rope.scaling.yarn_beta_fast"),
            )?,
            rope_yarn_beta_slow: get_f32(
                metadata,
                &format!("{prefix}.rope.scaling.yarn_beta_slow"),
            )?,
            dspark_block_size: (arch == Architecture::DFlash)
                .then(|| get_u32(metadata, &format!("{prefix}.block_size")))
                .transpose()?
                .map(|value| value as usize),
            dspark_target_layers: if arch == Architecture::DFlash {
                get_u32_array(metadata, &format!("{prefix}.target_layers"))?
                    .into_iter()
                    .map(|value| value as usize)
                    .collect()
            } else {
                Vec::new()
            },
            dspark_mask_token_id: (arch == Architecture::DFlash)
                .then(|| get_u32(metadata, "tokenizer.ggml.mask_token_id"))
                .transpose()?,
        })
    } else {
        None
    };
    let rope_theta = get_f32_opt(metadata, &format!("{prefix}.rope.freq_base"))?.unwrap_or(10000.0);
    let rope_theta_swa =
        get_f32_opt(metadata, &format!("{prefix}.rope.freq_base_swa"))?.unwrap_or(rope_theta);
    let rope_dim = get_u32_opt(metadata, &format!("{prefix}.rope.dimension_count"))?
        .map(|v| v as usize)
        .unwrap_or(0);
    let rope_dim_swa = get_u32_opt(metadata, &format!("{prefix}.rope.dimension_count_swa"))?
        .map(|v| v as usize)
        .unwrap_or(rope_dim);
    let rope_sections_vec = optional_metadata(get_u32_array(
        metadata,
        &format!("{prefix}.rope.dimension_sections"),
    ))?
    .unwrap_or_default();
    let rope_sections = [
        rope_sections_vec.first().copied().unwrap_or(0) as usize,
        rope_sections_vec.get(1).copied().unwrap_or(0) as usize,
        rope_sections_vec.get(2).copied().unwrap_or(0) as usize,
        rope_sections_vec.get(3).copied().unwrap_or(0) as usize,
    ];
    let norm_eps = if arch == Architecture::MuseGlimmer {
        get_f32(
            metadata,
            &format!("{prefix}.attention.layer_norm_rms_epsilon"),
        )?
    } else {
        get_f32_opt(
            metadata,
            &format!("{prefix}.attention.layer_norm_rms_epsilon"),
        )?
        .unwrap_or(1e-5)
    };
    let final_logit_softcapping = if arch == Architecture::MuseGlimmer {
        get_f32(metadata, &format!("{prefix}.final_logit_softcapping"))?
    } else {
        get_f32_opt(metadata, &format!("{prefix}.final_logit_softcapping"))?.unwrap_or(0.0)
    };
    let post_norm_eps = get_f32_opt(metadata, &format!("{prefix}.post_norm_epsilon"))?
        .unwrap_or_else(|| {
            if arch == Architecture::MuseGlimmer {
                1e-8
            } else {
                norm_eps
            }
        });
    let logit_scale = if arch == Architecture::MuseGlimmer {
        get_f32(metadata, &format!("{prefix}.logit_scale"))?
    } else {
        get_f32_opt(metadata, &format!("{prefix}.logit_scale"))?.unwrap_or(1.0)
    };

    // head_dim: explicit key_length or hidden_size / num_heads
    let head_dim = get_u32_opt(metadata, &format!("{prefix}.attention.key_length"))?
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_heads);
    let query_pre_attn_scalar = get_f32_opt(metadata, &format!("{prefix}.query_pre_attn_scalar"))?
        .unwrap_or_else(|| match arch {
            Architecture::Gemma => 256.0,
            // Gemma4 uses self.scaling = 1.0 (no pre-attn Q scaling); see runtime contract §2.
            // Gemma4Assistant (drafter) inherits the same Gemma4 attention semantics.
            Architecture::Gemma4 | Architecture::Gemma4Assistant => 1.0,
            _ => head_dim as f32,
        });
    let sliding_window = if arch == Architecture::MuseGlimmer {
        get_u32(metadata, &format!("{prefix}.attention.sliding_window"))? as usize
    } else {
        get_u32_opt(metadata, &format!("{prefix}.attention.sliding_window"))?.unwrap_or(0) as usize
    };
    let shared_kv_layers = get_u32_opt(metadata, &format!("{prefix}.attention.shared_kv_layers"))?
        .unwrap_or(0) as usize;
    let sliding_window_pattern_key = format!("{prefix}.attention.sliding_window_pattern");
    let sliding_window_pattern = if arch == Architecture::MuseGlimmer {
        muse_glimmer_sliding_window_pattern(metadata, &sliding_window_pattern_key, num_layers)?
    } else {
        optional_metadata(get_bool_array(metadata, &sliding_window_pattern_key))?
            .unwrap_or_default()
    };
    if arch == Architecture::MuseGlimmer
        && (!norm_eps.is_finite()
            || norm_eps <= 0.0
            || !post_norm_eps.is_finite()
            || post_norm_eps <= 0.0
            || !final_logit_softcapping.is_finite()
            || final_logit_softcapping <= 0.0
            || !logit_scale.is_finite()
            || logit_scale <= 0.0
            || sliding_window == 0
            || sliding_window_pattern.len() != num_layers)
    {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!(
                "muse-glimmer metadata contract violation: norm_eps={norm_eps}, post_norm_eps={post_norm_eps}, final_logit_softcapping={final_logit_softcapping}, logit_scale={logit_scale}, sliding_window={sliding_window}, sliding_window_pattern_len={}, num_layers={num_layers}",
                sliding_window_pattern.len()
            ),
        });
    }
    let key_length_full =
        get_u32_opt(metadata, &format!("{prefix}.attention.key_length"))?.unwrap_or(0) as usize;
    let key_length_swa =
        get_u32_opt(metadata, &format!("{prefix}.attention.key_length_swa"))?.unwrap_or(0) as usize;
    let value_length_swa = get_u32_opt(metadata, &format!("{prefix}.attention.value_length_swa"))?
        .unwrap_or(0) as usize;
    let embedding_length_per_layer_input = get_u32_opt(
        metadata,
        &format!("{prefix}.embedding_length_per_layer_input"),
    )?
    .unwrap_or(0) as usize;

    // MoE 메타 (gemma4 26B-A4B 등). dense 모델은 0.
    let expert_count =
        get_u32_opt(metadata, &format!("{prefix}.expert_count"))?.unwrap_or(0) as usize;
    let expert_used_count =
        get_u32_opt(metadata, &format!("{prefix}.expert_used_count"))?.unwrap_or(0) as usize;
    let expert_weights_scale =
        get_f32_opt(metadata, &format!("{prefix}.expert_weights_scale"))?.unwrap_or(1.0);
    let expert_feed_forward_length =
        get_u32_opt(metadata, &format!("{prefix}.expert_feed_forward_length"))?.unwrap_or(0)
            as usize;
    let expert_shared_count =
        get_u32_opt(metadata, &format!("{prefix}.expert_shared_count"))?.unwrap_or(0) as usize;
    let leading_dense_block_count =
        get_u32_opt(metadata, &format!("{prefix}.leading_dense_block_count"))?.unwrap_or(0)
            as usize;
    let expert_gating_func =
        get_u32_opt(metadata, &format!("{prefix}.expert_gating_func"))?.unwrap_or(0);
    let expert_weights_norm =
        get_bool_opt(metadata, &format!("{prefix}.expert_weights_norm"))?.unwrap_or(false);
    if matches!(arch, Architecture::DeepSeek4 | Architecture::DFlash) && expert_gating_func != 4 {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!(
                "{prefix}.expert_gating_func must be 4 (sqrt-softplus) for DeepSeek4, got {expert_gating_func}"
            ),
        });
    }

    // SSM/Delta Net parameters (Qwen3.5 etc.)
    let ssm_d_inner =
        get_u32_opt(metadata, &format!("{prefix}.ssm.inner_size"))?.unwrap_or(0) as usize;
    let ssm_d_state =
        get_u32_opt(metadata, &format!("{prefix}.ssm.state_size"))?.unwrap_or(0) as usize;
    let ssm_n_group =
        get_u32_opt(metadata, &format!("{prefix}.ssm.group_count"))?.unwrap_or(0) as usize;
    let ssm_dt_rank =
        get_u32_opt(metadata, &format!("{prefix}.ssm.time_step_rank"))?.unwrap_or(0) as usize;
    let ssm_conv_kernel =
        get_u32_opt(metadata, &format!("{prefix}.ssm.conv_kernel"))?.unwrap_or(0) as usize;
    let full_attention_interval =
        get_u32_opt(metadata, &format!("{prefix}.full_attention_interval"))?.unwrap_or(0) as usize;
    let layer_kinds =
        extract_layer_kinds(metadata, prefix, arch, num_layers, full_attention_interval)?;

    let tokens =
        optional_metadata(get_string_array(metadata, "tokenizer.ggml.tokens"))?.unwrap_or_default();
    let scores = optional_metadata(get_f32_array(metadata, "tokenizer.ggml.scores"))?;
    let token_types = optional_metadata(get_u32_array(metadata, "tokenizer.ggml.token_type"))?;
    let merges =
        optional_metadata(get_string_array(metadata, "tokenizer.ggml.merges"))?.unwrap_or_default();
    let added_tokens =
        optional_metadata(get_string_array(metadata, "tokenizer.ggml.added_tokens"))?
            .unwrap_or_default();
    let bos_id = get_u32_opt(metadata, "tokenizer.ggml.bos_token_id")?;
    let eos_id = get_u32_opt(metadata, "tokenizer.ggml.eos_token_id")?;
    let eot_id = get_u32_opt(metadata, "tokenizer.ggml.eot_token_id")?;
    let unknown_id = get_u32_opt(metadata, "tokenizer.ggml.unknown_token_id")?;
    let padding_id = get_u32_opt(metadata, "tokenizer.ggml.padding_token_id")?;
    let separator_id = match (
        get_u32_opt(metadata, "tokenizer.ggml.separator_token_id")?,
        get_u32_opt(metadata, "tokenizer.ggml.seperator_token_id")?,
    ) {
        (Some(standard), Some(legacy)) if standard != legacy => {
            return Err(LoaderError::ParseError {
                offset: 0,
                msg: format!("conflicting tokenizer separator token ids: {standard} != {legacy}"),
            });
        }
        (Some(id), _) | (_, Some(id)) => Some(id),
        (None, None) => None,
    };
    let tokenizer_model =
        optional_metadata(get_string(metadata, "tokenizer.ggml.model"))?.unwrap_or_default();
    let tokenizer_pre =
        optional_metadata(get_string(metadata, "tokenizer.ggml.pre"))?.map(str::to_owned);
    let chat_template =
        optional_metadata(get_string(metadata, "tokenizer.chat_template"))?.map(str::to_owned);
    let add_bos_token = get_bool_opt(metadata, "tokenizer.ggml.add_bos_token")?
        .unwrap_or(!matches!(arch, Architecture::Hy3 | Architecture::GlmDsa));
    let add_eos_token = get_bool_opt(metadata, "tokenizer.ggml.add_eos_token")?.unwrap_or(false);
    let add_sep_token = get_bool_opt(metadata, "tokenizer.ggml.add_sep_token")?.unwrap_or(false);
    let add_space_prefix =
        get_bool_opt(metadata, "tokenizer.ggml.add_space_prefix")?.unwrap_or(true);

    for (key, values_len) in [
        ("tokenizer.ggml.scores", scores.as_ref().map(Vec::len)),
        (
            "tokenizer.ggml.token_type",
            token_types.as_ref().map(Vec::len),
        ),
    ] {
        if let Some(len) = values_len.filter(|len| *len != tokens.len()) {
            return Err(LoaderError::ParseError {
                offset: 0,
                msg: format!("{key} has {len} entries, expected {}", tokens.len()),
            });
        }
    }
    let scores = scores.unwrap_or_default();
    let token_types = token_types.unwrap_or_default();
    if let Some(token_type) = token_types.iter().copied().find(|value| *value > 6) {
        return Err(LoaderError::ParseError {
            offset: 0,
            msg: format!("tokenizer.ggml.token_type contains unsupported value {token_type}"),
        });
    }

    let effective_vocab_size = if !tokens.is_empty() {
        tokens.len()
    } else {
        vocab_size
    };
    for (key, id) in [
        ("tokenizer.ggml.bos_token_id", bos_id),
        ("tokenizer.ggml.eos_token_id", eos_id),
        ("tokenizer.ggml.eot_token_id", eot_id),
        ("tokenizer.ggml.unknown_token_id", unknown_id),
        ("tokenizer.ggml.separator_token_id", separator_id),
        ("tokenizer.ggml.padding_token_id", padding_id),
        (
            "tokenizer.ggml.mask_token_id",
            deepseek4
                .as_ref()
                .and_then(|metadata| metadata.dspark_mask_token_id),
        ),
    ] {
        if id.is_some_and(|id| id as usize >= effective_vocab_size) {
            return Err(LoaderError::ParseError {
                offset: 0,
                msg: format!(
                    "{key} ({}) is outside vocabulary size {effective_vocab_size}",
                    id.unwrap()
                ),
            });
        }
    }

    // Gemma4 assistant (drafter) 전용 key 추출. 2026-07 포맷은
    // `gemma4-assistant` + `nextn.*` direct-vocab head를 쓰고, 이전 포맷은
    // `gemma4_assistant` + centroid-masked head를 쓴다. 공통 runtime
    // metadata로 정규화하되 head 선택 정보는 optional 값으로 보존한다.
    let assistant = if arch == Architecture::Gemma4Assistant {
        Some(AssistantMetadata {
            n_centroids: get_u32_opt(metadata, &format!("{prefix}.n_centroids"))?.unwrap_or(0),
            centroid_top_k: get_u32_opt(metadata, &format!("{prefix}.centroid_top_k"))?
                .unwrap_or(0),
            n_embd_backbone: get_u32(metadata, &format!("{prefix}.n_embd_backbone"))
                .or_else(|_| get_u32(metadata, &format!("{prefix}.embedding_length_out")))?,
            use_ordered_embeddings: get_bool_opt(
                metadata,
                &format!("{prefix}.use_ordered_embeddings"),
            )?
            .unwrap_or(false),
            requires_target_arch: optional_metadata(get_string(
                metadata,
                &format!("{prefix}.requires_target_arch"),
            ))?
            .unwrap_or("gemma4")
            .to_string(),
            shared_kv_layers: get_u32(metadata, &format!("{prefix}.attention.shared_kv_layers"))?,
            sliding_window_pattern: optional_metadata(get_bool_array(
                metadata,
                &format!("{prefix}.attention.sliding_window_pattern"),
            ))?
            .unwrap_or_default(),
            key_length_full: get_u32(metadata, &format!("{prefix}.attention.key_length"))?,
            key_length_swa: get_u32(metadata, &format!("{prefix}.attention.key_length_swa"))?,
            rope_freq_base_full: get_f32(metadata, &format!("{prefix}.rope.freq_base"))?,
            rope_freq_base_swa: get_f32(metadata, &format!("{prefix}.rope.freq_base_swa"))?,
            sliding_window: get_u32(metadata, &format!("{prefix}.attention.sliding_window"))?,
        })
    } else {
        None
    };

    Ok(ModelMetadata {
        architecture: arch,
        vocab_size: effective_vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        max_seq_len,
        rope_theta,
        rope_theta_swa,
        rope_dim,
        rope_dim_swa,
        rope_sections,
        norm_eps,
        final_logit_softcapping,
        post_norm_eps,
        logit_scale,
        query_pre_attn_scalar,
        sliding_window,
        shared_kv_layers,
        sliding_window_pattern,
        key_length_full,
        key_length_swa,
        value_length_swa,
        embedding_length_per_layer_input,
        expert_count,
        expert_used_count,
        expert_weights_scale,
        expert_feed_forward_length,
        expert_shared_count,
        leading_dense_block_count,
        expert_gating_func,
        expert_weights_norm,
        head_count_kv_per_layer,
        tokenizer: TokenizerData {
            vocab_size: effective_vocab_size,
            tokens,
            scores,
            token_types,
            merges,
            added_tokens,
            bos_id,
            eos_id,
            eot_id,
            unknown_id,
            separator_id,
            padding_id,
            model: tokenizer_model.to_string(),
            pre: tokenizer_pre,
            chat_template,
            add_bos_token,
            add_eos_token,
            add_sep_token,
            add_space_prefix,
        },
        ssm_d_inner,
        ssm_d_state,
        ssm_n_group,
        ssm_dt_rank,
        ssm_conv_kernel,
        full_attention_interval,
        layer_kinds,
        mtp,
        assistant,
        glm_indexer,
        deepseek4,
    })
}

fn extract_layer_kinds(
    metadata: &[(String, GGUFValue)],
    prefix: &str,
    arch: Architecture,
    num_layers: usize,
    full_attention_interval: usize,
) -> Result<Vec<ModelLayerKind>, LoaderError> {
    if arch == Architecture::NemotronHMoE {
        if let Ok(pattern) = get_string(metadata, &format!("{prefix}.hybrid_override_pattern")) {
            let kinds = decode_hybrid_pattern(pattern)
                .map_err(|err| LoaderError::ParseError {
                    offset: 0,
                    msg: format!(
                        "unsupported Nemotron-H layer pattern byte {:?} at index {}",
                        err.byte, err.index
                    ),
                })
                .map(|kinds| {
                    kinds
                        .into_iter()
                        .map(nemotron_layer_kind_to_model_kind)
                        .collect::<Vec<_>>()
                })?;
            if kinds.len() != num_layers {
                return Err(LoaderError::ParseError {
                    offset: 0,
                    msg: format!(
                        "Nemotron-H layer pattern length {} != block_count {num_layers}",
                        kinds.len()
                    ),
                });
            }
            return Ok(kinds);
        }
        return Ok(vec![ModelLayerKind::Attention; num_layers]);
    }

    Ok((0..num_layers)
        .map(|layer_idx| {
            if full_attention_interval > 0
                && layer_idx % full_attention_interval != full_attention_interval.saturating_sub(1)
            {
                ModelLayerKind::Recurrent
            } else {
                ModelLayerKind::Attention
            }
        })
        .collect())
}

fn nemotron_layer_kind_to_model_kind(kind: NemotronLayerKind) -> ModelLayerKind {
    match kind {
        NemotronLayerKind::Mamba2 => ModelLayerKind::Recurrent,
        NemotronLayerKind::MoE => ModelLayerKind::MoE,
        NemotronLayerKind::Attention => ModelLayerKind::Attention,
    }
}

pub fn infer_nemotron_layer_kinds_from_tensor_names<'a>(
    tensor_names: impl IntoIterator<Item = &'a str> + Clone,
    num_layers: usize,
) -> Result<Vec<ModelLayerKind>, LoaderError> {
    let mut kinds = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let kind = classify_layer_from_tensor_names(layer_idx, tensor_names.clone())
            .map_err(|msg| LoaderError::ParseError { offset: 0, msg })?
            .ok_or_else(|| LoaderError::MissingKey(format!("blk.{layer_idx}.<layer-kind>")))?;
        kinds.push(nemotron_layer_kind_to_model_kind(kind));
    }
    Ok(kinds)
}

pub fn collect_mtp_layer_tensors<'a>(
    tensor_names: impl IntoIterator<Item = &'a str>,
    metadata: &ModelMetadata,
) -> Result<Vec<MtpLayerTensors>, LoaderError> {
    let Some(mtp) = &metadata.mtp else {
        return Ok(Vec::new());
    };

    let names = tensor_names.into_iter().collect::<HashSet<_>>();
    let required = |name: String| {
        if names.contains(name.as_str()) {
            Ok(name)
        } else {
            Err(LoaderError::MissingKey(name))
        }
    };
    let optional = |name: String| names.contains(name.as_str()).then_some(name);

    let mut layers = Vec::with_capacity(mtp.nextn_predict_layers);
    for layer_index in mtp.first_mtp_layer..mtp.total_block_count {
        let prefix = format!("blk.{layer_index}.nextn");
        layers.push(MtpLayerTensors {
            layer_index,
            eh_proj_weight: required(format!("{prefix}.eh_proj.weight"))?,
            enorm_weight: required(format!("{prefix}.enorm.weight"))?,
            hnorm_weight: required(format!("{prefix}.hnorm.weight"))?,
            shared_head_norm_weight: required(format!("{prefix}.shared_head_norm.weight"))?,
            embed_tokens_weight: optional(format!("{prefix}.embed_tokens.weight")),
            shared_head_head_weight: optional(format!("{prefix}.shared_head_head.weight")),
        });
    }

    Ok(layers)
}

pub fn build_graph(meta: &ModelMetadata) -> Result<Graph, LoaderError> {
    match meta.architecture {
        Architecture::LLaMA
        | Architecture::Qwen2
        | Architecture::Qwen35
        | Architecture::Qwen35MoE
        | Architecture::NemotronHMoE
        | Architecture::Hy3
        | Architecture::GlmDsa => Ok(llama::build_llama_graph(meta)),
        Architecture::DeepSeek4 | Architecture::DFlash => {
            Ok(deepseek4::build_deepseek4_graph(meta))
        }
        // Gemma4 shares the structural graph builder with Gemma for now; the actual Gemma4-specific
        // forward semantics (ISWA, PLE, KV sharing, f_attention_scale=1.0) live in the engine path
        // and are layered on top. Graph-level split will come if/when builder-level differences
        // demand it.
        Architecture::Gemma | Architecture::Gemma4 => Ok(gemma::build_gemma_graph(meta)),
        Architecture::MuseGlimmer => Ok(muse_glimmer::build_muse_glimmer_graph(meta)),
        // Gemma4 assistant (drafter) GGUF lacks attn_k/attn_v tensors (KV-share with target)
        // and carries extra VQ codebooks. The generic Gemma graph builder cannot produce a
        // valid graph for it — drafter loading lives behind `rnb_mtp::Drafter::load_assistant`
        // per spec §"Stage D". Fail loud here so a stray `Engine::from_gguf` on a drafter file
        // does not silently fall through to a broken Gemma graph.
        Architecture::Gemma4Assistant => Err(LoaderError::UnsupportedArchitecture(
            "gemma4_assistant: drafter GGUF must be loaded via rnb_mtp::Drafter::load_assistant, \
             not the generic Engine path"
                .to_string(),
        )),
        Architecture::Phi => Ok(phi::build_phi_graph(meta)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::types::GGUFValue;

    fn make_llama_meta() -> Vec<(String, GGUFValue)> {
        vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("llama".to_string()),
            ),
            ("llama.embedding_length".to_string(), GGUFValue::U32(4096)),
            ("llama.block_count".to_string(), GGUFValue::U32(32)),
            ("llama.attention.head_count".to_string(), GGUFValue::U32(32)),
            (
                "llama.attention.head_count_kv".to_string(),
                GGUFValue::U32(8),
            ),
            (
                "llama.feed_forward_length".to_string(),
                GGUFValue::U32(11008),
            ),
            ("llama.context_length".to_string(), GGUFValue::U32(4096)),
            ("llama.rope.freq_base".to_string(), GGUFValue::F32(10000.0)),
            (
                "llama.attention.layer_norm_rms_epsilon".to_string(),
                GGUFValue::F32(1e-5),
            ),
        ]
    }

    #[test]
    fn test_extract_metadata_muse_glimmer_contract() {
        let sliding_pattern = (0..52)
            .map(|layer| GGUFValue::Bool(layer % 4 != 3))
            .collect();
        let meta = vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("muse-glimmer".to_string()),
            ),
            (
                "muse-glimmer.embedding_length".to_string(),
                GGUFValue::U32(6656),
            ),
            ("muse-glimmer.block_count".to_string(), GGUFValue::U32(52)),
            (
                "muse-glimmer.attention.head_count".to_string(),
                GGUFValue::U32(32),
            ),
            (
                "muse-glimmer.attention.head_count_kv".to_string(),
                GGUFValue::U32(2),
            ),
            (
                "muse-glimmer.attention.key_length".to_string(),
                GGUFValue::U32(128),
            ),
            (
                "muse-glimmer.attention.value_length".to_string(),
                GGUFValue::U32(128),
            ),
            (
                "muse-glimmer.feed_forward_length".to_string(),
                GGUFValue::U32(19968),
            ),
            (
                "muse-glimmer.context_length".to_string(),
                GGUFValue::U32(131072),
            ),
            (
                "muse-glimmer.rope.freq_base".to_string(),
                GGUFValue::F32(500000.0),
            ),
            (
                "muse-glimmer.attention.layer_norm_rms_epsilon".to_string(),
                GGUFValue::F32(1e-5),
            ),
            (
                "muse-glimmer.attention.sliding_window".to_string(),
                GGUFValue::U32(2048),
            ),
            (
                "muse-glimmer.attention.sliding_window_pattern".to_string(),
                GGUFValue::Array(sliding_pattern),
            ),
            (
                "muse-glimmer.final_logit_softcapping".to_string(),
                GGUFValue::F32(20.0),
            ),
            (
                "muse-glimmer.logit_scale".to_string(),
                GGUFValue::F32(0.19611613),
            ),
        ];

        let metadata = extract_metadata(&meta).unwrap();

        assert_eq!(metadata.architecture, Architecture::MuseGlimmer);
        assert_eq!(metadata.hidden_size, 6656);
        assert_eq!(metadata.num_layers, 52);
        assert_eq!(metadata.num_heads, 32);
        assert_eq!(metadata.num_kv_heads, 2);
        assert_eq!(metadata.head_dim, 128);
        assert_eq!(metadata.intermediate_size, 19968);
        assert_eq!(metadata.max_seq_len, 131072);
        assert_eq!(metadata.rope_theta, 500000.0);
        assert_eq!(metadata.sliding_window, 2048);
        assert_eq!(metadata.sliding_window_pattern.len(), 52);
        assert!(metadata.sliding_window_pattern[0]);
        assert!(!metadata.sliding_window_pattern[3]);
        assert_eq!(metadata.post_norm_eps, 1e-8);
        assert_eq!(metadata.final_logit_softcapping, 20.0);
        assert_eq!(metadata.logit_scale, 0.19611613);

        let mut scalar_pattern = meta.clone();
        scalar_pattern
            .iter_mut()
            .find(|(key, _)| key == "muse-glimmer.attention.sliding_window_pattern")
            .unwrap()
            .1 = GGUFValue::U32(4);
        let scalar_metadata = extract_metadata(&scalar_pattern).unwrap();
        assert_eq!(
            scalar_metadata.sliding_window_pattern,
            metadata.sliding_window_pattern
        );

        for required_key in [
            "muse-glimmer.final_logit_softcapping",
            "muse-glimmer.logit_scale",
            "muse-glimmer.attention.sliding_window",
            "muse-glimmer.attention.sliding_window_pattern",
        ] {
            let mut incomplete = meta.clone();
            incomplete.retain(|(key, _)| key != required_key);
            assert!(
                extract_metadata(&incomplete).is_err(),
                "missing {required_key} must fail"
            );
        }

        let mut short_pattern = meta.clone();
        let pattern = short_pattern
            .iter_mut()
            .find(|(key, _)| key == "muse-glimmer.attention.sliding_window_pattern")
            .unwrap();
        pattern.1 = GGUFValue::Array(vec![GGUFValue::Bool(true); 51]);
        assert!(matches!(
            extract_metadata(&short_pattern),
            Err(LoaderError::ParseError { msg, .. })
                if msg.contains("sliding_window_pattern_len=51")
        ));
    }

    #[test]
    fn test_detect_llama() {
        let meta = make_llama_meta();
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::LLaMA);
    }

    #[test]
    fn test_detect_gemma() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("gemma".to_string()),
        )];
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::Gemma);
    }

    #[test]
    fn test_detect_gemma4_is_dedicated_variant() {
        // Gemma4 is a separate arch variant from Gemma (Gemma3 / Gemma3n / Gemma2 / Gemma1).
        // See docs/gemma4-e2b-it-runtime-contract.md for why this split matters
        // (ISWA, PLE, shared-KV boundary, f_attention_scale = 1.0).
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("gemma4".to_string()),
        )];
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::Gemma4);
    }

    #[test]
    fn test_detect_gemma4_assistant_is_dedicated_variant() {
        // gemma4_assistant 는 drafter 전용 separate arch variant.
        // gemma4 와 별도로 식별돼서 AssistantMetadata 추출 분기로 들어가야 한다.
        // See docs/superpowers/specs/2026-05-13-gemma4-assistant-drafter-design.md
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("gemma4_assistant".to_string()),
        )];
        assert_eq!(
            detect_architecture(&meta).unwrap(),
            Architecture::Gemma4Assistant
        );
    }

    #[test]
    fn test_detect_gemma4_assistant_hyphenated_is_dedicated_variant() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("gemma4-assistant".to_string()),
        )];
        assert_eq!(
            detect_architecture(&meta).unwrap(),
            Architecture::Gemma4Assistant
        );
    }

    #[test]
    fn test_detect_gemma3_still_resolves_to_gemma() {
        // Gemma3 keeps using the Gemma arch variant (no Gemma3-specific split yet).
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("gemma3".to_string()),
        )];
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::Gemma);
    }

    #[test]
    fn test_detect_qwen35moe() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("qwen35moe".to_string()),
        )];
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::Qwen35MoE);
    }

    #[test]
    fn test_detect_qwen35moe_mtp_alias() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("qwen35moe_mtp".to_string()),
        )];
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::Qwen35MoE);
    }

    #[test]
    fn test_extract_metadata_hy_v3_contract() {
        let meta = vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("hy_v3".to_string()),
            ),
            ("hy_v3.embedding_length".to_string(), GGUFValue::U32(4096)),
            ("hy_v3.block_count".to_string(), GGUFValue::U32(81)),
            ("hy_v3.nextn_predict_layers".to_string(), GGUFValue::U32(1)),
            ("hy_v3.attention.head_count".to_string(), GGUFValue::U32(64)),
            (
                "hy_v3.attention.head_count_kv".to_string(),
                GGUFValue::U32(8),
            ),
            (
                "hy_v3.attention.key_length".to_string(),
                GGUFValue::U32(128),
            ),
            (
                "hy_v3.feed_forward_length".to_string(),
                GGUFValue::U32(13312),
            ),
            ("hy_v3.context_length".to_string(), GGUFValue::U32(262144)),
            (
                "hy_v3.attention.layer_norm_rms_epsilon".to_string(),
                GGUFValue::F32(1e-5),
            ),
            ("hy_v3.expert_count".to_string(), GGUFValue::U32(192)),
            ("hy_v3.expert_used_count".to_string(), GGUFValue::U32(8)),
            (
                "hy_v3.expert_feed_forward_length".to_string(),
                GGUFValue::U32(1536),
            ),
            ("hy_v3.expert_shared_count".to_string(), GGUFValue::U32(1)),
            (
                "hy_v3.leading_dense_block_count".to_string(),
                GGUFValue::U32(1),
            ),
            ("hy_v3.expert_gating_func".to_string(), GGUFValue::U32(2)),
            (
                "hy_v3.expert_weights_norm".to_string(),
                GGUFValue::Bool(true),
            ),
            (
                "hy_v3.expert_weights_scale".to_string(),
                GGUFValue::F32(2.826),
            ),
            (
                "tokenizer.chat_template".to_string(),
                GGUFValue::String("{{ messages[0].content }}".to_string()),
            ),
        ];

        let metadata = extract_metadata(&meta).unwrap();

        assert_eq!(metadata.architecture, Architecture::Hy3);
        assert_eq!(metadata.num_layers, 80);
        assert_eq!(metadata.num_heads, 64);
        assert_eq!(metadata.num_kv_heads, 8);
        assert_eq!(metadata.head_dim, 128);
        assert_eq!(metadata.intermediate_size, 13312);
        assert_eq!(metadata.max_seq_len, 262144);
        assert_eq!(metadata.expert_count, 192);
        assert_eq!(metadata.expert_used_count, 8);
        assert_eq!(metadata.expert_feed_forward_length, 1536);
        assert_eq!(metadata.expert_shared_count, 1);
        assert_eq!(metadata.leading_dense_block_count, 1);
        assert_eq!(metadata.expert_gating_func, 2);
        assert!(metadata.expert_weights_norm);
        assert!((metadata.expert_weights_scale - 2.826).abs() < f32::EPSILON);
        assert!(!metadata.tokenizer.add_bos_token);
        assert_eq!(
            metadata.tokenizer.chat_template.as_deref(),
            Some("{{ messages[0].content }}")
        );
        let mtp = metadata.mtp.as_ref().expect("MTP metadata");
        assert_eq!(mtp.total_block_count, 81);
        assert_eq!(mtp.trunk_layers, 80);
        assert_eq!(mtp.first_mtp_layer, 80);
        assert_eq!(mtp.nextn_predict_layers, 1);
    }

    #[test]
    fn test_extract_metadata_glm_dsa_excludes_nextn_block() {
        let meta = vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("glm-dsa".to_string()),
            ),
            ("glm-dsa.embedding_length".to_string(), GGUFValue::U32(6144)),
            ("glm-dsa.block_count".to_string(), GGUFValue::U32(79)),
            (
                "glm-dsa.nextn_predict_layers".to_string(),
                GGUFValue::U32(1),
            ),
            (
                "glm-dsa.attention.head_count".to_string(),
                GGUFValue::U32(64),
            ),
            (
                "glm-dsa.attention.head_count_kv".to_string(),
                GGUFValue::U32(1),
            ),
            (
                "glm-dsa.attention.key_length".to_string(),
                GGUFValue::U32(576),
            ),
            (
                "glm-dsa.feed_forward_length".to_string(),
                GGUFValue::U32(12288),
            ),
            (
                "glm-dsa.context_length".to_string(),
                GGUFValue::U32(1048576),
            ),
            (
                "glm-dsa.attention.layer_norm_rms_epsilon".to_string(),
                GGUFValue::F32(1e-5),
            ),
            ("glm-dsa.expert_count".to_string(), GGUFValue::U32(256)),
            ("glm-dsa.expert_used_count".to_string(), GGUFValue::U32(8)),
            (
                "glm-dsa.attention.indexer.top_k".to_string(),
                GGUFValue::U32(2048),
            ),
            (
                "glm-dsa.expert_feed_forward_length".to_string(),
                GGUFValue::U32(2048),
            ),
            ("glm-dsa.expert_shared_count".to_string(), GGUFValue::U32(1)),
            (
                "glm-dsa.leading_dense_block_count".to_string(),
                GGUFValue::U32(3),
            ),
            ("glm-dsa.expert_gating_func".to_string(), GGUFValue::U32(2)),
            (
                "glm-dsa.expert_weights_norm".to_string(),
                GGUFValue::Bool(true),
            ),
            (
                "glm-dsa.expert_weights_scale".to_string(),
                GGUFValue::F32(2.5),
            ),
        ];

        let metadata = extract_metadata(&meta).unwrap();

        assert_eq!(metadata.architecture, Architecture::GlmDsa);
        assert_eq!(metadata.num_layers, 78);
        assert_eq!(metadata.layer_kinds.len(), 78);
        assert!(metadata
            .layer_kinds
            .iter()
            .all(|kind| *kind == ModelLayerKind::Attention));
        assert_eq!(metadata.num_heads, 64);
        assert_eq!(metadata.num_kv_heads, 1);
        assert_eq!(metadata.head_dim, 576);
        assert_eq!(metadata.max_seq_len, 2048);
        assert_eq!(metadata.expert_count, 256);
        assert_eq!(metadata.expert_used_count, 8);
        assert_eq!(metadata.expert_feed_forward_length, 2048);
        assert_eq!(metadata.expert_shared_count, 1);
        assert_eq!(metadata.leading_dense_block_count, 3);
        assert_eq!(metadata.expert_gating_func, 2);
        assert!(metadata.expert_weights_norm);
        assert!((metadata.expert_weights_scale - 2.5).abs() < f32::EPSILON);
        assert!(!metadata.tokenizer.add_bos_token);
        let mtp = metadata.mtp.as_ref().expect("MTP metadata");
        assert_eq!(mtp.total_block_count, 79);
        assert_eq!(mtp.trunk_layers, 78);
        assert_eq!(mtp.first_mtp_layer, 78);
        assert_eq!(mtp.nextn_predict_layers, 1);
    }

    #[test]
    fn test_extract_metadata_qwen35moe_mtp_excludes_nextn_blocks_from_trunk_layers() {
        let meta = vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("qwen35moe".to_string()),
            ),
            (
                "qwen35moe.embedding_length".to_string(),
                GGUFValue::U32(2048),
            ),
            ("qwen35moe.block_count".to_string(), GGUFValue::U32(41)),
            (
                "qwen35moe.nextn_predict_layers".to_string(),
                GGUFValue::U32(1),
            ),
            (
                "qwen35moe.attention.head_count".to_string(),
                GGUFValue::U32(32),
            ),
            (
                "qwen35moe.attention.head_count_kv".to_string(),
                GGUFValue::U32(4),
            ),
            (
                "qwen35moe.expert_feed_forward_length".to_string(),
                GGUFValue::U32(768),
            ),
            (
                "qwen35moe.context_length".to_string(),
                GGUFValue::U32(262144),
            ),
            (
                "qwen35moe.attention.layer_norm_rms_epsilon".to_string(),
                GGUFValue::F32(1e-6),
            ),
            ("qwen35moe.expert_count".to_string(), GGUFValue::U32(128)),
            ("qwen35moe.expert_used_count".to_string(), GGUFValue::U32(8)),
            (
                "qwen35moe.full_attention_interval".to_string(),
                GGUFValue::U32(4),
            ),
        ];

        let metadata = extract_metadata(&meta).unwrap();

        assert_eq!(metadata.architecture, Architecture::Qwen35MoE);
        assert_eq!(metadata.num_layers, 40);
        assert_eq!(metadata.layer_kinds.len(), 40);
        assert_eq!(metadata.layer_kinds[39], ModelLayerKind::Attention);
        let mtp = metadata.mtp.as_ref().expect("MTP metadata");
        assert_eq!(mtp.total_block_count, 41);
        assert_eq!(mtp.trunk_layers, 40);
        assert_eq!(mtp.first_mtp_layer, 40);
        assert_eq!(mtp.nextn_predict_layers, 1);
    }

    #[test]
    fn test_detect_nemotron_h_moe() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("nemotron_h_moe".to_string()),
        )];
        assert_eq!(
            detect_architecture(&meta).unwrap(),
            Architecture::NemotronHMoE
        );
    }

    #[test]
    fn test_extract_metadata_nemotron_h_moe_text_backbone() {
        let meta = vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("nemotron_h_moe".to_string()),
            ),
            (
                "nemotron_h_moe.embedding_length".to_string(),
                GGUFValue::U32(2688),
            ),
            ("nemotron_h_moe.block_count".to_string(), GGUFValue::U32(4)),
            (
                "nemotron_h_moe.hybrid_override_pattern".to_string(),
                GGUFValue::String("ME*M".to_string()),
            ),
            (
                "nemotron_h_moe.attention.head_count".to_string(),
                GGUFValue::U32(32),
            ),
            (
                "nemotron_h_moe.attention.head_count_kv".to_string(),
                GGUFValue::Array(vec![
                    GGUFValue::I32(0),
                    GGUFValue::I32(2),
                    GGUFValue::I32(0),
                    GGUFValue::I32(0),
                ]),
            ),
            (
                "nemotron_h_moe.expert_feed_forward_length".to_string(),
                GGUFValue::U32(1856),
            ),
            (
                "nemotron_h_moe.context_length".to_string(),
                GGUFValue::U32(1_048_576),
            ),
            (
                "nemotron_h_moe.attention.layer_norm_rms_epsilon".to_string(),
                GGUFValue::F32(1e-5),
            ),
            (
                "nemotron_h_moe.ssm.inner_size".to_string(),
                GGUFValue::U32(5376),
            ),
            (
                "nemotron_h_moe.ssm.state_size".to_string(),
                GGUFValue::U32(128),
            ),
            (
                "nemotron_h_moe.ssm.group_count".to_string(),
                GGUFValue::U32(1),
            ),
            (
                "nemotron_h_moe.ssm.time_step_rank".to_string(),
                GGUFValue::U32(64),
            ),
            (
                "nemotron_h_moe.ssm.conv_kernel".to_string(),
                GGUFValue::U32(4),
            ),
        ];
        let metadata = extract_metadata(&meta).unwrap();
        assert_eq!(metadata.architecture, Architecture::NemotronHMoE);
        assert_eq!(metadata.hidden_size, 2688);
        assert_eq!(metadata.num_layers, 4);
        assert_eq!(metadata.num_kv_heads, 2);
        assert_eq!(metadata.max_seq_len, 1_048_576);
        assert_eq!(metadata.expert_feed_forward_length, 1856);
        assert_eq!(metadata.ssm_d_inner, 5376);
        assert_eq!(
            metadata.layer_kinds,
            vec![
                ModelLayerKind::Recurrent,
                ModelLayerKind::MoE,
                ModelLayerKind::Attention,
                ModelLayerKind::Recurrent
            ]
        );
        assert_eq!(
            metadata.head_count_kv_per_layer.as_deref(),
            Some([0, 2, 0, 0].as_slice())
        );
    }

    #[test]
    fn test_infer_nemotron_layer_kinds_from_tensor_names() {
        let names = vec![
            "blk.0.attn_norm.weight",
            "blk.0.ssm_in.weight",
            "blk.1.attn_norm.weight",
            "blk.1.ffn_gate_inp.weight",
            "blk.2.attn_norm.weight",
            "blk.2.attn_q.weight",
        ];

        let kinds = infer_nemotron_layer_kinds_from_tensor_names(names, 3).unwrap();

        assert_eq!(
            kinds,
            vec![
                ModelLayerKind::Recurrent,
                ModelLayerKind::MoE,
                ModelLayerKind::Attention,
            ]
        );
    }

    #[test]
    fn test_detect_phi() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("phi".to_string()),
        )];
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::Phi);
    }

    #[test]
    fn test_detect_unknown_architecture_is_rejected() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("mixtral".to_string()),
        )];

        let error = detect_architecture(&meta).unwrap_err();
        assert!(matches!(
            error,
            LoaderError::UnsupportedArchitecture(architecture) if architecture == "mixtral"
        ));
    }

    #[test]
    fn test_extract_metadata_gemma4_moe_26b_a4b() {
        // Gemma 4 26B-A4B 의 핵심 키 (필수 + MoE).
        let meta = vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("gemma4".to_string()),
            ),
            ("gemma4.embedding_length".to_string(), GGUFValue::U32(2816)),
            ("gemma4.block_count".to_string(), GGUFValue::U32(30)),
            (
                "gemma4.attention.head_count".to_string(),
                GGUFValue::U32(16),
            ),
            // head_count_kv 가 array (per-layer)
            (
                "gemma4.attention.head_count_kv".to_string(),
                GGUFValue::Array((0..30).map(|_| GGUFValue::I32(2)).collect()),
            ),
            (
                "gemma4.feed_forward_length".to_string(),
                GGUFValue::U32(2112),
            ),
            ("gemma4.context_length".to_string(), GGUFValue::U32(262144)),
            (
                "gemma4.rope.freq_base".to_string(),
                GGUFValue::F32(1_000_000.0),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".to_string(),
                GGUFValue::F32(1e-6),
            ),
            ("gemma4.expert_count".to_string(), GGUFValue::U32(128)),
            ("gemma4.expert_used_count".to_string(), GGUFValue::U32(8)),
            (
                "gemma4.expert_feed_forward_length".to_string(),
                GGUFValue::U32(704),
            ),
            (
                "gemma4.attention.key_length".to_string(),
                GGUFValue::U32(512),
            ),
        ];
        let m = extract_metadata(&meta).unwrap();
        assert_eq!(m.architecture, Architecture::Gemma4);
        assert_eq!(m.expert_count, 128);
        assert_eq!(m.expert_used_count, 8);
        assert_eq!(m.expert_feed_forward_length, 704);
        let kv_arr = m.head_count_kv_per_layer.expect("head_count_kv array 보존");
        assert_eq!(kv_arr.len(), 30);
        assert_eq!(kv_arr[0], 2);
        // num_kv_heads 는 array max 로 채워짐
        assert_eq!(m.num_kv_heads, 2);
    }

    #[test]
    fn test_extract_metadata_llama() {
        let meta = make_llama_meta();
        let m = extract_metadata(&meta).unwrap();
        assert_eq!(m.architecture, Architecture::LLaMA);
        assert_eq!(m.hidden_size, 4096);
        assert_eq!(m.num_layers, 32);
        assert_eq!(m.num_heads, 32);
        assert_eq!(m.num_kv_heads, 8);
        assert_eq!(m.head_dim, 128);
        assert_eq!(m.intermediate_size, 11008);
        assert!((m.rope_theta - 10000.0).abs() < 1.0);
    }

    #[test]
    fn optional_metadata_preserves_present_type_errors() {
        let mut meta = make_llama_meta();
        meta.push((
            "tokenizer.ggml.add_bos_token".to_string(),
            GGUFValue::U32(1),
        ));

        assert!(matches!(
            extract_metadata(&meta),
            Err(LoaderError::TypeMismatch { key, .. })
                if key == "tokenizer.ggml.add_bos_token"
        ));
    }

    #[test]
    fn tokenizer_metadata_preserves_types_and_special_ids() {
        let mut meta = make_llama_meta();
        meta.extend([
            (
                "tokenizer.ggml.model".to_string(),
                GGUFValue::String("llama".to_string()),
            ),
            (
                "tokenizer.ggml.pre".to_string(),
                GGUFValue::String("default".to_string()),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GGUFValue::Array(
                    ["<unk>", "<s>", "</s>", "word"]
                        .into_iter()
                        .map(|token| GGUFValue::String(token.to_string()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                GGUFValue::Array([2, 3, 3, 1].into_iter().map(GGUFValue::I32).collect()),
            ),
            ("tokenizer.ggml.bos_token_id".to_string(), GGUFValue::U32(1)),
            ("tokenizer.ggml.eos_token_id".to_string(), GGUFValue::U32(2)),
            (
                "tokenizer.ggml.unknown_token_id".to_string(),
                GGUFValue::U32(0),
            ),
            (
                "tokenizer.ggml.padding_token_id".to_string(),
                GGUFValue::U32(2),
            ),
            (
                "tokenizer.ggml.seperator_token_id".to_string(),
                GGUFValue::U32(3),
            ),
        ]);

        let tokenizer = extract_metadata(&meta).unwrap().tokenizer;
        assert_eq!(tokenizer.model, "llama");
        assert_eq!(tokenizer.pre.as_deref(), Some("default"));
        assert_eq!(tokenizer.token_types, vec![2, 3, 3, 1]);
        assert_eq!(tokenizer.bos_id, Some(1));
        assert_eq!(tokenizer.eos_id, Some(2));
        assert_eq!(tokenizer.unknown_id, Some(0));
        assert_eq!(tokenizer.padding_id, Some(2));
        assert_eq!(tokenizer.separator_id, Some(3));
    }

    #[test]
    fn tokenizer_metadata_rejects_mismatched_token_types() {
        let mut meta = make_llama_meta();
        meta.extend([
            (
                "tokenizer.ggml.tokens".to_string(),
                GGUFValue::Array(vec![GGUFValue::String("only".to_string())]),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                GGUFValue::Array(vec![GGUFValue::I32(1), GGUFValue::I32(3)]),
            ),
        ]);

        assert!(matches!(
            extract_metadata(&meta),
            Err(LoaderError::ParseError { msg, .. })
                if msg.contains("tokenizer.ggml.token_type has 2 entries, expected 1")
        ));
    }

    #[test]
    fn test_extract_metadata_deepseek4_contract() {
        let meta = vec![
            (
                "general.architecture".to_string(),
                GGUFValue::String("deepseek4".to_string()),
            ),
            (
                "deepseek4.embedding_length".to_string(),
                GGUFValue::U32(4096),
            ),
            ("deepseek4.block_count".to_string(), GGUFValue::U32(2)),
            (
                "deepseek4.attention.head_count".to_string(),
                GGUFValue::U32(64),
            ),
            (
                "deepseek4.attention.head_count_kv".to_string(),
                GGUFValue::U32(1),
            ),
            (
                "deepseek4.attention.key_length".to_string(),
                GGUFValue::U32(512),
            ),
            (
                "deepseek4.rope.dimension_count".to_string(),
                GGUFValue::U32(64),
            ),
            (
                "deepseek4.expert_feed_forward_length".to_string(),
                GGUFValue::U32(2048),
            ),
            ("deepseek4.expert_count".to_string(), GGUFValue::U32(256)),
            ("deepseek4.expert_used_count".to_string(), GGUFValue::U32(6)),
            (
                "deepseek4.expert_gating_func".to_string(),
                GGUFValue::U32(4),
            ),
            (
                "deepseek4.attention.q_lora_rank".to_string(),
                GGUFValue::U32(1024),
            ),
            (
                "deepseek4.attention.indexer.head_count".to_string(),
                GGUFValue::U32(64),
            ),
            (
                "deepseek4.attention.indexer.key_length".to_string(),
                GGUFValue::U32(128),
            ),
            (
                "deepseek4.attention.indexer.top_k".to_string(),
                GGUFValue::U32(512),
            ),
            (
                "deepseek4.attention.output_group_count".to_string(),
                GGUFValue::U32(8),
            ),
            (
                "deepseek4.attention.output_lora_rank".to_string(),
                GGUFValue::U32(1024),
            ),
            (
                "deepseek4.attention.compress_ratios".to_string(),
                GGUFValue::Array(vec![GGUFValue::I32(0), GGUFValue::I32(4)]),
            ),
            (
                "deepseek4.attention.compress_rope_freq_base".to_string(),
                GGUFValue::F32(160000.0),
            ),
            (
                "deepseek4.hyper_connection.count".to_string(),
                GGUFValue::U32(4),
            ),
            (
                "deepseek4.hyper_connection.sinkhorn_iterations".to_string(),
                GGUFValue::U32(20),
            ),
            (
                "deepseek4.hyper_connection.epsilon".to_string(),
                GGUFValue::F32(1e-6),
            ),
            ("deepseek4.hash_layer_count".to_string(), GGUFValue::U32(1)),
            (
                "deepseek4.swiglu_clamp_exp".to_string(),
                GGUFValue::Array(vec![GGUFValue::F32(10.0), GGUFValue::F32(10.0)]),
            ),
            (
                "deepseek4.swiglu_clamp_shexp".to_string(),
                GGUFValue::Array(vec![GGUFValue::F32(10.0), GGUFValue::F32(10.0)]),
            ),
            (
                "deepseek4.rope.scaling.factor".to_string(),
                GGUFValue::F32(16.0),
            ),
            (
                "deepseek4.rope.scaling.original_context_length".to_string(),
                GGUFValue::U32(65536),
            ),
            (
                "deepseek4.rope.scaling.yarn_beta_fast".to_string(),
                GGUFValue::F32(32.0),
            ),
            (
                "deepseek4.rope.scaling.yarn_beta_slow".to_string(),
                GGUFValue::F32(1.0),
            ),
        ];
        let metadata = extract_metadata(&meta).unwrap();
        assert_eq!(metadata.architecture, Architecture::DeepSeek4);
        assert_eq!(metadata.head_dim, 512);
        assert_eq!(metadata.rope_dim, 64);
        let deepseek4 = metadata.deepseek4.expect("DeepSeek4 metadata");
        assert_eq!(deepseek4.compress_ratios, vec![0, 4]);
        assert_eq!(deepseek4.indexer.top_k, 512);
        assert_eq!(deepseek4.hyper_connection_count, 4);
        assert_eq!(deepseek4.hash_layer_count, 1);

        let mut dflash_meta = meta
            .clone()
            .into_iter()
            .map(|(key, value)| (key.replacen("deepseek4.", "dflash.", 1), value))
            .collect::<Vec<_>>();
        dflash_meta
            .iter_mut()
            .find(|(key, _)| key == "general.architecture")
            .unwrap()
            .1 = GGUFValue::String("dflash".to_string());
        dflash_meta.extend([
            ("dflash.vocab_size".to_string(), GGUFValue::U32(151936)),
            ("dflash.block_size".to_string(), GGUFValue::U32(5)),
            (
                "dflash.target_layers".to_string(),
                GGUFValue::Array(vec![
                    GGUFValue::I32(41),
                    GGUFValue::I32(42),
                    GGUFValue::I32(43),
                ]),
            ),
            (
                "tokenizer.ggml.mask_token_id".to_string(),
                GGUFValue::U32(128799),
            ),
        ]);
        let metadata = extract_metadata(&dflash_meta).unwrap();
        assert_eq!(metadata.architecture, Architecture::DFlash);
        let dflash = metadata.deepseek4.expect("DFlash metadata");
        assert_eq!(dflash.dspark_block_size, Some(5));
        assert_eq!(dflash.dspark_target_layers, vec![41, 42, 43]);
        assert_eq!(dflash.dspark_mask_token_id, Some(128799));

        let mut invalid_mask = dflash_meta.clone();
        invalid_mask
            .iter_mut()
            .find(|(key, _)| key == "tokenizer.ggml.mask_token_id")
            .unwrap()
            .1 = GGUFValue::U32(151936);
        assert!(matches!(
            extract_metadata(&invalid_mask),
            Err(LoaderError::ParseError { msg, .. })
                if msg.contains("mask_token_id") && msg.contains("outside vocabulary")
        ));

        let mut invalid_heads = meta;
        invalid_heads
            .iter_mut()
            .find(|(key, _)| key == "deepseek4.attention.head_count")
            .unwrap()
            .1 = GGUFValue::U32(0);
        assert!(matches!(
            extract_metadata(&invalid_heads),
            Err(LoaderError::ParseError { msg, .. })
                if msg.contains("attention.head_count must be positive")
        ));
    }

    #[test]
    fn test_deepseek4_layer_f32_accepts_scalar_and_validates_ratios() {
        let metadata = vec![("deepseek4.clamp".to_string(), GGUFValue::F32(10.0))];
        assert_eq!(
            deepseek4_layer_f32(&metadata, "deepseek4.clamp", 3).unwrap(),
            vec![10.0; 3]
        );
        assert!(validate_deepseek4_compression_ratios("deepseek4", &[0, 4, 128]).is_ok());
        assert!(validate_deepseek4_compression_ratios("deepseek4", &[8]).is_err());
    }
}
