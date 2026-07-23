pub mod gemma;
pub mod llama;
pub mod phi;

use crate::error::LoaderError;
use crate::gguf::metadata::{
    get_bool, get_bool_array, get_bool_opt, get_f32, get_f32_array, get_f32_opt, get_string,
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
}

/// pm119: DSA lightning indexer 하이퍼파라미터 (GLM-5.2: heads=32, key=128, top_k=2048).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmIndexerMetadata {
    pub head_count: usize,
    pub key_length: usize,
    pub top_k: usize,
}

pub fn detect_architecture(metadata: &[(String, GGUFValue)]) -> Result<Architecture, LoaderError> {
    let arch = get_string(metadata, "general.architecture")?;
    match arch {
        "llama" => Ok(Architecture::LLaMA),
        "gemma" | "gemma2" | "gemma3" | "gemma3n" => Ok(Architecture::Gemma),
        "gemma4" => Ok(Architecture::Gemma4),
        "gemma4_assistant" => Ok(Architecture::Gemma4Assistant),
        "phi" => Ok(Architecture::Phi),
        "qwen2" => Ok(Architecture::Qwen2),
        "qwen35" => Ok(Architecture::Qwen35),
        "qwen35moe" | "qwen35moe_mtp" => Ok(Architecture::Qwen35MoE),
        "nemotron_h_moe" => Ok(Architecture::NemotronHMoE),
        "hy_v3" => Ok(Architecture::Hy3),
        "glm-dsa" => Ok(Architecture::GlmDsa),
        // LLaMA 호환 아키텍처는 자동 fallback
        other => {
            eprintln!(
                "Warning: unknown architecture '{other}', falling back to LLaMA-compatible mode"
            );
            Ok(Architecture::LLaMA)
        }
    }
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
    };
    // 실제 GGUF에 있는 키를 먼저 시도하고, 없으면 known prefix로 fallback
    let prefix = if get_u32(metadata, &format!("{arch_str}.embedding_length")).is_ok() {
        arch_str
    } else {
        known_prefix
    };

    let vocab_size = get_u32(metadata, &format!("{prefix}.vocab_size"))
        .or_else(|_| get_u32(metadata, "tokenizer.ggml.tokens"))
        .map(|v| v as usize)
        .unwrap_or(32000);

    let hidden_size = get_u32(metadata, &format!("{prefix}.embedding_length"))? as usize;
    let total_block_count = get_u32(metadata, &format!("{prefix}.block_count"))? as usize;
    let nextn_predict_layers =
        get_u32_opt(metadata, &format!("{prefix}.nextn_predict_layers")).unwrap_or(0) as usize;
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
    // attention.head_count_kv 는 단일 u32 또는 per-layer array (gemma4 26B-A4B 등 MoE 변형).
    // Array 면 Vec 보존하고, 단일값은 max (또는 첫 번째) 로 num_kv_heads 도 채움.
    let head_count_kv_per_layer: Option<Vec<usize>> =
        get_u32_array(metadata, &format!("{prefix}.attention.head_count_kv"))
            .ok()
            .map(|v| v.into_iter().map(|x| x as usize).collect());
    let num_kv_heads = match &head_count_kv_per_layer {
        Some(v) if !v.is_empty() => *v.iter().max().unwrap(),
        _ => get_u32_opt(metadata, &format!("{prefix}.attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or(num_heads),
    };
    let intermediate_size = get_u32(metadata, &format!("{prefix}.feed_forward_length"))
        .map(|v| v as usize)
        .or_else(|_| {
            get_u32_array(metadata, &format!("{prefix}.feed_forward_length"))
                .map(|vals| vals.into_iter().map(|v| v as usize).max().unwrap_or(0))
        })
        .or_else(|_| {
            // MoE-only models (e.g. qwen35moe) have no dense FFN, so
            // `feed_forward_length` is absent. Fall back to
            // `expert_feed_forward_length` so scratch/debug sizing stays sane.
            get_u32(metadata, &format!("{prefix}.expert_feed_forward_length")).map(|v| v as usize)
        })?;
    let declared_max_seq_len = get_u32(metadata, &format!("{prefix}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(4096);
    // The current GLM DSA path evaluates dense attention and is equivalent to
    // the model's sparse top-k attention only while every cached token fits in
    // the selected set. Keep the advertised 1M context disabled until the
    // IndexShare selector is wired.
    let max_seq_len = if arch == Architecture::GlmDsa {
        get_u32_opt(metadata, &format!("{prefix}.attention.indexer.top_k"))
            .map(|top_k| declared_max_seq_len.min(top_k as usize))
            .unwrap_or(declared_max_seq_len)
    } else {
        declared_max_seq_len
    };
    // pm119: DSA lightning indexer 메타 (GlmDsa 한정, 세 키 모두 있을 때만).
    let glm_indexer = if arch == Architecture::GlmDsa {
        match (
            get_u32_opt(metadata, &format!("{prefix}.attention.indexer.head_count")),
            get_u32_opt(metadata, &format!("{prefix}.attention.indexer.key_length")),
            get_u32_opt(metadata, &format!("{prefix}.attention.indexer.top_k")),
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
    let rope_theta = get_f32_opt(metadata, &format!("{prefix}.rope.freq_base")).unwrap_or(10000.0);
    let rope_theta_swa =
        get_f32_opt(metadata, &format!("{prefix}.rope.freq_base_swa")).unwrap_or(rope_theta);
    let rope_dim = get_u32_opt(metadata, &format!("{prefix}.rope.dimension_count"))
        .map(|v| v as usize)
        .unwrap_or(0);
    let rope_dim_swa = get_u32_opt(metadata, &format!("{prefix}.rope.dimension_count_swa"))
        .map(|v| v as usize)
        .unwrap_or(rope_dim);
    let rope_sections_vec =
        get_u32_array(metadata, &format!("{prefix}.rope.dimension_sections")).unwrap_or_default();
    let rope_sections = [
        rope_sections_vec.first().copied().unwrap_or(0) as usize,
        rope_sections_vec.get(1).copied().unwrap_or(0) as usize,
        rope_sections_vec.get(2).copied().unwrap_or(0) as usize,
        rope_sections_vec.get(3).copied().unwrap_or(0) as usize,
    ];
    let norm_eps = get_f32_opt(
        metadata,
        &format!("{prefix}.attention.layer_norm_rms_epsilon"),
    )
    .unwrap_or(1e-5);
    let final_logit_softcapping =
        get_f32_opt(metadata, &format!("{prefix}.final_logit_softcapping")).unwrap_or(0.0);

    // head_dim: explicit key_length or hidden_size / num_heads
    let head_dim = get_u32_opt(metadata, &format!("{prefix}.attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_heads);
    let query_pre_attn_scalar = get_f32_opt(metadata, &format!("{prefix}.query_pre_attn_scalar"))
        .unwrap_or_else(|| match arch {
            Architecture::Gemma => 256.0,
            // Gemma4 uses self.scaling = 1.0 (no pre-attn Q scaling); see runtime contract §2.
            // Gemma4Assistant (drafter) inherits the same Gemma4 attention semantics.
            Architecture::Gemma4 | Architecture::Gemma4Assistant => 1.0,
            _ => head_dim as f32,
        });
    let sliding_window =
        get_u32_opt(metadata, &format!("{prefix}.attention.sliding_window")).unwrap_or(0) as usize;
    let shared_kv_layers = get_u32_opt(metadata, &format!("{prefix}.attention.shared_kv_layers"))
        .unwrap_or(0) as usize;
    let sliding_window_pattern = get_bool_array(
        metadata,
        &format!("{prefix}.attention.sliding_window_pattern"),
    )
    .unwrap_or_default();
    let key_length_full =
        get_u32_opt(metadata, &format!("{prefix}.attention.key_length")).unwrap_or(0) as usize;
    let key_length_swa =
        get_u32_opt(metadata, &format!("{prefix}.attention.key_length_swa")).unwrap_or(0) as usize;
    let value_length_swa = get_u32_opt(metadata, &format!("{prefix}.attention.value_length_swa"))
        .unwrap_or(0) as usize;
    let embedding_length_per_layer_input = get_u32_opt(
        metadata,
        &format!("{prefix}.embedding_length_per_layer_input"),
    )
    .unwrap_or(0) as usize;

    // MoE 메타 (gemma4 26B-A4B 등). dense 모델은 0.
    let expert_count =
        get_u32_opt(metadata, &format!("{prefix}.expert_count")).unwrap_or(0) as usize;
    let expert_used_count =
        get_u32_opt(metadata, &format!("{prefix}.expert_used_count")).unwrap_or(0) as usize;
    let expert_weights_scale =
        get_f32_opt(metadata, &format!("{prefix}.expert_weights_scale")).unwrap_or(1.0);
    let expert_feed_forward_length =
        get_u32_opt(metadata, &format!("{prefix}.expert_feed_forward_length")).unwrap_or(0)
            as usize;
    let expert_shared_count =
        get_u32_opt(metadata, &format!("{prefix}.expert_shared_count")).unwrap_or(0) as usize;
    let leading_dense_block_count =
        get_u32_opt(metadata, &format!("{prefix}.leading_dense_block_count")).unwrap_or(0) as usize;
    let expert_gating_func =
        get_u32_opt(metadata, &format!("{prefix}.expert_gating_func")).unwrap_or(0);
    let expert_weights_norm =
        get_bool_opt(metadata, &format!("{prefix}.expert_weights_norm")).unwrap_or(false);

    // SSM/Delta Net parameters (Qwen3.5 etc.)
    let ssm_d_inner =
        get_u32_opt(metadata, &format!("{prefix}.ssm.inner_size")).unwrap_or(0) as usize;
    let ssm_d_state =
        get_u32_opt(metadata, &format!("{prefix}.ssm.state_size")).unwrap_or(0) as usize;
    let ssm_n_group =
        get_u32_opt(metadata, &format!("{prefix}.ssm.group_count")).unwrap_or(0) as usize;
    let ssm_dt_rank =
        get_u32_opt(metadata, &format!("{prefix}.ssm.time_step_rank")).unwrap_or(0) as usize;
    let ssm_conv_kernel =
        get_u32_opt(metadata, &format!("{prefix}.ssm.conv_kernel")).unwrap_or(0) as usize;
    let full_attention_interval =
        get_u32_opt(metadata, &format!("{prefix}.full_attention_interval")).unwrap_or(0) as usize;
    let layer_kinds =
        extract_layer_kinds(metadata, prefix, arch, num_layers, full_attention_interval)?;

    // 토크나이저 데이터 파싱 (없으면 빈 벡터로 fallback)
    let tokens = get_string_array(metadata, "tokenizer.ggml.tokens").unwrap_or_default();
    let scores = get_f32_array(metadata, "tokenizer.ggml.scores").unwrap_or_default();
    let merges = get_string_array(metadata, "tokenizer.ggml.merges").unwrap_or_default();
    let bos_id = get_u32_opt(metadata, "tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos_id = get_u32_opt(metadata, "tokenizer.ggml.eos_token_id").unwrap_or(2);
    let tokenizer_model = get_string(metadata, "tokenizer.ggml.model").unwrap_or_default();
    let chat_template = metadata
        .iter()
        .any(|(key, _)| key == "tokenizer.chat_template")
        .then(|| get_string(metadata, "tokenizer.chat_template").map(str::to_owned))
        .transpose()?;
    let add_bos_token = get_bool_opt(metadata, "tokenizer.ggml.add_bos_token")
        .unwrap_or(!matches!(arch, Architecture::Hy3 | Architecture::GlmDsa));
    let add_space_prefix =
        get_bool_opt(metadata, "tokenizer.ggml.add_space_prefix").unwrap_or(true);

    // vocab_size는 tokens 배열이 있으면 그 길이, 없으면 메타데이터 값 사용
    let effective_vocab_size = if !tokens.is_empty() {
        tokens.len()
    } else {
        vocab_size
    };

    // Gemma4 assistant (drafter) 전용 key 추출. dense/MoE target 모델은 None.
    // sliding_window_pattern, key_length_full/swa, rope.freq_base_swa 같은 sub-namespace 키는
    // `{prefix}.<sub>` 형태로 쌓이고, prefix 는 known_prefix == "gemma4_assistant" 와 동일.
    let assistant = if arch == Architecture::Gemma4Assistant {
        Some(AssistantMetadata {
            n_centroids: get_u32(metadata, &format!("{prefix}.n_centroids"))?,
            centroid_top_k: get_u32(metadata, &format!("{prefix}.centroid_top_k"))?,
            n_embd_backbone: get_u32(metadata, &format!("{prefix}.n_embd_backbone"))?,
            use_ordered_embeddings: get_bool(metadata, &format!("{prefix}.use_ordered_embeddings"))
                .unwrap_or(false),
            requires_target_arch: get_string(metadata, &format!("{prefix}.requires_target_arch"))?
                .to_string(),
            shared_kv_layers: get_u32(metadata, &format!("{prefix}.attention.shared_kv_layers"))?,
            sliding_window_pattern: get_bool_array(
                metadata,
                &format!("{prefix}.attention.sliding_window_pattern"),
            )
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
            merges,
            bos_id,
            eos_id,
            model: tokenizer_model.to_string(),
            chat_template,
            add_bos_token,
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
        // Gemma4 shares the structural graph builder with Gemma for now; the actual Gemma4-specific
        // forward semantics (ISWA, PLE, KV sharing, f_attention_scale=1.0) live in the engine path
        // and are layered on top. Graph-level split will come if/when builder-level differences
        // demand it.
        Architecture::Gemma | Architecture::Gemma4 => Ok(gemma::build_gemma_graph(meta)),
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
    fn test_detect_unknown_falls_back_to_llama() {
        let meta = vec![(
            "general.architecture".to_string(),
            GGUFValue::String("mixtral".to_string()),
        )];
        // 알 수 없는 아키텍처는 LLaMA로 fallback
        assert_eq!(detect_architecture(&meta).unwrap(), Architecture::LLaMA);
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
}
