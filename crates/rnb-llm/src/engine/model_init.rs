use super::policy;
use super::types::ModelMetadata;
use crate::error::{LlmError, Result};
use crate::tokenizer::Tokenizer;
use rnb_loader::{LoadedModel, TokenizerData};
use std::collections::HashSet;

#[derive(Clone, Copy)]
enum TokenizerKind {
    SentencePiece,
    Gpt2,
    Gemma4,
}

pub(super) fn build_tokenizer(tok_data: &TokenizerData) -> Result<(Tokenizer, usize)> {
    if tok_data.tokens.is_empty() {
        return Err(LlmError::Tokenizer(
            "GGUF is missing tokenizer.ggml.tokens; placeholder and Hugging Face JSON-only tokenizers are unsupported"
                .to_string(),
        ));
    }
    if !tok_data.token_types.is_empty() && tok_data.token_types.len() != tok_data.tokens.len() {
        return Err(LlmError::Tokenizer(format!(
            "tokenizer.ggml.token_type has {} entries, expected {}",
            tok_data.token_types.len(),
            tok_data.tokens.len()
        )));
    }
    if tok_data.add_eos_token || tok_data.add_sep_token {
        return Err(LlmError::Unsupported(
            "tokenizer.ggml.add_eos_token and add_sep_token are not supported".to_string(),
        ));
    }

    let (kind, default_bos, default_eos, default_unknown) = match tok_data.model.as_str() {
        "llama" => (TokenizerKind::SentencePiece, Some(1), Some(2), Some(0)),
        "gpt2" => (TokenizerKind::Gpt2, Some(11), Some(11), None),
        "gemma4" => (TokenizerKind::Gemma4, None, None, None),
        model => {
            return Err(LlmError::Unsupported(format!(
                "tokenizer.ggml.model={model:?}; supported models are \"llama\", \"gpt2\", and \"gemma4\""
            )));
        }
    };

    let bos = tok_data.bos_id.or(default_bos).ok_or_else(|| {
        LlmError::Tokenizer("GGUF is missing tokenizer.ggml.bos_token_id".to_string())
    })?;
    let eos = tok_data.eos_id.or(default_eos).ok_or_else(|| {
        LlmError::Tokenizer("GGUF is missing tokenizer.ggml.eos_token_id".to_string())
    })?;
    for (name, id) in [
        ("bos_token_id", Some(bos)),
        ("eos_token_id", Some(eos)),
        ("eot_token_id", tok_data.eot_id),
        ("unknown_token_id", tok_data.unknown_id.or(default_unknown)),
        ("separator_token_id", tok_data.separator_id),
        ("padding_token_id", tok_data.padding_id),
    ] {
        if id.is_some_and(|id| id as usize >= tok_data.tokens.len()) {
            return Err(LlmError::Tokenizer(format!(
                "tokenizer.ggml.{name} ({}) is outside vocabulary size {}",
                id.unwrap(),
                tok_data.tokens.len()
            )));
        }
    }

    let mut seen = HashSet::with_capacity(tok_data.tokens.len());
    if let Some(token) = tok_data
        .tokens
        .iter()
        .find(|token| !seen.insert(token.as_str()))
    {
        return Err(LlmError::Tokenizer(format!(
            "tokenizer.ggml.tokens contains duplicate token {token:?}"
        )));
    }

    let special = crate::tokenizer::vocab::SpecialTokens {
        bos,
        eos,
        pad: tok_data.padding_id,
    };
    let mut vocab = crate::tokenizer::vocab::Vocab::new(tok_data.tokens.clone(), special);
    let added_token_ids = tok_data
        .added_tokens
        .iter()
        .map(|token| {
            vocab.token_id(token).ok_or_else(|| {
                LlmError::Tokenizer(format!(
                    "tokenizer.ggml.added_tokens entry {token:?} is absent from tokenizer.ggml.tokens"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(id) = tok_data
        .token_types
        .iter()
        .enumerate()
        .filter_map(|(id, token_type)| matches!(token_type, 2 | 3 | 4).then_some(id as u32))
        .chain(added_token_ids.iter().copied())
        .find(|id| tok_data.tokens[*id as usize].is_empty())
    {
        return Err(LlmError::Tokenizer(format!(
            "atomic tokenizer token id {id} is empty"
        )));
    }
    vocab.set_token_metadata(
        tok_data.token_types.clone(),
        tok_data.unknown_id.or(default_unknown),
        tok_data.separator_id,
        added_token_ids,
    );

    let merges = tok_data
        .merges
        .iter()
        .enumerate()
        .map(|(rank, rule)| {
            let split = rule
                .char_indices()
                .find(|(index, ch)| *index > 0 && *ch == ' ')
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    LlmError::Tokenizer(format!(
                        "tokenizer.ggml.merges[{rank}] has no token separator"
                    ))
                })?;
            let left = &rule[..split];
            let right = &rule[split + 1..];
            if right.is_empty() {
                return Err(LlmError::Tokenizer(format!(
                    "tokenizer.ggml.merges[{rank}] has an empty right token"
                )));
            }
            let left_id = vocab.token_id(left).ok_or_else(|| {
                LlmError::Tokenizer(format!(
                    "tokenizer.ggml.merges[{rank}] references unknown left token {left:?}"
                ))
            })?;
            let right_id = vocab.token_id(right).ok_or_else(|| {
                LlmError::Tokenizer(format!(
                    "tokenizer.ggml.merges[{rank}] references unknown right token {right:?}"
                ))
            })?;
            Ok((left_id, right_id))
        })
        .collect::<Result<Vec<_>>>()?;

    eprintln!(
        "[INFO] Raw tokenizer model: {:?}, pre: {:?}",
        tok_data.model, tok_data.pre
    );
    let mut tokenizer = match kind {
        TokenizerKind::Gpt2 => {
            let llama4 = tok_data.pre.as_deref() == Some("llama4");
            eprintln!(
                "[INFO] Tokenizer: GPT-2 BPE{}",
                if llama4 {
                    " + Llama 4 pre-tokenizer"
                } else {
                    ""
                }
            );
            let mut tokenizer = if llama4 {
                crate::tokenizer::bpe::Tokenizer::new_gpt2_llama4(vocab, merges)
            } else {
                crate::tokenizer::bpe::Tokenizer::new_gpt2(vocab, merges)
            };
            tokenizer.set_add_bos_token(tok_data.add_bos_token);
            tokenizer
        }
        TokenizerKind::Gemma4 if policy::gemma_tokenizer_bpe_enabled() => {
            eprintln!("[INFO] Tokenizer: Gemma4 BPE");
            crate::tokenizer::bpe::Tokenizer::new_gemma4_bpe(
                vocab,
                merges,
                tok_data.scores.clone(),
                tok_data.add_bos_token,
            )
        }
        TokenizerKind::Gemma4 | TokenizerKind::SentencePiece => {
            eprintln!("[INFO] Tokenizer: SentencePiece");
            crate::tokenizer::bpe::Tokenizer::new_sentencepiece_with_config(
                vocab,
                merges,
                tok_data.scores.clone(),
                tok_data.add_bos_token,
                tok_data.add_space_prefix,
            )
        }
    };
    tokenizer.set_chat_template(tok_data.chat_template.clone());
    tokenizer.set_model_stop_tokens(
        tok_data
            .eot_id
            .filter(|token| *token != eos)
            .into_iter()
            .collect(),
    );

    Ok((tokenizer, tok_data.tokens.len()))
}

pub(super) fn build_model_metadata(model: &LoadedModel, vocab_size: usize) -> ModelMetadata {
    let max_seq_len = match policy::max_ctx_override() {
        Some(requested) if model.metadata.architecture == rnb_loader::Architecture::GlmDsa => {
            // pm119 2b: DSA indexer opt-in + weight 존재 시 top_k clamp 해제 —
            // top_k 초과 attend 는 selected-set attention 이 담당 (KV 할당은
            // 사용자 지정 RNB_MAX_CTX 그대로).
            let indexer_ready = policy::env_string("RNB_GLM_DSA_INDEXER").as_deref() == Some("1")
                && model.metadata.glm_indexer.is_some()
                && model.weights.contains_key("blk.0.indexer.proj.weight");
            if indexer_ready {
                requested
            } else {
                requested.min(model.metadata.max_seq_len)
            }
        }
        Some(requested) => requested,
        None => model.metadata.max_seq_len,
    };
    ModelMetadata {
        num_layers: model.metadata.num_layers,
        num_heads: model.metadata.num_heads,
        num_kv_heads: model.metadata.num_kv_heads,
        head_dim: model.metadata.head_dim,
        vocab_size,
        max_seq_len,
        hidden_dim: model.metadata.hidden_size,
        rope_theta: model.metadata.rope_theta,
        rope_theta_swa: model.metadata.rope_theta_swa,
        rope_dim: model.metadata.rope_dim,
        rope_dim_swa: model.metadata.rope_dim_swa,
        rope_sections: model.metadata.rope_sections,
        norm_eps: model.metadata.norm_eps,
        final_logit_softcapping: model.metadata.final_logit_softcapping,
        post_norm_eps: model.metadata.post_norm_eps,
        logit_scale: model.metadata.logit_scale,
        query_pre_attn_scalar: model.metadata.query_pre_attn_scalar,
        sliding_window: model.metadata.sliding_window,
        shared_kv_layers: model.metadata.shared_kv_layers,
        sliding_window_pattern: model.metadata.sliding_window_pattern.clone(),
        key_length_full: model.metadata.key_length_full,
        key_length_swa: model.metadata.key_length_swa,
        value_length_swa: model.metadata.value_length_swa,
        head_count_kv_per_layer: model.metadata.head_count_kv_per_layer.clone(),
        embedding_length_per_layer_input: model.metadata.embedding_length_per_layer_input,
        expert_used_count: model.metadata.expert_used_count,
        expert_weights_scale: model.metadata.expert_weights_scale,
        ssm_d_inner: model.metadata.ssm_d_inner,
        ssm_d_state: model.metadata.ssm_d_state,
        ssm_n_group: model.metadata.ssm_n_group,
        ssm_dt_rank: model.metadata.ssm_dt_rank,
        ssm_conv_kernel: model.metadata.ssm_conv_kernel,
        full_attention_interval: model.metadata.full_attention_interval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_data(model: &str, tokens: &[&str]) -> TokenizerData {
        let mut data = TokenizerData::placeholder(tokens.len());
        data.model = model.to_string();
        data.tokens = tokens.iter().map(|token| (*token).to_string()).collect();
        data.bos_id = Some(1);
        data.eos_id = Some(2);
        data.add_bos_token = false;
        data
    }

    #[test]
    fn tokenizer_builder_rejects_missing_vocab() {
        let data = TokenizerData::placeholder(32_000);
        assert!(matches!(
            build_tokenizer(&data),
            Err(LlmError::Tokenizer(message))
                if message.contains("missing tokenizer.ggml.tokens")
        ));
    }

    #[test]
    fn tokenizer_builder_rejects_unknown_model_instead_of_falling_back() {
        let data = tokenizer_data("rwkv", &["x", "<bos>", "<eos>"]);
        assert!(matches!(
            build_tokenizer(&data),
            Err(LlmError::Unsupported(message))
                if message.contains("tokenizer.ggml.model=\"rwkv\"")
        ));
    }

    #[test]
    fn tokenizer_builder_connects_token_types_and_special_ids() {
        let mut data = tokenizer_data(
            "gpt2",
            &["x", "<bos>", "<eos>", "SPECIAL", "<pad>", "<unk>"],
        );
        data.token_types = vec![1, 3, 3, 3, 3, 2];
        data.padding_id = Some(4);

        let (tokenizer, vocab_size) = build_tokenizer(&data).unwrap();
        assert_eq!(vocab_size, 6);
        assert_eq!(tokenizer.vocab.special.pad, Some(4));
        assert_eq!(tokenizer.vocab.unknown_id(), Some(5));
        assert_eq!(tokenizer.encode("SPECIAL"), vec![3]);
    }

    #[test]
    fn tokenizer_builder_preserves_distinct_eot_stop_token() {
        let mut data = tokenizer_data("gpt2", &["x", "<bos>", "<eos>", "<eot>"]);
        data.eot_id = Some(3);

        let (tokenizer, _) = build_tokenizer(&data).unwrap();

        assert_eq!(tokenizer.model_stop_tokens(), &[3]);
    }

    #[test]
    fn tokenizer_builder_rejects_empty_atomic_token() {
        let mut data = tokenizer_data("gpt2", &["", "<bos>", "<eos>"]);
        data.token_types = vec![3, 3, 3];

        assert!(matches!(
            build_tokenizer(&data),
            Err(LlmError::Tokenizer(message)) if message.contains("token id 0 is empty")
        ));
    }

    #[test]
    fn tokenizer_builder_rejects_malformed_merge_rules() {
        let mut data = tokenizer_data("gpt2", &["x", "<bos>", "<eos>"]);
        data.merges.push("missing-separator".to_string());

        assert!(matches!(
            build_tokenizer(&data),
            Err(LlmError::Tokenizer(message))
                if message.contains("merges[0] has no token separator")
        ));
    }

    #[test]
    fn tokenizer_builder_rejects_unimplemented_add_special_policy() {
        let mut data = tokenizer_data("gpt2", &["x", "<bos>", "<eos>"]);
        data.add_eos_token = true;

        assert!(matches!(
            build_tokenizer(&data),
            Err(LlmError::Unsupported(message))
                if message.contains("add_eos_token")
        ));
    }
}
