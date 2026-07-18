use super::policy;
use super::types::ModelMetadata;
use crate::tokenizer::Tokenizer;
use rnb_loader::LoadedModel;

pub(super) fn build_tokenizer(model: &LoadedModel) -> (Tokenizer, usize) {
    let tok_data = &model.metadata.tokenizer;
    let tokens: Vec<String> = if !tok_data.tokens.is_empty() {
        tok_data.tokens.clone()
    } else {
        (0..model.metadata.vocab_size)
            .map(|i| format!("<tok_{i}>"))
            .collect()
    };
    let vocab_size = tokens.len();
    let special = crate::tokenizer::vocab::SpecialTokens {
        bos: tok_data.bos_id,
        eos: tok_data.eos_id,
        pad: None,
    };
    let vocab = crate::tokenizer::vocab::Vocab::new(tokens, special);

    let merges: Vec<(u32, u32)> = tok_data
        .merges
        .iter()
        .filter_map(|rule| {
            let mut parts = rule.splitn(2, ' ');
            let left = parts.next()?;
            let right = parts.next()?;
            let left_id = vocab.token_id(left)?;
            let right_id = vocab.token_id(right)?;
            Some((left_id, right_id))
        })
        .collect();
    eprintln!("[INFO] Raw tokenizer model: {:?}", tok_data.model);

    let mut tokenizer = if tok_data.model == "gpt2" {
        eprintln!("[INFO] Tokenizer: GPT-2 BPE");
        let mut tok = crate::tokenizer::bpe::Tokenizer::new_gpt2(vocab, merges);
        // Honor `tokenizer.ggml.add_bos_token` from GGUF metadata. Qwen3.5 sets
        // this to `false`; new_gpt2 defaults to `true`, which would prepend
        // an unwanted BOS and shift logits vs. llama.cpp.
        tok.set_add_bos_token(tok_data.add_bos_token);
        tok
    } else if tok_data.model == "gemma4" && policy::gemma_tokenizer_bpe_enabled() {
        eprintln!("[INFO] Tokenizer: Gemma4 BPE (opt-in)");
        crate::tokenizer::bpe::Tokenizer::new_gemma4_bpe(
            vocab,
            merges,
            tok_data.scores.clone(),
            tok_data.add_bos_token,
        )
    } else {
        eprintln!("[INFO] Tokenizer: SentencePiece");
        crate::tokenizer::bpe::Tokenizer::new_sentencepiece_with_config(
            vocab,
            merges,
            tok_data.scores.clone(),
            tok_data.add_bos_token,
            tok_data.add_space_prefix,
        )
    };
    tokenizer.set_chat_template(tok_data.chat_template.clone());

    (tokenizer, vocab_size)
}

pub(super) fn build_model_metadata(model: &LoadedModel, vocab_size: usize) -> ModelMetadata {
    let max_seq_len = match policy::max_ctx_override() {
        Some(requested) if model.metadata.architecture == rnb_loader::Architecture::GlmDsa => {
            requested.min(model.metadata.max_seq_len)
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
