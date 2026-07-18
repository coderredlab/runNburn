use super::quantized_dispatch;

#[derive(Clone)]
pub struct ModelMetadata {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub rope_theta: f32,
    pub rope_theta_swa: f32,
    pub rope_dim: usize,
    pub rope_dim_swa: usize,
    pub rope_sections: [usize; 4],
    pub norm_eps: f32,
    pub final_logit_softcapping: f32,
    pub query_pre_attn_scalar: f32,
    pub sliding_window: usize,
    pub shared_kv_layers: usize,
    pub sliding_window_pattern: Vec<bool>,
    /// Full attention layer 의 key dim. Gemma4 E4B = 512. SWA layer 와 다른 head_dim 분기에 사용.
    /// 0 = head_dim 그대로.
    pub key_length_full: usize,
    pub key_length_swa: usize,
    pub value_length_swa: usize,
    pub head_count_kv_per_layer: Option<Vec<usize>>,
    pub embedding_length_per_layer_input: usize,
    pub expert_used_count: usize,
    pub expert_weights_scale: f32,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv_kernel: usize,
    pub full_attention_interval: usize,
}

/// Pre-allocated buffers for seq_len=1 decode. Avoids heap allocation per token.
#[derive(Clone)]
pub(crate) struct ScratchBuffers {
    pub(super) hidden: Vec<f32>,
    pub(super) norm_buf: Vec<f32>,
    pub(super) norm_buf2: Vec<f32>,

    pub(super) q_buf: Vec<f32>,
    pub(super) k_buf: Vec<f32>,
    pub(super) v_buf: Vec<f32>,
    // cu29 Phase 2: hd=128 fused QKV+RoPE path 결과 (K/V f16 bits, attention
    // backend H2D 필요 없이 KvCache append_bits_range 에 그대로 사용).
    // hd!=128 path 에서는 0-len 유지.
    pub(super) k_bits_buf: Vec<u16>,
    pub(super) v_bits_buf: Vec<u16>,
    pub(super) attn_out: Vec<f32>,
    pub(super) proj_buf: Vec<f32>,
    pub(super) q_split: Vec<f32>,
    pub(super) gate_split: Vec<f32>,

    pub(super) ffn_gate: Vec<f32>,
    pub(super) ffn_up: Vec<f32>,
    pub(super) ffn_down: Vec<f32>,

    pub(super) qkv_buf: Vec<f32>,
    pub(super) z_buf: Vec<f32>,
    pub(super) alpha_buf: Vec<f32>,
    pub(super) beta_buf: Vec<f32>,
    pub(super) conv_input: Vec<f32>,
    pub(super) conv_out: Vec<f32>,
    pub(super) gdn_q: Vec<f32>,
    pub(super) gdn_k: Vec<f32>,
    pub(super) gdn_v: Vec<f32>,
    pub(super) gdn_q_norm: Vec<f32>,
    pub(super) gdn_k_norm: Vec<f32>,
    pub(super) gdn_q_rep: Vec<f32>,
    pub(super) gdn_k_rep: Vec<f32>,
    pub(super) delta_out: Vec<f32>,
    pub(super) gated_out: Vec<f32>,
    pub(super) ssm_proj: Vec<f32>,

    pub(super) logits: Vec<f32>,
    pub(super) backend_argmax_token: Option<u32>,
    pub(super) backend_argmax_only: bool,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) arch_scratch: quantized_dispatch::ArchScratchBuffers,
}

impl ScratchBuffers {
    pub(super) fn new(metadata: &ModelMetadata, ffn_inner_dim: usize) -> Self {
        let hidden_dim = metadata.hidden_dim;
        let num_heads = metadata.num_heads;
        let num_kv_heads = metadata.num_kv_heads;
        let head_dim = metadata.head_dim;
        let vocab_size = metadata.vocab_size;
        let d_inner = metadata.ssm_d_inner;
        let d_state = metadata.ssm_d_state;
        let n_group = metadata.ssm_n_group;
        let dt_rank = metadata.ssm_dt_rank;
        let conv_kernel = metadata.ssm_conv_kernel;
        let conv_channels = if d_inner > 0 {
            d_inner + 2 * n_group * d_state
        } else {
            0
        };
        let num_v_heads = dt_rank;
        let q_dim = if n_group > 0 { d_state * n_group } else { 0 };
        let k_dim = q_dim;
        let qk_rep_dim = num_v_heads * d_state;
        let v_dim = if dt_rank > 0 {
            (d_inner / dt_rank) * num_v_heads
        } else {
            0
        };

        Self {
            hidden: vec![0.0; hidden_dim],
            norm_buf: vec![0.0; hidden_dim],
            norm_buf2: vec![0.0; (num_heads * head_dim).max(hidden_dim)],

            q_buf: vec![0.0; num_heads * head_dim * 2],
            k_buf: vec![0.0; num_kv_heads * head_dim],
            v_buf: vec![0.0; num_kv_heads * head_dim],
            // cu29 Phase 2: hd=128 fused QKV+RoPE path 만 사용. 다른 head_dim
            // 에서는 비어둠 (alloc 비용 무시 — 1024 elem * 2 byte = 2KB).
            k_bits_buf: if head_dim == 128 {
                vec![0u16; num_kv_heads * head_dim]
            } else {
                Vec::new()
            },
            v_bits_buf: if head_dim == 128 {
                vec![0u16; num_kv_heads * head_dim]
            } else {
                Vec::new()
            },
            attn_out: vec![0.0; num_heads * head_dim],
            proj_buf: vec![0.0; hidden_dim],
            q_split: vec![0.0; num_heads * head_dim],
            gate_split: vec![0.0; num_heads * head_dim],

            ffn_gate: vec![0.0; ffn_inner_dim],
            ffn_up: vec![0.0; ffn_inner_dim],
            ffn_down: vec![0.0; hidden_dim],

            qkv_buf: vec![0.0; conv_channels],
            z_buf: vec![0.0; d_inner],
            alpha_buf: vec![0.0; num_v_heads],
            beta_buf: vec![0.0; num_v_heads],
            conv_input: vec![0.0; conv_kernel.max(1) * conv_channels.max(1)],
            conv_out: vec![0.0; conv_channels],
            gdn_q: vec![0.0; q_dim],
            gdn_k: vec![0.0; k_dim],
            gdn_v: vec![0.0; v_dim],
            gdn_q_norm: vec![0.0; q_dim],
            gdn_k_norm: vec![0.0; k_dim],
            gdn_q_rep: vec![0.0; qk_rep_dim],
            gdn_k_rep: vec![0.0; qk_rep_dim],
            delta_out: vec![0.0; d_inner],
            gated_out: vec![0.0; d_inner],
            ssm_proj: vec![0.0; hidden_dim],

            logits: vec![0.0; vocab_size],
            backend_argmax_token: None,
            backend_argmax_only: false,
            arch_scratch: quantized_dispatch::ArchScratchBuffers::new(hidden_dim, ffn_inner_dim),
        }
    }
}
