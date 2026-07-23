pub(in crate::runtime) const GGML_F32: u32 = 0;
pub(in crate::runtime) const GGML_Q4_K: u32 = 12;
pub(in crate::runtime) const GGML_Q6_K: u32 = 14;
pub(in crate::runtime) const GGML_Q8_0: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpVerifyBufferPlan {
    pub window_tokens: usize,
    pub hidden_dim: usize,
    pub prefix_count: usize,
    pub token_id_bytes: usize,
    pub target_token_bytes: usize,
    pub hidden_row_bytes: usize,
    pub scratch_hidden_bytes: usize,
    pub prefix_index_bytes: usize,
}

impl MtpVerifyBufferPlan {
    pub fn total_device_bytes(&self) -> usize {
        self.token_id_bytes
            + self.target_token_bytes
            + self.hidden_row_bytes
            + self.scratch_hidden_bytes
            + self.prefix_index_bytes
    }
}

pub fn qwen35_mtp_verify_buffer_plan(
    window_tokens: usize,
    hidden_dim: usize,
    prefix_count: usize,
) -> Result<MtpVerifyBufferPlan, String> {
    if window_tokens == 0 {
        return Err("MTP verify window requires at least one token".to_string());
    }
    if hidden_dim == 0 {
        return Err("MTP verify window requires non-zero hidden_dim".to_string());
    }

    let token_id_bytes = window_tokens
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "MTP verify token id buffer size overflow".to_string())?;
    let target_token_bytes = token_id_bytes;
    let hidden_row_bytes = window_tokens
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "MTP verify hidden row buffer size overflow".to_string())?;
    let scratch_hidden_bytes = window_tokens
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "MTP verify scratch hidden buffer size overflow".to_string())?;
    let prefix_index_bytes = prefix_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "MTP verify prefix index buffer size overflow".to_string())?;

    Ok(MtpVerifyBufferPlan {
        window_tokens,
        hidden_dim,
        prefix_count,
        token_id_bytes,
        target_token_bytes,
        hidden_row_bytes,
        scratch_hidden_bytes,
        prefix_index_bytes,
    })
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyDeviceBuffers {
    pub(in crate::runtime) token_ids_dev: u64,
    pub(in crate::runtime) target_tokens_dev: u64,
    pub(in crate::runtime) hidden_rows_dev: u64,
    pub(in crate::runtime) scratch_hidden_dev: u64,
    pub(in crate::runtime) prefix_indices_dev: u64,
    pub(in crate::runtime) plan: MtpVerifyBufferPlan,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyGdnProjectionBuffers {
    pub(in crate::runtime) qkv_dev: u64,
    pub(in crate::runtime) gate_dev: u64,
    pub(in crate::runtime) alpha_dev: u64,
    pub(in crate::runtime) beta_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) qkv_rows: usize,
    pub(in crate::runtime) gate_rows: usize,
    pub(in crate::runtime) alpha_rows: usize,
    pub(in crate::runtime) beta_rows: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyAttentionQkvProjectionBuffers {
    pub(in crate::runtime) q_dev: u64,
    pub(in crate::runtime) k_dev: u64,
    pub(in crate::runtime) v_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) q_rows: usize,
    pub(in crate::runtime) k_rows: usize,
    pub(in crate::runtime) v_rows: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyAttentionQkNormRopeBuffers {
    pub(in crate::runtime) q_dev: u64,
    pub(in crate::runtime) gate_dev: Option<u64>,
    pub(in crate::runtime) k_bits_dev: u64,
    pub(in crate::runtime) v_bits_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) q_rows: usize,
    pub(in crate::runtime) kv_rows: usize,
    pub(in crate::runtime) head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyAttentionOutputBuffers {
    pub(in crate::runtime) k_f32_dev: u64,
    pub(in crate::runtime) v_f32_dev: u64,
    pub(in crate::runtime) attn_out_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) q_rows: usize,
    pub(in crate::runtime) kv_rows: usize,
    pub(in crate::runtime) head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyGdnConvBuffers {
    pub(in crate::runtime) conv_state_dev: u64,
    pub(in crate::runtime) device_resident_state: bool,
    pub(in crate::runtime) conv_input_dev: u64,
    pub(in crate::runtime) conv_out_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) channels: usize,
    pub(in crate::runtime) kernel_size: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyGdnDeltaInputBuffers {
    pub(in crate::runtime) q_dev: u64,
    pub(in crate::runtime) k_dev: u64,
    pub(in crate::runtime) v_dev: u64,
    pub(in crate::runtime) gate_dev: u64,
    pub(in crate::runtime) beta_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) num_heads: usize,
    pub(in crate::runtime) head_k_dim: usize,
    pub(in crate::runtime) head_v_dim: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyGdnDeltaScanBuffers {
    pub(in crate::runtime) output_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) num_heads: usize,
    pub(in crate::runtime) head_v_dim: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyGdnSsmOutBuffers {
    pub(in crate::runtime) gated_dev: u64,
    pub(in crate::runtime) ssm_out_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) hidden_dim: usize,
    pub(in crate::runtime) d_inner: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct MtpVerifyQwen35RouterBuffers {
    pub(in crate::runtime) logits_dev: u64,
    pub(in crate::runtime) expert_ids_dev: u64,
    pub(in crate::runtime) route_weights_dev: u64,
    pub(in crate::runtime) token_ids_dev: u64,
    pub(in crate::runtime) window_tokens: usize,
    pub(in crate::runtime) n_expert: usize,
    pub(in crate::runtime) n_expert_used: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct Qwen35MtpGdnProjectionRequest<'a> {
    pub(in crate::runtime) attn_norm: &'a [f32],
    pub(in crate::runtime) qkv_q4k: &'a [u8],
    pub(in crate::runtime) qkv_quant: u32,
    pub(in crate::runtime) qkv_rows: usize,
    pub(in crate::runtime) qkv_cols: usize,
    pub(in crate::runtime) gate_q4k: &'a [u8],
    pub(in crate::runtime) gate_quant: u32,
    pub(in crate::runtime) gate_rows: usize,
    pub(in crate::runtime) gate_cols: usize,
    pub(in crate::runtime) alpha_q4k: &'a [u8],
    pub(in crate::runtime) alpha_f32: &'a [f32],
    pub(in crate::runtime) alpha_quant: u32,
    pub(in crate::runtime) alpha_rows: usize,
    pub(in crate::runtime) alpha_cols: usize,
    pub(in crate::runtime) beta_q4k: &'a [u8],
    pub(in crate::runtime) beta_f32: &'a [f32],
    pub(in crate::runtime) beta_quant: u32,
    pub(in crate::runtime) beta_rows: usize,
    pub(in crate::runtime) beta_cols: usize,
    pub(in crate::runtime) norm_eps: f32,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct Qwen35MtpAttentionQkvProjectionRequest<'a> {
    pub(in crate::runtime) attn_norm: &'a [f32],
    pub(in crate::runtime) q_q4k: &'a [u8],
    pub(in crate::runtime) q_quant: u32,
    pub(in crate::runtime) q_rows: usize,
    pub(in crate::runtime) q_cols: usize,
    pub(in crate::runtime) k_q4k: &'a [u8],
    pub(in crate::runtime) k_quant: u32,
    pub(in crate::runtime) k_rows: usize,
    pub(in crate::runtime) k_cols: usize,
    pub(in crate::runtime) v_q4k: &'a [u8],
    pub(in crate::runtime) v_quant: u32,
    pub(in crate::runtime) v_rows: usize,
    pub(in crate::runtime) v_cols: usize,
    pub(in crate::runtime) norm_eps: f32,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct Qwen35MtpAttentionQkNormRopeRequest<'a> {
    pub(in crate::runtime) q_norm: &'a [f32],
    pub(in crate::runtime) k_norm: &'a [f32],
    pub(in crate::runtime) num_heads: usize,
    pub(in crate::runtime) num_kv_heads: usize,
    pub(in crate::runtime) head_dim: usize,
    pub(in crate::runtime) rope_dim: usize,
    pub(in crate::runtime) rope_neox: bool,
    pub(in crate::runtime) rope_theta: f32,
    pub(in crate::runtime) rope_freq_factors: &'a [f32],
    pub(in crate::runtime) pos_start: usize,
    pub(in crate::runtime) norm_eps: f32,
    pub(in crate::runtime) q_unit_offset: bool,
    pub(in crate::runtime) k_unit_offset: bool,
    pub(in crate::runtime) v_no_scale_norm: bool,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct Qwen35MtpAttentionOutputRequest {
    pub(in crate::runtime) num_heads: usize,
    pub(in crate::runtime) num_kv_heads: usize,
    pub(in crate::runtime) scale: f32,
    pub(in crate::runtime) window: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(in crate::runtime) struct Qwen35MtpAttentionOutputWithPriorRequest {
    pub(in crate::runtime) prior_k_bits_dev: u64,
    pub(in crate::runtime) prior_v_bits_dev: u64,
    pub(in crate::runtime) prior_tokens: usize,
    pub(in crate::runtime) num_heads: usize,
    pub(in crate::runtime) num_kv_heads: usize,
    pub(in crate::runtime) scale: f32,
    pub(in crate::runtime) window: usize,
}

#[allow(dead_code)]
pub(in crate::runtime) struct Qwen35MtpGdnMoeLayerRequest<'a> {
    pub(in crate::runtime) projection: Qwen35MtpGdnProjectionRequest<'a>,
    pub(in crate::runtime) conv_state: &'a [f32],
    pub(in crate::runtime) conv_kernel: &'a [f32],
    pub(in crate::runtime) kernel_size: usize,
    pub(in crate::runtime) dt_bias: &'a [f32],
    pub(in crate::runtime) ssm_a: &'a [f32],
    pub(in crate::runtime) num_k_heads: usize,
    pub(in crate::runtime) num_v_heads: usize,
    pub(in crate::runtime) head_k_dim: usize,
    pub(in crate::runtime) head_v_dim: usize,
    pub(in crate::runtime) delta_state: &'a mut [f32],
    pub(in crate::runtime) sync_delta_state_to_host: bool,
    pub(in crate::runtime) ssm_norm: &'a [f32],
    pub(in crate::runtime) ssm_out_q4k: &'a [u8],
    pub(in crate::runtime) ssm_out_quant: u32,
    pub(in crate::runtime) ssm_out_rows: usize,
    pub(in crate::runtime) ssm_out_cols: usize,
    pub(in crate::runtime) post_attn_norm: &'a [f32],
    pub(in crate::runtime) router_w: &'a [f32],
    pub(in crate::runtime) n_expert: usize,
    pub(in crate::runtime) n_expert_used: usize,
    pub(in crate::runtime) gate_all: &'a [u8],
    pub(in crate::runtime) up_all: &'a [u8],
    pub(in crate::runtime) down_all: &'a [u8],
    pub(in crate::runtime) down_quant: u32,
    pub(in crate::runtime) shared_input_scale: &'a [f32],
    pub(in crate::runtime) shared_gate: &'a [u8],
    pub(in crate::runtime) shared_gate_quant: u32,
    pub(in crate::runtime) shared_up: &'a [u8],
    pub(in crate::runtime) shared_up_quant: u32,
    pub(in crate::runtime) shared_down: &'a [u8],
    pub(in crate::runtime) shared_down_quant: u32,
    pub(in crate::runtime) n_ff: usize,
    pub(in crate::runtime) n_embd: usize,
    pub(in crate::runtime) ffn_gate_q4k: &'a [u8],
    pub(in crate::runtime) ffn_gate_rows: usize,
    pub(in crate::runtime) ffn_gate_cols: usize,
    pub(in crate::runtime) ffn_up_q4k: &'a [u8],
    pub(in crate::runtime) ffn_up_rows: usize,
    pub(in crate::runtime) ffn_up_cols: usize,
    pub(in crate::runtime) ffn_down: &'a [u8],
    pub(in crate::runtime) ffn_down_quant: u32,
    pub(in crate::runtime) ffn_down_rows: usize,
    pub(in crate::runtime) ffn_down_cols: usize,
    pub(in crate::runtime) norm_eps: f32,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceVerifyGdnMoeLayer<'a> {
    pub layer_index: usize,
    pub attn_norm: &'a [f32],
    pub qkv_q4k: &'a [u8],
    pub qkv_quant: u32,
    pub qkv_rows: usize,
    pub qkv_cols: usize,
    pub gate_q4k: &'a [u8],
    pub gate_quant: u32,
    pub gate_rows: usize,
    pub gate_cols: usize,
    pub alpha_q4k: &'a [u8],
    pub alpha_f32: &'a [f32],
    pub alpha_quant: u32,
    pub alpha_rows: usize,
    pub alpha_cols: usize,
    pub beta_q4k: &'a [u8],
    pub beta_f32: &'a [f32],
    pub beta_quant: u32,
    pub beta_rows: usize,
    pub beta_cols: usize,
    pub conv_state: &'a [f32],
    pub conv_kernel: &'a [f32],
    pub kernel_size: usize,
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub delta_state: &'a mut [f32],
    pub sync_delta_state_to_host: bool,
    pub ssm_norm: &'a [f32],
    pub ssm_out_q4k: &'a [u8],
    pub ssm_out_quant: u32,
    pub ssm_out_rows: usize,
    pub ssm_out_cols: usize,
    pub post_attn_norm: &'a [f32],
    pub router_w: &'a [f32],
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub down_quant: u32,
    pub shared_input_scale: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_gate_quant: u32,
    pub shared_up: &'a [u8],
    pub shared_up_quant: u32,
    pub shared_down: &'a [u8],
    pub shared_down_quant: u32,
    pub n_ff: usize,
    pub n_embd: usize,
    pub ffn_gate_q4k: &'a [u8],
    pub ffn_gate_rows: usize,
    pub ffn_gate_cols: usize,
    pub ffn_up_q4k: &'a [u8],
    pub ffn_up_rows: usize,
    pub ffn_up_cols: usize,
    pub ffn_down: &'a [u8],
    pub ffn_down_quant: u32,
    pub ffn_down_rows: usize,
    pub ffn_down_cols: usize,
}

#[derive(Debug, Default)]
pub struct Qwen35MtpDeviceVerifyAttentionMoeLayer<'a> {
    pub layer_index: usize,
    pub kv_source_layer: Option<usize>,
    pub attn_norm: &'a [f32],
    pub q_q4k: &'a [u8],
    pub q_quant: u32,
    pub q_rows: usize,
    pub q_cols: usize,
    pub k_q4k: &'a [u8],
    pub k_quant: u32,
    pub k_rows: usize,
    pub k_cols: usize,
    pub v_q4k: &'a [u8],
    pub v_quant: u32,
    pub v_rows: usize,
    pub v_cols: usize,
    pub prior_k_bits: &'a [u16],
    pub prior_v_bits: &'a [u16],
    pub prior_tokens: usize,
    pub prior_sequence_epoch: u64,
    pub kvarn_prior: Option<rnb_backend_api::KvarnDecodeRequest<'a>>,
    pub attention_scale: f32,
    pub o_q4k: &'a [u8],
    pub o_quant: u32,
    pub o_rows: usize,
    pub o_cols: usize,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub qk_norm_unit_offset: bool,
    pub v_no_scale_norm: bool,
    pub rope_freq_factors: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub post_attn_norm_unit_offset: bool,
    pub ffn_norm_unit_offset: bool,
    pub post_ffw_norm_unit_offset: bool,
    pub ffn_norm: &'a [f32],
    pub post_ffw_norm: &'a [f32],
    pub out_scale: &'a [f32],
    pub ple_gate: &'a [u8],
    pub ple_gate_quant: u32,
    pub ple_gate_rows: usize,
    pub ple_gate_cols: usize,
    pub ple_proj: &'a [u8],
    pub ple_proj_quant: u32,
    pub ple_proj_rows: usize,
    pub ple_proj_cols: usize,
    pub ple_post_norm: &'a [f32],
    pub ple_post_norm_unit_offset: bool,
    pub ple_input: &'a [f32],
    pub ffn_uses_gelu: bool,
    pub ffn_gate_q4k: &'a [u8],
    pub ffn_gate_rows: usize,
    pub ffn_gate_cols: usize,
    pub ffn_up_q4k: &'a [u8],
    pub ffn_up_rows: usize,
    pub ffn_up_cols: usize,
    pub ffn_down: &'a [u8],
    pub ffn_down_quant: u32,
    pub ffn_down_rows: usize,
    pub ffn_down_cols: usize,
    pub router_w: &'a [f32],
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_gating_func: u32,
    pub shared_expert_gated: bool,
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub down_quant: u32,
    pub shared_input_scale: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_gate_quant: u32,
    pub shared_up: &'a [u8],
    pub shared_up_quant: u32,
    pub shared_down: &'a [u8],
    pub shared_down_quant: u32,
    pub n_ff: usize,
    pub n_embd: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MtpDeviceVerifyLayerKind {
    AttentionMoe(usize),
    GdnMoe(usize),
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceVerifyRequest<'a> {
    pub verify_tokens: &'a [u32],
    pub prefix_tokens: &'a [usize],
    pub pos_start: usize,
    pub hidden_dim: usize,
    pub rope_dim: usize,
    pub rope_neox: bool,
    pub rope_theta: f32,
    pub include_bonus: bool,
    pub token_embd_q4k: &'a [u8],
    pub token_embd_quant: u32,
    pub token_embd_rows: usize,
    pub token_embd_cols: usize,
    pub embedding_scale: f32,
    pub layer_order: &'a [Qwen35MtpDeviceVerifyLayerKind],
    pub attention_moe_layers: &'a [Qwen35MtpDeviceVerifyAttentionMoeLayer<'a>],
    pub gdn_moe_layers: &'a mut [Qwen35MtpDeviceVerifyGdnMoeLayer<'a>],
    pub output_q6k: &'a [u8],
    pub output_quant: u32,
    pub output_rows: usize,
    pub output_cols: usize,
    pub output_norm: &'a [f32],
    pub norm_eps: f32,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceDraftRequest<'a> {
    pub projected_hidden: &'a [f32],
    pub pos_start: usize,
    pub hidden_dim: usize,
    pub rope_dim: usize,
    pub rope_neox: bool,
    pub rope_theta: f32,
    pub layer: &'a Qwen35MtpDeviceVerifyAttentionMoeLayer<'a>,
    pub output_weight: &'a [u8],
    pub output_quant: u32,
    pub output_rows: usize,
    pub output_cols: usize,
    pub output_norm: &'a [f32],
    pub norm_eps: f32,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceDraftResult {
    pub token: u32,
    pub hidden: Vec<f32>,
    pub attention_kv: Qwen35MtpDeviceVerifyAttentionKvState,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceVerifyResult {
    pub target_tokens: Vec<u32>,
    pub mtp_hidden_rows: Vec<f32>,
    pub hidden_dim: usize,
    pub prefix_states: Vec<Qwen35MtpDeviceVerifyPrefixState>,
    pub ssm_final_states: Vec<Qwen35MtpDeviceVerifySsmLayerFinalState>,
    pub attention_kv_states: Vec<Qwen35MtpDeviceVerifyAttentionKvState>,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceVerifyAttentionKvState {
    pub layer_idx: usize,
    pub window_tokens: usize,
    pub kv_rows: usize,
    pub k_bits: Vec<u16>,
    pub v_bits: Vec<u16>,
    pub device_resident: bool,
}

#[derive(Debug)]
pub(in crate::runtime) struct Qwen35MtpGdnMoeLayerStateCapture {
    pub(in crate::runtime) prefix_states: Vec<Qwen35MtpDeviceVerifyPrefixState>,
    pub(in crate::runtime) final_state: Qwen35MtpDeviceVerifySsmLayerFinalState,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceVerifySsmLayerFinalState {
    pub layer_idx: usize,
    pub conv_state: Vec<f32>,
    pub device_resident: bool,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceVerifyPrefixState {
    pub prefix_tokens: usize,
    pub layers: Vec<Qwen35MtpDeviceVerifySsmLayerPrefixState>,
}

#[derive(Debug)]
pub struct Qwen35MtpDeviceVerifySsmLayerPrefixState {
    pub layer_idx: usize,
    pub conv_state: Vec<f32>,
    pub resident_conv_snapshot: Option<crate::runtime::DeltaStateSnapshot>,
    pub resident_delta_snapshot: Option<crate::runtime::DeltaStateSnapshot>,
}
