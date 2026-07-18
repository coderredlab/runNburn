use rnb_backend_api::{DeviceTensorDesc, DeviceTensorId, MoeRouteSlot};
use rnb_loader::GGMLType;
use rnb_memory::ExpertBundleObservationReceipt;

use super::{backend, dequant, dequant_type, Result};

pub fn qwen_moe_decode_sparse_batch_enabled(route_count: usize, all_high_precision: bool) -> bool {
    backend::tuning::qwen_moe_gate_up_enabled()
        && backend::tuning::qwen_moe_batch_enabled()
        && route_count < 32
        && all_high_precision
}

pub fn qwen_moe_prefill_enabled() -> bool {
    backend::tuning::prefill_moe_enabled()
}

pub fn qwen_moe_prefill_enabled_for_seq(seq_len: usize) -> bool {
    backend::tuning::prefill_moe_enabled_for_seq(seq_len)
}

pub fn qwen35_device_moe_inplace_residual_enabled() -> bool {
    backend::tuning::qwen35_device_moe_inplace_residual_enabled()
}

fn qwen_moe_prefill_route_hist_enabled() -> bool {
    backend::tuning::moe_route_hist_enabled()
}

pub fn log_qwen_moe_prefill_route_hist(
    layer_idx: usize,
    seq_len: usize,
    n_expert: usize,
    sparse_slots: &[MoeRouteSlot],
) {
    if !qwen_moe_prefill_route_hist_enabled() || !(layer_idx == 0 || layer_idx == 3) {
        return;
    }
    let mut counts = vec![0usize; n_expert];
    for slot in sparse_slots {
        counts[slot.expert] += 1;
    }
    let mut nonzero: Vec<usize> = counts.into_iter().filter(|&count| count > 0).collect();
    nonzero.sort_unstable();
    let unique = nonzero.len();
    let max = nonzero.last().copied().unwrap_or(0);
    let p50 = nonzero.get(unique / 2).copied().unwrap_or(0);
    let p90 = nonzero
        .get(unique.saturating_mul(9) / 10)
        .copied()
        .unwrap_or(0);
    eprintln!(
        "[cuda-route-hist] layer={} tokens={} slots={} unique={} max={} p50={} p90={}",
        layer_idx,
        seq_len,
        sparse_slots.len(),
        unique,
        max,
        p50,
        p90
    );
}

#[derive(Debug)]
pub struct MtpDeviceVerifyGdnMoeLayer<'a> {
    pub layer_index: usize,
    pub attn_norm: &'a [f32],
    pub qkv_q4k: &'a [u8],
    pub qkv_quant: GGMLType,
    pub qkv_rows: usize,
    pub qkv_cols: usize,
    pub gate_q4k: &'a [u8],
    pub gate_rows: usize,
    pub gate_cols: usize,
    pub alpha_q4k: &'a [u8],
    pub alpha_f32: &'a [f32],
    pub alpha_quant: GGMLType,
    pub alpha_rows: usize,
    pub alpha_cols: usize,
    pub beta_q4k: &'a [u8],
    pub beta_f32: &'a [f32],
    pub beta_quant: GGMLType,
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
    pub ssm_out_quant: GGMLType,
    pub ssm_out_rows: usize,
    pub ssm_out_cols: usize,
    pub post_attn_norm: &'a [f32],
    pub router_w: &'a [f32],
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub down_quant: GGMLType,
    pub shared_input_scale: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_up: &'a [u8],
    pub shared_down: &'a [u8],
    pub shared_down_quant: GGMLType,
    pub n_ff: usize,
    pub n_embd: usize,
    pub ffn_gate_q4k: &'a [u8],
    pub ffn_gate_rows: usize,
    pub ffn_gate_cols: usize,
    pub ffn_up_q4k: &'a [u8],
    pub ffn_up_rows: usize,
    pub ffn_up_cols: usize,
    pub ffn_down: &'a [u8],
    pub ffn_down_quant: GGMLType,
    pub ffn_down_rows: usize,
    pub ffn_down_cols: usize,
}

#[derive(Debug)]
pub struct MtpDeviceVerifyAttentionMoeLayer<'a> {
    pub layer_index: usize,
    pub attn_norm: &'a [f32],
    pub q_q4k: &'a [u8],
    pub q_quant: GGMLType,
    pub q_rows: usize,
    pub q_cols: usize,
    pub k_q4k: &'a [u8],
    pub k_quant: GGMLType,
    pub k_rows: usize,
    pub k_cols: usize,
    pub v_q4k: &'a [u8],
    pub v_quant: GGMLType,
    pub v_rows: usize,
    pub v_cols: usize,
    pub prior_k_bits: Vec<u16>,
    pub prior_v_bits: Vec<u16>,
    pub prior_tokens: usize,
    pub o_q4k: &'a [u8],
    pub o_quant: GGMLType,
    pub o_rows: usize,
    pub o_cols: usize,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub ffn_gate_q4k: &'a [u8],
    pub ffn_gate_rows: usize,
    pub ffn_gate_cols: usize,
    pub ffn_up_q4k: &'a [u8],
    pub ffn_up_rows: usize,
    pub ffn_up_cols: usize,
    pub ffn_down: &'a [u8],
    pub ffn_down_quant: GGMLType,
    pub ffn_down_rows: usize,
    pub ffn_down_cols: usize,
    pub router_w: &'a [f32],
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub down_quant: GGMLType,
    pub shared_input_scale: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_up: &'a [u8],
    pub shared_down: &'a [u8],
    pub shared_down_quant: GGMLType,
    pub n_ff: usize,
    pub n_embd: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDeviceVerifyLayerKind {
    AttentionMoe(usize),
    GdnMoe(usize),
}

#[derive(Debug)]
pub struct MtpDeviceVerifyWindowRequest<'a> {
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
    pub layer_order: &'a [MtpDeviceVerifyLayerKind],
    pub attention_moe_layers: &'a [MtpDeviceVerifyAttentionMoeLayer<'a>],
    pub gdn_moe_layers: &'a mut [MtpDeviceVerifyGdnMoeLayer<'a>],
    pub output_q6k: &'a [u8],
    pub output_quant: u32,
    pub output_rows: usize,
    pub output_cols: usize,
    pub output_norm: &'a [f32],
    pub norm_eps: f32,
}

#[derive(Debug)]
pub struct MtpDeviceVerifyWindowResult {
    pub target_tokens: Vec<u32>,
    pub mtp_hidden_rows: Vec<f32>,
    pub hidden_dim: usize,
    pub prefix_states: Vec<MtpDeviceVerifyPrefixState>,
    pub ssm_final_states: Vec<MtpDeviceVerifySsmLayerFinalState>,
    pub attention_kv_states: Vec<MtpDeviceVerifyAttentionKvState>,
}

#[derive(Debug)]
pub struct MtpDeviceVerifyPrefixState {
    pub prefix_tokens: usize,
    pub layers: Vec<MtpDeviceVerifySsmLayerPrefixState>,
}

#[derive(Debug)]
pub struct MtpDeviceVerifySsmLayerPrefixState {
    pub layer_idx: usize,
    pub conv_state: Vec<f32>,
    pub resident_delta_snapshot: Option<backend::DeltaStateSnapshot>,
}

#[derive(Debug)]
pub struct MtpDeviceVerifySsmLayerFinalState {
    pub layer_idx: usize,
    pub conv_state: Vec<f32>,
}

#[derive(Debug)]
pub struct MtpDeviceVerifyAttentionKvState {
    pub layer_idx: usize,
    pub window_tokens: usize,
    pub kv_rows: usize,
    pub k_bits: Vec<u16>,
    pub v_bits: Vec<u16>,
}

pub fn qwen35_mtp_device_verify_window(
    request: MtpDeviceVerifyWindowRequest<'_>,
) -> Result<MtpDeviceVerifyWindowResult> {
    let backend_attention_layers = request
        .attention_moe_layers
        .iter()
        .map(|layer| backend::Qwen35MtpDeviceVerifyAttentionMoeLayer {
            layer_index: layer.layer_index,
            attn_norm: layer.attn_norm,
            q_q4k: layer.q_q4k,
            q_quant: layer.q_quant as u32,
            q_rows: layer.q_rows,
            q_cols: layer.q_cols,
            k_q4k: layer.k_q4k,
            k_quant: layer.k_quant as u32,
            k_rows: layer.k_rows,
            k_cols: layer.k_cols,
            v_q4k: layer.v_q4k,
            v_quant: layer.v_quant as u32,
            v_rows: layer.v_rows,
            v_cols: layer.v_cols,
            prior_k_bits: &layer.prior_k_bits,
            prior_v_bits: &layer.prior_v_bits,
            prior_tokens: layer.prior_tokens,
            o_q4k: layer.o_q4k,
            o_quant: layer.o_quant as u32,
            o_rows: layer.o_rows,
            o_cols: layer.o_cols,
            q_norm: layer.q_norm,
            k_norm: layer.k_norm,
            post_attn_norm: layer.post_attn_norm,
            ffn_norm: layer.ffn_norm,
            ffn_gate_q4k: layer.ffn_gate_q4k,
            ffn_gate_rows: layer.ffn_gate_rows,
            ffn_gate_cols: layer.ffn_gate_cols,
            ffn_up_q4k: layer.ffn_up_q4k,
            ffn_up_rows: layer.ffn_up_rows,
            ffn_up_cols: layer.ffn_up_cols,
            ffn_down: layer.ffn_down,
            ffn_down_quant: layer.ffn_down_quant as u32,
            ffn_down_rows: layer.ffn_down_rows,
            ffn_down_cols: layer.ffn_down_cols,
            router_w: layer.router_w,
            n_expert: layer.n_expert,
            n_expert_used: layer.n_expert_used,
            gate_all: layer.gate_all,
            up_all: layer.up_all,
            down_all: layer.down_all,
            down_quant: layer.down_quant as u32,
            shared_input_scale: layer.shared_input_scale,
            shared_gate: layer.shared_gate,
            shared_up: layer.shared_up,
            shared_down: layer.shared_down,
            shared_down_quant: layer.shared_down_quant as u32,
            n_ff: layer.n_ff,
            n_embd: layer.n_embd,
        })
        .collect::<Vec<_>>();
    let backend_layer_order = request
        .layer_order
        .iter()
        .map(|kind| match kind {
            MtpDeviceVerifyLayerKind::AttentionMoe(index) => {
                backend::Qwen35MtpDeviceVerifyLayerKind::AttentionMoe(*index)
            }
            MtpDeviceVerifyLayerKind::GdnMoe(index) => {
                backend::Qwen35MtpDeviceVerifyLayerKind::GdnMoe(*index)
            }
        })
        .collect::<Vec<_>>();
    let mut backend_layers = request
        .gdn_moe_layers
        .iter_mut()
        .map(|layer| backend::Qwen35MtpDeviceVerifyGdnMoeLayer {
            layer_index: layer.layer_index,
            attn_norm: layer.attn_norm,
            qkv_q4k: layer.qkv_q4k,
            qkv_quant: layer.qkv_quant as u32,
            qkv_rows: layer.qkv_rows,
            qkv_cols: layer.qkv_cols,
            gate_q4k: layer.gate_q4k,
            gate_rows: layer.gate_rows,
            gate_cols: layer.gate_cols,
            alpha_q4k: layer.alpha_q4k,
            alpha_f32: layer.alpha_f32,
            alpha_quant: layer.alpha_quant as u32,
            alpha_rows: layer.alpha_rows,
            alpha_cols: layer.alpha_cols,
            beta_q4k: layer.beta_q4k,
            beta_f32: layer.beta_f32,
            beta_quant: layer.beta_quant as u32,
            beta_rows: layer.beta_rows,
            beta_cols: layer.beta_cols,
            conv_state: layer.conv_state,
            conv_kernel: layer.conv_kernel,
            kernel_size: layer.kernel_size,
            dt_bias: layer.dt_bias,
            ssm_a: layer.ssm_a,
            num_k_heads: layer.num_k_heads,
            num_v_heads: layer.num_v_heads,
            head_k_dim: layer.head_k_dim,
            head_v_dim: layer.head_v_dim,
            delta_state: layer.delta_state,
            sync_delta_state_to_host: layer.sync_delta_state_to_host,
            ssm_norm: layer.ssm_norm,
            ssm_out_q4k: layer.ssm_out_q4k,
            ssm_out_quant: layer.ssm_out_quant as u32,
            ssm_out_rows: layer.ssm_out_rows,
            ssm_out_cols: layer.ssm_out_cols,
            post_attn_norm: layer.post_attn_norm,
            router_w: layer.router_w,
            n_expert: layer.n_expert,
            n_expert_used: layer.n_expert_used,
            gate_all: layer.gate_all,
            up_all: layer.up_all,
            down_all: layer.down_all,
            down_quant: layer.down_quant as u32,
            shared_input_scale: layer.shared_input_scale,
            shared_gate: layer.shared_gate,
            shared_up: layer.shared_up,
            shared_down: layer.shared_down,
            shared_down_quant: layer.shared_down_quant as u32,
            n_ff: layer.n_ff,
            n_embd: layer.n_embd,
            ffn_gate_q4k: layer.ffn_gate_q4k,
            ffn_gate_rows: layer.ffn_gate_rows,
            ffn_gate_cols: layer.ffn_gate_cols,
            ffn_up_q4k: layer.ffn_up_q4k,
            ffn_up_rows: layer.ffn_up_rows,
            ffn_up_cols: layer.ffn_up_cols,
            ffn_down: layer.ffn_down,
            ffn_down_quant: layer.ffn_down_quant as u32,
            ffn_down_rows: layer.ffn_down_rows,
            ffn_down_cols: layer.ffn_down_cols,
        })
        .collect::<Vec<_>>();
    backend::qwen35_mtp_device_verify_window(backend::Qwen35MtpDeviceVerifyRequest {
        verify_tokens: request.verify_tokens,
        prefix_tokens: request.prefix_tokens,
        pos_start: request.pos_start,
        hidden_dim: request.hidden_dim,
        rope_dim: request.rope_dim,
        rope_neox: request.rope_neox,
        rope_theta: request.rope_theta,
        include_bonus: request.include_bonus,
        token_embd_q4k: request.token_embd_q4k,
        token_embd_quant: request.token_embd_quant,
        token_embd_rows: request.token_embd_rows,
        token_embd_cols: request.token_embd_cols,
        layer_order: &backend_layer_order,
        attention_moe_layers: &backend_attention_layers,
        gdn_moe_layers: &mut backend_layers,
        output_q6k: request.output_q6k,
        output_quant: request.output_quant,
        output_rows: request.output_rows,
        output_cols: request.output_cols,
        output_norm: request.output_norm,
        norm_eps: request.norm_eps,
    })
    .map(|result| MtpDeviceVerifyWindowResult {
        target_tokens: result.target_tokens,
        mtp_hidden_rows: result.mtp_hidden_rows,
        hidden_dim: result.hidden_dim,
        prefix_states: result
            .prefix_states
            .into_iter()
            .map(|prefix| MtpDeviceVerifyPrefixState {
                prefix_tokens: prefix.prefix_tokens,
                layers: prefix
                    .layers
                    .into_iter()
                    .map(|layer| MtpDeviceVerifySsmLayerPrefixState {
                        layer_idx: layer.layer_idx,
                        conv_state: layer.conv_state,
                        resident_delta_snapshot: layer.resident_delta_snapshot,
                    })
                    .collect(),
            })
            .collect(),
        ssm_final_states: result
            .ssm_final_states
            .into_iter()
            .map(|state| MtpDeviceVerifySsmLayerFinalState {
                layer_idx: state.layer_idx,
                conv_state: state.conv_state,
            })
            .collect(),
        attention_kv_states: result
            .attention_kv_states
            .into_iter()
            .map(|state| MtpDeviceVerifyAttentionKvState {
                layer_idx: state.layer_idx,
                window_tokens: state.window_tokens,
                kv_rows: state.kv_rows,
                k_bits: state.k_bits,
                v_bits: state.v_bits,
            })
            .collect(),
    })
    .map_err(|err| format!("Qwen35 MTP device verify failed: {err}"))
}

#[cfg(test)]
mod mtp_device_verify_tests {
    use super::*;

    #[test]
    fn qwen35_mtp_device_verify_refuses_stub_execution() {
        let verify_tokens = [10_u32, 11];
        let prefix_tokens = [1_usize];
        let token_embd = vec![0u8; 12 * 144];
        let output_q6k = vec![0u8; 8 * 210];
        let output_norm = vec![1.0f32; 256];
        let layer_order = [];
        let attention_moe_layers = [];
        let mut gdn_moe_layers = [];
        let request = MtpDeviceVerifyWindowRequest {
            verify_tokens: &verify_tokens,
            prefix_tokens: &prefix_tokens,
            pos_start: 1139,
            hidden_dim: 256,
            rope_dim: 256,
            rope_neox: true,
            rope_theta: 10000.0,
            include_bonus: false,
            token_embd_q4k: &token_embd,
            token_embd_quant: 12,
            token_embd_rows: 12,
            token_embd_cols: 256,
            layer_order: &layer_order,
            attention_moe_layers: &attention_moe_layers,
            gdn_moe_layers: &mut gdn_moe_layers,
            output_q6k: &output_q6k,
            output_quant: 14,
            output_rows: 8,
            output_cols: 256,
            output_norm: &output_norm,
            norm_eps: 1.0e-5,
        };

        let err = qwen35_mtp_device_verify_window(request).unwrap_err();

        assert!(err.contains("not implemented"));
        assert!(err.contains("bytes="));
        assert!(err.contains("pos_start=1139"));
    }

    #[test]
    fn qwen35_mtp_device_verify_request_carries_gdn_moe_layer_graph() {
        let verify_tokens = [10_u32, 11];
        let prefix_tokens = [1_usize];
        let token_embd = vec![0u8; 12 * 144];
        let output_q6k = vec![0u8; 8 * 210];
        let output_norm = vec![1.0f32; 256];
        let attention_moe_layers = [];
        let layer_order = [];
        let mut delta_state = Vec::<f32>::new();
        let layer = MtpDeviceVerifyGdnMoeLayer {
            layer_index: 0,
            attn_norm: &[],
            qkv_q4k: &[],
            qkv_quant: GGMLType::Q4_K,
            qkv_rows: 0,
            qkv_cols: 0,
            gate_q4k: &[],
            gate_rows: 0,
            gate_cols: 0,
            alpha_q4k: &[],
            alpha_f32: &[],
            alpha_quant: GGMLType::Q4_K,
            alpha_rows: 0,
            alpha_cols: 0,
            beta_q4k: &[],
            beta_f32: &[],
            beta_quant: GGMLType::Q4_K,
            beta_rows: 0,
            beta_cols: 0,
            conv_state: &[],
            conv_kernel: &[],
            kernel_size: 0,
            dt_bias: &[],
            ssm_a: &[],
            num_k_heads: 0,
            num_v_heads: 0,
            head_k_dim: 0,
            head_v_dim: 0,
            delta_state: delta_state.as_mut_slice(),
            sync_delta_state_to_host: false,
            ssm_norm: &[],
            ssm_out_q4k: &[],
            ssm_out_quant: GGMLType::Q4_K,
            ssm_out_rows: 0,
            ssm_out_cols: 0,
            post_attn_norm: &[],
            router_w: &[],
            n_expert: 0,
            n_expert_used: 0,
            gate_all: &[],
            up_all: &[],
            down_all: &[],
            down_quant: GGMLType::Q4_K,
            shared_input_scale: &[],
            shared_gate: &[],
            shared_up: &[],
            shared_down: &[],
            shared_down_quant: GGMLType::Q4_K,
            n_ff: 0,
            n_embd: 256,
            ffn_gate_q4k: &[],
            ffn_gate_rows: 0,
            ffn_gate_cols: 0,
            ffn_up_q4k: &[],
            ffn_up_rows: 0,
            ffn_up_cols: 0,
            ffn_down: &[],
            ffn_down_quant: GGMLType::Q4_K,
            ffn_down_rows: 0,
            ffn_down_cols: 0,
        };
        let mut layers = [layer];
        let request = MtpDeviceVerifyWindowRequest {
            verify_tokens: &verify_tokens,
            prefix_tokens: &prefix_tokens,
            pos_start: 1139,
            hidden_dim: 256,
            rope_dim: 256,
            rope_neox: true,
            rope_theta: 10000.0,
            include_bonus: false,
            token_embd_q4k: &token_embd,
            token_embd_quant: 12,
            token_embd_rows: 12,
            token_embd_cols: 256,
            layer_order: &layer_order,
            attention_moe_layers: &attention_moe_layers,
            gdn_moe_layers: &mut layers,
            output_q6k: &output_q6k,
            output_quant: 14,
            output_rows: 8,
            output_cols: 256,
            output_norm: &output_norm,
            norm_eps: 1.0e-5,
        };

        let err = qwen35_mtp_device_verify_window(request).unwrap_err();

        assert!(err.contains("GDN attn_norm length mismatch"));
    }

    #[test]
    fn qwen35_mtp_device_verify_validates_layer_graph_before_stub_error() {
        let verify_tokens = [10_u32, 11];
        let prefix_tokens = [1_usize];
        let token_embd = vec![0u8; 12 * 144];
        let output_q6k = vec![0u8; 8 * 210];
        let output_norm = vec![1.0f32; 256];
        let attention_moe_layers = [];
        let layer_order = [];
        let mut delta_state = Vec::<f32>::new();
        let layer = MtpDeviceVerifyGdnMoeLayer {
            layer_index: 0,
            attn_norm: &[],
            qkv_q4k: &[],
            qkv_quant: GGMLType::Q4_K,
            qkv_rows: 0,
            qkv_cols: 0,
            gate_q4k: &[],
            gate_rows: 0,
            gate_cols: 0,
            alpha_q4k: &[],
            alpha_f32: &[],
            alpha_quant: GGMLType::Q4_K,
            alpha_rows: 0,
            alpha_cols: 0,
            beta_q4k: &[],
            beta_f32: &[],
            beta_quant: GGMLType::Q4_K,
            beta_rows: 0,
            beta_cols: 0,
            conv_state: &[],
            conv_kernel: &[],
            kernel_size: 0,
            dt_bias: &[],
            ssm_a: &[],
            num_k_heads: 0,
            num_v_heads: 0,
            head_k_dim: 0,
            head_v_dim: 0,
            delta_state: delta_state.as_mut_slice(),
            sync_delta_state_to_host: false,
            ssm_norm: &[],
            ssm_out_q4k: &[],
            ssm_out_quant: GGMLType::Q4_K,
            ssm_out_rows: 0,
            ssm_out_cols: 0,
            post_attn_norm: &[],
            router_w: &[],
            n_expert: 0,
            n_expert_used: 0,
            gate_all: &[],
            up_all: &[],
            down_all: &[],
            down_quant: GGMLType::Q4_K,
            shared_input_scale: &[],
            shared_gate: &[],
            shared_up: &[],
            shared_down: &[],
            shared_down_quant: GGMLType::Q4_K,
            n_ff: 0,
            n_embd: 128,
            ffn_gate_q4k: &[],
            ffn_gate_rows: 0,
            ffn_gate_cols: 0,
            ffn_up_q4k: &[],
            ffn_up_rows: 0,
            ffn_up_cols: 0,
            ffn_down: &[],
            ffn_down_quant: GGMLType::Q4_K,
            ffn_down_rows: 0,
            ffn_down_cols: 0,
        };
        let mut layers = [layer];
        let request = MtpDeviceVerifyWindowRequest {
            verify_tokens: &verify_tokens,
            prefix_tokens: &prefix_tokens,
            pos_start: 1139,
            hidden_dim: 256,
            rope_dim: 256,
            rope_neox: true,
            rope_theta: 10000.0,
            include_bonus: false,
            token_embd_q4k: &token_embd,
            token_embd_quant: 12,
            token_embd_rows: 12,
            token_embd_cols: 256,
            layer_order: &layer_order,
            attention_moe_layers: &attention_moe_layers,
            gdn_moe_layers: &mut layers,
            output_q6k: &output_q6k,
            output_quant: 14,
            output_rows: 8,
            output_cols: 256,
            output_norm: &output_norm,
            norm_eps: 1.0e-5,
        };

        let err = qwen35_mtp_device_verify_window(request).unwrap_err();

        assert!(err.contains("gdn_moe_layers[0] n_embd"));
        assert!(!err.contains("not implemented"));
    }
}

pub fn qwen_moe_prefill_router_logits(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
) -> Result<Vec<f32>> {
    backend::f32_gemm_batch(router_w, n_expert, hidden_dim, norm_all)
        .map_err(|err| format!("CUDA prefill MoE router failed: {err}"))
}

pub fn qwen_moe_prefill_sparse_slots_device_topk(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    seq_len: usize,
    n_expert_used: usize,
) -> Result<Vec<MoeRouteSlot>> {
    let (expert_ids, route_weights, token_ids) = qwen_moe_prefill_sparse_route_arrays_device_topk(
        router_w,
        n_expert,
        hidden_dim,
        norm_all,
        seq_len,
        n_expert_used,
    )?;
    Ok(expert_ids
        .into_iter()
        .zip(route_weights)
        .zip(token_ids)
        .map(|((expert, weight), token)| MoeRouteSlot::new(expert as usize, token, weight))
        .collect())
}

pub fn qwen_moe_prefill_sparse_route_arrays_device_topk(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    seq_len: usize,
    n_expert_used: usize,
) -> Result<(Vec<u32>, Vec<f32>, Vec<u32>)> {
    let (expert_ids, route_weights, token_ids) = backend::qwen35_prefill_device_topk_route_slots(
        router_w,
        n_expert,
        hidden_dim,
        norm_all,
        seq_len,
        n_expert_used,
    )
    .map_err(|err| format!("CUDA prefill MoE device top-k route failed: {err}"))?;
    if expert_ids.len() != route_weights.len() || expert_ids.len() != token_ids.len() {
        return Err("CUDA prefill MoE device top-k returned mismatched arrays".to_string());
    }
    Ok((expert_ids, route_weights, token_ids))
}

pub fn qwen_moe_prefill_sparse_slots(
    router_logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<MoeRouteSlot> {
    let mut sparse_slots = Vec::with_capacity(seq_len * n_expert_used);
    let selected_count = n_expert_used.min(n_expert);
    if selected_count == 0 {
        return sparse_slots;
    }
    for t in 0..seq_len {
        let logits = &router_logits[t * n_expert..(t + 1) * n_expert];
        let mut idx_stack = [0usize; 256];
        let mut idx_vec;
        let idx: &mut [usize] = if n_expert <= idx_stack.len() {
            &mut idx_stack[..n_expert]
        } else {
            idx_vec = vec![0usize; n_expert];
            &mut idx_vec
        };
        for (i, dst) in idx.iter_mut().enumerate() {
            *dst = i;
        }
        if selected_count < n_expert {
            idx.select_nth_unstable_by(selected_count, |&a, &b| {
                logits[b]
                    .partial_cmp(&logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
        }
        let selected = &idx[..selected_count];
        let mut selected_stack = [(0usize, 0.0f32); 32];
        let mut selected_vec;
        let selected_logits: &mut [(usize, f32)] = if selected.len() <= selected_stack.len() {
            &mut selected_stack[..selected.len()]
        } else {
            selected_vec = vec![(0usize, 0.0f32); selected.len()];
            &mut selected_vec
        };
        for (slot, &expert) in selected_logits.iter_mut().zip(selected.iter()) {
            *slot = (expert, logits[expert]);
        }
        sort_selected_logits_for_qwen_route(selected_logits);
        normalize_selected_softmax_routes(selected_logits);
        for &(expert, route) in selected_logits.iter() {
            sparse_slots.push(MoeRouteSlot::new(expert, t as u32, route));
        }
    }
    sparse_slots
}

pub fn qwen_moe_prefill_sparse_slots_expert_major(
    router_logits: &[f32],
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Vec<MoeRouteSlot> {
    let selected_count = n_expert_used.min(n_expert);
    if selected_count == 0 {
        return Vec::new();
    }

    let mut token_major = Vec::with_capacity(seq_len * selected_count);
    let mut expert_counts = vec![0usize; n_expert];
    for t in 0..seq_len {
        let logits = &router_logits[t * n_expert..(t + 1) * n_expert];
        let mut idx_stack = [0usize; 256];
        let mut idx_vec;
        let idx: &mut [usize] = if n_expert <= idx_stack.len() {
            &mut idx_stack[..n_expert]
        } else {
            idx_vec = vec![0usize; n_expert];
            &mut idx_vec
        };
        for (i, dst) in idx.iter_mut().enumerate() {
            *dst = i;
        }
        if selected_count < n_expert {
            idx.select_nth_unstable_by(selected_count, |&a, &b| {
                logits[b]
                    .partial_cmp(&logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
        }
        let selected = &idx[..selected_count];
        let mut selected_stack = [(0usize, 0.0f32); 32];
        let mut selected_vec;
        let selected_logits: &mut [(usize, f32)] = if selected.len() <= selected_stack.len() {
            &mut selected_stack[..selected.len()]
        } else {
            selected_vec = vec![(0usize, 0.0f32); selected.len()];
            &mut selected_vec
        };
        for (slot, &expert) in selected_logits.iter_mut().zip(selected.iter()) {
            *slot = (expert, logits[expert]);
        }
        sort_selected_logits_for_qwen_route(selected_logits);
        normalize_selected_softmax_routes(selected_logits);
        for &(expert, route) in selected_logits.iter() {
            token_major.push(MoeRouteSlot::new(expert, t as u32, route));
            expert_counts[expert] += 1;
        }
    }

    let mut expert_offsets = vec![0usize; n_expert];
    let mut offset = 0usize;
    for (expert, count) in expert_counts.iter().copied().enumerate() {
        expert_offsets[expert] = offset;
        offset += count;
    }
    let mut cursors = expert_offsets;
    let mut expert_major = vec![MoeRouteSlot::new(0, 0, 0.0); token_major.len()];
    for slot in token_major {
        let dst = cursors[slot.expert];
        expert_major[dst] = slot;
        cursors[slot.expert] += 1;
    }
    expert_major
}

fn sort_selected_logits_for_qwen_route(selected_logits: &mut [(usize, f32)]) {
    selected_logits.sort_by(|(expert_a, logit_a), (expert_b, logit_b)| {
        logit_b
            .partial_cmp(logit_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| expert_a.cmp(expert_b))
    });
}

fn normalize_selected_softmax_routes(selected_logits: &mut [(usize, f32)]) {
    if selected_logits.is_empty() {
        return;
    }
    let max_l = selected_logits
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, weight) in selected_logits.iter_mut() {
        *weight = (*weight - max_l).exp();
        sum += *weight;
    }
    for (_, weight) in selected_logits.iter_mut() {
        *weight /= sum;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_shared_expert_batch(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> Result<Vec<f32>> {
    let result = if backend::tuning::shared_f32_enabled() {
        let gate_f32 =
            dequant::dequantize_bytes_to_f32(shared_gate_bytes, dequant_type(shared_gate_quant));
        let up_f32 =
            dequant::dequantize_bytes_to_f32(shared_up_bytes, dequant_type(shared_up_quant));
        let down_f32 =
            dequant::dequantize_bytes_to_f32(shared_down_bytes, dequant_type(shared_down_quant));
        if gate_f32.len() != n_ff * n_embd
            || up_f32.len() != n_ff * n_embd
            || down_f32.len() != n_embd * n_ff
        {
            return Err("CUDA shared f32 MoE dequant shape mismatch".into());
        }
        backend::f32_shared_expert(
            &gate_f32,
            &up_f32,
            &down_f32,
            route_weights,
            n_ff,
            n_embd,
            norm_all,
        )
    } else {
        backend::qwen35_sparse_experts_by_token(
            gate,
            up,
            down,
            route_weights,
            token_ids,
            seq_len,
            shared_down_quant as u32,
            n_ff,
            n_embd,
            norm_all,
        )
    };
    result.map_err(|err| format!("CUDA prefill MoE shared expert batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_combined_f32_shared_sparse_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_route_weights: &[f32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> Result<Vec<f32>> {
    if shared_gate_quant == GGMLType::Q4_K
        && shared_up_quant == GGMLType::Q4_K
        && matches!(shared_down_quant, GGMLType::Q4_K | GGMLType::Q6_K)
    {
        match backend::qwen35_prefill_moe_q4_shared_sparse_by_token_cached(
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            shared_route_weights,
            gate,
            up,
            down,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            shared_down_quant as u32,
            down_quant as u32,
            n_ff,
            n_embd,
            norm_all,
        ) {
            Ok(Some(output)) => return Ok(output),
            Ok(None) => {}
            Err(err) => {
                return Err(format!(
                    "CUDA prefill MoE cached Q4 shared+sparse failed: {err}"
                ))
            }
        }
    }

    let shared_gate =
        dequant::dequantize_bytes_to_f32(shared_gate_bytes, dequant_type(shared_gate_quant));
    let shared_up =
        dequant::dequantize_bytes_to_f32(shared_up_bytes, dequant_type(shared_up_quant));
    let shared_down =
        dequant::dequantize_bytes_to_f32(shared_down_bytes, dequant_type(shared_down_quant));
    backend::qwen35_prefill_moe_f32_shared_sparse_by_token(
        &shared_gate,
        &shared_up,
        &shared_down,
        shared_route_weights,
        gate,
        up,
        down,
        expert_ids,
        route_weights,
        token_ids,
        seq_len,
        down_quant as u32,
        n_ff,
        n_embd,
        norm_all,
    )
    .map_err(|err| format!("CUDA prefill MoE combined shared+sparse failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_combined_f32_shared_sparse_selected_base_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_route_weights: &[f32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> Result<Vec<f32>> {
    if shared_gate_quant == GGMLType::Q4_K
        && shared_up_quant == GGMLType::Q4_K
        && matches!(shared_down_quant, GGMLType::Q4_K | GGMLType::Q6_K)
    {
        match backend::qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            shared_route_weights,
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            shared_down_quant as u32,
            down_quant as u32,
            n_ff,
            n_embd,
            norm_all,
        ) {
            Ok(Some(output)) => return Ok(output),
            Ok(None) => {}
            Err(err) => {
                return Err(format!(
                    "CUDA prefill MoE cached selected-base Q4 shared+sparse failed: {err}"
                ))
            }
        }
    }

    let shared_gate =
        dequant::dequantize_bytes_to_f32(shared_gate_bytes, dequant_type(shared_gate_quant));
    let shared_up =
        dequant::dequantize_bytes_to_f32(shared_up_bytes, dequant_type(shared_up_quant));
    let shared_down =
        dequant::dequantize_bytes_to_f32(shared_down_bytes, dequant_type(shared_down_quant));
    backend::qwen35_prefill_moe_f32_shared_sparse_selected_base_by_token(
        &shared_gate,
        &shared_up,
        &shared_down,
        shared_route_weights,
        gate_all,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        seq_len,
        down_quant as u32,
        n_ff,
        n_embd,
        norm_all,
    )
    .map_err(|err| format!("CUDA prefill MoE selected-base combined shared+sparse failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    shared_route_weights: &[f32],
    seq_len: usize,
    n_expert_used: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    if shared_gate_quant == GGMLType::Q4_K
        && shared_up_quant == GGMLType::Q4_K
        && matches!(shared_down_quant, GGMLType::Q4_K | GGMLType::Q6_K)
    {
        match backend::qwen35_prefill_moe_q4_shared_sparse_device_topk_selected_base_by_token_cached(
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            shared_route_weights,
            gate_all,
            up_all,
            down_all,
            router_w,
            n_expert,
            hidden_dim,
            norm_all,
            seq_len,
            n_expert_used,
            shared_down_quant as u32,
            down_quant as u32,
            n_ff,
            n_embd,
        ) {
            Ok(Some(output)) => return Ok(output),
            Ok(None) => {}
            Err(err) => {
                return Err(format!(
                "CUDA prefill MoE cached device-topk selected-base Q4 shared+sparse failed: {err}"
            ))
            }
        }
    }

    let shared_gate =
        dequant::dequantize_bytes_to_f32(shared_gate_bytes, dequant_type(shared_gate_quant));
    let shared_up =
        dequant::dequantize_bytes_to_f32(shared_up_bytes, dequant_type(shared_up_quant));
    let shared_down =
        dequant::dequantize_bytes_to_f32(shared_down_bytes, dequant_type(shared_down_quant));
    backend::qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_by_token(
        &shared_gate,
        &shared_up,
        &shared_down,
        shared_route_weights,
        gate_all,
        up_all,
        down_all,
        router_w,
        n_expert,
        hidden_dim,
        norm_all,
        seq_len,
        n_expert_used,
        down_quant as u32,
        n_ff,
        n_embd,
    )
    .map_err(|err| {
        format!("CUDA prefill MoE device-topk selected-base combined shared+sparse failed: {err}")
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenMoeDeviceInputOutput {
    pub output_id: DeviceTensorId,
    pub output_desc: DeviceTensorDesc,
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    shared_input_scale: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    input_id: DeviceTensorId,
    input_desc: DeviceTensorDesc,
    residual_id: DeviceTensorId,
    residual_desc: DeviceTensorDesc,
    seq_len: usize,
    n_expert_used: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
) -> Result<Option<QwenMoeDeviceInputOutput>> {
    qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input_impl(
        shared_gate_bytes,
        shared_gate_quant,
        shared_up_bytes,
        shared_up_quant,
        shared_down_bytes,
        shared_down_quant,
        shared_input_scale,
        gate_all,
        up_all,
        down_all,
        router_w,
        n_expert,
        hidden_dim,
        input_id,
        input_desc,
        residual_id,
        residual_desc,
        seq_len,
        n_expert_used,
        down_quant,
        n_ff,
        n_embd,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    shared_input_scale: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    input_id: DeviceTensorId,
    input_desc: DeviceTensorDesc,
    residual_id: DeviceTensorId,
    residual_desc: DeviceTensorDesc,
    seq_len: usize,
    n_expert_used: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
) -> Result<Option<QwenMoeDeviceInputOutput>> {
    qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input_impl(
        shared_gate_bytes,
        shared_gate_quant,
        shared_up_bytes,
        shared_up_quant,
        shared_down_bytes,
        shared_down_quant,
        shared_input_scale,
        gate_all,
        up_all,
        down_all,
        router_w,
        n_expert,
        hidden_dim,
        input_id,
        input_desc,
        residual_id,
        residual_desc,
        seq_len,
        n_expert_used,
        down_quant,
        n_ff,
        n_embd,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn qwen_moe_prefill_combined_f32_shared_sparse_device_topk_selected_base_device_input_impl(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    shared_input_scale: &[f32],
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    input_id: DeviceTensorId,
    input_desc: DeviceTensorDesc,
    residual_id: DeviceTensorId,
    residual_desc: DeviceTensorDesc,
    seq_len: usize,
    n_expert_used: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    reuse_residual_output: bool,
) -> Result<Option<QwenMoeDeviceInputOutput>> {
    if shared_down_quant != GGMLType::F32
        && !matches!(shared_down_quant, GGMLType::Q4_K | GGMLType::Q6_K)
    {
        return Ok(None);
    }
    let shared_gate =
        dequant::dequantize_bytes_to_f32(shared_gate_bytes, dequant_type(shared_gate_quant));
    let shared_up =
        dequant::dequantize_bytes_to_f32(shared_up_bytes, dequant_type(shared_up_quant));
    let shared_down =
        dequant::dequantize_bytes_to_f32(shared_down_bytes, dequant_type(shared_down_quant));
    let output = if reuse_residual_output {
        backend::qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input_reuse_residual(
            &shared_gate,
            &shared_up,
            &shared_down,
            shared_input_scale,
            gate_all,
            up_all,
            down_all,
            router_w,
            n_expert,
            hidden_dim,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            seq_len,
            n_expert_used,
            down_quant as u32,
            n_ff,
            n_embd,
        )
    } else {
        backend::qwen35_prefill_moe_f32_shared_sparse_device_topk_selected_base_device_input(
            &shared_gate,
            &shared_up,
            &shared_down,
            shared_input_scale,
            gate_all,
            up_all,
            down_all,
            router_w,
            n_expert,
            hidden_dim,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            seq_len,
            n_expert_used,
            down_quant as u32,
            n_ff,
            n_embd,
        )
    }
    .map_err(|err| {
        format!("CUDA prefill MoE device-input selected-base combined shared+sparse failed: {err}")
    })?;
    Ok(output.map(|output| QwenMoeDeviceInputOutput {
        output_id: output.output_id,
        output_desc: output.output_desc,
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_combined_f32_shared_sparse_full_layer_by_token(
    shared_gate_bytes: &[u8],
    shared_gate_quant: GGMLType,
    shared_up_bytes: &[u8],
    shared_up_quant: GGMLType,
    shared_down_bytes: &[u8],
    shared_down_quant: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_route_weights: &[f32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> Result<Vec<f32>> {
    if shared_gate_quant == GGMLType::Q4_K
        && shared_up_quant == GGMLType::Q4_K
        && matches!(shared_down_quant, GGMLType::Q4_K | GGMLType::Q6_K)
    {
        match backend::qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached(
            shared_gate_bytes,
            shared_up_bytes,
            shared_down_bytes,
            shared_route_weights,
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            shared_down_quant as u32,
            down_quant as u32,
            n_ff,
            n_embd,
            norm_all,
        ) {
            Ok(Some(output)) => return Ok(output),
            Ok(None) => {}
            Err(err) => {
                return Err(format!(
                    "CUDA prefill MoE cached full-layer Q4 shared+sparse failed: {err}"
                ))
            }
        }
    }

    let shared_gate =
        dequant::dequantize_bytes_to_f32(shared_gate_bytes, dequant_type(shared_gate_quant));
    let shared_up =
        dequant::dequantize_bytes_to_f32(shared_up_bytes, dequant_type(shared_up_quant));
    let shared_down =
        dequant::dequantize_bytes_to_f32(shared_down_bytes, dequant_type(shared_down_quant));
    backend::qwen35_prefill_moe_f32_shared_sparse_full_layer_by_token(
        &shared_gate,
        &shared_up,
        &shared_down,
        shared_route_weights,
        gate_all,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        seq_len,
        down_quant as u32,
        n_ff,
        n_embd,
        norm_all,
    )
    .map_err(|err| format!("CUDA prefill MoE full-layer combined shared+sparse failed: {err}"))
}

pub fn qwen_moe_register_layer(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
) -> Result<bool> {
    backend::qwen35_register_moe_layer(gate_all, up_all, down_all, down_quant as u32, n_ff, n_embd)
        .map_err(|err| format!("CUDA MoE layer registration failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_sparse_experts_by_token(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> Result<Vec<f32>> {
    backend::qwen35_sparse_experts_by_token(
        gate,
        up,
        down,
        route_weights,
        token_ids,
        seq_len,
        down_quant as u32,
        n_ff,
        n_embd,
        norm_all,
    )
    .map_err(|err| format!("CUDA prefill MoE sparse batch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prefill_sparse_experts_selected_base_by_token(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    norm_all: &[f32],
) -> Result<Vec<f32>> {
    backend::qwen35_sparse_experts_selected_base_by_token(
        gate_all,
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        seq_len,
        down_quant as u32,
        n_ff,
        n_embd,
        norm_all,
    )
    .map_err(|err| format!("CUDA prefill MoE selected-base sparse batch failed: {err}"))
}

pub fn qwen_moe_decode_expert(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    if !backend::tuning::qwen_moe_gate_up_enabled() {
        return None;
    }
    backend::qwen35_expert(gate, up, down, down_quant as u32, n_ff, n_embd, input).ok()
}

pub fn qwen_moe_decode_gate_up(
    gate: &[u8],
    up: &[u8],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    if !backend::tuning::qwen_moe_gate_up_enabled() {
        return None;
    }
    Some((
        backend::q4k_gemv(gate, n_ff, n_embd, input).ok()?,
        backend::q4k_gemv(up, n_ff, n_embd, input).ok()?,
    ))
}

pub fn qwen_moe_decode_down(
    down_quant: GGMLType,
    down: &[u8],
    n_embd: usize,
    n_ff: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    if !backend::tuning::qwen_moe_gate_up_enabled() {
        return None;
    }
    match down_quant {
        GGMLType::Q4_K => backend::q4k_gemv(down, n_embd, n_ff, input).ok(),
        GGMLType::Q5_K => backend::q5k_gemv(down, n_embd, n_ff, input).ok(),
        GGMLType::Q6_K => backend::q6k_gemv(down, n_embd, n_ff, input).ok(),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_prepare_selected_bundle_residency(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    n_ff: usize,
    n_embd: usize,
) -> std::result::Result<Vec<bool>, String> {
    backend::qwen35_prepare_selected_bundle_residency(
        gate,
        up,
        down,
        route_weights,
        layer_idx,
        selected_expert_ids,
        bundle_observation_receipt,
        n_ff,
        n_embd,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_decode_sparse_experts_per_slot_resident(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    backend::qwen35_sparse_experts_per_slot_resident(
        gate,
        up,
        down,
        down_quant as u32,
        n_ff,
        n_embd,
        input,
    )
}

pub fn qwen_moe_decode_sparse_experts_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<(), String> {
    backend::qwen35_sparse_experts_into(
        gate,
        up,
        down,
        route_weights,
        layer_idx,
        selected_expert_ids,
        bundle_observation_receipt,
        down_quant as u32,
        n_ff,
        n_embd,
        input,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_decode_sparse_experts_iq4xs_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<(), String> {
    backend::qwen35_sparse_experts_iq4xs_into(
        gate,
        up,
        down,
        route_weights,
        down_quant as u32,
        n_ff,
        n_embd,
        input,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_decode_sparse_experts_add_residual_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    residual: &mut [f32],
) -> std::result::Result<(), String> {
    backend::qwen35_sparse_experts_add_residual_into(
        gate,
        up,
        down,
        route_weights,
        layer_idx,
        selected_expert_ids,
        bundle_observation_receipt,
        down_quant as u32,
        n_ff,
        n_embd,
        input,
        residual,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_decode_sparse_experts_iq4xs_add_residual_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    residual: &mut [f32],
) -> std::result::Result<(), String> {
    backend::qwen35_sparse_experts_iq4xs_add_residual_into(
        gate,
        up,
        down,
        route_weights,
        down_quant as u32,
        n_ff,
        n_embd,
        input,
        residual,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_moe_decode_shared_sparse_experts_into(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route: f32,
    shared_down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> std::result::Result<(), String> {
    backend::qwen35_decode_moe_shared_sparse_into(
        gate,
        up,
        down,
        route_weights,
        layer_idx,
        selected_expert_ids,
        bundle_observation_receipt,
        down_quant as u32,
        shared_gate,
        shared_up,
        shared_down,
        shared_route,
        shared_down_quant as u32,
        n_ff,
        n_embd,
        input,
        out,
    )
}

fn qwen_moe_device_roundtrip_supports_quant(down_quant: GGMLType) -> bool {
    matches!(down_quant, GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K)
}

pub fn qwen_moe_decode_sparse_experts(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    layer_idx: Option<usize>,
    selected_expert_ids: &[usize],
    bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    if backend::tuning::qwen_moe_device_decode_enabled()
        && qwen_moe_device_roundtrip_supports_quant(down_quant)
    {
        backend::qwen35_sparse_experts_device_roundtrip(
            gate,
            up,
            down,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant as u32,
            n_ff,
            n_embd,
            input,
        )
    } else {
        backend::qwen35_sparse_experts(
            gate,
            up,
            down,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant as u32,
            n_ff,
            n_embd,
            input,
        )
    }
}

pub fn qwen_moe_decode_sparse_experts_iq4xs(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    down_quant: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    backend::qwen35_sparse_experts_iq4xs(
        gate,
        up,
        down,
        route_weights,
        down_quant as u32,
        n_ff,
        n_embd,
        input,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_selected_softmax_routes, qwen_moe_device_roundtrip_supports_quant,
        qwen_moe_prefill_sparse_slots,
    };
    use rnb_loader::GGMLType;

    #[test]
    fn device_roundtrip_quant_capability_matches_backend_kernel_table() {
        for quant in [GGMLType::Q4_K, GGMLType::Q5_K, GGMLType::Q6_K] {
            assert!(qwen_moe_device_roundtrip_supports_quant(quant));
        }
        for quant in [
            GGMLType::F32,
            GGMLType::F16,
            GGMLType::BF16,
            GGMLType::Q4_0,
            GGMLType::Q4_1,
            GGMLType::Q5_0,
            GGMLType::Q5_1,
            GGMLType::Q8_0,
            GGMLType::Q8_1,
            GGMLType::Q2_K,
            GGMLType::Q3_K,
            GGMLType::IQ4_XS,
            GGMLType::I32,
        ] {
            assert!(!qwen_moe_device_roundtrip_supports_quant(quant));
        }
    }

    #[test]
    fn selected_softmax_routes_normalizes_only_selected_logits() {
        let mut selected = [(3usize, 7.0f32), (1usize, 5.0f32), (4usize, 4.0f32)];

        normalize_selected_softmax_routes(&mut selected);

        let sum = selected.iter().map(|(_, weight)| *weight).sum::<f32>();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert_eq!(selected[0].0, 3);
        assert!(selected[0].1 > selected[1].1);
        assert!(selected[1].1 > selected[2].1);
    }

    #[test]
    fn selected_softmax_matches_full_softmax_renormalized_routes() {
        let logits = [-3.0f32, 5.0, 1.0, 7.0, 4.0];
        let mut selected = [
            (3usize, logits[3]),
            (1usize, logits[1]),
            (4usize, logits[4]),
        ];

        normalize_selected_softmax_routes(&mut selected);

        let max_all = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut full_probs = [0.0f32; 5];
        let mut full_sum = 0.0f32;
        for (prob, &logit) in full_probs.iter_mut().zip(logits.iter()) {
            *prob = (logit - max_all).exp();
            full_sum += *prob;
        }
        for prob in &mut full_probs {
            *prob /= full_sum;
        }
        let selected_sum = [3usize, 1, 4]
            .iter()
            .map(|&expert| full_probs[expert])
            .sum::<f32>();

        for &(expert, route) in &selected {
            let expected = full_probs[expert] / selected_sum;
            assert!((route - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn expert_major_prefill_sparse_slots_match_sorted_token_major_slots() {
        let router_logits = [
            0.1f32, 3.0, 1.0, 2.0, //
            4.0, 0.5, 2.5, 1.5, //
            0.0, 2.0, 5.0, 1.0,
        ];
        let mut expected = qwen_moe_prefill_sparse_slots(&router_logits, 3, 4, 2);
        expected.sort_unstable_by_key(|slot| (slot.expert, slot.token));

        let actual = super::qwen_moe_prefill_sparse_slots_expert_major(&router_logits, 3, 4, 2);

        assert_eq!(actual, expected);
    }

    #[test]
    fn prefill_sparse_slots_break_tied_logits_by_lower_expert_id() {
        let n_expert = 64usize;
        let router_logits = vec![0.0f32; n_expert];
        let mut actual = qwen_moe_prefill_sparse_slots(&router_logits, 1, n_expert, 2);
        actual.sort_unstable_by_key(|slot| (slot.expert, slot.token));

        let experts = actual.iter().map(|slot| slot.expert).collect::<Vec<_>>();
        assert_eq!(experts, vec![0, 1]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_topk_prefill_sparse_slots_match_host_for_tied_logits() -> Result<(), String> {
        let n_expert = 64usize;
        let hidden_dim = 1usize;
        let seq_len = 2usize;
        let n_expert_used = 2usize;
        let router_w = vec![0.0f32; n_expert * hidden_dim];
        let norm_all = vec![1.0f32; seq_len * hidden_dim];
        let router_logits = vec![0.0f32; seq_len * n_expert];
        let mut expected =
            qwen_moe_prefill_sparse_slots(&router_logits, seq_len, n_expert, n_expert_used);
        expected.sort_unstable_by_key(|slot| (slot.expert, slot.token));

        let mut actual = super::qwen_moe_prefill_sparse_slots_device_topk(
            &router_w,
            n_expert,
            hidden_dim,
            &norm_all,
            seq_len,
            n_expert_used,
        )?;
        actual.sort_unstable_by_key(|slot| (slot.expert, slot.token));

        assert_eq!(
            actual
                .iter()
                .map(|slot| (slot.expert, slot.token))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|slot| (slot.expert, slot.token))
                .collect::<Vec<_>>()
        );
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.weight - expected.weight).abs() < 1.0e-6);
        }
        Ok(())
    }
}
