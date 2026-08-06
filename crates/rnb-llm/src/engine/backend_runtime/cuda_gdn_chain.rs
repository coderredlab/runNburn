//! cu203: Qwen GDN decode 층 core device chain facade.
//!
//! `gdn_decode.rs` 의 단계별 host↔device 왕복(qkv/gate/alpha/beta/conv/delta/
//! gated norm/ssm_out/residual, 층당 수십 회 memcpy)을 backend chain 한 번으로
//! 대체한다. 지원 조합이 아니면 `Ok(false)` 로 기존 경로에 후퇴한다.
//!
//! conv/delta state 는 backend resident registry 가 진실이 된다 — host 사본
//! sync 는 `materialize_sequence_state`(snapshot/checkpoint) 가 담당하고, 새
//! 시퀀스는 `clear_sequence_state` 가 registry 를 비운다 (기존 delta 계약과 동일).

#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_variables))]

#[cfg(feature = "cuda")]
use super::super::cpu_runtime::kernels;
use super::super::layer_weights::GdnLayerWeights;
use super::super::ModelMetadata;
#[cfg(feature = "cuda")]
use super::super::{cuda_runtime, policy};
#[cfg(feature = "cuda")]
use rnb_loader::GGMLType;

pub(in crate::engine) struct GdnDecodeChainStates<'a> {
    pub conv_state: &'a mut [f32],
    pub delta_state: &'a mut [f32],
}

/// GDN core 를 device chain 으로 실행한다. 성공 시 `hidden` 에 ssm residual add
/// 까지 반영되고 `true` 를 반환한다. FFN 은 caller 가 이어서 실행한다.
pub(in crate::engine) fn try_gdn_decode_core_chain_if_supported(
    metadata: &ModelMetadata,
    w: &GdnLayerWeights,
    states: GdnDecodeChainStates<'_>,
    hidden: &mut [f32],
) -> crate::error::Result<bool> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (metadata, w, states, hidden);
        Ok(false)
    }
    #[cfg(feature = "cuda")]
    {
        if !policy::qwen35_gdn_decode_chain_enabled() {
            return Ok(false);
        }
        if !matches!(w.qkv_weight.ggml_type, GGMLType::Q4_K | GGMLType::Q6_K)
            || w.gate_weight.ggml_type != GGMLType::Q4_K
            || !matches!(
                w.ssm_out.ggml_type,
                GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
            )
            || w.ssm_alpha.ggml_type != GGMLType::F32
            || w.ssm_beta.ggml_type != GGMLType::F32
        {
            return Ok(false);
        }
        let n_embd = metadata.hidden_dim;
        let d_inner = metadata.ssm_d_inner;
        let num_v_heads = metadata.ssm_dt_rank;
        let num_k_heads = metadata.ssm_n_group;
        let head_k_dim = metadata.ssm_d_state;
        let conv_kernel = metadata.ssm_conv_kernel;
        if num_v_heads == 0 || num_k_heads == 0 || num_v_heads % num_k_heads != 0 {
            return Ok(false);
        }
        let head_v_dim = d_inner / num_v_heads;
        let conv_channels = d_inner + 2 * num_k_heads * head_k_dim;
        // q8dot 커널 admission (기존 decode 경로와 같은 rows/blocks 조건).
        if n_embd % 256 != 0
            || d_inner % 256 != 0
            || conv_channels < 1024
            || d_inner < 1024
            || n_embd / 256 < 4
            || d_inner / 256 < 4
        {
            return Ok(false);
        }
        if !cuda_runtime::qwen35_gdn_decode_core_chain_admitted(
            w.qkv_weight.ggml_type,
            w.ssm_out.ggml_type,
            n_embd,
            conv_channels,
            d_inner,
        ) {
            return Ok(false);
        }
        let (Some(qkv_bytes), Some(gate_bytes), Some(ssm_out_bytes)) = (
            w.qkv_weight.data.as_bytes(),
            w.gate_weight.data.as_bytes(),
            w.ssm_out.data.as_bytes(),
        ) else {
            return Ok(false);
        };
        if w.ssm_alpha.data.dtype() != rnb_core::tensor::DType::F32
            || w.ssm_beta.data.dtype() != rnb_core::tensor::DType::F32
        {
            return Ok(false);
        }
        let alpha_f32 = kernels::tensor_as_f32_slice(&w.ssm_alpha.data);
        let beta_f32 = kernels::tensor_as_f32_slice(&w.ssm_beta.data);
        if alpha_f32.len() != num_v_heads * n_embd || beta_f32.len() != num_v_heads * n_embd {
            return Ok(false);
        }
        let attn_norm = kernels::tensor_as_f32_slice(&w.attn_norm);
        let dt_bias = kernels::tensor_as_f32_slice(&w.ssm_dt_bias);
        let ssm_a = kernels::tensor_as_f32_slice(&w.ssm_a);
        let conv_kernel_weights = kernels::tensor_as_f32_slice(&w.ssm_conv1d);
        let ssm_norm = kernels::tensor_as_f32_slice(&w.ssm_norm);
        if attn_norm.len() != n_embd
            || dt_bias.len() != num_v_heads
            || ssm_a.len() != num_v_heads
            || ssm_norm.len() != head_v_dim
            || conv_kernel_weights.len() != conv_kernel * conv_channels
        {
            return Ok(false);
        }
        cuda_runtime::qwen35_gdn_decode_core_chain(cuda_runtime::QwenGdnDecodeChainCall {
            hidden,
            conv_state: states.conv_state,
            delta_state: states.delta_state,
            attn_norm,
            qkv_weights: qkv_bytes,
            qkv_quant: w.qkv_weight.ggml_type,
            gate_weights: gate_bytes,
            alpha_weights: alpha_f32,
            beta_weights: beta_f32,
            dt_bias,
            ssm_a,
            conv_kernel_weights,
            ssm_norm,
            ssm_out_weights: ssm_out_bytes,
            ssm_out_quant: w.ssm_out.ggml_type,
            n_embd,
            conv_channels,
            conv_kernel,
            d_inner,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            norm_eps: metadata.norm_eps,
        })
        .map_err(crate::error::LlmError::Forward)?;
        Ok(true)
    }
}
