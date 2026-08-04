#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(feature = "metal")]
use crate::engine::quantized_weight_types::backend_ggml_type;
use crate::engine::quantized_weight_types::QuantizedWeight;

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_attn_decode_into_if_supported(
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> crate::error::Result<bool> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attn_decode_into_if_supported(
            q,
            k_cache,
            v_cache,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            sliding_window,
            has_softcap,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            q,
            k_cache,
            v_cache,
            &attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            sliding_window,
            has_softcap,
        );
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_attn_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    norm_weight: &[f32],
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    o_weight: &QuantizedWeight,
    ffn_norm_weight: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    prior_k: &[u16],
    prior_v: &[u16],
    pos: usize,
    hidden_dim: usize,
    q_dim: usize,
    q_out_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    n_rot: usize,
    capacity: usize,
    ffn_dim: usize,
    eps: f32,
    theta: f32,
    scale: f32,
) -> crate::error::Result<bool> {
    let (Some(q_v), Some(k_v), Some(v_v), Some(o_v), Some(fg_v), Some(fu_v), Some(fd_v)) = (
        q_weight.backend_view(),
        k_weight.backend_view(),
        v_weight.backend_view(),
        o_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attn_layer_into_if_supported(
            layer,
            hidden,
            norm_weight,
            backend_ggml_type(q_v.quant()),
            backend_ggml_type(k_v.quant()),
            backend_ggml_type(v_v.quant()),
            backend_ggml_type(o_v.quant()),
            q_v.raw(),
            k_v.raw(),
            v_v.raw(),
            q_norm_weight,
            k_norm_weight,
            o_v.raw(),
            ffn_norm_weight,
            backend_ggml_type(fg_v.quant()),
            fg_v.raw(),
            backend_ggml_type(fu_v.quant()),
            fu_v.raw(),
            backend_ggml_type(fd_v.quant()),
            fd_v.raw(),
            prior_k,
            prior_v,
            pos,
            hidden_dim,
            q_dim,
            q_out_dim,
            kv_dim,
            head_dim,
            num_heads,
            num_kv_heads,
            n_rot,
            capacity,
            ffn_dim,
            eps,
            theta,
            scale,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (q_v, k_v, v_v, o_v, fg_v, fu_v, fd_v);
        Ok(false)
    }
}

#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
pub(in crate::engine) fn metal_decode_attn_carrier_kv_filled(layer: usize) -> Option<usize> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_attn_carrier_kv_filled(layer);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        None
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_gdn_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_weight: &QuantizedWeight,
    ffn_norm_weight: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    ffn_dim: usize,
    eps: f32,
) -> crate::error::Result<bool> {
    let (
        Some(qkv_v),
        Some(gate_v),
        Some(alpha_v),
        Some(beta_v),
        Some(ssm_out_v),
        Some(fg_v),
        Some(fu_v),
        Some(fd_v),
    ) = (
        qkv_weight.backend_view(),
        gate_weight.backend_view(),
        alpha_weight.backend_view(),
        beta_weight.backend_view(),
        ssm_out_weight.backend_view(),
        ffn_gate_weight.backend_view(),
        ffn_up_weight.backend_view(),
        ffn_down_weight.backend_view(),
    )
    else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_gdn_layer_into_if_supported(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            backend_ggml_type(qkv_v.quant()),
            qkv_v.raw(),
            backend_ggml_type(gate_v.quant()),
            gate_v.raw(),
            backend_ggml_type(alpha_v.quant()),
            alpha_v.raw(),
            backend_ggml_type(beta_v.quant()),
            beta_v.raw(),
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            backend_ggml_type(ssm_out_v.quant()),
            ssm_out_v.raw(),
            ffn_norm_weight,
            backend_ggml_type(fg_v.quant()),
            fg_v.raw(),
            backend_ggml_type(fu_v.quant()),
            fu_v.raw(),
            backend_ggml_type(fd_v.quant()),
            fd_v.raw(),
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            ffn_dim,
            eps,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (qkv_v, gate_v, alpha_v, beta_v, ssm_out_v, fg_v, fu_v, fd_v);
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_gdn_core_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_weight: &QuantizedWeight,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
) -> crate::error::Result<bool> {
    let (Some(qkv_v), Some(gate_v), Some(alpha_v), Some(beta_v), Some(ssm_out_v)) = (
        qkv_weight.backend_view(),
        gate_weight.backend_view(),
        alpha_weight.backend_view(),
        beta_weight.backend_view(),
        ssm_out_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_gdn_core_into_if_supported(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            backend_ggml_type(qkv_v.quant()),
            qkv_v.raw(),
            backend_ggml_type(gate_v.quant()),
            gate_v.raw(),
            backend_ggml_type(alpha_v.quant()),
            alpha_v.raw(),
            backend_ggml_type(beta_v.quant()),
            beta_v.raw(),
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            backend_ggml_type(ssm_out_v.quant()),
            ssm_out_v.raw(),
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            eps,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (qkv_v, gate_v, alpha_v, beta_v, ssm_out_v);
        Ok(false)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_gdn_moe_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    alpha_weight: &QuantizedWeight,
    beta_weight: &QuantizedWeight,
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_weight: &QuantizedWeight,
    ffn_norm_weight: &[f32],
    moe_w: &crate::engine::layer_weights::SharedExpertMoELayerWeights,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
) -> crate::error::Result<bool> {
    let (Some(qkv_v), Some(gate_v), Some(alpha_v), Some(beta_v), Some(ssm_out_v)) = (
        qkv_weight.backend_view(),
        gate_weight.backend_view(),
        alpha_weight.backend_view(),
        beta_weight.backend_view(),
        ssm_out_weight.backend_view(),
    ) else {
        return Ok(false);
    };
    let Some(router_w) = moe_w.router_f32() else {
        return Ok(false);
    };
    let (Some(gate_exps), Some(up_exps), Some(down_exps)) = (
        moe_w.gate_exps_bytes(),
        moe_w.up_exps_bytes(),
        moe_w.down_exps_bytes(),
    ) else {
        return Ok(false);
    };
    let (Some(shared_gate), Some(shared_up), Some(shared_down)) = (
        moe_w.shared_gate.data.as_bytes(),
        moe_w.shared_up.data.as_bytes(),
        moe_w.shared_down.data.as_bytes(),
    ) else {
        return Ok(false);
    };
    let shared_input_scale = crate::engine::kernels::tensor_as_f32_slice(&moe_w.shared_input_scale);
    let expert_bytes = crate::engine::models::shared_expert_moe::moe_types::sparse_expert_bytes(
        moe_w.n_embd,
        moe_w.n_ff,
        moe_w.gate_quant,
        moe_w.up_quant,
        moe_w.down_quant,
    )
    .expect("qwen35 sparse expert bytes");
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return metal_runtime::metal_gdn_moe_layer_into_if_supported(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            backend_ggml_type(qkv_v.quant()),
            qkv_v.raw(),
            backend_ggml_type(gate_v.quant()),
            gate_v.raw(),
            backend_ggml_type(alpha_v.quant()),
            alpha_v.raw(),
            backend_ggml_type(beta_v.quant()),
            beta_v.raw(),
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            backend_ggml_type(ssm_out_v.quant()),
            ssm_out_v.raw(),
            ffn_norm_weight,
            router_w,
            moe_w.gate_quant,
            gate_exps,
            expert_bytes.gate,
            moe_w.up_quant,
            up_exps,
            expert_bytes.up,
            moe_w.down_quant,
            down_exps,
            expert_bytes.down,
            shared_input_scale,
            moe_w.shared_gate.ggml_type,
            shared_gate,
            moe_w.shared_up.ggml_type,
            shared_up,
            moe_w.shared_down.ggml_type,
            shared_down,
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            moe_w.n_ff,
            moe_w.n_expert,
            moe_w.n_expert_used,
            eps,
        )
        .map_err(crate::error::LlmError::Forward);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            qkv_v,
            gate_v,
            alpha_v,
            beta_v,
            ssm_out_v,
            router_w,
            gate_exps,
            up_exps,
            down_exps,
            shared_gate,
            shared_up,
            shared_down,
            shared_input_scale,
            expert_bytes,
        );
        Ok(false)
    }
}
