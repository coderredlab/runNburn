#[allow(unused_imports)]
use crate::engine::layer_weights::LayerType;
#[cfg(feature = "metal")]
use crate::engine::metal_runtime;
#[cfg(feature = "metal")]
use crate::engine::quantized_weight_types::backend_ggml_type;
use crate::engine::quantized_weight_types::QuantizedWeight;
#[cfg(feature = "metal")]
use rnb_loader::GGMLType;

/// chain run 1개 layer 의 입력 state(borrowed). backend 호출이 끝난 뒤 이 borrow 를
/// 명시적으로 해제하고 GDN state 를 kv_cache 에 반영한다.
/// - Attn: 첫 token KV init 용 prior KV(host f16). filled!=0 이면 빈 slice.
/// - Gdn: 직전 token conv/delta state.
#[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
pub(in crate::engine) enum ChainLayerInput<'a> {
    Attn {
        prior_k: &'a [u16],
        prior_v: &'a [u16],
        kvarn: Option<crate::runtime::KvarnChainView<'a>>,
    },
    Gdn {
        conv_state: &'a [f32],
        delta_state: &'a [f32],
    },
}

/// chain run 1개 attention layer 의 shape(decode loop 이 `resolve_attention_layout` +
/// `resolve_rope_params` 로 1회 계산해 넘긴다). carrier entry 생성에 그대로 사용.
#[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
pub(in crate::engine) struct ChainAttnShape {
    pub q_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub n_rot: usize,
    pub pos: usize,
    pub theta: f32,
    pub scale: f32,
    pub post_norm_eps: f32,
    pub apply_rope: bool,
    pub sliding_window: Option<usize>,
    pub muse_semantics: bool,
}

#[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
pub(in crate::engine) struct MetalDecodeOutputArgmax<'a> {
    pub norm_weight: &'a [f32],
    pub output_weight: &'a QuantizedWeight,
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
    pub excluded_token: Option<u32>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
struct QwenMoeChainMaterials<'a> {
    router_w: &'a [f32],
    gate_exps: &'a [u8],
    gate_expert_bytes: usize,
    up_exps: &'a [u8],
    up_expert_bytes: usize,
    down_exps: &'a [u8],
    down_expert_bytes: usize,
    shared_input_scale: &'a [f32],
    shared_gate: &'a [u8],
    shared_up: &'a [u8],
    shared_down: &'a [u8],
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn qwen_moe_chain_materials(
    moe_w: &crate::engine::layer_weights::SharedExpertMoELayerWeights,
) -> std::result::Result<QwenMoeChainMaterials<'_>, &'static str> {
    let shared_q8_0 = moe_w.shared_gate.ggml_type == GGMLType::Q8_0
        && moe_w.shared_up.ggml_type == GGMLType::Q8_0
        && moe_w.shared_down.ggml_type == GGMLType::Q8_0;
    let shared_q4 = moe_w.shared_gate.ggml_type == GGMLType::Q4_K
        && moe_w.shared_up.ggml_type == GGMLType::Q4_K
        && moe_w.shared_down.ggml_type == moe_w.down_quant;
    if moe_w.gate_quant != GGMLType::Q4_K
        || moe_w.up_quant != GGMLType::Q4_K
        || !matches!(
            moe_w.down_quant,
            GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K
        )
        || !(shared_q8_0 || shared_q4)
    {
        return Err("unsupported qwen moe quant");
    }
    let Some(router_w) = moe_w.router_f32() else {
        return Err("missing qwen moe router");
    };
    let (Some(gate_exps), Some(up_exps), Some(down_exps)) = (
        moe_w.gate_exps_bytes(),
        moe_w.up_exps_bytes(),
        moe_w.down_exps_bytes(),
    ) else {
        return Err("missing qwen moe expert raw");
    };
    let (Some(shared_gate), Some(shared_up), Some(shared_down)) = (
        moe_w.shared_gate.data.as_bytes(),
        moe_w.shared_up.data.as_bytes(),
        moe_w.shared_down.data.as_bytes(),
    ) else {
        return Err("missing qwen moe shared raw");
    };
    let Some(expert_bytes) =
        crate::engine::models::shared_expert_moe::moe_types::sparse_expert_bytes(
            moe_w.n_embd,
            moe_w.n_ff,
            moe_w.gate_quant,
            moe_w.up_quant,
            moe_w.down_quant,
        )
    else {
        return Err("unsupported qwen moe expert shape");
    };
    Ok(QwenMoeChainMaterials {
        router_w,
        gate_exps,
        gate_expert_bytes: expert_bytes.gate,
        up_exps,
        up_expert_bytes: expert_bytes.up,
        down_exps,
        down_expert_bytes: expert_bytes.down,
        shared_input_scale: crate::engine::kernels::tensor_as_f32_slice(&moe_w.shared_input_scale),
        shared_gate,
        shared_up,
        shared_down,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::engine) struct MetalDecodeChainRunReport {
    pub did_run: bool,
    pub qwen_moe_layers: usize,
    pub fallback_reason: Option<&'static str>,
    pub output_argmax_token: Option<u32>,
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_decode_chain_runtime_result(
    result: std::result::Result<metal_runtime::MetalDecodeChainReport, String>,
) -> crate::error::Result<MetalDecodeChainRunReport> {
    result
        .map(|report| MetalDecodeChainRunReport {
            did_run: report.did_run,
            qwen_moe_layers: report.qwen_moe_layers,
            fallback_reason: report.fallback_reason,
            output_argmax_token: report.output_argmax.token_id,
        })
        .map_err(crate::error::LlmError::Forward)
}

/// 연속된 attn carrier + GDN carrier layer N개를 단일 command buffer 로 실행하는 facade(2단계).
/// 1단계 GDN-only(`metal_gdn_decode_chain_run`)를 일반화 — `layers` 가 `&LayerType`(attn/gdn
/// 혼합)이라 9B 의 32 layer 를 전부 하나의 command buffer 로 묶는다. 각 layer 에서 quant raw
/// view + f32 weight 를 추출해 seam `MetalChainLayer` enum 으로 변환 → `metal_decode_chain_run`.
///
/// borrow 안전: `inputs`(attn prior KV / gdn conv·delta)는 caller 가 owned clone 으로 넘겨
/// kv_cache 가변 borrow 와 충돌 안 함. attn 은 KV device 소유라 out_states 가 `None`,
/// gdn 만 `Some((conv_new, delta_new))` → caller 가 ssm_state 에 write.
/// GDN shape 는 모든 GDN layer 동일(caller 가 metadata 에서 1회 계산), attn shape 는
/// `attn_shapes`(layer 별, 9B 는 동일하지만 self-contained).
///
/// 모든 layer 가 backend_view + quant code 지원이어야 한다. 하나라도 미충족이면 Ok(false)
/// → caller 가 per-layer 경로로 fallback.
/// milestone 5(MTP): 단일/배치 공용 구현. `out_attn_kv=None` → 단일(batch=1) `decode_chain_run`,
/// `Some` → `decode_chain_run_batched_collect_attn_kv`(B lane; attn window K/V 를 host commit 용
/// 으로 반환). lane 별 report(Vec) 반환 — 단일 wrapper 는 [0] 만 추출한다.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
fn metal_decode_chain_run_impl(
    hidden: &mut [f32],
    batch: usize,
    layers: &[(usize, &LayerType)],
    inputs: &[ChainLayerInput<'_>],
    attn_shapes: &[Option<ChainAttnShape>],
    out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
    out_attn_kv: Option<&mut Vec<Option<(Vec<u16>, Vec<u16>)>>>,
    out_gdn_state_handle: Option<&mut Option<u64>>,
    out_output_logits: Option<&mut Vec<f32>>,
    capacity: usize,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
    output_argmax: Option<MetalDecodeOutputArgmax<'_>>,
) -> crate::error::Result<Vec<MetalDecodeChainRunReport>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use crate::engine::kernels;
        let fallback = |reason: &'static str| {
            vec![
                MetalDecodeChainRunReport {
                    fallback_reason: Some(reason),
                    ..MetalDecodeChainRunReport::default()
                };
                batch
            ]
        };
        assert_eq!(layers.len(), inputs.len());
        assert_eq!(layers.len(), attn_shapes.len());
        assert_eq!(layers.len(), out_states.len());
        if layers.is_empty() {
            return Ok(vec![
                MetalDecodeChainRunReport {
                    did_run: true,
                    ..MetalDecodeChainRunReport::default()
                };
                batch
            ]);
        }
        // 각 layer 의 quant backend_view 를 먼저 모은다(raw borrow 수명을 specs 와 묶기 위해
        // 별도 Vec 에 보관). 하나라도 view 가 없거나 layer 종류 미지원이면 chain 포기.
        // Dense Attn은 q/k/v/o + 3개 FFN, Attention+MoE는 q/k/v/o만, Gdn은
        // qkv/gate/alpha/beta/ssm_out + dense일 때만 3개 FFN view를 보관한다.
        let mut views: Vec<Vec<_>> = Vec::with_capacity(layers.len());
        for (_, lt) in layers {
            match lt {
                LayerType::Attention(w) => {
                    let (Some(q_v), Some(k_v), Some(v_v), Some(o_v)) = (
                        w.q_weight.backend_view(),
                        w.k_weight.backend_view(),
                        w.v_weight.backend_view(),
                        w.o_weight.backend_view(),
                    ) else {
                        return Ok(fallback("missing attention backend view"));
                    };
                    if w.shared_expert_moe.is_some() {
                        views.push(vec![q_v, k_v, v_v, o_v]);
                    } else {
                        let (Some(fg_v), Some(fu_v), Some(fd_v)) = (
                            w.ffn_gate_weight.backend_view(),
                            w.ffn_up_weight.backend_view(),
                            w.ffn_down_weight.backend_view(),
                        ) else {
                            return Ok(fallback("missing attention ffn backend view"));
                        };
                        let mut layer_views = vec![q_v, k_v, v_v, o_v, fg_v, fu_v, fd_v];
                        if let Some(attn_gate) = &w.attn_gate_weight {
                            let Some(attn_gate_v) = attn_gate.backend_view() else {
                                return Ok(fallback("missing attention gate backend view"));
                            };
                            layer_views.push(attn_gate_v);
                        }
                        views.push(layer_views);
                    }
                }
                LayerType::GatedDeltaNet(w) => {
                    let (Some(qkv_v), Some(gate_v), Some(alpha_v), Some(beta_v), Some(ssm_out_v)) = (
                        w.qkv_weight.backend_view(),
                        w.gate_weight.backend_view(),
                        w.ssm_alpha.backend_view(),
                        w.ssm_beta.backend_view(),
                        w.ssm_out.backend_view(),
                    ) else {
                        return Ok(fallback("missing gdn backend view"));
                    };
                    if w.shared_expert_moe.is_some() {
                        views.push(vec![qkv_v, gate_v, alpha_v, beta_v, ssm_out_v]);
                    } else {
                        let (Some(fg_v), Some(fu_v), Some(fd_v)) = (
                            w.ffn_gate_weight.backend_view(),
                            w.ffn_up_weight.backend_view(),
                            w.ffn_down_weight.backend_view(),
                        ) else {
                            return Ok(fallback("missing dense gdn ffn backend view"));
                        };
                        views.push(vec![
                            qkv_v, gate_v, alpha_v, beta_v, ssm_out_v, fg_v, fu_v, fd_v,
                        ]);
                    }
                }
                _ => return Ok(fallback("unsupported chain layer")),
            }
        }
        // views 가 살아있는 동안 raw()/quant() borrow 로 seam spec 구성.
        let mut specs = Vec::with_capacity(layers.len());
        for (i, (layer_idx, lt)) in layers.iter().enumerate() {
            let v = &views[i];
            match lt {
                LayerType::Attention(w) => {
                    let (
                        ChainLayerInput::Attn {
                            prior_k,
                            prior_v,
                            kvarn,
                        },
                        Some(shape),
                    ) = (&inputs[i], &attn_shapes[i])
                    else {
                        return Ok(fallback("attention chain input mismatch"));
                    };
                    let (Some(q_norm), Some(k_norm)) = (w.q_norm.as_ref(), w.k_norm.as_ref())
                    else {
                        return Ok(fallback("missing attention qk norm"));
                    };
                    let q_ggml = backend_ggml_type(v[0].quant());
                    let k_ggml = backend_ggml_type(v[1].quant());
                    let v_ggml = backend_ggml_type(v[2].quant());
                    let o_ggml = backend_ggml_type(v[3].quant());
                    if let Some(moe_w) = w.shared_expert_moe.as_ref() {
                        // q/k/v/o decode GEMV 지원 집합. 실제 code 변환/검증은 runtime
                        // attn_moe_qwen_chain_ref_if_supported 가 수행(단일 소스). 여기선
                        // 미지원 quant 를 조기 fallback 시켜 이유를 명확히 남긴다.
                        let attn_quant_ok = |t: GGMLType| {
                            matches!(
                                t,
                                GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K | GGMLType::Q8_0
                            )
                        };
                        if !attn_quant_ok(q_ggml)
                            || !attn_quant_ok(k_ggml)
                            || !attn_quant_ok(v_ggml)
                            || !attn_quant_ok(o_ggml)
                        {
                            return Ok(fallback("unsupported qwen attention quant"));
                        }
                        let moe = match qwen_moe_chain_materials(moe_w) {
                            Ok(materials) => materials,
                            Err(reason) => return Ok(fallback(reason)),
                        };
                        specs.push(metal_runtime::MetalChainLayer::AttnMoeQwen(
                            metal_runtime::MetalAttnMoeQwenChainLayer {
                                layer: *layer_idx,
                                norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                q_norm_weight: kernels::tensor_as_f32_slice(q_norm),
                                k_norm_weight: kernels::tensor_as_f32_slice(k_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.ffn_norm),
                                q_ggml,
                                q_raw: v[0].raw(),
                                k_ggml,
                                k_raw: v[1].raw(),
                                v_ggml,
                                v_raw: v[2].raw(),
                                o_ggml,
                                o_raw: v[3].raw(),
                                router_w: moe.router_w,
                                gate_exps_ggml: moe_w.gate_quant,
                                gate_exps_raw: moe.gate_exps,
                                gate_expert_bytes: moe.gate_expert_bytes,
                                up_exps_ggml: moe_w.up_quant,
                                up_exps_raw: moe.up_exps,
                                up_expert_bytes: moe.up_expert_bytes,
                                down_exps_ggml: moe_w.down_quant,
                                down_exps_raw: moe.down_exps,
                                down_expert_bytes: moe.down_expert_bytes,
                                shared_input_scale: moe.shared_input_scale,
                                shared_gate_ggml: moe_w.shared_gate.ggml_type,
                                shared_gate_raw: moe.shared_gate,
                                shared_up_ggml: moe_w.shared_up.ggml_type,
                                shared_up_raw: moe.shared_up,
                                shared_down_ggml: moe_w.shared_down.ggml_type,
                                shared_down_raw: moe.shared_down,
                                prior_k,
                                prior_v,
                                kvarn: *kvarn,
                                pos: shape.pos,
                                hidden_dim,
                                q_dim: shape.q_dim,
                                q_out_dim: shape.q_out_dim,
                                kv_dim: shape.kv_dim,
                                head_dim: shape.head_dim,
                                num_heads: shape.num_heads,
                                num_kv_heads: shape.num_kv_heads,
                                n_rot: shape.n_rot,
                                capacity,
                                n_ff: moe_w.n_ff,
                                n_expert: moe_w.n_expert,
                                n_expert_used: moe_w.n_expert_used,
                                eps,
                                theta: shape.theta,
                                scale: shape.scale,
                            },
                        ));
                    } else {
                        let (attn_gate_ggml, attn_gate_raw) = if w.attn_gate_weight.is_some() {
                            (Some(backend_ggml_type(v[7].quant())), Some(v[7].raw()))
                        } else {
                            (None, None)
                        };
                        let post_attn_norm_weight =
                            w.post_attn_norm.as_ref().map(kernels::tensor_as_f32_slice);
                        let post_ffn_norm_weight =
                            w.post_ffw_norm.as_ref().map(kernels::tensor_as_f32_slice);
                        specs.push(metal_runtime::MetalChainLayer::Attn(
                            metal_runtime::MetalAttnChainLayer {
                                layer: *layer_idx,
                                norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                q_norm_weight: kernels::tensor_as_f32_slice(q_norm),
                                k_norm_weight: kernels::tensor_as_f32_slice(k_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.ffn_norm),
                                post_attn_norm_weight,
                                post_ffn_norm_weight,
                                q_ggml,
                                q_raw: v[0].raw(),
                                k_ggml,
                                k_raw: v[1].raw(),
                                v_ggml,
                                v_raw: v[2].raw(),
                                o_ggml,
                                o_raw: v[3].raw(),
                                attn_gate_ggml,
                                attn_gate_raw,
                                ffn_gate_ggml: backend_ggml_type(v[4].quant()),
                                ffn_gate_raw: v[4].raw(),
                                ffn_up_ggml: backend_ggml_type(v[5].quant()),
                                ffn_up_raw: v[5].raw(),
                                ffn_down_ggml: backend_ggml_type(v[6].quant()),
                                ffn_down_raw: v[6].raw(),
                                prior_k,
                                prior_v,
                                pos: shape.pos,
                                hidden_dim,
                                q_dim: shape.q_dim,
                                q_out_dim: shape.q_out_dim,
                                kv_dim: shape.kv_dim,
                                head_dim: shape.head_dim,
                                num_heads: shape.num_heads,
                                num_kv_heads: shape.num_kv_heads,
                                n_rot: shape.n_rot,
                                capacity,
                                ffn_dim: w.ffn_gate_weight.rows,
                                eps,
                                post_norm_eps: shape.post_norm_eps,
                                theta: shape.theta,
                                scale: shape.scale,
                                apply_rope: shape.apply_rope,
                                sliding_window: shape.sliding_window,
                                muse_semantics: shape.muse_semantics,
                            },
                        ));
                    }
                }
                LayerType::GatedDeltaNet(w) => {
                    let ChainLayerInput::Gdn {
                        conv_state,
                        delta_state,
                    } = &inputs[i]
                    else {
                        return Ok(fallback("gdn chain input mismatch"));
                    };
                    if let Some(moe_w) = w.shared_expert_moe.as_ref() {
                        let moe = match qwen_moe_chain_materials(moe_w) {
                            Ok(materials) => materials,
                            Err(reason) => return Ok(fallback(reason)),
                        };
                        specs.push(metal_runtime::MetalChainLayer::GdnMoeQwen(
                            metal_runtime::MetalGdnMoeQwenChainLayer {
                                layer: *layer_idx,
                                conv_state,
                                delta_state,
                                attn_norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                dt_bias_weight: kernels::tensor_as_f32_slice(&w.ssm_dt_bias),
                                ssm_a_weight: kernels::tensor_as_f32_slice(&w.ssm_a),
                                conv1d_weight: kernels::tensor_as_f32_slice(&w.ssm_conv1d),
                                ssm_norm_weight: kernels::tensor_as_f32_slice(&w.ssm_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.post_attn_norm),
                                qkv_ggml: backend_ggml_type(v[0].quant()),
                                qkv_raw: v[0].raw(),
                                gate_ggml: backend_ggml_type(v[1].quant()),
                                gate_raw: v[1].raw(),
                                alpha_ggml: backend_ggml_type(v[2].quant()),
                                alpha_raw: v[2].raw(),
                                beta_ggml: backend_ggml_type(v[3].quant()),
                                beta_raw: v[3].raw(),
                                ssm_out_ggml: backend_ggml_type(v[4].quant()),
                                ssm_out_raw: v[4].raw(),
                                router_w: moe.router_w,
                                gate_exps_ggml: moe_w.gate_quant,
                                gate_exps_raw: moe.gate_exps,
                                gate_expert_bytes: moe.gate_expert_bytes,
                                up_exps_ggml: moe_w.up_quant,
                                up_exps_raw: moe.up_exps,
                                up_expert_bytes: moe.up_expert_bytes,
                                down_exps_ggml: moe_w.down_quant,
                                down_exps_raw: moe.down_exps,
                                down_expert_bytes: moe.down_expert_bytes,
                                shared_input_scale: moe.shared_input_scale,
                                shared_gate_ggml: moe_w.shared_gate.ggml_type,
                                shared_gate_raw: moe.shared_gate,
                                shared_up_ggml: moe_w.shared_up.ggml_type,
                                shared_up_raw: moe.shared_up,
                                shared_down_ggml: moe_w.shared_down.ggml_type,
                                shared_down_raw: moe.shared_down,
                                hidden_dim,
                                conv_channels,
                                conv_kernel,
                                z_dim,
                                num_v_heads,
                                num_k_heads,
                                head_k_dim,
                                head_v_dim,
                                n_ff: moe_w.n_ff,
                                n_expert: moe_w.n_expert,
                                n_expert_used: moe_w.n_expert_used,
                                eps,
                            },
                        ));
                    } else {
                        specs.push(metal_runtime::MetalChainLayer::Gdn(
                            metal_runtime::MetalGdnChainLayer {
                                layer: *layer_idx,
                                conv_state,
                                delta_state,
                                attn_norm_weight: kernels::tensor_as_f32_slice(&w.attn_norm),
                                dt_bias_weight: kernels::tensor_as_f32_slice(&w.ssm_dt_bias),
                                ssm_a_weight: kernels::tensor_as_f32_slice(&w.ssm_a),
                                conv1d_weight: kernels::tensor_as_f32_slice(&w.ssm_conv1d),
                                ssm_norm_weight: kernels::tensor_as_f32_slice(&w.ssm_norm),
                                ffn_norm_weight: kernels::tensor_as_f32_slice(&w.post_attn_norm),
                                qkv_ggml: backend_ggml_type(v[0].quant()),
                                qkv_raw: v[0].raw(),
                                gate_ggml: backend_ggml_type(v[1].quant()),
                                gate_raw: v[1].raw(),
                                alpha_ggml: backend_ggml_type(v[2].quant()),
                                alpha_raw: v[2].raw(),
                                beta_ggml: backend_ggml_type(v[3].quant()),
                                beta_raw: v[3].raw(),
                                ssm_out_ggml: backend_ggml_type(v[4].quant()),
                                ssm_out_raw: v[4].raw(),
                                ffn_gate_ggml: backend_ggml_type(v[5].quant()),
                                ffn_gate_raw: v[5].raw(),
                                ffn_up_ggml: backend_ggml_type(v[6].quant()),
                                ffn_up_raw: v[6].raw(),
                                ffn_down_ggml: backend_ggml_type(v[7].quant()),
                                ffn_down_raw: v[7].raw(),
                                hidden_dim,
                                conv_channels,
                                conv_kernel,
                                z_dim,
                                num_v_heads,
                                num_k_heads,
                                head_k_dim,
                                head_v_dim,
                                ffn_dim: w.ffn_gate_weight.rows,
                                eps,
                            },
                        ));
                    }
                }
                _ => return Ok(fallback("unsupported chain layer")),
            }
        }
        let output_view = output_argmax
            .as_ref()
            .and_then(|tail| tail.output_weight.backend_view());
        let output_tail = match (output_argmax.as_ref(), output_view.as_ref()) {
            (Some(tail), Some(view)) => Some(metal_runtime::MetalDecodeOutputArgmax {
                norm_weight: tail.norm_weight,
                output_ggml: backend_ggml_type(view.quant()),
                output_raw: view.raw(),
                rows: tail.rows,
                cols: tail.cols,
                eps: tail.eps,
                excluded_token: tail.excluded_token,
            }),
            (Some(_), None) => {
                return Ok(fallback("missing output backend view"));
            }
            (None, _) => None,
        };
        return match out_attn_kv {
            Some(out_kv) => {
                let state_handle = out_gdn_state_handle
                    .expect("batched decode chain requires GDN state handle output");
                metal_decode_chain_runtime_result_batched(
                    metal_runtime::metal_decode_chain_run_batched_collect_attn_kv(
                        hidden,
                        batch,
                        &specs,
                        out_states,
                        out_kv,
                        state_handle,
                        output_tail,
                        out_output_logits,
                    ),
                )
            }
            None => metal_decode_chain_runtime_result(metal_runtime::metal_decode_chain_run(
                hidden,
                &specs,
                out_states,
                output_tail,
            ))
            .map(|report| vec![report]),
        };
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            hidden,
            batch,
            layers,
            inputs,
            attn_shapes,
            out_states,
            out_attn_kv,
            out_gdn_state_handle,
            out_output_logits,
            capacity,
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            eps,
            output_argmax,
        );
        Ok(vec![MetalDecodeChainRunReport::default(); batch])
    }
}

/// 단일-토큰 decode chain facade(기존 시그니처 유지). `metal_decode_chain_run_impl` 에 batch=1,
/// out_attn_kv=None 으로 위임하고 lane 0 report 를 반환한다.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
pub(in crate::engine) fn metal_decode_chain_run(
    hidden: &mut [f32],
    layers: &[(usize, &LayerType)],
    inputs: &[ChainLayerInput<'_>],
    attn_shapes: &[Option<ChainAttnShape>],
    out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
    capacity: usize,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
    output_argmax: Option<MetalDecodeOutputArgmax<'_>>,
) -> crate::error::Result<MetalDecodeChainRunReport> {
    metal_decode_chain_run_impl(
        hidden,
        1,
        layers,
        inputs,
        attn_shapes,
        out_states,
        None,
        None,
        None,
        capacity,
        hidden_dim,
        conv_channels,
        conv_kernel,
        z_dim,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        eps,
        output_argmax,
    )
    .map(|reports| reports.into_iter().next().unwrap_or_default())
}

/// milestone 5(MTP): 배치(B-lane) decode chain facade. 단일과 동일한 spec 빌드를 쓰되 `hidden`
/// 은 `[batch*hidden_dim]`, `out_attn_kv` 에 attn layer 별 window post-rope f16 K/V 를 채워
/// 반환한다(엔진이 accept-n 커밋에서 host kv_cache 에 append). 반환: lane 별 report.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(all(feature = "metal", not(feature = "cuda"))),
    allow(dead_code, unused_variables)
)]
pub(in crate::engine) fn metal_decode_chain_run_batched(
    hidden: &mut [f32],
    batch: usize,
    layers: &[(usize, &LayerType)],
    inputs: &[ChainLayerInput<'_>],
    attn_shapes: &[Option<ChainAttnShape>],
    out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
    out_attn_kv: &mut Vec<Option<(Vec<u16>, Vec<u16>)>>,
    out_gdn_state_handle: &mut Option<u64>,
    out_output_logits: Option<&mut Vec<f32>>,
    capacity: usize,
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
    output_argmax: Option<MetalDecodeOutputArgmax<'_>>,
) -> crate::error::Result<Vec<MetalDecodeChainRunReport>> {
    metal_decode_chain_run_impl(
        hidden,
        batch,
        layers,
        inputs,
        attn_shapes,
        out_states,
        Some(out_attn_kv),
        Some(out_gdn_state_handle),
        out_output_logits,
        capacity,
        hidden_dim,
        conv_channels,
        conv_kernel,
        z_dim,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        eps,
        output_argmax,
    )
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_decode_chain_runtime_result_batched(
    result: std::result::Result<Vec<metal_runtime::MetalDecodeChainReport>, String>,
) -> crate::error::Result<Vec<MetalDecodeChainRunReport>> {
    result
        .map(|reports| {
            reports
                .into_iter()
                .map(|report| MetalDecodeChainRunReport {
                    did_run: report.did_run,
                    qwen_moe_layers: report.qwen_moe_layers,
                    fallback_reason: report.fallback_reason,
                    output_argmax_token: report.output_argmax.token_id,
                })
                .collect()
        })
        .map_err(crate::error::LlmError::Forward)
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod metal_decode_chain_facade_tests {
    use super::*;

    #[test]
    fn chain_layer_input_borrows_attention_and_gdn_state() {
        let prior_k = [1, 2];
        let prior_v = [3, 4];
        let conv_state = [5.0];
        let delta_state = [6.0];
        let attention = ChainLayerInput::Attn {
            prior_k: &prior_k,
            prior_v: &prior_v,
            kvarn: None,
        };
        let gdn = ChainLayerInput::Gdn {
            conv_state: &conv_state,
            delta_state: &delta_state,
        };

        match attention {
            ChainLayerInput::Attn {
                prior_k, prior_v, ..
            } => {
                assert_eq!(prior_k, [1, 2]);
                assert_eq!(prior_v, [3, 4]);
            }
            ChainLayerInput::Gdn { .. } => panic!("expected borrowed attention input"),
        }
        match gdn {
            ChainLayerInput::Gdn {
                conv_state,
                delta_state,
            } => {
                assert_eq!(conv_state, [5.0]);
                assert_eq!(delta_state, [6.0]);
            }
            ChainLayerInput::Attn { .. } => panic!("expected borrowed GDN input"),
        }
    }
}
