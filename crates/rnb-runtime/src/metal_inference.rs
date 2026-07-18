pub use rnb_backend_metal::QwenMoeLlamaIdPrefillOutput;
use rnb_backend_metal::{
    QwenMoeLlamaIdPrefillRequest, QwenMoeLlamaIdQuant, QwenMoeLlamaIdQuantSet, TensoropsQuant,
};
use rnb_loader::GGMLType;

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug)]
pub struct MetalPrefillProjTrace {
    pub role: &'static str,
    pub layer_idx: usize,
    pub timing_enabled: bool,
}

pub struct MetalPrefillAtnCoreRequest<'a> {
    pub hidden: &'a [f32],
    pub attn_norm_w: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_weight_ggml: GGMLType,
    pub q_weight_raw: &'a [u8],
    pub q_weight_rows: usize,
    pub q_weight_cols: usize,
    pub k_weight_ggml: GGMLType,
    pub k_weight_raw: &'a [u8],
    pub k_weight_rows: usize,
    pub k_weight_cols: usize,
    pub v_weight_ggml: GGMLType,
    pub v_weight_raw: &'a [u8],
    pub v_weight_rows: usize,
    pub v_weight_cols: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub scale: f32,
    pub norm_eps: f32,
    pub pos_start: usize,
}

pub struct MetalPrefillAtnCoreOut {
    pub attn_out: Vec<f32>,
    pub k_bits: Vec<u16>,
    pub v_bits: Vec<u16>,
}

pub struct MetalPrefillAtnFullLayerRequest<'a> {
    pub core: MetalPrefillAtnCoreRequest<'a>,
    pub o_weight_ggml: GGMLType,
    pub o_weight_raw: &'a [u8],
    pub o_weight_rows: usize,
    pub o_weight_cols: usize,
    pub ffn_norm_w: &'a [f32],
    pub ffn_gate_weight_ggml: GGMLType,
    pub ffn_gate_weight_raw: &'a [u8],
    pub ffn_gate_weight_rows: usize,
    pub ffn_gate_weight_cols: usize,
    pub ffn_up_weight_ggml: GGMLType,
    pub ffn_up_weight_raw: &'a [u8],
    pub ffn_up_weight_rows: usize,
    pub ffn_up_weight_cols: usize,
    pub ffn_down_weight_ggml: GGMLType,
    pub ffn_down_weight_raw: &'a [u8],
    pub ffn_down_weight_rows: usize,
    pub ffn_down_weight_cols: usize,
    pub ffn_dim: usize,
}

pub struct MetalPrefillAtnFullLayerOut {
    pub hidden: Vec<f32>,
    pub k_bits: Vec<u16>,
    pub v_bits: Vec<u16>,
}

pub struct MetalPrefillAtnOTailRequest<'a> {
    pub core: MetalPrefillAtnCoreRequest<'a>,
    pub o_weight_ggml: GGMLType,
    pub o_weight_raw: &'a [u8],
    pub o_weight_rows: usize,
    pub o_weight_cols: usize,
}

pub struct MetalPrefillAtnOTailOut {
    pub hidden: Vec<f32>,
    pub k_bits: Vec<u16>,
    pub v_bits: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35RouteAlgorithm {
    SelectedSoftmaxTopKLowerExpertTieV1,
}

pub struct Qwen35RoutePolicy {
    pub algorithm: Qwen35RouteAlgorithm,
    pub n_expert: usize,
    pub n_expert_used: usize,
}

pub struct MetalQuantWeightRef<'a> {
    pub ggml_type: GGMLType,
    pub raw: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

pub struct MetalGgmlQuantSet {
    pub gate: GGMLType,
    pub up: GGMLType,
    pub down: GGMLType,
}

pub enum MetalGdnWeightRef<'a> {
    F32 {
        values: &'a [f32],
        rows: usize,
        cols: usize,
    },
    Quant(MetalQuantWeightRef<'a>),
}

pub struct MetalPrefillAtnCoreSpec<'a> {
    pub attn_norm_w: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_weight: MetalQuantWeightRef<'a>,
    pub k_weight: MetalQuantWeightRef<'a>,
    pub v_weight: MetalQuantWeightRef<'a>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub scale: f32,
    pub norm_eps: f32,
    pub pos_start: usize,
}

pub struct MetalPrefillAtnOTailSpec<'a> {
    pub core: MetalPrefillAtnCoreSpec<'a>,
    pub o_weight: MetalQuantWeightRef<'a>,
}

pub struct MetalQwenPrefillGdnSpec<'a> {
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_group: usize,
    pub dt_rank: usize,
    pub conv_kernel_size: usize,
    pub attn_norm_w: &'a [f32],
    pub qkv_weight: MetalGdnWeightRef<'a>,
    pub gate_weight: MetalGdnWeightRef<'a>,
    pub alpha_weight: MetalGdnWeightRef<'a>,
    pub beta_weight: MetalGdnWeightRef<'a>,
    pub conv_state: &'a [f32],
    pub conv_kernel: &'a [f32],
    pub dt_bias: &'a [f32],
    pub ssm_a: &'a [f32],
    pub delta_state: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out_weight: MetalGdnWeightRef<'a>,
    pub post_attn_norm_w: &'a [f32],
    pub norm_eps: f32,
}

pub struct MetalQwenMoePrefillWeights<'a> {
    pub ffn_norm_w: &'a [f32],
    pub router_w: &'a [f32],
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_expert_bytes: usize,
    pub down_expert_bytes: usize,
    pub shared_input_scale: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_up: &'a [u8],
    pub shared_down: &'a [u8],
    pub sparse_quant: MetalGgmlQuantSet,
    pub shared_quant: MetalGgmlQuantSet,
    pub route: Qwen35RoutePolicy,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub norm_eps: f32,
}

pub enum MetalQwenPrefillChainLayer<'a> {
    Attention {
        layer_idx: usize,
        core: MetalPrefillAtnOTailSpec<'a>,
        moe: MetalQwenMoePrefillWeights<'a>,
    },
    Gdn {
        layer_idx: usize,
        layer: MetalQwenPrefillGdnSpec<'a>,
        moe: MetalQwenMoePrefillWeights<'a>,
    },
}

pub struct MetalQwenPrefillChainOut {
    pub hidden: Vec<f32>,
    pub attention_kv: Vec<(usize, Vec<u16>, Vec<u16>)>,
    pub gdn_states: Vec<(usize, Vec<f32>, Vec<f32>)>,
    pub hidden_uploads: usize,
    pub hidden_readbacks: usize,
    pub intermediate_hidden_transfers: usize,
}

// SAFETY: MetalBackend 는 !Send+!Sync 라 thread_local 필수.
// PoC: rayon 워커마다 Metal device 하나씩 생성 — overhead 있지만
//      decode-path 는 single-thread 에서 호출되므로 실제론 1개.
// module-level hoist: GEMV 와 FFN chain 이 같은 backend 인스턴스를 공유 →
// resident/carrier 캐시 재사용.
thread_local! {
    static METAL: rnb_backend_metal::MetalBackend = rnb_backend_metal::MetalBackend::new();
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_quant(
    ggml_type: GGMLType,
) -> Option<(rnb_backend_metal::TensoropsQuant, usize)> {
    match ggml_type {
        GGMLType::Q4_K => Some((rnb_backend_metal::TensoropsQuant::Q4K, 144)),
        GGMLType::Q5_K => Some((rnb_backend_metal::TensoropsQuant::Q5K, 176)),
        GGMLType::Q6_K => Some((rnb_backend_metal::TensoropsQuant::Q6K, 210)),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_quant_weight<'a>(
    weight: &MetalQuantWeightRef<'a>,
    expected_rows: usize,
    expected_cols: usize,
) -> Option<rnb_backend_metal::PrefillAtnCoreWeightView<'a>> {
    if weight.rows != expected_rows
        || weight.cols != expected_cols
        || weight.cols == 0
        || weight.cols % 256 != 0
    {
        return None;
    }
    let (quant, block_bytes) = qwen_prefill_chain_quant(weight.ggml_type)?;
    let expected_raw_len = weight
        .rows
        .checked_mul(weight.cols / 256)?
        .checked_mul(block_bytes)?;
    if weight.raw.len() != expected_raw_len {
        return None;
    }
    Some(rnb_backend_metal::PrefillAtnCoreWeightView {
        raw: weight.raw,
        quant,
        rows: weight.rows,
        cols: weight.cols,
    })
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_gdn_weight<'a>(
    weight: &MetalGdnWeightRef<'a>,
    expected_rows: usize,
    expected_cols: usize,
) -> Option<rnb_backend_metal::GdnBackendWeightRef<'a>> {
    match weight {
        MetalGdnWeightRef::F32 { values, rows, cols } => {
            let expected_len = expected_rows.checked_mul(expected_cols)?;
            if *rows != expected_rows || *cols != expected_cols || values.len() != expected_len {
                return None;
            }
            Some(rnb_backend_metal::GdnBackendWeightRef::F32 {
                values: *values,
                rows: *rows,
                cols: *cols,
            })
        }
        MetalGdnWeightRef::Quant(weight) => {
            qwen_prefill_chain_quant_weight(weight, expected_rows, expected_cols)
                .map(rnb_backend_metal::GdnBackendWeightRef::Quant)
        }
    }
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_attention_spec<'a>(
    spec: &MetalPrefillAtnOTailSpec<'a>,
) -> Option<rnb_backend_metal::PrefillAtnOTailBackendSpecRef<'a>> {
    let core = &spec.core;
    if core.seq_len == 0
        || core.num_heads == 0
        || core.num_kv_heads == 0
        || core.hidden_dim == 0
        || core.num_heads % core.num_kv_heads != 0
        || core.head_dim != 256
        || core.pos_start != 0
        || core.n_rot > core.head_dim
        || core.n_rot % 2 != 0
        || !core.norm_eps.is_finite()
        || core.norm_eps <= 0.0
        || (core.n_rot > 0 && (!core.rope_theta.is_finite() || core.rope_theta <= 0.0))
        || !core.scale.is_finite()
    {
        return None;
    }
    let q_dim = core.num_heads.checked_mul(core.head_dim)?;
    let kv_dim = core.num_kv_heads.checked_mul(core.head_dim)?;
    let q_weight_rows = q_dim.checked_mul(2)?;
    let hidden_elements = core.seq_len.checked_mul(core.hidden_dim)?;
    let q_elements = core.seq_len.checked_mul(core.q_dim)?;
    let kv_elements = core.seq_len.checked_mul(core.kv_dim)?;
    if core.q_dim != q_dim
        || core.kv_dim != kv_dim
        || core.attn_norm_w.len() != core.hidden_dim
        || core.q_norm_w.len() != core.head_dim
        || core.k_norm_w.len() != core.head_dim
        || u32::try_from(hidden_elements).is_err()
        || u32::try_from(q_elements).is_err()
        || u32::try_from(kv_elements).is_err()
        || [
            core.seq_len,
            core.num_heads,
            core.num_kv_heads,
            core.head_dim,
            core.hidden_dim,
            core.q_dim,
            q_weight_rows,
            core.kv_dim,
            core.n_rot,
            core.pos_start,
        ]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return None;
    }
    let q_weight = qwen_prefill_chain_quant_weight(&core.q_weight, q_weight_rows, core.hidden_dim)?;
    let k_weight = qwen_prefill_chain_quant_weight(&core.k_weight, core.kv_dim, core.hidden_dim)?;
    let v_weight = qwen_prefill_chain_quant_weight(&core.v_weight, core.kv_dim, core.hidden_dim)?;
    let o_weight = qwen_prefill_chain_quant_weight(&spec.o_weight, core.hidden_dim, core.q_dim)?;
    Some(rnb_backend_metal::PrefillAtnOTailBackendSpecRef {
        core: rnb_backend_metal::PrefillAtnCoreBackendSpecRef {
            attn_norm_w: core.attn_norm_w,
            q_norm_w: core.q_norm_w,
            k_norm_w: core.k_norm_w,
            q_weight,
            k_weight,
            v_weight,
            seq_len: core.seq_len,
            num_heads: core.num_heads,
            num_kv_heads: core.num_kv_heads,
            head_dim: core.head_dim,
            hidden_dim: core.hidden_dim,
            q_dim: core.q_dim,
            kv_dim: core.kv_dim,
            n_rot: core.n_rot,
            rope_theta: core.rope_theta,
            scale: core.scale,
            norm_eps: core.norm_eps,
            pos_start: core.pos_start,
        },
        o_weight,
    })
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_gdn_elements_fit_u32(elements: &[usize]) -> bool {
    elements
        .iter()
        .all(|&elements| u32::try_from(elements).is_ok())
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_gdn_spec<'a>(
    spec: &MetalQwenPrefillGdnSpec<'a>,
) -> Option<rnb_backend_metal::QwenPrefillGdnBackendSpecRef<'a>> {
    if spec.seq_len == 0
        || spec.hidden_dim == 0
        || spec.d_inner == 0
        || spec.d_state == 0
        || spec.d_state > 256
        || spec.n_group == 0
        || spec.dt_rank == 0
        || spec.conv_kernel_size < 2
        || spec.d_inner % spec.dt_rank != 0
        || spec.dt_rank % spec.n_group != 0
    {
        return None;
    }
    let grouped_state = spec.n_group.checked_mul(spec.d_state)?;
    let conv_channels = spec.d_inner.checked_add(grouped_state.checked_mul(2)?)?;
    let head_v_dim = spec.d_inner / spec.dt_rank;
    let conv_state_len = spec
        .conv_kernel_size
        .checked_sub(1)?
        .checked_mul(conv_channels)?;
    let conv_kernel_len = spec.conv_kernel_size.checked_mul(conv_channels)?;
    let delta_state_len = spec.d_inner.checked_mul(spec.d_state)?;
    let hidden_elements = spec.seq_len.checked_mul(spec.hidden_dim)?;
    let conv_rows = spec
        .seq_len
        .checked_add(spec.conv_kernel_size.checked_sub(1)?)?;
    let conv_input_elements = conv_rows.checked_mul(conv_channels)?;
    let conv_data_elements = spec.seq_len.checked_mul(conv_channels)?;
    let qk_elements = spec.seq_len.checked_mul(grouped_state)?;
    let inner_elements = spec.seq_len.checked_mul(spec.d_inner)?;
    let final_qk_width = spec.dt_rank.checked_mul(spec.d_state)?;
    let final_qk_elements = spec.seq_len.checked_mul(final_qk_width)?;
    let su_workspace = spec.d_inner.checked_mul(48)?;
    let kh_workspace = spec.dt_rank.checked_mul(48)?.checked_mul(spec.d_state)?;
    let step5_workspace = spec
        .dt_rank
        .checked_mul(head_v_dim)?
        .checked_mul(spec.d_state)?;
    let step4_a_workspace = spec.dt_rank.checked_mul(48)?.checked_mul(176)?;
    let step4_b_workspace = spec.dt_rank.checked_mul(176)?.checked_mul(head_v_dim)?;
    let step4_workspace = spec.d_inner.checked_mul(48)?;
    if !qwen_prefill_chain_gdn_elements_fit_u32(&[
        hidden_elements,
        delta_state_len,
        conv_input_elements,
        conv_data_elements,
        final_qk_elements,
        qk_elements,
        inner_elements,
        conv_state_len,
        conv_kernel_len,
        su_workspace,
        kh_workspace,
        step5_workspace,
        step4_a_workspace,
        step4_b_workspace,
        step4_workspace,
    ]) {
        return None;
    }
    if spec.attn_norm_w.len() != spec.hidden_dim
        || spec.conv_state.len() != conv_state_len
        || spec.conv_kernel.len() != conv_kernel_len
        || spec.dt_bias.len() != spec.dt_rank
        || spec.ssm_a.len() != spec.dt_rank
        || spec.delta_state.len() != delta_state_len
        || spec.ssm_norm.len() != head_v_dim
        || spec.post_attn_norm_w.len() != spec.hidden_dim
        || [
            spec.seq_len,
            spec.hidden_dim,
            spec.d_inner,
            spec.d_state,
            spec.n_group,
            spec.dt_rank,
            spec.conv_kernel_size,
            conv_channels,
            head_v_dim,
        ]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return None;
    }
    let qkv_weight =
        qwen_prefill_chain_gdn_weight(&spec.qkv_weight, conv_channels, spec.hidden_dim)?;
    let gate_weight =
        qwen_prefill_chain_gdn_weight(&spec.gate_weight, spec.d_inner, spec.hidden_dim)?;
    let alpha_weight =
        qwen_prefill_chain_gdn_weight(&spec.alpha_weight, spec.dt_rank, spec.hidden_dim)?;
    let beta_weight =
        qwen_prefill_chain_gdn_weight(&spec.beta_weight, spec.dt_rank, spec.hidden_dim)?;
    let ssm_out_weight =
        qwen_prefill_chain_gdn_weight(&spec.ssm_out_weight, spec.hidden_dim, spec.d_inner)?;
    Some(rnb_backend_metal::QwenPrefillGdnBackendSpecRef {
        seq_len: spec.seq_len,
        hidden_dim: spec.hidden_dim,
        d_inner: spec.d_inner,
        d_state: spec.d_state,
        n_group: spec.n_group,
        dt_rank: spec.dt_rank,
        conv_kernel_size: spec.conv_kernel_size,
        attn_norm_w: spec.attn_norm_w,
        qkv_weight,
        gate_weight,
        alpha_weight,
        beta_weight,
        conv_state: spec.conv_state,
        conv_kernel: spec.conv_kernel,
        dt_bias: spec.dt_bias,
        ssm_a: spec.ssm_a,
        delta_state: spec.delta_state,
        ssm_norm: spec.ssm_norm,
        ssm_out_weight,
        post_attn_norm_w: spec.post_attn_norm_w,
        norm_eps: spec.norm_eps,
    })
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_moe_quant(quant: GGMLType) -> Option<rnb_backend_metal::QwenMoeLlamaIdQuant> {
    match quant {
        GGMLType::Q4_K => Some(rnb_backend_metal::QwenMoeLlamaIdQuant::Q4K),
        GGMLType::Q5_K => Some(rnb_backend_metal::QwenMoeLlamaIdQuant::Q5K),
        GGMLType::Q6_K => Some(rnb_backend_metal::QwenMoeLlamaIdQuant::Q6K),
        GGMLType::Q8_0 => Some(rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_moe_weight_bytes(
    quant: rnb_backend_metal::QwenMoeLlamaIdQuant,
    input_dim: usize,
    output_dim: usize,
) -> Option<usize> {
    let (block_elements, block_bytes) = match quant {
        rnb_backend_metal::QwenMoeLlamaIdQuant::Q4K => (256, 144),
        rnb_backend_metal::QwenMoeLlamaIdQuant::Q5K => (256, 176),
        rnb_backend_metal::QwenMoeLlamaIdQuant::Q6K => (256, 210),
        rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero => (32, 34),
    };
    if input_dim == 0 || output_dim == 0 || input_dim % block_elements != 0 {
        return None;
    }
    output_dim
        .checked_mul(input_dim / block_elements)?
        .checked_mul(block_bytes)
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_norm_eps_supported(norm_eps: f32) -> bool {
    norm_eps.is_finite() && norm_eps > 0.0
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_moe_spec<'a>(
    spec: &MetalQwenMoePrefillWeights<'a>,
    seq_len: usize,
    hidden_dim: usize,
) -> Option<rnb_backend_metal::QwenMoePrefillBackendSpecRef<'a>> {
    if seq_len == 0
        || spec.hidden_dim != hidden_dim
        || spec.hidden_dim == 0
        || spec.ffn_dim == 0
        || !qwen_prefill_chain_norm_eps_supported(spec.norm_eps)
        || spec.route.n_expert == 0
        || spec.route.n_expert_used == 0
        || spec.route.n_expert_used > spec.route.n_expert
        || spec.route.n_expert_used > 8
        || spec.ffn_norm_w.len() != spec.hidden_dim
        || spec.shared_input_scale.len() != spec.hidden_dim
        || [
            seq_len,
            spec.hidden_dim,
            spec.ffn_dim,
            spec.route.n_expert,
            spec.route.n_expert_used,
        ]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return None;
    }
    let router_len = spec.route.n_expert.checked_mul(spec.hidden_dim)?;
    let route_slots = seq_len.checked_mul(spec.route.n_expert_used)?;
    let hidden_elements = seq_len.checked_mul(spec.hidden_dim)?;
    let ffn_elements = seq_len.checked_mul(spec.ffn_dim)?;
    let expert_token_slots = seq_len.checked_mul(spec.route.n_expert)?;
    if spec.router_w.len() != router_len
        || u32::try_from(route_slots).is_err()
        || u32::try_from(hidden_elements).is_err()
        || u32::try_from(ffn_elements).is_err()
        || u32::try_from(expert_token_slots).is_err()
        || route_slots
            .checked_mul(std::mem::size_of::<u32>())
            .is_none()
        || route_slots.checked_mul(spec.hidden_dim).is_none()
        || route_slots.checked_mul(spec.ffn_dim).is_none()
    {
        return None;
    }
    let sparse_down = match spec.sparse_quant.down {
        GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K => {
            qwen_prefill_chain_moe_quant(spec.sparse_quant.down)?
        }
        _ => return None,
    };
    if spec.sparse_quant.gate != GGMLType::Q4_K || spec.sparse_quant.up != GGMLType::Q4_K {
        return None;
    }
    let shared_quant = match (
        spec.shared_quant.gate,
        spec.shared_quant.up,
        spec.shared_quant.down,
    ) {
        (GGMLType::Q4_K, GGMLType::Q4_K, GGMLType::Q4_K | GGMLType::Q6_K) => {
            rnb_backend_metal::QwenMoeLlamaIdQuantSet {
                gate: rnb_backend_metal::QwenMoeLlamaIdQuant::Q4K,
                up: rnb_backend_metal::QwenMoeLlamaIdQuant::Q4K,
                down: qwen_prefill_chain_moe_quant(spec.shared_quant.down)?,
            }
        }
        (GGMLType::Q8_0, GGMLType::Q8_0, GGMLType::Q8_0) => {
            rnb_backend_metal::QwenMoeLlamaIdQuantSet {
                gate: rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero,
                up: rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero,
                down: rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero,
            }
        }
        _ => return None,
    };
    let sparse_quant = rnb_backend_metal::QwenMoeLlamaIdQuantSet {
        gate: rnb_backend_metal::QwenMoeLlamaIdQuant::Q4K,
        up: rnb_backend_metal::QwenMoeLlamaIdQuant::Q4K,
        down: sparse_down,
    };
    let expected_gate =
        qwen_prefill_chain_moe_weight_bytes(sparse_quant.gate, spec.hidden_dim, spec.ffn_dim)?;
    let expected_up =
        qwen_prefill_chain_moe_weight_bytes(sparse_quant.up, spec.hidden_dim, spec.ffn_dim)?;
    let expected_down =
        qwen_prefill_chain_moe_weight_bytes(sparse_quant.down, spec.ffn_dim, spec.hidden_dim)?;
    let expected_gate_arena = spec.route.n_expert.checked_mul(expected_gate)?;
    let expected_up_arena = spec.route.n_expert.checked_mul(expected_up)?;
    let expected_down_arena = spec.route.n_expert.checked_mul(expected_down)?;
    if spec.gate_expert_bytes != expected_gate
        || spec.up_expert_bytes != expected_up
        || spec.down_expert_bytes != expected_down
        || spec.gate_all.len() != expected_gate_arena
        || spec.up_all.len() != expected_up_arena
        || spec.down_all.len() != expected_down_arena
    {
        return None;
    }
    let shared_gate_bytes =
        qwen_prefill_chain_moe_weight_bytes(shared_quant.gate, spec.hidden_dim, spec.ffn_dim)?;
    let shared_up_bytes =
        qwen_prefill_chain_moe_weight_bytes(shared_quant.up, spec.hidden_dim, spec.ffn_dim)?;
    let shared_down_bytes =
        qwen_prefill_chain_moe_weight_bytes(shared_quant.down, spec.ffn_dim, spec.hidden_dim)?;
    if spec.shared_gate.len() != shared_gate_bytes
        || spec.shared_up.len() != shared_up_bytes
        || spec.shared_down.len() != shared_down_bytes
    {
        return None;
    }
    let route_algorithm = match spec.route.algorithm {
        Qwen35RouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1 => {
            rnb_backend_metal::QwenRouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1
        }
    };
    Some(rnb_backend_metal::QwenMoePrefillBackendSpecRef {
        ffn_norm_w: spec.ffn_norm_w,
        router_w: spec.router_w,
        gate_all: spec.gate_all,
        up_all: spec.up_all,
        down_all: spec.down_all,
        gate_expert_bytes: spec.gate_expert_bytes,
        up_expert_bytes: spec.up_expert_bytes,
        down_expert_bytes: spec.down_expert_bytes,
        shared_input_scale: spec.shared_input_scale,
        shared_gate: spec.shared_gate,
        shared_up: spec.shared_up,
        shared_down: spec.shared_down,
        sparse_quant,
        shared_quant,
        route_algorithm,
        n_expert: spec.route.n_expert,
        n_expert_used: spec.route.n_expert_used,
        hidden_dim: spec.hidden_dim,
        ffn_dim: spec.ffn_dim,
        norm_eps: spec.norm_eps,
    })
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_trace_enabled() -> bool {
    std::env::var("RNB_METAL_QWEN_PREFILL_CHAIN_TRACE").as_deref() == Ok("1")
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_trace_reject(layer_idx: Option<usize>, reason: &str) {
    if qwen_prefill_chain_trace_enabled() {
        eprintln!(
            "RNB_METAL_QWEN_PREFILL_CHAIN reject layer={} reason={reason}",
            layer_idx
                .map(|layer_idx| layer_idx.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
    }
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_expected_quant_bytes(weight: &MetalQuantWeightRef<'_>) -> Option<usize> {
    let (_, block_bytes) = qwen_prefill_chain_quant(weight.ggml_type)?;
    if weight.cols == 0 || weight.cols % 256 != 0 {
        return None;
    }
    weight
        .rows
        .checked_mul(weight.cols / 256)?
        .checked_mul(block_bytes)
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_trace_attention_reject(
    layer_idx: usize,
    spec: &MetalPrefillAtnOTailSpec<'_>,
) {
    if !qwen_prefill_chain_trace_enabled() {
        return;
    }
    let core = &spec.core;
    let q_dim = core.num_heads.checked_mul(core.head_dim);
    let kv_dim = core.num_kv_heads.checked_mul(core.head_dim);
    let q_rows = q_dim.and_then(|q_dim| q_dim.checked_mul(2));
    eprintln!(
        "RNB_METAL_QWEN_PREFILL_CHAIN reject layer={layer_idx} stage=attention \
         seq={} hidden={} heads={} kv_heads={} head_dim={} pos_start={} n_rot={} \
         q_dim={}/expected={:?} kv_dim={}/expected={:?} \
         norms={}/{}/{} expected={}/{}/{} \
         q={:?}:{:?}x{} bytes={}/expected={:?} \
         k={:?}:{}x{} bytes={}/expected={:?} \
         v={:?}:{}x{} bytes={}/expected={:?} \
         o={:?}:{}x{} bytes={}/expected={:?}",
        core.seq_len,
        core.hidden_dim,
        core.num_heads,
        core.num_kv_heads,
        core.head_dim,
        core.pos_start,
        core.n_rot,
        core.q_dim,
        q_dim,
        core.kv_dim,
        kv_dim,
        core.attn_norm_w.len(),
        core.q_norm_w.len(),
        core.k_norm_w.len(),
        core.hidden_dim,
        core.head_dim,
        core.head_dim,
        core.q_weight.ggml_type,
        q_rows,
        core.q_weight.cols,
        core.q_weight.raw.len(),
        qwen_prefill_chain_expected_quant_bytes(&core.q_weight),
        core.k_weight.ggml_type,
        core.k_weight.rows,
        core.k_weight.cols,
        core.k_weight.raw.len(),
        qwen_prefill_chain_expected_quant_bytes(&core.k_weight),
        core.v_weight.ggml_type,
        core.v_weight.rows,
        core.v_weight.cols,
        core.v_weight.raw.len(),
        qwen_prefill_chain_expected_quant_bytes(&core.v_weight),
        spec.o_weight.ggml_type,
        spec.o_weight.rows,
        spec.o_weight.cols,
        spec.o_weight.raw.len(),
        qwen_prefill_chain_expected_quant_bytes(&spec.o_weight),
    );
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_trace_gdn_reject(layer_idx: usize, spec: &MetalQwenPrefillGdnSpec<'_>) {
    if !qwen_prefill_chain_trace_enabled() {
        return;
    }
    let conv_channels = spec
        .n_group
        .checked_mul(spec.d_state)
        .and_then(|grouped| grouped.checked_mul(2))
        .and_then(|grouped| spec.d_inner.checked_add(grouped));
    let conv_state_expected = conv_channels.and_then(|channels| {
        spec.conv_kernel_size
            .checked_sub(1)
            .and_then(|kernel| kernel.checked_mul(channels))
    });
    let conv_kernel_expected =
        conv_channels.and_then(|channels| spec.conv_kernel_size.checked_mul(channels));
    let delta_expected = spec.d_inner.checked_mul(spec.d_state);
    let head_v_dim = (spec.dt_rank != 0).then(|| spec.d_inner / spec.dt_rank);
    eprintln!(
        "RNB_METAL_QWEN_PREFILL_CHAIN reject layer={layer_idx} stage=gdn \
         seq={} hidden={} d_inner={} d_state={} n_group={} dt_rank={} kernel={} \
         conv_channels={:?} conv_state={}/expected={:?} conv_kernel={}/expected={:?} \
         delta_state={}/expected={:?} attn_norm={}/expected={} dt_bias={}/expected={} \
         ssm_a={}/expected={} ssm_norm={}/expected={:?} post_norm={}/expected={} \
         qkv={} gate={} alpha={} beta={} ssm_out={}",
        spec.seq_len,
        spec.hidden_dim,
        spec.d_inner,
        spec.d_state,
        spec.n_group,
        spec.dt_rank,
        spec.conv_kernel_size,
        conv_channels,
        spec.conv_state.len(),
        conv_state_expected,
        spec.conv_kernel.len(),
        conv_kernel_expected,
        spec.delta_state.len(),
        delta_expected,
        spec.attn_norm_w.len(),
        spec.hidden_dim,
        spec.dt_bias.len(),
        spec.dt_rank,
        spec.ssm_a.len(),
        spec.dt_rank,
        spec.ssm_norm.len(),
        head_v_dim,
        spec.post_attn_norm_w.len(),
        spec.hidden_dim,
        qwen_prefill_chain_gdn_weight_trace(&spec.qkv_weight),
        qwen_prefill_chain_gdn_weight_trace(&spec.gate_weight),
        qwen_prefill_chain_gdn_weight_trace(&spec.alpha_weight),
        qwen_prefill_chain_gdn_weight_trace(&spec.beta_weight),
        qwen_prefill_chain_gdn_weight_trace(&spec.ssm_out_weight),
    );
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_gdn_weight_trace(weight: &MetalGdnWeightRef<'_>) -> String {
    match weight {
        MetalGdnWeightRef::F32 { values, rows, cols } => {
            format!("F32:{rows}x{cols}:values={}", values.len())
        }
        MetalGdnWeightRef::Quant(weight) => format!(
            "{:?}:{}x{}:bytes={}:expected={:?}",
            weight.ggml_type,
            weight.rows,
            weight.cols,
            weight.raw.len(),
            qwen_prefill_chain_expected_quant_bytes(weight)
        ),
    }
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_trace_moe_reject(
    layer_idx: usize,
    seq_len: usize,
    hidden_dim: usize,
    spec: &MetalQwenMoePrefillWeights<'_>,
) {
    if !qwen_prefill_chain_trace_enabled() {
        return;
    }
    let sparse_gate = qwen_prefill_chain_moe_quant(spec.sparse_quant.gate);
    let sparse_up = qwen_prefill_chain_moe_quant(spec.sparse_quant.up);
    let sparse_down = qwen_prefill_chain_moe_quant(spec.sparse_quant.down);
    let shared_gate = qwen_prefill_chain_moe_quant(spec.shared_quant.gate);
    let shared_up = qwen_prefill_chain_moe_quant(spec.shared_quant.up);
    let shared_down = qwen_prefill_chain_moe_quant(spec.shared_quant.down);
    let gate_bytes = sparse_gate.and_then(|quant| {
        qwen_prefill_chain_moe_weight_bytes(quant, spec.hidden_dim, spec.ffn_dim)
    });
    let up_bytes = sparse_up.and_then(|quant| {
        qwen_prefill_chain_moe_weight_bytes(quant, spec.hidden_dim, spec.ffn_dim)
    });
    let down_bytes = sparse_down.and_then(|quant| {
        qwen_prefill_chain_moe_weight_bytes(quant, spec.ffn_dim, spec.hidden_dim)
    });
    let shared_gate_bytes = shared_gate.and_then(|quant| {
        qwen_prefill_chain_moe_weight_bytes(quant, spec.hidden_dim, spec.ffn_dim)
    });
    let shared_up_bytes = shared_up.and_then(|quant| {
        qwen_prefill_chain_moe_weight_bytes(quant, spec.hidden_dim, spec.ffn_dim)
    });
    let shared_down_bytes = shared_down.and_then(|quant| {
        qwen_prefill_chain_moe_weight_bytes(quant, spec.ffn_dim, spec.hidden_dim)
    });
    eprintln!(
        "RNB_METAL_QWEN_PREFILL_CHAIN reject layer={layer_idx} stage=moe \
         seq={seq_len} hidden={}/chain_hidden={hidden_dim} ffn={} experts={}/used={} \
         norms={}/expected={} router={}/expected={:?} shared_scale={}/expected={} \
         sparse={:?}/{:?}/{:?} bytes_per={}/{}/{} expected={:?}/{:?}/{:?} \
         arenas={}/{}/{} expected={:?}/{:?}/{:?} \
         shared={:?}/{:?}/{:?} bytes={}/{}/{} expected={:?}/{:?}/{:?}",
        spec.hidden_dim,
        spec.ffn_dim,
        spec.route.n_expert,
        spec.route.n_expert_used,
        spec.ffn_norm_w.len(),
        spec.hidden_dim,
        spec.router_w.len(),
        spec.route.n_expert.checked_mul(spec.hidden_dim),
        spec.shared_input_scale.len(),
        spec.hidden_dim,
        spec.sparse_quant.gate,
        spec.sparse_quant.up,
        spec.sparse_quant.down,
        spec.gate_expert_bytes,
        spec.up_expert_bytes,
        spec.down_expert_bytes,
        gate_bytes,
        up_bytes,
        down_bytes,
        spec.gate_all.len(),
        spec.up_all.len(),
        spec.down_all.len(),
        gate_bytes.and_then(|bytes| bytes.checked_mul(spec.route.n_expert)),
        up_bytes.and_then(|bytes| bytes.checked_mul(spec.route.n_expert)),
        down_bytes.and_then(|bytes| bytes.checked_mul(spec.route.n_expert)),
        spec.shared_quant.gate,
        spec.shared_quant.up,
        spec.shared_quant.down,
        spec.shared_gate.len(),
        spec.shared_up.len(),
        spec.shared_down.len(),
        shared_gate_bytes,
        shared_up_bytes,
        shared_down_bytes,
    );
}

#[cfg(target_os = "macos")]
fn qwen_prefill_chain_layer_is_contiguous(
    previous_layer_idx: Option<usize>,
    layer_idx: usize,
) -> bool {
    previous_layer_idx
        .map(|previous| previous.checked_add(1) == Some(layer_idx))
        .unwrap_or(true)
}

#[cfg(target_os = "macos")]
pub fn metal_qwen_prefill_chain_run(
    hidden: &[f32],
    layers: &[MetalQwenPrefillChainLayer<'_>],
) -> Result<Option<MetalQwenPrefillChainOut>> {
    if layers.is_empty() {
        qwen_prefill_chain_trace_reject(None, "conversion_empty_layers");
        return Ok(None);
    }
    let mut backend_specs = Vec::with_capacity(layers.len());
    let mut chain_shape: Option<(usize, usize)> = None;
    let mut previous_layer_idx: Option<usize> = None;
    for layer in layers {
        let (layer_idx, seq_len, hidden_dim) = match layer {
            MetalQwenPrefillChainLayer::Attention {
                layer_idx, core, ..
            } => (*layer_idx, core.core.seq_len, core.core.hidden_dim),
            MetalQwenPrefillChainLayer::Gdn {
                layer_idx, layer, ..
            } => (*layer_idx, layer.seq_len, layer.hidden_dim),
        };
        if !qwen_prefill_chain_layer_is_contiguous(previous_layer_idx, layer_idx) {
            qwen_prefill_chain_trace_reject(
                Some(layer_idx),
                &format!(
                    "conversion_non_contiguous_layer previous={previous_layer_idx:?} actual={layer_idx}"
                ),
            );
            return Ok(None);
        }
        previous_layer_idx = Some(layer_idx);
        match chain_shape {
            Some(expected) if expected != (seq_len, hidden_dim) => {
                qwen_prefill_chain_trace_reject(
                    Some(layer_idx),
                    &format!(
                        "conversion_chain_shape expected={expected:?} actual=({seq_len},{hidden_dim})"
                    ),
                );
                return Ok(None);
            }
            None => {
                if seq_len
                    .checked_mul(hidden_dim)
                    .map_or(true, |expected| hidden.len() != expected)
                {
                    qwen_prefill_chain_trace_reject(
                        Some(layer_idx),
                        &format!(
                            "conversion_hidden_len actual={} expected={:?}",
                            hidden.len(),
                            seq_len.checked_mul(hidden_dim)
                        ),
                    );
                    return Ok(None);
                }
                chain_shape = Some((seq_len, hidden_dim));
            }
            _ => {}
        }
        let backend_spec = match layer {
            MetalQwenPrefillChainLayer::Attention {
                layer_idx,
                core,
                moe,
            } => rnb_backend_metal::QwenPrefillChainSpecRef::Attention {
                layer_idx: *layer_idx,
                core: match qwen_prefill_chain_attention_spec(core) {
                    Some(core) => core,
                    None => {
                        qwen_prefill_chain_trace_attention_reject(*layer_idx, core);
                        return Ok(None);
                    }
                },
                moe: match qwen_prefill_chain_moe_spec(moe, seq_len, hidden_dim) {
                    Some(moe) => moe,
                    None => {
                        qwen_prefill_chain_trace_moe_reject(*layer_idx, seq_len, hidden_dim, moe);
                        return Ok(None);
                    }
                },
            },
            MetalQwenPrefillChainLayer::Gdn {
                layer_idx,
                layer,
                moe,
            } => rnb_backend_metal::QwenPrefillChainSpecRef::Gdn {
                layer_idx: *layer_idx,
                layer: match qwen_prefill_chain_gdn_spec(layer) {
                    Some(layer) => layer,
                    None => {
                        qwen_prefill_chain_trace_gdn_reject(*layer_idx, layer);
                        return Ok(None);
                    }
                },
                moe: match qwen_prefill_chain_moe_spec(moe, seq_len, hidden_dim) {
                    Some(moe) => moe,
                    None => {
                        qwen_prefill_chain_trace_moe_reject(*layer_idx, seq_len, hidden_dim, moe);
                        return Ok(None);
                    }
                },
            },
        };
        backend_specs.push(backend_spec);
    }

    let Some(output) =
        METAL.with(|backend| backend.qwen_prefill_chain_run(hidden, &backend_specs))?
    else {
        qwen_prefill_chain_trace_reject(None, "backend_returned_none");
        return Ok(None);
    };
    Ok(Some(MetalQwenPrefillChainOut {
        hidden: output.hidden,
        attention_kv: output.attention_kv,
        gdn_states: output.gdn_states,
        hidden_uploads: output.hidden_uploads,
        hidden_readbacks: output.hidden_readbacks,
        intermediate_hidden_transfers: output.intermediate_hidden_transfers,
    }))
}

fn metal_prefill_gdn_full_ffn_enabled() -> bool {
    std::env::var("RNB_METAL_PREFILL_GDN_FULL_FFN")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

fn env_falsey(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .map(|v| {
            matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(false)
}

fn metal_prefill_atn_full_layer_requested() -> bool {
    !env_falsey("RNB_METAL_PREFILL_ATN_FULL_LAYER")
}

pub fn metal_prefill_atn_o_tail_requested() -> bool {
    std::env::var("RNB_METAL_PREFILL_ATN_O_TAIL").as_deref() == Ok("1")
}

#[derive(Clone, Copy, Default)]
struct AtnFullCounters {
    expected_dense_atn_layers: usize,
    core_hit: usize,
    full_layer_hit: usize,
    fallback: usize,
    opt_out: usize,
    skip_by_reason: usize,
    backend_err: usize,
}

thread_local! {
    static ATN_FULL_COUNTERS: std::cell::Cell<AtnFullCounters> =
        std::cell::Cell::new(AtnFullCounters::default());
}

fn atn_full_counter_update(f: impl FnOnce(&mut AtnFullCounters)) {
    ATN_FULL_COUNTERS.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    });
}

pub fn metal_prefill_atn_full_expected_dense_layer() {
    atn_full_counter_update(|c| c.expected_dense_atn_layers += 1);
}

pub fn metal_prefill_atn_full_record_core_hit() {
    atn_full_counter_update(|c| c.core_hit += 1);
}

pub fn metal_prefill_atn_full_record_full_layer_hit() {
    atn_full_counter_update(|c| c.full_layer_hit += 1);
}

pub fn metal_prefill_atn_full_record_skip() {
    atn_full_counter_update(|c| {
        c.fallback += 1;
        c.skip_by_reason += 1;
    });
}

pub fn metal_prefill_atn_full_record_adapter_reject() {
    atn_full_counter_update(|c| {
        if metal_prefill_atn_full_layer_requested() {
            c.fallback += 1;
            c.skip_by_reason += 1;
        } else {
            c.opt_out += 1;
        }
    });
}

pub fn metal_prefill_atn_full_record_backend_err() {
    atn_full_counter_update(|c| {
        c.fallback += 1;
        c.backend_err += 1;
    });
}

pub fn metal_prefill_atn_full_timing_enabled() -> bool {
    std::env::var("RNB_METAL_PREFILL_ATN_FULL_TIME").as_deref() == Ok("1")
}

pub fn metal_prefill_atn_full_counters_reset() {
    ATN_FULL_COUNTERS.with(|c| c.set(AtnFullCounters::default()));
}

pub fn metal_prefill_atn_full_counters_report(label: &str) {
    if !metal_prefill_atn_full_timing_enabled() {
        return;
    }
    ATN_FULL_COUNTERS.with(|c| {
        let v = c.get();
        eprintln!(
            "[prefill-atn-full] {label} expected_dense_atn_layers={} core_hit={} full_layer_hit={} fallback={} opt_out={} skip_by_reason={} backend_err={}",
            v.expected_dense_atn_layers,
            v.core_hit,
            v.full_layer_hit,
            v.fallback,
            v.opt_out,
            v.skip_by_reason,
            v.backend_err
        );
    });
}

#[derive(Clone, Copy, Default)]
struct AtnOTailCounters {
    expected_dense_atn_layers: usize,
    o_tail_hit: usize,
    fallback: usize,
    opt_out: usize,
    skip_by_reason: usize,
    backend_err: usize,
}

thread_local! {
    static ATN_O_TAIL_COUNTERS: std::cell::Cell<AtnOTailCounters> =
        std::cell::Cell::new(AtnOTailCounters::default());
}

fn atn_o_tail_counter_update(f: impl FnOnce(&mut AtnOTailCounters)) {
    ATN_O_TAIL_COUNTERS.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    });
}

pub fn metal_prefill_atn_o_tail_expected_dense_layer() {
    atn_o_tail_counter_update(|c| c.expected_dense_atn_layers += 1);
}

pub fn metal_prefill_atn_o_tail_record_hit() {
    atn_o_tail_counter_update(|c| c.o_tail_hit += 1);
}

pub fn metal_prefill_atn_o_tail_record_skip() {
    atn_o_tail_counter_update(|c| {
        c.fallback += 1;
        c.skip_by_reason += 1;
    });
}

pub fn metal_prefill_atn_o_tail_record_adapter_reject() {
    atn_o_tail_counter_update(|c| {
        if metal_prefill_atn_o_tail_requested() {
            c.fallback += 1;
            c.skip_by_reason += 1;
        } else {
            c.opt_out += 1;
        }
    });
}

pub fn metal_prefill_atn_o_tail_record_backend_err() {
    atn_o_tail_counter_update(|c| {
        c.fallback += 1;
        c.backend_err += 1;
    });
}

pub fn metal_prefill_atn_o_tail_timing_enabled() -> bool {
    std::env::var("RNB_METAL_PREFILL_ATN_O_TAIL_TIME").as_deref() == Ok("1")
        || metal_prefill_atn_full_timing_enabled()
}

pub fn metal_prefill_atn_o_tail_counters_reset() {
    ATN_O_TAIL_COUNTERS.with(|c| c.set(AtnOTailCounters::default()));
}

pub fn metal_prefill_atn_o_tail_counters_report(label: &str) {
    if !metal_prefill_atn_o_tail_timing_enabled() {
        return;
    }
    ATN_O_TAIL_COUNTERS.with(|c| {
        let v = c.get();
        eprintln!(
            "[prefill-atn-o-tail] {label} expected_dense_atn_layers={} o_tail_hit={} fallback={} opt_out={} skip_by_reason={} backend_err={}",
            v.expected_dense_atn_layers,
            v.o_tail_hit,
            v.fallback,
            v.opt_out,
            v.skip_by_reason,
            v.backend_err
        );
    });
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalDecodeParityPolicy {
    pub parity_requested: bool,
    pub timing_enabled: bool,
    pub legacy_decode_chain_enabled: bool,
    pub legacy_gdn_layer_enabled: bool,
    pub legacy_attn_layer_enabled: bool,
    pub qwen_moe_decode_chain_enabled: bool,
    pub kv_int8_requested: bool,
    pub collect_gpu_time: bool,
    pub delta_resident: bool,
}

impl MetalDecodeParityPolicy {
    pub fn legacy_carrier_enabled(self) -> bool {
        self.legacy_decode_chain_enabled && self.legacy_gdn_layer_enabled
    }

    pub fn kv_int8_requires_carrier_error(
        self,
        carrier_chain_enabled: bool,
        attn_layer_enabled: bool,
    ) -> Option<&'static str> {
        if self.kv_int8_requested && !(carrier_chain_enabled && attn_layer_enabled) {
            Some("RNB_METAL_KV_INT8=1 requires Metal carrier chain and attention layer")
        } else {
            None
        }
    }
}

pub fn metal_decode_parity_policy() -> MetalDecodeParityPolicy {
    MetalDecodeParityPolicy {
        parity_requested: std::env::var("RNB_METAL_DECODE_PARITY_CHAIN")
            .ok()
            .map(|v| {
                !matches!(
                    v.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false),
        timing_enabled: std::env::var("RNB_METAL_DECODE_PARITY_TIME").as_deref() == Ok("1"),
        legacy_decode_chain_enabled: std::env::var("RNB_METAL_DECODE_CHAIN").as_deref() != Ok("0"),
        legacy_gdn_layer_enabled: std::env::var("RNB_METAL_GDN_LAYER").as_deref() != Ok("0"),
        legacy_attn_layer_enabled: std::env::var("RNB_METAL_ATTN_LAYER").as_deref() != Ok("0"),
        qwen_moe_decode_chain_enabled: !env_falsey("RNB_METAL_QWEN35_MOE_DECODE_CHAIN"),
        kv_int8_requested: std::env::var("RNB_METAL_KV_INT8").as_deref() == Ok("1"),
        collect_gpu_time: std::env::var("RNB_METAL_CHAIN_GPU_TIME").as_deref() == Ok("1"),
        delta_resident: std::env::var("RNB_METAL_DELTA_RESIDENT").as_deref() != Ok("0"),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalOutputArgmaxReport {
    pub attempted: bool,
    pub did_run: bool,
    pub token_id: Option<u32>,
    pub cmd_buffers: usize,
    pub wait_us: u64,
    pub readback_bytes: usize,
    pub fallback_reason: Option<&'static str>,
}

impl MetalOutputArgmaxReport {
    pub fn fallback(reason: &'static str) -> Self {
        Self {
            attempted: true,
            fallback_reason: Some(reason),
            ..Self::default()
        }
    }
}

impl From<rnb_backend_metal::OutputArgmaxReport> for MetalOutputArgmaxReport {
    fn from(report: rnb_backend_metal::OutputArgmaxReport) -> Self {
        Self {
            attempted: report.attempted,
            did_run: report.did_run,
            token_id: report.token_id,
            cmd_buffers: report.cmd_buffers,
            wait_us: report.wait_us,
            readback_bytes: report.readback_bytes,
            fallback_reason: report.fallback_reason,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalDecodeChainReport {
    pub did_run: bool,
    pub pm51_extension_hit: bool,
    pub legacy_chain_hit: bool,
    pub qwen_moe_layers: usize,
    pub fallback_reason: Option<&'static str>,
    pub cmd_buffers: usize,
    pub wait_us: u64,
    pub upload_bytes: usize,
    pub readback_bytes: usize,
    pub state_syncs: usize,
    pub output_argmax: MetalOutputArgmaxReport,
}

impl MetalDecodeChainReport {
    pub fn fallback(reason: &'static str, backend_err: bool) -> Self {
        Self {
            did_run: false,
            fallback_reason: Some(if backend_err { "backend error" } else { reason }),
            ..Self::default()
        }
    }
}

impl From<rnb_backend_metal::DecodeChainReport> for MetalDecodeChainReport {
    fn from(report: rnb_backend_metal::DecodeChainReport) -> Self {
        Self {
            did_run: report.did_run,
            pm51_extension_hit: report.pm51_extension_hit,
            legacy_chain_hit: report.legacy_chain_hit,
            qwen_moe_layers: report.qwen_moe_layers,
            fallback_reason: report.fallback_reason,
            cmd_buffers: report.cmd_buffers,
            wait_us: report.wait_us,
            upload_bytes: report.upload_bytes,
            readback_bytes: report.readback_bytes,
            state_syncs: report.state_syncs,
            output_argmax: MetalOutputArgmaxReport {
                attempted: report.output_argmax.attempted,
                did_run: report.output_argmax.did_run,
                token_id: report.output_argmax.token_id,
                cmd_buffers: report.output_argmax.cmd_buffers,
                wait_us: report.output_argmax.wait_us,
                readback_bytes: report.output_argmax.readback_bytes,
                fallback_reason: report.output_argmax.fallback_reason,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalDecodeParityCounters {
    pub expected_decode_tokens: usize,
    pub chain_hit_tokens: usize,
    pub fallback_tokens: usize,
    pub opt_out_tokens: usize,
    pub backend_err: usize,
    pub cmd_buffers: usize,
    pub wait_us: u64,
    pub upload_bytes: usize,
    pub readback_bytes: usize,
    pub argmax_device_hits: usize,
    pub state_syncs: usize,
}

thread_local! {
    static METAL_DECODE_PARITY_COUNTERS: std::cell::Cell<MetalDecodeParityCounters> =
        std::cell::Cell::new(MetalDecodeParityCounters::default());
}

fn metal_decode_parity_counter_update(f: impl FnOnce(&mut MetalDecodeParityCounters)) {
    METAL_DECODE_PARITY_COUNTERS.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    });
}

pub fn metal_decode_parity_counters_reset() {
    METAL_DECODE_PARITY_COUNTERS.with(|c| c.set(MetalDecodeParityCounters::default()));
}

pub fn metal_decode_parity_counters_snapshot() -> MetalDecodeParityCounters {
    METAL_DECODE_PARITY_COUNTERS.with(|c| c.get())
}

pub fn metal_decode_parity_record_expected_token() {
    metal_decode_parity_counter_update(|c| c.expected_decode_tokens += 1);
}

pub fn metal_decode_parity_record_opt_out() {
    metal_decode_parity_counter_update(|c| c.opt_out_tokens += 1);
}

pub fn metal_decode_parity_record_chain_report(report: MetalDecodeChainReport) {
    metal_decode_parity_counter_update(|c| {
        if report.did_run {
            c.chain_hit_tokens += 1;
        } else {
            c.fallback_tokens += 1;
            if report.fallback_reason == Some("backend error") {
                c.backend_err += 1;
            }
        }
        c.cmd_buffers += report.cmd_buffers + report.output_argmax.cmd_buffers;
        c.wait_us += report.wait_us + report.output_argmax.wait_us;
        c.upload_bytes += report.upload_bytes;
        c.readback_bytes += report.readback_bytes + report.output_argmax.readback_bytes;
        if report.output_argmax.did_run {
            c.argmax_device_hits += 1;
        }
        c.state_syncs += report.state_syncs;
    });
}

pub fn metal_decode_parity_record_output_argmax(report: MetalOutputArgmaxReport) {
    metal_decode_parity_counter_update(|c| {
        c.cmd_buffers += report.cmd_buffers;
        c.wait_us += report.wait_us;
        c.readback_bytes += report.readback_bytes;
        if report.did_run {
            c.argmax_device_hits += 1;
        }
    });
}

pub fn metal_decode_parity_counters_report(label: &str) {
    if !metal_decode_parity_policy().timing_enabled {
        return;
    }
    let c = metal_decode_parity_counters_snapshot();
    eprintln!(
        "[metal-decode-parity] {label} expected_decode_tokens={} chain_hit_tokens={} fallback_tokens={} opt_out_tokens={} backend_err={} cmd_buffers={} wait_us={} wait_ms={:.3} upload_bytes={} readback_bytes={} argmax_device_hits={} state_syncs={}",
        c.expected_decode_tokens,
        c.chain_hit_tokens,
        c.fallback_tokens,
        c.opt_out_tokens,
        c.backend_err,
        c.cmd_buffers,
        c.wait_us,
        c.wait_us as f64 / 1000.0,
        c.upload_bytes,
        c.readback_bytes,
        c.argmax_device_hits,
        c.state_syncs
    );
}

/// sequence 경계(새 prompt/turn, in-process engine 재사용)에서 carrier 의 device-resident
/// sequence state(attention/GDN carrier + 내장 KvResident.filled)를 evict 한다. 호출 안 하면
/// 직전 sequence 의 stale device KV 가 살아 새 sequence prefill 재동기화(`if filled == 0`)를
/// skip → 오답(pm17 footgun). cuda `clear_sequence_state_cache` 와 같은 역할.
pub fn metal_clear_sequence_state() -> Result<()> {
    METAL.with(|b| b.clear_sequence_state());
    Ok(())
}

/// pm31: device 잔류 delta_state(residency 경로)를 host `out` 으로 sync 한다. speculative
/// checkpoint(`materialize_sequence_state`) 직전 stale host delta 를 최신화하는 데 쓴다.
/// carrier 없는 layer(attn/미초기화 GDN)는 `false`(host `out` 미변경). cuda
/// `sync_delta_state_cache` 와 같은 역할.
pub fn metal_sync_delta_state(layer: usize, out: &mut [f32]) -> Result<bool> {
    Ok(METAL.with(|b| b.sync_delta_state(layer, out)))
}

pub fn metal_attn_carrier_kv_filled(layer: usize) -> Option<usize> {
    METAL.with(|b| b.attn_carrier_kv_filled(layer))
}

/// Metal PoC decode GEMV. Gate: `RNB_METAL_DECODE=1`.
///
/// Q4_K 전용. 다른 quant 타입은 Ok(false) → CPU fallback.
pub fn decode_gemv_into_if_supported(
    ggml_type: GGMLType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
    _label: &str,
) -> Result<bool> {
    if std::env::var("RNB_METAL_DECODE").as_deref() != Ok("1") {
        return Ok(false);
    }
    if ggml_type != GGMLType::Q4_K {
        return Ok(false);
    }
    // zero-copy NoCopy + thread_local resident cache: raw 는 mmap 된 GGUF weight
    // 포인터(loader 소유)라 첫 token 에 1회 wrap(복사 0), 이후 캐시 hit 으로 input 만 업로드.
    // RNB_METAL_SIMD_GEMV=1 이면 SIMD-group 협력 커널(memory-bandwidth 최적화).
    let use_simd = std::env::var("RNB_METAL_SIMD_GEMV").as_deref() == Ok("1");
    let out = METAL.with(|b| {
        if use_simd {
            b.gemv_q4k_simd_resident(raw, input, rows, cols)
        } else {
            b.gemv_q4k_resident(raw, input, rows, cols)
        }
    });
    output.copy_from_slice(&out);
    Ok(true)
}

pub fn metal_output_argmax_token_if_supported(
    ggml_type: GGMLType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Option<u32> {
    if std::env::var("RNB_METAL_OUTPUT_ARGMAX").as_deref() != Ok("1") {
        return None;
    }
    if input.len() != cols || rows == 0 {
        return None;
    }
    let supported = matches!(ggml_type, GGMLType::Q4_K | GGMLType::Q6_K);
    if !supported {
        return None;
    }

    let backend_report = METAL.with(|b| match ggml_type {
        GGMLType::Q4_K => b.output_argmax_q4k_simd_resident(raw, input, rows, cols),
        GGMLType::Q6_K => b.output_argmax_q6k_simd_resident(raw, input, rows, cols),
        _ => unreachable!("unsupported Metal output argmax quant"),
    });
    let report = MetalOutputArgmaxReport::from(backend_report);
    metal_decode_parity_record_output_argmax(report);
    report.token_id
}

/// attention O chain (o_proj + residual). `RNB_METAL_O_CHAIN=1` 일 때만 활성.
/// o weight Q4_K 한정. attn_out/hidden 은 host slice. 성공 시 Ok(true) + hidden(residual) 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_o_chain_into_if_supported(
    o_ggml: GGMLType,
    attn_out: &[f32],
    o_raw: &[u8],
    hidden: &mut [f32],
    hidden_dim: usize,
    q_dim: usize,
) -> Result<bool> {
    if std::env::var("RNB_METAL_O_CHAIN").as_deref() != Ok("1") {
        return Ok(false);
    }
    if o_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let out =
        METAL.with(|b| b.attention_o_chain_resident(attn_out, hidden, o_raw, hidden_dim, q_dim));
    hidden.copy_from_slice(&out);
    Ok(true)
}

/// single-token decode attention compute (QK^T → online softmax → AV).
/// `RNB_METAL_ATTN_COMPUTE=1` 일 때만 활성. CPU `attention_decode_flash`
/// (production default) 와 token-identical. q/k_cache/v_cache 는 host slice,
/// KV 는 f16 bits(u16). sliding_window/softcap 미지원(있으면 host fallback).
/// 성공 시 Ok(true) + attn_out 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_attn_decode_into_if_supported(
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
) -> Result<bool> {
    if std::env::var("RNB_METAL_ATTN_COMPUTE").as_deref() != Ok("1") {
        return Ok(false);
    }
    // 커널 미지원 케이스(sliding window / softcap / lane-local 상한)는 host 로.
    if sliding_window.is_some()
        || has_softcap
        || head_dim > 256
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
    {
        return Ok(false);
    }
    let out = METAL.with(|b| {
        b.attn_decode(
            q,
            k_cache,
            v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
        )
    });
    attn_out[..out.len()].copy_from_slice(&out);
    Ok(true)
}

/// KV residency attention compute. `RNB_METAL_ATTN_RESIDENT=1` 일 때만 활성.
/// host KV(전체 kv_len)에서 device 에 아직 없는 token 만 incremental 복사 후 q 로
/// attention. 매 토큰 전체 KV 업로드 제거. layer 별 device KV buffer 유지.
/// sliding_window/softcap 미지원 시 host fallback. 성공 시 Ok(true) + attn_out 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_attn_decode_kv_resident_into_if_supported(
    layer: usize,
    q: &[f32],
    k_all: &[u16],
    v_all: &[u16],
    attn_out: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
    capacity: usize,
    sliding_window: Option<usize>,
    has_softcap: bool,
) -> Result<bool> {
    if std::env::var("RNB_METAL_ATTN_RESIDENT").as_deref() != Ok("1") {
        return Ok(false);
    }
    if sliding_window.is_some()
        || has_softcap
        || head_dim > 256
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
    {
        return Ok(false);
    }
    let Some(out) = METAL.with(|b| {
        b.attn_decode_kv_resident(
            layer,
            k_all,
            v_all,
            q,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            scale,
            capacity,
        )
    }) else {
        return Ok(false);
    };
    attn_out[..out.len()].copy_from_slice(&out);
    Ok(true)
}

/// text M-RoPE in-place(q/k). `RNB_METAL_ROPE=1` 일 때만 활성. CPU
/// `rope_mrope_text_inplace` 와 token-identical. q/k 는 host slice(in-place 갱신).
/// apply_k=false 면 q 만(Gemma4 q-only reuse). 성공 시 Ok(true).
#[allow(clippy::too_many_arguments)]
pub fn metal_rope_mrope_into_if_supported(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    mrope_dim: usize,
    theta: f32,
    pos: usize,
    apply_k: bool,
) -> Result<bool> {
    if std::env::var("RNB_METAL_ROPE").as_deref() != Ok("1") {
        return Ok(false);
    }
    let q_out = METAL.with(|b| b.rope_mrope(q, head_dim, q_dim, mrope_dim, theta, pos));
    q.copy_from_slice(&q_out);
    if apply_k {
        let k_out = METAL.with(|b| b.rope_mrope(k, head_dim, kv_dim, mrope_dim, theta, pos));
        k.copy_from_slice(&k_out);
    }
    Ok(true)
}

/// Attention layer 전체 device-resident carrier. pm25: default ON(opt-out) —
/// `RNB_METAL_ATTN_LAYER="0"` 으로만 끈다.
/// 표준 Qwen3 attention layer(norm→q/k/v GEMV→q/k norm→rope→kv_append→attn→o→
/// residual, 단일 command buffer). q/k/o Q4_K + v Q4_K|Q6_K 한정 — 그 외 quant
/// 또는 gemma/gated 는 caller 가 조건 미충족으로 호출 안 함. hidden(host)은 norm
/// 입력, in-place residual 갱신. prior_k/prior_v 는 prefill KV(host f16, [pos*kv_dim])
/// — 첫 호출에만 device init, 이후 device 누적. 성공 시 Ok(true) + hidden 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_attn_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    norm_weight: &[f32],
    q_ggml: GGMLType,
    k_ggml: GGMLType,
    v_ggml: GGMLType,
    o_ggml: GGMLType,
    q_raw: &[u8],
    k_raw: &[u8],
    v_raw: &[u8],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    o_raw: &[u8],
    ffn_norm_weight: &[f32],
    ffn_gate_ggml: GGMLType,
    ffn_gate_raw: &[u8],
    ffn_up_ggml: GGMLType,
    ffn_up_raw: &[u8],
    ffn_down_ggml: GGMLType,
    ffn_down_raw: &[u8],
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
) -> Result<bool> {
    if std::env::var("RNB_METAL_ATTN_LAYER").as_deref() == Ok("0") {
        return Ok(false);
    }
    // q/k/o 는 Q4_K, v 는 Q4_K 또는 Q6_K(Q4_K_M attn_v) 만. 그 외는 host fallback.
    if q_ggml != GGMLType::Q4_K || k_ggml != GGMLType::Q4_K || o_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let v_is_q6k = match v_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => return Ok(false),
    };
    // FFN: gate/up Q4_K, down Q4_K|Q6_K (9B ffn_down 은 Q6_K). 그 외는 fallback.
    if ffn_gate_ggml != GGMLType::Q4_K || ffn_up_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let ffn_down_is_q6k = match ffn_down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => return Ok(false),
    };
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 || head_dim > 256 {
        return Ok(false);
    }
    let out = METAL.with(|b| {
        b.attn_layer_resident(
            layer,
            hidden,
            norm_weight,
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight,
            k_norm_weight,
            o_raw,
            v_is_q6k,
            ffn_norm_weight,
            ffn_gate_raw,
            ffn_up_raw,
            ffn_down_raw,
            ffn_down_is_q6k,
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
    });
    hidden.copy_from_slice(&out);
    Ok(true)
}

/// 연속 GDN run 1개 layer 의 chain 입력. rnb-llm 은 `rnb-backend-metal` 에 직접
/// 의존하지 않으므로(crate 의존 방향 rnb-llm→rnb-runtime→rnb-backend-metal) backend
/// 소유 타입 `GdnChainSpecRef` 를 직접 못 만든다. 그래서 seam(이 crate)에 rnb-llm
/// 친화 spec 타입을 두고, seam 이 `GdnChainSpecRef` 로 변환한다. weight raw/f32 는
/// mmap borrow(`'a`), conv/delta state 는 borrow 충돌 회피용으로 owned(`Vec<f32>`).
#[cfg(feature = "metal")]
pub struct MetalGdnChainLayer<'a> {
    pub layer: usize,
    pub conv_state: Vec<f32>,
    pub delta_state: Vec<f32>,
    pub attn_norm_weight: &'a [f32],
    pub dt_bias_weight: &'a [f32],
    pub ssm_a_weight: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub ssm_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub qkv_ggml: GGMLType,
    pub qkv_raw: &'a [u8],
    pub gate_ggml: GGMLType,
    pub gate_raw: &'a [u8],
    pub alpha_ggml: GGMLType,
    pub alpha_raw: &'a [u8],
    pub beta_ggml: GGMLType,
    pub beta_raw: &'a [u8],
    pub ssm_out_ggml: GGMLType,
    pub ssm_out_raw: &'a [u8],
    pub ffn_gate_ggml: GGMLType,
    pub ffn_gate_raw: &'a [u8],
    pub ffn_up_ggml: GGMLType,
    pub ffn_up_raw: &'a [u8],
    pub ffn_down_ggml: GGMLType,
    pub ffn_down_raw: &'a [u8],
    pub hidden_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub z_dim: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub ffn_dim: usize,
    pub eps: f32,
}

#[cfg(feature = "metal")]
pub struct MetalGdnMoeQwenChainLayer<'a> {
    pub layer: usize,
    pub conv_state: Vec<f32>,
    pub delta_state: Vec<f32>,
    pub attn_norm_weight: &'a [f32],
    pub dt_bias_weight: &'a [f32],
    pub ssm_a_weight: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub ssm_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub qkv_ggml: GGMLType,
    pub qkv_raw: &'a [u8],
    pub gate_ggml: GGMLType,
    pub gate_raw: &'a [u8],
    pub alpha_ggml: GGMLType,
    pub alpha_raw: &'a [u8],
    pub beta_ggml: GGMLType,
    pub beta_raw: &'a [u8],
    pub ssm_out_ggml: GGMLType,
    pub ssm_out_raw: &'a [u8],
    pub router_w: &'a [f32],
    pub gate_exps_ggml: GGMLType,
    pub gate_exps_raw: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_exps_ggml: GGMLType,
    pub up_exps_raw: &'a [u8],
    pub up_expert_bytes: usize,
    pub down_exps_ggml: GGMLType,
    pub down_exps_raw: &'a [u8],
    pub down_expert_bytes: usize,
    pub shared_input_scale: &'a [f32],
    pub shared_gate_ggml: GGMLType,
    pub shared_gate_raw: &'a [u8],
    pub shared_up_ggml: GGMLType,
    pub shared_up_raw: &'a [u8],
    pub shared_down_ggml: GGMLType,
    pub shared_down_raw: &'a [u8],
    pub hidden_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub z_dim: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub eps: f32,
}

/// 연속 chain run 1개 attention layer 의 chain 입력(2단계). `MetalGdnChainLayer` 와
/// 대칭 — rnb-llm 이 backend 타입을 직접 못 만드므로 seam 친화 spec 을 두고 seam 이
/// `AttnChainSpecRef` 로 변환한다. weight raw/f32 는 mmap borrow(`'a`), prior KV 는
/// 첫 token device init 용 host f16 bits(borrow). q/k/o/ffn_gate/ffn_up = Q4_K 가드는
/// seam 변환에서 검사하고, v/ffn_down 만 Q4_K|Q6_K 라 ggml 타입을 들고 다닌다.
#[cfg(feature = "metal")]
pub struct MetalAttnChainLayer<'a> {
    pub layer: usize,
    pub norm_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub q_ggml: GGMLType,
    pub q_raw: &'a [u8],
    pub k_ggml: GGMLType,
    pub k_raw: &'a [u8],
    pub v_ggml: GGMLType,
    pub v_raw: &'a [u8],
    pub o_ggml: GGMLType,
    pub o_raw: &'a [u8],
    pub ffn_gate_ggml: GGMLType,
    pub ffn_gate_raw: &'a [u8],
    pub ffn_up_ggml: GGMLType,
    pub ffn_up_raw: &'a [u8],
    pub ffn_down_ggml: GGMLType,
    pub ffn_down_raw: &'a [u8],
    pub prior_k: &'a [u16],
    pub prior_v: &'a [u16],
    pub pos: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub n_rot: usize,
    pub capacity: usize,
    pub ffn_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub scale: f32,
}

#[cfg(feature = "metal")]
pub struct MetalAttnMoeQwenChainLayer<'a> {
    pub layer: usize,
    pub norm_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub ffn_norm_weight: &'a [f32],
    pub q_ggml: GGMLType,
    pub q_raw: &'a [u8],
    pub k_ggml: GGMLType,
    pub k_raw: &'a [u8],
    pub v_ggml: GGMLType,
    pub v_raw: &'a [u8],
    pub o_ggml: GGMLType,
    pub o_raw: &'a [u8],
    pub router_w: &'a [f32],
    pub gate_exps_ggml: GGMLType,
    pub gate_exps_raw: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_exps_ggml: GGMLType,
    pub up_exps_raw: &'a [u8],
    pub up_expert_bytes: usize,
    pub down_exps_ggml: GGMLType,
    pub down_exps_raw: &'a [u8],
    pub down_expert_bytes: usize,
    pub shared_input_scale: &'a [f32],
    pub shared_gate_ggml: GGMLType,
    pub shared_gate_raw: &'a [u8],
    pub shared_up_ggml: GGMLType,
    pub shared_up_raw: &'a [u8],
    pub shared_down_ggml: GGMLType,
    pub shared_down_raw: &'a [u8],
    pub prior_k: &'a [u16],
    pub prior_v: &'a [u16],
    pub pos: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub q_out_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub n_rot: usize,
    pub capacity: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub eps: f32,
    pub theta: f32,
    pub scale: f32,
}

/// chain run 의 per-layer spec — attn / gdn 둘 다 단일 command buffer 에 합류시키기 위한
/// enum(2단계). `metal_decode_chain_run` 이 backend `ChainLayerSpecRef` enum 으로 변환.
#[cfg(feature = "metal")]
pub enum MetalChainLayer<'a> {
    Attn(MetalAttnChainLayer<'a>),
    AttnMoeQwen(MetalAttnMoeQwenChainLayer<'a>),
    Gdn(MetalGdnChainLayer<'a>),
    GdnMoeQwen(MetalGdnMoeQwenChainLayer<'a>),
}

#[cfg(feature = "metal")]
pub struct MetalDecodeOutputArgmax<'a> {
    pub norm_weight: &'a [f32],
    pub output_ggml: GGMLType,
    pub output_raw: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
}

/// 지원 quant(Q4_K=0/Q5_K=1/Q6_K=2/Q8_0=3)을 code 로. 그 외는 None → host fallback.
/// `metal_gdn_layer_into_if_supported` 의 동일 매핑.
#[cfg(feature = "metal")]
fn gdn_quant_code(t: GGMLType) -> Option<u8> {
    match t {
        GGMLType::Q4_K => Some(0),
        GGMLType::Q5_K => Some(1),
        GGMLType::Q6_K => Some(2),
        GGMLType::Q8_0 => Some(3),
        // pm26: 27B GDN 의 ssm_alpha/beta 는 F32 raw. encode_gemv_quant case 4 →
        // gemv_f32_chain 으로 무손실 device GEMV. 이게 없으면 None → carrier 전체 포기 →
        // 48 GDN host per-op 추락(27B decode 466ms 중 carrier 0.8%, MLX 대비 8.3x).
        GGMLType::F32 => Some(4),
        _ => None,
    }
}

fn qwen_moe_decode_down_quant_code(t: GGMLType) -> Option<u8> {
    match t {
        GGMLType::Q4_K => Some(0),
        GGMLType::Q5_K => Some(1),
        GGMLType::Q6_K => Some(2),
        _ => None,
    }
}

#[cfg(feature = "metal")]
fn gdn_moe_qwen_chain_ref_if_supported<'a>(
    s: &'a MetalGdnMoeQwenChainLayer<'a>,
) -> std::result::Result<rnb_backend_metal::GdnMoeQwenChainSpecRef<'a>, &'static str> {
    let (Some(qkv_q), Some(gate_q), Some(alpha_q), Some(beta_q), Some(ssm_out_q)) = (
        gdn_quant_code(s.qkv_ggml),
        gdn_quant_code(s.gate_ggml),
        gdn_quant_code(s.alpha_ggml),
        gdn_quant_code(s.beta_ggml),
        gdn_quant_code(s.ssm_out_ggml),
    ) else {
        return Err("unsupported gdn quant");
    };
    if s.num_k_heads == 0 || s.num_v_heads % s.num_k_heads != 0 || s.head_k_dim > 256 {
        return Err("unsupported gdn shape");
    }
    if s.gate_exps_ggml != GGMLType::Q4_K
        || s.up_exps_ggml != GGMLType::Q4_K
        || s.shared_gate_ggml != GGMLType::Q4_K
        || s.shared_up_ggml != GGMLType::Q4_K
    {
        return Err("unsupported qwen moe gate/up quant");
    }
    let Some(down_quant) = qwen_moe_decode_down_quant_code(s.down_exps_ggml) else {
        return Err("unsupported qwen moe down quant");
    };
    if s.shared_down_ggml != s.down_exps_ggml {
        return Err("unsupported qwen moe shared down quant");
    }
    if s.n_expert_used == 0
        || s.n_expert_used > 31
        || s.n_expert_used > s.n_expert
        || s.hidden_dim == 0
        || s.n_ff == 0
    {
        return Err("unsupported qwen moe shape");
    }
    if s.router_w.len() != s.n_expert * s.hidden_dim {
        return Err("qwen moe router length mismatch");
    }
    if s.shared_input_scale.len() != s.hidden_dim {
        return Err("qwen moe shared input scale length mismatch");
    }
    if s.gate_exps_raw.len() != s.n_expert * s.gate_expert_bytes
        || s.up_exps_raw.len() != s.n_expert * s.up_expert_bytes
        || s.down_exps_raw.len() != s.n_expert * s.down_expert_bytes
        || s.shared_gate_raw.len() != s.gate_expert_bytes
        || s.shared_up_raw.len() != s.up_expert_bytes
        || s.shared_down_raw.len() != s.down_expert_bytes
    {
        return Err("qwen moe raw length mismatch");
    }

    Ok(rnb_backend_metal::GdnMoeQwenChainSpecRef {
        layer: s.layer,
        conv_state: &s.conv_state,
        delta_state: &s.delta_state,
        attn_norm_weight: s.attn_norm_weight,
        dt_bias_weight: s.dt_bias_weight,
        ssm_a_weight: s.ssm_a_weight,
        conv1d_weight: s.conv1d_weight,
        ssm_norm_weight: s.ssm_norm_weight,
        ffn_norm_weight: s.ffn_norm_weight,
        qkv_raw: s.qkv_raw,
        gate_raw: s.gate_raw,
        alpha_raw: s.alpha_raw,
        beta_raw: s.beta_raw,
        ssm_out_raw: s.ssm_out_raw,
        router_w: s.router_w,
        gate_exps_raw: s.gate_exps_raw,
        gate_expert_bytes: s.gate_expert_bytes,
        up_exps_raw: s.up_exps_raw,
        up_expert_bytes: s.up_expert_bytes,
        down_exps_raw: s.down_exps_raw,
        down_expert_bytes: s.down_expert_bytes,
        shared_input_scale: s.shared_input_scale,
        shared_gate_raw: s.shared_gate_raw,
        shared_up_raw: s.shared_up_raw,
        shared_down_raw: s.shared_down_raw,
        qkv_q,
        gate_q,
        alpha_q,
        beta_q,
        ssm_out_q,
        down_quant,
        hidden_dim: s.hidden_dim,
        conv_channels: s.conv_channels,
        conv_kernel: s.conv_kernel,
        z_dim: s.z_dim,
        num_v_heads: s.num_v_heads,
        num_k_heads: s.num_k_heads,
        head_k_dim: s.head_k_dim,
        head_v_dim: s.head_v_dim,
        n_ff: s.n_ff,
        n_expert: s.n_expert,
        n_expert_used: s.n_expert_used,
        eps: s.eps,
    })
}

#[cfg(feature = "metal")]
fn attn_moe_qwen_chain_ref_if_supported<'a>(
    s: &'a MetalAttnMoeQwenChainLayer<'a>,
) -> std::result::Result<rnb_backend_metal::AttnMoeQwenChainSpecRef<'a>, &'static str> {
    if s.q_ggml != GGMLType::Q4_K || s.k_ggml != GGMLType::Q4_K || s.o_ggml != GGMLType::Q4_K {
        return Err("unsupported attention quant");
    }
    let v_is_q6k = match s.v_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => return Err("unsupported attention v quant"),
    };
    if s.num_kv_heads == 0 || s.num_heads % s.num_kv_heads != 0 || s.head_dim > 256 {
        return Err("unsupported attention shape");
    }
    if s.gate_exps_ggml != GGMLType::Q4_K
        || s.up_exps_ggml != GGMLType::Q4_K
        || s.shared_gate_ggml != GGMLType::Q4_K
        || s.shared_up_ggml != GGMLType::Q4_K
    {
        return Err("unsupported qwen moe gate/up quant");
    }
    let Some(down_quant) = qwen_moe_decode_down_quant_code(s.down_exps_ggml) else {
        return Err("unsupported qwen moe down quant");
    };
    if s.shared_down_ggml != s.down_exps_ggml {
        return Err("unsupported qwen moe shared down quant");
    }
    if s.n_expert_used == 0
        || s.n_expert_used > 31
        || s.n_expert_used > s.n_expert
        || s.hidden_dim == 0
        || s.n_ff == 0
    {
        return Err("unsupported qwen moe shape");
    }
    if s.router_w.len() != s.n_expert * s.hidden_dim {
        return Err("qwen moe router length mismatch");
    }
    if s.shared_input_scale.len() != s.hidden_dim {
        return Err("qwen moe shared input scale length mismatch");
    }
    if s.gate_exps_raw.len() != s.n_expert * s.gate_expert_bytes
        || s.up_exps_raw.len() != s.n_expert * s.up_expert_bytes
        || s.down_exps_raw.len() != s.n_expert * s.down_expert_bytes
        || s.shared_gate_raw.len() != s.gate_expert_bytes
        || s.shared_up_raw.len() != s.up_expert_bytes
        || s.shared_down_raw.len() != s.down_expert_bytes
    {
        return Err("qwen moe raw length mismatch");
    }

    Ok(rnb_backend_metal::AttnMoeQwenChainSpecRef {
        layer: s.layer,
        norm_weight: s.norm_weight,
        q_norm_weight: s.q_norm_weight,
        k_norm_weight: s.k_norm_weight,
        ffn_norm_weight: s.ffn_norm_weight,
        q_raw: s.q_raw,
        k_raw: s.k_raw,
        v_raw: s.v_raw,
        o_raw: s.o_raw,
        v_is_q6k,
        prior_k: s.prior_k,
        prior_v: s.prior_v,
        pos: s.pos,
        router_w: s.router_w,
        gate_exps_raw: s.gate_exps_raw,
        gate_expert_bytes: s.gate_expert_bytes,
        up_exps_raw: s.up_exps_raw,
        up_expert_bytes: s.up_expert_bytes,
        down_exps_raw: s.down_exps_raw,
        down_expert_bytes: s.down_expert_bytes,
        shared_input_scale: s.shared_input_scale,
        shared_gate_raw: s.shared_gate_raw,
        shared_up_raw: s.shared_up_raw,
        shared_down_raw: s.shared_down_raw,
        down_quant,
        hidden_dim: s.hidden_dim,
        q_dim: s.q_dim,
        q_out_dim: s.q_out_dim,
        kv_dim: s.kv_dim,
        head_dim: s.head_dim,
        num_heads: s.num_heads,
        num_kv_heads: s.num_kv_heads,
        n_rot: s.n_rot,
        capacity: s.capacity,
        n_ff: s.n_ff,
        n_expert: s.n_expert,
        n_expert_used: s.n_expert_used,
        eps: s.eps,
        theta: s.theta,
        scale: s.scale,
    })
}

#[cfg(test)]
mod pm26_gdn_quant_tests {
    use super::gdn_quant_code;
    use rnb_loader::GGMLType;

    #[test]
    fn gdn_quant_code_supports_f32() {
        // pm26: 27B GDN 의 ssm_alpha/beta 가 F32 raw. carrier(gemv_f32_chain)가 F32 를
        // 지원하므로 Some(4) 를 반환해야 한다(이전엔 None → 48 GDN host per-op 추락 = 8.3x).
        assert_eq!(gdn_quant_code(GGMLType::F32), Some(4));
        // 기존 quant 회귀 없음.
        assert_eq!(gdn_quant_code(GGMLType::Q4_K), Some(0));
        assert_eq!(gdn_quant_code(GGMLType::Q8_0), Some(3));
    }
}

#[cfg(feature = "metal")]
fn decode_chain_options(policy: MetalDecodeParityPolicy) -> rnb_backend_metal::DecodeChainOptions {
    rnb_backend_metal::DecodeChainOptions {
        collect_timing: policy.timing_enabled || policy.collect_gpu_time,
        delta_resident: policy.delta_resident,
    }
}

/// 연속된 attn carrier + GDN carrier layer N개를 단일 command buffer 로 실행하는 seam(2단계).
/// 1단계 GDN-only 를 일반화 — specs 가 attn/gdn 혼합 enum 이라 9B 의 32 layer(attn 8 + gdn 24)
/// 를 전부 하나의 command buffer 로 묶는다. `RNB_METAL_DECODE_CHAIN=1` 게이트는 caller(decode
/// loop)가 판정 — 이 함수는 진입했으면 chain 을 실행한다. 모든 layer 의 quant 가 지원 code 로
/// 변환되고 shape 가 유효해야 한다(아니면 Ok(false) → caller 가 per-layer fallback).
///
/// out_states: GDN 은 `Some((conv_new, delta_new))`, attn 은 `None`(KV device 소유).
/// `hidden` 은 첫 layer 입력(in)이자 마지막 layer 출력(out). 성공 시 Ok(true).
///
/// `decode_chain_run` 의 CommandBuffer/encoder 는 backend 메서드 안에서 commit·
/// waitUntilCompleted 까지 완결하므로 `METAL.with(...)` closure 밖으로 새지 않는다.
#[cfg(feature = "metal")]
pub fn metal_decode_chain_run(
    hidden: &mut [f32],
    specs: &[MetalChainLayer<'_>],
    out_states: &mut [Option<(Vec<f32>, Vec<f32>)>],
    output_argmax: Option<MetalDecodeOutputArgmax<'_>>,
) -> Result<MetalDecodeChainReport> {
    if specs.is_empty() {
        return Ok(MetalDecodeChainReport {
            did_run: true,
            ..MetalDecodeChainReport::default()
        });
    }
    let policy = metal_decode_parity_policy();
    // 모든 layer 의 quant code 변환 + ChainLayerSpecRef 구성. 하나라도 미지원이면 chain 포기.
    let mut refs: Vec<rnb_backend_metal::ChainLayerSpecRef<'_>> = Vec::with_capacity(specs.len());
    for s in specs {
        match s {
            MetalChainLayer::Attn(a) => {
                // q/k/o/ffn_gate/ffn_up = Q4_K, v/ffn_down = Q4_K|Q6_K. 그 외 quant 는 미지원.
                if a.q_ggml != GGMLType::Q4_K
                    || a.k_ggml != GGMLType::Q4_K
                    || a.o_ggml != GGMLType::Q4_K
                    || a.ffn_gate_ggml != GGMLType::Q4_K
                    || a.ffn_up_ggml != GGMLType::Q4_K
                {
                    return Ok(MetalDecodeChainReport::fallback(
                        "unsupported attention quant",
                        false,
                    ));
                }
                let v_is_q6k = match a.v_ggml {
                    GGMLType::Q4_K => false,
                    GGMLType::Q6_K => true,
                    _ => {
                        return Ok(MetalDecodeChainReport::fallback(
                            "unsupported attention v quant",
                            false,
                        ))
                    }
                };
                let ffn_down_is_q6k = match a.ffn_down_ggml {
                    GGMLType::Q4_K => false,
                    GGMLType::Q6_K => true,
                    _ => {
                        return Ok(MetalDecodeChainReport::fallback(
                            "unsupported attention ffn_down quant",
                            false,
                        ))
                    }
                };
                if a.num_kv_heads == 0 || a.num_heads % a.num_kv_heads != 0 || a.head_dim > 256 {
                    return Ok(MetalDecodeChainReport::fallback(
                        "unsupported attention shape",
                        false,
                    ));
                }
                refs.push(rnb_backend_metal::ChainLayerSpecRef::Attn(
                    rnb_backend_metal::AttnChainSpecRef {
                        layer: a.layer,
                        norm_weight: a.norm_weight,
                        q_norm_weight: a.q_norm_weight,
                        k_norm_weight: a.k_norm_weight,
                        ffn_norm_weight: a.ffn_norm_weight,
                        q_raw: a.q_raw,
                        k_raw: a.k_raw,
                        v_raw: a.v_raw,
                        o_raw: a.o_raw,
                        ffn_gate_raw: a.ffn_gate_raw,
                        ffn_up_raw: a.ffn_up_raw,
                        ffn_down_raw: a.ffn_down_raw,
                        v_is_q6k,
                        ffn_down_is_q6k,
                        prior_k: a.prior_k,
                        prior_v: a.prior_v,
                        pos: a.pos,
                        hidden_dim: a.hidden_dim,
                        q_dim: a.q_dim,
                        q_out_dim: a.q_out_dim,
                        kv_dim: a.kv_dim,
                        head_dim: a.head_dim,
                        num_heads: a.num_heads,
                        num_kv_heads: a.num_kv_heads,
                        n_rot: a.n_rot,
                        capacity: a.capacity,
                        ffn_dim: a.ffn_dim,
                        eps: a.eps,
                        theta: a.theta,
                        scale: a.scale,
                    },
                ));
            }
            MetalChainLayer::AttnMoeQwen(s) => {
                let spec = match attn_moe_qwen_chain_ref_if_supported(s) {
                    Ok(spec) => spec,
                    Err(reason) => return Ok(MetalDecodeChainReport::fallback(reason, false)),
                };
                refs.push(rnb_backend_metal::ChainLayerSpecRef::AttnMoeQwen(spec));
            }
            MetalChainLayer::Gdn(s) => {
                let (
                    Some(qkv_q),
                    Some(gate_q),
                    Some(alpha_q),
                    Some(beta_q),
                    Some(ssm_out_q),
                    Some(ffn_gate_q),
                    Some(ffn_up_q),
                    Some(ffn_down_q),
                ) = (
                    gdn_quant_code(s.qkv_ggml),
                    gdn_quant_code(s.gate_ggml),
                    gdn_quant_code(s.alpha_ggml),
                    gdn_quant_code(s.beta_ggml),
                    gdn_quant_code(s.ssm_out_ggml),
                    gdn_quant_code(s.ffn_gate_ggml),
                    gdn_quant_code(s.ffn_up_ggml),
                    gdn_quant_code(s.ffn_down_ggml),
                )
                else {
                    return Ok(MetalDecodeChainReport::fallback(
                        "unsupported gdn quant",
                        false,
                    ));
                };
                if s.num_k_heads == 0 || s.num_v_heads % s.num_k_heads != 0 || s.head_k_dim > 256 {
                    return Ok(MetalDecodeChainReport::fallback(
                        "unsupported gdn shape",
                        false,
                    ));
                }
                refs.push(rnb_backend_metal::ChainLayerSpecRef::Gdn(
                    rnb_backend_metal::GdnChainSpecRef {
                        layer: s.layer,
                        conv_state: &s.conv_state,
                        delta_state: &s.delta_state,
                        attn_norm_weight: s.attn_norm_weight,
                        dt_bias_weight: s.dt_bias_weight,
                        ssm_a_weight: s.ssm_a_weight,
                        conv1d_weight: s.conv1d_weight,
                        ssm_norm_weight: s.ssm_norm_weight,
                        ffn_norm_weight: s.ffn_norm_weight,
                        qkv_raw: s.qkv_raw,
                        gate_raw: s.gate_raw,
                        alpha_raw: s.alpha_raw,
                        beta_raw: s.beta_raw,
                        ssm_out_raw: s.ssm_out_raw,
                        ffn_gate_raw: s.ffn_gate_raw,
                        ffn_up_raw: s.ffn_up_raw,
                        ffn_down_raw: s.ffn_down_raw,
                        qkv_q,
                        gate_q,
                        alpha_q,
                        beta_q,
                        ssm_out_q,
                        ffn_gate_q,
                        ffn_up_q,
                        ffn_down_q,
                        hidden_dim: s.hidden_dim,
                        conv_channels: s.conv_channels,
                        conv_kernel: s.conv_kernel,
                        z_dim: s.z_dim,
                        num_v_heads: s.num_v_heads,
                        num_k_heads: s.num_k_heads,
                        head_k_dim: s.head_k_dim,
                        head_v_dim: s.head_v_dim,
                        ffn_dim: s.ffn_dim,
                        eps: s.eps,
                    },
                ));
            }
            MetalChainLayer::GdnMoeQwen(s) => {
                let spec = match gdn_moe_qwen_chain_ref_if_supported(s) {
                    Ok(spec) => spec,
                    Err(reason) => return Ok(MetalDecodeChainReport::fallback(reason, false)),
                };
                refs.push(rnb_backend_metal::ChainLayerSpecRef::GdnMoeQwen(spec));
            }
        }
    }
    let output_argmax_ref = if let Some(output) = output_argmax {
        let output_quant = match output.output_ggml {
            GGMLType::Q4_K => 0,
            GGMLType::Q6_K => 2,
            _ => {
                return Ok(MetalDecodeChainReport::fallback(
                    "unsupported output quant",
                    false,
                ))
            }
        };
        Some(rnb_backend_metal::DecodeOutputArgmaxSpecRef {
            norm_weight: output.norm_weight,
            output_raw: output.output_raw,
            output_quant,
            rows: output.rows,
            cols: output.cols,
            eps: output.eps,
        })
    } else {
        None
    };

    // CommandBuffer 는 backend 메서드 안에서 완결 → closure 안에서 atomic.
    let options = decode_chain_options(policy);
    let mut report: MetalDecodeChainReport = METAL
        .with(|b| b.decode_chain_run(hidden, &refs, out_states, options, output_argmax_ref))
        .into();
    if report.did_run {
        if policy.parity_requested {
            report.pm51_extension_hit = true;
        } else {
            report.legacy_chain_hit = true;
        }
    }
    metal_decode_parity_record_chain_report(report);
    Ok(report)
}

/// GDN layer 전체 device-resident carrier. pm25: default ON(opt-out) —
/// `RNB_METAL_GDN_LAYER="0"` 으로만 끈다.
/// qkv/gate/alpha/beta/ssm_out/ffn_gate/ffn_up = Q4_K, ffn_down = Q4_K|Q6_K 한정.
/// conv_state·delta_state 는 host(ssm_state)에서 받아 device sync 후 갱신본 반환.
/// 성공 시 Ok(true) + hidden/conv_state/delta_state in-place 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_gdn_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_ggml: GGMLType,
    qkv_raw: &[u8],
    gate_ggml: GGMLType,
    gate_raw: &[u8],
    alpha_ggml: GGMLType,
    alpha_raw: &[u8],
    beta_ggml: GGMLType,
    beta_raw: &[u8],
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_ggml: GGMLType,
    ssm_out_raw: &[u8],
    ffn_norm_weight: &[f32],
    ffn_gate_ggml: GGMLType,
    ffn_gate_raw: &[u8],
    ffn_up_ggml: GGMLType,
    ffn_up_raw: &[u8],
    ffn_down_ggml: GGMLType,
    ffn_down_raw: &[u8],
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
) -> Result<bool> {
    if std::env::var("RNB_METAL_GDN_LAYER").as_deref() != Ok("1") {
        return Ok(false);
    }
    // pm26: carrier(metal_decode_chain_run)와 동일한 gdn_quant_code 사용. 이전엔 별도 내부
    // quant_code 가 F32 를 None 처리해, carrier 실패로 이 per-op fallback 에 들어온 27B 의
    // F32 ssm_alpha/beta 가 또 host per-op 로 빠지는 불일치(carrier 성공 시 미발현이나 통일).
    let (
        Some(qkv_q),
        Some(gate_q),
        Some(alpha_q),
        Some(beta_q),
        Some(ssm_out_q),
        Some(ffn_gate_q),
        Some(ffn_up_q),
        Some(ffn_down_q),
    ) = (
        gdn_quant_code(qkv_ggml),
        gdn_quant_code(gate_ggml),
        gdn_quant_code(alpha_ggml),
        gdn_quant_code(beta_ggml),
        gdn_quant_code(ssm_out_ggml),
        gdn_quant_code(ffn_gate_ggml),
        gdn_quant_code(ffn_up_ggml),
        gdn_quant_code(ffn_down_ggml),
    )
    else {
        return Ok(false);
    };
    if num_k_heads == 0 || num_v_heads % num_k_heads != 0 || head_k_dim > 256 {
        return Ok(false);
    }
    let (h, cs, ds) = METAL.with(|b| {
        b.gdn_layer_resident(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            qkv_raw,
            gate_raw,
            alpha_raw,
            beta_raw,
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            ssm_out_raw,
            ffn_norm_weight,
            ffn_gate_raw,
            ffn_up_raw,
            ffn_down_raw,
            qkv_q,
            gate_q,
            alpha_q,
            beta_q,
            ssm_out_q,
            ffn_gate_q,
            ffn_up_q,
            ffn_down_q,
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
    });
    hidden.copy_from_slice(&h);
    conv_state.copy_from_slice(&cs);
    delta_state.copy_from_slice(&ds);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_gdn_core_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_ggml: GGMLType,
    qkv_raw: &[u8],
    gate_ggml: GGMLType,
    gate_raw: &[u8],
    alpha_ggml: GGMLType,
    alpha_raw: &[u8],
    beta_ggml: GGMLType,
    beta_raw: &[u8],
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_ggml: GGMLType,
    ssm_out_raw: &[u8],
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    eps: f32,
) -> Result<bool> {
    if std::env::var("RNB_METAL_QWEN35_MOE_GDN_CORE").as_deref() != Ok("1") {
        return Ok(false);
    }
    let (Some(qkv_q), Some(gate_q), Some(alpha_q), Some(beta_q), Some(ssm_out_q)) = (
        gdn_quant_code(qkv_ggml),
        gdn_quant_code(gate_ggml),
        gdn_quant_code(alpha_ggml),
        gdn_quant_code(beta_ggml),
        gdn_quant_code(ssm_out_ggml),
    ) else {
        return Ok(false);
    };
    if num_k_heads == 0 || num_v_heads % num_k_heads != 0 || head_k_dim > 256 {
        return Ok(false);
    }
    let (h, cs, ds) = METAL.with(|b| {
        b.gdn_core_resident(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            qkv_raw,
            gate_raw,
            alpha_raw,
            beta_raw,
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            ssm_out_raw,
            qkv_q,
            gate_q,
            alpha_q,
            beta_q,
            ssm_out_q,
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
    });
    hidden.copy_from_slice(&h);
    conv_state.copy_from_slice(&cs);
    delta_state.copy_from_slice(&ds);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_gdn_moe_layer_into_if_supported(
    layer: usize,
    hidden: &mut [f32],
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    attn_norm_weight: &[f32],
    qkv_ggml: GGMLType,
    qkv_raw: &[u8],
    gate_ggml: GGMLType,
    gate_raw: &[u8],
    alpha_ggml: GGMLType,
    alpha_raw: &[u8],
    beta_ggml: GGMLType,
    beta_raw: &[u8],
    dt_bias_weight: &[f32],
    ssm_a_weight: &[f32],
    conv1d_weight: &[f32],
    ssm_norm_weight: &[f32],
    ssm_out_ggml: GGMLType,
    ssm_out_raw: &[u8],
    ffn_norm_weight: &[f32],
    router_w: &[f32],
    gate_exps_ggml: GGMLType,
    gate_exps_raw: &[u8],
    gate_expert_bytes: usize,
    up_exps_ggml: GGMLType,
    up_exps_raw: &[u8],
    up_expert_bytes: usize,
    down_exps_ggml: GGMLType,
    down_exps_raw: &[u8],
    down_expert_bytes: usize,
    shared_input_scale: &[f32],
    shared_gate_ggml: GGMLType,
    shared_gate_raw: &[u8],
    shared_up_ggml: GGMLType,
    shared_up_raw: &[u8],
    shared_down_ggml: GGMLType,
    shared_down_raw: &[u8],
    hidden_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    z_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    n_ff: usize,
    n_expert: usize,
    n_expert_used: usize,
    eps: f32,
) -> Result<bool> {
    if std::env::var("RNB_METAL_QWEN35_MOE_GDN_FULL_LAYER").as_deref() != Ok("1") {
        return Ok(false);
    }
    let (Some(qkv_q), Some(gate_q), Some(alpha_q), Some(beta_q), Some(ssm_out_q)) = (
        gdn_quant_code(qkv_ggml),
        gdn_quant_code(gate_ggml),
        gdn_quant_code(alpha_ggml),
        gdn_quant_code(beta_ggml),
        gdn_quant_code(ssm_out_ggml),
    ) else {
        return Ok(false);
    };
    if gate_exps_ggml != GGMLType::Q4_K
        || up_exps_ggml != GGMLType::Q4_K
        || shared_gate_ggml != GGMLType::Q4_K
        || shared_up_ggml != GGMLType::Q4_K
        || shared_down_ggml != down_exps_ggml
    {
        return Ok(false);
    }
    let down_quant = match down_exps_ggml {
        GGMLType::Q5_K => 1,
        GGMLType::Q6_K => 2,
        _ => return Ok(false),
    };
    if num_k_heads == 0
        || num_v_heads % num_k_heads != 0
        || head_k_dim > 256
        || n_expert_used == 0
        || n_expert_used > 31
        || hidden.len() != hidden_dim
        || router_w.len() != n_expert * hidden_dim
        || shared_input_scale.len() != hidden_dim
        || gate_exps_raw.is_empty()
        || up_exps_raw.is_empty()
        || down_exps_raw.is_empty()
        || shared_gate_raw.is_empty()
        || shared_up_raw.is_empty()
        || shared_down_raw.is_empty()
        || gate_expert_bytes == 0
        || up_expert_bytes == 0
        || down_expert_bytes == 0
    {
        return Ok(false);
    }
    let (h, cs, ds) = METAL.with(|b| {
        b.gdn_moe_layer_resident(
            layer,
            hidden,
            conv_state,
            delta_state,
            attn_norm_weight,
            qkv_raw,
            gate_raw,
            alpha_raw,
            beta_raw,
            dt_bias_weight,
            ssm_a_weight,
            conv1d_weight,
            ssm_norm_weight,
            ssm_out_raw,
            ffn_norm_weight,
            router_w,
            gate_exps_raw,
            gate_expert_bytes,
            up_exps_raw,
            up_expert_bytes,
            down_exps_raw,
            down_expert_bytes,
            shared_input_scale,
            shared_gate_raw,
            shared_up_raw,
            shared_down_raw,
            qkv_q,
            gate_q,
            alpha_q,
            beta_q,
            ssm_out_q,
            down_quant,
            hidden_dim,
            conv_channels,
            conv_kernel,
            z_dim,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            n_ff,
            n_expert,
            n_expert_used,
            eps,
        )
    });
    hidden.copy_from_slice(&h);
    conv_state.copy_from_slice(&cs);
    delta_state.copy_from_slice(&ds);
    Ok(true)
}

/// GDN qkv+gate device-resident chain. `RNB_METAL_GDN_CHAIN=1` 일 때만 활성.
/// qkv/gate 모두 Q4_K 일 때 단일 command buffer 2 GEMV(dispatch 2→1).
/// norm_input 은 norm 완료된 host slice. 성공 시 Ok(true) + qkv/gate out 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_gdn_inproj_chain_into_if_supported(
    qkv_ggml: GGMLType,
    gate_ggml: GGMLType,
    norm_input: &[f32],
    qkv_raw: &[u8],
    gate_raw: &[u8],
    qkv_out: &mut [f32],
    gate_out: &mut [f32],
    hidden_dim: usize,
    qkv_dim: usize,
    gate_dim: usize,
) -> Result<bool> {
    if std::env::var("RNB_METAL_GDN_CHAIN").as_deref() != Ok("1") {
        return Ok(false);
    }
    if qkv_ggml != GGMLType::Q4_K || gate_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let (qkv, gate) = METAL.with(|b| {
        b.gdn_inproj_chain_resident(norm_input, qkv_raw, gate_raw, hidden_dim, qkv_dim, gate_dim)
    });
    qkv_out.copy_from_slice(&qkv);
    gate_out.copy_from_slice(&gate);
    Ok(true)
}

/// QKV projection device-resident chain. `RNB_METAL_QKV_CHAIN=1` 일 때만 활성.
/// q/k/v 모두 Q4_K 일 때 단일 command buffer 3 GEMV(dispatch 3→1). norm_input 은
/// chain 진입 전 norm 완료된 host slice. 성공 시 Ok(true) + q/k/v out 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_qkv_chain_into_if_supported(
    q_ggml: GGMLType,
    k_ggml: GGMLType,
    v_ggml: GGMLType,
    norm_input: &[f32],
    q_raw: &[u8],
    k_raw: &[u8],
    v_raw: &[u8],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    hidden_dim: usize,
    q_out_dim: usize,
    kv_dim: usize,
) -> Result<bool> {
    if std::env::var("RNB_METAL_QKV_CHAIN").as_deref() != Ok("1") {
        return Ok(false);
    }
    if q_ggml != GGMLType::Q4_K || k_ggml != GGMLType::Q4_K || v_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let (q, k, v) = METAL.with(|b| {
        b.attention_qkv_chain_resident(
            norm_input, q_raw, k_raw, v_raw, hidden_dim, q_out_dim, kv_dim,
        )
    });
    q_out.copy_from_slice(&q);
    k_out.copy_from_slice(&k);
    v_out.copy_from_slice(&v);
    Ok(true)
}

/// FFN device-resident chain. `RNB_METAL_FFN_CHAIN=1` 일 때만 활성.
/// gate/up 은 Q4_K, down 은 Q4_K 또는 Q6_K raw mmap bytes(loader 소유).
/// 본선 dense 모델(qwen3.5-9B 등)은 down 이 Q6_K 라 mixed quant 를 허용한다.
/// 성공 시 Ok(true) + hidden 갱신.
#[allow(clippy::too_many_arguments)]
pub fn metal_ffn_chain_into_if_supported(
    gate_ggml: GGMLType,
    up_ggml: GGMLType,
    down_ggml: GGMLType,
    gate_raw: &[u8],
    up_raw: &[u8],
    down_raw: &[u8],
    norm_weight: &[f32],
    hidden: &mut [f32],
    hidden_dim: usize,
    ffn_dim: usize,
    norm_eps: f32,
) -> Result<bool> {
    if std::env::var("RNB_METAL_FFN_CHAIN").as_deref() != Ok("1") {
        return Ok(false);
    }
    // gate/up 은 Q4_K(silu chain GEMV 커널 한정). down 은 Q4_K 또는 Q6_K 허용.
    // 그 외 quant(Q5_0/Q8_0 등 MoE 계열)는 Ok(false) → CPU fallback.
    if gate_ggml != GGMLType::Q4_K || up_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let down_is_q6k = match down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => return Ok(false),
    };
    let out = METAL.with(|b| {
        b.ffn_chain_q4k_resident(
            hidden,
            norm_weight,
            gate_raw,
            up_raw,
            down_raw,
            hidden_dim,
            ffn_dim,
            norm_eps,
            down_is_q6k,
        )
    });
    hidden.copy_from_slice(&out);
    Ok(true)
}

/// pm33: prefill FFN batch GEMM chain. `RNB_METAL_PREFILL_FFN=1` 일 때만 활성.
/// `metal_ffn_chain_into_if_supported`(decode, single-token)의 M>1 아날로그.
/// norm 은 caller 가 적용(`normed` 입력), residual 도 caller 처리(`out` = down 결과, residual 전).
/// gate/up 은 Q4_K, down 은 Q4_K 또는 Q6_K raw mmap bytes. 성공 시 Ok(true) + out 채움.
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_ffn_chain_into_if_supported(
    gate_ggml: GGMLType,
    up_ggml: GGMLType,
    down_ggml: GGMLType,
    gate_raw: &[u8],
    up_raw: &[u8],
    down_raw: &[u8],
    normed: &[f32],
    out: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<bool> {
    // pm37: default ON(metal prefill GPU 승격, 27B token-identical 검증). opt-out=RNB_METAL_PREFILL_FFN=0.
    if std::env::var("RNB_METAL_PREFILL_FFN").as_deref() == Ok("0") {
        return Ok(false);
    }
    if gate_ggml != GGMLType::Q4_K || up_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let down_is_q6k = match down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => return Ok(false),
    };
    let r = METAL.with(|b| {
        b.prefill_ffn_chain(
            normed,
            gate_raw,
            up_raw,
            down_raw,
            down_is_q6k,
            seq_len,
            hidden_dim,
            ffn_dim,
        )
    });
    out.copy_from_slice(&r);
    Ok(true)
}

fn metal_qwen_moe_expert_batch_enabled(group_len: usize) -> bool {
    match std::env::var("RNB_METAL_QWEN35_MOE_EXPERT_BATCH") {
        Ok(value) => !matches!(
            value.to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => group_len >= 16,
    }
}

fn metal_qwen_moe_prefill_router_requested() -> bool {
    std::env::var("RNB_METAL_QWEN35_PREFILL_ROUTER")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

fn metal_qwen_moe_prefill_route_topk_requested() -> bool {
    std::env::var("RNB_METAL_QWEN35_PREFILL_ROUTE_TOPK")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_prefill_router_logits_if_supported(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    seq_len: usize,
) -> Result<Option<Vec<f32>>> {
    if !metal_qwen_moe_prefill_router_requested() {
        return Ok(None);
    }
    if router_w.len() != n_expert * hidden_dim {
        return Err(format!(
            "metal qwen moe router weight len mismatch: got {}, expected {}",
            router_w.len(),
            n_expert * hidden_dim
        ));
    }
    if norm_all.len() != seq_len * hidden_dim {
        return Err(format!(
            "metal qwen moe router norm len mismatch: got {}, expected {}",
            norm_all.len(),
            seq_len * hidden_dim
        ));
    }
    let out = METAL.with(|b| {
        b.qwen_moe_prefill_router_logits(router_w, norm_all, n_expert, hidden_dim, seq_len)
    });
    if out.len() != seq_len * n_expert {
        return Err(format!(
            "metal qwen moe router logits len mismatch: got {}, expected {}",
            out.len(),
            seq_len * n_expert
        ));
    }
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_prefill_route_arrays_if_supported(
    router_w: &[f32],
    n_expert: usize,
    hidden_dim: usize,
    norm_all: &[f32],
    seq_len: usize,
    n_expert_used: usize,
) -> Result<Option<(Vec<u32>, Vec<f32>, Vec<u32>)>> {
    if !metal_qwen_moe_prefill_router_requested()
        || !metal_qwen_moe_prefill_route_topk_requested()
        || n_expert_used == 0
        || n_expert_used > 32
    {
        return Ok(None);
    }
    if router_w.len() != n_expert * hidden_dim {
        return Err(format!(
            "metal qwen moe route weight len mismatch: got {}, expected {}",
            router_w.len(),
            n_expert * hidden_dim
        ));
    }
    if norm_all.len() != seq_len * hidden_dim {
        return Err(format!(
            "metal qwen moe route norm len mismatch: got {}, expected {}",
            norm_all.len(),
            seq_len * hidden_dim
        ));
    }
    let (expert_ids, route_weights, token_ids) = METAL.with(|b| {
        b.qwen_moe_prefill_route_arrays(
            router_w,
            norm_all,
            n_expert,
            hidden_dim,
            seq_len,
            n_expert_used,
        )
    });
    let slots = seq_len * n_expert_used.min(n_expert);
    if expert_ids.len() != slots || route_weights.len() != slots || token_ids.len() != slots {
        return Err(format!(
            "metal qwen moe route arrays len mismatch: experts={} weights={} tokens={} expected={slots}",
            expert_ids.len(),
            route_weights.len(),
            token_ids.len()
        ));
    }
    Ok(Some((expert_ids, route_weights, token_ids)))
}

/// Qwen3.6 MoE expert-group prefill path. This reuses the dense FFN Metal chain
/// for one already-routed expert group: gate/up must be Q4_K, down Q4_K or Q6_K,
/// input is the token-major normalized activations for that expert group.
#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_expert_ffn_into_if_supported(
    gate_ggml: GGMLType,
    up_ggml: GGMLType,
    down_ggml: GGMLType,
    gate_raw: &[u8],
    up_raw: &[u8],
    down_raw: &[u8],
    normed: &[f32],
    out: &mut [f32],
    group_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<bool> {
    if !metal_qwen_moe_expert_batch_enabled(group_len) {
        return Ok(false);
    }
    if gate_ggml != GGMLType::Q4_K || up_ggml != GGMLType::Q4_K {
        return Ok(false);
    }
    let down_is_q6k = match down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => return Ok(false),
    };
    if group_len == 0
        || normed.len() != group_len * hidden_dim
        || out.len() != group_len * hidden_dim
    {
        return Ok(false);
    }
    let r = METAL.with(|b| {
        b.prefill_ffn_chain(
            normed,
            gate_raw,
            up_raw,
            down_raw,
            down_is_q6k,
            group_len,
            hidden_dim,
            ffn_dim,
        )
    });
    out.copy_from_slice(&r);
    Ok(true)
}

fn metal_qwen_moe_prefill_accum_enabled(seq_len: usize) -> bool {
    match std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ACCUM") {
        Ok(value) => !matches!(
            value.to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => seq_len >= 1024,
    }
}

fn metal_qwen_moe_prefill_accum_trace_enabled() -> bool {
    std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_ACCUM_TRACE")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(false)
}

pub fn metal_qwen_moe_prefill_mulmmid_v3_requested() -> bool {
    std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V3")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
        .unwrap_or(false)
        && !metal_qwen_moe_llama_id_requested()
}

pub fn metal_qwen_moe_llama_id_requested() -> bool {
    !env_falsey("RNB_METAL_QWEN35_MOE_LLAMA_ID")
}

pub fn metal_qwen_moe_prefill_mulmmid_v4_requested() -> bool {
    std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
        .unwrap_or(false)
        && !metal_qwen_moe_llama_id_requested()
}

pub struct MetalQwenMoeLlamaIdPrefillRequest<'a> {
    pub gate_all: &'a [u8],
    pub up_all: &'a [u8],
    pub down_all: &'a [u8],
    pub gate_expert_bytes: usize,
    pub up_expert_bytes: usize,
    pub down_expert_bytes: usize,
    pub selected_experts: &'a [u32],
    pub route_weights: &'a [f32],
    pub shared_gate: &'a [u8],
    pub shared_up: &'a [u8],
    pub shared_down: &'a [u8],
    pub shared_route_weights: &'a [f32],
    pub gate_quant: GGMLType,
    pub up_quant: GGMLType,
    pub down_quant: GGMLType,
    pub shared_gate_quant: GGMLType,
    pub shared_up_quant: GGMLType,
    pub shared_down_quant: GGMLType,
    pub norm_all: &'a [f32],
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub layer_idx: usize,
}

fn qwen_moe_llama_id_quant_sets(
    request: &MetalQwenMoeLlamaIdPrefillRequest<'_>,
) -> Option<(QwenMoeLlamaIdQuantSet, QwenMoeLlamaIdQuantSet)> {
    let sparse_down = match request.down_quant {
        GGMLType::Q4_K => QwenMoeLlamaIdQuant::Q4K,
        GGMLType::Q5_K => QwenMoeLlamaIdQuant::Q5K,
        GGMLType::Q6_K => QwenMoeLlamaIdQuant::Q6K,
        _ => return None,
    };
    if request.gate_quant != GGMLType::Q4_K || request.up_quant != GGMLType::Q4_K {
        return None;
    }
    let shared_quant = match (
        request.shared_gate_quant,
        request.shared_up_quant,
        request.shared_down_quant,
    ) {
        (GGMLType::Q8_0, GGMLType::Q8_0, GGMLType::Q8_0) => QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q8Zero,
            up: QwenMoeLlamaIdQuant::Q8Zero,
            down: QwenMoeLlamaIdQuant::Q8Zero,
        },
        (GGMLType::Q4_K, GGMLType::Q4_K, GGMLType::Q4_K) => QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q4K,
        },
        (GGMLType::Q4_K, GGMLType::Q4_K, GGMLType::Q6_K) => QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: QwenMoeLlamaIdQuant::Q6K,
        },
        _ => return None,
    };
    Some((
        QwenMoeLlamaIdQuantSet {
            gate: QwenMoeLlamaIdQuant::Q4K,
            up: QwenMoeLlamaIdQuant::Q4K,
            down: sparse_down,
        },
        shared_quant,
    ))
}

pub fn metal_qwen_moe_llama_id_prefill(
    request: MetalQwenMoeLlamaIdPrefillRequest<'_>,
) -> Result<Option<QwenMoeLlamaIdPrefillOutput>> {
    let Some((sparse_quant, shared_quant)) = qwen_moe_llama_id_quant_sets(&request) else {
        return Ok(None);
    };
    METAL.with(|backend| {
        backend.qwen_moe_llama_id_prefill(QwenMoeLlamaIdPrefillRequest {
            gate_all: request.gate_all,
            up_all: request.up_all,
            down_all: request.down_all,
            gate_expert_bytes: request.gate_expert_bytes,
            up_expert_bytes: request.up_expert_bytes,
            down_expert_bytes: request.down_expert_bytes,
            selected_experts: request.selected_experts,
            route_weights: request.route_weights,
            shared_gate: request.shared_gate,
            shared_up: request.shared_up,
            shared_down: request.shared_down,
            shared_route_weights: request.shared_route_weights,
            sparse_quant,
            shared_quant,
            norm_all: request.norm_all,
            seq_len: request.seq_len,
            hidden_dim: request.hidden_dim,
            ffn_dim: request.ffn_dim,
            layer_idx: request.layer_idx,
        })
    })
}

fn metal_qwen_moe_prefill_shared_accum_enabled(seq_len: usize) -> bool {
    if !metal_qwen_moe_prefill_accum_enabled(seq_len) {
        return false;
    }
    std::env::var("RNB_METAL_QWEN35_MOE_PREFILL_SHARED_ACCUM")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

/// Qwen3.6 MoE sparse prefill opt-in: keep quantized expert weights raw and
/// accumulate expert-group outputs on Metal, then read back the final sparse
/// token-major output once per layer.
#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_prefill_sparse_accum_if_supported(
    gate_ggml: GGMLType,
    up_ggml: GGMLType,
    down_ggml: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_all: &[f32],
) -> Result<Option<Vec<f32>>> {
    let trace = metal_qwen_moe_prefill_accum_trace_enabled();
    if !metal_qwen_moe_prefill_accum_enabled(seq_len) {
        if trace {
            eprintln!("[metal-qwen-moe-prefill-accum] skip reason=policy seq_len={seq_len}");
        }
        return Ok(None);
    }
    if gate_ggml != GGMLType::Q4_K || up_ggml != GGMLType::Q4_K {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] skip reason=gate_up_quant gate={gate_ggml:?} up={up_ggml:?}"
            );
        }
        return Ok(None);
    }
    let down_is_q6k = match down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => {
            if trace {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] skip reason=down_quant down={down_ggml:?}"
                );
            }
            return Ok(None);
        }
    };
    if expert_ids.is_empty()
        || expert_ids.len() != route_weights.len()
        || expert_ids.len() != token_ids.len()
        || norm_all.len() != seq_len * hidden_dim
    {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] skip reason=shape experts={} weights={} tokens={} norm={} expected_norm={}",
                expert_ids.len(),
                route_weights.len(),
                token_ids.len(),
                norm_all.len(),
                seq_len * hidden_dim
            );
        }
        return Ok(None);
    }
    let supported = METAL.with(|b| b.qwen_moe_prefill_sparse_accum_supported(down_is_q6k));
    if !supported {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] skip reason=backend_unsupported down_is_q6k={down_is_q6k}"
            );
        }
        return Ok(None);
    }
    if trace {
        eprintln!(
            "[metal-qwen-moe-prefill-accum] use slots={} seq_len={seq_len} hidden={hidden_dim} ffn={ffn_dim} down_is_q6k={down_is_q6k}",
            expert_ids.len()
        );
    }
    let out = METAL.with(|b| {
        b.qwen_moe_prefill_sparse_accum(
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            down_is_q6k,
            seq_len,
            hidden_dim,
            ffn_dim,
            norm_all,
        )
    });
    Ok(Some(out))
}

/// Qwen3.6 MoE prefill shared+sparse accumulator. The shared expert is encoded
/// into the same token-major accumulator before sparse experts, so the caller
/// reads one combined output and avoids a separate shared CPU batch + host add.
#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_prefill_shared_sparse_accum_if_supported(
    shared_gate_ggml: GGMLType,
    shared_up_ggml: GGMLType,
    shared_down_ggml: GGMLType,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route_weights: &[f32],
    shared_token_ids: &[u32],
    gate_ggml: GGMLType,
    up_ggml: GGMLType,
    down_ggml: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_all: &[f32],
) -> Result<Option<Vec<f32>>> {
    let trace = metal_qwen_moe_prefill_accum_trace_enabled();
    if !metal_qwen_moe_prefill_shared_accum_enabled(seq_len) {
        if trace {
            eprintln!("[metal-qwen-moe-prefill-accum] skip reason=shared_policy seq_len={seq_len}");
        }
        return Ok(None);
    }
    if shared_gate_ggml != GGMLType::Q4_K
        || shared_up_ggml != GGMLType::Q4_K
        || gate_ggml != GGMLType::Q4_K
        || up_ggml != GGMLType::Q4_K
        || shared_down_ggml != down_ggml
    {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] skip reason=shared_quant shared_gate={shared_gate_ggml:?} shared_up={shared_up_ggml:?} shared_down={shared_down_ggml:?} gate={gate_ggml:?} up={up_ggml:?} down={down_ggml:?}"
            );
        }
        return Ok(None);
    }
    let down_is_q6k = match down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => {
            if trace {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] skip reason=shared_down_quant down={down_ggml:?}"
                );
            }
            return Ok(None);
        }
    };
    if expert_ids.is_empty()
        || expert_ids.len() != route_weights.len()
        || expert_ids.len() != token_ids.len()
        || shared_route_weights.len() != seq_len
        || shared_token_ids.len() != seq_len
        || norm_all.len() != seq_len * hidden_dim
    {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] skip reason=shared_shape experts={} weights={} tokens={} shared_weights={} shared_tokens={} norm={} expected_norm={}",
                expert_ids.len(),
                route_weights.len(),
                token_ids.len(),
                shared_route_weights.len(),
                shared_token_ids.len(),
                norm_all.len(),
                seq_len * hidden_dim
            );
        }
        return Ok(None);
    }
    let supported = METAL.with(|b| b.qwen_moe_prefill_sparse_accum_supported(down_is_q6k));
    if !supported {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] skip reason=shared_backend_unsupported down_is_q6k={down_is_q6k}"
            );
        }
        return Ok(None);
    }
    if trace {
        eprintln!(
            "[metal-qwen-moe-prefill-accum] use shared+sparse slots={} seq_len={seq_len} hidden={hidden_dim} ffn={ffn_dim} down_is_q6k={down_is_q6k}",
            expert_ids.len()
        );
    }
    let out = METAL.with(|b| {
        b.qwen_moe_prefill_shared_sparse_accum(
            shared_gate,
            shared_up,
            shared_down,
            shared_route_weights,
            shared_token_ids,
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            down_is_q6k,
            seq_len,
            hidden_dim,
            ffn_dim,
            norm_all,
        )
    });
    Ok(Some(out))
}

/// Qwen3.6 MoE prefill mul-mm-id v3 opt-in facade. This keeps the v3 entry
/// token-major and Q4_K/Q6_K-only; unsupported requests fall back to the existing path.
#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_prefill_shared_sparse_selected_base_v4_if_supported(
    shared_gate_ggml: GGMLType,
    shared_up_ggml: GGMLType,
    shared_down_ggml: GGMLType,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route_weights: &[f32],
    shared_token_ids: &[u32],
    gate_ggml: GGMLType,
    up_ggml: GGMLType,
    down_ggml: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_all: &[f32],
) -> Result<Option<Vec<f32>>> {
    if !metal_qwen_moe_prefill_mulmmid_v4_requested() {
        return Ok(None);
    }
    let trace = metal_qwen_moe_prefill_accum_trace_enabled();
    let down_is_q6k = match down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => {
            if trace {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] mode=mulmmid_v4 fallback_reason=unsupported_quant"
                );
            }
            return Ok(None);
        }
    };
    if shared_gate_ggml != GGMLType::Q4_K
        || shared_up_ggml != GGMLType::Q4_K
        || shared_down_ggml != down_ggml
        || gate_ggml != GGMLType::Q4_K
        || up_ggml != GGMLType::Q4_K
    {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=mulmmid_v4 fallback_reason=unsupported_quant"
            );
        }
        return Ok(None);
    }
    if expert_ids.is_empty()
        || expert_ids.len() != route_weights.len()
        || expert_ids.len() != token_ids.len()
        || shared_route_weights.len() != seq_len
        || shared_token_ids.len() != seq_len
        || norm_all.len() != seq_len * hidden_dim
    {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=mulmmid_v4 fallback_reason=invalid_shape"
            );
        }
        return Ok(None);
    }
    let supported = METAL.with(|b| b.qwen_moe_prefill_mulmmid_v4_supported(down_is_q6k));
    if !supported {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=mulmmid_v4 fallback_reason=pipeline_unavailable"
            );
        }
        return Ok(None);
    }
    let out = METAL.with(|b| {
        b.qwen_moe_prefill_shared_sparse_accum(
            shared_gate,
            shared_up,
            shared_down,
            shared_route_weights,
            shared_token_ids,
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            down_is_q6k,
            seq_len,
            hidden_dim,
            ffn_dim,
            norm_all,
        )
    });
    Ok(Some(out))
}

/// Qwen3.6 MoE prefill mul-mm-id v3 opt-in facade. This keeps the v3 entry
/// token-major and Q4_K/Q6_K-only; unsupported requests fall back to the existing path.
#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_prefill_shared_sparse_selected_base_v3_if_supported(
    shared_gate_ggml: GGMLType,
    shared_up_ggml: GGMLType,
    shared_down_ggml: GGMLType,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route_weights: &[f32],
    shared_token_ids: &[u32],
    gate_ggml: GGMLType,
    up_ggml: GGMLType,
    down_ggml: GGMLType,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_all: &[f32],
) -> Result<Option<Vec<f32>>> {
    if !metal_qwen_moe_prefill_mulmmid_v3_requested() {
        return Ok(None);
    }
    let trace = metal_qwen_moe_prefill_accum_trace_enabled();
    let down_is_q6k = match down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => {
            if trace {
                eprintln!(
                    "[metal-qwen-moe-prefill-accum] mode=mulmmid_v3 fallback_reason=unsupported_quant"
                );
            }
            return Ok(None);
        }
    };
    if shared_gate_ggml != GGMLType::Q4_K
        || shared_up_ggml != GGMLType::Q4_K
        || shared_down_ggml != down_ggml
        || gate_ggml != GGMLType::Q4_K
        || up_ggml != GGMLType::Q4_K
    {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=mulmmid_v3 fallback_reason=unsupported_quant"
            );
        }
        return Ok(None);
    }
    if expert_ids.is_empty()
        || expert_ids.len() != route_weights.len()
        || expert_ids.len() != token_ids.len()
        || shared_route_weights.len() != seq_len
        || shared_token_ids.len() != seq_len
        || norm_all.len() != seq_len * hidden_dim
    {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=mulmmid_v3 fallback_reason=invalid_shape"
            );
        }
        return Ok(None);
    }
    let supported = METAL.with(|b| b.qwen_moe_prefill_mulmmid_v3_supported(down_is_q6k));
    if !supported {
        if trace {
            eprintln!(
                "[metal-qwen-moe-prefill-accum] mode=mulmmid_v3 fallback_reason=pipeline_unavailable"
            );
        }
        return Ok(None);
    }
    let out = METAL.with(|b| {
        b.qwen_moe_prefill_shared_sparse_accum(
            shared_gate,
            shared_up,
            shared_down,
            shared_route_weights,
            shared_token_ids,
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            expert_ids,
            route_weights,
            token_ids,
            down_is_q6k,
            seq_len,
            hidden_dim,
            ffn_dim,
            norm_all,
        )
    });
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_decode_sparse_experts_id_into_if_supported(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    gate_expert_bytes: usize,
    up_expert_bytes: usize,
    down_expert_bytes: usize,
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    shared_expert_id: u32,
    down_ggml: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    if std::env::var("RNB_METAL_QWEN35_MOE_DECODE_BATCH").as_deref() != Ok("1") {
        return Ok(false);
    }
    if env_falsey("RNB_METAL_QWEN35_MOE_DECODE_ID") {
        return Ok(false);
    }
    if gate_all.is_empty()
        || up_all.is_empty()
        || down_all.is_empty()
        || shared_gate.is_empty()
        || shared_up.is_empty()
        || shared_down.is_empty()
        || expert_ids.is_empty()
        || expert_ids.len() != route_weights.len()
        || input.len() != n_embd
        || out.len() != n_embd
    {
        return Ok(false);
    }
    let down_quant = match down_ggml {
        GGMLType::Q5_K => 1,
        GGMLType::Q6_K => 2,
        _ => return Ok(false),
    };
    let result = METAL.with(|b| {
        b.qwen_moe_decode_sparse_experts_id(
            gate_all,
            up_all,
            down_all,
            gate_expert_bytes,
            up_expert_bytes,
            down_expert_bytes,
            shared_gate,
            shared_up,
            shared_down,
            expert_ids,
            route_weights,
            shared_expert_id,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
    });
    out.copy_from_slice(&result);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_qwen_moe_decode_sparse_experts_into_if_supported(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    down_ggml: GGMLType,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    if std::env::var("RNB_METAL_QWEN35_MOE_DECODE_BATCH").as_deref() != Ok("1") {
        return Ok(false);
    }
    if gate.is_empty()
        || gate.len() != up.len()
        || gate.len() != down.len()
        || gate.len() != route_weights.len()
        || input.len() != n_embd
        || out.len() != n_embd
    {
        return Ok(false);
    }
    let Some(down_quant) = qwen_moe_decode_down_quant_code(down_ggml) else {
        return Ok(false);
    };
    let result = METAL.with(|b| {
        b.qwen_moe_decode_sparse_experts(
            gate,
            up,
            down,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
        )
    });
    out.copy_from_slice(&result);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_glm_moe_decode_iq2xxs_iq3xxs_into_if_supported(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    shared_route_weight: f32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
    out: &mut [f32],
    gate_up_iq2s: bool,
    down_is_iq4xs: bool,
    shared_gate_up_q6k: bool,
    shared_down_q8_0: bool,
) -> Result<bool> {
    if env_falsey("RNB_METAL_GLM_MOE_DECODE") {
        return Ok(false);
    }
    if gate.is_empty()
        || gate.len() > 8
        || gate.len() != up.len()
        || gate.len() != down.len()
        || gate.len() != route_weights.len()
        || shared_gate.is_empty()
        || shared_up.is_empty()
        || shared_down.is_empty()
        || n_ff == 0
        || n_embd == 0
        || n_ff % 256 != 0
        || n_embd % 256 != 0
        || input.len() != n_embd
        || out.len() != n_embd
    {
        return Ok(false);
    }
    let result = METAL.with(|backend| {
        backend.glm_moe_decode_iq2xxs_iq3xxs(
            gate,
            up,
            down,
            route_weights,
            shared_gate,
            shared_up,
            shared_down,
            shared_route_weight,
            n_ff,
            n_embd,
            input,
            rnb_backend_metal::GlmMoeQuantSelect {
                gate_up_iq2s,
                down_iq4xs: down_is_iq4xs,
                shared_gate_up_q6k,
                shared_down_q8_0,
            },
        )
    });
    out.copy_from_slice(&result);
    Ok(true)
}

/// pm113: GLM MoE prefill token-batch 를 Metal 로 라우팅. slot 배열은
/// `[seq][sparse+1]` flatten (마지막 slot 이 shared expert, route weight 는
/// 토큰별 gate_scalar). `RNB_METAL_GLM_MOE_PREFILL_BATCH=0` opt-out, 기본 ON.
#[allow(clippy::too_many_arguments)]
pub fn metal_glm_moe_prefill_iq_batch_into_if_supported(
    gate: &[&[u8]],
    up: &[&[u8]],
    down: &[&[u8]],
    route_weights: &[f32],
    seq_len: usize,
    sparse_slots: usize,
    n_ff: usize,
    n_embd: usize,
    input_all: &[f32],
    out: &mut [f32],
    gate_up_iq2s: bool,
    down_is_iq4xs: bool,
    shared_gate_up_q6k: bool,
    shared_down_q8_0: bool,
    file_regions: Option<&[rnb_core::tensor::FileBackedRegion; 3]>,
) -> Result<bool> {
    if env_falsey("RNB_METAL_GLM_MOE_PREFILL_BATCH") || env_falsey("RNB_METAL_GLM_MOE_DECODE") {
        return Ok(false);
    }
    let slots = sparse_slots + 1;
    if seq_len == 0
        || sparse_slots == 0
        || sparse_slots > 8
        || gate.len() != seq_len * slots
        || up.len() != seq_len * slots
        || down.len() != seq_len * slots
        || route_weights.len() != seq_len * slots
        || n_ff == 0
        || n_embd == 0
        || n_ff % 256 != 0
        || n_embd % 256 != 0
        || input_all.len() != seq_len * n_embd
        || out.len() != seq_len * n_embd
    {
        return Ok(false);
    }
    let direct_file = file_regions.map(|regions| rnb_backend_metal::GlmPrefillDirectFile {
        paths: [regions[0].path(), regions[1].path(), regions[2].path()],
        base_offsets: [
            regions[0].file_offset(),
            regions[1].file_offset(),
            regions[2].file_offset(),
        ],
        base_addrs: [
            regions[0].host_address(),
            regions[1].host_address(),
            regions[2].host_address(),
        ],
        lens: [regions[0].len(), regions[1].len(), regions[2].len()],
    });
    let result = METAL.with(|backend| {
        backend.glm_moe_prefill_iq_batch(
            gate,
            up,
            down,
            route_weights,
            seq_len,
            sparse_slots,
            n_ff,
            n_embd,
            input_all,
            rnb_backend_metal::GlmMoeQuantSelect {
                gate_up_iq2s,
                down_iq4xs: down_is_iq4xs,
                shared_gate_up_q6k,
                shared_down_q8_0,
            },
            direct_file.as_ref(),
        )
    });
    out.copy_from_slice(&result);
    Ok(true)
}

/// pm112: GLM MLA decode dense GEMV (Q5_K | Q8_0) 를 Metal 로 라우팅.
/// weight 는 backend resident 캐시로 1회 wrap. 기본 ON —
/// `RNB_METAL_GLM_MLA_GEMV=0` 은 진단 opt-out.
pub fn metal_glm_mla_gemv_into_if_supported(
    ggml_type: GGMLType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    if env_falsey("RNB_METAL_GLM_MLA_GEMV") {
        return Ok(false);
    }
    if rows == 0 || input.len() != cols || out.len() != rows {
        return Ok(false);
    }
    let result = match ggml_type {
        GGMLType::Q5_K if cols % 256 == 0 => {
            METAL.with(|backend| backend.gemv_q5k_mla_resident(raw, input, rows, cols))
        }
        GGMLType::Q8_0 if cols % 32 == 0 => {
            METAL.with(|backend| backend.gemv_q8_0_mla_resident(raw, input, rows, cols))
        }
        _ => return Ok(false),
    };
    out.copy_from_slice(&result);
    Ok(true)
}

/// pm112: GLM MLA per-head Q8_0 GEMV 배치 (k_b/v_b, 64 heads → encode 64 + wait 1회).
/// input 은 `[heads*cols]` packing, out 은 `[heads*rows_per_head]`.
/// `RNB_METAL_GLM_MLA_GEMV=0` 은 단발 GEMV 와 같은 opt-out 을 공유한다.
pub fn metal_glm_mla_head_gemv_into_if_supported(
    ggml_type: GGMLType,
    raw: &[u8],
    heads: usize,
    rows_per_head: usize,
    cols: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    if env_falsey("RNB_METAL_GLM_MLA_GEMV") {
        return Ok(false);
    }
    if ggml_type != GGMLType::Q8_0
        || heads == 0
        || cols % 32 != 0
        || input.len() != heads * cols
        || out.len() != heads * rows_per_head
    {
        return Ok(false);
    }
    let result = METAL.with(|backend| {
        backend.glm_mla_head_gemv_q8_0_resident(raw, input, heads, rows_per_head, cols)
    });
    out.copy_from_slice(&result);
    Ok(true)
}

/// pm113: GLM MLA prefill slot-batch GEMV (slot = token*heads+head 전체를 단일
/// dispatch). `RNB_METAL_GLM_MLA_GEMV=0` opt-out은 decode 경로와 공유.
#[allow(clippy::too_many_arguments)]
pub fn metal_glm_mla_head_slots_gemv_into_if_supported(
    ggml_type: GGMLType,
    raw: &[u8],
    slots: usize,
    heads: usize,
    rows_per_head: usize,
    cols: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    if env_falsey("RNB_METAL_GLM_MLA_GEMV") {
        return Ok(false);
    }
    if ggml_type != GGMLType::Q8_0
        || heads == 0
        || slots == 0
        || cols % 32 != 0
        || input.len() != slots * cols
        || out.len() != slots * rows_per_head
    {
        return Ok(false);
    }
    let result = METAL.with(|backend| {
        backend.glm_mla_head_gemv_q8_0_slots_resident(raw, input, slots, heads, rows_per_head, cols)
    });
    out.copy_from_slice(&result);
    Ok(true)
}

/// pm112: GLM MLA front chain (q_a Q5K → rms → q_b Q8_0, kv_a Q8_0, k_b Q8_0
/// 64-head) 를 단일 command buffer 로. `RNB_METAL_GLM_MLA_CHAIN=0` opt-out.
#[allow(clippy::too_many_arguments)]
pub fn metal_glm_mla_front_into_if_supported(
    qa_quant: GGMLType,
    qb_quant: GGMLType,
    kva_quant: GGMLType,
    kb_quant: GGMLType,
    qa: &[u8],
    qa_norm: &[u8],
    qb: &[u8],
    kva: &[u8],
    kb: &[u8],
    hidden: usize,
    q_rank: usize,
    q_dim: usize,
    kv_dim: usize,
    heads: usize,
    kv_rank: usize,
    q_nope: usize,
    value_dim: usize,
    eps: f32,
    input: &[f32],
    q_out: &mut [f32],
    kv_out: &mut [f32],
    qabs_out: &mut [f32],
) -> Result<bool> {
    if env_falsey("RNB_METAL_GLM_MLA_CHAIN") {
        return Ok(false);
    }
    if qa_quant != GGMLType::Q5_K
        || qb_quant != GGMLType::Q8_0
        || kva_quant != GGMLType::Q8_0
        || kb_quant != GGMLType::Q8_0
        || hidden % 256 != 0
        || q_rank % 256 != 0
        || q_nope % 32 != 0
        || heads == 0
        || q_dim % heads != 0
        || heads * value_dim != q_dim
        || qa_norm.len() != q_rank * std::mem::size_of::<f32>()
        || input.len() != hidden
        || q_out.len() != q_dim
        || kv_out.len() != kv_dim
        || qabs_out.len() != heads * kv_rank
    {
        return Ok(false);
    }
    METAL.with(|backend| {
        backend.glm_mla_front(
            qa, qa_norm, qb, kva, kb, hidden, q_rank, q_dim, kv_dim, heads, kv_rank, q_nope,
            value_dim, eps, input, q_out, kv_out, qabs_out,
        )
    });
    Ok(true)
}

/// pm112: GLM MLA back chain (v_b Q8_0 64-head → o Q5K) 를 단일 command buffer 로.
#[allow(clippy::too_many_arguments)]
pub fn metal_glm_mla_back_into_if_supported(
    vb_quant: GGMLType,
    o_quant: GGMLType,
    vb: &[u8],
    o: &[u8],
    hidden: usize,
    q_rank: usize,
    q_dim: usize,
    kv_dim: usize,
    heads: usize,
    kv_rank: usize,
    q_nope: usize,
    value_dim: usize,
    eps: f32,
    latent: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    if env_falsey("RNB_METAL_GLM_MLA_CHAIN") {
        return Ok(false);
    }
    if vb_quant != GGMLType::Q8_0
        || o_quant != GGMLType::Q5_K
        || kv_rank % 32 != 0
        || heads == 0
        || heads * value_dim != q_dim
        || q_dim % 256 != 0
        || latent.len() != heads * kv_rank
        || out.len() != hidden
    {
        return Ok(false);
    }
    METAL.with(|backend| {
        backend.glm_mla_back(
            vb, o, hidden, q_rank, q_dim, kv_dim, heads, kv_rank, q_nope, value_dim, eps, latent,
            out,
        )
    });
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gdn_f32_dual_proj_if_supported(
    left_ggml: GGMLType,
    right_ggml: GGMLType,
    left_raw: &[u8],
    right_raw: &[u8],
    normed: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    n_out: usize,
) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
    if normed.len() != seq_len * hidden_dim {
        return Ok(None);
    }
    if let (Some(left_quant), Some(right_quant)) = (
        tensorops_quant_from_ggml(left_ggml),
        tensorops_quant_from_ggml(right_ggml),
    ) {
        if std::env::var("RNB_METAL_PREFILL_GDN_INPROJ").as_deref() == Ok("0") {
            return Ok(None);
        }
        return Ok(Some(METAL.with(|b| {
            b.prefill_gdn_quant_dual_proj(
                normed,
                left_raw,
                left_quant,
                right_raw,
                right_quant,
                seq_len,
                hidden_dim,
                n_out,
            )
        })));
    }
    let requested = std::env::var("RNB_METAL_PREFILL_GDN_F32_DUAL")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false);
    if !requested {
        return Ok(None);
    }
    if left_ggml != GGMLType::F32 || right_ggml != GGMLType::F32 {
        return Ok(None);
    }
    if left_raw.len() < n_out * hidden_dim * std::mem::size_of::<f32>()
        || right_raw.len() < n_out * hidden_dim * std::mem::size_of::<f32>()
    {
        return Ok(None);
    }
    Ok(Some(METAL.with(|b| {
        b.prefill_gdn_f32_dual_proj(normed, left_raw, right_raw, seq_len, hidden_dim, n_out)
    })))
}

/// pm35 M2 + pm36: prefill GDN proj(in_proj/gate/ssm_out) single GEMM. weight Q4_K|Q5_K|Q6_K
/// 만 GPU, 그 외 host fallback. RNB_METAL_PREFILL_GDN_INPROJ=1 opt-in. out[seq_len*n_out] 채우고
/// used=true. pm36: Q5_K 추가(27B ssm_out + 9B 전체 GDN projection).
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gdn_proj_into_if_supported(
    weight_ggml: GGMLType,
    weight_raw: &[u8],
    normed: &[f32],
    out: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    n_out: usize,
) -> Result<bool> {
    metal_prefill_gdn_proj_into_if_supported_with_trace(
        weight_ggml,
        weight_raw,
        normed,
        out,
        seq_len,
        hidden_dim,
        n_out,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gdn_proj_into_if_supported_with_trace(
    weight_ggml: GGMLType,
    weight_raw: &[u8],
    normed: &[f32],
    out: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    n_out: usize,
    trace: Option<MetalPrefillProjTrace>,
) -> Result<bool> {
    // pm37: default ON(metal prefill GPU 승격, 27B token-identical 검증). opt-out=RNB_METAL_PREFILL_GDN_INPROJ=0.
    if std::env::var("RNB_METAL_PREFILL_GDN_INPROJ").as_deref() == Ok("0") {
        return Ok(false);
    }
    let quant = match weight_ggml {
        GGMLType::Q4_K => TensoropsQuant::Q4K,
        GGMLType::Q5_K => TensoropsQuant::Q5K,
        GGMLType::Q6_K => TensoropsQuant::Q6K,
        // Q8_0/F32 등 → host fallback (해당 tensorops 커널 없음 — 블록 크기 오독 방지)
        _ => return Ok(false),
    };
    let backend_trace = trace.map(|trace| rnb_backend_metal::PrefillProjTrace {
        role: trace.role,
        layer_idx: trace.layer_idx,
        timing_enabled: trace.timing_enabled,
    });
    let r = METAL.with(|b| {
        b.prefill_gdn_proj_with_trace(
            normed,
            weight_raw,
            quant,
            seq_len,
            hidden_dim,
            n_out,
            backend_trace,
        )
    });
    out.copy_from_slice(&r);
    Ok(true)
}

fn tensorops_quant_from_ggml(ggml: GGMLType) -> Option<rnb_backend_metal::TensoropsQuant> {
    match ggml {
        GGMLType::Q4_K => Some(rnb_backend_metal::TensoropsQuant::Q4K),
        GGMLType::Q5_K => Some(rnb_backend_metal::TensoropsQuant::Q5K),
        GGMLType::Q6_K => Some(rnb_backend_metal::TensoropsQuant::Q6K),
        _ => None,
    }
}

pub fn metal_prefill_atn_core_if_supported(
    req: MetalPrefillAtnCoreRequest<'_>,
) -> Result<Option<MetalPrefillAtnCoreOut>> {
    if !metal_prefill_atn_full_layer_requested() {
        atn_full_counter_update(|c| c.opt_out += 1);
        return Ok(None);
    }
    let Some(q_quant) = tensorops_quant_from_ggml(req.q_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(k_quant) = tensorops_quant_from_ggml(req.k_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(v_quant) = tensorops_quant_from_ggml(req.v_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };

    let backend_req = rnb_backend_metal::PrefillAtnCoreBackendRequest {
        hidden: req.hidden,
        attn_norm_w: req.attn_norm_w,
        q_norm_w: req.q_norm_w,
        k_norm_w: req.k_norm_w,
        q_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.q_weight_raw,
            quant: q_quant,
            rows: req.q_weight_rows,
            cols: req.q_weight_cols,
        },
        k_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.k_weight_raw,
            quant: k_quant,
            rows: req.k_weight_rows,
            cols: req.k_weight_cols,
        },
        v_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.v_weight_raw,
            quant: v_quant,
            rows: req.v_weight_rows,
            cols: req.v_weight_cols,
        },
        seq_len: req.seq_len,
        num_heads: req.num_heads,
        num_kv_heads: req.num_kv_heads,
        head_dim: req.head_dim,
        hidden_dim: req.hidden_dim,
        q_dim: req.q_dim,
        kv_dim: req.kv_dim,
        n_rot: req.n_rot,
        rope_theta: req.rope_theta,
        scale: req.scale,
        norm_eps: req.norm_eps,
        pos_start: req.pos_start,
    };
    let result = METAL.with(|b| b.prefill_atn_core_if_supported(backend_req));
    match result {
        Ok(Some((attn_out, k_bits, v_bits))) => {
            metal_prefill_atn_full_record_core_hit();
            Ok(Some(MetalPrefillAtnCoreOut {
                attn_out,
                k_bits,
                v_bits,
            }))
        }
        Ok(None) => {
            metal_prefill_atn_full_record_skip();
            Ok(None)
        }
        Err(err) => {
            metal_prefill_atn_full_record_backend_err();
            Err(err)
        }
    }
}

pub fn metal_prefill_atn_o_tail_if_supported(
    req: MetalPrefillAtnOTailRequest<'_>,
) -> Result<Option<MetalPrefillAtnOTailOut>> {
    if !metal_prefill_atn_o_tail_requested() {
        atn_o_tail_counter_update(|c| c.opt_out += 1);
        return Ok(None);
    }
    let Some(q_quant) = tensorops_quant_from_ggml(req.core.q_weight_ggml) else {
        metal_prefill_atn_o_tail_record_skip();
        return Ok(None);
    };
    let Some(k_quant) = tensorops_quant_from_ggml(req.core.k_weight_ggml) else {
        metal_prefill_atn_o_tail_record_skip();
        return Ok(None);
    };
    let Some(v_quant) = tensorops_quant_from_ggml(req.core.v_weight_ggml) else {
        metal_prefill_atn_o_tail_record_skip();
        return Ok(None);
    };
    let Some(o_quant) = tensorops_quant_from_ggml(req.o_weight_ggml) else {
        metal_prefill_atn_o_tail_record_skip();
        return Ok(None);
    };

    let core = rnb_backend_metal::PrefillAtnCoreBackendRequest {
        hidden: req.core.hidden,
        attn_norm_w: req.core.attn_norm_w,
        q_norm_w: req.core.q_norm_w,
        k_norm_w: req.core.k_norm_w,
        q_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.core.q_weight_raw,
            quant: q_quant,
            rows: req.core.q_weight_rows,
            cols: req.core.q_weight_cols,
        },
        k_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.core.k_weight_raw,
            quant: k_quant,
            rows: req.core.k_weight_rows,
            cols: req.core.k_weight_cols,
        },
        v_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.core.v_weight_raw,
            quant: v_quant,
            rows: req.core.v_weight_rows,
            cols: req.core.v_weight_cols,
        },
        seq_len: req.core.seq_len,
        num_heads: req.core.num_heads,
        num_kv_heads: req.core.num_kv_heads,
        head_dim: req.core.head_dim,
        hidden_dim: req.core.hidden_dim,
        q_dim: req.core.q_dim,
        kv_dim: req.core.kv_dim,
        n_rot: req.core.n_rot,
        rope_theta: req.core.rope_theta,
        scale: req.core.scale,
        norm_eps: req.core.norm_eps,
        pos_start: req.core.pos_start,
    };
    let backend_req = rnb_backend_metal::PrefillAtnOTailBackendRequest {
        core,
        o_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.o_weight_raw,
            quant: o_quant,
            rows: req.o_weight_rows,
            cols: req.o_weight_cols,
        },
    };
    let result = METAL.with(|b| b.prefill_atn_o_tail_if_supported(backend_req));
    match result {
        Ok(Some((hidden, k_bits, v_bits))) => {
            metal_prefill_atn_o_tail_record_hit();
            Ok(Some(MetalPrefillAtnOTailOut {
                hidden,
                k_bits,
                v_bits,
            }))
        }
        Ok(None) => {
            metal_prefill_atn_o_tail_record_skip();
            Ok(None)
        }
        Err(err) => {
            metal_prefill_atn_o_tail_record_backend_err();
            Err(err)
        }
    }
}

pub fn metal_prefill_atn_full_layer_if_supported(
    req: MetalPrefillAtnFullLayerRequest<'_>,
) -> Result<Option<MetalPrefillAtnFullLayerOut>> {
    if !metal_prefill_atn_full_layer_requested() {
        atn_full_counter_update(|c| c.opt_out += 1);
        return Ok(None);
    }
    let Some(q_quant) = tensorops_quant_from_ggml(req.core.q_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(k_quant) = tensorops_quant_from_ggml(req.core.k_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(v_quant) = tensorops_quant_from_ggml(req.core.v_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(o_quant) = tensorops_quant_from_ggml(req.o_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(ffn_gate_quant) = tensorops_quant_from_ggml(req.ffn_gate_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(ffn_up_quant) = tensorops_quant_from_ggml(req.ffn_up_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };
    let Some(ffn_down_quant) = tensorops_quant_from_ggml(req.ffn_down_weight_ggml) else {
        metal_prefill_atn_full_record_skip();
        return Ok(None);
    };

    let core = rnb_backend_metal::PrefillAtnCoreBackendRequest {
        hidden: req.core.hidden,
        attn_norm_w: req.core.attn_norm_w,
        q_norm_w: req.core.q_norm_w,
        k_norm_w: req.core.k_norm_w,
        q_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.core.q_weight_raw,
            quant: q_quant,
            rows: req.core.q_weight_rows,
            cols: req.core.q_weight_cols,
        },
        k_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.core.k_weight_raw,
            quant: k_quant,
            rows: req.core.k_weight_rows,
            cols: req.core.k_weight_cols,
        },
        v_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.core.v_weight_raw,
            quant: v_quant,
            rows: req.core.v_weight_rows,
            cols: req.core.v_weight_cols,
        },
        seq_len: req.core.seq_len,
        num_heads: req.core.num_heads,
        num_kv_heads: req.core.num_kv_heads,
        head_dim: req.core.head_dim,
        hidden_dim: req.core.hidden_dim,
        q_dim: req.core.q_dim,
        kv_dim: req.core.kv_dim,
        n_rot: req.core.n_rot,
        rope_theta: req.core.rope_theta,
        scale: req.core.scale,
        norm_eps: req.core.norm_eps,
        pos_start: req.core.pos_start,
    };
    let backend_req = rnb_backend_metal::PrefillAtnFullLayerBackendRequest {
        core,
        o_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.o_weight_raw,
            quant: o_quant,
            rows: req.o_weight_rows,
            cols: req.o_weight_cols,
        },
        ffn_norm_w: req.ffn_norm_w,
        ffn_gate_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.ffn_gate_weight_raw,
            quant: ffn_gate_quant,
            rows: req.ffn_gate_weight_rows,
            cols: req.ffn_gate_weight_cols,
        },
        ffn_up_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.ffn_up_weight_raw,
            quant: ffn_up_quant,
            rows: req.ffn_up_weight_rows,
            cols: req.ffn_up_weight_cols,
        },
        ffn_down_weight: rnb_backend_metal::PrefillAtnCoreWeightView {
            raw: req.ffn_down_weight_raw,
            quant: ffn_down_quant,
            rows: req.ffn_down_weight_rows,
            cols: req.ffn_down_weight_cols,
        },
        ffn_dim: req.ffn_dim,
    };
    let result = METAL.with(|b| b.prefill_atn_full_layer_if_supported(backend_req));
    match result {
        Ok(Some((hidden, k_bits, v_bits))) => {
            metal_prefill_atn_full_record_full_layer_hit();
            Ok(Some(MetalPrefillAtnFullLayerOut {
                hidden,
                k_bits,
                v_bits,
            }))
        }
        Ok(None) => {
            metal_prefill_atn_full_record_skip();
            Ok(None)
        }
        Err(err) => {
            metal_prefill_atn_full_record_backend_err();
            Err(err)
        }
    }
}

/// pm43: GDN prefill conv1d+silu Metal seam(gpu_gdn 공통 facade가 cuda/metal 분기로 호출).
/// input[(seq_len+kernel_size-1)*channels] + weight[kernel_size*channels] → out[seq_len*channels].
/// f32 exact(quant 무관, token-identical). default ON, opt-out=RNB_METAL_PREFILL_CONV1D=0.
pub fn metal_prefill_conv1d_silu_into_if_supported(
    input: &[f32],
    weight: &[f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) -> Option<Vec<f32>> {
    if std::env::var("RNB_METAL_PREFILL_CONV1D").as_deref() == Ok("0") {
        return None;
    }
    let r = METAL.with(|b| b.prefill_conv1d_silu(input, weight, seq_len, channels, kernel_size));
    Some(r)
}

/// pm48 ①: dense causal GQA prefill attention compute Metal seam(host 입출력, 1차).
/// q[seq*nh*hd] f32(rope/qk_norm 완료), k/v[kv*nkv*hd] f16 bits → out[seq*nh*hd] f32.
/// rnb-llm 은 backend env 를 직접 안 읽으므로 gate(`RNB_METAL_PREFILL_FLASH_ATTN`)를 여기서 read.
/// **default OFF(첫 검증용) — ABAB ≥10% 확인 후 ON 승격.** sliding_window/softcap 있으면 호출자가
/// 미진입(이 함수는 dense Qwen 전용). head_dim!=256 또는 nh%nkv!=0 또는 non-M5 면 None(CPU fallback).
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_attention_flash_if_supported(
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
) -> Option<Vec<f32>> {
    // default OFF: opt-in 으로만 진입(검증 후 default ON 승격 예정).
    if std::env::var("RNB_METAL_PREFILL_FLASH_ATTN").as_deref() != Ok("1") {
        return None;
    }
    // dense Qwen 전용: sliding window / softcap 은 미지원(Gemma 등은 CPU 경로 유지).
    if has_sliding_window || has_softcap {
        return None;
    }
    // simdgroup matmul2d 커널은 HD=256 컴파일타임 고정 + GQA(nh%nkv==0) 전제.
    if head_dim != 256 || num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return None;
    }
    METAL.with(|b| {
        b.prefill_flash_attention(
            q,
            k_f16,
            v_f16,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    })
}

/// pm48 ②: prefill attention 2차 device-resident chain seam(rope/qk_norm→cast→flash 단일
/// command buffer). 입력(host): q_proj(gate split 후, norm 전), k_proj(norm 전), v(f32),
/// q_norm/k_norm weight. 반환 `(attn_out, k_f16, v_f16)` — k_f16/v_f16 은 device 에서 만든
/// f16 KV bits(호출자가 host KV cache 에 write). gate `RNB_METAL_PREFILL_ATTN_CHAIN`. 1차와
/// 마찬가지로 dense Qwen 전용(head_dim==256, GQA, n_rot partial, pos_start==0 prefill).
/// None 반환(gate OFF / shape 미충족 / non-M5) 시 caller 가 기존 CPU rope/qk_norm + 1차/CPU
/// attention 으로 fallback.
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_attn_chain_if_supported(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
    eps: f32,
    pos_start: usize,
    scale: f32,
    has_sliding_window: bool,
    has_softcap: bool,
    layer_idx: usize,
    timing_enabled: bool,
) -> Option<(Vec<f32>, Vec<u16>, Vec<u16>)> {
    // pm48 ②: 27B ABAB −10.43%/−12.36%(2회, 겹침0, argmax token-identical) → (A) dual-path
    // winner ≥10% 충족 → default ON 승격. opt-out=`RNB_METAL_PREFILL_ATTN_CHAIN=0`.
    if std::env::var("RNB_METAL_PREFILL_ATTN_CHAIN").as_deref() == Ok("0") {
        return None;
    }
    // dense Qwen 전용: sliding window / softcap 미지원. prefill only(pos_start==0).
    if has_sliding_window || has_softcap || pos_start != 0 {
        return None;
    }
    // flash 커널 HD=256 고정 + GQA(nh%nkv==0).
    if head_dim != 256 || num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return None;
    }
    METAL.with(|b| {
        b.prefill_attn_chain(
            q_in,
            k_in,
            v_in,
            q_norm_w,
            k_norm_w,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            n_rot,
            theta,
            eps,
            pos_start,
            scale,
            layer_idx,
            timing_enabled,
        )
    })
}

/// pm43: GDN prefill gated RMSNorm+SiLU Metal seam(gpu_gdn 공통 facade). output[rows*cols] +
/// z[rows*cols] + norm[cols](ssm_norm) → gated[rows*cols] = rmsnorm(output per row)·silu(z).
/// rows=seq_len*num_v_heads, cols=head_v_dim. default ON, opt-out=RNB_METAL_PREFILL_GATED_NORM=0.
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gated_norm_silu_into_if_supported(
    output: &[f32],
    z: &[f32],
    norm: &[f32],
    _seq_len: usize,
    rows: usize,
    cols: usize,
    norm_eps: f32,
) -> Option<Vec<f32>> {
    if std::env::var("RNB_METAL_PREFILL_GATED_NORM").as_deref() == Ok("0") {
        return None;
    }
    let r = METAL.with(|b| b.prefill_gated_norm_silu(output, z, norm, rows, cols, norm_eps));
    Some(r)
}

/// pm44 M1: GDN prefill gated→ssm_out fused Metal seam. gated_norm_silu(out_in,z,norm) → ssm_out
/// proj(weight Q4_K|Q5_K|Q6_K) 를 단일 command buffer device-resident 로. host 왕복(gated readback +
/// ssm_out upload + commit/wait 1회) 제거. default ON, opt-out=RNB_METAL_PREFILL_GATED_PROJ=0.
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gated_norm_silu_project_into_if_supported(
    output: &[f32],
    z: &[f32],
    norm: &[f32],
    weight_ggml: GGMLType,
    weight_raw: &[u8],
    seq_len: usize,
    head_v_dim: usize,
    n_out: usize,
    norm_eps: f32,
) -> Option<Vec<f32>> {
    if std::env::var("RNB_METAL_PREFILL_GATED_PROJ").as_deref() == Ok("0") {
        return None;
    }
    let quant = match weight_ggml {
        GGMLType::Q4_K => TensoropsQuant::Q4K,
        GGMLType::Q5_K => TensoropsQuant::Q5K,
        GGMLType::Q6_K => TensoropsQuant::Q6K,
        _ => return None, // tensorops 커널 없는 quant → 분리 경로 fallback
    };
    let r = METAL.with(|b| {
        b.prefill_gated_norm_silu_project(
            output, z, norm, weight_raw, quant, seq_len, head_v_dim, n_out, norm_eps,
        )
    });
    Some(r)
}

/// pm39 M2/M3: chunk_size 를 threadgroup memory budget 에서 산정(고정값 금지 — CLAUDE.md 비례 정책).
/// 커널 dynamic threadgroup: u_corr(cs*head_v_dim) + kk_sh(cs²) + qk_sh(cs²) f32. 정적 g_cum[256]=1KiB
/// 여유 후 31KiB(=7936 f32) budget. 2·cs² + head_v_dim·cs ≤ budget 의 최대 cs.
///   cs ≤ (−hv + √(hv² + 8·budget)) / 4.
/// 측정 override: RNB_METAL_DELTA_SCAN_CHUNK=<n>(cs 스윕용). 하한 1, 상한 64(launch/측정 안정).
fn pick_delta_scan_chunk_size(head_v_dim: usize) -> usize {
    if let Ok(s) = std::env::var("RNB_METAL_DELTA_SCAN_CHUNK") {
        if let Ok(c) = s.parse::<usize>() {
            if c >= 1 {
                return c;
            }
        }
    }
    const BUDGET_F32: usize = (31 * 1024) / 4; // 7936 floats
    let hv = head_v_dim as f64;
    let cs_max = ((-hv + (hv * hv + 8.0 * BUDGET_F32 as f64).sqrt()) / 4.0).floor() as usize;
    cs_max.clamp(1, 64)
}

/// pm45 M2: GDN prefill conv→delta device-resident chain Metal seam. conv1d_silu →
/// split_conv_qkv → l2_norm(q,k) → repeat_qk → scale(q) → delta_net_scan_chunk 를 단일
/// command buffer 로 묶어 op 별 host 왕복 제거. conv_input(raw, scale 미적용)/conv_weight
/// (w.ssm_conv1d) + gate/beta + state(delta_state) → (output[seq*num_v_heads*head_v_dim],
/// state_after). chunk_size 는 `pick_delta_scan_chunk_size`(delta scan seam 과 동일 산정 +
/// 동일 budget 가드)로 — 다르면 token 깨짐. default ON, opt-out=RNB_METAL_PREFILL_GDN_CONV_DELTA=0.
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gdn_conv_delta_chain_into_if_supported(
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    norm_eps: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if std::env::var("RNB_METAL_PREFILL_GDN_CONV_DELTA").as_deref() == Ok("0") {
        return None;
    }
    // delta scan seam(metal_prefill_delta_net_scan_into_if_supported)과 동일한 chunk_size 산정.
    // 다르면 chunkwise scan 결과가 갈려 token 깨짐.
    let chunk_size = pick_delta_scan_chunk_size(head_v_dim);
    // 동일 budget 가드(거대 head_v_dim → GPU 미지원 → CPU fallback). delta scan seam 과 일치.
    if (chunk_size * head_v_dim + 2 * chunk_size * chunk_size + 256) * 4 > 32 * 1024 {
        return None;
    }
    let r = METAL.with(|b| {
        b.prefill_gdn_conv_delta_chain(
            conv_input,
            conv_weight,
            gate,
            beta,
            state,
            seq_len,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            norm_eps,
        )
    });
    Some(r)
}

/// pm45 M3-1: GDN prefill full chain(conv→delta→gated→ssm_out) Metal seam. M2 chain(conv→delta)
/// 끝 delta output 을 readback 하지 않고 같은 command buffer 에 이어서 gated_rmsnorm_silu →
/// cast → ssm_out GEMM_v2 를 device-resident 로. M2↔M1 사이 host 왕복(delta output readback +
/// gated input upload) 제거. M2 seam 입력 + M1 입력(z/ssm_norm/ssm_out weight) 추가.
/// chunk_size 산정/budget 가드는 M2 seam 과 동일(token 동일성 사활). ssm_out weight 가 tensorops
/// 미지원 quant 면 None(분리 경로 fallback). default ON, opt-out=RNB_METAL_PREFILL_GDN_FULL=0.
/// 반환 `(proj[seq*n_out], state_after)`.
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gdn_full_chain_into_if_supported(
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    z: &[f32],
    ssm_norm: &[f32],
    ssm_out_ggml: GGMLType,
    ssm_out_raw: &[u8],
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    n_out: usize,
    norm_eps: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if std::env::var("RNB_METAL_PREFILL_GDN_FULL").as_deref() == Ok("0") {
        return None;
    }
    // ssm_out weight 가 tensorops GEMM_v2 지원 quant 인지 먼저 확인(미지원이면 분리 경로 fallback).
    let quant = match ssm_out_ggml {
        GGMLType::Q4_K => TensoropsQuant::Q4K,
        GGMLType::Q5_K => TensoropsQuant::Q5K,
        GGMLType::Q6_K => TensoropsQuant::Q6K,
        _ => return None,
    };
    // M2 seam 과 동일한 chunk_size 산정 + budget 가드(다르면 chunkwise scan 결과가 갈려 token 깨짐).
    let chunk_size = pick_delta_scan_chunk_size(head_v_dim);
    if (chunk_size * head_v_dim + 2 * chunk_size * chunk_size + 256) * 4 > 32 * 1024 {
        return None;
    }
    let r = METAL.with(|b| {
        b.prefill_gdn_full_chain(
            conv_input,
            conv_weight,
            gate,
            beta,
            state,
            z,
            ssm_norm,
            ssm_out_raw,
            quant,
            seq_len,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            n_out,
            norm_eps,
        )
    });
    Some(r)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_gdn_full_ffn_chain_into_if_supported(
    hidden: &[f32],
    conv_input: &[f32],
    conv_weight: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &[f32],
    z: &[f32],
    ssm_norm: &[f32],
    ssm_out_ggml: GGMLType,
    ssm_out_raw: &[u8],
    post_norm_w: &[f32],
    ffn_gate_ggml: GGMLType,
    ffn_gate_raw: &[u8],
    ffn_up_ggml: GGMLType,
    ffn_up_raw: &[u8],
    ffn_down_ggml: GGMLType,
    ffn_down_raw: &[u8],
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_eps: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if !metal_prefill_gdn_full_ffn_enabled() {
        return None;
    }
    if std::env::var("RNB_METAL_PREFILL_GDN_FULL").as_deref() == Ok("0") {
        return None;
    }
    let ssm_out_quant = match ssm_out_ggml {
        GGMLType::Q4_K => TensoropsQuant::Q4K,
        GGMLType::Q5_K => TensoropsQuant::Q5K,
        GGMLType::Q6_K => TensoropsQuant::Q6K,
        _ => return None,
    };
    if ffn_gate_ggml != GGMLType::Q4_K || ffn_up_ggml != GGMLType::Q4_K {
        return None;
    }
    let ffn_down_is_q6k = match ffn_down_ggml {
        GGMLType::Q4_K => false,
        GGMLType::Q6_K => true,
        _ => return None,
    };
    let chunk_size = pick_delta_scan_chunk_size(head_v_dim);
    if (chunk_size * head_v_dim + 2 * chunk_size * chunk_size + 256) * 4 > 32 * 1024 {
        return None;
    }
    METAL.with(|b| {
        b.prefill_gdn_full_ffn_chain_if_supported(
            hidden,
            conv_input,
            conv_weight,
            gate,
            beta,
            state,
            z,
            ssm_norm,
            ssm_out_raw,
            ssm_out_quant,
            post_norm_w,
            ffn_gate_raw,
            ffn_up_raw,
            ffn_down_raw,
            ffn_down_is_q6k,
            seq_len,
            conv_channels,
            conv_kernel,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            chunk_size,
            hidden_dim,
            ffn_dim,
            norm_eps,
        )
    })
}

/// pm39 M3: prefill GDN delta scan(순차 recurrence)을 Metal GPU chunkwise parallel scan 으로.
/// `state` 는 chunk hand-off 로 in-place 갱신, `out`[seq_len*num_heads*head_v_dim] 채우고 used=true.
/// GQA 는 caller 가 q/k 를 num_heads(=num_v_heads) 로 repeat 푼 뒤 넘긴다.
/// pm39 M3 default ON 승격: Qwen3.5-9B + Qwen3.6-27B(hk=hv=128) token-identical + 27B ABAB
/// prefill −9.9%(median, 겹침0). opt-out=RNB_METAL_PREFILL_GDN_SCAN=0.
#[allow(clippy::too_many_arguments)]
pub fn metal_prefill_delta_net_scan_into_if_supported(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<bool> {
    // default ON(opt-out). 9B+27B token-identical + ABAB winner 검증 → 승격(MEMORY: 채택이면 default ON).
    if std::env::var("RNB_METAL_PREFILL_GDN_SCAN").as_deref() == Ok("0") {
        return Ok(false);
    }
    let chunk_size = pick_delta_scan_chunk_size(head_v_dim);
    // threadgroup budget 초과(거대 head_v_dim) → CPU fallback. cs 비례 산정이 보통 budget 안에 들지만
    // cs=1 에서도 초과하는 극단 shape 는 GPU 미지원 → panic 대신 Ok(false)(capability gate).
    if (chunk_size * head_v_dim + 2 * chunk_size * chunk_size + 256) * 4 > 32 * 1024 {
        return Ok(false);
    }
    let (output, new_state) = METAL.with(|b| {
        b.delta_net_scan_chunk(
            q, k, v, gate, beta, state, seq_len, num_heads, head_k_dim, head_v_dim, chunk_size,
        )
    });
    out.copy_from_slice(&output);
    state.copy_from_slice(&new_state);
    Ok(true)
}

#[cfg(test)]
mod gdn_proj_wrapper_tests {
    use super::*;

    #[test]
    fn gdn_proj_wrapper_quant_guard_rejects_unsupported_quant() {
        // cargo test 병렬 → 전역 env mutate race 직렬화 (자체 Mutex; policy.rs env_lock 은 private).
        static LK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LK.lock().expect("env lock");
        // pm37: default ON 승격. opt-out(=0) → Ok(false) (device 불요).
        std::env::set_var("RNB_METAL_PREFILL_GDN_INPROJ", "0");
        let mut out = vec![0f32; 4];
        let r = metal_prefill_gdn_proj_into_if_supported(
            GGMLType::Q4_K,
            &[0u8; 144],
            &[0f32; 4],
            &mut out,
            1,
            4,
            4,
        );
        assert!(!r.unwrap(), "opt-out(=0)이면 Ok(false)");
        // Q8_0(미지원 tensorops) → quant 가드 reject(device 도달 전). default ON 이라 env
        // 미설정도 통과하지만 Q8_0 은 가드(_ => Ok(false))에서 device 전 reject.
        std::env::remove_var("RNB_METAL_PREFILL_GDN_INPROJ");
        let r = metal_prefill_gdn_proj_into_if_supported(
            GGMLType::Q8_0,
            &[0u8; 34],
            &[0f32; 4],
            &mut out,
            1,
            4,
            4,
        );
        assert!(
            !r.unwrap(),
            "Q8_0 는 tensorops 커널 없어 가드에서 Ok(false)"
        );
    }

    #[test]
    fn gdn_dual_proj_quantized_obeys_inproj_opt_out_before_device() {
        static LK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LK.lock().expect("env lock");
        std::env::set_var("RNB_METAL_PREFILL_GDN_INPROJ", "0");

        let r = metal_prefill_gdn_f32_dual_proj_if_supported(
            GGMLType::Q4_K,
            GGMLType::Q4_K,
            &[0u8; 144],
            &[0u8; 144],
            &[0f32; 4],
            1,
            4,
            4,
        );

        std::env::remove_var("RNB_METAL_PREFILL_GDN_INPROJ");
        assert!(
            r.unwrap().is_none(),
            "quantized dual opt-out must skip device"
        );
    }
}

#[cfg(test)]
mod gdn_full_ffn_policy_tests {
    use super::metal_prefill_gdn_full_ffn_enabled;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn gdn_full_ffn_defaults_on_and_supports_explicit_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        std::env::remove_var("RNB_METAL_PREFILL_GDN_FULL_FFN");
        assert!(metal_prefill_gdn_full_ffn_enabled());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_PREFILL_GDN_FULL_FFN", value);
            assert!(
                !metal_prefill_gdn_full_ffn_enabled(),
                "{value} should opt out of GDN full+FFN default"
            );
        }

        std::env::set_var("RNB_METAL_PREFILL_GDN_FULL_FFN", "1");
        assert!(metal_prefill_gdn_full_ffn_enabled());
        std::env::remove_var("RNB_METAL_PREFILL_GDN_FULL_FFN");
    }
}

#[cfg(test)]
mod atn_full_policy_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn counters() -> AtnFullCounters {
        ATN_FULL_COUNTERS.with(|c| c.get())
    }

    #[test]
    fn atn_full_layer_is_default_on_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_TIME");
        assert!(metal_prefill_atn_full_layer_requested());
        assert!(!metal_prefill_atn_full_timing_enabled());

        std::env::set_var("RNB_METAL_PREFILL_ATN_FULL_LAYER", "1");
        assert!(metal_prefill_atn_full_layer_requested());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_PREFILL_ATN_FULL_LAYER", value);
            assert!(
                !metal_prefill_atn_full_layer_requested(),
                "{value} should reject ATN full-layer opt-in"
            );
        }

        std::env::set_var("RNB_METAL_PREFILL_ATN_FULL_TIME", "1");
        assert!(metal_prefill_atn_full_timing_enabled());
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_TIME");
    }

    #[test]
    fn atn_full_adapter_reject_splits_opt_out_from_fallback() {
        let _guard = env_lock().lock().expect("env lock");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER");
        metal_prefill_atn_full_counters_reset();

        metal_prefill_atn_full_expected_dense_layer();
        metal_prefill_atn_full_record_adapter_reject();
        let v = counters();
        assert_eq!(v.expected_dense_atn_layers, 1);
        assert_eq!(v.opt_out, 0);
        assert_eq!(v.fallback, 1);
        assert_eq!(v.skip_by_reason, 1);

        std::env::set_var("RNB_METAL_PREFILL_ATN_FULL_LAYER", "0");
        metal_prefill_atn_full_counters_reset();
        metal_prefill_atn_full_expected_dense_layer();
        metal_prefill_atn_full_record_adapter_reject();
        let v = counters();
        assert_eq!(v.expected_dense_atn_layers, 1);
        assert_eq!(v.opt_out, 1);
        assert_eq!(v.fallback, 0);
        assert_eq!(v.skip_by_reason, 0);
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER");
    }
}

#[cfg(test)]
mod atn_o_tail_policy_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn counters() -> AtnOTailCounters {
        ATN_O_TAIL_COUNTERS.with(|c| c.get())
    }

    #[test]
    fn atn_o_tail_is_exact_opt_in_and_independent_of_full_layer_gates() {
        let _guard = env_lock().lock().expect("env lock");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_O_TAIL");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER_TAIL");
        assert!(!metal_prefill_atn_o_tail_requested());

        for value in ["0", "false", "off", "no", "true", "yes", "on"] {
            std::env::set_var("RNB_METAL_PREFILL_ATN_O_TAIL", value);
            assert!(
                !metal_prefill_atn_o_tail_requested(),
                "{value} must not enable exact opt-in o-tail"
            );
        }

        std::env::set_var("RNB_METAL_PREFILL_ATN_O_TAIL", "1");
        std::env::set_var("RNB_METAL_PREFILL_ATN_FULL_LAYER", "0");
        std::env::set_var("RNB_METAL_PREFILL_ATN_FULL_LAYER_TAIL", "0");
        assert!(metal_prefill_atn_o_tail_requested());

        std::env::remove_var("RNB_METAL_PREFILL_ATN_O_TAIL");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER");
        std::env::remove_var("RNB_METAL_PREFILL_ATN_FULL_LAYER_TAIL");
    }

    #[test]
    fn atn_o_tail_expected_counter_is_independent_from_hit_and_fallback() {
        let _guard = env_lock().lock().expect("env lock");
        metal_prefill_atn_o_tail_counters_reset();

        metal_prefill_atn_o_tail_expected_dense_layer();
        let v = counters();
        assert_eq!(v.expected_dense_atn_layers, 1);
        assert_eq!(v.o_tail_hit, 0);
        assert_eq!(v.fallback, 0);

        metal_prefill_atn_o_tail_record_hit();
        let v = counters();
        assert_eq!(v.expected_dense_atn_layers, 1);
        assert_eq!(v.o_tail_hit, 1);
        assert_eq!(v.fallback, 0);
    }
}

#[cfg(test)]
mod metal_qwen_moe_prefill_accum_policy_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn qwen_moe_llama_id_request(
        gate_quant: GGMLType,
        up_quant: GGMLType,
        down_quant: GGMLType,
        shared_gate_quant: GGMLType,
        shared_up_quant: GGMLType,
        shared_down_quant: GGMLType,
    ) -> MetalQwenMoeLlamaIdPrefillRequest<'static> {
        MetalQwenMoeLlamaIdPrefillRequest {
            gate_all: &[],
            up_all: &[],
            down_all: &[],
            gate_expert_bytes: 0,
            up_expert_bytes: 0,
            down_expert_bytes: 0,
            selected_experts: &[],
            route_weights: &[],
            shared_gate: &[],
            shared_up: &[],
            shared_down: &[],
            shared_route_weights: &[],
            gate_quant,
            up_quant,
            down_quant,
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            norm_all: &[],
            seq_len: 0,
            hidden_dim: 0,
            ffn_dim: 0,
            layer_idx: 0,
        }
    }

    #[test]
    fn qwen_moe_llama_id_type_positions_accept_only_product_tuple() {
        for (quants, expected_sparse_down, expected_shared) in [
            (
                [
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                ],
                QwenMoeLlamaIdQuant::Q4K,
                QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q4K,
                    up: QwenMoeLlamaIdQuant::Q4K,
                    down: QwenMoeLlamaIdQuant::Q4K,
                },
            ),
            (
                [
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                    GGMLType::Q6_K,
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                    GGMLType::Q6_K,
                ],
                QwenMoeLlamaIdQuant::Q6K,
                QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q4K,
                    up: QwenMoeLlamaIdQuant::Q4K,
                    down: QwenMoeLlamaIdQuant::Q6K,
                },
            ),
            (
                [
                    GGMLType::Q4_K,
                    GGMLType::Q4_K,
                    GGMLType::Q5_K,
                    GGMLType::Q8_0,
                    GGMLType::Q8_0,
                    GGMLType::Q8_0,
                ],
                QwenMoeLlamaIdQuant::Q5K,
                QwenMoeLlamaIdQuantSet {
                    gate: QwenMoeLlamaIdQuant::Q8Zero,
                    up: QwenMoeLlamaIdQuant::Q8Zero,
                    down: QwenMoeLlamaIdQuant::Q8Zero,
                },
            ),
        ] {
            let request = qwen_moe_llama_id_request(
                quants[0], quants[1], quants[2], quants[3], quants[4], quants[5],
            );
            let (sparse, shared) =
                qwen_moe_llama_id_quant_sets(&request).expect("supported product tuple");
            assert_eq!(sparse.gate, QwenMoeLlamaIdQuant::Q4K);
            assert_eq!(sparse.up, QwenMoeLlamaIdQuant::Q4K);
            assert_eq!(sparse.down, expected_sparse_down);
            assert_eq!(shared, expected_shared);
        }

        for quants in [
            [
                GGMLType::Q8_0,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q8_0,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q8_0,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q8_0,
                GGMLType::Q4_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q5_K,
            ],
            [
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
                GGMLType::Q4_K,
            ],
        ] {
            let request = qwen_moe_llama_id_request(
                quants[0], quants[1], quants[2], quants[3], quants[4], quants[5],
            );
            assert!(qwen_moe_llama_id_quant_sets(&request).is_none());
            assert!(metal_qwen_moe_llama_id_prefill(request)
                .expect("unsupported quant tuple is a fallback")
                .is_none());
        }
    }

    #[test]
    fn qwen_moe_llama_id_success_forwards_backend_output() {
        const HIDDEN_DIM: usize = 256;
        const FFN_DIM: usize = 256;
        const Q4_K_BLOCK_BYTES: usize = 144;
        const Q5_K_BLOCK_BYTES: usize = 176;
        const Q6_K_BLOCK_BYTES: usize = 210;
        const Q8_0_BLOCK_BYTES: usize = 34;
        let _guard = env_lock().lock().expect("env lock");
        let env_key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let previous = std::env::var(env_key).ok();
        std::env::set_var(env_key, "1");

        let matrix_bytes = |quant| match quant {
            GGMLType::Q4_K => FFN_DIM * (HIDDEN_DIM / 256) * Q4_K_BLOCK_BYTES,
            GGMLType::Q5_K => FFN_DIM * (HIDDEN_DIM / 256) * Q5_K_BLOCK_BYTES,
            GGMLType::Q6_K => FFN_DIM * (HIDDEN_DIM / 256) * Q6_K_BLOCK_BYTES,
            GGMLType::Q8_0 => FFN_DIM * (HIDDEN_DIM / 32) * Q8_0_BLOCK_BYTES,
            other => panic!("unsupported test quant {other:?}"),
        };
        let q4_matrix_bytes = matrix_bytes(GGMLType::Q4_K);
        let gate_all = vec![0; q4_matrix_bytes];
        let up_all = vec![0; q4_matrix_bytes];
        let selected_experts = [0];
        let route_weights = [1.0];
        let shared_route_weights = [1.0];
        let norm_all = vec![0.0; HIDDEN_DIM];

        for (down_quant, shared_gate_quant, shared_up_quant, shared_down_quant) in [
            (
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
            ),
            (
                GGMLType::Q6_K,
                GGMLType::Q4_K,
                GGMLType::Q4_K,
                GGMLType::Q6_K,
            ),
            (
                GGMLType::Q5_K,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
            ),
            (
                GGMLType::Q6_K,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
                GGMLType::Q8_0,
            ),
        ] {
            let down_expert_bytes = matrix_bytes(down_quant);
            let down_all = vec![0; down_expert_bytes];
            let shared_gate = vec![0; matrix_bytes(shared_gate_quant)];
            let shared_up = vec![0; matrix_bytes(shared_up_quant)];
            let shared_down = vec![0; matrix_bytes(shared_down_quant)];
            let output = metal_qwen_moe_llama_id_prefill(MetalQwenMoeLlamaIdPrefillRequest {
                gate_all: &gate_all,
                up_all: &up_all,
                down_all: &down_all,
                gate_expert_bytes: q4_matrix_bytes,
                up_expert_bytes: q4_matrix_bytes,
                down_expert_bytes,
                selected_experts: &selected_experts,
                route_weights: &route_weights,
                shared_gate: &shared_gate,
                shared_up: &shared_up,
                shared_down: &shared_down,
                shared_route_weights: &shared_route_weights,
                gate_quant: GGMLType::Q4_K,
                up_quant: GGMLType::Q4_K,
                down_quant,
                shared_gate_quant,
                shared_up_quant,
                shared_down_quant,
                norm_all: &norm_all,
                seq_len: 1,
                hidden_dim: HIDDEN_DIM,
                ffn_dim: FFN_DIM,
                layer_idx: 0,
            })
            .expect("supported product tuple must not error")
            .expect("TensorOps-capable Metal device must run the product seam");
            assert_eq!(output.values, vec![0.0; HIDDEN_DIM]);
        }
        match previous {
            Some(value) => std::env::set_var(env_key, value),
            None => std::env::remove_var(env_key),
        }
    }

    #[test]
    fn qwen_moe_prefill_accum_defaults_to_long_prefill_only_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_METAL_QWEN35_MOE_PREFILL_ACCUM";
        let previous = std::env::var(key).ok();
        std::env::remove_var(key);

        assert!(!metal_qwen_moe_prefill_accum_enabled(5));
        assert!(!metal_qwen_moe_prefill_accum_enabled(1023));
        assert!(metal_qwen_moe_prefill_accum_enabled(1024));

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !metal_qwen_moe_prefill_accum_enabled(4096),
                "{value} should opt out"
            );
        }

        std::env::set_var(key, "1");
        assert!(metal_qwen_moe_prefill_accum_enabled(5));

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn qwen_moe_prefill_shared_accum_defaults_on_for_long_prefill_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let accum_key = "RNB_METAL_QWEN35_MOE_PREFILL_ACCUM";
        let shared_key = "RNB_METAL_QWEN35_MOE_PREFILL_SHARED_ACCUM";
        let previous_accum = std::env::var(accum_key).ok();
        let previous_shared = std::env::var(shared_key).ok();
        std::env::remove_var(accum_key);
        std::env::remove_var(shared_key);

        assert!(!metal_qwen_moe_prefill_shared_accum_enabled(1023));
        assert!(metal_qwen_moe_prefill_shared_accum_enabled(1024));

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(shared_key, value);
            assert!(
                !metal_qwen_moe_prefill_shared_accum_enabled(4096),
                "{value} should opt out"
            );
        }

        std::env::set_var(accum_key, "0");
        std::env::set_var(shared_key, "1");
        assert!(
            !metal_qwen_moe_prefill_shared_accum_enabled(4096),
            "base accum opt-out must disable shared accum too"
        );

        match previous_accum {
            Some(value) => std::env::set_var(accum_key, value),
            None => std::env::remove_var(accum_key),
        }
        match previous_shared {
            Some(value) => std::env::set_var(shared_key, value),
            None => std::env::remove_var(shared_key),
        }
    }

    #[test]
    fn qwen_moe_prefill_mulmmid_v3_defaults_off_with_truthy_opt_in() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V3";
        let llama_id_key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let llama_id_previous = std::env::var(llama_id_key).ok();
        let previous = std::env::var(key).ok();

        std::env::remove_var(llama_id_key);
        std::env::remove_var(key);
        assert!(!metal_qwen_moe_prefill_mulmmid_v3_requested());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !metal_qwen_moe_prefill_mulmmid_v3_requested(),
                "{value} should opt out"
            );
        }

        std::env::set_var(key, "1");
        assert!(metal_qwen_moe_prefill_mulmmid_v3_requested());

        for value in ["true", "on", "yes"] {
            std::env::set_var(key, value);
            assert!(metal_qwen_moe_prefill_mulmmid_v3_requested(), "{value}");
        }

        std::env::set_var(key, "typo");
        assert!(!metal_qwen_moe_prefill_mulmmid_v3_requested());

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        match llama_id_previous {
            Some(value) => std::env::set_var(llama_id_key, value),
            None => std::env::remove_var(llama_id_key),
        }
    }

    #[test]
    fn qwen_moe_llama_id_defaults_on_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let previous = std::env::var(key).ok();

        std::env::remove_var(key);
        assert!(metal_qwen_moe_llama_id_requested());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !metal_qwen_moe_llama_id_requested(),
                "{value} should opt out"
            );
        }

        for value in ["1", "true", "on", "yes", "typo"] {
            std::env::set_var(key, value);
            assert!(
                metal_qwen_moe_llama_id_requested(),
                "{value} should keep the default enabled"
            );
        }

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn qwen_moe_llama_id_takes_priority_over_mulmmid_envs() {
        let _guard = env_lock().lock().expect("env lock");
        let llama_id_key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let v3_key = "RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V3";
        let v4_key = "RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4";
        let llama_id_previous = std::env::var(llama_id_key).ok();
        let v3_previous = std::env::var(v3_key).ok();
        let v4_previous = std::env::var(v4_key).ok();

        std::env::set_var(llama_id_key, "1");
        std::env::set_var(v3_key, "1");
        std::env::set_var(v4_key, "true");

        assert!(metal_qwen_moe_llama_id_requested());
        assert!(!metal_qwen_moe_prefill_mulmmid_v3_requested());
        assert!(!metal_qwen_moe_prefill_mulmmid_v4_requested());

        match llama_id_previous {
            Some(value) => std::env::set_var(llama_id_key, value),
            None => std::env::remove_var(llama_id_key),
        }
        match v3_previous {
            Some(value) => std::env::set_var(v3_key, value),
            None => std::env::remove_var(v3_key),
        }
        match v4_previous {
            Some(value) => std::env::set_var(v4_key, value),
            None => std::env::remove_var(v4_key),
        }
    }

    #[test]
    fn qwen_moe_prefill_mulmmid_v4_defaults_off_with_truthy_opt_in() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_METAL_QWEN35_MOE_PREFILL_MULMMID_V4";
        let llama_id_key = "RNB_METAL_QWEN35_MOE_LLAMA_ID";
        let llama_id_previous = std::env::var(llama_id_key).ok();
        let previous = std::env::var(key).ok();

        std::env::remove_var(llama_id_key);
        std::env::remove_var(key);
        assert!(!metal_qwen_moe_prefill_mulmmid_v4_requested());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !metal_qwen_moe_prefill_mulmmid_v4_requested(),
                "{value} should opt out"
            );
        }

        for value in ["1", "true", "on", "yes"] {
            std::env::set_var(key, value);
            assert!(
                metal_qwen_moe_prefill_mulmmid_v4_requested(),
                "{value} should opt in"
            );
        }

        std::env::set_var(key, "typo");
        assert!(!metal_qwen_moe_prefill_mulmmid_v4_requested());

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        match llama_id_previous {
            Some(value) => std::env::set_var(llama_id_key, value),
            None => std::env::remove_var(llama_id_key),
        }
    }

    #[test]
    fn qwen_moe_expert_batch_defaults_to_long_prefill_only_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_METAL_QWEN35_MOE_EXPERT_BATCH";
        let previous = std::env::var(key).ok();
        std::env::remove_var(key);

        assert!(!metal_qwen_moe_expert_batch_enabled(5));
        assert!(!metal_qwen_moe_expert_batch_enabled(15));
        assert!(metal_qwen_moe_expert_batch_enabled(16));

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !metal_qwen_moe_expert_batch_enabled(4096),
                "{value} should opt out"
            );
        }

        std::env::set_var(key, "1");
        assert!(metal_qwen_moe_expert_batch_enabled(5));

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn qwen_moe_prefill_router_defaults_on_with_falsey_opt_out() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "RNB_METAL_QWEN35_PREFILL_ROUTER";
        let previous = std::env::var(key).ok();
        std::env::remove_var(key);

        assert!(metal_qwen_moe_prefill_router_requested());

        for value in ["0", "false", "off", "no"] {
            std::env::set_var(key, value);
            assert!(
                !metal_qwen_moe_prefill_router_requested(),
                "{value} should opt out"
            );
        }

        std::env::set_var(key, "1");
        assert!(metal_qwen_moe_prefill_router_requested());

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

#[cfg(test)]
mod metal_decode_parity_policy_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_env() {
        for key in [
            "RNB_METAL_DECODE_PARITY_CHAIN",
            "RNB_METAL_DECODE_PARITY_TIME",
            "RNB_METAL_DECODE_CHAIN",
            "RNB_METAL_GDN_LAYER",
            "RNB_METAL_ATTN_LAYER",
            "RNB_METAL_QWEN35_MOE_DECODE_CHAIN",
            "RNB_METAL_KV_INT8",
            "RNB_METAL_CHAIN_GPU_TIME",
            "RNB_METAL_DELTA_RESIDENT",
        ] {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn metal_decode_policy_keeps_parity_opt_in_and_defaults_qwen_moe_on() {
        let _guard = env_lock().lock().expect("env lock");
        clear_env();
        let policy = metal_decode_parity_policy();
        assert!(!policy.parity_requested);
        assert!(!policy.timing_enabled);
        assert!(policy.legacy_decode_chain_enabled);
        assert!(policy.legacy_gdn_layer_enabled);
        assert!(policy.legacy_attn_layer_enabled);
        assert!(policy.qwen_moe_decode_chain_enabled);
        assert!(policy.delta_resident);

        for value in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_DECODE_PARITY_CHAIN", value);
            assert!(
                !metal_decode_parity_policy().parity_requested,
                "{value} should opt out of pm51 parity chain"
            );
        }

        for value in ["0", "false", "off", "no"] {
            std::env::set_var("RNB_METAL_QWEN35_MOE_DECODE_CHAIN", value);
            assert!(
                !metal_decode_parity_policy().qwen_moe_decode_chain_enabled,
                "{value} should opt out of the Qwen MoE decode chain"
            );
        }

        std::env::set_var("RNB_METAL_DECODE_PARITY_CHAIN", "1");
        std::env::set_var("RNB_METAL_DECODE_PARITY_TIME", "1");
        std::env::set_var("RNB_METAL_QWEN35_MOE_DECODE_CHAIN", "1");
        let policy = metal_decode_parity_policy();
        assert!(policy.parity_requested);
        assert!(policy.timing_enabled);
        assert!(policy.qwen_moe_decode_chain_enabled);
        clear_env();
    }

    #[cfg(feature = "metal")]
    fn qwen_moe_base_layer(down_ggml: GGMLType) -> MetalGdnMoeQwenChainLayer<'static> {
        static F32_1: [f32; 1] = [1.0];
        static F32_4: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        static RAW: [u8; 4096] = [0x11; 4096];
        static ROUTER: [f32; 4096] = [0.0; 4096];
        static SCALE: [f32; 256] = [1.0; 256];

        MetalGdnMoeQwenChainLayer {
            layer: 3,
            conv_state: vec![0.0; 4],
            delta_state: vec![0.0; 4],
            attn_norm_weight: &F32_1,
            dt_bias_weight: &F32_1,
            ssm_a_weight: &F32_1,
            conv1d_weight: &F32_4,
            ssm_norm_weight: &F32_1,
            ffn_norm_weight: &F32_1,
            qkv_ggml: GGMLType::Q4_K,
            qkv_raw: &RAW,
            gate_ggml: GGMLType::Q4_K,
            gate_raw: &RAW,
            alpha_ggml: GGMLType::F32,
            alpha_raw: &RAW,
            beta_ggml: GGMLType::F32,
            beta_raw: &RAW,
            ssm_out_ggml: GGMLType::Q4_K,
            ssm_out_raw: &RAW,
            router_w: &ROUTER,
            gate_exps_ggml: GGMLType::Q4_K,
            gate_exps_raw: &RAW,
            gate_expert_bytes: 256,
            up_exps_ggml: GGMLType::Q4_K,
            up_exps_raw: &RAW,
            up_expert_bytes: 256,
            down_exps_ggml: down_ggml,
            down_exps_raw: &RAW,
            down_expert_bytes: 256,
            shared_input_scale: &SCALE,
            shared_gate_ggml: GGMLType::Q4_K,
            shared_gate_raw: &RAW[..256],
            shared_up_ggml: GGMLType::Q4_K,
            shared_up_raw: &RAW[..256],
            shared_down_ggml: down_ggml,
            shared_down_raw: &RAW[..256],
            hidden_dim: 256,
            conv_channels: 1,
            conv_kernel: 4,
            z_dim: 1,
            num_v_heads: 1,
            num_k_heads: 1,
            head_k_dim: 1,
            head_v_dim: 1,
            n_ff: 256,
            n_expert: 16,
            n_expert_used: 2,
            eps: 1e-6,
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qwen_moe_decode_chain_conversion_accepts_actual_down_tuples() {
        for (down, expected_code) in [
            (GGMLType::Q4_K, 0),
            (GGMLType::Q5_K, 1),
            (GGMLType::Q6_K, 2),
        ] {
            let layer = qwen_moe_base_layer(down);
            let spec = gdn_moe_qwen_chain_ref_if_supported(&layer).expect("supported down tuple");
            assert_eq!(spec.down_quant, expected_code);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qwen_moe_decode_chain_conversion_rejects_mixed_down_tuple() {
        let mut layer = qwen_moe_base_layer(GGMLType::Q5_K);
        layer.shared_down_ggml = GGMLType::Q6_K;
        let err = match gdn_moe_qwen_chain_ref_if_supported(&layer) {
            Ok(_) => panic!("mixed down quant must be rejected"),
            Err(err) => err,
        };
        assert_eq!(err, "unsupported qwen moe shared down quant");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qwen_moe_decode_chain_conversion_rejects_too_many_active_experts() {
        let mut layer = qwen_moe_base_layer(GGMLType::Q6_K);
        layer.n_expert = 64;
        layer.n_expert_used = 32;
        let err = match gdn_moe_qwen_chain_ref_if_supported(&layer) {
            Ok(_) => panic!("n_expert_used above route slot limit must be rejected"),
            Err(err) => err,
        };
        assert_eq!(err, "unsupported qwen moe shape");
    }

    #[test]
    fn metal_decode_parity_policy_owns_legacy_chain_and_kv_int8_env() {
        let _guard = env_lock().lock().expect("env lock");
        clear_env();
        std::env::set_var("RNB_METAL_DECODE_CHAIN", "0");
        std::env::set_var("RNB_METAL_GDN_LAYER", "0");
        std::env::set_var("RNB_METAL_ATTN_LAYER", "0");
        std::env::set_var("RNB_METAL_KV_INT8", "1");
        std::env::set_var("RNB_METAL_CHAIN_GPU_TIME", "1");
        std::env::set_var("RNB_METAL_DELTA_RESIDENT", "0");

        let policy = metal_decode_parity_policy();
        assert!(!policy.legacy_decode_chain_enabled);
        assert!(!policy.legacy_gdn_layer_enabled);
        assert!(!policy.legacy_attn_layer_enabled);
        assert!(policy.kv_int8_requested);
        assert!(policy.collect_gpu_time);
        assert!(!policy.delta_resident);
        assert!(policy
            .kv_int8_requires_carrier_error(false, false)
            .is_some());
        clear_env();
    }

    #[test]
    fn metal_decode_parity_counters_split_hit_fallback_and_backend_error() {
        let _guard = env_lock().lock().expect("env lock");
        clear_env();
        metal_decode_parity_counters_reset();

        metal_decode_parity_record_expected_token();
        metal_decode_parity_record_chain_report(MetalDecodeChainReport {
            did_run: true,
            pm51_extension_hit: true,
            legacy_chain_hit: false,
            qwen_moe_layers: 1,
            fallback_reason: None,
            cmd_buffers: 2,
            wait_us: 300,
            upload_bytes: 16,
            readback_bytes: 4,
            state_syncs: 1,
            output_argmax: MetalOutputArgmaxReport {
                attempted: true,
                did_run: true,
                token_id: Some(7),
                cmd_buffers: 1,
                wait_us: 50,
                readback_bytes: 4,
                fallback_reason: None,
            },
        });
        metal_decode_parity_record_expected_token();
        metal_decode_parity_record_chain_report(MetalDecodeChainReport::fallback(
            "unsupported quant",
            true,
        ));

        let counters = metal_decode_parity_counters_snapshot();
        assert_eq!(counters.expected_decode_tokens, 2);
        assert_eq!(counters.chain_hit_tokens, 1);
        assert_eq!(counters.fallback_tokens, 1);
        assert_eq!(counters.backend_err, 1);
        assert_eq!(counters.cmd_buffers, 3);
        assert_eq!(counters.wait_us, 350);
        assert_eq!(counters.upload_bytes, 16);
        assert_eq!(counters.readback_bytes, 8);
        assert_eq!(counters.argmax_device_hits, 1);
        assert_eq!(counters.state_syncs, 1);
        clear_env();
    }

    #[test]
    fn metal_output_argmax_is_opt_in_and_quant_gated() {
        let _guard = env_lock().lock().expect("env lock");
        clear_env();
        std::env::remove_var("RNB_METAL_OUTPUT_ARGMAX");

        let input = vec![0.0f32; 256];
        assert_eq!(
            metal_output_argmax_token_if_supported(GGMLType::Q6_K, &[], 1, 256, &input),
            None
        );

        std::env::set_var("RNB_METAL_OUTPUT_ARGMAX", "1");
        assert_eq!(
            metal_output_argmax_token_if_supported(GGMLType::F32, &[], 1, 256, &input),
            None
        );

        std::env::remove_var("RNB_METAL_OUTPUT_ARGMAX");
        clear_env();
    }
}

#[cfg(all(test, target_os = "macos"))]
mod qwen_prefill_chain_conversion_tests {
    use super::*;

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_shared_quant_tuples() {
        let hidden_dim = 256;
        let ffn_dim = 256;
        let sparse_bytes = 256 * 144;
        let shared_q8_bytes = 256 * (256 / 32) * 34;
        let shared_q4_bytes = 256 * 144;
        let shared_q6_down_bytes = 256 * 210;
        let norm = vec![1.0; hidden_dim];
        let router = vec![0.0; hidden_dim];
        let shared_input_scale = vec![0.0; hidden_dim];
        let sparse_gate = vec![0; sparse_bytes];
        let sparse_up = vec![0; sparse_bytes];
        let sparse_down = vec![0; sparse_bytes];
        let shared_q8 = vec![0; shared_q8_bytes];
        let shared_q4_gate = vec![0; shared_q4_bytes];
        let shared_q4_up = vec![0; shared_q4_bytes];
        let shared_q6_down = vec![0; shared_q6_down_bytes];

        let q8 = MetalQwenMoePrefillWeights {
            ffn_norm_w: &norm,
            router_w: &router,
            gate_all: &sparse_gate,
            up_all: &sparse_up,
            down_all: &sparse_down,
            gate_expert_bytes: sparse_bytes,
            up_expert_bytes: sparse_bytes,
            down_expert_bytes: sparse_bytes,
            shared_input_scale: &shared_input_scale,
            shared_gate: &shared_q8,
            shared_up: &shared_q8,
            shared_down: &shared_q8,
            sparse_quant: MetalGgmlQuantSet {
                gate: GGMLType::Q4_K,
                up: GGMLType::Q4_K,
                down: GGMLType::Q4_K,
            },
            shared_quant: MetalGgmlQuantSet {
                gate: GGMLType::Q8_0,
                up: GGMLType::Q8_0,
                down: GGMLType::Q8_0,
            },
            route: Qwen35RoutePolicy {
                algorithm: Qwen35RouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1,
                n_expert: 1,
                n_expert_used: 1,
            },
            hidden_dim,
            ffn_dim,
            norm_eps: 3.0e-5,
        };
        let converted = qwen_prefill_chain_moe_spec(&q8, 1, hidden_dim).expect("exact Q8 tuple");
        assert!(matches!(
            converted.shared_quant.gate,
            rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero
        ));
        assert!(matches!(
            converted.shared_quant.up,
            rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero
        ));
        assert!(matches!(
            converted.shared_quant.down,
            rnb_backend_metal::QwenMoeLlamaIdQuant::Q8Zero
        ));
        assert_eq!(converted.norm_eps.to_bits(), 3.0e-5f32.to_bits());

        let q4_q6 = MetalQwenMoePrefillWeights {
            shared_gate: &shared_q4_gate,
            shared_up: &shared_q4_up,
            shared_down: &shared_q6_down,
            shared_quant: MetalGgmlQuantSet {
                gate: GGMLType::Q4_K,
                up: GGMLType::Q4_K,
                down: GGMLType::Q6_K,
            },
            ..q8
        };
        let converted = qwen_prefill_chain_moe_spec(&q4_q6, 1, hidden_dim).expect("Q4/Q4/Q6 tuple");
        assert!(matches!(
            converted.shared_quant.down,
            rnb_backend_metal::QwenMoeLlamaIdQuant::Q6K
        ));

        let mixed = MetalQwenMoePrefillWeights {
            shared_gate: &shared_q8,
            shared_up: &shared_q8,
            shared_down: &shared_q4_gate,
            shared_quant: MetalGgmlQuantSet {
                gate: GGMLType::Q8_0,
                up: GGMLType::Q8_0,
                down: GGMLType::Q4_K,
            },
            ..q4_q6
        };
        assert!(qwen_prefill_chain_moe_spec(&mixed, 1, hidden_dim).is_none());
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_rejects_rank_nine() {
        let hidden_dim = 256;
        let ffn_dim = 256;
        let n_expert = 9;
        let sparse_expert_bytes = 256 * 144;
        let shared_q8_bytes = 256 * (256 / 32) * 34;
        let norm = vec![1.0; hidden_dim];
        let router = vec![0.0; n_expert * hidden_dim];
        let shared_input_scale = vec![0.0; hidden_dim];
        let sparse_gate = vec![0; n_expert * sparse_expert_bytes];
        let sparse_up = vec![0; n_expert * sparse_expert_bytes];
        let sparse_down = vec![0; n_expert * sparse_expert_bytes];
        let shared_q8 = vec![0; shared_q8_bytes];
        let mut rank = MetalQwenMoePrefillWeights {
            ffn_norm_w: &norm,
            router_w: &router,
            gate_all: &sparse_gate,
            up_all: &sparse_up,
            down_all: &sparse_down,
            gate_expert_bytes: sparse_expert_bytes,
            up_expert_bytes: sparse_expert_bytes,
            down_expert_bytes: sparse_expert_bytes,
            shared_input_scale: &shared_input_scale,
            shared_gate: &shared_q8,
            shared_up: &shared_q8,
            shared_down: &shared_q8,
            sparse_quant: MetalGgmlQuantSet {
                gate: GGMLType::Q4_K,
                up: GGMLType::Q4_K,
                down: GGMLType::Q4_K,
            },
            shared_quant: MetalGgmlQuantSet {
                gate: GGMLType::Q8_0,
                up: GGMLType::Q8_0,
                down: GGMLType::Q8_0,
            },
            route: Qwen35RoutePolicy {
                algorithm: Qwen35RouteAlgorithm::SelectedSoftmaxTopKLowerExpertTieV1,
                n_expert,
                n_expert_used: 8,
            },
            hidden_dim,
            ffn_dim,
            norm_eps: 1.0e-6,
        };
        assert!(qwen_prefill_chain_moe_spec(&rank, 1, hidden_dim).is_some());
        rank.route.n_expert_used = 9;
        assert!(qwen_prefill_chain_moe_spec(&rank, 1, hidden_dim).is_none());
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_contiguous_positions() {
        assert!(qwen_prefill_chain_layer_is_contiguous(None, 7));
        assert!(qwen_prefill_chain_layer_is_contiguous(Some(7), 8));
        assert!(qwen_prefill_chain_layer_is_contiguous(Some(39), 40));
        assert!(!qwen_prefill_chain_layer_is_contiguous(Some(7), 7));
        assert!(!qwen_prefill_chain_layer_is_contiguous(Some(7), 9));
        assert!(!qwen_prefill_chain_layer_is_contiguous(Some(7), 6));
        assert!(!qwen_prefill_chain_layer_is_contiguous(Some(usize::MAX), 0));
    }
    #[test]
    fn qwen_moe_llama_prefill_layer_chain_preserves_non_default_norm_eps() {
        assert!(qwen_prefill_chain_norm_eps_supported(3.0e-5));
        assert!(qwen_prefill_chain_norm_eps_supported(f32::MIN_POSITIVE));
        assert!(!qwen_prefill_chain_norm_eps_supported(0.0));
        assert!(!qwen_prefill_chain_norm_eps_supported(-1.0e-6));
        assert!(!qwen_prefill_chain_norm_eps_supported(f32::NAN));
        assert!(!qwen_prefill_chain_norm_eps_supported(f32::INFINITY));
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_rejects_u32_conv_data_overflow() {
        const SEQ_LEN: usize = 65_536;
        const CONV_CHANNELS: usize = 65_536;
        assert_eq!(
            SEQ_LEN.checked_mul(CONV_CHANNELS),
            Some(u32::MAX as usize + 1)
        );
        assert!(!qwen_prefill_chain_gdn_elements_fit_u32(&[
            SEQ_LEN * CONV_CHANNELS
        ]));
        assert!(qwen_prefill_chain_gdn_elements_fit_u32(&[
            (SEQ_LEN - 1) * CONV_CHANNELS
        ]));
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_rejects_u32_delta_state_overflow() {
        const D_INNER: usize = 16_777_216;
        const D_STATE: usize = 256;
        assert_eq!(D_INNER.checked_mul(D_STATE), Some(u32::MAX as usize + 1));
        assert!(!qwen_prefill_chain_gdn_elements_fit_u32(&[
            D_INNER * D_STATE
        ]));
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_rejects_u32_conv_input_overflow() {
        const SEQ_LEN: usize = 1;
        const CONV_KERNEL: usize = 65_536;
        const CONV_CHANNELS: usize = 65_536;
        let conv_rows = SEQ_LEN + CONV_KERNEL - 1;
        assert_eq!(
            conv_rows.checked_mul(CONV_CHANNELS),
            Some(u32::MAX as usize + 1)
        );
        assert!(!qwen_prefill_chain_gdn_elements_fit_u32(&[
            conv_rows * CONV_CHANNELS
        ]));
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_rejects_u32_final_qk_overflow() {
        const SEQ_LEN: usize = 65_536;
        const DT_RANK: usize = 256;
        const D_STATE: usize = 256;
        assert_eq!(
            SEQ_LEN.checked_mul(DT_RANK * D_STATE),
            Some(u32::MAX as usize + 1)
        );
        assert!(!qwen_prefill_chain_gdn_elements_fit_u32(&[SEQ_LEN
            * DT_RANK
            * D_STATE]));
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_attention_rejects_odd_n_rot() {
        let hidden_dim = 256;
        let q_dim = 256;
        let kv_dim = 256;
        let attn_norm = vec![1.0; hidden_dim];
        let q_norm = vec![1.0; 256];
        let k_norm = vec![1.0; 256];
        let q_raw = vec![0; q_dim * 2 * 144];
        let k_raw = vec![0; kv_dim * 144];
        let v_raw = vec![0; kv_dim * 144];
        let o_raw = vec![0; hidden_dim * 144];
        let mut spec = MetalPrefillAtnOTailSpec {
            core: MetalPrefillAtnCoreSpec {
                attn_norm_w: &attn_norm,
                q_norm_w: &q_norm,
                k_norm_w: &k_norm,
                q_weight: MetalQuantWeightRef {
                    ggml_type: GGMLType::Q4_K,
                    raw: &q_raw,
                    rows: q_dim * 2,
                    cols: hidden_dim,
                },
                k_weight: MetalQuantWeightRef {
                    ggml_type: GGMLType::Q4_K,
                    raw: &k_raw,
                    rows: kv_dim,
                    cols: hidden_dim,
                },
                v_weight: MetalQuantWeightRef {
                    ggml_type: GGMLType::Q4_K,
                    raw: &v_raw,
                    rows: kv_dim,
                    cols: hidden_dim,
                },
                seq_len: 1,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 256,
                hidden_dim,
                q_dim,
                kv_dim,
                n_rot: 64,
                rope_theta: 10_000_000.0,
                scale: 0.0625,
                norm_eps: 1.0e-6,
                pos_start: 0,
            },
            o_weight: MetalQuantWeightRef {
                ggml_type: GGMLType::Q4_K,
                raw: &o_raw,
                rows: hidden_dim,
                cols: q_dim,
            },
        };
        assert!(qwen_prefill_chain_attention_spec(&spec).is_some());
        spec.core.n_rot = 63;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_none());
        spec.core.n_rot = 64;
        spec.core.norm_eps = f32::NAN;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_none());
        spec.core.norm_eps = 0.0;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_none());
        spec.core.norm_eps = 1.0e-6;
        spec.core.rope_theta = f32::NAN;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_none());
        spec.core.rope_theta = 0.0;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_none());
        spec.core.rope_theta = 10_000_000.0;
        spec.core.scale = f32::NAN;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_none());
        spec.core.scale = f32::INFINITY;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_none());
        spec.core.scale = 0.0625;
        assert!(qwen_prefill_chain_attention_spec(&spec).is_some());
    }

    #[test]
    fn qwen_moe_llama_prefill_layer_chain_gdn_rejects_kernel_one() {
        let hidden_dim = 256;
        let d_inner = 256;
        let d_state = 1;
        let n_group = 1;
        let dt_rank = 1;
        let conv_channels = d_inner + 2 * n_group * d_state;
        let attn_norm = vec![1.0; hidden_dim];
        let qkv = vec![0.0; conv_channels * hidden_dim];
        let gate = vec![0.0; d_inner * hidden_dim];
        let alpha = vec![0.0; dt_rank * hidden_dim];
        let beta = vec![0.0; dt_rank * hidden_dim];
        let conv_state = vec![0.0; conv_channels];
        let conv_kernel = vec![0.0; 2 * conv_channels];
        let dt_bias = vec![0.0; dt_rank];
        let ssm_a = vec![0.0; dt_rank];
        let delta_state = vec![0.0; d_inner * d_state];
        let ssm_norm = vec![1.0; d_inner / dt_rank];
        let ssm_out = vec![0.0; hidden_dim * d_inner];
        let post_norm = vec![1.0; hidden_dim];
        let mut spec = MetalQwenPrefillGdnSpec {
            seq_len: 1,
            hidden_dim,
            d_inner,
            d_state,
            n_group,
            dt_rank,
            conv_kernel_size: 2,
            attn_norm_w: &attn_norm,
            qkv_weight: MetalGdnWeightRef::F32 {
                values: &qkv,
                rows: conv_channels,
                cols: hidden_dim,
            },
            gate_weight: MetalGdnWeightRef::F32 {
                values: &gate,
                rows: d_inner,
                cols: hidden_dim,
            },
            alpha_weight: MetalGdnWeightRef::F32 {
                values: &alpha,
                rows: dt_rank,
                cols: hidden_dim,
            },
            beta_weight: MetalGdnWeightRef::F32 {
                values: &beta,
                rows: dt_rank,
                cols: hidden_dim,
            },
            conv_state: &conv_state,
            conv_kernel: &conv_kernel,
            dt_bias: &dt_bias,
            ssm_a: &ssm_a,
            delta_state: &delta_state,
            ssm_norm: &ssm_norm,
            ssm_out_weight: MetalGdnWeightRef::F32 {
                values: &ssm_out,
                rows: hidden_dim,
                cols: d_inner,
            },
            post_attn_norm_w: &post_norm,
            norm_eps: 1.0e-6,
        };
        assert!(qwen_prefill_chain_gdn_spec(&spec).is_some());
        spec.conv_kernel_size = 1;
        assert!(qwen_prefill_chain_gdn_spec(&spec).is_none());
    }
}
