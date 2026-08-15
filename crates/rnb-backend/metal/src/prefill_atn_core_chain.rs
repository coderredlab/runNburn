//! pm50 M1: Qwen dense gated attention prefill core carrier.
//!
//! q/k/v projection, q/gate split, qk_norm+RoPE, flash attention, gate apply를
//! 단일 command buffer로 묶고 기존 `PrefillFusedAttention` seam과 같은
//! `(attn_out, k_f16, v_f16)`을 host로 돌려준다.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder,
};

use crate::compute::{
    self, encode_cast_f32_to_f16, encode_flash_attn_prefill, encode_prefill_gate_apply,
    encode_prefill_rope_qk_norm, encode_prefill_split_q_gate, encode_rms_norm_batch,
    encode_silu_mul_to_f16, MetalContext,
};
use crate::ffn_chain::{
    empty_f16_buf, empty_f16_buf_with_zeroed_tail, empty_f32_buf, f32_buf, private_f16_buf,
    private_f32_buf, readback, shared_f32_buf, shared_u32_buf, u32_buf, QwenMoeLlamaIdStage,
    QwenMoeLlamaIdStageSampler,
};
use crate::{PrefillAtnOTailBackendSpecRef, TensoropsQuant};

pub(crate) struct PrefillAtnCoreCarrier {
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub n_rot: usize,
    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_full_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_gated_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    seq_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_len_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nh_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nkv_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hd_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nrot_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    theta_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    pos_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    rope_cos_sin_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    scale_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_cols_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_n_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct PrefillAtnFullLayerCarrier {
    pub core: PrefillAtnCoreCarrier,
    pub ffn_dim: usize,
    o_in_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    o_proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_act_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}
pub(crate) struct DflashFullLayerCarrier {
    layer: PrefillAtnFullLayerCarrier,
    attention: crate::dflash_attention::DflashAttentionCarrier,
}
pub(crate) struct DflashOutputTop1Carrier {
    pub batch: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    output_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normalized_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    normalized_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    token_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    probability_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    normalized_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    vocab_size_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}
pub(crate) struct DflashCacheSeedCarrier {
    pub capacity: usize,
    pub layer_count: usize,
    pub feature_dim: usize,
    pub hidden_dim: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    features_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    features_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    encoder_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    projected_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    fused_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    fused_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_norm_w_dev: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    kv_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_f16_dev: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    v_f16_dev: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    rope_cos_sin_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    feature_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    token_count_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    feature_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    num_kv_heads_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    head_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct MusePrefillOTailFfnCarrier {
    pub seq_len: usize,
    pub q_dim: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    attention_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attention_gated_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_out_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    o_in_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    o_proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_attn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_ffn_norm_w_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_normed_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_up_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_act_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_down_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    seq_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ffn_elems_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    norm_eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    post_norm_eps_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct MusePrefillFullLayerCarrier {
    pub seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    core: PrefillAtnCoreCarrier,
    tail: MusePrefillOTailFfnCarrier,
    attention_gate_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) struct MusePrefillLayerRangeState {
    hidden_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_elements: usize,
    feature_dev: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    feature_elements: usize,
    hidden_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    feature_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    feature_total_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    feature_base_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl MusePrefillLayerRangeState {
    pub(crate) fn new(
        ctx: &MetalContext,
        hidden: &[f32],
        hidden_dim: usize,
        feature_count: usize,
    ) -> Self {
        let hidden_dev = empty_f32_buf(ctx, hidden.len());
        copy_f32(hidden, &hidden_dev);
        let feature_dim = hidden_dim
            .checked_mul(feature_count)
            .expect("Muse DFlash feature width overflow");
        let feature_elements = hidden
            .len()
            .checked_mul(feature_count)
            .expect("Muse DFlash feature buffer overflow");
        Self {
            hidden_dev,
            hidden_elements: hidden.len(),
            feature_dev: (feature_count != 0).then(|| empty_f32_buf(ctx, feature_elements)),
            feature_elements,
            hidden_dim_buf: u32_buf(ctx, hidden_dim as u32),
            feature_dim_buf: u32_buf(ctx, feature_dim as u32),
            feature_total_buf: u32_buf(ctx, hidden.len() as u32),
            feature_base_bufs: (0..feature_count)
                .map(|index| u32_buf(ctx, (index * hidden_dim) as u32))
                .collect(),
        }
    }

    fn encode_feature(
        &self,
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        feature_index: Option<usize>,
    ) {
        let (Some(features), Some(feature_index)) = (&self.feature_dev, feature_index) else {
            return;
        };
        compute::encode_muse_capture_feature(
            ctx,
            encoder,
            &self.hidden_dev,
            features,
            &self.hidden_dim_buf,
            &self.feature_dim_buf,
            &self.feature_base_bufs[feature_index],
            &self.feature_total_buf,
            self.hidden_elements,
        );
        compute::chain_barrier_resources(ctx, encoder, [&*self.hidden_dev, &**features]);
    }

    pub(crate) fn finish(&self) -> Vec<f32> {
        readback(&self.hidden_dev, self.hidden_elements)
    }

    pub(crate) fn finish_features(&self) -> Vec<f32> {
        self.feature_dev
            .as_ref()
            .map(|features| readback(features, self.feature_elements))
            .unwrap_or_default()
    }
}

type MusePrefillLayerRangeStageTrace = (
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
);

pub(crate) struct MusePrefillLayerRangePending {
    layer_idx: usize,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    k_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_len: usize,
    completed: bool,
    stage_trace: Option<MusePrefillLayerRangeStageTrace>,
}

impl Drop for MusePrefillLayerRangePending {
    fn drop(&mut self) {
        if !self.completed {
            self.command.waitUntilCompleted();
        }
    }
}
pub(crate) struct PrefillAtnOTailCarrier {
    pub core: PrefillAtnCoreCarrier,
    o_in_f16_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    o_proj_dev: Retained<ProtocolObject<dyn MTLBuffer>>,
    q_dim_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

fn prefill_rope_cos_sin(
    seq_len: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
    pos_start: usize,
) -> Vec<f32> {
    let n_rot = n_rot.min(head_dim);
    if n_rot == 0 {
        return vec![1.0, 0.0];
    }
    let theta_scale = theta.powf(-2.0f32 / n_rot as f32);
    let mut table = Vec::with_capacity(seq_len * n_rot);
    for token in 0..seq_len {
        let mut angle = (pos_start + token) as f32;
        for _ in 0..(n_rot / 2) {
            table.push(angle.cos());
            table.push(angle.sin());
            angle *= theta_scale;
        }
    }
    table
}

fn prefill_imrope_cos_sin(
    positions: &[[u32; 4]],
    n_rot: usize,
    sections: [usize; 4],
    theta: f32,
) -> Vec<f32> {
    if n_rot == 0 {
        return vec![1.0, 0.0];
    }
    let half = n_rot / 2;
    let section_pairs = sections.iter().sum::<usize>();
    debug_assert_eq!(section_pairs, half);
    let theta_scale = theta.powf(-2.0f32 / n_rot as f32);
    let mut table = Vec::with_capacity(positions.len() * n_rot);
    for position in positions {
        let mut frequency = 1.0f32;
        for pair in 0..half {
            let sector = pair % section_pairs;
            let axis = if sector % 3 == 1 && sector < 3 * sections[1] {
                1
            } else if sector % 3 == 2 && sector < 3 * sections[2] {
                2
            } else if sector % 3 == 0 && sector < 3 * sections[0] {
                0
            } else {
                3
            };
            let angle = position[axis] as f32 * frequency;
            table.push(angle.cos());
            table.push(angle.sin());
            frequency *= theta_scale;
        }
    }
    table
}

impl PrefillAtnCoreCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        pos_start: usize,
    ) -> Self {
        let kv_elems = seq_len * kv_dim;
        let padded_kv_elems = seq_len.next_multiple_of(64) * kv_dim;
        Self {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            hidden_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            attn_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            q_norm_w_dev: empty_f32_buf(ctx, head_dim),
            k_norm_w_dev: empty_f32_buf(ctx, head_dim),
            normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            q_full_dev: empty_f32_buf(ctx, seq_len * q_dim * 2),
            q_dev: empty_f32_buf(ctx, seq_len * q_dim),
            gate_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            v_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            q_normed_dev: empty_f32_buf(ctx, seq_len * q_dim),
            k_normed_dev: empty_f32_buf(ctx, seq_len * kv_dim),
            k_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            v_f16_dev: empty_f16_buf_with_zeroed_tail(ctx, kv_elems, padded_kv_elems),
            attn_out_dev: empty_f32_buf(ctx, seq_len * q_dim),
            attn_gated_dev: empty_f32_buf(ctx, seq_len * q_dim),
            seq_buf: u32_buf(ctx, seq_len as u32),
            kv_len_buf: u32_buf(ctx, seq_len as u32),
            nh_buf: u32_buf(ctx, num_heads as u32),
            nkv_buf: u32_buf(ctx, num_kv_heads as u32),
            hd_buf: u32_buf(ctx, head_dim as u32),
            nrot_buf: u32_buf(ctx, n_rot as u32),
            theta_buf: f32_buf(ctx, rope_theta),
            eps_buf: f32_buf(ctx, norm_eps),
            pos_buf: u32_buf(ctx, pos_start as u32),
            rope_cos_sin_dev: shared_f32_buf(
                ctx,
                &prefill_rope_cos_sin(seq_len, head_dim, n_rot, rope_theta, pos_start),
            ),
            scale_buf: f32_buf(ctx, scale),
            hidden_cols_buf: u32_buf(ctx, hidden_dim as u32),
            q_n_buf: u32_buf(ctx, q_dim as u32),
            kv_n_buf: u32_buf(ctx, kv_dim as u32),
            hidden_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
            q_elems_buf: u32_buf(ctx, (seq_len * q_dim) as u32),
            kv_elems_buf: u32_buf(ctx, (seq_len * kv_dim) as u32),
        }
    }

    fn upload(&self, hidden: &[f32], attn_norm_w: &[f32], q_norm_w: &[f32], k_norm_w: &[f32]) {
        assert_eq!(hidden.len(), self.seq_len * self.hidden_dim);
        assert_eq!(attn_norm_w.len(), self.hidden_dim);
        assert_eq!(q_norm_w.len(), self.head_dim);
        assert_eq!(k_norm_w.len(), self.head_dim);
        copy_f32(hidden, &self.hidden_dev);
        copy_f32(attn_norm_w, &self.attn_norm_w_dev);
        copy_f32(q_norm_w, &self.q_norm_w_dev);
        copy_f32(k_norm_w, &self.k_norm_w_dev);
    }

    fn update_rope_cos_sin(
        &self,
        positions: Option<&[[u32; 4]]>,
        sections: [usize; 4],
        theta: f32,
        pos_start: usize,
    ) {
        let table = positions.map_or_else(
            || prefill_rope_cos_sin(self.seq_len, self.head_dim, self.n_rot, theta, pos_start),
            |positions| prefill_imrope_cos_sin(positions, self.n_rot, sections, theta),
        );
        copy_f32(&table, &self.rope_cos_sin_dev);
    }
}

impl PrefillAtnFullLayerCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        pos_start: usize,
    ) -> Self {
        let core = PrefillAtnCoreCarrier::new(
            ctx,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        );
        Self {
            core,
            ffn_dim,
            o_in_f16_dev: empty_f16_buf(ctx, seq_len * q_dim),
            o_proj_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            ffn_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            ffn_normed_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            ffn_normed_f16_dev: empty_f16_buf(ctx, seq_len * hidden_dim),
            ffn_gate_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            ffn_up_dev: empty_f32_buf(ctx, seq_len * ffn_dim),
            ffn_act_f16_dev: empty_f16_buf(ctx, seq_len * ffn_dim),
            ffn_down_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
            ffn_dim_buf: u32_buf(ctx, ffn_dim as u32),
            ffn_elems_buf: u32_buf(ctx, (seq_len * ffn_dim) as u32),
        }
    }

    fn upload(
        &self,
        hidden: &[f32],
        attn_norm_w: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        ffn_norm_w: &[f32],
    ) {
        self.core.upload(hidden, attn_norm_w, q_norm_w, k_norm_w);
        assert_eq!(ffn_norm_w.len(), self.core.hidden_dim);
        copy_f32(ffn_norm_w, &self.ffn_norm_w_dev);
    }

    fn upload_weights(
        &self,
        attn_norm_w: &[f32],
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        ffn_norm_w: &[f32],
    ) {
        assert_eq!(attn_norm_w.len(), self.core.hidden_dim);
        assert_eq!(q_norm_w.len(), self.core.head_dim);
        assert_eq!(k_norm_w.len(), self.core.head_dim);
        assert_eq!(ffn_norm_w.len(), self.core.hidden_dim);
        copy_f32(attn_norm_w, &self.core.attn_norm_w_dev);
        copy_f32(q_norm_w, &self.core.q_norm_w_dev);
        copy_f32(k_norm_w, &self.core.k_norm_w_dev);
        copy_f32(ffn_norm_w, &self.ffn_norm_w_dev);
    }
}
impl DflashFullLayerCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn_dim: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        sliding_window: usize,
    ) -> Self {
        Self {
            layer: PrefillAtnFullLayerCarrier::new(
                ctx,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                hidden_dim,
                q_dim,
                kv_dim,
                ffn_dim,
                head_dim,
                rope_theta,
                scale,
                norm_eps,
                0,
            ),
            attention: crate::dflash_attention::DflashAttentionCarrier::new(
                ctx,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
            ),
        }
    }
}
impl DflashOutputTop1Carrier {
    pub(crate) fn new(
        ctx: &MetalContext,
        batch: usize,
        hidden_dim: usize,
        vocab_size: usize,
        norm_eps: f32,
    ) -> Self {
        Self {
            batch,
            hidden_dim,
            vocab_size,
            output_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            normalized_dev: empty_f32_buf(ctx, batch * hidden_dim),
            normalized_f16_dev: empty_f16_buf(ctx, batch * hidden_dim),
            logits_dev: empty_f32_buf(ctx, batch * vocab_size),
            token_dev: shared_u32_buf(ctx, &vec![0; batch]),
            probability_dev: empty_f32_buf(ctx, batch),
            hidden_dim_buf: u32_buf(ctx, hidden_dim as u32),
            normalized_elems_buf: u32_buf(ctx, (batch * hidden_dim) as u32),
            vocab_size_buf: u32_buf(ctx, vocab_size as u32),
            batch_buf: u32_buf(ctx, batch as u32),
            eps_buf: f32_buf(ctx, norm_eps),
        }
    }

    fn upload_output_norm(&self, output_norm_w: &[f32]) {
        assert_eq!(output_norm_w.len(), self.hidden_dim);
        copy_f32(output_norm_w, &self.output_norm_w_dev);
    }

    fn read_output(&self) -> (Vec<u32>, Vec<f32>) {
        let tokens = unsafe {
            std::slice::from_raw_parts(self.token_dev.contents().as_ptr().cast::<u32>(), self.batch)
                .to_vec()
        };
        let probabilities = unsafe {
            std::slice::from_raw_parts(
                self.probability_dev.contents().as_ptr().cast::<f32>(),
                self.batch,
            )
            .to_vec()
        };
        (tokens, probabilities)
    }
}
impl DflashCacheSeedCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        capacity: usize,
        layer_count: usize,
        feature_dim: usize,
        hidden_dim: usize,
        num_kv_heads: usize,
        head_dim: usize,
        norm_eps: f32,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        Self {
            capacity,
            layer_count,
            feature_dim,
            hidden_dim,
            num_kv_heads,
            head_dim,
            kv_dim,
            features_dev: empty_f32_buf(ctx, capacity * feature_dim),
            features_f16_dev: private_f16_buf(ctx, capacity * feature_dim),
            encoder_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            projected_dev: private_f32_buf(ctx, capacity * hidden_dim),
            fused_dev: private_f32_buf(ctx, capacity * hidden_dim),
            fused_f16_dev: private_f16_buf(ctx, capacity * hidden_dim),
            k_norm_w_dev: (0..layer_count)
                .map(|_| empty_f32_buf(ctx, head_dim))
                .collect(),
            kv_dev: private_f32_buf(ctx, capacity * kv_dim),
            k_normed_dev: private_f32_buf(ctx, capacity * kv_dim),
            k_f16_dev: (0..layer_count)
                .map(|_| empty_f16_buf(ctx, capacity * kv_dim))
                .collect(),
            v_f16_dev: (0..layer_count)
                .map(|_| empty_f16_buf(ctx, capacity * kv_dim))
                .collect(),
            rope_cos_sin_dev: empty_f32_buf(ctx, capacity * head_dim),
            feature_dim_buf: u32_buf(ctx, feature_dim as u32),
            hidden_dim_buf: u32_buf(ctx, hidden_dim as u32),
            kv_dim_buf: u32_buf(ctx, kv_dim as u32),
            token_count_buf: u32_buf(ctx, 0),
            feature_elems_buf: u32_buf(ctx, 0),
            hidden_elems_buf: u32_buf(ctx, 0),
            kv_elems_buf: u32_buf(ctx, 0),
            num_kv_heads_buf: u32_buf(ctx, num_kv_heads as u32),
            head_dim_buf: u32_buf(ctx, head_dim as u32),
            eps_buf: f32_buf(ctx, norm_eps),
        }
    }

    fn upload(
        &self,
        features: &[f32],
        encoder_norm_w: &[f32],
        k_norm_weights: &[&[f32]],
        token_count: usize,
        start_position: usize,
        rope_theta: f32,
    ) {
        debug_assert!(token_count <= self.capacity);
        debug_assert_eq!(features.len(), token_count * self.feature_dim);
        debug_assert_eq!(encoder_norm_w.len(), self.hidden_dim);
        debug_assert_eq!(k_norm_weights.len(), self.layer_count);
        copy_f32(features, &self.features_dev);
        copy_f32(encoder_norm_w, &self.encoder_norm_w_dev);
        for (weight, buffer) in k_norm_weights.iter().zip(&self.k_norm_w_dev) {
            debug_assert_eq!(weight.len(), self.head_dim);
            copy_f32(weight, buffer);
        }
        copy_f32(
            &prefill_rope_cos_sin(
                token_count,
                self.head_dim,
                self.head_dim,
                rope_theta,
                start_position,
            ),
            &self.rope_cos_sin_dev,
        );
        for (buffer, value) in [
            (&self.token_count_buf, token_count),
            (&self.feature_elems_buf, token_count * self.feature_dim),
            (&self.hidden_elems_buf, token_count * self.hidden_dim),
            (&self.kv_elems_buf, token_count * self.kv_dim),
        ] {
            unsafe {
                *(buffer.contents().as_ptr() as *mut u32) = value as u32;
            }
        }
    }
}

impl MusePrefillOTailFfnCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        q_dim: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        norm_eps: f32,
        post_norm_eps: f32,
    ) -> Self {
        Self {
            seq_len,
            q_dim,
            hidden_dim,
            ffn_dim,
            attention_gate_dev: empty_f32_buf(ctx, seq_len * q_dim),
            attention_gated_dev: private_f32_buf(ctx, seq_len * q_dim),
            attn_out_dev: empty_f32_buf(ctx, seq_len * q_dim),
            hidden_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            o_in_f16_dev: private_f16_buf(ctx, seq_len * q_dim),
            o_proj_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            post_attn_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            ffn_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            post_ffn_norm_w_dev: empty_f32_buf(ctx, hidden_dim),
            ffn_normed_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            ffn_normed_f16_dev: private_f16_buf(ctx, seq_len * hidden_dim),
            ffn_gate_dev: private_f32_buf(ctx, seq_len * ffn_dim),
            ffn_up_dev: private_f32_buf(ctx, seq_len * ffn_dim),
            ffn_act_f16_dev: private_f16_buf(ctx, seq_len * ffn_dim),
            ffn_down_dev: private_f32_buf(ctx, seq_len * hidden_dim),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
            hidden_dim_buf: u32_buf(ctx, hidden_dim as u32),
            ffn_dim_buf: u32_buf(ctx, ffn_dim as u32),
            seq_buf: u32_buf(ctx, seq_len as u32),
            q_elems_buf: u32_buf(ctx, (seq_len * q_dim) as u32),
            hidden_elems_buf: u32_buf(ctx, (seq_len * hidden_dim) as u32),
            ffn_elems_buf: u32_buf(ctx, (seq_len * ffn_dim) as u32),
            norm_eps_buf: f32_buf(ctx, norm_eps),
            post_norm_eps_buf: f32_buf(ctx, post_norm_eps),
        }
    }

    fn upload_attention(&self, attn_out: &[f32]) {
        debug_assert_eq!(attn_out.len(), self.seq_len * self.q_dim);
        copy_f32(attn_out, &self.attn_out_dev);
    }

    fn upload_op_weights(&self, attention_gate: Option<&[f32]>, req: MuseOTailFfnOpsRequest<'_>) {
        debug_assert_eq!(req.post_attn_norm_w.len(), self.hidden_dim);
        debug_assert_eq!(req.ffn_norm_w.len(), self.hidden_dim);
        debug_assert_eq!(req.post_ffn_norm_w.len(), self.hidden_dim);
        if let Some(gate) = attention_gate {
            debug_assert_eq!(gate.len(), self.seq_len * self.q_dim);
            copy_f32(gate, &self.attention_gate_dev);
        }
        copy_f32(req.post_attn_norm_w, &self.post_attn_norm_w_dev);
        copy_f32(req.ffn_norm_w, &self.ffn_norm_w_dev);
        copy_f32(req.post_ffn_norm_w, &self.post_ffn_norm_w_dev);
    }

    fn upload_ops(&self, attention_gate: Option<&[f32]>, req: MuseOTailFfnOpsRequest<'_>) {
        debug_assert_eq!(req.hidden.len(), self.seq_len * self.hidden_dim);
        self.upload_op_weights(attention_gate, req);
        copy_f32(req.hidden, &self.hidden_dev);
    }
}

impl MusePrefillFullLayerCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn_dim: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        post_norm_eps: f32,
    ) -> Self {
        let core = PrefillAtnCoreCarrier::new(
            ctx,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            head_dim,
            rope_theta,
            scale,
            norm_eps,
            0,
        );
        let tail = MusePrefillOTailFfnCarrier::new(
            ctx,
            seq_len,
            q_dim,
            hidden_dim,
            ffn_dim,
            norm_eps,
            post_norm_eps,
        );
        Self {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            core,
            tail,
            attention_gate_dev: empty_f32_buf(ctx, seq_len * q_dim),
        }
    }
}

impl PrefillAtnOTailCarrier {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &MetalContext,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        scale: f32,
        norm_eps: f32,
        pos_start: usize,
    ) -> Self {
        let core = PrefillAtnCoreCarrier::new(
            ctx,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_dim,
            kv_dim,
            n_rot,
            rope_theta,
            scale,
            norm_eps,
            pos_start,
        );
        Self {
            core,
            o_in_f16_dev: empty_f16_buf(ctx, seq_len * q_dim),
            o_proj_dev: empty_f32_buf(ctx, seq_len * hidden_dim),
            q_dim_buf: u32_buf(ctx, q_dim as u32),
        }
    }

    fn upload_hidden(&self, hidden: &[f32]) {
        assert_eq!(hidden.len(), self.core.seq_len * self.core.hidden_dim);
        copy_f32(hidden, &self.core.hidden_dev);
    }
}

fn copy_f32(src: &[f32], dst: &ProtocolObject<dyn MTLBuffer>) {
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.contents().as_ptr() as *mut f32, src.len());
    }
}

pub(crate) fn ensure_command_completed(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<(), String> {
    let status = cmd.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    let error = cmd
        .error()
        .map(|err| format!("{err:?}"))
        .unwrap_or_else(|| "no NSError attached".to_string());
    Err(format!(
        "Metal prefill ATN core: command buffer failed status={status:?} error={error}"
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_quant_gemm_v2(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    quant: TensoropsQuant,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    w_off: u32,
    in_f16_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    n_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    m_buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    m: usize,
) {
    match quant {
        TensoropsQuant::Q4K if m <= 16 => compute::encode_gemm_q4k_tensorops_v2_64x8_packed(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q5K if m <= 16 => compute::encode_gemm_q5k_tensorops_v2_64x8_packed(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q6K if m <= 16 => compute::encode_gemm_q6k_tensorops_v2_64x32(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q4K => compute::encode_gemm_q4k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q5K => compute::encode_gemm_q5k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q6K => compute::encode_gemm_q6k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q8_0 => compute::encode_gemm_q8_0_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q2K => compute::encode_gemm_q2k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
        TensoropsQuant::Q3K => compute::encode_gemm_q3k_tensorops_v2(
            ctx, enc, w_buf, w_off, in_f16_buf, out_buf, n_buf, k_buf, m_buf, n, m,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct PrefillAtnCoreDispatchRequest<'a> {
    pub hidden: &'a [f32],
    pub attn_norm_w: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub q_w_off: u32,
    pub q_quant: TensoropsQuant,
    pub k_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub k_w_off: u32,
    pub k_quant: TensoropsQuant,
    pub v_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub v_w_off: u32,
    pub v_quant: TensoropsQuant,
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct PrefillAtnFullLayerDispatchRequest<'a> {
    pub core: PrefillAtnCoreDispatchRequest<'a>,
    pub o_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub o_w_off: u32,
    pub o_quant: TensoropsQuant,
    pub ffn_norm_w: &'a [f32],
    pub ffn_gate_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_gate_w_off: u32,
    pub ffn_gate_quant: TensoropsQuant,
    pub ffn_up_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_up_w_off: u32,
    pub ffn_up_quant: TensoropsQuant,
    pub ffn_down_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_down_w_off: u32,
    pub ffn_down_quant: TensoropsQuant,
}
#[allow(clippy::too_many_arguments)]
pub(crate) struct DflashFullLayerDispatchRequest<'a> {
    pub layer: PrefillAtnFullLayerDispatchRequest<'a>,
    pub prior_k: &'a [u16],
    pub prior_v: &'a [u16],
    pub position: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}
pub(crate) struct DflashOutputTop1DispatchRequest<'a> {
    pub carrier: &'a DflashOutputTop1Carrier,
    pub output_norm_w: &'a [f32],
    pub output_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub output_w_off: u32,
    pub output_quant: TensoropsQuant,
}

pub(crate) struct DflashFullLayerDispatchOutput {
    pub hidden: Vec<f32>,
    pub output_top1: Option<(Vec<u32>, Vec<f32>)>,
}
#[derive(Clone, Copy)]
pub(crate) struct DflashCacheSeedLayerDispatchRequest<'a> {
    pub k_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub k_w_off: u32,
    pub k_quant: TensoropsQuant,
    pub v_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub v_w_off: u32,
    pub v_quant: TensoropsQuant,
    pub k_norm_w: &'a [f32],
}

pub(crate) struct DflashCacheSeedDispatchRequest<'a> {
    pub features: &'a [f32],
    pub encoder_norm_w: &'a [f32],
    pub encoder_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub encoder_w_off: u32,
    pub encoder_quant: TensoropsQuant,
    pub layers: &'a [DflashCacheSeedLayerDispatchRequest<'a>],
    pub token_count: usize,
    pub start_position: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct MuseOTailFfnOpsRequest<'a> {
    pub hidden: &'a [f32],
    pub post_attn_norm_w: &'a [f32],
    pub ffn_norm_w: &'a [f32],
    pub post_ffn_norm_w: &'a [f32],
    pub o_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub o_w_off: u32,
    pub o_quant: TensoropsQuant,
    pub ffn_gate_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_gate_w_off: u32,
    pub ffn_gate_quant: TensoropsQuant,
    pub ffn_up_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_up_w_off: u32,
    pub ffn_up_quant: TensoropsQuant,
    pub ffn_down_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub ffn_down_w_off: u32,
    pub ffn_down_quant: TensoropsQuant,
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct MusePrefillOTailFfnDispatchRequest<'a> {
    pub attn_out: &'a [f32],
    pub ops: MuseOTailFfnOpsRequest<'a>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct MuseTargetAttentionOTailFfnDispatchRequest<'a> {
    pub query: &'a [f32],
    pub key: &'a ProtocolObject<dyn MTLBuffer>,
    pub value: &'a ProtocolObject<dyn MTLBuffer>,
    pub kv_len: usize,
    pub sliding_window: Option<usize>,
    pub scale: f32,
    pub attention_gate: &'a [f32],
    pub ops: MuseOTailFfnOpsRequest<'a>,
}
#[derive(Clone, Copy)]
pub(crate) struct MuseTargetKvBuffers<'a> {
    pub key: &'a ProtocolObject<dyn MTLBuffer>,
    pub value: &'a ProtocolObject<dyn MTLBuffer>,
    pub pos_start: usize,
    pub kv_len: usize,
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct MusePrefillFullLayerDispatchRequest<'a> {
    pub hidden: &'a [f32],
    pub attn_norm_w: &'a [f32],
    pub q_norm_w: &'a [f32],
    pub k_norm_w: &'a [f32],
    pub q_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub q_w_off: u32,
    pub q_quant: TensoropsQuant,
    pub k_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub k_w_off: u32,
    pub k_quant: TensoropsQuant,
    pub v_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub v_w_off: u32,
    pub v_quant: TensoropsQuant,
    pub attention_gate_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    pub attention_gate_w_off: u32,
    pub attention_gate_quant: TensoropsQuant,
    pub apply_rope: bool,
    pub rope_theta: f32,
    pub pos_start: usize,
    pub sliding_window: Option<usize>,
    pub target_kv: Option<MuseTargetKvBuffers<'a>>,
    pub ops: MuseOTailFfnOpsRequest<'a>,
    pub norm_eps: f32,
    pub scale: f32,
}

pub(crate) struct PrefillAtnOTailDispatchRequest<'a> {
    pub hidden: &'a [f32],
    pub spec: PrefillAtnOTailBackendSpecRef<'a>,
}

#[derive(Clone, Copy)]
struct PrefillAtnCoreOpsWeights<'a> {
    q_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    q_w_off: u32,
    q_quant: TensoropsQuant,
    k_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    k_w_off: u32,
    k_quant: TensoropsQuant,
    v_w_buf: &'a ProtocolObject<dyn MTLBuffer>,
    v_w_off: u32,
    v_quant: TensoropsQuant,
}

impl<'a> PrefillAtnCoreOpsWeights<'a> {
    fn from_dispatch(req: &PrefillAtnCoreDispatchRequest<'a>) -> Self {
        Self {
            q_w_buf: req.q_w_buf,
            q_w_off: req.q_w_off,
            q_quant: req.q_quant,
            k_w_buf: req.k_w_buf,
            k_w_off: req.k_w_off,
            k_quant: req.k_quant,
            v_w_buf: req.v_w_buf,
            v_w_off: req.v_w_off,
            v_quant: req.v_quant,
        }
    }
}

#[derive(Clone, Copy)]
enum PrefillAtnNormMode {
    LegacyTree,
    Exact {
        eps: f32,
        n_rot: usize,
        imrope: bool,
    },
}

fn encode_atn_core_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnCoreCarrier,
    weights: PrefillAtnCoreOpsWeights<'_>,
    attn_norm_w: &ProtocolObject<dyn MTLBuffer>,
    q_norm_w: &ProtocolObject<dyn MTLBuffer>,
    k_norm_w: &ProtocolObject<dyn MTLBuffer>,
    norm_mode: PrefillAtnNormMode,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
    mut stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<(), String> {
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.begin(enc, QwenMoeLlamaIdStage::Map);
    }
    match norm_mode {
        PrefillAtnNormMode::LegacyTree => {
            encode_rms_norm_batch(
                ctx,
                enc,
                hidden_in,
                attn_norm_w,
                &carrier.normed_dev,
                &carrier.hidden_cols_buf,
                &carrier.eps_buf,
                carrier.seq_len,
            );
            compute::chain_barrier(ctx, enc);
        }
        PrefillAtnNormMode::Exact { eps, .. } => {
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                hidden_in,
                attn_norm_w,
                &carrier.normed_dev,
                carrier.seq_len,
                carrier.hidden_dim,
                eps,
            )
            .map_err(|error| {
                format!("Metal prefill ATN exact residual RMS norm failed: {error:?}")
            })?;
        }
    }
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.normed_dev,
        &carrier.normed_f16_dev,
        &carrier.hidden_elems_buf,
        carrier.seq_len * carrier.hidden_dim,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Gate);
    }
    encode_quant_gemm_v2(
        ctx,
        enc,
        weights.q_quant,
        weights.q_w_buf,
        weights.q_w_off,
        &carrier.normed_f16_dev,
        &carrier.q_full_dev,
        &carrier.q_n_buf,
        &carrier.hidden_cols_buf,
        &carrier.seq_buf,
        carrier.q_dim * 2,
        carrier.seq_len,
    );
    encode_quant_gemm_v2(
        ctx,
        enc,
        weights.k_quant,
        weights.k_w_buf,
        weights.k_w_off,
        &carrier.normed_f16_dev,
        &carrier.k_dev,
        &carrier.kv_n_buf,
        &carrier.hidden_cols_buf,
        &carrier.seq_buf,
        carrier.kv_dim,
        carrier.seq_len,
    );
    encode_quant_gemm_v2(
        ctx,
        enc,
        weights.v_quant,
        weights.v_w_buf,
        weights.v_w_off,
        &carrier.normed_f16_dev,
        &carrier.v_dev,
        &carrier.kv_n_buf,
        &carrier.hidden_cols_buf,
        &carrier.seq_buf,
        carrier.kv_dim,
        carrier.seq_len,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Up);
    }
    encode_prefill_split_q_gate(
        ctx,
        enc,
        &carrier.q_full_dev,
        &carrier.q_dev,
        &carrier.gate_dev,
        &carrier.seq_buf,
        &carrier.nh_buf,
        &carrier.hd_buf,
        carrier.seq_len * carrier.q_dim,
    );
    compute::chain_barrier(ctx, enc);
    match norm_mode {
        PrefillAtnNormMode::LegacyTree => {
            encode_prefill_rope_qk_norm(
                ctx,
                enc,
                &carrier.q_dev,
                q_norm_w,
                &carrier.q_normed_dev,
                &carrier.nh_buf,
                &carrier.hd_buf,
                &carrier.nrot_buf,
                &carrier.theta_buf,
                &carrier.eps_buf,
                &carrier.pos_buf,
                carrier.seq_len,
                carrier.num_heads,
            );
            encode_prefill_rope_qk_norm(
                ctx,
                enc,
                &carrier.k_dev,
                k_norm_w,
                &carrier.k_normed_dev,
                &carrier.nkv_buf,
                &carrier.hd_buf,
                &carrier.nrot_buf,
                &carrier.theta_buf,
                &carrier.eps_buf,
                &carrier.pos_buf,
                carrier.seq_len,
                carrier.num_kv_heads,
            );
            compute::chain_barrier(ctx, enc);
        }
        PrefillAtnNormMode::Exact { eps, n_rot, imrope } => {
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                &carrier.q_dev,
                q_norm_w,
                &carrier.q_normed_dev,
                carrier.seq_len * carrier.num_heads,
                carrier.head_dim,
                eps,
            )
            .map_err(|error| format!("Metal prefill ATN exact q RMS norm failed: {error:?}"))?;
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                &carrier.k_dev,
                k_norm_w,
                &carrier.k_normed_dev,
                carrier.seq_len * carrier.num_kv_heads,
                carrier.head_dim,
                eps,
            )
            .map_err(|error| format!("Metal prefill ATN exact k RMS norm failed: {error:?}"))?;
            if imrope {
                compute::encode_prefill_imrope_only(
                    ctx,
                    enc,
                    &carrier.q_normed_dev,
                    &carrier.q_normed_dev,
                    &carrier.rope_cos_sin_dev,
                    carrier.num_heads,
                    carrier.head_dim,
                    n_rot,
                    carrier.seq_len,
                )
                .map_err(|error| format!("Metal prefill ATN q IMRoPE failed: {error:?}"))?;
                compute::encode_prefill_imrope_only(
                    ctx,
                    enc,
                    &carrier.k_normed_dev,
                    &carrier.k_normed_dev,
                    &carrier.rope_cos_sin_dev,
                    carrier.num_kv_heads,
                    carrier.head_dim,
                    n_rot,
                    carrier.seq_len,
                )
                .map_err(|error| format!("Metal prefill ATN k IMRoPE failed: {error:?}"))?;
            } else {
                compute::encode_prefill_rope_only(
                    ctx,
                    enc,
                    &carrier.q_normed_dev,
                    &carrier.q_normed_dev,
                    &carrier.rope_cos_sin_dev,
                    carrier.num_heads,
                    carrier.head_dim,
                    n_rot,
                    carrier.seq_len,
                )
                .map_err(|error| format!("Metal prefill ATN q RoPE failed: {error:?}"))?;
                compute::encode_prefill_rope_only(
                    ctx,
                    enc,
                    &carrier.k_normed_dev,
                    &carrier.k_normed_dev,
                    &carrier.rope_cos_sin_dev,
                    carrier.num_kv_heads,
                    carrier.head_dim,
                    n_rot,
                    carrier.seq_len,
                )
                .map_err(|error| format!("Metal prefill ATN k RoPE failed: {error:?}"))?;
            }
        }
    }
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Activation);
    }
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.k_normed_dev,
        kv_out.0,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.v_dev,
        kv_out.1,
        &carrier.kv_elems_buf,
        carrier.seq_len * carrier.kv_dim,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Down);
    }
    encode_flash_attn_prefill(
        ctx,
        enc,
        &carrier.q_normed_dev,
        kv_out.0,
        kv_out.1,
        &carrier.attn_out_dev,
        &carrier.nh_buf,
        &carrier.nkv_buf,
        &carrier.kv_len_buf,
        &carrier.seq_buf,
        &carrier.scale_buf,
        None,
        None,
        carrier.num_heads,
        carrier.seq_len,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
        sampler.begin(enc, QwenMoeLlamaIdStage::Reduce);
    }
    encode_prefill_gate_apply(
        ctx,
        enc,
        &carrier.attn_out_dev,
        &carrier.gate_dev,
        &carrier.attn_gated_dev,
        &carrier.q_elems_buf,
        carrier.seq_len * carrier.q_dim,
    );
    compute::chain_barrier(ctx, enc);
    if let Some(sampler) = stage_sampler.as_deref_mut() {
        sampler.end(enc);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_prefill_atn_o_tail_bound_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnOTailCarrier,
    weights: PrefillAtnCoreOpsWeights<'_>,
    attn_norm_w: &ProtocolObject<dyn MTLBuffer>,
    q_norm_w: &ProtocolObject<dyn MTLBuffer>,
    k_norm_w: &ProtocolObject<dyn MTLBuffer>,
    norm_mode: PrefillAtnNormMode,
    o_w_buf: &ProtocolObject<dyn MTLBuffer>,
    o_w_off: u32,
    o_quant: TensoropsQuant,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
    stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<(), String> {
    let core = &carrier.core;
    encode_atn_core_ops(
        ctx,
        enc,
        core,
        weights,
        attn_norm_w,
        q_norm_w,
        k_norm_w,
        norm_mode,
        hidden_in,
        kv_out,
        stage_sampler,
    )?;
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &core.attn_gated_dev,
        &carrier.o_in_f16_dev,
        &core.q_elems_buf,
        core.seq_len * core.q_dim,
    );
    compute::chain_barrier(ctx, enc);
    encode_quant_gemm_v2(
        ctx,
        enc,
        o_quant,
        o_w_buf,
        o_w_off,
        &carrier.o_in_f16_dev,
        hidden_out,
        &core.hidden_cols_buf,
        &carrier.q_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    compute::chain_barrier(ctx, enc);
    crate::ffn_chain::encode_residual_add(
        ctx,
        enc,
        hidden_out,
        hidden_in,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    compute::chain_barrier(ctx, enc);
    Ok(())
}

pub(crate) fn encode_prefill_atn_o_tail_ops_profiled(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnOTailCarrier,
    spec: PrefillAtnOTailBackendSpecRef<'_>,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
    stage_sampler: Option<&mut QwenMoeLlamaIdStageSampler>,
) -> Result<(), String> {
    let core = &carrier.core;
    let spec_core = spec.core;
    let carrier_shape = (
        core.seq_len,
        core.num_heads,
        core.num_kv_heads,
        core.head_dim,
        core.hidden_dim,
        core.q_dim,
        core.kv_dim,
    );
    let spec_shape = (
        spec_core.seq_len,
        spec_core.num_heads,
        spec_core.num_kv_heads,
        spec_core.head_dim,
        spec_core.hidden_dim,
        spec_core.q_dim,
        spec_core.kv_dim,
    );
    if carrier_shape != spec_shape {
        return Err(format!(
            "Metal prefill ATN o-tail: carrier/spec shape mismatch: carrier={carrier_shape:?} spec={spec_shape:?}"
        ));
    }
    if spec_core.attn_norm_w.len() != core.hidden_dim
        || spec_core.q_norm_w.len() != core.head_dim
        || spec_core.k_norm_w.len() != core.head_dim
    {
        return Err(format!(
            "Metal prefill ATN o-tail: norm length mismatch: attn={} q={} k={} expected={}/{}/{}",
            spec_core.attn_norm_w.len(),
            spec_core.q_norm_w.len(),
            spec_core.k_norm_w.len(),
            core.hidden_dim,
            core.head_dim,
            core.head_dim,
        ));
    }
    core.update_rope_cos_sin(
        spec_core.imrope_positions,
        spec_core.imrope_sections,
        spec_core.rope_theta,
        spec_core.pos_start,
    );
    let hidden_bytes = core
        .seq_len
        .checked_mul(core.hidden_dim)
        .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "Metal prefill ATN o-tail: hidden buffer size overflow".to_string())?;
    let kv_bytes = core
        .seq_len
        .checked_mul(core.kv_dim)
        .and_then(|len| len.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| "Metal prefill ATN o-tail: KV buffer size overflow".to_string())?;
    if hidden_in.length() < hidden_bytes || hidden_out.length() < hidden_bytes {
        return Err(format!(
            "Metal prefill ATN o-tail: hidden buffer too small: in={} out={} required={hidden_bytes}",
            hidden_in.length(),
            hidden_out.length(),
        ));
    }
    if kv_out.0.length() < kv_bytes || kv_out.1.length() < kv_bytes {
        return Err(format!(
            "Metal prefill ATN o-tail: KV buffer too small: k={} v={} required={kv_bytes}",
            kv_out.0.length(),
            kv_out.1.length(),
        ));
    }

    // 한 command buffer에서 큰 shape carrier를 여러 layer가 공유하므로 norm source는
    // 호출별 immutable buffer로 유지해 뒤 layer의 CPU 준비가 앞 dispatch를 덮지 않게 한다.
    let attn_norm_w = shared_f32_buf(ctx, spec_core.attn_norm_w);
    let q_norm_w = shared_f32_buf(ctx, spec_core.q_norm_w);
    let k_norm_w = shared_f32_buf(ctx, spec_core.k_norm_w);
    let (q_w_buf, q_w_off) = compute::wrap_nocopy(ctx, spec_core.q_weight.raw);
    let (k_w_buf, k_w_off) = compute::wrap_nocopy(ctx, spec_core.k_weight.raw);
    let (v_w_buf, v_w_off) = compute::wrap_nocopy(ctx, spec_core.v_weight.raw);
    let (o_w_buf, o_w_off) = compute::wrap_nocopy(ctx, spec.o_weight.raw);

    encode_prefill_atn_o_tail_bound_ops(
        ctx,
        enc,
        carrier,
        PrefillAtnCoreOpsWeights {
            q_w_buf: &q_w_buf,
            q_w_off,
            q_quant: spec_core.q_weight.quant,
            k_w_buf: &k_w_buf,
            k_w_off,
            k_quant: spec_core.k_weight.quant,
            v_w_buf: &v_w_buf,
            v_w_off,
            v_quant: spec_core.v_weight.quant,
        },
        &attn_norm_w,
        &q_norm_w,
        &k_norm_w,
        PrefillAtnNormMode::Exact {
            eps: spec_core.norm_eps,
            n_rot: spec_core.n_rot,
            imrope: spec_core.imrope_positions.is_some(),
        },
        &o_w_buf,
        o_w_off,
        spec.o_weight.quant,
        hidden_in,
        hidden_out,
        kv_out,
        stage_sampler,
    )?;
    Ok(())
}

pub(crate) fn encode_prefill_atn_o_tail_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    carrier: &PrefillAtnOTailCarrier,
    spec: PrefillAtnOTailBackendSpecRef<'_>,
    hidden_in: &ProtocolObject<dyn MTLBuffer>,
    hidden_out: &ProtocolObject<dyn MTLBuffer>,
    kv_out: (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ),
) -> Result<(), String> {
    encode_prefill_atn_o_tail_ops_profiled(
        ctx, enc, carrier, spec, hidden_in, hidden_out, kv_out, None,
    )
}

pub(crate) fn prefill_atn_core_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillAtnCoreCarrier,
    req: PrefillAtnCoreDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    carrier.upload(req.hidden, req.attn_norm_w, req.q_norm_w, req.k_norm_w);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal prefill ATN core: command buffer creation failed".to_string())?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| "Metal prefill ATN core: compute encoder creation failed".to_string())?;

    encode_atn_core_ops(
        ctx,
        &enc,
        carrier,
        PrefillAtnCoreOpsWeights::from_dispatch(&req),
        &carrier.attn_norm_w_dev,
        &carrier.q_norm_w_dev,
        &carrier.k_norm_w_dev,
        PrefillAtnNormMode::LegacyTree,
        &carrier.hidden_dev,
        (&carrier.k_f16_dev, &carrier.v_f16_dev),
        None,
    )?;

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let attn_out = {
        let c = carrier.attn_gated_dev.contents();
        unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const f32, carrier.seq_len * carrier.q_dim)
        }
        .to_vec()
    };
    let k_bits = {
        let c = carrier.k_f16_dev.contents();
        unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const u16, carrier.seq_len * carrier.kv_dim)
        }
        .to_vec()
    };
    let v_bits = {
        let c = carrier.v_f16_dev.contents();
        unsafe {
            std::slice::from_raw_parts(c.as_ptr() as *const u16, carrier.seq_len * carrier.kv_dim)
        }
        .to_vec()
    };
    Ok((attn_out, k_bits, v_bits))
}

pub(crate) fn prefill_atn_o_tail_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillAtnOTailCarrier,
    req: PrefillAtnOTailDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    let core = &carrier.core;
    carrier.upload_hidden(req.hidden);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal prefill ATN o-tail: command buffer creation failed".to_string())?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| "Metal prefill ATN o-tail: compute encoder creation failed".to_string())?;

    encode_prefill_atn_o_tail_ops(
        ctx,
        &enc,
        carrier,
        req.spec,
        &core.hidden_dev,
        &carrier.o_proj_dev,
        (&core.k_f16_dev, &core.v_f16_dev),
    )?;

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let hidden = readback(&carrier.o_proj_dev, core.seq_len * core.hidden_dim);
    let k_bits = {
        let c = core.k_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    let v_bits = {
        let c = core.v_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    Ok((hidden, k_bits, v_bits))
}

fn encode_muse_o_tail_ffn_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &MusePrefillOTailFfnCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    attn_out: &ProtocolObject<dyn MTLBuffer>,
    req: MuseOTailFfnOpsRequest<'_>,
) {
    encode_cast_f32_to_f16(
        ctx,
        enc,
        attn_out,
        &carrier.o_in_f16_dev,
        &carrier.q_elems_buf,
        carrier.seq_len * carrier.q_dim,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.o_in_f16_dev]);
    encode_quant_gemm_v2(
        ctx,
        enc,
        req.o_quant,
        req.o_w_buf,
        req.o_w_off,
        &carrier.o_in_f16_dev,
        &carrier.o_proj_dev,
        &carrier.hidden_dim_buf,
        &carrier.q_dim_buf,
        &carrier.seq_buf,
        carrier.hidden_dim,
        carrier.seq_len,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.o_proj_dev]);
    encode_rms_norm_batch(
        ctx,
        enc,
        &carrier.o_proj_dev,
        &carrier.post_attn_norm_w_dev,
        &carrier.ffn_normed_dev,
        &carrier.hidden_dim_buf,
        &carrier.post_norm_eps_buf,
        carrier.seq_len,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.ffn_normed_dev]);
    crate::ffn_chain::encode_residual_add(
        ctx,
        enc,
        hidden_dev,
        &carrier.ffn_normed_dev,
        &carrier.hidden_elems_buf,
        carrier.seq_len * carrier.hidden_dim,
    );
    compute::chain_barrier_resources(ctx, enc, [hidden_dev, &*carrier.ffn_normed_dev]);
    encode_rms_norm_batch(
        ctx,
        enc,
        hidden_dev,
        &carrier.ffn_norm_w_dev,
        &carrier.ffn_normed_dev,
        &carrier.hidden_dim_buf,
        &carrier.norm_eps_buf,
        carrier.seq_len,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.ffn_normed_dev]);
    encode_cast_f32_to_f16(
        ctx,
        enc,
        &carrier.ffn_normed_dev,
        &carrier.ffn_normed_f16_dev,
        &carrier.hidden_elems_buf,
        carrier.seq_len * carrier.hidden_dim,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.ffn_normed_f16_dev]);
    for (quant, weight, offset, output) in [
        (
            req.ffn_gate_quant,
            req.ffn_gate_w_buf,
            req.ffn_gate_w_off,
            &*carrier.ffn_gate_dev,
        ),
        (
            req.ffn_up_quant,
            req.ffn_up_w_buf,
            req.ffn_up_w_off,
            &*carrier.ffn_up_dev,
        ),
    ] {
        encode_quant_gemm_v2(
            ctx,
            enc,
            quant,
            weight,
            offset,
            &carrier.ffn_normed_f16_dev,
            output,
            &carrier.ffn_dim_buf,
            &carrier.hidden_dim_buf,
            &carrier.seq_buf,
            carrier.ffn_dim,
            carrier.seq_len,
        );
    }
    compute::chain_barrier_resources(ctx, enc, [&*carrier.ffn_gate_dev, &*carrier.ffn_up_dev]);
    encode_silu_mul_to_f16(
        ctx,
        enc,
        &carrier.ffn_gate_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_elems_buf,
        carrier.seq_len * carrier.ffn_dim,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.ffn_act_f16_dev]);
    encode_quant_gemm_v2(
        ctx,
        enc,
        req.ffn_down_quant,
        req.ffn_down_w_buf,
        req.ffn_down_w_off,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_down_dev,
        &carrier.hidden_dim_buf,
        &carrier.ffn_dim_buf,
        &carrier.seq_buf,
        carrier.hidden_dim,
        carrier.seq_len,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.ffn_down_dev]);
    encode_rms_norm_batch(
        ctx,
        enc,
        &carrier.ffn_down_dev,
        &carrier.post_ffn_norm_w_dev,
        &carrier.ffn_normed_dev,
        &carrier.hidden_dim_buf,
        &carrier.post_norm_eps_buf,
        carrier.seq_len,
    );
    compute::chain_barrier_resources(ctx, enc, [&*carrier.ffn_normed_dev]);
    crate::ffn_chain::encode_residual_add(
        ctx,
        enc,
        hidden_dev,
        &carrier.ffn_normed_dev,
        &carrier.hidden_elems_buf,
        carrier.seq_len * carrier.hidden_dim,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_muse_full_layer_ops(
    ctx: &MetalContext,
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    carrier: &MusePrefillFullLayerCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    req: &MusePrefillFullLayerDispatchRequest<'_>,
) -> Result<(), String> {
    let core = &carrier.core;
    let diagnostic_skip = (core.seq_len <= 8)
        .then(|| std::env::var("RNB_METAL_MUSE_SKIP_SEGMENT").ok())
        .flatten();

    if diagnostic_skip.as_deref() != Some("a") {
        if req.target_kv.is_some() {
            encode_rms_norm_batch(
                ctx,
                enc,
                hidden_dev,
                &core.attn_norm_w_dev,
                &core.normed_dev,
                &core.hidden_cols_buf,
                &core.eps_buf,
                core.seq_len,
            );
        } else {
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                hidden_dev,
                &core.attn_norm_w_dev,
                &core.normed_dev,
                core.seq_len,
                core.hidden_dim,
                req.norm_eps,
            )
            .map_err(|error| format!("Metal Muse prefill attention RMS norm failed: {error:?}"))?;
        }
        compute::chain_barrier_resources(ctx, enc, [&*core.normed_dev]);
        encode_cast_f32_to_f16(
            ctx,
            &enc,
            &core.normed_dev,
            &core.normed_f16_dev,
            &core.hidden_elems_buf,
            core.seq_len * core.hidden_dim,
        );
        compute::chain_barrier(ctx, &enc);
        encode_quant_gemm_v2(
            ctx,
            &enc,
            req.q_quant,
            req.q_w_buf,
            req.q_w_off,
            &core.normed_f16_dev,
            &core.q_dev,
            &core.q_n_buf,
            &core.hidden_cols_buf,
            &core.seq_buf,
            core.q_dim,
            core.seq_len,
        );
        encode_quant_gemm_v2(
            ctx,
            &enc,
            req.k_quant,
            req.k_w_buf,
            req.k_w_off,
            &core.normed_f16_dev,
            &core.k_dev,
            &core.kv_n_buf,
            &core.hidden_cols_buf,
            &core.seq_buf,
            core.kv_dim,
            core.seq_len,
        );
        encode_quant_gemm_v2(
            ctx,
            &enc,
            req.v_quant,
            req.v_w_buf,
            req.v_w_off,
            &core.normed_f16_dev,
            &core.v_dev,
            &core.kv_n_buf,
            &core.hidden_cols_buf,
            &core.seq_buf,
            core.kv_dim,
            core.seq_len,
        );
        encode_quant_gemm_v2(
            ctx,
            &enc,
            req.attention_gate_quant,
            req.attention_gate_w_buf,
            req.attention_gate_w_off,
            &core.normed_f16_dev,
            &carrier.attention_gate_dev,
            &core.q_n_buf,
            &core.hidden_cols_buf,
            &core.seq_buf,
            core.q_dim,
            core.seq_len,
        );
        compute::chain_barrier(ctx, &enc);
    }

    if diagnostic_skip.as_deref() != Some("b") {
        if req.target_kv.is_some() {
            encode_rms_norm_batch(
                ctx,
                enc,
                &core.q_dev,
                &core.q_norm_w_dev,
                &core.q_normed_dev,
                &core.hd_buf,
                &core.eps_buf,
                core.seq_len * core.num_heads,
            );
            encode_rms_norm_batch(
                ctx,
                enc,
                &core.k_dev,
                &core.k_norm_w_dev,
                &core.k_normed_dev,
                &core.hd_buf,
                &core.eps_buf,
                core.seq_len * core.num_kv_heads,
            );
        } else {
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                &core.q_dev,
                &core.q_norm_w_dev,
                &core.q_normed_dev,
                core.seq_len * core.num_heads,
                core.head_dim,
                req.norm_eps,
            )
            .map_err(|error| format!("Metal Muse prefill Q RMS norm failed: {error:?}"))?;
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                ctx,
                enc,
                &core.k_dev,
                &core.k_norm_w_dev,
                &core.k_normed_dev,
                core.seq_len * core.num_kv_heads,
                core.head_dim,
                req.norm_eps,
            )
            .map_err(|error| format!("Metal Muse prefill K RMS norm failed: {error:?}"))?;
        }
        compute::chain_barrier(ctx, &enc);
        if req.apply_rope {
            compute::encode_prefill_rope_only(
                ctx,
                &enc,
                &core.q_normed_dev,
                &core.q_normed_dev,
                &core.rope_cos_sin_dev,
                core.num_heads,
                core.head_dim,
                core.head_dim,
                core.seq_len,
            )
            .map_err(|error| format!("Metal Muse prefill Q RoPE failed: {error:?}"))?;
            compute::encode_prefill_rope_only(
                ctx,
                &enc,
                &core.k_normed_dev,
                &core.k_normed_dev,
                &core.rope_cos_sin_dev,
                core.num_kv_heads,
                core.head_dim,
                core.head_dim,
                core.seq_len,
            )
            .map_err(|error| format!("Metal Muse prefill K RoPE failed: {error:?}"))?;
        }
        compute::chain_barrier(ctx, &enc);
        encode_cast_f32_to_f16(
            ctx,
            &enc,
            &core.k_normed_dev,
            &core.k_f16_dev,
            &core.kv_elems_buf,
            core.seq_len * core.kv_dim,
        );
        encode_cast_f32_to_f16(
            ctx,
            &enc,
            &core.v_dev,
            &core.v_f16_dev,
            &core.kv_elems_buf,
            core.seq_len * core.kv_dim,
        );
        let (attention_key, attention_value, attention_kv_len) = if let Some(target) = req.target_kv
        {
            let pos_start_buf = u32_buf(ctx, target.pos_start as u32);
            compute::encode_kv_append_batch(
                ctx,
                enc,
                &core.k_normed_dev,
                &core.v_dev,
                target.key,
                target.value,
                &core.kv_n_buf,
                &pos_start_buf,
                &core.seq_buf,
                core.seq_len * core.kv_dim,
            );
            compute::chain_barrier_resources(ctx, enc, [target.key, target.value]);
            (target.key, target.value, target.kv_len)
        } else {
            (&*core.k_f16_dev, &*core.v_f16_dev, core.seq_len)
        };
        compute::chain_barrier(ctx, &enc);
        crate::dflash_attention::encode_muse_target_attention_f16_hd128(
            ctx,
            &enc,
            &core.q_normed_dev,
            attention_key,
            attention_value,
            &core.attn_out_dev,
            core.seq_len,
            attention_kv_len,
            core.num_heads,
            core.num_kv_heads,
            req.sliding_window,
            req.scale,
        )?;
        compute::chain_barrier(ctx, &enc);
        encode_prefill_gate_apply(
            ctx,
            &enc,
            &core.attn_out_dev,
            &carrier.attention_gate_dev,
            &core.attn_gated_dev,
            &core.q_elems_buf,
            core.seq_len * core.q_dim,
        );
        compute::chain_barrier(ctx, &enc);
    }
    if diagnostic_skip.as_deref() != Some("c") {
        encode_muse_o_tail_ffn_ops(
            ctx,
            enc,
            &carrier.tail,
            hidden_dev,
            &core.attn_gated_dev,
            req.ops,
        );
    }
    compute::chain_barrier(ctx, &enc);
    Ok(())
}

pub(crate) fn prefill_muse_full_layer_dispatch(
    ctx: &MetalContext,
    carrier: &MusePrefillFullLayerCarrier,
    req: MusePrefillFullLayerDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    let core = &carrier.core;
    core.upload(req.hidden, req.attn_norm_w, req.q_norm_w, req.k_norm_w);
    carrier.tail.upload_ops(None, req.ops);

    let cmd = ctx.queue.commandBuffer().ok_or_else(|| {
        "Metal Muse prefill full layer: command buffer creation failed".to_string()
    })?;
    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
        "Metal Muse prefill full layer: compute encoder creation failed".to_string()
    })?;
    encode_muse_full_layer_ops(ctx, &enc, carrier, &core.hidden_dev, &req)?;

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let hidden = readback(
        &carrier.tail.hidden_dev,
        carrier.seq_len * carrier.tail.hidden_dim,
    );
    if carrier.seq_len <= 8
        && std::env::var_os("RNB_METAL_MUSE_STAGE_COMPARE").is_some()
        && !{
            static TRACED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            TRACED.swap(true, std::sync::atomic::Ordering::Relaxed)
        }
    {
        let trace = |label: &str, values: Vec<f32>| {
            let hash = values.iter().fold(0xcbf29ce484222325u64, |hash, value| {
                (hash ^ value.to_bits() as u64).wrapping_mul(0x100000001b3)
            });
            eprintln!(
                "[muse-stage] reference {label} hash={hash:016x} first={:?}",
                &values[..values.len().min(8)]
            );
        };
        trace(
            "q",
            readback(&core.q_normed_dev, carrier.seq_len * carrier.q_dim),
        );
        trace(
            "k",
            readback(&core.k_normed_dev, carrier.seq_len * carrier.kv_dim),
        );
        trace("v", readback(&core.v_dev, carrier.seq_len * carrier.kv_dim));
        trace(
            "attn",
            readback(&core.attn_out_dev, carrier.seq_len * carrier.q_dim),
        );
        trace(
            "gated",
            readback(&core.attn_gated_dev, carrier.seq_len * carrier.q_dim),
        );
        trace(
            "o",
            readback(
                &carrier.tail.o_proj_dev,
                carrier.seq_len * carrier.tail.hidden_dim,
            ),
        );
        trace("hidden", hidden.clone());
    }
    let k_bits = unsafe {
        std::slice::from_raw_parts(
            core.k_f16_dev.contents().as_ptr().cast::<u16>(),
            carrier.seq_len * carrier.kv_dim,
        )
        .to_vec()
    };

    let v_bits = unsafe {
        std::slice::from_raw_parts(
            core.v_f16_dev.contents().as_ptr().cast::<u16>(),
            carrier.seq_len * carrier.kv_dim,
        )
        .to_vec()
    };
    Ok((hidden, k_bits, v_bits))
}

pub(crate) fn prefill_muse_layer_range_submit<F>(
    ctx: &MetalContext,
    state: &mut MusePrefillLayerRangeState,
    carrier: &MusePrefillFullLayerCarrier,
    layer_idx: usize,
    feature_index: Option<usize>,
    req: MusePrefillFullLayerDispatchRequest<'_>,
    after_layer: F,
) -> Result<MusePrefillLayerRangePending, String>
where
    F: FnOnce(
        &ProtocolObject<dyn MTLComputeCommandEncoder>,
        &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<(), String>,
{
    let core = &carrier.core;
    if state.hidden_elements != core.seq_len * core.hidden_dim {
        return Err("Metal Muse prefill range hidden shape mismatch".to_string());
    }
    copy_f32(req.attn_norm_w, &core.attn_norm_w_dev);
    copy_f32(req.q_norm_w, &core.q_norm_w_dev);
    copy_f32(req.k_norm_w, &core.k_norm_w_dev);
    core.update_rope_cos_sin(None, [0; 4], req.rope_theta, req.pos_start);
    carrier.tail.upload_op_weights(None, req.ops);
    let command = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal Muse prefill range command buffer creation failed".to_string())?;
    let encoder = compute::chain_compute_encoder(ctx, &command);
    state.encode_feature(ctx, &encoder, feature_index);
    encode_muse_full_layer_ops(ctx, &encoder, carrier, &state.hidden_dev, &req)?;
    after_layer(&encoder, &state.hidden_dev)?;
    encoder.endEncoding();
    command.commit();
    Ok(MusePrefillLayerRangePending {
        layer_idx,
        command,
        k_f16_dev: core.k_f16_dev.clone(),
        v_f16_dev: core.v_f16_dev.clone(),
        kv_len: core.seq_len * core.kv_dim,
        stage_trace: (core.seq_len <= 8
            && layer_idx == 0
            && std::env::var_os("RNB_METAL_MUSE_STAGE_COMPARE").is_some())
        .then(|| {
            (
                core.q_dev.clone(),
                core.seq_len * core.q_dim,
                core.k_dev.clone(),
                core.seq_len * core.kv_dim,
                core.q_normed_dev.clone(),
                core.seq_len * core.q_dim,
                core.k_normed_dev.clone(),
                core.seq_len * core.kv_dim,
                core.v_dev.clone(),
                core.seq_len * core.kv_dim,
                core.attn_out_dev.clone(),
                core.seq_len * core.q_dim,
                core.attn_gated_dev.clone(),
                core.seq_len * core.q_dim,
                carrier.tail.o_proj_dev.clone(),
                core.seq_len * core.hidden_dim,
            )
        }),
        completed: false,
    })
}

pub(crate) fn prefill_muse_layer_range_complete(
    mut pending: MusePrefillLayerRangePending,
    on_kv: &mut impl FnMut(usize, &[u16], &[u16]) -> Result<(), String>,
) -> Result<(), String> {
    pending.command.waitUntilCompleted();
    pending.completed = true;
    ensure_command_completed(&pending.command)?;
    if std::env::var("RNB_METAL_MUSE_LAYER_GPU").as_deref() == Ok("1") {
        eprintln!(
            "[muse-layer-gpu] layer={} gpu_ms={:.3}",
            pending.layer_idx,
            (pending.command.GPUEndTime() - pending.command.GPUStartTime()) * 1000.0,
        );
    }
    if let Some((q_raw, qrn, k_raw, krn, q, qn, k, kn, v, vn, attn, an, gated, gn, o, on)) =
        pending.stage_trace.as_ref()
    {
        let trace = |label: &str, values: Vec<f32>| {
            let hash = values.iter().fold(0xcbf29ce484222325u64, |hash, value| {
                (hash ^ value.to_bits() as u64).wrapping_mul(0x100000001b3)
            });
            eprintln!(
                "[muse-stage] reference-range {label} hash={hash:016x} first={:?}",
                &values[..values.len().min(8)]
            );
        };
        trace("q_raw", readback(q_raw, *qrn));
        trace("k_raw", readback(k_raw, *krn));
        trace("q", readback(q, *qn));
        trace("k", readback(k, *kn));
        trace("v", readback(v, *vn));
        trace("attn", readback(attn, *an));
        trace("gated", readback(gated, *gn));
        trace("o", readback(o, *on));
    }

    let k_bits = unsafe {
        std::slice::from_raw_parts(
            pending.k_f16_dev.contents().as_ptr().cast::<u16>(),
            pending.kv_len,
        )
    };
    let v_bits = unsafe {
        std::slice::from_raw_parts(
            pending.v_f16_dev.contents().as_ptr().cast::<u16>(),
            pending.kv_len,
        )
    };
    on_kv(pending.layer_idx, k_bits, v_bits)
}

pub(crate) fn prefill_muse_o_tail_ffn_dispatch(
    ctx: &MetalContext,
    carrier: &MusePrefillOTailFfnCarrier,
    req: MusePrefillOTailFfnDispatchRequest<'_>,
) -> Result<Vec<f32>, String> {
    carrier.upload_attention(req.attn_out);
    carrier.upload_ops(None, req.ops);
    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal Muse prefill O+FFN: command buffer creation failed".to_string())?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| "Metal Muse prefill O+FFN: compute encoder creation failed".to_string())?;
    encode_muse_o_tail_ffn_ops(
        ctx,
        &enc,
        carrier,
        &carrier.hidden_dev,
        &carrier.attn_out_dev,
        req.ops,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;
    Ok(readback(
        &carrier.hidden_dev,
        carrier.seq_len * carrier.hidden_dim,
    ))
}

pub(crate) fn prefill_muse_target_attention_o_tail_ffn_dispatch(
    ctx: &MetalContext,
    target: &crate::dflash_attention::MuseTargetAttentionCarrier,
    carrier: &MusePrefillOTailFfnCarrier,
    req: MuseTargetAttentionOTailFfnDispatchRequest<'_>,
) -> Result<Vec<f32>, String> {
    target.upload_query(req.query)?;
    carrier.upload_ops(Some(req.attention_gate), req.ops);
    let cmd = ctx.queue.commandBuffer().ok_or_else(|| {
        "Metal Muse target attention+O+FFN: command buffer creation failed".to_string()
    })?;
    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
        "Metal Muse target attention+O+FFN: compute encoder creation failed".to_string()
    })?;
    target.encode(
        ctx,
        &enc,
        req.key,
        req.value,
        req.kv_len,
        req.sliding_window,
        req.scale,
    )?;
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    encode_prefill_gate_apply(
        ctx,
        &enc,
        target.output_buffer(),
        &carrier.attention_gate_dev,
        &carrier.attention_gated_dev,
        &carrier.q_elems_buf,
        carrier.seq_len * carrier.q_dim,
    );
    encode_muse_o_tail_ffn_ops(
        ctx,
        &enc,
        carrier,
        &carrier.hidden_dev,
        &carrier.attention_gated_dev,
        req.ops,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;
    Ok(readback(
        &carrier.hidden_dev,
        carrier.seq_len * carrier.hidden_dim,
    ))
}

pub(crate) fn dflash_full_layer_dispatch(
    ctx: &MetalContext,
    carrier: &DflashFullLayerCarrier,
    hidden_dev: &ProtocolObject<dyn MTLBuffer>,
    upload_hidden: bool,
    wait_readback: bool,
    req: DflashFullLayerDispatchRequest<'_>,
    output_top1: Option<DflashOutputTop1DispatchRequest<'_>>,
) -> Result<DflashFullLayerDispatchOutput, String> {
    let layer = &carrier.layer;
    let core = &layer.core;
    let base = &req.layer;
    if upload_hidden {
        copy_f32(base.core.hidden, hidden_dev);
    }
    layer.upload_weights(
        base.core.attn_norm_w,
        base.core.q_norm_w,
        base.core.k_norm_w,
        base.ffn_norm_w,
    );
    core.update_rope_cos_sin(None, [0; 4], req.rope_theta, req.position);
    let context_len = carrier
        .attention
        .upload_context(req.prior_k, req.prior_v, req.position)?;
    if let Some(output) = output_top1.as_ref() {
        if output.carrier.hidden_dim != core.hidden_dim
            || output.carrier.batch == 0
            || output.carrier.batch + 1 > core.seq_len
        {
            return Err("Metal DFlash output tail shape mismatch".to_string());
        }
        output.carrier.upload_output_norm(output.output_norm_w);
    }
    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal DFlash full layer command buffer creation failed".to_string())?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| "Metal DFlash full layer encoder creation failed".to_string())?;

    crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
        ctx,
        &enc,
        hidden_dev,
        &core.attn_norm_w_dev,
        &core.normed_dev,
        core.seq_len,
        core.hidden_dim,
        req.norm_eps,
    )
    .map_err(|error| format!("Metal DFlash attention RMS norm failed: {error:?}"))?;
    compute::chain_barrier_resources(ctx, &enc, [&*core.normed_dev]);
    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &core.normed_dev,
        &core.normed_f16_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    compute::chain_barrier(ctx, &enc);
    for (quant, weight, offset, output, n_buf, n) in [
        (
            base.core.q_quant,
            base.core.q_w_buf,
            base.core.q_w_off,
            &*core.q_dev,
            &*core.q_n_buf,
            core.q_dim,
        ),
        (
            base.core.k_quant,
            base.core.k_w_buf,
            base.core.k_w_off,
            &*core.k_dev,
            &*core.kv_n_buf,
            core.kv_dim,
        ),
        (
            base.core.v_quant,
            base.core.v_w_buf,
            base.core.v_w_off,
            &*core.v_dev,
            &*core.kv_n_buf,
            core.kv_dim,
        ),
    ] {
        encode_quant_gemm_v2(
            ctx,
            &enc,
            quant,
            weight,
            offset,
            &core.normed_f16_dev,
            output,
            n_buf,
            &core.hidden_cols_buf,
            &core.seq_buf,
            n,
            core.seq_len,
        );
    }
    compute::chain_barrier(ctx, &enc);
    compute::encode_prefill_neox_qk_norm_table(
        ctx,
        &enc,
        &core.q_dev,
        &core.q_norm_w_dev,
        &core.q_normed_dev,
        &core.rope_cos_sin_dev,
        &core.nh_buf,
        &core.hd_buf,
        &core.eps_buf,
        core.seq_len,
        core.num_heads,
    );
    compute::encode_prefill_neox_qk_norm_table(
        ctx,
        &enc,
        &core.k_dev,
        &core.k_norm_w_dev,
        &core.k_normed_dev,
        &core.rope_cos_sin_dev,
        &core.nkv_buf,
        &core.hd_buf,
        &core.eps_buf,
        core.seq_len,
        core.num_kv_heads,
    );
    compute::chain_barrier(ctx, &enc);
    carrier.attention.encode_from_buffers(
        ctx,
        &enc,
        &core.q_normed_dev,
        &core.k_normed_dev,
        &core.v_dev,
        &core.attn_out_dev,
        context_len,
        req.position,
    )?;
    compute::chain_barrier(ctx, &enc);

    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &core.attn_out_dev,
        &layer.o_in_f16_dev,
        &core.q_elems_buf,
        core.seq_len * core.q_dim,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*layer.o_in_f16_dev]);
    encode_quant_gemm_v2(
        ctx,
        &enc,
        base.o_quant,
        base.o_w_buf,
        base.o_w_off,
        &layer.o_in_f16_dev,
        &layer.o_proj_dev,
        &core.hidden_cols_buf,
        &layer.q_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*layer.o_proj_dev]);
    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        hidden_dev,
        &layer.o_proj_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    compute::chain_barrier_resources(ctx, &enc, [hidden_dev]);

    crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
        ctx,
        &enc,
        hidden_dev,
        &layer.ffn_norm_w_dev,
        &layer.ffn_normed_dev,
        core.seq_len,
        core.hidden_dim,
        req.norm_eps,
    )
    .map_err(|error| format!("Metal DFlash FFN RMS norm failed: {error:?}"))?;
    compute::chain_barrier_resources(ctx, &enc, [&*layer.ffn_normed_dev]);
    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &layer.ffn_normed_dev,
        &layer.ffn_normed_f16_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*layer.ffn_normed_f16_dev]);
    for (quant, weight, offset, output) in [
        (
            base.ffn_gate_quant,
            base.ffn_gate_w_buf,
            base.ffn_gate_w_off,
            &*layer.ffn_gate_dev,
        ),
        (
            base.ffn_up_quant,
            base.ffn_up_w_buf,
            base.ffn_up_w_off,
            &*layer.ffn_up_dev,
        ),
    ] {
        encode_quant_gemm_v2(
            ctx,
            &enc,
            quant,
            weight,
            offset,
            &layer.ffn_normed_f16_dev,
            output,
            &layer.ffn_dim_buf,
            &core.hidden_cols_buf,
            &core.seq_buf,
            layer.ffn_dim,
            core.seq_len,
        );
    }
    compute::chain_barrier_resources(ctx, &enc, [&*layer.ffn_gate_dev, &*layer.ffn_up_dev]);
    encode_silu_mul_to_f16(
        ctx,
        &enc,
        &layer.ffn_gate_dev,
        &layer.ffn_up_dev,
        &layer.ffn_act_f16_dev,
        &layer.ffn_elems_buf,
        core.seq_len * layer.ffn_dim,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*layer.ffn_act_f16_dev]);
    encode_quant_gemm_v2(
        ctx,
        &enc,
        base.ffn_down_quant,
        base.ffn_down_w_buf,
        base.ffn_down_w_off,
        &layer.ffn_act_f16_dev,
        &layer.ffn_down_dev,
        &core.hidden_cols_buf,
        &layer.ffn_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*layer.ffn_down_dev]);
    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        hidden_dev,
        &layer.ffn_down_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    if let Some(output) = output_top1.as_ref() {
        let tail = output.carrier;
        compute::chain_barrier_resources(ctx, &enc, [hidden_dev]);
        for row in 0..tail.batch {
            crate::ffn_chain::encode_rms_norm_io_offset(
                ctx,
                &enc,
                hidden_dev,
                (row + 1) * core.hidden_dim * std::mem::size_of::<f32>(),
                &tail.output_norm_w_dev,
                &tail.normalized_dev,
                row * core.hidden_dim * std::mem::size_of::<f32>(),
                &tail.hidden_dim_buf,
                &tail.eps_buf,
            );
        }
        compute::chain_barrier_resources(ctx, &enc, [&*tail.normalized_dev]);
        encode_cast_f32_to_f16(
            ctx,
            &enc,
            &tail.normalized_dev,
            &tail.normalized_f16_dev,
            &tail.normalized_elems_buf,
            tail.batch * tail.hidden_dim,
        );
        compute::chain_barrier_resources(ctx, &enc, [&*tail.normalized_f16_dev]);
        encode_quant_gemm_v2(
            ctx,
            &enc,
            output.output_quant,
            output.output_w_buf,
            output.output_w_off,
            &tail.normalized_f16_dev,
            &tail.logits_dev,
            &tail.vocab_size_buf,
            &tail.hidden_dim_buf,
            &tail.batch_buf,
            tail.vocab_size,
            tail.batch,
        );
        compute::chain_barrier_resources(ctx, &enc, [&*tail.logits_dev]);
        for row in 0..tail.batch {
            compute::encode_top1_probability_f32_at(
                ctx,
                &enc,
                &tail.logits_dev,
                row * tail.vocab_size * std::mem::size_of::<f32>(),
                &tail.token_dev,
                row * std::mem::size_of::<u32>(),
                &tail.probability_dev,
                row * std::mem::size_of::<f32>(),
                &tail.vocab_size_buf,
            );
        }
    }
    enc.endEncoding();
    cmd.commit();
    if wait_readback || output_top1.is_some() {
        cmd.waitUntilCompleted();
        ensure_command_completed(&cmd)?;
    }
    let hidden = if wait_readback {
        readback(hidden_dev, core.seq_len * core.hidden_dim)
    } else {
        Vec::new()
    };
    let output_top1 = output_top1.map(|output| output.carrier.read_output());
    Ok(DflashFullLayerDispatchOutput {
        hidden,
        output_top1,
    })
}
pub(crate) fn dflash_cache_seed_dispatch(
    ctx: &MetalContext,
    carrier: &DflashCacheSeedCarrier,
    req: DflashCacheSeedDispatchRequest<'_>,
) -> Result<Vec<(Vec<u16>, Vec<u16>)>, String> {
    let k_norm_weights = req
        .layers
        .iter()
        .map(|layer| layer.k_norm_w)
        .collect::<Vec<_>>();
    carrier.upload(
        req.features,
        req.encoder_norm_w,
        &k_norm_weights,
        req.token_count,
        req.start_position,
        req.rope_theta,
    );

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| "Metal DFlash cache seed command buffer creation failed".to_string())?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| "Metal DFlash cache seed encoder creation failed".to_string())?;

    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.features_dev,
        &carrier.features_f16_dev,
        &carrier.feature_elems_buf,
        req.token_count * carrier.feature_dim,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*carrier.features_f16_dev]);
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.encoder_quant,
        req.encoder_w_buf,
        req.encoder_w_off,
        &carrier.features_f16_dev,
        &carrier.projected_dev,
        &carrier.hidden_dim_buf,
        &carrier.feature_dim_buf,
        &carrier.token_count_buf,
        carrier.hidden_dim,
        req.token_count,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*carrier.projected_dev]);
    crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
        ctx,
        &enc,
        &carrier.projected_dev,
        &carrier.encoder_norm_w_dev,
        &carrier.fused_dev,
        req.token_count,
        carrier.hidden_dim,
        req.norm_eps,
    )
    .map_err(|error| format!("Metal DFlash encoder RMS norm failed: {error:?}"))?;
    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.fused_dev,
        &carrier.fused_f16_dev,
        &carrier.hidden_elems_buf,
        req.token_count * carrier.hidden_dim,
    );
    compute::chain_barrier_resources(ctx, &enc, [&*carrier.fused_f16_dev]);

    for (layer_index, layer) in req.layers.iter().enumerate() {
        encode_quant_gemm_v2(
            ctx,
            &enc,
            layer.k_quant,
            layer.k_w_buf,
            layer.k_w_off,
            &carrier.fused_f16_dev,
            &carrier.kv_dev,
            &carrier.kv_dim_buf,
            &carrier.hidden_dim_buf,
            &carrier.token_count_buf,
            carrier.kv_dim,
            req.token_count,
        );
        compute::chain_barrier_resources(ctx, &enc, [&*carrier.kv_dev]);
        compute::encode_prefill_neox_qk_norm_table(
            ctx,
            &enc,
            &carrier.kv_dev,
            &carrier.k_norm_w_dev[layer_index],
            &carrier.k_normed_dev,
            &carrier.rope_cos_sin_dev,
            &carrier.num_kv_heads_buf,
            &carrier.head_dim_buf,
            &carrier.eps_buf,
            req.token_count,
            carrier.num_kv_heads,
        );
        compute::chain_barrier_resources(ctx, &enc, [&*carrier.k_normed_dev]);
        encode_cast_f32_to_f16(
            ctx,
            &enc,
            &carrier.k_normed_dev,
            &carrier.k_f16_dev[layer_index],
            &carrier.kv_elems_buf,
            req.token_count * carrier.kv_dim,
        );
        compute::chain_barrier_resources(ctx, &enc, [&*carrier.k_f16_dev[layer_index]]);

        encode_quant_gemm_v2(
            ctx,
            &enc,
            layer.v_quant,
            layer.v_w_buf,
            layer.v_w_off,
            &carrier.fused_f16_dev,
            &carrier.kv_dev,
            &carrier.kv_dim_buf,
            &carrier.hidden_dim_buf,
            &carrier.token_count_buf,
            carrier.kv_dim,
            req.token_count,
        );
        compute::chain_barrier_resources(ctx, &enc, [&*carrier.kv_dev]);
        encode_cast_f32_to_f16(
            ctx,
            &enc,
            &carrier.kv_dev,
            &carrier.v_f16_dev[layer_index],
            &carrier.kv_elems_buf,
            req.token_count * carrier.kv_dim,
        );
        compute::chain_barrier_resources(ctx, &enc, [&*carrier.v_f16_dev[layer_index]]);
    }

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let elements = req.token_count * carrier.kv_dim;
    Ok(carrier
        .k_f16_dev
        .iter()
        .zip(&carrier.v_f16_dev)
        .map(|(keys, values)| unsafe {
            (
                std::slice::from_raw_parts(keys.contents().as_ptr() as *const u16, elements)
                    .to_vec(),
                std::slice::from_raw_parts(values.contents().as_ptr() as *const u16, elements)
                    .to_vec(),
            )
        })
        .collect())
}

pub(crate) fn prefill_atn_full_layer_dispatch(
    ctx: &MetalContext,
    carrier: &PrefillAtnFullLayerCarrier,
    req: PrefillAtnFullLayerDispatchRequest<'_>,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u16>), String> {
    let core = &carrier.core;
    carrier.upload(
        req.core.hidden,
        req.core.attn_norm_w,
        req.core.q_norm_w,
        req.core.k_norm_w,
        req.ffn_norm_w,
    );

    let cmd = ctx.queue.commandBuffer().ok_or_else(|| {
        "Metal prefill ATN full layer: command buffer creation failed".to_string()
    })?;
    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
        "Metal prefill ATN full layer: compute encoder creation failed".to_string()
    })?;

    encode_atn_core_ops(
        ctx,
        &enc,
        core,
        PrefillAtnCoreOpsWeights::from_dispatch(&req.core),
        &core.attn_norm_w_dev,
        &core.q_norm_w_dev,
        &core.k_norm_w_dev,
        PrefillAtnNormMode::LegacyTree,
        &core.hidden_dev,
        (&core.k_f16_dev, &core.v_f16_dev),
        None,
    )?;

    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &core.attn_gated_dev,
        &carrier.o_in_f16_dev,
        &core.q_elems_buf,
        core.seq_len * core.q_dim,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.o_quant,
        req.o_w_buf,
        req.o_w_off,
        &carrier.o_in_f16_dev,
        &carrier.o_proj_dev,
        &core.hidden_cols_buf,
        &carrier.q_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        &core.hidden_dev,
        &carrier.o_proj_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );

    encode_rms_norm_batch(
        ctx,
        &enc,
        &core.hidden_dev,
        &carrier.ffn_norm_w_dev,
        &carrier.ffn_normed_dev,
        &core.hidden_cols_buf,
        &core.eps_buf,
        core.seq_len,
    );
    encode_cast_f32_to_f16(
        ctx,
        &enc,
        &carrier.ffn_normed_dev,
        &carrier.ffn_normed_f16_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.ffn_gate_quant,
        req.ffn_gate_w_buf,
        req.ffn_gate_w_off,
        &carrier.ffn_normed_f16_dev,
        &carrier.ffn_gate_dev,
        &carrier.ffn_dim_buf,
        &core.hidden_cols_buf,
        &core.seq_buf,
        carrier.ffn_dim,
        core.seq_len,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.ffn_up_quant,
        req.ffn_up_w_buf,
        req.ffn_up_w_off,
        &carrier.ffn_normed_f16_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_dim_buf,
        &core.hidden_cols_buf,
        &core.seq_buf,
        carrier.ffn_dim,
        core.seq_len,
    );
    encode_silu_mul_to_f16(
        ctx,
        &enc,
        &carrier.ffn_gate_dev,
        &carrier.ffn_up_dev,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_elems_buf,
        core.seq_len * carrier.ffn_dim,
    );
    encode_quant_gemm_v2(
        ctx,
        &enc,
        req.ffn_down_quant,
        req.ffn_down_w_buf,
        req.ffn_down_w_off,
        &carrier.ffn_act_f16_dev,
        &carrier.ffn_down_dev,
        &core.hidden_cols_buf,
        &carrier.ffn_dim_buf,
        &core.seq_buf,
        core.hidden_dim,
        core.seq_len,
    );
    crate::ffn_chain::encode_residual_add(
        ctx,
        &enc,
        &core.hidden_dev,
        &carrier.ffn_down_dev,
        &core.hidden_elems_buf,
        core.seq_len * core.hidden_dim,
    );

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    ensure_command_completed(&cmd)?;

    let hidden = readback(&core.hidden_dev, core.seq_len * core.hidden_dim);
    let k_bits = {
        let c = core.k_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    let v_bits = {
        let c = core.v_f16_dev.contents();
        unsafe { std::slice::from_raw_parts(c.as_ptr() as *const u16, core.seq_len * core.kv_dim) }
            .to_vec()
    };
    Ok((hidden, k_bits, v_bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLDevice, MTLResourceOptions};
    use rnb_cpu::kernels::{
        norm::rms_norm_into,
        rope::{rope_imrope_inplace, rope_partial_inplace},
    };

    fn rms_norm_f64(input: &[f32], weight: &[f32], eps: f32, rows: usize) -> Vec<f32> {
        let cols = weight.len();
        assert_eq!(input.len(), rows * cols);
        let mut output = Vec::with_capacity(input.len());
        for row in input.chunks_exact(cols) {
            let mean_sq = row
                .iter()
                .map(|&value| {
                    let value = f64::from(value);
                    value * value
                })
                .sum::<f64>()
                / cols as f64;
            let inverse_rms = 1.0 / (mean_sq + f64::from(eps)).sqrt();
            output.extend(row.iter().zip(weight).map(|(&value, &scale)| {
                (f64::from(value) * inverse_rms * f64::from(scale)) as f32
            }));
        }
        output
    }

    fn metal_rms_norm(
        ctx: &MetalContext,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        rows: usize,
    ) -> Vec<f32> {
        let input_dev = shared_f32_buf(ctx, input);
        let weight_dev = shared_f32_buf(ctx, weight);
        let output_dev = empty_f32_buf(ctx, input.len());
        let cols_dev = u32_buf(ctx, weight.len() as u32);
        let eps_dev = f32_buf(ctx, eps);
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        encode_rms_norm_batch(
            ctx,
            &enc,
            &input_dev,
            &weight_dev,
            &output_dev,
            &cols_dev,
            &eps_dev,
            rows,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        ensure_command_completed(&cmd).expect("command completed");
        readback(&output_dev, input.len())
    }

    fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
        actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max)
    }

    fn assert_f32_bits(label: &str, got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len(), "{label} length");
        for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{label} bit mismatch at {index}: got={got:?} expected={expected:?}"
            );
        }
    }

    // RoPE 회전 적용은 Metal 커널이 FMA(fma)로 융합하는 반면 CPU 참조는 분리된
    // mul/sub 라 몇 ULP 어긋난다(cos/sin 테이블 자체는 CPU 계산으로 bit-identical).
    // 따라서 회전 결과는 bit-exact 가 아니라 tight tolerance 로 검증한다.
    fn assert_f32_close(label: &str, got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len(), "{label} length");
        for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            let tol = 1.0e-5 * expected.abs().max(1.0);
            assert!(
                (got - expected).abs() <= tol,
                "{label} mismatch at {index}: got={got:?} expected={expected:?} tol={tol}"
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_prefill_exact_qk_norm_and_rope_match_cpu_bits() {
        const SEQ_LEN: usize = 4;
        const HEAD_DIM: usize = 256;
        const N_ROT: usize = 64;
        const THETA: f32 = 10_000_000.0;
        const EPS: f32 = 1.0e-6;

        let ctx = compute::build_metal_context().expect("Metal context");
        for (num_heads, pos_start, label) in [(16usize, 0usize, "q"), (2usize, 4552usize, "k")] {
            let cols = HEAD_DIM;
            let rows = SEQ_LEN * num_heads;
            let len = rows * cols;
            let input: Vec<f32> = (0..len)
                .map(|index| {
                    let centered = (index.wrapping_mul(37) % 257) as f32 - 128.0;
                    centered * 0.000_976_562_5
                })
                .collect();
            let weight: Vec<f32> = (0..cols)
                .map(|index| 0.75 + (index % 17) as f32 * 0.015_625)
                .collect();
            let mut norm_expected = vec![0.0f32; len];
            rms_norm_into(&input, &weight, EPS, &mut norm_expected);
            let mut rope_expected = norm_expected.clone();
            rope_partial_inplace(
                &mut rope_expected,
                pos_start,
                HEAD_DIM,
                num_heads * HEAD_DIM,
                N_ROT,
                THETA,
            );

            let input_dev = shared_f32_buf(&ctx, &input);
            let weight_dev = shared_f32_buf(&ctx, &weight);
            let norm_dev = empty_f32_buf(&ctx, len);
            let rope_dev = empty_f32_buf(&ctx, len);
            let cmd = ctx.queue.commandBuffer().expect("command buffer");
            let rope_cos_sin_dev = shared_f32_buf(
                &ctx,
                &prefill_rope_cos_sin(SEQ_LEN, HEAD_DIM, N_ROT, THETA, pos_start),
            );
            let enc = compute::try_chain_compute_encoder(&ctx, &cmd).expect("compute encoder");
            crate::ffn_chain::encode_qwen_prefill_rms_norm_exact(
                &ctx,
                &enc,
                &input_dev,
                &weight_dev,
                &norm_dev,
                rows,
                cols,
                EPS,
            )
            .expect("exact RMS encode");
            compute::encode_prefill_rope_only(
                &ctx,
                &enc,
                &norm_dev,
                &rope_dev,
                &rope_cos_sin_dev,
                num_heads,
                HEAD_DIM,
                N_ROT,
                SEQ_LEN,
            )
            .expect("RoPE-only encode");
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            ensure_command_completed(&cmd).expect("command completed");

            assert_f32_bits(
                &format!("{label} qk norm"),
                &readback(&norm_dev, len),
                &norm_expected,
            );
            assert_f32_close(
                &format!("{label} RoPE"),
                &readback(&rope_dev, len),
                &rope_expected,
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn qwen_prefill_imrope_matches_cpu() {
        const SEQ_LEN: usize = 4;
        const NUM_HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        const N_ROT: usize = 64;
        const THETA: f32 = 10_000_000.0;
        const SECTIONS: [usize; 4] = [11, 11, 10, 0];
        let positions = [[0, 0, 0, 0], [1, 5, 9, 1], [2, 7, 3, 2], [11, 11, 11, 11]];
        let len = SEQ_LEN * NUM_HEADS * HEAD_DIM;
        let input = (0..len)
            .map(|index| ((index.wrapping_mul(29) % 251) as f32 - 125.0) / 128.0)
            .collect::<Vec<_>>();
        let mut expected = input.clone();
        rope_imrope_inplace(
            &mut expected,
            &positions,
            HEAD_DIM,
            NUM_HEADS * HEAD_DIM,
            N_ROT,
            SECTIONS,
            THETA,
        );

        let ctx = compute::build_metal_context().expect("Metal context");
        let input_dev = shared_f32_buf(&ctx, &input);
        let output_dev = empty_f32_buf(&ctx, len);
        let table_dev = shared_f32_buf(
            &ctx,
            &prefill_imrope_cos_sin(&positions, N_ROT, SECTIONS, THETA),
        );
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = compute::try_chain_compute_encoder(&ctx, &cmd).expect("compute encoder");
        compute::encode_prefill_imrope_only(
            &ctx,
            &enc,
            &input_dev,
            &output_dev,
            &table_dev,
            NUM_HEADS,
            HEAD_DIM,
            N_ROT,
            SEQ_LEN,
        )
        .expect("IMRoPE encode");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        ensure_command_completed(&cmd).expect("command completed");

        assert_f32_close("IMRoPE", &readback(&output_dev, len), &expected);
    }
    #[test]
    #[ignore = "requires a Metal device"]
    fn muse_prefill_o_tail_ffn_norms_match_f64_cpu() {
        const SEQ_LEN: usize = 2;
        const Q_DIM: usize = 256;
        const HIDDEN_DIM: usize = 256;
        const FFN_DIM: usize = 256;
        const NORM_EPS: f32 = 1.0e-6;
        const POST_NORM_EPS: f32 = 1.0e-5;

        let ctx = compute::build_metal_context().expect("Metal context");
        let hidden = (0..SEQ_LEN * HIDDEN_DIM)
            .map(|index| ((index * 29 % 251) as f32 - 125.0) * 0.001)
            .collect::<Vec<_>>();
        let attn_out = (0..SEQ_LEN * Q_DIM)
            .map(|index| ((index * 17 % 239) as f32 - 119.0) * 0.0015)
            .collect::<Vec<_>>();
        let post_attn_norm_w = (0..HIDDEN_DIM)
            .map(|index| 0.8 + (index % 13) as f32 * 0.01)
            .collect::<Vec<_>>();
        let ffn_norm_w = (0..HIDDEN_DIM)
            .map(|index| 0.7 + (index % 11) as f32 * 0.0125)
            .collect::<Vec<_>>();
        let post_ffn_norm_w = (0..HIDDEN_DIM)
            .map(|index| 0.9 + (index % 7) as f32 * 0.015)
            .collect::<Vec<_>>();
        let mut block = crate::tests_fixture::q4k_block_fixed();
        block[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.005).to_le_bytes());
        let q4_weight = |rows: usize, cols: usize| {
            block
                .iter()
                .cycle()
                .take(rows * (cols / 256) * 144)
                .copied()
                .collect::<Vec<_>>()
        };
        let o_weight = q4_weight(HIDDEN_DIM, Q_DIM);
        let gate_weight = q4_weight(FFN_DIM, HIDDEN_DIM);
        let up_weight = q4_weight(FFN_DIM, HIDDEN_DIM);
        let down_weight = q4_weight(HIDDEN_DIM, FFN_DIM);
        let round_f16 = |values: &[f32]| {
            values
                .iter()
                .map(|&value| half::f16::from_f32(value).to_f32())
                .collect::<Vec<_>>()
        };

        let o_proj = crate::tests_fixture::q4k_gemm_reference(
            &o_weight,
            HIDDEN_DIM,
            Q_DIM,
            &round_f16(&attn_out),
            SEQ_LEN,
        );
        let post_attn = rms_norm_f64(&o_proj, &post_attn_norm_w, POST_NORM_EPS, SEQ_LEN);
        let residual_1 = hidden
            .iter()
            .zip(&post_attn)
            .map(|(&hidden, &residual)| hidden + residual)
            .collect::<Vec<_>>();
        let pre_ffn = rms_norm_f64(&residual_1, &ffn_norm_w, NORM_EPS, SEQ_LEN);
        let pre_ffn_f16 = round_f16(&pre_ffn);
        let gate = crate::tests_fixture::q4k_gemm_reference(
            &gate_weight,
            FFN_DIM,
            HIDDEN_DIM,
            &pre_ffn_f16,
            SEQ_LEN,
        );
        let up = crate::tests_fixture::q4k_gemm_reference(
            &up_weight,
            FFN_DIM,
            HIDDEN_DIM,
            &pre_ffn_f16,
            SEQ_LEN,
        );
        let activation = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| half::f16::from_f32((gate / (1.0 + (-gate).exp())) * up).to_f32())
            .collect::<Vec<_>>();
        let down = crate::tests_fixture::q4k_gemm_reference(
            &down_weight,
            HIDDEN_DIM,
            FFN_DIM,
            &activation,
            SEQ_LEN,
        );
        let post_ffn = rms_norm_f64(&down, &post_ffn_norm_w, POST_NORM_EPS, SEQ_LEN);
        let expected = residual_1
            .iter()
            .zip(&post_ffn)
            .map(|(&hidden, &residual)| hidden + residual)
            .collect::<Vec<_>>();

        let post_attn_norm_error = max_abs_diff(
            &metal_rms_norm(&ctx, &o_proj, &post_attn_norm_w, POST_NORM_EPS, SEQ_LEN),
            &post_attn,
        );
        let pre_ffn_norm_error = max_abs_diff(
            &metal_rms_norm(&ctx, &residual_1, &ffn_norm_w, NORM_EPS, SEQ_LEN),
            &pre_ffn,
        );
        let post_ffn_norm_error = max_abs_diff(
            &metal_rms_norm(&ctx, &down, &post_ffn_norm_w, POST_NORM_EPS, SEQ_LEN),
            &post_ffn,
        );
        assert!(
            post_attn_norm_error < 2.0e-6,
            "post-attention norm max_abs={post_attn_norm_error}"
        );
        assert!(
            pre_ffn_norm_error < 2.0e-6,
            "pre-FFN norm max_abs={pre_ffn_norm_error}"
        );
        assert!(
            post_ffn_norm_error < 2.0e-6,
            "post-FFN norm max_abs={post_ffn_norm_error}"
        );

        let shared_bytes = |data: &[u8]| unsafe {
            let ptr = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
                .expect("weight pointer");
            ctx.device
                .newBufferWithBytes_length_options(
                    ptr,
                    data.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("weight buffer")
        };
        let o_weight_dev = shared_bytes(&o_weight);
        let gate_weight_dev = shared_bytes(&gate_weight);
        let up_weight_dev = shared_bytes(&up_weight);
        let down_weight_dev = shared_bytes(&down_weight);
        let carrier = MusePrefillOTailFfnCarrier::new(
            &ctx,
            SEQ_LEN,
            Q_DIM,
            HIDDEN_DIM,
            FFN_DIM,
            NORM_EPS,
            POST_NORM_EPS,
        );
        let actual = prefill_muse_o_tail_ffn_dispatch(
            &ctx,
            &carrier,
            MusePrefillOTailFfnDispatchRequest {
                attn_out: &attn_out,
                ops: MuseOTailFfnOpsRequest {
                    hidden: &hidden,
                    post_attn_norm_w: &post_attn_norm_w,
                    ffn_norm_w: &ffn_norm_w,
                    post_ffn_norm_w: &post_ffn_norm_w,
                    o_w_buf: &o_weight_dev,
                    o_w_off: 0,
                    o_quant: TensoropsQuant::Q4K,
                    ffn_gate_w_buf: &gate_weight_dev,
                    ffn_gate_w_off: 0,
                    ffn_gate_quant: TensoropsQuant::Q4K,
                    ffn_up_w_buf: &up_weight_dev,
                    ffn_up_w_off: 0,
                    ffn_up_quant: TensoropsQuant::Q4K,
                    ffn_down_w_buf: &down_weight_dev,
                    ffn_down_w_off: 0,
                    ffn_down_quant: TensoropsQuant::Q4K,
                },
            },
        )
        .expect("Muse O+FFN dispatch");
        let full_chain_error = max_abs_diff(&actual, &expected);
        assert!(
            full_chain_error < 5.0e-3,
            "full chain max_abs={full_chain_error}"
        );
        eprintln!(
            "Muse O+FFN oracle: post_attn={post_attn_norm_error:.9e} \
             pre_ffn={pre_ffn_norm_error:.9e} post_ffn={post_ffn_norm_error:.9e} \
             full={full_chain_error:.9e}"
        );
    }
}
