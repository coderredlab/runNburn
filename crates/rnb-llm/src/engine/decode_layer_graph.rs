#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) struct Cu68LayerGraphRequest {
    pub(in crate::engine) layer_graph_enabled: bool,
    pub(in crate::engine) device_qkv_enabled: bool,
    pub(in crate::engine) used_fused_hd128: bool,
    pub(in crate::engine) chain_emits_hidden_carrier: bool,
    pub(in crate::engine) rms_used_cuda: bool,
    pub(in crate::engine) has_gated_attn: bool,
    pub(in crate::engine) gemma4_reuse_q_only: bool,
    pub(in crate::engine) gemma4_attn_rot_active: bool,
    pub(in crate::engine) has_sliding_window: bool,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) enum Cu68LayerGraphDecision {
    Eligible,
    Rejected(Cu68LayerGraphRejectReason),
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) enum Cu68LayerGraphRejectReason {
    Disabled,
    DeviceQkvDisabled,
    FusedHd128,
    NoHiddenCarrier,
    HostRmsNorm,
    GatedAttention,
    ReuseQOnly,
    AttentionRotation,
    SlidingWindow,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) struct Cu69LayerGraphRequest {
    pub(in crate::engine) dense_chain_graph_enabled: bool,
    pub(in crate::engine) architecture_is_gemma4: bool,
    pub(in crate::engine) device_qkv_enabled: bool,
    pub(in crate::engine) chain_emits_hidden_carrier: bool,
    pub(in crate::engine) rms_used_cuda: bool,
    pub(in crate::engine) attn_out_on_device: bool,
    pub(in crate::engine) hidden_carrier_available: bool,
    pub(in crate::engine) skip_h2d_hidden: bool,
    pub(in crate::engine) skip_d2h_hidden: bool,
    pub(in crate::engine) has_gated_attn: bool,
    pub(in crate::engine) gemma4_reuse_q_only: bool,
    pub(in crate::engine) gemma4_attn_rot_active: bool,
    pub(in crate::engine) has_sliding_window: bool,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) enum Cu69DenseChainGraphDecision {
    Eligible,
    Rejected(Cu69DenseChainGraphRejectReason),
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) enum Cu69DenseChainGraphRejectReason {
    Disabled,
    NonGemma4,
    DeviceQkvDisabled,
    NoHiddenCarrier,
    HostRmsNorm,
    AttentionOutputOnHost,
    HostHiddenUpload,
    HostHiddenDownload,
    GatedAttention,
    ReuseQOnly,
    AttentionRotation,
    SlidingWindow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) struct Cu71LayerSegmentGraphRequest {
    pub(in crate::engine) layer_segment_graph_enabled: bool,
    pub(in crate::engine) architecture_is_gemma4: bool,
    pub(in crate::engine) device_qkv_enabled: bool,
    pub(in crate::engine) chain_emits_hidden_carrier: bool,
    pub(in crate::engine) rms_used_cuda: bool,
    pub(in crate::engine) attn_out_on_device: bool,
    pub(in crate::engine) hidden_carrier_available: bool,
    pub(in crate::engine) skip_h2d_hidden: bool,
    pub(in crate::engine) skip_d2h_hidden: bool,
    pub(in crate::engine) has_gated_attn: bool,
    pub(in crate::engine) gemma4_reuse_q_only: bool,
    pub(in crate::engine) gemma4_attn_rot_active: bool,
    pub(in crate::engine) has_sliding_window: bool,
    pub(in crate::engine) long_kv_split_preferred: bool,
    pub(in crate::engine) dense_chain_graph_supported: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) enum Cu71LayerSegmentGraphDecision {
    Eligible,
    Rejected(Cu71LayerSegmentGraphRejectReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) enum Cu71LayerSegmentGraphRejectReason {
    Disabled,
    NonGemma4,
    DeviceQkvDisabled,
    NoHiddenCarrier,
    HostRmsNorm,
    AttentionOutputOnHost,
    HostHiddenUpload,
    HostHiddenDownload,
    GatedAttention,
    ReuseQOnly,
    AttentionRotation,
    SlidingWindow,
    SplitKAttentionPreferred,
    DenseChainUnsupported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) struct Cu72HiddenPersistenceLayer {
    pub(in crate::engine) layer_idx: usize,
    pub(in crate::engine) request: Cu71LayerSegmentGraphRequest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) struct Cu72HiddenPersistenceSegment {
    pub(in crate::engine) start_layer_idx: usize,
    pub(in crate::engine) end_layer_idx: usize,
    pub(in crate::engine) layer_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::engine) struct Cu72HiddenPersistenceTraceSummary {
    pub(in crate::engine) layer_count: usize,
    pub(in crate::engine) eligible_layer_count: usize,
    pub(in crate::engine) segments: Vec<Cu72HiddenPersistenceSegment>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::engine) struct Cu72HiddenPersistenceTrace {
    layers: Vec<Cu72HiddenPersistenceLayer>,
}

impl Cu72HiddenPersistenceTrace {
    #[cfg(any(feature = "cuda", test))]
    pub(in crate::engine) fn new() -> Self {
        Self::default()
    }

    #[cfg(any(feature = "cuda", test))]
    pub(in crate::engine) fn record_layer(&mut self, layer: Cu72HiddenPersistenceLayer) {
        self.layers.push(layer);
    }

    pub(in crate::engine) fn summary(&self) -> Cu72HiddenPersistenceTraceSummary {
        let eligible_layer_count = self
            .layers
            .iter()
            .filter(|layer| {
                cu71_layer_segment_graph_decision(layer.request)
                    == Cu71LayerSegmentGraphDecision::Eligible
            })
            .count();
        Cu72HiddenPersistenceTraceSummary {
            layer_count: self.layers.len(),
            eligible_layer_count,
            segments: cu72_hidden_persistence_segments(&self.layers),
        }
    }

    pub(in crate::engine) fn emit_trace(&self, pos: usize) {
        let summary = self.summary();
        let segments = if summary.segments.is_empty() {
            "none".to_string()
        } else {
            summary
                .segments
                .iter()
                .map(|segment| {
                    format!(
                        "{}-{}:{}",
                        segment.start_layer_idx, segment.end_layer_idx, segment.layer_count
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        eprintln!(
            "[cu72 hidden-persistence] pos={pos} layers={} eligible={} segments={segments}",
            summary.layer_count, summary.eligible_layer_count
        );
    }
}

#[cfg(any(feature = "cuda", test))]
pub(in crate::engine) fn cu68_layer_graph_decision(
    request: Cu68LayerGraphRequest,
) -> Cu68LayerGraphDecision {
    if !request.layer_graph_enabled {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::Disabled);
    }
    if !request.device_qkv_enabled {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::DeviceQkvDisabled);
    }
    if request.used_fused_hd128 {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::FusedHd128);
    }
    if !request.chain_emits_hidden_carrier {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::NoHiddenCarrier);
    }
    if !request.rms_used_cuda {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::HostRmsNorm);
    }
    if request.has_gated_attn {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::GatedAttention);
    }
    if request.gemma4_reuse_q_only {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::ReuseQOnly);
    }
    if request.gemma4_attn_rot_active {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::AttentionRotation);
    }
    if request.has_sliding_window {
        return Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::SlidingWindow);
    }
    Cu68LayerGraphDecision::Eligible
}

#[cfg(any(feature = "cuda", test))]
pub(in crate::engine) fn cu69_dense_chain_graph_decision(
    request: Cu69LayerGraphRequest,
) -> Cu69DenseChainGraphDecision {
    if !request.dense_chain_graph_enabled {
        return Cu69DenseChainGraphDecision::Rejected(Cu69DenseChainGraphRejectReason::Disabled);
    }
    if !request.architecture_is_gemma4 {
        return Cu69DenseChainGraphDecision::Rejected(Cu69DenseChainGraphRejectReason::NonGemma4);
    }
    if !request.device_qkv_enabled {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::DeviceQkvDisabled,
        );
    }
    if !request.chain_emits_hidden_carrier || !request.hidden_carrier_available {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::NoHiddenCarrier,
        );
    }
    if !request.rms_used_cuda {
        return Cu69DenseChainGraphDecision::Rejected(Cu69DenseChainGraphRejectReason::HostRmsNorm);
    }
    if !request.attn_out_on_device {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::AttentionOutputOnHost,
        );
    }
    if !request.skip_h2d_hidden {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::HostHiddenUpload,
        );
    }
    if !request.skip_d2h_hidden {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::HostHiddenDownload,
        );
    }
    if request.has_gated_attn {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::GatedAttention,
        );
    }
    if request.gemma4_reuse_q_only {
        return Cu69DenseChainGraphDecision::Rejected(Cu69DenseChainGraphRejectReason::ReuseQOnly);
    }
    if request.gemma4_attn_rot_active {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::AttentionRotation,
        );
    }
    if request.has_sliding_window {
        return Cu69DenseChainGraphDecision::Rejected(
            Cu69DenseChainGraphRejectReason::SlidingWindow,
        );
    }
    Cu69DenseChainGraphDecision::Eligible
}

pub(in crate::engine) fn cu71_layer_segment_graph_decision(
    request: Cu71LayerSegmentGraphRequest,
) -> Cu71LayerSegmentGraphDecision {
    if !request.layer_segment_graph_enabled {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::Disabled,
        );
    }
    if !request.architecture_is_gemma4 {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::NonGemma4,
        );
    }
    if !request.device_qkv_enabled {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::DeviceQkvDisabled,
        );
    }
    if !request.chain_emits_hidden_carrier || !request.hidden_carrier_available {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::NoHiddenCarrier,
        );
    }
    if !request.rms_used_cuda {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::HostRmsNorm,
        );
    }
    if !request.attn_out_on_device {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::AttentionOutputOnHost,
        );
    }
    if !request.skip_h2d_hidden {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::HostHiddenUpload,
        );
    }
    if !request.skip_d2h_hidden {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::HostHiddenDownload,
        );
    }
    if request.has_gated_attn {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::GatedAttention,
        );
    }
    if request.gemma4_reuse_q_only {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::ReuseQOnly,
        );
    }
    if request.gemma4_attn_rot_active {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::AttentionRotation,
        );
    }
    if request.has_sliding_window {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::SlidingWindow,
        );
    }
    if request.long_kv_split_preferred {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::SplitKAttentionPreferred,
        );
    }
    if !request.dense_chain_graph_supported {
        return Cu71LayerSegmentGraphDecision::Rejected(
            Cu71LayerSegmentGraphRejectReason::DenseChainUnsupported,
        );
    }
    Cu71LayerSegmentGraphDecision::Eligible
}

pub(in crate::engine) fn cu72_hidden_persistence_segments(
    layers: &[Cu72HiddenPersistenceLayer],
) -> Vec<Cu72HiddenPersistenceSegment> {
    fn flush_segment(
        out: &mut Vec<Cu72HiddenPersistenceSegment>,
        start_layer_idx: usize,
        end_layer_idx: usize,
        layer_count: usize,
    ) {
        if layer_count >= 2 {
            out.push(Cu72HiddenPersistenceSegment {
                start_layer_idx,
                end_layer_idx,
                layer_count,
            });
        }
    }

    let mut segments = Vec::new();
    let mut active_start = 0usize;
    let mut active_end = 0usize;
    let mut active_count = 0usize;

    for layer in layers {
        let eligible = cu71_layer_segment_graph_decision(layer.request)
            == Cu71LayerSegmentGraphDecision::Eligible;
        if !eligible {
            flush_segment(&mut segments, active_start, active_end, active_count);
            active_count = 0;
            continue;
        }

        if active_count == 0 {
            active_start = layer.layer_idx;
            active_end = layer.layer_idx;
            active_count = 1;
            continue;
        }

        if layer.layer_idx == active_end + 1 {
            active_end = layer.layer_idx;
            active_count += 1;
        } else {
            flush_segment(&mut segments, active_start, active_end, active_count);
            active_start = layer.layer_idx;
            active_end = layer.layer_idx;
            active_count = 1;
        }
    }

    flush_segment(&mut segments, active_start, active_end, active_count);
    segments
}

#[cfg(any(feature = "cuda", test))]
pub(in crate::engine) fn cu68_qkv_graph_enabled(
    cu65_qkv_graph_enabled: bool,
    cu68_attention_graph_enabled: bool,
) -> bool {
    cu65_qkv_graph_enabled || cu68_attention_graph_enabled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eligible_request() -> Cu68LayerGraphRequest {
        Cu68LayerGraphRequest {
            layer_graph_enabled: true,
            device_qkv_enabled: true,
            used_fused_hd128: false,
            chain_emits_hidden_carrier: true,
            rms_used_cuda: true,
            has_gated_attn: false,
            gemma4_reuse_q_only: false,
            gemma4_attn_rot_active: false,
            has_sliding_window: false,
        }
    }

    #[test]
    fn cu68_layer_graph_accepts_only_global_attention_device_chain_layers() {
        let decision = cu68_layer_graph_decision(eligible_request());

        assert_eq!(
            decision,
            Cu68LayerGraphDecision::Eligible,
            "global attention device-QKV chain layers are the only cu68 capture target"
        );
    }

    #[test]
    fn cu68_layer_graph_rejects_dynamic_or_host_mutating_branches() {
        let mut req = eligible_request();
        req.has_sliding_window = true;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::SlidingWindow)
        );

        let mut req = eligible_request();
        req.has_gated_attn = true;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::GatedAttention)
        );

        let mut req = eligible_request();
        req.gemma4_attn_rot_active = true;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::AttentionRotation)
        );

        let mut req = eligible_request();
        req.gemma4_reuse_q_only = true;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::ReuseQOnly)
        );
    }

    #[test]
    fn cu68_layer_graph_rejects_without_required_device_infra() {
        let mut req = eligible_request();
        req.layer_graph_enabled = false;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::Disabled)
        );

        let mut req = eligible_request();
        req.device_qkv_enabled = false;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::DeviceQkvDisabled)
        );

        let mut req = eligible_request();
        req.chain_emits_hidden_carrier = false;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::NoHiddenCarrier)
        );

        let mut req = eligible_request();
        req.rms_used_cuda = false;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::HostRmsNorm)
        );

        let mut req = eligible_request();
        req.used_fused_hd128 = true;
        assert_eq!(
            cu68_layer_graph_decision(req),
            Cu68LayerGraphDecision::Rejected(Cu68LayerGraphRejectReason::FusedHd128)
        );
    }

    #[test]
    fn cu68_attention_graph_also_enables_qkv_graph() {
        assert!(cu68_qkv_graph_enabled(false, true));
        assert!(cu68_qkv_graph_enabled(true, false));
        assert!(!cu68_qkv_graph_enabled(false, false));
    }

    fn cu69_eligible_dense_chain_request() -> Cu69LayerGraphRequest {
        Cu69LayerGraphRequest {
            dense_chain_graph_enabled: true,
            architecture_is_gemma4: true,
            device_qkv_enabled: true,
            chain_emits_hidden_carrier: true,
            rms_used_cuda: true,
            attn_out_on_device: true,
            hidden_carrier_available: true,
            skip_h2d_hidden: true,
            skip_d2h_hidden: true,
            has_gated_attn: false,
            gemma4_reuse_q_only: false,
            gemma4_attn_rot_active: false,
            has_sliding_window: false,
        }
    }

    #[test]
    fn cu69_dense_chain_graph_accepts_only_gemma4_device_carrier_chain() {
        assert_eq!(
            cu69_dense_chain_graph_decision(cu69_eligible_dense_chain_request()),
            Cu69DenseChainGraphDecision::Eligible
        );
    }

    #[test]
    fn cu69_dense_chain_graph_rejects_non_gemma4_architectures() {
        let mut req = cu69_eligible_dense_chain_request();
        req.architecture_is_gemma4 = false;

        assert_eq!(
            cu69_dense_chain_graph_decision(req),
            Cu69DenseChainGraphDecision::Rejected(Cu69DenseChainGraphRejectReason::NonGemma4)
        );
    }

    #[test]
    fn cu69_dense_chain_graph_rejects_without_device_carriers() {
        let mut req = cu69_eligible_dense_chain_request();
        req.attn_out_on_device = false;
        assert_eq!(
            cu69_dense_chain_graph_decision(req),
            Cu69DenseChainGraphDecision::Rejected(
                Cu69DenseChainGraphRejectReason::AttentionOutputOnHost
            )
        );

        let mut req = cu69_eligible_dense_chain_request();
        req.hidden_carrier_available = false;
        assert_eq!(
            cu69_dense_chain_graph_decision(req),
            Cu69DenseChainGraphDecision::Rejected(Cu69DenseChainGraphRejectReason::NoHiddenCarrier)
        );

        let mut req = cu69_eligible_dense_chain_request();
        req.skip_h2d_hidden = false;
        assert_eq!(
            cu69_dense_chain_graph_decision(req),
            Cu69DenseChainGraphDecision::Rejected(
                Cu69DenseChainGraphRejectReason::HostHiddenUpload
            )
        );

        let mut req = cu69_eligible_dense_chain_request();
        req.skip_d2h_hidden = false;
        assert_eq!(
            cu69_dense_chain_graph_decision(req),
            Cu69DenseChainGraphDecision::Rejected(
                Cu69DenseChainGraphRejectReason::HostHiddenDownload
            )
        );
    }

    fn cu71_eligible_layer_segment_request() -> Cu71LayerSegmentGraphRequest {
        Cu71LayerSegmentGraphRequest {
            layer_segment_graph_enabled: true,
            architecture_is_gemma4: true,
            device_qkv_enabled: true,
            chain_emits_hidden_carrier: true,
            rms_used_cuda: true,
            attn_out_on_device: true,
            hidden_carrier_available: true,
            skip_h2d_hidden: true,
            skip_d2h_hidden: true,
            has_gated_attn: false,
            gemma4_reuse_q_only: false,
            gemma4_attn_rot_active: false,
            has_sliding_window: false,
            long_kv_split_preferred: false,
            dense_chain_graph_supported: true,
        }
    }

    #[test]
    fn cu71_layer_segment_graph_accepts_gemma4_global_attention_device_segment() {
        assert_eq!(
            cu71_layer_segment_graph_decision(cu71_eligible_layer_segment_request()),
            Cu71LayerSegmentGraphDecision::Eligible
        );
    }

    #[test]
    fn cu71_layer_segment_graph_rejects_eager_island_branches() {
        let mut req = cu71_eligible_layer_segment_request();
        req.has_sliding_window = true;
        assert_eq!(
            cu71_layer_segment_graph_decision(req),
            Cu71LayerSegmentGraphDecision::Rejected(
                Cu71LayerSegmentGraphRejectReason::SlidingWindow
            )
        );

        let mut req = cu71_eligible_layer_segment_request();
        req.has_gated_attn = true;
        assert_eq!(
            cu71_layer_segment_graph_decision(req),
            Cu71LayerSegmentGraphDecision::Rejected(
                Cu71LayerSegmentGraphRejectReason::GatedAttention
            )
        );

        let mut req = cu71_eligible_layer_segment_request();
        req.gemma4_attn_rot_active = true;
        assert_eq!(
            cu71_layer_segment_graph_decision(req),
            Cu71LayerSegmentGraphDecision::Rejected(
                Cu71LayerSegmentGraphRejectReason::AttentionRotation
            )
        );

        let mut req = cu71_eligible_layer_segment_request();
        req.gemma4_reuse_q_only = true;
        assert_eq!(
            cu71_layer_segment_graph_decision(req),
            Cu71LayerSegmentGraphDecision::Rejected(Cu71LayerSegmentGraphRejectReason::ReuseQOnly)
        );
    }

    #[test]
    fn cu71_layer_segment_graph_preserves_split_k_and_dense_guards() {
        let mut req = cu71_eligible_layer_segment_request();
        req.long_kv_split_preferred = true;
        assert_eq!(
            cu71_layer_segment_graph_decision(req),
            Cu71LayerSegmentGraphDecision::Rejected(
                Cu71LayerSegmentGraphRejectReason::SplitKAttentionPreferred
            )
        );

        let mut req = cu71_eligible_layer_segment_request();
        req.dense_chain_graph_supported = false;
        assert_eq!(
            cu71_layer_segment_graph_decision(req),
            Cu71LayerSegmentGraphDecision::Rejected(
                Cu71LayerSegmentGraphRejectReason::DenseChainUnsupported
            )
        );
    }

    #[test]
    fn cu72_gemma4_hidden_persistence_plans_contiguous_global_layers_only() {
        let mut layers = vec![
            Cu72HiddenPersistenceLayer {
                layer_idx: 4,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 5,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 6,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 8,
                request: cu71_eligible_layer_segment_request(),
            },
        ];
        layers[2].request.has_sliding_window = true;

        assert_eq!(
            cu72_hidden_persistence_segments(&layers),
            vec![Cu72HiddenPersistenceSegment {
                start_layer_idx: 4,
                end_layer_idx: 5,
                layer_count: 2,
            }],
            "cu72 should keep only 2+ contiguous global-attention layers that can carry hidden on device"
        );
    }

    #[test]
    fn cu72_gemma4_hidden_persistence_breaks_on_eager_islands() {
        let mut layers = vec![
            Cu72HiddenPersistenceLayer {
                layer_idx: 10,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 11,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 12,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 13,
                request: cu71_eligible_layer_segment_request(),
            },
        ];
        layers[1].request.has_gated_attn = true;
        layers[2].request.gemma4_attn_rot_active = true;
        layers[3].request.long_kv_split_preferred = true;

        assert_eq!(
            cu72_hidden_persistence_segments(&layers),
            Vec::<Cu72HiddenPersistenceSegment>::new(),
            "dynamic/eager islands must break the hidden-persistence segment instead of being fused through"
        );
    }

    #[test]
    fn cu72_hidden_persistence_trace_records_runtime_segments() {
        let mut trace = Cu72HiddenPersistenceTrace::new();
        let mut rejected = cu71_eligible_layer_segment_request();
        rejected.has_sliding_window = true;

        for layer in [
            Cu72HiddenPersistenceLayer {
                layer_idx: 4,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 5,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 6,
                request: rejected,
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 7,
                request: cu71_eligible_layer_segment_request(),
            },
            Cu72HiddenPersistenceLayer {
                layer_idx: 8,
                request: cu71_eligible_layer_segment_request(),
            },
        ] {
            trace.record_layer(layer);
        }

        let summary = trace.summary();

        assert_eq!(summary.layer_count, 5);
        assert_eq!(summary.eligible_layer_count, 4);
        assert_eq!(
            summary.segments,
            vec![
                Cu72HiddenPersistenceSegment {
                    start_layer_idx: 4,
                    end_layer_idx: 5,
                    layer_count: 2,
                },
                Cu72HiddenPersistenceSegment {
                    start_layer_idx: 7,
                    end_layer_idx: 8,
                    layer_count: 2,
                },
            ],
            "runtime trace must feed the cu72 segment planner instead of leaving it as dead scaffold"
        );
    }
}
