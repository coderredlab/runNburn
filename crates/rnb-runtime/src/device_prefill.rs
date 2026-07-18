use crate::backend::BackendKind;
use rnb_backend_api::{DeviceTensorDesc, DeviceTensorId, DeviceTransferCounters};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePrefillSessionConfig {
    pub backend: BackendKind,
    pub seq_len: usize,
    pub hidden_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePrefillProfile {
    pub transfers: DeviceTransferCounters,
    pub transfer_breakdown: DevicePrefillTransferBreakdown,
    pub fallback_reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DevicePrefillTransferBreakdown {
    pub hidden_h2d_bytes: usize,
    pub hidden_d2h_bytes: usize,
    pub router_logits_d2h_bytes: usize,
    pub route_h2d_bytes: usize,
    pub state_d2h_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronPrefillWorkspaceRequest {
    pub seq_len: usize,
    pub hidden_dim: usize,
    pub n_expert: usize,
    pub expert_used: usize,
    pub n_ff: usize,
    pub shared_ff: usize,
    pub free_vram_bytes: usize,
    pub total_vram_bytes: usize,
    pub reserve_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronPrefillWorkspaceDecision {
    Fits,
    Chunk { chunk_len: usize },
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronPrefillWorkspacePlan {
    pub seq_len: usize,
    pub chunk_len: usize,
    pub hidden_dim: usize,
    pub n_expert: usize,
    pub expert_used: usize,
    pub n_ff: usize,
    pub shared_ff: usize,
    pub free_vram_bytes: usize,
    pub total_vram_bytes: usize,
    pub reserve_bytes: usize,
    pub usable_vram_bytes: usize,
    pub route_slots: usize,
    pub normalized_bytes: usize,
    pub router_logits_bytes: usize,
    pub route_bytes: usize,
    pub hidden_bytes: usize,
    pub persistent_hidden_bytes: usize,
    pub moe_intermediate_bytes: usize,
    pub mamba_state_sync_bytes: usize,
    pub attention_handoff_bytes: usize,
    pub required_workspace_bytes: usize,
    pub decision: NemotronPrefillWorkspaceDecision,
}

const NEMOTRON_PREFILL_WORKSPACE_MIN_CHUNK_LEN: usize = 32;
const F32_BYTES: usize = std::mem::size_of::<f32>();
const U32_BYTES: usize = std::mem::size_of::<u32>();
const ROUTE_SLOT_BYTES: usize = U32_BYTES + U32_BYTES + F32_BYTES;

pub fn plan_nemotron_prefill_workspace(
    request: NemotronPrefillWorkspaceRequest,
) -> Result<NemotronPrefillWorkspacePlan, String> {
    validate_nemotron_prefill_workspace_request(request)?;

    let usable_vram_bytes = request
        .free_vram_bytes
        .saturating_sub(request.reserve_bytes);
    let full = build_nemotron_prefill_workspace_plan(
        request,
        request.seq_len,
        usable_vram_bytes,
        NemotronPrefillWorkspaceDecision::Fits,
    )?;
    if full.required_workspace_bytes <= usable_vram_bytes {
        return Ok(full);
    }

    let mut low = 1usize;
    let mut high = request.seq_len.saturating_sub(1);
    let mut best_chunk_len = 0usize;
    while low <= high {
        let mid = low + (high - low) / 2;
        let candidate = build_nemotron_prefill_workspace_plan(
            request,
            mid,
            usable_vram_bytes,
            NemotronPrefillWorkspaceDecision::Fallback,
        )?;
        if candidate.required_workspace_bytes <= usable_vram_bytes {
            best_chunk_len = mid;
            low = mid.saturating_add(1);
        } else {
            high = mid.saturating_sub(1);
        }
    }

    let fallback_chunk_len = best_chunk_len.max(1).min(request.seq_len);
    let min_chunk_len = request
        .seq_len
        .min(NEMOTRON_PREFILL_WORKSPACE_MIN_CHUNK_LEN);
    if best_chunk_len >= min_chunk_len {
        return build_nemotron_prefill_workspace_plan(
            request,
            best_chunk_len,
            usable_vram_bytes,
            NemotronPrefillWorkspaceDecision::Chunk {
                chunk_len: best_chunk_len,
            },
        );
    }

    build_nemotron_prefill_workspace_plan(
        request,
        fallback_chunk_len,
        usable_vram_bytes,
        NemotronPrefillWorkspaceDecision::Fallback,
    )
}

fn validate_nemotron_prefill_workspace_request(
    request: NemotronPrefillWorkspaceRequest,
) -> Result<(), String> {
    for (name, value) in [
        ("seq_len", request.seq_len),
        ("hidden_dim", request.hidden_dim),
        ("n_expert", request.n_expert),
        ("expert_used", request.expert_used),
        ("n_ff", request.n_ff),
        ("shared_ff", request.shared_ff),
    ] {
        if value == 0 {
            return Err(format!(
                "{name} must be > 0 for Nemotron prefill workspace plan"
            ));
        }
    }
    if request.expert_used > request.n_expert {
        return Err(format!(
            "expert_used must be <= n_expert for Nemotron prefill workspace plan: expert_used={} n_expert={}",
            request.expert_used, request.n_expert
        ));
    }
    Ok(())
}

fn build_nemotron_prefill_workspace_plan(
    request: NemotronPrefillWorkspaceRequest,
    chunk_len: usize,
    usable_vram_bytes: usize,
    decision: NemotronPrefillWorkspaceDecision,
) -> Result<NemotronPrefillWorkspacePlan, String> {
    let route_slots = checked_mul(chunk_len, request.expert_used, "route slots")?;
    let hidden_elems = checked_mul(chunk_len, request.hidden_dim, "hidden elements")?;
    let hidden_bytes = checked_mul(hidden_elems, F32_BYTES, "hidden bytes")?;
    let normalized_bytes = hidden_bytes;
    let router_logits_elems = checked_mul(chunk_len, request.n_expert, "router logits elements")?;
    let router_logits_bytes = checked_mul(router_logits_elems, F32_BYTES, "router logits bytes")?;
    let route_bytes = checked_mul(route_slots, ROUTE_SLOT_BYTES, "route bytes")?;
    let routed_intermediate_elems =
        checked_mul(route_slots, request.n_ff, "routed intermediate elements")?;
    let routed_intermediate_bytes = checked_mul(
        routed_intermediate_elems,
        F32_BYTES,
        "routed intermediate bytes",
    )?;
    let shared_intermediate_elems =
        checked_mul(chunk_len, request.shared_ff, "shared intermediate elements")?;
    let shared_intermediate_bytes = checked_mul(
        shared_intermediate_elems,
        F32_BYTES,
        "shared intermediate bytes",
    )?;
    let moe_intermediate_bytes = checked_add(
        routed_intermediate_bytes,
        shared_intermediate_bytes,
        "moe intermediate bytes",
    )?;
    let persistent_hidden_bytes = checked_mul(hidden_bytes, 2, "persistent hidden bytes")?;
    let route_workspace_bytes = checked_add(
        checked_add(
            normalized_bytes,
            router_logits_bytes,
            "router workspace bytes",
        )?,
        route_bytes,
        "route workspace bytes",
    )?;
    let moe_workspace_bytes = checked_add(
        route_workspace_bytes,
        moe_intermediate_bytes,
        "moe workspace bytes",
    )?;

    let mamba_state_sync_bytes = 0;
    let attention_handoff_bytes = hidden_bytes;
    let max_layer_temp_bytes = moe_workspace_bytes
        .max(mamba_state_sync_bytes)
        .max(attention_handoff_bytes);
    let required_workspace_bytes = checked_add(
        persistent_hidden_bytes,
        max_layer_temp_bytes,
        "required workspace bytes",
    )?;

    Ok(NemotronPrefillWorkspacePlan {
        seq_len: request.seq_len,
        chunk_len,
        hidden_dim: request.hidden_dim,
        n_expert: request.n_expert,
        expert_used: request.expert_used,
        n_ff: request.n_ff,
        shared_ff: request.shared_ff,
        free_vram_bytes: request.free_vram_bytes,
        total_vram_bytes: request.total_vram_bytes,
        reserve_bytes: request.reserve_bytes,
        usable_vram_bytes,
        route_slots,
        normalized_bytes,
        router_logits_bytes,
        route_bytes,
        hidden_bytes,
        persistent_hidden_bytes,
        moe_intermediate_bytes,
        mamba_state_sync_bytes,
        attention_handoff_bytes,
        required_workspace_bytes,
        decision,
    })
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| format!("Nemotron prefill workspace {label} overflow: {lhs} * {rhs}"))
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
    lhs.checked_add(rhs)
        .ok_or_else(|| format!("Nemotron prefill workspace {label} overflow: {lhs} + {rhs}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePrefillSession {
    config: DevicePrefillSessionConfig,
    profile: DevicePrefillProfile,
    current_hidden: Option<DeviceTensorId>,
}

impl DevicePrefillSession {
    pub fn new(config: DevicePrefillSessionConfig) -> Result<Self, String> {
        if config.seq_len == 0 || config.hidden_dim == 0 {
            return Err(format!(
                "invalid device prefill shape: seq_len={} hidden_dim={}",
                config.seq_len, config.hidden_dim
            ));
        }
        Ok(Self {
            config,
            profile: DevicePrefillProfile {
                transfers: DeviceTransferCounters::new(),
                transfer_breakdown: DevicePrefillTransferBreakdown::default(),
                fallback_reasons: Vec::new(),
            },
            current_hidden: None,
        })
    }

    pub fn config(&self) -> &DevicePrefillSessionConfig {
        &self.config
    }

    pub fn profile(&self) -> &DevicePrefillProfile {
        &self.profile
    }

    pub fn current_hidden(&self) -> Option<DeviceTensorId> {
        self.current_hidden
    }

    pub fn set_current_hidden(
        &mut self,
        id: DeviceTensorId,
        desc: DeviceTensorDesc,
    ) -> Result<(), String> {
        if id.backend() != self.config.backend {
            return Err(format!(
                "device tensor backend mismatch: session={:?} tensor={:?}",
                self.config.backend,
                id.backend()
            ));
        }
        if desc.rows() != self.config.seq_len || desc.cols() != self.config.hidden_dim {
            return Err(format!(
                "device hidden shape mismatch: got {}x{}, expected {}x{}",
                desc.rows(),
                desc.cols(),
                self.config.seq_len,
                self.config.hidden_dim
            ));
        }
        self.current_hidden = Some(id);
        Ok(())
    }

    pub fn record_h2d(&mut self, bytes: usize) {
        self.profile.transfers.record_h2d(bytes);
    }

    pub fn record_d2h(&mut self, bytes: usize) {
        self.profile.transfers.record_d2h(bytes);
    }

    pub fn record_hidden_h2d(&mut self, bytes: usize) {
        self.profile.transfers.record_h2d(bytes);
        self.profile.transfer_breakdown.hidden_h2d_bytes = self
            .profile
            .transfer_breakdown
            .hidden_h2d_bytes
            .saturating_add(bytes);
    }

    pub fn record_hidden_d2h(&mut self, bytes: usize) {
        self.profile.transfers.record_d2h(bytes);
        self.profile.transfer_breakdown.hidden_d2h_bytes = self
            .profile
            .transfer_breakdown
            .hidden_d2h_bytes
            .saturating_add(bytes);
    }

    pub fn record_router_logits_d2h(&mut self, bytes: usize) {
        self.profile.transfers.record_d2h(bytes);
        self.profile.transfer_breakdown.router_logits_d2h_bytes = self
            .profile
            .transfer_breakdown
            .router_logits_d2h_bytes
            .saturating_add(bytes);
    }

    pub fn record_route_h2d(&mut self, bytes: usize) {
        self.profile.transfers.record_h2d(bytes);
        self.profile.transfer_breakdown.route_h2d_bytes = self
            .profile
            .transfer_breakdown
            .route_h2d_bytes
            .saturating_add(bytes);
    }

    pub fn record_state_d2h(&mut self, bytes: usize) {
        self.profile.transfers.record_d2h(bytes);
        self.profile.transfer_breakdown.state_d2h_bytes = self
            .profile
            .transfer_breakdown
            .state_d2h_bytes
            .saturating_add(bytes);
    }

    pub fn record_fallback(&mut self, reason: impl Into<String>) {
        self.profile.fallback_reasons.push(reason.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_backend_api::{DeviceTensorRole, ScalarType};

    #[test]
    fn session_tracks_current_hidden_and_transfer_bytes() {
        let mut session = DevicePrefillSession::new(DevicePrefillSessionConfig {
            backend: BackendKind::Cuda,
            seq_len: 2,
            hidden_dim: 4,
        })
        .unwrap();
        let id = DeviceTensorId::new(BackendKind::Cuda, 7);
        let desc = DeviceTensorDesc::new(2, 4, ScalarType::F32, DeviceTensorRole::Hidden);

        session.set_current_hidden(id, desc).unwrap();
        session.record_h2d(desc.byte_len().unwrap());

        assert_eq!(session.current_hidden(), Some(id));
        assert_eq!(session.profile().transfers.h2d_bytes(), 32);
    }

    #[test]
    fn session_records_transfer_breakdown() {
        let mut session = DevicePrefillSession::new(DevicePrefillSessionConfig {
            backend: BackendKind::Cuda,
            seq_len: 2,
            hidden_dim: 4,
        })
        .unwrap();

        session.record_hidden_d2h(32);
        session.record_router_logits_d2h(16);
        session.record_route_h2d(24);
        session.record_state_d2h(8);

        assert_eq!(session.profile().transfers.d2h_bytes(), 56);
        assert_eq!(session.profile().transfers.h2d_bytes(), 24);
        assert_eq!(session.profile().transfer_breakdown.hidden_d2h_bytes, 32);
        assert_eq!(
            session.profile().transfer_breakdown.router_logits_d2h_bytes,
            16
        );
        assert_eq!(session.profile().transfer_breakdown.route_h2d_bytes, 24);
        assert_eq!(session.profile().transfer_breakdown.state_d2h_bytes, 8);
    }

    #[test]
    fn session_records_fallback_reason() {
        let mut session = DevicePrefillSession::new(DevicePrefillSessionConfig {
            backend: BackendKind::Cuda,
            seq_len: 1,
            hidden_dim: 4,
        })
        .unwrap();

        session.record_fallback("unsupported mamba layer");

        assert_eq!(
            session.profile().fallback_reasons,
            vec!["unsupported mamba layer"]
        );
    }

    fn workspace_request_with_free_vram(free_vram_bytes: usize) -> NemotronPrefillWorkspaceRequest {
        NemotronPrefillWorkspaceRequest {
            seq_len: 128,
            hidden_dim: 16,
            n_expert: 8,
            expert_used: 2,
            n_ff: 64,
            shared_ff: 32,
            free_vram_bytes,
            total_vram_bytes: 10_000_000,
            reserve_bytes: 0,
        }
    }

    #[test]
    fn nemotron_prefill_workspace_plan_fits_when_usable_covers_required() {
        let request = NemotronPrefillWorkspaceRequest {
            reserve_bytes: 1_000_000,
            ..workspace_request_with_free_vram(10_000_000)
        };

        let plan = plan_nemotron_prefill_workspace(request).unwrap();

        assert_eq!(plan.decision, NemotronPrefillWorkspaceDecision::Fits);
        assert_eq!(plan.seq_len, 128);
        assert_eq!(plan.chunk_len, 128);
        assert_eq!(plan.route_slots, 256);
        assert_eq!(plan.hidden_bytes, 8192);
        assert_eq!(plan.normalized_bytes, 8192);
        assert_eq!(plan.router_logits_bytes, 4096);
        assert_eq!(plan.route_bytes, 3072);
        assert_eq!(plan.moe_intermediate_bytes, 81920);
        assert_eq!(plan.attention_handoff_bytes, 8192);
        assert_eq!(plan.persistent_hidden_bytes, 16384);
        assert_eq!(plan.required_workspace_bytes, 113664);
        assert_eq!(plan.usable_vram_bytes, 9_000_000);
    }

    #[test]
    fn nemotron_prefill_workspace_plan_chunks_when_full_sequence_does_not_fit() {
        let plan =
            plan_nemotron_prefill_workspace(workspace_request_with_free_vram(50_000)).unwrap();

        assert_eq!(
            plan.decision,
            NemotronPrefillWorkspaceDecision::Chunk { chunk_len: 56 }
        );
        assert_eq!(plan.chunk_len, 56);
        assert_eq!(plan.route_slots, 112);
        assert_eq!(plan.required_workspace_bytes, 49728);
    }

    #[test]
    fn nemotron_prefill_workspace_plan_falls_back_when_min_chunk_does_not_fit() {
        let plan =
            plan_nemotron_prefill_workspace(workspace_request_with_free_vram(20_000)).unwrap();

        assert_eq!(plan.decision, NemotronPrefillWorkspaceDecision::Fallback);
        assert_eq!(plan.chunk_len, 22);
        assert_eq!(plan.required_workspace_bytes, 19536);
    }

    #[test]
    fn nemotron_prefill_workspace_plan_rejects_zero_shape() {
        let err = plan_nemotron_prefill_workspace(NemotronPrefillWorkspaceRequest {
            seq_len: 0,
            ..workspace_request_with_free_vram(10_000_000)
        })
        .unwrap_err();

        assert!(err.contains("seq_len must be > 0"));
    }
}
