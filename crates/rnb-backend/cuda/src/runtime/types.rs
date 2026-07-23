use super::*;
use rnb_memory::{ExpertBundleCacheStats, ExpertBundleReuseHistory, SparseExpertCacheKey};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum DeviceTensorStorage {
    Owned,
    NemotronWorkspace {
        arena_id: u64,
        offset: usize,
        bytes: usize,
    },
}

impl DeviceTensorStorage {
    pub(super) fn is_owned(self) -> bool {
        matches!(self, Self::Owned)
    }

    #[allow(dead_code)]
    pub(super) fn workspace_arena_id(self) -> Option<u64> {
        match self {
            Self::Owned => None,
            Self::NemotronWorkspace { arena_id, .. } => Some(arena_id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::runtime) enum NemotronRoutePackStorage {
    Owned,
    Workspace { arena_id: u64 },
}

impl NemotronRoutePackStorage {
    #[allow(dead_code)]
    pub(in crate::runtime) fn workspace_arena_id(self) -> Option<u64> {
        match self {
            Self::Owned => None,
            Self::Workspace { arena_id } => Some(arena_id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronPrefillWorkspaceConfig {
    pub hidden_bytes: usize,
    pub normalized_bytes: usize,
    pub router_logits_bytes: usize,
    pub route_bytes: usize,
    pub moe_shared_mid_bytes: usize,
    pub moe_sparse_mid_bytes: usize,
    pub required_workspace_bytes: usize,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronPrefillWorkspaceSummary {
    pub active: bool,
    pub arena_bytes: usize,
    pub live_leases: usize,
    pub hit_bytes: usize,
    pub miss_bytes: usize,
    pub owned_alloc_count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaWeightResidencyCounters {
    pub expanded_diag_bytes: u64,
    pub native_f32_bytes: u64,
    pub packed_q8dot_bytes: u64,
    pub raw_quant_bytes: u64,
    pub transient_quant_upload_bytes: u64,
    pub q4_expanded_f16_bytes: u64,
    pub q4_expanded_f32_bytes: u64,
    pub q6_expanded_f16_bytes: u64,
    pub q6_expanded_f32_bytes: u64,
    pub q4_raw_quant_bytes: u64,
    pub q6_raw_quant_bytes: u64,
    pub q4_transient_quant_upload_bytes: u64,
    pub q6_transient_quant_upload_bytes: u64,
}

impl CudaWeightResidencyCounters {
    pub fn delta(self, before: Self) -> Self {
        Self {
            expanded_diag_bytes: self
                .expanded_diag_bytes
                .saturating_sub(before.expanded_diag_bytes),
            native_f32_bytes: self
                .native_f32_bytes
                .saturating_sub(before.native_f32_bytes),
            packed_q8dot_bytes: self
                .packed_q8dot_bytes
                .saturating_sub(before.packed_q8dot_bytes),
            raw_quant_bytes: self.raw_quant_bytes.saturating_sub(before.raw_quant_bytes),
            transient_quant_upload_bytes: self
                .transient_quant_upload_bytes
                .saturating_sub(before.transient_quant_upload_bytes),
            q4_expanded_f16_bytes: self
                .q4_expanded_f16_bytes
                .saturating_sub(before.q4_expanded_f16_bytes),
            q4_expanded_f32_bytes: self
                .q4_expanded_f32_bytes
                .saturating_sub(before.q4_expanded_f32_bytes),
            q6_expanded_f16_bytes: self
                .q6_expanded_f16_bytes
                .saturating_sub(before.q6_expanded_f16_bytes),
            q6_expanded_f32_bytes: self
                .q6_expanded_f32_bytes
                .saturating_sub(before.q6_expanded_f32_bytes),
            q4_raw_quant_bytes: self
                .q4_raw_quant_bytes
                .saturating_sub(before.q4_raw_quant_bytes),
            q6_raw_quant_bytes: self
                .q6_raw_quant_bytes
                .saturating_sub(before.q6_raw_quant_bytes),
            q4_transient_quant_upload_bytes: self
                .q4_transient_quant_upload_bytes
                .saturating_sub(before.q4_transient_quant_upload_bytes),
            q6_transient_quant_upload_bytes: self
                .q6_transient_quant_upload_bytes
                .saturating_sub(before.q6_transient_quant_upload_bytes),
        }
    }

    pub fn record_q4_expanded_f16(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q4_expanded_f16_bytes = self.q4_expanded_f16_bytes.saturating_add(bytes);
        self.expanded_diag_bytes = self.expanded_diag_bytes.saturating_add(bytes);
    }

    pub fn record_q4_expanded_f32(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q4_expanded_f32_bytes = self.q4_expanded_f32_bytes.saturating_add(bytes);
        self.expanded_diag_bytes = self.expanded_diag_bytes.saturating_add(bytes);
    }

    pub fn record_q6_expanded_f16(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q6_expanded_f16_bytes = self.q6_expanded_f16_bytes.saturating_add(bytes);
        self.expanded_diag_bytes = self.expanded_diag_bytes.saturating_add(bytes);
    }

    pub fn record_q6_expanded_f32(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q6_expanded_f32_bytes = self.q6_expanded_f32_bytes.saturating_add(bytes);
        self.expanded_diag_bytes = self.expanded_diag_bytes.saturating_add(bytes);
    }

    pub fn record_native_f32(&mut self, bytes: usize) {
        self.native_f32_bytes = self.native_f32_bytes.saturating_add(bytes as u64);
    }

    pub fn record_packed_q8dot(&mut self, bytes: usize) {
        self.packed_q8dot_bytes = self.packed_q8dot_bytes.saturating_add(bytes as u64);
    }

    pub fn record_q4_raw_quant(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q4_raw_quant_bytes = self.q4_raw_quant_bytes.saturating_add(bytes);
        self.raw_quant_bytes = self.raw_quant_bytes.saturating_add(bytes);
    }

    pub fn record_q6_raw_quant(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q6_raw_quant_bytes = self.q6_raw_quant_bytes.saturating_add(bytes);
        self.raw_quant_bytes = self.raw_quant_bytes.saturating_add(bytes);
    }

    pub fn record_q4_transient_quant_upload(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q4_transient_quant_upload_bytes =
            self.q4_transient_quant_upload_bytes.saturating_add(bytes);
        self.transient_quant_upload_bytes = self.transient_quant_upload_bytes.saturating_add(bytes);
    }

    pub fn record_q6_transient_quant_upload(&mut self, bytes: usize) {
        let bytes = bytes as u64;
        self.q6_transient_quant_upload_bytes =
            self.q6_transient_quant_upload_bytes.saturating_add(bytes);
        self.transient_quant_upload_bytes = self.transient_quant_upload_bytes.saturating_add(bytes);
    }

    #[cfg(test)]
    pub fn record_q4_expanded_f16_for_test(&mut self, bytes: usize) {
        self.record_q4_expanded_f16(bytes);
    }

    #[cfg(test)]
    pub fn record_q4_expanded_f32_for_test(&mut self, bytes: usize) {
        self.record_q4_expanded_f32(bytes);
    }

    #[cfg(test)]
    pub fn record_q6_expanded_f16_for_test(&mut self, bytes: usize) {
        self.record_q6_expanded_f16(bytes);
    }

    #[cfg(test)]
    pub fn record_q6_expanded_f32_for_test(&mut self, bytes: usize) {
        self.record_q6_expanded_f32(bytes);
    }

    #[cfg(test)]
    pub fn record_native_f32_for_test(&mut self, bytes: usize) {
        self.record_native_f32(bytes);
    }

    #[cfg(test)]
    pub fn record_packed_q8dot_for_test(&mut self, bytes: usize) {
        self.record_packed_q8dot(bytes);
    }
}

#[derive(Debug)]
/// Reusable device buffers for the persistent decode dispatch.
/// Allocated lazily on first use and reused across all subsequent decode
/// tokens to avoid the per-call mem_alloc / mem_free churn that dominates
/// wall time on small models (Gemma4 E2B).  Per-layer KV cache slots are
/// allocated once at the configured `max_seq_len * kv_dim_max * 2` size and
/// held for the lifetime of the CudaState.
pub(super) struct PersistentDecodeReusable {
    pub(super) num_layers: usize,
    pub(super) max_seq_len: u32,
    // cu104: batch-prefill scratch capacity (token slots for hidden/q_buf/
    // attn_out/gate/up/ffn batch buffers). Sized to the actual batch seq_len,
    // NOT max_seq_len (= max_ctx) — otherwise a 47-token prefill over-allocates
    // 4096-slot buffers (gate/up alone = 4096×n_ff×4 ≈ 200MB) and OOMs on large
    // context models. KV cache stays on max_seq_len (needs full context).
    pub(super) batch_seq_cap: u32,
    pub(super) kv_dim_max: u32,
    pub(super) q_dim_max: u32,
    pub(super) n_ff_max: u32,
    pub(super) hidden_dim: u32,
    pub(super) vocab_size: u32,
    pub(super) ple_dim: u32,
    pub(super) k_cache_devs: Vec<u64>,
    pub(super) v_cache_devs: Vec<u64>,
    /// cu76: track last dispatch's rope_pos to detect new prompt.
    /// If next rope_pos != last_rope_pos + 1, re-upload host KV (new sequence).
    /// `None` = first dispatch (must upload).
    pub(super) last_rope_pos: Option<u32>,
    pub(super) resident_kv_tokens: usize,
    pub(super) layers_dev: u64,
    pub(super) hidden_dev: u64,
    pub(super) normed_dev: u64,
    pub(super) attn_out_dev: u64,
    pub(super) q_buf_dev: u64,
    pub(super) k_buf_dev: u64,
    pub(super) v_buf_dev: u64,
    pub(super) gate_buf_dev: u64,
    pub(super) up_buf_dev: u64,
    pub(super) ple_gate_buf_dev: u64,
    // cu102 M4 batch FFN: batch-slot buffers (hidden_slots × dim). ffn_normed =
    // ffn_norm output (gate/up input), ffn_down = down output. gate_buf/up_buf
    // are reallocated to hidden_slots × n_ff_max for the batch FFN phase.
    pub(super) ffn_normed_dev: u64,
    pub(super) ffn_down_dev: u64,
    // cu105 QKV batch-tiling: attn_norm output as batch slots (hidden_slots ×
    // hidden_dim) = Q/K/V projection input for the batch GEMM. k_buf/v_buf are
    // also reallocated to hidden_slots × kv_dim_max so phase A3 reads each
    // token's K/V after the batch projection.
    pub(super) attn_normed_dev: u64,
    pub(super) logits_dev: u64,
    pub(super) argmax_dev: u64,
}

pub(super) struct DeviceTensorSlot {
    pub(super) ptr: u64,
    pub(super) capacity: usize,
    pub(super) desc: rnb_backend_api::DeviceTensorDesc,
    pub(super) storage: DeviceTensorStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NemotronWorkspaceSlice {
    pub(super) offset: usize,
    pub(super) bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NemotronPrefillWorkspaceLayout {
    pub(super) hidden_a: NemotronWorkspaceSlice,
    pub(super) hidden_b: NemotronWorkspaceSlice,
    pub(super) normalized: NemotronWorkspaceSlice,
    pub(super) router_logits: NemotronWorkspaceSlice,
    pub(super) route_pack: NemotronWorkspaceSlice,
    pub(super) moe_shared_mid: NemotronWorkspaceSlice,
    pub(super) moe_sparse_mid: NemotronWorkspaceSlice,
    pub(super) total_bytes: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct NemotronPrefillWorkspaceStats {
    pub(super) hit_bytes: usize,
    pub(super) miss_bytes: usize,
    pub(super) owned_alloc_count: usize,
    pub(super) live_leases: usize,
}

#[derive(Debug)]
pub(super) struct NemotronPrefillWorkspaceArena {
    pub(super) id: u64,
    pub(super) ptr: u64,
    #[allow(dead_code)]
    pub(super) capacity: usize,
    #[allow(dead_code)]
    pub(super) active: bool,
    #[allow(dead_code)]
    pub(super) layout: NemotronPrefillWorkspaceLayout,
    pub(super) stats: NemotronPrefillWorkspaceStats,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Qwen35DecodeFailurePoint {
    AdmissionAfterObservation,
    ResidualBeforeSelectedObservation,
    CombinedShared,
    CombinedAdd,
    CombinedDtoh,
    CombinedSync,
}

pub(super) struct CudaState {
    pub(super) lib_handle: usize,
    pub(super) ctx: usize,
    pub(super) stream: usize,
    pub(super) copy_stream: usize,
    pub(super) api: CudaApi,
    pub(super) device_residency_plan: rnb_memory::DeviceResidencyPlan,
    pub(super) cublas: Option<CublasState>,
    pub(super) device_tensors: HashMap<u64, DeviceTensorSlot>,
    pub(super) next_device_tensor_id: u64,
    pub(super) nemotron_prefill_workspace: Option<NemotronPrefillWorkspaceArena>,
    #[allow(dead_code)]
    pub(super) next_nemotron_prefill_workspace_id: u64,
    pub(super) compute_weights: Option<u64>,
    pub(super) compute_weights_capacity: usize,
    pub(super) transient_q4_f16_pool: Vec<TransientQ4F16Slot>,
    pub(super) transient_q4_f16_pool_cursor: usize,
    pub(super) resident_q4k: HashMap<(usize, usize), ResidentQ4k>,
    pub(super) resident_q4k_slabs: HashMap<u64, ResidentQ4kSlab>,
    pub(super) resident_q4k_lru: VecDeque<((usize, usize), u64)>,
    pub(super) resident_q4k_epoch: u64,
    pub(super) resident_q4k_bytes: usize,
    pub(super) resident_q4k_limit: usize,
    pub(super) qwen35_target_decode_q4k_limit_checked: bool,
    pub(super) nemotron_decode_q4k_limit_checked: bool,
    pub(super) resident_q4k_touch_hits_auto: bool,
    pub(super) qwen35_selected_base_admission_history: HashMap<(usize, usize), u32>,
    pub(super) qwen35_expert_bundle_reuse_history: ExpertBundleReuseHistory,
    pub(super) qwen35_q2q3_bundle_ownership: ResidentQ4kBundleOwnership,
    pub(super) qwen35_q2q3_resident_payload_bytes: u64,
    #[cfg(test)]
    pub(super) qwen35_resident_alloc_ooms_remaining: usize,
    #[cfg(test)]
    pub(super) qwen35_decode_failure_point: Option<Qwen35DecodeFailurePoint>,
    pub(super) resident_q4k_arena: Option<u64>,
    pub(super) resident_q4k_arena_capacity: usize,
    pub(super) resident_q4k_arena_offset: usize,
    pub(super) resident_q8_f32: HashMap<Q8F32Key, ResidentQ8F32>,
    pub(super) resident_q8_f32_lru: VecDeque<(Q8F32Key, u64)>,
    pub(super) resident_q8_f32_epoch: u64,
    pub(super) resident_q8_f32_bytes: usize,
    pub(super) resident_q8_f32_limit: usize,
    pub(super) resident_q8_quant: HashMap<Q8F32Key, ResidentQ8F32>,
    pub(super) resident_q4_packed: HashMap<Q4PackedKey, ResidentQ4Packed>,
    pub(super) resident_q4_packed_bytes: usize,
    pub(super) resident_q4_packed_limit: usize,
    pub(super) resident_q4_f32: HashMap<Q4F32Key, ResidentQ4F32>,
    pub(super) resident_q4_f32_bytes: usize,
    pub(super) resident_q4_f32_limit: usize,
    pub(super) resident_q6_packed: HashMap<Q6PackedKey, ResidentQ6Packed>,
    pub(super) resident_q6_packed_bytes: usize,
    pub(super) resident_q6_packed_limit: usize,
    pub(super) resident_q6_f32: HashMap<Q6F32Key, ResidentQ6F32>,
    pub(super) resident_q6_f32_bytes: usize,
    pub(super) resident_q6_f32_limit: usize,
    pub(super) resident_q6_f16: HashMap<Q6F16Key, ResidentQ6F16>,
    pub(super) resident_q6_f16_bytes: usize,
    pub(super) resident_q6_f16_limit: usize,
    pub(super) weight_residency_counters: CudaWeightResidencyCounters,
    pub(super) resident_f32: HashMap<F32Key, ResidentF32>,
    pub(super) resident_rope_tables: HashMap<RopeTableKey, ResidentRopeTable>,
    pub(super) resident_moe_layers: HashMap<Qwen35MoeLayerKey, ResidentMoeLayer>,
    pub(super) resident_moe_layer_lru: VecDeque<(Qwen35MoeLayerKey, u64)>,
    pub(super) resident_moe_layer_epoch: u64,
    pub(super) resident_moe_layer_bytes: usize,
    pub(super) resident_moe_layer_limit: usize,
    pub(super) qwen35_moe_model_bytes: usize,
    pub(super) qwen35_moe_min_layer_bytes: usize,
    pub(super) qwen35_moe_layer_cache_enabled: bool,
    pub(super) compute_input: Option<u64>,
    pub(super) compute_input_capacity: usize,
    pub(super) compute_output: Option<u64>,
    pub(super) compute_output_capacity: usize,
    pub(super) compute_aux_output: Option<u64>,
    pub(super) compute_aux_output_capacity: usize,
    pub(super) compute_mid_a: Option<u64>,
    pub(super) compute_mid_a_capacity: usize,
    pub(super) compute_mid_b: Option<u64>,
    pub(super) compute_mid_b_capacity: usize,
    pub(super) compute_gate_ptrs: Option<u64>,
    pub(super) compute_gate_ptrs_capacity: usize,
    pub(super) compute_up_ptrs: Option<u64>,
    pub(super) compute_up_ptrs_capacity: usize,
    pub(super) compute_down_ptrs: Option<u64>,
    pub(super) compute_down_ptrs_capacity: usize,
    pub(super) compute_full_gate: Option<u64>,
    pub(super) compute_full_gate_capacity: usize,
    pub(super) compute_full_up: Option<u64>,
    pub(super) compute_full_up_capacity: usize,
    pub(super) compute_full_down: Option<u64>,
    pub(super) compute_full_down_capacity: usize,
    pub(super) compute_temp_slab: Option<u64>,
    pub(super) compute_temp_slab_capacity: usize,
    pub(super) qwen35_selected_base_stream_slab_a: Option<u64>,
    pub(super) qwen35_selected_base_stream_slab_a_capacity: usize,
    pub(super) qwen35_selected_base_stream_slab_b: Option<u64>,
    pub(super) qwen35_selected_base_stream_slab_b_capacity: usize,
    pub(super) qwen35_selected_base_temp_slab_cache: Option<Qwen35SelectedBaseTempSlabCache>,
    #[cfg(test)]
    pub(super) last_qwen35_selected_sparse_boundary_stats:
        Option<Qwen35SelectedSparseBoundaryStats>,
    pub(super) qwen35_packed_act: Option<u64>,
    pub(super) qwen35_packed_act_capacity: usize,
    // cu29 Phase 2: hd128 RoPE-only QKV path 의 GPU output buffers.
    // Q4K QKV GEMV → q_dev/k_dev/v_dev (f32) → RoPE kernel → 아래 3개로 write.
    // q_rope: f32 (attention input 그대로), k/v_bits: f16 packed (KV cache 용).
    pub(super) compute_q_rope_out: Option<u64>,
    pub(super) compute_q_rope_out_capacity: usize,
    pub(super) compute_k_bits_out: Option<u64>,
    pub(super) compute_k_bits_out_capacity: usize,
    pub(super) compute_v_bits_out: Option<u64>,
    pub(super) compute_v_bits_out_capacity: usize,
    // cu41 Phase 1: decode loop 의 device-resident hidden state carrier (raw
    // hidden 용 — chain function 의 hidden_carrier_dev). attention/gdn 와 공유되는
    // compute_full_gate cache 와 분리, chain call 사이에 살아남음.
    pub(super) decode_hidden_carrier: Option<u64>,
    pub(super) decode_hidden_carrier_capacity: usize,
    // cu41 step 8: RMS norm 전용 input buffer — compute_input cache 와 분리 (다음
    // op 의 alloc 으로 overwrite race 회피).
    pub(super) decode_rms_input: Option<u64>,
    pub(super) decode_rms_input_capacity: usize,
    // cu42 step 12: RMS norm 결과 (norm_buf) 용 별도 carrier. QKV/q4k_gemv 의
    // device input 으로 사용. hidden carrier 와 분리 — raw hidden 과 norm 결과 가
    // 다른 데이터라.
    pub(super) decode_norm_buf_carrier: Option<u64>,
    pub(super) decode_norm_buf_carrier_capacity: usize,
    // cu47 step 32: attention forward 의 device output buffer. attention_decode_cached
    // 의 결과를 host scratch.attn_out 대신 device 에 유지. chain function 의 attn_out
    // H2D round-trip 제거.
    pub(super) decode_attn_out_carrier: Option<u64>,
    pub(super) decode_attn_out_carrier_capacity: usize,
    // cu49 step 38: K/V projection device output buffer. K/V projection 결과를
    // device 에 유지 → KV cache 의 host f16 변환 + H2D round-trip 제거.
    pub(super) decode_k_carrier: Option<u64>,
    pub(super) decode_k_carrier_capacity: usize,
    pub(super) decode_v_carrier: Option<u64>,
    pub(super) decode_v_carrier_capacity: usize,
    // cu52 step 47: K/V projection f32 → f16 변환 후 device buffer. KV cache 의
    // attention compute 가 f16 input 받으므로 변환 필요.
    pub(super) decode_k_f16_carrier: Option<u64>,
    pub(super) decode_k_f16_carrier_capacity: usize,
    pub(super) decode_v_f16_carrier: Option<u64>,
    pub(super) decode_v_f16_carrier_capacity: usize,
    pub(super) decode_q_carrier: Option<u64>,
    pub(super) decode_q_carrier_capacity: usize,
    pub(super) cu65_graph_pos: Option<u64>,
    pub(super) cu65_graph_pos_capacity: usize,
    pub(super) cu68_graph_kv_len: Option<u64>,
    pub(super) cu68_graph_kv_len_capacity: usize,
    pub(super) pending_nemotron_prefill_sparse: Option<PendingNemotronPrefillSparse>,
    pub(super) host_temp_slab: Option<usize>,
    pub(super) host_temp_slab_capacity: usize,
    pub(super) direct_file_reader: rnb_memory::moe_cold_io::DirectFileReaderCache,
    pub(super) host_sparse_input_slab: Option<usize>,
    pub(super) host_sparse_input_slab_capacity: usize,
    pub(super) registered_host_ranges: Vec<RegisteredHostRange>,
    pub(super) compute_route: Option<u64>,
    pub(super) compute_route_capacity: usize,
    pub(super) compute_token_ids: Option<u64>,
    pub(super) compute_token_ids_capacity: usize,
    pub(super) compute_group_meta: Option<u64>,
    pub(super) compute_group_meta_capacity: usize,
    pub(super) gemma_ple_base: Option<u64>,
    pub(super) gemma_ple_base_capacity: usize,
    pub(super) gemma_ple_base_len: usize,
    pub(super) mtp_verify_token_ids: Option<u64>,
    pub(super) mtp_verify_token_ids_capacity: usize,
    pub(super) mtp_verify_target_tokens: Option<u64>,
    pub(super) mtp_verify_target_tokens_capacity: usize,
    pub(super) mtp_verify_hidden_rows: Option<u64>,
    pub(super) mtp_verify_hidden_rows_capacity: usize,
    pub(super) mtp_verify_scratch_hidden: Option<u64>,
    pub(super) mtp_verify_scratch_hidden_capacity: usize,
    pub(super) mtp_verify_prefix_indices: Option<u64>,
    pub(super) mtp_verify_prefix_indices_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_qkv: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_qkv_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_gate: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_gate_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_alpha: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_alpha_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_beta: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_beta_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_conv_state: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_conv_state_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_conv_input: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_conv_input_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_conv_out: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_conv_out_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_q: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_q_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_k: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_k_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_v: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_v_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_gate: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_gate_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_beta: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_beta_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_out: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_delta_out_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_k_f32: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_k_f32_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_v_f32: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_v_f32_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_q_compact: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_q_compact_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_gate: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_gate_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_prior_kv: Vec<MtpVerifyAttentionPriorKvCache>,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_shared_window_kv: Vec<MtpVerifyAttentionSharedWindowKvCache>,
    pub(super) decode_attention_kv: HashMap<usize, DecodeAttentionKvCache>,
    pub(super) decode_attention_kvarn: HashMap<usize, KvarnDecodeAttentionCache>,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_out: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_attention_out_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_gated: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_gated_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_ssm_out: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_gdn_ssm_out_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_logits: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_logits_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_expert_ids: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_expert_ids_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_route_weights: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_route_weights_capacity: usize,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_token_ids: Option<u64>,
    #[allow(dead_code)]
    pub(super) mtp_verify_router_token_ids_capacity: usize,
    pub(super) qwen35_mtp_expert_history: HashMap<usize, HashSet<u32>>,
    pub(super) qwen35_mtp_expert_observations: HashMap<usize, usize>,
    pub(super) resident_delta_states: HashMap<(usize, usize), ResidentDeltaState>,
    pub(super) mtp_verify_snapshot_pool: Vec<MtpVerifySnapshotPoolEntry>,
    pub(super) nemotron_decode_sparse_calls: usize,
    pub(super) q4k_gemv_module: Option<usize>,
    pub(super) nemotron_selected_module: Option<usize>,
    pub(super) persistent_decode_module: Option<usize>,
    /// Per-CudaState reusable buffers for the persistent decode path.
    /// Allocated lazily on the first dispatch and grown when shapes change.
    pub(super) persistent_decode_ctx: Option<PersistentDecodeReusable>,
    // cu62: persistent 2×u32 counter buffer (8 bytes). lazy alloc on first megakernel launch.
    pub(super) cu62_counter_dev: Option<u64>,
    pub(super) qwen35_sparse_graphs: HashMap<SparseMoeGraphKey, SparseMoeGraph>,
    pub(super) qwen35_compound_graphs: HashMap<Qwen35CompoundGraphKey, SparseMoeGraph>,
    pub(super) mtp_verify_selected_graphs: HashMap<MtpVerifySelectedGraphKey, SparseMoeGraph>,
    pub(super) mtp_verify_gdn_graph_warmed: HashSet<MtpVerifyGdnGraphKey>,
    pub(super) mtp_verify_gdn_graphs: HashMap<MtpVerifyGdnGraphKey, SparseMoeGraph>,
    pub(super) mtp_verify_attention_graph_warmed: HashSet<MtpVerifyAttentionGraphKey>,
    pub(super) mtp_verify_attention_graphs: HashMap<MtpVerifyAttentionGraphKey, SparseMoeGraph>,
    pub(super) mtp_verify_segment_graph_warmed: HashSet<MtpVerifySegmentGraphKey>,
    pub(super) mtp_verify_segment_graphs: HashMap<MtpVerifySegmentGraphKey, SparseMoeGraph>,
    pub(super) mtp_verify_segment_capture_active: bool,
    pub(super) cu65_qkv_graph_warmed: HashSet<Cu65QkvGraphKey>,
    pub(super) cu65_qkv_graphs: HashMap<Cu65QkvGraphKey, SparseMoeGraph>,
    pub(super) cu68_attention_graph_warmed: HashSet<Cu68AttentionGraphKey>,
    pub(super) cu68_attention_graphs: HashMap<Cu68AttentionGraphKey, SparseMoeGraph>,
    pub(super) dense_expert_graph_warmed: HashSet<DenseExpertGraphKey>,
    pub(super) dense_expert_graphs: HashMap<DenseExpertGraphKey, SparseMoeGraph>,
    pub(super) cu69_dense_chain_graph_warmed: HashSet<DenseChainGraphKey>,
    pub(super) cu69_dense_chain_graphs: HashMap<DenseChainGraphKey, SparseMoeGraph>,
    #[allow(dead_code)]
    pub(super) cu71_layer_segment_graph_warmed: HashSet<LayerSegmentGraphKey>,
    pub(super) cu71_layer_segment_graphs: HashMap<LayerSegmentGraphKey, SparseMoeGraph>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseTempSlabCacheKey {
    pub(super) gate_ptr: usize,
    pub(super) gate_len: usize,
    pub(super) up_ptr: usize,
    pub(super) up_len: usize,
    pub(super) down_ptr: usize,
    pub(super) down_len: usize,
    pub(super) expert_ids: Vec<u32>,
    pub(super) down_quant: u32,
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
    pub(super) range_upload: bool,
}

pub(super) struct Qwen35SelectedBaseTempSlabCache {
    pub(super) key: Qwen35SelectedBaseTempSlabCacheKey,
    pub(super) slab_dev: u64,
    pub(super) slab_bytes: usize,
    pub(super) slab_capacity: usize,
    pub(super) expert_slab_indices: Vec<u32>,
    pub(super) gate_base: u64,
    pub(super) up_base: u64,
    pub(super) down_base: u64,
    pub(super) gate_expert_bytes: usize,
    pub(super) up_expert_bytes: usize,
    pub(super) down_expert_bytes: usize,
    pub(super) group_meta2: Vec<u32>,
    pub(super) group_meta4: Vec<u32>,
    pub(super) group_meta8: Vec<u32>,
    pub(super) group_meta16: Vec<u32>,
    pub(super) group_meta32: Vec<u32>,
    pub(super) group_meta64: Vec<u32>,
    pub(super) copy_stream_upload_pending: bool,
}

pub(super) struct PendingNemotronPrefillSparse {
    pub(super) slab: u64,
    pub(super) up_keys: Vec<(usize, usize)>,
    pub(super) down_keys: Vec<(usize, usize)>,
    pub(super) up_ptrs: Vec<u64>,
    pub(super) down_ptrs: Vec<u64>,
}

#[derive(Debug, Default)]
pub(super) struct MtpVerifyAttentionPriorKvCache {
    pub(super) layer_index: usize,
    pub(super) kv_rows: usize,
    pub(super) k_bits_dev: Option<u64>,
    pub(super) k_bits_capacity: usize,
    pub(super) v_bits_dev: Option<u64>,
    pub(super) v_bits_capacity: usize,
    pub(super) cached_tokens: usize,
    pub(super) sequence_epoch: u64,
    pub(super) host_k_bits: Vec<u16>,
    pub(super) host_v_bits: Vec<u16>,
}

#[derive(Debug, Default)]
pub(super) struct MtpVerifyAttentionSharedWindowKvCache {
    pub(super) layer_index: usize,
    pub(super) kv_rows: usize,
    pub(super) k_bits_dev: Option<u64>,
    pub(super) k_bits_capacity: usize,
    pub(super) v_bits_dev: Option<u64>,
    pub(super) v_bits_capacity: usize,
    pub(super) pos_start: usize,
    pub(super) window_tokens: usize,
    pub(super) sequence_epoch: u64,
}

#[derive(Debug, Default)]
pub(super) struct DecodeAttentionKvCache {
    pub(super) kv_rows: usize,
    pub(super) k_bits_dev: Option<u64>,
    pub(super) k_bits_capacity: usize,
    pub(super) v_bits_dev: Option<u64>,
    pub(super) v_bits_capacity: usize,
    pub(super) cached_tokens: usize,
    pub(super) host_k_base: usize,
    pub(super) host_v_base: usize,
}

#[derive(Debug, Default)]
pub(super) struct KvarnDecodeAttentionCache {
    pub(super) kv_rows: usize,
    pub(super) key_bits: u8,
    pub(super) value_bits: u8,
    pub(super) group: usize,
    pub(super) sink_tokens: usize,
    pub(super) block_bytes: usize,
    pub(super) records_dev: Option<u64>,
    pub(super) records_capacity: usize,
    pub(super) uploaded_record_bytes: usize,
    pub(super) host_records_base: usize,
    pub(super) f16_dev: Option<u64>,
    pub(super) f16_capacity: usize,
    pub(super) uploaded_sink_keys: usize,
    pub(super) uploaded_sink_values: usize,
    pub(super) host_sink_k_base: usize,
    pub(super) host_sink_v_base: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SparseMoeGraphKey {
    pub(super) down_quant: u32,
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
    pub(super) selected: usize,
    pub(super) input_dev: u64,
    pub(super) gate_dev: u64,
    pub(super) up_dev: u64,
    pub(super) output_dev: u64,
    pub(super) gate_ptrs_dev: u64,
    pub(super) up_ptrs_dev: u64,
    pub(super) down_ptrs_dev: u64,
    pub(super) route_dev: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Qwen35CompoundGraphKey {
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
    pub(super) zero_output: bool,
    pub(super) output_len: usize,
    pub(super) pack_group_count: usize,
    pub(super) down_group_count: usize,
    pub(super) input_dev: u64,
    pub(super) packed_dev: u64,
    pub(super) output_dev: u64,
    pub(super) gate_ptrs_dev: u64,
    pub(super) up_ptrs_dev: u64,
    pub(super) down_ptrs_dev: u64,
    pub(super) token_ids_dev: u64,
    pub(super) gate_meta_dev: u64,
    pub(super) pack_group_offsets_dev: u64,
    pub(super) down_meta_dev: u64,
    pub(super) route_dev: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct MtpVerifySelectedGraphKey {
    pub(super) q8_gate_up: bool,
    pub(super) down_quant: u32,
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
    pub(super) selected: usize,
    pub(super) input_dev: u64,
    pub(super) hidden_dev: u64,
    pub(super) gate_dev: u64,
    pub(super) up_dev: u64,
    pub(super) output_dev: u64,
    pub(super) gate_ptrs_dev: u64,
    pub(super) up_ptrs_dev: u64,
    pub(super) down_ptrs_dev: u64,
    pub(super) route_dev: u64,
    pub(super) input_qs_dev: u64,
    pub(super) input_ds_dev: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct MtpVerifyGdnGraphKey {
    pub(super) layer_idx: usize,
    pub(super) model_weight_ptr: usize,
    pub(super) hidden_dev: u64,
    pub(super) conv_state_ptr: usize,
    pub(super) delta_state_ptr: usize,
    pub(super) q8_selected_gate_up: bool,
    pub(super) pair2_selected_gate_up: bool,
    pub(super) pair2_selected_gate_up_silu: bool,
    pub(super) pair2_selected_down: bool,
    pub(super) pair2_selected_map: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct MtpVerifyAttentionGraphKey {
    pub(super) layer_idx: usize,
    pub(super) model_weight_ptr: usize,
    pub(super) hidden_dev: u64,
    pub(super) segment: u8,
    pub(super) q8_selected_gate_up: bool,
    pub(super) pair2_selected_gate_up: bool,
    pub(super) pair2_selected_gate_up_silu: bool,
    pub(super) pair2_selected_down: bool,
    pub(super) pair2_selected_map: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::runtime) struct MtpVerifySegmentGraphKey {
    pub(in crate::runtime) first_layer_idx: usize,
    pub(in crate::runtime) layer_count: usize,
    pub(in crate::runtime) model_weight_ptr: usize,
    pub(in crate::runtime) hidden_dev: u64,
    pub(in crate::runtime) q8_selected_gate_up: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum MtpVerifySegmentGraphStep {
    Disabled,
    Warm,
    Capture,
    Replay,
}

pub(super) struct SparseMoeGraph {
    pub(super) graph: usize,
    pub(super) exec: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Cu68AttentionGraphKey {
    pub(super) layer_idx: usize,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) q_dev: u64,
    pub(super) k_dev: u64,
    pub(super) v_dev: u64,
    pub(super) output_dev: u64,
    pub(super) kv_len_dev: u64,
    pub(super) scale_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Cu65QkvGraphKey {
    pub(super) layer_idx: usize,
    pub(super) q_rows: usize,
    pub(super) kv_dim: usize,
    pub(super) n_embd: usize,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) actual_head_dim: usize,
    pub(super) rope_theta_bits: u32,
    pub(super) norm_eps_bits: u32,
    pub(super) norm_carrier_dev: u64,
    pub(super) q_dev: u64,
    pub(super) k_dev: u64,
    pub(super) v_dev: u64,
    pub(super) q_carrier_dev: u64,
    pub(super) k_f16_dev: u64,
    pub(super) v_f16_dev: u64,
    pub(super) pos_dev: u64,
    pub(super) q_weight_ptr: usize,
    pub(super) q_weight_len: usize,
    pub(super) k_weight_ptr: usize,
    pub(super) k_weight_len: usize,
    pub(super) v_weight_ptr: usize,
    pub(super) v_weight_len: usize,
    pub(super) q_norm_ptr: usize,
    pub(super) q_norm_len: usize,
    pub(super) q_norm_hash: u64,
    pub(super) k_norm_ptr: usize,
    pub(super) k_norm_len: usize,
    pub(super) k_norm_hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DenseExpertGraphKey {
    pub(super) down_quant: u32,
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
    pub(super) gelu: bool,
    pub(super) q8dot_gate_up: bool,
    pub(super) q8dot_down: bool,
    pub(super) q8_input_prequantized: bool,
    pub(super) input_dev: u64,
    pub(super) output_dev: u64,
    pub(super) gate_dev: u64,
    pub(super) up_dev: u64,
    pub(super) gate_weight: usize,
    pub(super) up_weight: usize,
    pub(super) down_weight: usize,
    pub(super) packed_gate: u64,
    pub(super) packed_up: u64,
    pub(super) packed_q4_down: u64,
    pub(super) packed_q6_down_qs: u64,
    pub(super) packed_q6_down_d_super: u64,
    pub(super) packed_q6_down_sub_scale: u64,
    pub(super) q8_qs_dev: u64,
    pub(super) q8_ds_dev: u64,
    pub(super) down_q8_qs_dev: u64,
    pub(super) down_q8_ds_dev: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DenseChainGraphKey {
    pub(super) down_quant: u32,
    pub(super) o_cols: usize,
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
    pub(super) norm_eps_bits: u32,
    pub(super) ffn_uses_gelu: bool,
    pub(super) combined_norms: bool,
    pub(super) o_q8dot: bool,
    pub(super) q8dot_gate_up: bool,
    pub(super) q8dot_down: bool,
    pub(super) q8_input_prequantized: bool,
    pub(super) has_post_attn_norm: bool,
    pub(super) has_post_ffn_norm: bool,
    pub(super) has_ple: bool,
    pub(super) has_layer_output_scale: bool,
    pub(super) ple_weight_kind: u8,
    pub(super) ple_dim: usize,
    pub(super) ple_gate_gelu: bool,
    pub(super) layer_output_scale_bits: u32,
    pub(super) unit_offset_post_attn_norm: bool,
    pub(super) unit_offset_ffn_norm: bool,
    pub(super) unit_offset_ple_norm: bool,
    pub(super) hidden_dev: u64,
    pub(super) attn_out_dev: u64,
    pub(super) ple_input_dev: u64,
    pub(super) normed_dev: u64,
    pub(super) proj_dev: u64,
    pub(super) gate_dev: u64,
    pub(super) up_dev: u64,
    pub(super) packed_gate: u64,
    pub(super) packed_up: u64,
    pub(super) packed_q4_down: u64,
    pub(super) packed_q6_down_qs: u64,
    pub(super) packed_q6_down_d_super: u64,
    pub(super) packed_q6_down_sub_scale: u64,
    pub(super) o_q8_qs_dev: u64,
    pub(super) o_q8_ds_dev: u64,
    pub(super) q8_qs_dev: u64,
    pub(super) q8_ds_dev: u64,
    pub(super) down_q8_qs_dev: u64,
    pub(super) down_q8_ds_dev: u64,
    pub(super) o_weight_dev: u64,
    pub(super) gate_weight_dev: u64,
    pub(super) up_weight_dev: u64,
    pub(super) down_weight_dev: u64,
    pub(super) ple_gate_weight_dev: u64,
    pub(super) ple_proj_weight_dev: u64,
    pub(super) o_weight: usize,
    pub(super) gate_weight: usize,
    pub(super) up_weight: usize,
    pub(super) down_weight: usize,
    pub(super) post_attn_norm_weight: usize,
    pub(super) ffn_norm_weight: usize,
    pub(super) post_ffn_norm_weight: usize,
    pub(super) ple_gate_weight: usize,
    pub(super) ple_proj_weight: usize,
    pub(super) ple_post_norm_weight: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct LayerSegmentKvBucketKey {
    pub(super) layer_idx: usize,
    pub(super) page_size: usize,
    pub(super) max_len: usize,
    pub(super) kv_row_width: usize,
    pub(super) k_identity: u64,
    pub(super) v_identity: u64,
}

#[allow(dead_code)]
impl LayerSegmentKvBucketKey {
    pub(super) fn from_bucket_view(view: rnb_backend_api::KvBucketView) -> Self {
        Self {
            layer_idx: view.layer_idx(),
            page_size: view.page_size(),
            max_len: view.max_len(),
            kv_row_width: view.kv_row_width(),
            k_identity: view.k_identity(),
            v_identity: view.v_identity(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cu71LayerSegmentGraphRuntimeContext {
    pub layer_idx: usize,
    pub q_rows: usize,
    pub kv_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub attention_scale: f32,
    pub q_quant: u32,
    pub k_quant: u32,
    pub v_quant: u32,
    pub q_weight_identity: u64,
    pub k_weight_identity: u64,
    pub v_weight_identity: u64,
    pub q_norm_hash: u64,
    pub k_norm_hash: u64,
    pub q_carrier_dev: u64,
    pub k_f16_dev: u64,
    pub v_f16_dev: u64,
    pub kv_bucket: rnb_backend_api::KvBucketView,
    pub long_kv_split_preferred: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct LayerSegmentGraphKey {
    pub(super) layer_idx: usize,
    pub(super) n_embd: usize,
    pub(super) q_rows: usize,
    pub(super) kv_dim: usize,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rope_theta_bits: u32,
    pub(super) norm_eps_bits: u32,
    pub(super) attention_scale_bits: u32,
    pub(super) q_quant: u32,
    pub(super) k_quant: u32,
    pub(super) v_quant: u32,
    pub(super) o_quant: u32,
    pub(super) gate_quant: u32,
    pub(super) up_quant: u32,
    pub(super) down_quant: u32,
    pub(super) q_carrier_dev: u64,
    pub(super) k_carrier_dev: u64,
    pub(super) v_carrier_dev: u64,
    pub(super) k_f16_dev: u64,
    pub(super) v_f16_dev: u64,
    pub(super) kv_cache_identity: u64,
    pub(super) kv_bucket: LayerSegmentKvBucketKey,
    pub(super) kv_len_dev: u64,
    pub(super) attn_out_dev: u64,
    pub(super) hidden_dev: u64,
    pub(super) normed_dev: u64,
    pub(super) proj_dev: u64,
    pub(super) gate_dev: u64,
    pub(super) up_dev: u64,
    pub(super) q_norm_hash: u64,
    pub(super) k_norm_hash: u64,
    pub(super) post_attn_norm_hash: u64,
    pub(super) ffn_norm_hash: u64,
    pub(super) post_ffn_norm_hash: u64,
    pub(super) q_weight_identity: u64,
    pub(super) k_weight_identity: u64,
    pub(super) v_weight_identity: u64,
    pub(super) o_weight_identity: u64,
    pub(super) gate_weight_identity: u64,
    pub(super) up_weight_identity: u64,
    pub(super) down_weight_identity: u64,
    pub(super) packed_gate_identity: u64,
    pub(super) packed_up_identity: u64,
    pub(super) packed_down_identity: u64,
    pub(super) global_attention: bool,
    pub(super) has_ple: bool,
    pub(super) has_layer_output_scale: bool,
    pub(super) has_post_attn_norm: bool,
    pub(super) has_post_ffn_norm: bool,
    pub(super) ffn_uses_gelu: bool,
    pub(super) q8dot_qkv: bool,
    pub(super) q8dot_o: bool,
    pub(super) q8dot_gate_up: bool,
    pub(super) q8dot_down: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Cu71LayerSegmentCaptureInputs {
    pub(super) qkv_ready: bool,
    pub(super) attention_ready: bool,
    pub(super) dense_ready: bool,
    pub(super) long_kv_split_preferred: bool,
    pub(super) would_allocate_during_capture: bool,
    pub(super) q_carrier_dev: u64,
    pub(super) k_carrier_dev: u64,
    pub(super) v_carrier_dev: u64,
    pub(super) kv_cache_identity: u64,
    pub(super) attn_out_dev: u64,
    pub(super) hidden_dev: u64,
    pub(super) dense_graph_identity: u64,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cu71LayerSegmentCaptureDecision {
    Eligible,
    Rejected(Cu71LayerSegmentCaptureRejectReason),
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cu71LayerSegmentCaptureRejectReason {
    QkvNotReady,
    AttentionNotReady,
    DenseNotReady,
    MissingStableBuffer,
    SplitKAttentionPreferred,
    CaptureWouldAllocate,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cu71LayerSegmentGraphStep {
    Disabled,
    Rejected(Cu71LayerSegmentCaptureRejectReason),
    Warm,
    Capture,
    Replay,
}

#[allow(dead_code)]
pub(super) fn cu71_layer_segment_capture_decision(
    inputs: Cu71LayerSegmentCaptureInputs,
) -> Cu71LayerSegmentCaptureDecision {
    if !inputs.qkv_ready {
        return Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::QkvNotReady,
        );
    }
    if !inputs.attention_ready {
        return Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::AttentionNotReady,
        );
    }
    if !inputs.dense_ready {
        return Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::DenseNotReady,
        );
    }
    if inputs.long_kv_split_preferred {
        return Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::SplitKAttentionPreferred,
        );
    }
    if inputs.would_allocate_during_capture {
        return Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::CaptureWouldAllocate,
        );
    }
    if inputs.q_carrier_dev == 0
        || inputs.k_carrier_dev == 0
        || inputs.v_carrier_dev == 0
        || inputs.kv_cache_identity == 0
        || inputs.attn_out_dev == 0
        || inputs.hidden_dev == 0
        || inputs.dense_graph_identity == 0
    {
        return Cu71LayerSegmentCaptureDecision::Rejected(
            Cu71LayerSegmentCaptureRejectReason::MissingStableBuffer,
        );
    }
    Cu71LayerSegmentCaptureDecision::Eligible
}

#[allow(dead_code)]
pub(super) fn cu71_layer_segment_graph_step(
    enabled: bool,
    inputs: Cu71LayerSegmentCaptureInputs,
    key: LayerSegmentGraphKey,
    warmed: &mut HashSet<LayerSegmentGraphKey>,
    captured: &HashMap<LayerSegmentGraphKey, SparseMoeGraph>,
) -> Cu71LayerSegmentGraphStep {
    if !enabled {
        return Cu71LayerSegmentGraphStep::Disabled;
    }
    if let Cu71LayerSegmentCaptureDecision::Rejected(reason) =
        cu71_layer_segment_capture_decision(inputs)
    {
        return Cu71LayerSegmentGraphStep::Rejected(reason);
    }
    if captured.contains_key(&key) {
        return Cu71LayerSegmentGraphStep::Replay;
    }
    if warmed.contains(&key) {
        return Cu71LayerSegmentGraphStep::Capture;
    }
    warmed.insert(key);
    Cu71LayerSegmentGraphStep::Warm
}

#[derive(Debug, Default)]
pub(super) struct ResidentQ4kBundleOwnership {
    pub(super) role_owners: HashMap<(usize, usize), HashSet<SparseExpertCacheKey>>,
    pub(super) bundle_roles: HashMap<SparseExpertCacheKey, HashSet<(usize, usize)>>,
    pub(super) bundle_epochs: HashMap<SparseExpertCacheKey, u64>,
    pub(super) bundle_lru: VecDeque<(SparseExpertCacheKey, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ResidentQ4kEvictionUnit {
    OwnedClosure {
        roles: Vec<(usize, usize)>,
        bundles: Vec<SparseExpertCacheKey>,
    },
    UnownedRole {
        role: (usize, usize),
    },
}

impl ResidentQ4kEvictionUnit {
    pub(super) fn roles(&self) -> &[(usize, usize)] {
        match self {
            Self::OwnedClosure { roles, .. } => roles,
            Self::UnownedRole { role } => std::slice::from_ref(role),
        }
    }

    pub(super) fn bundles(&self) -> &[SparseExpertCacheKey] {
        match self {
            Self::OwnedClosure { bundles, .. } => bundles,
            Self::UnownedRole { .. } => &[],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ResidentQ4kEvictionPlan {
    pub(super) units: Vec<ResidentQ4kEvictionUnit>,
    pub(super) reload_payload_bytes: u64,
    pub(super) releasable_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ResidentQ4kAdmissionResult {
    pub(super) uploaded: bool,
    pub(super) evictions: ExpertBundleCacheStats,
}

pub(super) struct ResidentQ4k {
    pub(super) ptr: u64,
    pub(super) bytes: usize,
    pub(super) epoch: u64,
    pub(super) owned_alloc: bool,
    pub(super) slab_base: Option<u64>,
    pub(super) pinned: bool,
}

pub(super) struct ResidentQ4kSlab {
    pub(super) bytes: usize,
    pub(super) live_entries: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RegisteredHostRange {
    pub(super) base: usize,
    pub(super) bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Q8F32Key {
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

pub(super) struct ResidentQ8F32 {
    pub(super) ptr: u64,
    pub(super) bytes: usize,
    pub(super) epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Q4PackedKey {
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) rows: usize,
    pub(super) blocks_per_row: usize,
}

pub(super) struct ResidentQ4Packed {
    pub(super) ptr: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Q4F32Key {
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) rows: usize,
    pub(super) blocks_per_row: usize,
}

pub(super) struct ResidentQ4F32 {
    pub(super) ptr: u64,
}

pub(super) struct TransientQ4F16Slot {
    pub(super) buffer: Option<u64>,
    pub(super) capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Q6PackedKey {
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) rows: usize,
    pub(super) blocks_per_row: usize,
}

pub(super) struct ResidentQ6Packed {
    pub(super) qs_ptr: u64,
    pub(super) d_super_ptr: u64,
    pub(super) sub_scale_ptr: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Q6F32Key {
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) rows: usize,
    pub(super) blocks_per_row: usize,
}

pub(super) struct ResidentQ6F32 {
    pub(super) ptr: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Q6F16Key {
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) rows: usize,
    pub(super) blocks_per_row: usize,
}

pub(super) struct ResidentQ6F16 {
    pub(super) ptr: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct F32Key {
    pub(super) ptr: usize,
    pub(super) len: usize,
    pub(super) bit_hash: u64,
}

pub(super) struct ResidentF32 {
    pub(super) ptr: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct RopeTableKey {
    pub(super) head_dim: usize,
    pub(super) seq_len: usize,
    pub(super) pos_start: usize,
    pub(super) rope_theta_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResidentRopeTable {
    pub(super) sin_ptr: u64,
    pub(super) cos_ptr: u64,
    pub(super) bytes: usize,
}

pub(super) fn f32_key(data: &[f32]) -> F32Key {
    let mut bit_hash = 0xcbf29ce484222325_u64;
    for value in data {
        bit_hash ^= value.to_bits() as u64;
        bit_hash = bit_hash.wrapping_mul(0x100000001b3);
    }
    // cu114: content(len+bit_hash) 기반 키. 이전엔 ptr 도 키에 넣어서, 같은 norm
    // weight 가 매 호출 다른 임시 slice 주소로 넘어오면 cache miss → 매 token·매 layer
    // H2D 재upload(decode H2D 5626회). 내용 같으면 주소 무관 hit 시켜 H2D 제거.
    F32Key {
        ptr: 0,
        len: data.len(),
        bit_hash,
    }
}

pub(super) fn q8_f32_key(weights: &[u8], rows: usize, cols: usize) -> Q8F32Key {
    Q8F32Key {
        ptr: weights.as_ptr() as usize,
        len: weights.len(),
        rows,
        cols,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Qwen35MoeLayerKey {
    pub(super) gate_ptr: usize,
    pub(super) gate_len: usize,
    pub(super) up_ptr: usize,
    pub(super) up_len: usize,
    pub(super) down_ptr: usize,
    pub(super) down_len: usize,
    pub(super) down_quant: u32,
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
}

pub(super) struct ResidentMoeLayer {
    pub(super) gate_base: u64,
    pub(super) up_base: u64,
    pub(super) down_base: u64,
    pub(super) ptr: u64,
    pub(super) bytes: usize,
    pub(super) epoch: u64,
}

pub(super) struct ResidentDeltaState {
    pub(super) ptr: u64,
}

pub(super) struct MtpVerifySnapshotPoolEntry {
    pub(super) ptr: u64,
    pub(super) capacity: usize,
    pub(super) in_use: bool,
}

#[derive(Debug)]
pub struct DeltaStateSnapshot {
    pub(crate) ptr: u64,
    pub(crate) bytes: usize,
    pub(crate) pool_slot: Option<usize>,
}

pub(super) struct CublasState {
    pub(super) lib_handle: usize,
    pub(super) api: CublasApi,
    pub(super) handle: usize,
}
