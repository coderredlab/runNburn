use super::super::*;

struct CudaLibraryGuard(Option<usize>);

impl CudaLibraryGuard {
    fn new(handle: usize) -> Self {
        Self(Some(handle))
    }

    fn handle(&self) -> usize {
        self.0.expect("CUDA library handle initialized")
    }

    fn release(mut self) -> usize {
        self.0.take().expect("CUDA library handle initialized")
    }
}

impl Drop for CudaLibraryGuard {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            unsafe {
                crate::dynlib::close(handle);
            }
        }
    }
}

struct CudaOpenResources<'a> {
    api: &'a CudaApi,
    ctx: Option<usize>,
    stream: Option<usize>,
    copy_stream: Option<usize>,
}

impl<'a> CudaOpenResources<'a> {
    fn new(api: &'a CudaApi) -> Self {
        Self {
            api,
            ctx: None,
            stream: None,
            copy_stream: None,
        }
    }

    fn release(mut self) -> (usize, usize, usize) {
        (
            self.ctx.take().expect("CUDA context initialized"),
            self.stream.take().expect("CUDA stream initialized"),
            self.copy_stream
                .take()
                .expect("CUDA copy stream initialized"),
        )
    }
}

impl Drop for CudaOpenResources<'_> {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx {
            let _ = unsafe { self.api.ctx_set_current(ctx) };
        }
        if let Some(stream) = self.copy_stream.take() {
            let _ = unsafe { self.api.stream_destroy(stream) };
        }
        if let Some(stream) = self.stream.take() {
            let _ = unsafe { self.api.stream_destroy(stream) };
        }
        if let Some(ctx) = self.ctx.take() {
            let _ = unsafe { self.api.ctx_destroy(ctx) };
        }
    }
}

impl CudaState {
    pub(in crate::runtime) fn configured_device_ordinal() -> i32 {
        std::env::var("RNB_CUDA_DEVICE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(0)
    }

    pub(in crate::runtime) fn open() -> Result<Self, String> {
        Self::open_ordinal(Self::configured_device_ordinal())
    }

    pub(in crate::runtime) fn open_ordinal(ordinal: i32) -> Result<Self, String> {
        let library = CudaLibraryGuard::new(dlopen_cuda()?);
        let api = unsafe { CudaApi::load(library.handle())? };
        let mut open_resources = CudaOpenResources::new(&api);
        unsafe { api.init(0)? };
        let device = unsafe { api.device_get(ordinal)? };
        // cu58 pre diag: visible device count + 선택된 device 의 name + PCI bus ID dump
        if std::env::var("RNB_CUDA_DIAG").ok().as_deref() == Some("1") || ordinal != 0 {
            let count = unsafe { api.device_get_count() }.unwrap_or(-1);
            let name =
                unsafe { api.device_get_name(device) }.unwrap_or_else(|_| "<unknown>".to_string());
            let pci = unsafe { api.device_get_pci_bus_id(device) }
                .unwrap_or_else(|_| "<unknown>".to_string());
            eprintln!(
                "[cuda] visible_count={count} ordinal={ordinal} device_handle={device} name=\"{name}\" pci_bus_id={pci}"
            );
        }
        let ctx = unsafe { api.ctx_create(0, device)? };
        open_resources.ctx = Some(ctx);
        unsafe { api.ctx_set_current(ctx)? };
        let (initial_free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
        let dynamic_reserve_bytes = device_residency_configured_reserve_mib(
            total_bytes / (1024 * 1024),
            mtp_device_verify_env_enabled(),
        )?
        .saturating_mul(1024 * 1024);
        let device_residency_plan = rnb_memory::DeviceResidencyPlan::from_snapshot(
            total_bytes,
            initial_free_bytes,
            dynamic_reserve_bytes,
        );
        let global_resident_limit = device_residency_plan.resident_limit_bytes;
        let resident_moe_layer_limit = moe_layer_cache_limit(&api)?.min(global_resident_limit);
        let resident_q4k_limit = q4k_resident_cache_limit(&api)?.min(global_resident_limit);
        let resident_q8_f32_limit = q8_f32_cache_limit()?.min(global_resident_limit);
        let resident_q4_packed_limit = q4_packed_cache_limit(&api)?.min(global_resident_limit);
        let resident_q4_f32_limit = q4_f32_cache_limit(&api)?.min(global_resident_limit);
        let resident_q6_packed_limit = q6_packed_cache_limit(&api)?.min(global_resident_limit);
        let resident_q6_f32_limit = q6_f32_cache_limit()?.min(global_resident_limit);
        let resident_q6_f16_limit = q6_f16_cache_limit(&api)?.min(global_resident_limit);
        if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            eprintln!(
                "[cuda] unified residency budget: total={}MiB free={}MiB dynamic_reserve={}MiB resident_limit={}MiB",
                total_bytes / (1024 * 1024),
                initial_free_bytes / (1024 * 1024),
                device_residency_plan.dynamic_reserve_bytes / (1024 * 1024),
                global_resident_limit / (1024 * 1024),
            );
        }
        let stream = unsafe { api.stream_create(0)? };
        open_resources.stream = Some(stream);
        let copy_stream = unsafe { api.stream_create(0)? };
        open_resources.copy_stream = Some(copy_stream);
        let (ctx, stream, copy_stream) = open_resources.release();
        let lib_handle = library.release();
        Ok(Self {
            lib_handle,
            ctx,
            stream,
            copy_stream,
            api,
            device_residency_plan,
            cublas: None,
            device_tensors: HashMap::new(),
            next_device_tensor_id: 1,
            nemotron_prefill_workspace: None,
            next_nemotron_prefill_workspace_id: 1,
            compute_weights: None,
            compute_weights_capacity: 0,
            transient_q4_f16_pool: Vec::new(),
            transient_q4_f16_pool_cursor: 0,
            resident_q4k: HashMap::new(),
            resident_q4k_slabs: HashMap::new(),
            resident_q4k_lru: VecDeque::new(),
            resident_q4k_epoch: 0,
            resident_q4k_bytes: 0,
            resident_q4k_limit,
            qwen35_target_decode_q4k_limit_checked: false,
            nemotron_decode_q4k_limit_checked: false,
            resident_q4k_touch_hits_auto: false,
            qwen35_selected_base_admission_history: HashMap::new(),
            qwen35_expert_bundle_reuse_history: rnb_memory::ExpertBundleReuseHistory::new(0),
            qwen35_q2q3_bundle_ownership: ResidentQ4kBundleOwnership::default(),
            qwen35_q2q3_resident_payload_bytes: 0,
            #[cfg(test)]
            qwen35_resident_alloc_ooms_remaining: 0,
            #[cfg(test)]
            qwen35_decode_failure_point: None,
            resident_q4k_arena: None,
            resident_q4k_arena_capacity: 0,
            resident_q4k_arena_offset: 0,
            resident_q8_f32: HashMap::new(),
            resident_q8_f32_lru: VecDeque::new(),
            resident_q8_f32_epoch: 0,
            resident_q8_f32_bytes: 0,
            resident_q8_f32_limit,
            resident_q8_quant: HashMap::new(),
            resident_q4_packed: HashMap::new(),
            resident_q4_packed_bytes: 0,
            resident_q4_packed_limit,
            resident_q4_f32: HashMap::new(),
            resident_q4_f32_bytes: 0,
            resident_q4_f32_limit,
            resident_q6_packed: HashMap::new(),
            resident_q6_packed_bytes: 0,
            resident_q6_packed_limit,
            resident_q6_f32: HashMap::new(),
            resident_q6_f32_bytes: 0,
            resident_q6_f32_limit,
            resident_q6_f16: HashMap::new(),
            resident_q6_f16_bytes: 0,
            resident_q6_f16_limit,
            weight_residency_counters: CudaWeightResidencyCounters::default(),
            resident_f32: HashMap::new(),
            resident_rope_tables: HashMap::new(),
            resident_moe_layers: HashMap::new(),
            resident_moe_layer_lru: VecDeque::new(),
            resident_moe_layer_epoch: 0,
            resident_moe_layer_bytes: 0,
            resident_moe_layer_limit,
            qwen35_moe_model_bytes: 0,
            qwen35_moe_min_layer_bytes: 0,
            qwen35_moe_layer_cache_enabled: tuning::moe_layer_cache_enabled(),
            compute_input: None,
            compute_input_capacity: 0,
            compute_output: None,
            compute_output_capacity: 0,
            compute_aux_output: None,
            compute_aux_output_capacity: 0,
            compute_mid_a: None,
            compute_mid_a_capacity: 0,
            compute_mid_b: None,
            compute_mid_b_capacity: 0,
            compute_gate_ptrs: None,
            compute_gate_ptrs_capacity: 0,
            compute_up_ptrs: None,
            compute_up_ptrs_capacity: 0,
            compute_down_ptrs: None,
            compute_down_ptrs_capacity: 0,
            compute_full_gate: None,
            compute_full_gate_capacity: 0,
            compute_full_up: None,
            compute_full_up_capacity: 0,
            compute_full_down: None,
            compute_full_down_capacity: 0,
            compute_temp_slab: None,
            compute_temp_slab_capacity: 0,
            qwen35_selected_base_stream_slab_a: None,
            qwen35_selected_base_stream_slab_a_capacity: 0,
            qwen35_selected_base_stream_slab_b: None,
            qwen35_selected_base_stream_slab_b_capacity: 0,
            qwen35_selected_base_temp_slab_cache: None,
            #[cfg(test)]
            last_qwen35_selected_sparse_boundary_stats: None,
            qwen35_packed_act: None,
            qwen35_packed_act_capacity: 0,
            compute_q_rope_out: None,
            compute_q_rope_out_capacity: 0,
            compute_k_bits_out: None,
            compute_k_bits_out_capacity: 0,
            compute_v_bits_out: None,
            compute_v_bits_out_capacity: 0,
            decode_hidden_carrier: None,
            decode_hidden_carrier_capacity: 0,
            decode_rms_input: None,
            decode_rms_input_capacity: 0,
            decode_norm_buf_carrier: None,
            decode_norm_buf_carrier_capacity: 0,
            decode_attn_out_carrier: None,
            decode_attn_out_carrier_capacity: 0,
            decode_k_carrier: None,
            decode_k_carrier_capacity: 0,
            decode_v_carrier: None,
            decode_v_carrier_capacity: 0,
            decode_k_f16_carrier: None,
            decode_k_f16_carrier_capacity: 0,
            decode_v_f16_carrier: None,
            decode_v_f16_carrier_capacity: 0,
            decode_q_carrier: None,
            decode_q_carrier_capacity: 0,
            cu65_graph_pos: None,
            cu65_graph_pos_capacity: 0,
            cu68_graph_kv_len: None,
            cu68_graph_kv_len_capacity: 0,
            pending_nemotron_prefill_sparse: None,
            host_temp_slab: None,
            host_temp_slab_capacity: 0,
            direct_file_reader: rnb_memory::moe_cold_io::DirectFileReaderCache::default(),
            host_sparse_input_slab: None,
            host_sparse_input_slab_capacity: 0,
            registered_host_ranges: Vec::new(),
            compute_route: None,
            compute_route_capacity: 0,
            compute_token_ids: None,
            compute_token_ids_capacity: 0,
            compute_group_meta: None,
            compute_group_meta_capacity: 0,
            gemma_ple_base: None,
            gemma_ple_base_capacity: 0,
            gemma_ple_base_len: 0,
            mtp_verify_token_ids: None,
            mtp_verify_token_ids_capacity: 0,
            mtp_verify_target_tokens: None,
            mtp_verify_target_tokens_capacity: 0,
            mtp_verify_hidden_rows: None,
            mtp_verify_hidden_rows_capacity: 0,
            mtp_verify_scratch_hidden: None,
            mtp_verify_scratch_hidden_capacity: 0,
            mtp_verify_prefix_indices: None,
            mtp_verify_prefix_indices_capacity: 0,
            mtp_verify_gdn_qkv: None,
            mtp_verify_gdn_qkv_capacity: 0,
            mtp_verify_gdn_gate: None,
            mtp_verify_gdn_gate_capacity: 0,
            mtp_verify_gdn_alpha: None,
            mtp_verify_gdn_alpha_capacity: 0,
            mtp_verify_gdn_beta: None,
            mtp_verify_gdn_beta_capacity: 0,
            mtp_verify_gdn_conv_state: None,
            mtp_verify_gdn_conv_state_capacity: 0,
            mtp_verify_gdn_conv_input: None,
            mtp_verify_gdn_conv_input_capacity: 0,
            mtp_verify_gdn_conv_out: None,
            mtp_verify_gdn_conv_out_capacity: 0,
            mtp_verify_gdn_delta_q: None,
            mtp_verify_gdn_delta_q_capacity: 0,
            mtp_verify_gdn_delta_k: None,
            mtp_verify_gdn_delta_k_capacity: 0,
            mtp_verify_gdn_delta_v: None,
            mtp_verify_gdn_delta_v_capacity: 0,
            mtp_verify_gdn_delta_gate: None,
            mtp_verify_gdn_delta_gate_capacity: 0,
            mtp_verify_gdn_delta_beta: None,
            mtp_verify_gdn_delta_beta_capacity: 0,
            mtp_verify_gdn_delta_out: None,
            mtp_verify_gdn_delta_out_capacity: 0,
            mtp_verify_attention_k_f32: None,
            mtp_verify_attention_k_f32_capacity: 0,
            mtp_verify_attention_v_f32: None,
            mtp_verify_attention_v_f32_capacity: 0,
            mtp_verify_attention_q_compact: None,
            mtp_verify_attention_q_compact_capacity: 0,
            mtp_verify_attention_gate: None,
            mtp_verify_attention_gate_capacity: 0,
            mtp_verify_attention_prior_kv: Vec::new(),
            mtp_verify_attention_shared_window_kv: Vec::new(),
            decode_attention_kv: HashMap::new(),
            decode_attention_kvarn: HashMap::new(),
            mtp_verify_attention_out: None,
            mtp_verify_attention_out_capacity: 0,
            mtp_verify_gdn_gated: None,
            mtp_verify_gdn_gated_capacity: 0,
            mtp_verify_gdn_ssm_out: None,
            mtp_verify_gdn_ssm_out_capacity: 0,
            mtp_verify_router_logits: None,
            mtp_verify_router_logits_capacity: 0,
            mtp_verify_router_expert_ids: None,
            mtp_verify_router_expert_ids_capacity: 0,
            mtp_verify_router_route_weights: None,
            mtp_verify_router_route_weights_capacity: 0,
            mtp_verify_router_token_ids: None,
            mtp_verify_router_token_ids_capacity: 0,
            qwen35_mtp_expert_history: HashMap::new(),
            qwen35_mtp_expert_observations: HashMap::new(),
            resident_delta_states: HashMap::new(),
            mtp_verify_snapshot_pool: Vec::new(),
            nemotron_decode_sparse_calls: 0,
            q4k_gemv_module: None,
            nemotron_selected_module: None,
            persistent_decode_module: None,
            persistent_decode_ctx: None,
            qwen35_sparse_graphs: HashMap::new(),
            qwen35_compound_graphs: HashMap::new(),
            mtp_verify_selected_graphs: HashMap::new(),
            mtp_verify_gdn_graph_warmed: HashSet::new(),
            mtp_verify_gdn_graphs: HashMap::new(),
            mtp_verify_attention_graph_warmed: HashSet::new(),
            mtp_verify_attention_graphs: HashMap::new(),
            mtp_verify_segment_graph_warmed: HashSet::new(),
            mtp_verify_segment_graphs: HashMap::new(),
            mtp_verify_segment_capture_active: false,
            cu65_qkv_graph_warmed: HashSet::new(),
            cu65_qkv_graphs: HashMap::new(),
            cu68_attention_graph_warmed: HashSet::new(),
            cu68_attention_graphs: HashMap::new(),
            dense_expert_graph_warmed: HashSet::new(),
            dense_expert_graphs: HashMap::new(),
            cu69_dense_chain_graph_warmed: HashSet::new(),
            cu69_dense_chain_graphs: HashMap::new(),
            cu71_layer_segment_graph_warmed: HashSet::new(),
            cu71_layer_segment_graphs: HashMap::new(),
            cu62_counter_dev: None,
        })
    }

    #[track_caller]
    pub(in crate::runtime) fn stream_synchronize(&self) -> Result<(), String> {
        if crate::tuning::cu63_sync_diag() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static CTR: AtomicU64 = AtomicU64::new(0);
            let c = CTR.fetch_add(1, Ordering::Relaxed);
            let loc = std::panic::Location::caller();
            eprintln!("[sync-diag] #{c} {loc}");
        }
        self.set_current()?;
        unsafe { self.api.stream_synchronize(self.stream) }
    }

    #[cfg(test)]
    pub(in crate::runtime) fn mem_alloc(&self, bytes: usize) -> Result<u64, String> {
        self.set_current()?;
        unsafe { self.api.mem_alloc(bytes) }
    }

    pub(in crate::runtime) fn set_current(&self) -> Result<(), String> {
        unsafe { self.api.ctx_set_current(self.ctx) }
    }

    pub(in crate::runtime) fn cublas_state_mut(&mut self) -> Result<&mut CublasState, String> {
        self.set_current()?;
        if self.cublas.is_none() {
            self.cublas = Some(CublasState::open()?);
        }
        self.cublas
            .as_mut()
            .ok_or_else(|| "missing cuBLAS state".to_string())
    }
}

impl Drop for CudaState {
    fn drop(&mut self) {
        let _ = self.set_current();
        if let Some(cublas) = self.cublas.take() {
            drop(cublas);
        }
        if let Some(ptr) = self.compute_weights.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        for slot in self.transient_q4_f16_pool.drain(..) {
            if let Some(ptr) = slot.buffer {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
        }
        for (_, tensor) in self.device_tensors.drain() {
            if tensor.storage.is_owned() {
                let _ = unsafe { self.api.mem_free(tensor.ptr) };
            }
        }
        if let Some(workspace) = self.nemotron_prefill_workspace.take() {
            let _ = unsafe { self.api.mem_free(workspace.ptr) };
        }
        cache_stats()
            .remove_expert_bundle_resident_payload(self.qwen35_q2q3_resident_payload_bytes);
        self.qwen35_q2q3_resident_payload_bytes = 0;
        self.qwen35_q2q3_bundle_ownership = ResidentQ4kBundleOwnership::default();
        for (_, entry) in self.resident_q4k.drain() {
            if entry.slab_base.is_none() && entry.owned_alloc {
                let _ = unsafe { self.api.mem_free(entry.ptr) };
            }
        }
        for (ptr, _) in self.resident_q4k_slabs.drain() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.resident_q4k_arena.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        for (_, entry) in self.resident_q8_f32.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_q8_quant.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_q4_packed.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_q4_f32.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_q6_packed.drain() {
            let _ = unsafe { self.api.mem_free(entry.qs_ptr) };
            let _ = unsafe { self.api.mem_free(entry.d_super_ptr) };
            let _ = unsafe { self.api.mem_free(entry.sub_scale_ptr) };
        }
        for (_, entry) in self.resident_q6_f32.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_q6_f16.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_f32.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_rope_tables.drain() {
            let _ = unsafe { self.api.mem_free(entry.sin_ptr) };
            let _ = unsafe { self.api.mem_free(entry.cos_ptr) };
        }
        for (_, entry) in self.resident_moe_layers.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        if let Some(ptr) = self.compute_input.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_output.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_aux_output.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_mid_a.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_mid_b.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_gate_ptrs.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_up_ptrs.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_down_ptrs.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_full_gate.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_full_up.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_full_down.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_temp_slab.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.qwen35_selected_base_stream_slab_a.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.qwen35_selected_base_stream_slab_b.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(cache) = self.qwen35_selected_base_temp_slab_cache.take() {
            let _ = unsafe { self.api.mem_free(cache.slab_dev) };
        }
        if let Some(ptr) = self.qwen35_packed_act.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_q_rope_out.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_k_bits_out.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_v_bits_out.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_hidden_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_rms_input.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_norm_buf_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_attn_out_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_k_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_v_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_k_f16_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_v_f16_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.decode_q_carrier.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.cu65_graph_pos.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.cu68_graph_kv_len.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(prefetch) = self.pending_nemotron_prefill_sparse.take() {
            let _ = unsafe { self.api.stream_synchronize(self.copy_stream) };
            let _ = unsafe { self.api.mem_free(prefetch.slab) };
        }
        if let Some(ptr) = self.host_temp_slab.take() {
            let _ = unsafe { self.api.mem_free_host(ptr as *mut libc::c_void) };
        }
        if let Some(ptr) = self.host_sparse_input_slab.take() {
            let _ = unsafe { self.api.mem_free_host(ptr as *mut libc::c_void) };
        }
        let _ = self.clear_host_registered_ranges();
        if let Some(ptr) = self.compute_route.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_token_ids.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.compute_group_meta.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.gemma_ple_base.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_token_ids.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_target_tokens.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_hidden_rows.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_scratch_hidden.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_prefix_indices.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_qkv.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_gate.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_alpha.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_beta.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_conv_state.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_conv_input.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_conv_out.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_delta_q.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_delta_k.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_delta_v.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_delta_gate.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_delta_beta.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_delta_out.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_attention_k_f32.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_attention_v_f32.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_attention_q_compact.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_attention_gate.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        for cache in self.mtp_verify_attention_prior_kv.drain(..) {
            if let Some(ptr) = cache.k_bits_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
            if let Some(ptr) = cache.v_bits_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
        }
        for cache in self.mtp_verify_attention_shared_window_kv.drain(..) {
            if let Some(ptr) = cache.k_bits_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
            if let Some(ptr) = cache.v_bits_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
        }
        for (_, cache) in self.decode_attention_kv.drain() {
            if let Some(ptr) = cache.k_bits_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
            if let Some(ptr) = cache.v_bits_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
        }
        for (_, cache) in self.decode_attention_kvarn.drain() {
            if let Some(ptr) = cache.records_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
            if let Some(ptr) = cache.f16_dev {
                let _ = unsafe { self.api.mem_free(ptr) };
            }
        }
        if let Some(ptr) = self.mtp_verify_attention_out.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_gated.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_gdn_ssm_out.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_router_logits.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_router_expert_ids.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_router_route_weights.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        if let Some(ptr) = self.mtp_verify_router_token_ids.take() {
            let _ = unsafe { self.api.mem_free(ptr) };
        }
        for entry in self.mtp_verify_snapshot_pool.drain(..) {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, entry) in self.resident_delta_states.drain() {
            let _ = unsafe { self.api.mem_free(entry.ptr) };
        }
        for (_, graph) in self.qwen35_sparse_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.qwen35_compound_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.mtp_verify_selected_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.mtp_verify_gdn_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.mtp_verify_attention_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.mtp_verify_segment_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.cu65_qkv_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.cu68_attention_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.dense_expert_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.cu69_dense_chain_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        for (_, graph) in self.cu71_layer_segment_graphs.drain() {
            let _ = unsafe { self.api.graph_exec_destroy(graph.exec as *mut libc::c_void) };
            let _ = unsafe { self.api.graph_destroy(graph.graph as *mut libc::c_void) };
        }
        if let Some(module) = self.q4k_gemv_module.take() {
            let _ = unsafe { self.api.module_unload(module as *mut libc::c_void) };
        }
        if let Some(module) = self.nemotron_selected_module.take() {
            let _ = unsafe { self.api.module_unload(module as *mut libc::c_void) };
        }
        if let Some(module) = self.persistent_decode_module.take() {
            let _ = unsafe { self.api.module_unload(module as *mut libc::c_void) };
        }
        let _ = unsafe { self.api.stream_destroy(self.stream) };
        let _ = unsafe { self.api.stream_destroy(self.copy_stream) };
        let _ = unsafe { self.api.ctx_destroy(self.ctx) };
        unsafe {
            crate::dynlib::close(self.lib_handle);
        }
    }
}

impl CublasState {
    pub(in crate::runtime) fn open() -> Result<Self, String> {
        let lib_handle = dlopen_cublas()?;
        let api = unsafe { CublasApi::load(lib_handle)? };
        let handle = unsafe { api.create()? };
        // cu25: optional TF32 tensor core mode for sgemm. Default OFF (strict
        // fp32) to preserve token-identical output. Opt-in via
        // RNB_CUDA_CUBLAS_TF32=1 trades 1-ULP fp32→tf32 rounding for tensor
        // core throughput on Ampere+ (sm_80+) GPUs.
        let mode = match std::env::var("RNB_CUDA_CUBLAS_TF32").ok().as_deref() {
            Some(v) if matches!(v, "1" | "true" | "on" | "yes" | "TF32") => {
                CUBLAS_TF32_TENSOR_OP_MATH
            }
            Some("pedantic") => CUBLAS_PEDANTIC_MATH,
            _ => CUBLAS_DEFAULT_MATH,
        };
        if mode != CUBLAS_DEFAULT_MATH {
            unsafe { api.set_math_mode(handle, mode)? };
            if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
                eprintln!("[cuda] cuBLAS math mode set to {mode}");
            }
        }
        Ok(Self {
            lib_handle,
            api,
            handle,
        })
    }
}

impl Drop for CublasState {
    fn drop(&mut self) {
        let _ = unsafe { self.api.destroy(self.handle) };
        unsafe {
            crate::dynlib::close(self.lib_handle);
        }
    }
}
