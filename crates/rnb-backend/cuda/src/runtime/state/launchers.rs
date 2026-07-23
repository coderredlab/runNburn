use super::super::*;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Q6PackedBatchKernelPlan {
    Seq8Warp4,
    Seq4Warp4,
    Seq4Warp8,
    Warp8,
}

fn q6k_packed_batch_kernel_plan(seq_len: usize, blocks_per_row: usize) -> Q6PackedBatchKernelPlan {
    let seq8 = tuning::q6k_packed_batch_seq8_enabled(seq_len, blocks_per_row);
    if seq8 {
        return Q6PackedBatchKernelPlan::Seq8Warp4;
    }
    let seq4 = std::env::var("RNB_CUDA_Q6_PACKED_BATCH_Q8DOT_SEQ4")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(seq_len > 1);
    if seq4 && tuning::q6k_packed_batch_warp4_enabled(blocks_per_row) {
        return Q6PackedBatchKernelPlan::Seq4Warp4;
    }
    if seq4 {
        return Q6PackedBatchKernelPlan::Seq4Warp8;
    }
    Q6PackedBatchKernelPlan::Warp8
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(in crate::runtime) enum Q6PackedBatchKernelPlanForTest {
    Seq8Warp4,
    Seq4Warp4,
    Seq4Warp8,
    Warp8,
}

#[cfg(test)]
impl From<Q6PackedBatchKernelPlan> for Q6PackedBatchKernelPlanForTest {
    fn from(plan: Q6PackedBatchKernelPlan) -> Self {
        match plan {
            Q6PackedBatchKernelPlan::Seq8Warp4 => Self::Seq8Warp4,
            Q6PackedBatchKernelPlan::Seq4Warp4 => Self::Seq4Warp4,
            Q6PackedBatchKernelPlan::Seq4Warp8 => Self::Seq4Warp8,
            Q6PackedBatchKernelPlan::Warp8 => Self::Warp8,
        }
    }
}

#[cfg(test)]
pub(in crate::runtime) fn q6k_packed_batch_kernel_plan_for_test(
    seq_len: usize,
    blocks_per_row: usize,
) -> Q6PackedBatchKernelPlanForTest {
    q6k_packed_batch_kernel_plan(seq_len, blocks_per_row).into()
}

impl CudaState {
    pub(in crate::runtime) fn launch_zero_f32(
        &mut self,
        output_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_zero_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_qwen35_router_topk_softmax_f32(
        &mut self,
        logits_dev: u64,
        expert_ids_dev: u64,
        route_weights_dev: u64,
        token_ids_dev: u64,
        window_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
    ) -> Result<(), String> {
        if window_tokens == 0 {
            return Err("Qwen35 router top-k requires at least one token".to_string());
        }
        if n_expert == 0 || n_expert_used == 0 {
            return Err(format!(
                "Qwen35 router top-k requires non-zero experts and selected experts: experts={n_expert}, used={n_expert_used}"
            ));
        }
        let mut logits_arg = logits_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut route_weights_arg = route_weights_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut tokens_arg = u32::try_from(window_tokens).map_err(|_| {
            format!("Qwen35 router top-k token count exceeds CUDA u32 limit: {window_tokens}")
        })?;
        let mut n_expert_arg = u32::try_from(n_expert).map_err(|_| {
            format!("Qwen35 router top-k expert count exceeds CUDA u32 limit: {n_expert}")
        })?;
        let mut n_expert_used_arg = u32::try_from(n_expert_used).map_err(|_| {
            format!(
                "Qwen35 router top-k selected expert count exceeds CUDA u32 limit: {n_expert_used}"
            )
        })?;
        self.launch_cached_gemv(
            "rnb_qwen35_router_topk_softmax_f32",
            &[
                (&mut logits_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut tokens_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_expert_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_expert_used_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (tokens_arg, 1, 1),
            (32, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_qwen35_router_topk_logits_f32(
        &mut self,
        logits_dev: u64,
        expert_ids_dev: u64,
        route_logits_dev: u64,
        token_ids_dev: u64,
        window_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
    ) -> Result<(), String> {
        if window_tokens == 0 {
            return Err("Qwen35 router top-k logits requires at least one token".to_string());
        }
        if n_expert == 0 || n_expert_used == 0 {
            return Err(format!(
                "Qwen35 router top-k logits requires non-zero experts and selected experts: experts={n_expert}, used={n_expert_used}"
            ));
        }
        let mut logits_arg = logits_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut route_logits_arg = route_logits_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut tokens_arg = u32::try_from(window_tokens).map_err(|_| {
            format!(
                "Qwen35 router top-k logits token count exceeds CUDA u32 limit: {window_tokens}"
            )
        })?;
        let mut n_expert_arg = u32::try_from(n_expert).map_err(|_| {
            format!("Qwen35 router top-k logits expert count exceeds CUDA u32 limit: {n_expert}")
        })?;
        let mut n_expert_used_arg = u32::try_from(n_expert_used).map_err(|_| {
            format!(
                "Qwen35 router top-k logits selected expert count exceeds CUDA u32 limit: {n_expert_used}"
            )
        })?;
        self.launch_cached_gemv(
            "rnb_qwen35_router_topk_logits_f32",
            &[
                (&mut logits_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_logits_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut tokens_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_expert_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_expert_used_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (tokens_arg, 1, 1),
            (32, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_sigmoid_topk_route_f32(
        &mut self,
        logits_dev: u64,
        bias_dev: Option<u64>,
        expert_ids_dev: u64,
        route_weights_dev: u64,
        token_ids_dev: u64,
        window_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        expert_weight_scale: f32,
    ) -> Result<(), String> {
        if window_tokens == 0 {
            return Err("Nemotron route pack requires at least one token".to_string());
        }
        if n_expert == 0 || n_expert_used == 0 {
            return Err(format!(
                "Nemotron route pack requires non-zero experts and selected experts: experts={n_expert}, used={n_expert_used}"
            ));
        }
        if n_expert_used > 32 {
            return Err(format!(
                "Nemotron route pack supports up to 32 selected experts, got {n_expert_used}"
            ));
        }
        let mut logits_arg = logits_dev;
        let mut bias_arg = bias_dev.unwrap_or(0);
        let mut expert_ids_arg = expert_ids_dev;
        let mut route_weights_arg = route_weights_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut tokens_arg = u32::try_from(window_tokens).map_err(|_| {
            format!("Nemotron route pack token count exceeds CUDA u32 limit: {window_tokens}")
        })?;
        let mut n_expert_arg = u32::try_from(n_expert).map_err(|_| {
            format!("Nemotron route pack expert count exceeds CUDA u32 limit: {n_expert}")
        })?;
        let mut n_expert_used_arg = u32::try_from(n_expert_used).map_err(|_| {
            format!(
                "Nemotron route pack selected expert count exceeds CUDA u32 limit: {n_expert_used}"
            )
        })?;
        let mut scale_arg = expert_weight_scale;
        let mut has_bias_arg = u32::from(bias_dev.is_some());
        self.launch_cached_gemv(
            "rnb_nemotron_sigmoid_topk_route_f32",
            &[
                (&mut logits_arg as *mut u64).cast::<libc::c_void>(),
                (&mut bias_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut tokens_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_expert_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_expert_used_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                (&mut has_bias_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (tokens_arg, 1, 1),
            (1, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_reorder_route_slots(
        &mut self,
        expert_ids_in: u64,
        route_weights_in: u64,
        token_ids_in: u64,
        order_indices: u64,
        expert_ids_out: u64,
        route_weights_out: u64,
        token_ids_out: u64,
        slots: usize,
    ) -> Result<(), String> {
        if slots == 0 {
            return Err("Nemotron route reorder requires at least one slot".to_string());
        }
        let mut expert_ids_in_arg = expert_ids_in;
        let mut route_weights_in_arg = route_weights_in;
        let mut token_ids_in_arg = token_ids_in;
        let mut order_indices_arg = order_indices;
        let mut expert_ids_out_arg = expert_ids_out;
        let mut route_weights_out_arg = route_weights_out;
        let mut token_ids_out_arg = token_ids_out;
        let mut slots_arg = u32::try_from(slots)
            .map_err(|_| format!("Nemotron route reorder slots exceed CUDA u32 limit: {slots}"))?;
        self.launch_cached_gemv(
            "rnb_nemotron_reorder_route_slots",
            &[
                (&mut expert_ids_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_weights_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut order_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_weights_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut slots_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((slots as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qwen35_build_q4k_full_layer_slot_ptrs(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        down_ptrs_dev: u64,
        expert_ids_dev: u64,
        pair_slots_dev: u64,
        slots_per_token: usize,
        gate_base: u64,
        up_base: u64,
        down_base: u64,
        gate_expert_bytes: usize,
        down_expert_bytes: usize,
        slots: usize,
    ) -> Result<(), String> {
        if slots == 0 {
            return Err("Qwen35 full-layer slot pointer build requires slots".to_string());
        }
        let mut gate_ptrs_arg = gate_ptrs_dev;
        let mut up_ptrs_arg = up_ptrs_dev;
        let mut down_ptrs_arg = down_ptrs_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut pair_slots_arg = pair_slots_dev;
        let mut gate_base_arg = gate_base;
        let mut up_base_arg = up_base;
        let mut down_base_arg = down_base;
        let mut gate_expert_bytes_arg = u32::try_from(gate_expert_bytes).map_err(|_| {
            format!(
                "Qwen35 full-layer gate expert bytes exceeds CUDA u32 limit: {gate_expert_bytes}"
            )
        })?;
        let mut down_expert_bytes_arg = u32::try_from(down_expert_bytes).map_err(|_| {
            format!(
                "Qwen35 full-layer down expert bytes exceeds CUDA u32 limit: {down_expert_bytes}"
            )
        })?;
        let mut slots_per_token_arg = u32::try_from(slots_per_token).map_err(|_| {
            format!("Qwen35 full-layer slots per token exceeds CUDA u32 limit: {slots_per_token}")
        })?;
        let mut slots_arg = u32::try_from(slots)
            .map_err(|_| format!("Qwen35 full-layer slot count exceeds CUDA u32 limit: {slots}"))?;
        self.launch_cached_gemv(
            "rnb_qwen35_build_q4k_full_layer_slot_ptrs",
            &[
                (&mut gate_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut down_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pair_slots_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_base_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_base_arg as *mut u64).cast::<libc::c_void>(),
                (&mut down_base_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_expert_bytes_arg as *mut u32).cast::<libc::c_void>(),
                (&mut down_expert_bytes_arg as *mut u32).cast::<libc::c_void>(),
                (&mut slots_per_token_arg as *mut u32).cast::<libc::c_void>(),
                (&mut slots_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((slots_arg.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qwen35_build_q4k_compact_slot_ptrs(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        down_ptrs_dev: u64,
        expert_ids_dev: u64,
        expert_slab_indices_dev: u64,
        gate_base: u64,
        up_base: u64,
        down_base: u64,
        gate_expert_bytes: usize,
        up_expert_bytes: usize,
        down_expert_bytes: usize,
        slots: usize,
    ) -> Result<(), String> {
        if slots == 0 {
            return Err("Qwen35 compact slot pointer build requires slots".to_string());
        }
        let mut gate_ptrs_arg = gate_ptrs_dev;
        let mut up_ptrs_arg = up_ptrs_dev;
        let mut down_ptrs_arg = down_ptrs_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut expert_slab_indices_arg = expert_slab_indices_dev;
        let mut gate_base_arg = gate_base;
        let mut up_base_arg = up_base;
        let mut down_base_arg = down_base;
        let mut gate_expert_bytes_arg = u32::try_from(gate_expert_bytes).map_err(|_| {
            format!(
                "Qwen35 compact slot gate expert bytes exceeds CUDA u32 limit: {gate_expert_bytes}"
            )
        })?;
        let mut up_expert_bytes_arg = u32::try_from(up_expert_bytes).map_err(|_| {
            format!("Qwen35 compact slot up expert bytes exceeds CUDA u32 limit: {up_expert_bytes}")
        })?;
        let mut down_expert_bytes_arg = u32::try_from(down_expert_bytes).map_err(|_| {
            format!(
                "Qwen35 compact slot down expert bytes exceeds CUDA u32 limit: {down_expert_bytes}"
            )
        })?;
        let mut slots_arg = u32::try_from(slots)
            .map_err(|_| format!("Qwen35 compact slot count exceeds CUDA u32 limit: {slots}"))?;
        self.launch_cached_gemv(
            "rnb_qwen35_build_q4k_compact_slot_ptrs",
            &[
                (&mut gate_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut down_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_slab_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_base_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_base_arg as *mut u64).cast::<libc::c_void>(),
                (&mut down_base_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_expert_bytes_arg as *mut u32).cast::<libc::c_void>(),
                (&mut up_expert_bytes_arg as *mut u32).cast::<libc::c_void>(),
                (&mut down_expert_bytes_arg as *mut u32).cast::<libc::c_void>(),
                (&mut slots_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((slots_arg.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qwen35_build_q4k_mixed_slot_ptrs(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        down_ptrs_dev: u64,
        expert_ids_dev: u64,
        gate_expert_ptrs_dev: u64,
        up_expert_ptrs_dev: u64,
        down_expert_ptrs_dev: u64,
        slots: usize,
    ) -> Result<(), String> {
        if slots == 0 {
            return Err("Qwen35 mixed slot pointer build requires slots".to_string());
        }
        let mut gate_ptrs_arg = gate_ptrs_dev;
        let mut up_ptrs_arg = up_ptrs_dev;
        let mut down_ptrs_arg = down_ptrs_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut gate_expert_ptrs_arg = gate_expert_ptrs_dev;
        let mut up_expert_ptrs_arg = up_expert_ptrs_dev;
        let mut down_expert_ptrs_arg = down_expert_ptrs_dev;
        let mut slots_arg = u32::try_from(slots)
            .map_err(|_| format!("Qwen35 mixed slot count exceeds CUDA u32 limit: {slots}"))?;
        self.launch_cached_gemv(
            "rnb_qwen35_build_q4k_mixed_slot_ptrs",
            &[
                (&mut gate_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut down_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_expert_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_expert_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut down_expert_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut slots_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((slots_arg.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_qwen35_shared_route_sigmoid_f32(
        &mut self,
        route_dev: u64,
        input_dev: u64,
        scale_dev: u64,
        rows: usize,
        hidden_dim: usize,
    ) -> Result<(), String> {
        if rows == 0 || hidden_dim == 0 {
            return Err(format!(
                "Qwen35 shared route requires non-zero rows and hidden_dim: rows={rows}, hidden_dim={hidden_dim}"
            ));
        }
        let mut route_arg = route_dev;
        let mut input_arg = input_dev;
        let mut scale_arg = scale_dev;
        let mut rows_arg = u32::try_from(rows)
            .map_err(|_| format!("Qwen35 shared route rows exceeds CUDA u32 limit: {rows}"))?;
        let mut hidden_dim_arg = u32::try_from(hidden_dim).map_err(|_| {
            format!("Qwen35 shared route hidden_dim exceeds CUDA u32 limit: {hidden_dim}")
        })?;
        self.launch_cached_gemv(
            "rnb_qwen35_shared_route_sigmoid_f32",
            &[
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut scale_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut hidden_dim_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows_arg, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_mamba2_split_projection(
        &mut self,
        projected_dev: u64,
        dt_bias_dev: u64,
        z_dev: u64,
        conv_dev: u64,
        dt_dev: u64,
        seq_len: usize,
        d_inner: usize,
        conv_channels: usize,
        num_heads: usize,
    ) -> Result<(), String> {
        let rows = d_inner
            .checked_add(conv_channels)
            .and_then(|v| v.checked_add(num_heads))
            .ok_or_else(|| "Mamba2 split rows overflow".to_string())?;
        let total = seq_len
            .checked_mul(rows)
            .ok_or_else(|| "Mamba2 split total overflow".to_string())?;
        if total == 0 {
            return Ok(());
        }
        let total_u32 = u32::try_from(total)
            .map_err(|_| format!("Mamba2 split total exceeds CUDA u32 limit: {total}"))?;
        let mut projected_arg = projected_dev;
        let mut dt_bias_arg = dt_bias_dev;
        let mut z_arg = z_dev;
        let mut conv_arg = conv_dev;
        let mut dt_arg = dt_dev;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("Mamba2 split seq_len exceeds CUDA u32 limit: {seq_len}"))?;
        let mut d_inner_arg = u32::try_from(d_inner)
            .map_err(|_| format!("Mamba2 split d_inner exceeds CUDA u32 limit: {d_inner}"))?;
        let mut conv_channels_arg = u32::try_from(conv_channels).map_err(|_| {
            format!("Mamba2 split conv_channels exceeds CUDA u32 limit: {conv_channels}")
        })?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("Mamba2 split num_heads exceeds CUDA u32 limit: {num_heads}"))?;
        self.launch_cached_gemv(
            "rnb_nemotron_mamba2_split_projection",
            &[
                (&mut projected_arg as *mut u64).cast::<libc::c_void>(),
                (&mut dt_bias_arg as *mut u64).cast::<libc::c_void>(),
                (&mut z_arg as *mut u64).cast::<libc::c_void>(),
                (&mut conv_arg as *mut u64).cast::<libc::c_void>(),
                (&mut dt_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut d_inner_arg as *mut u32).cast::<libc::c_void>(),
                (&mut conv_channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((total_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_gdn_build_conv_input_f32(
        &mut self,
        output_dev: u64,
        conv_state_dev: u64,
        qkv_rows_dev: u64,
        window_tokens: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Result<(), String> {
        let state_rows = kernel_size
            .checked_sub(1)
            .ok_or_else(|| "GDN conv input kernel_size must be non-zero".to_string())?;
        let len = window_tokens
            .checked_add(state_rows)
            .and_then(|rows| rows.checked_mul(channels))
            .ok_or_else(|| {
                format!(
                    "GDN conv input element count overflow: tokens={window_tokens}, channels={channels}, kernel_size={kernel_size}"
                )
            })?;
        let len_u32 = u32::try_from(len)
            .map_err(|_| format!("GDN conv input len exceeds CUDA u32 limit: {len}"))?;
        let mut output_arg = output_dev;
        let mut conv_state_arg = conv_state_dev;
        let mut qkv_rows_arg = qkv_rows_dev;
        let mut window_tokens_arg = u32::try_from(window_tokens).map_err(|_| {
            format!("GDN conv input window_tokens exceeds CUDA u32 limit: {window_tokens}")
        })?;
        let mut channels_arg = u32::try_from(channels)
            .map_err(|_| format!("GDN conv input channels exceeds CUDA u32 limit: {channels}"))?;
        let mut kernel_size_arg = u32::try_from(kernel_size).map_err(|_| {
            format!("GDN conv input kernel_size exceeds CUDA u32 limit: {kernel_size}")
        })?;
        self.launch_cached_gemv(
            "rnb_gdn_build_conv_input_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut conv_state_arg as *mut u64).cast::<libc::c_void>(),
                (&mut qkv_rows_arg as *mut u64).cast::<libc::c_void>(),
                (&mut window_tokens_arg as *mut u32).cast::<libc::c_void>(),
                (&mut channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kernel_size_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_ssm_conv1d_silu_dev(
        &mut self,
        input_dev: u64,
        kernel_dev: u64,
        output_dev: u64,
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Result<(), String> {
        let output_len = seq_len.checked_mul(channels).ok_or_else(|| {
            format!("GDN conv output len overflow: seq_len={seq_len}, channels={channels}")
        })?;
        let output_len_u32 = u32::try_from(output_len)
            .map_err(|_| format!("GDN conv output len exceeds CUDA u32 limit: {output_len}"))?;
        let mut output_arg = output_dev;
        let mut input_arg = input_dev;
        let mut kernel_arg = kernel_dev;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("GDN conv seq_len exceeds CUDA u32 limit: {seq_len}"))?;
        let mut channels_arg = u32::try_from(channels)
            .map_err(|_| format!("GDN conv channels exceeds CUDA u32 limit: {channels}"))?;
        let mut kernel_size_arg = u32::try_from(kernel_size)
            .map_err(|_| format!("GDN conv kernel_size exceeds CUDA u32 limit: {kernel_size}"))?;
        self.launch_cached_gemv(
            "rnb_ssm_conv1d_silu",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut kernel_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kernel_size_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((output_len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_mamba2_conv1d_bias_silu_dev(
        &mut self,
        input_dev: u64,
        kernel_dev: u64,
        bias_dev: u64,
        output_dev: u64,
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Result<(), String> {
        let output_len = seq_len.checked_mul(channels).ok_or_else(|| {
            format!("Mamba2 conv output len overflow: seq_len={seq_len}, channels={channels}")
        })?;
        if output_len == 0 {
            return Ok(());
        }
        let output_len_u32 = u32::try_from(output_len)
            .map_err(|_| format!("Mamba2 conv output len exceeds CUDA u32 limit: {output_len}"))?;
        let mut output_arg = output_dev;
        let mut input_arg = input_dev;
        let mut kernel_arg = kernel_dev;
        let mut bias_arg = bias_dev;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("Mamba2 conv seq_len exceeds CUDA u32 limit: {seq_len}"))?;
        let mut channels_arg = u32::try_from(channels)
            .map_err(|_| format!("Mamba2 conv channels exceeds CUDA u32 limit: {channels}"))?;
        let mut kernel_size_arg = u32::try_from(kernel_size).map_err(|_| {
            format!("Mamba2 conv kernel_size exceeds CUDA u32 limit: {kernel_size}")
        })?;
        self.launch_cached_gemv(
            "rnb_nemotron_mamba2_conv1d_bias_silu",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut kernel_arg as *mut u64).cast::<libc::c_void>(),
                (&mut bias_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kernel_size_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((output_len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn launch_nemotron_mamba2_add_residual(
        &mut self,
        output_dev: u64,
        proj_dev: u64,
        residual_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let len_u32 = u32::try_from(len)
            .map_err(|_| format!("Mamba2 residual len exceeds CUDA u32 limit: {len}"))?;
        let mut output_arg = output_dev;
        let mut proj_arg = proj_dev;
        let mut residual_arg = residual_dev;
        let mut len_arg = len_u32;
        self.launch_cached_gemv(
            "rnb_nemotron_mamba2_add_residual",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut proj_arg as *mut u64).cast::<libc::c_void>(),
                (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_gdn_prepare_delta_qkv_f32(
        &mut self,
        q_dev: u64,
        k_dev: u64,
        v_dev: u64,
        conv_out_dev: u64,
        window_tokens: usize,
        conv_channels: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        eps: f32,
        q_scale: f32,
    ) -> Result<(), String> {
        let rows = window_tokens.checked_mul(num_v_heads).ok_or_else(|| {
            format!("GDN delta qkv row count overflow: tokens={window_tokens}, heads={num_v_heads}")
        })?;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut conv_arg = conv_out_dev;
        let mut window_tokens_arg = u32::try_from(window_tokens).map_err(|_| {
            format!("GDN delta qkv window_tokens exceeds CUDA u32 limit: {window_tokens}")
        })?;
        let mut conv_channels_arg = u32::try_from(conv_channels).map_err(|_| {
            format!("GDN delta qkv conv_channels exceeds CUDA u32 limit: {conv_channels}")
        })?;
        let mut num_k_heads_arg = u32::try_from(num_k_heads).map_err(|_| {
            format!("GDN delta qkv num_k_heads exceeds CUDA u32 limit: {num_k_heads}")
        })?;
        let mut num_v_heads_arg = u32::try_from(num_v_heads).map_err(|_| {
            format!("GDN delta qkv num_v_heads exceeds CUDA u32 limit: {num_v_heads}")
        })?;
        let mut head_k_dim_arg = u32::try_from(head_k_dim).map_err(|_| {
            format!("GDN delta qkv head_k_dim exceeds CUDA u32 limit: {head_k_dim}")
        })?;
        let mut head_v_dim_arg = u32::try_from(head_v_dim).map_err(|_| {
            format!("GDN delta qkv head_v_dim exceeds CUDA u32 limit: {head_v_dim}")
        })?;
        let mut eps_arg = eps;
        let mut q_scale_arg = q_scale;
        let (kernel, grid, block) = if crate::tuning::mtp_verify_gdn_qkv_warp_enabled(window_tokens)
        {
            (
                "rnb_gdn_prepare_delta_qkv_f32_warp",
                (((rows as u32).saturating_add(7)) / 8, 1, 1),
                (256, 1, 1),
            )
        } else {
            (
                "rnb_gdn_prepare_delta_qkv_f32",
                (((rows as u32).saturating_add(255)) / 256, 1, 1),
                (256, 1, 1),
            )
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut conv_arg as *mut u64).cast::<libc::c_void>(),
                (&mut window_tokens_arg as *mut u32).cast::<libc::c_void>(),
                (&mut conv_channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_k_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_v_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_k_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_v_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut q_scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    pub(in crate::runtime) fn launch_gdn_prepare_delta_gate_beta_f32(
        &mut self,
        gate_dev: u64,
        beta_dev: u64,
        alpha_dev: u64,
        beta_raw_dev: u64,
        dt_bias_dev: u64,
        ssm_a_dev: u64,
        len: usize,
        num_heads: usize,
    ) -> Result<(), String> {
        let len_u32 = u32::try_from(len)
            .map_err(|_| format!("GDN delta gate len exceeds CUDA u32 limit: {len}"))?;
        let mut gate_arg = gate_dev;
        let mut beta_arg = beta_dev;
        let mut alpha_arg = alpha_dev;
        let mut beta_raw_arg = beta_raw_dev;
        let mut dt_bias_arg = dt_bias_dev;
        let mut ssm_a_arg = ssm_a_dev;
        let mut len_arg = len_u32;
        let mut num_heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("GDN delta gate num_heads exceeds CUDA u32 limit: {num_heads}"))?;
        self.launch_cached_gemv(
            "rnb_gdn_prepare_delta_gate_beta_f32",
            &[
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut alpha_arg as *mut u64).cast::<libc::c_void>(),
                (&mut beta_raw_arg as *mut u64).cast::<libc::c_void>(),
                (&mut dt_bias_arg as *mut u64).cast::<libc::c_void>(),
                (&mut ssm_a_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_heads_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_gdn_gated_norm_silu_dev(
        &mut self,
        output_dev: u64,
        delta_dev: u64,
        z_dev: u64,
        norm_dev: u64,
        rows: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut delta_arg = delta_dev;
        let mut z_arg = z_dev;
        let mut norm_arg = norm_dev;
        let mut rows_arg = u32::try_from(rows)
            .map_err(|_| format!("GDN gated norm rows exceeds CUDA u32 limit: {rows}"))?;
        let mut head_dim_arg = u32::try_from(head_dim)
            .map_err(|_| format!("GDN gated norm head_dim exceeds CUDA u32 limit: {head_dim}"))?;
        let mut eps_arg = eps;
        self.launch_cached_gemv(
            "rnb_gdn_gated_norm_silu",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut delta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut z_arg as *mut u64).cast::<libc::c_void>(),
                (&mut norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_f32_to_f16(
        &mut self,
        input_dev: u64,
        output_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let len_u32 = u32::try_from(len)
            .map_err(|_| format!("rnb_f32_to_f16 len exceeds CUDA kernel u32 limit: {len}"))?;
        let mut input_arg = input_dev;
        let mut output_arg = output_dev;
        let mut len_arg = len_u32;
        self.launch_cached_gemv(
            "rnb_f32_to_f16",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_f16_to_f32(
        &mut self,
        input_dev: u64,
        output_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let len_u32 = u32::try_from(len)
            .map_err(|_| format!("rnb_f16_to_f32 len exceeds CUDA kernel u32 limit: {len}"))?;
        let mut input_arg = input_dev;
        let mut output_arg = output_dev;
        let mut len_arg = len_u32;
        self.launch_cached_gemv(
            "rnb_f16_to_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_split_gated_attention_q_f32(
        &mut self,
        q_full_dev: u64,
        q_out_dev: u64,
        gate_out_dev: u64,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let total = seq_len
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(head_dim))
            .ok_or_else(|| {
                format!(
                    "split gated attention q size overflow: seq_len={seq_len} num_heads={num_heads} head_dim={head_dim}"
                )
            })?;
        let total_u32 = u32::try_from(total)
            .map_err(|_| format!("split gated attention q len exceeds u32: {total}"))?;
        let mut q_full_arg = q_full_dev;
        let mut q_out_arg = q_out_dev;
        let mut gate_arg = gate_out_dev;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("split gated attention q seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("split gated attention q num_heads exceeds u32: {num_heads}"))?;
        let mut head_dim_arg = u32::try_from(head_dim)
            .map_err(|_| format!("split gated attention q head_dim exceeds u32: {head_dim}"))?;
        self.launch_cached_gemv(
            "rnb_split_gated_attention_q_f32",
            &[
                (&mut q_full_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((total_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_sigmoid_mul_inplace(
        &mut self,
        values_dev: u64,
        gate_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let len_u32 = u32::try_from(len)
            .map_err(|_| format!("sigmoid mul len exceeds CUDA kernel u32 limit: {len}"))?;
        let mut values_arg = values_dev;
        let mut gate_arg = gate_dev;
        let mut len_arg = len_u32;
        self.launch_cached_gemv(
            "rnb_sigmoid_mul_inplace",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_u32.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    // cu28: Llama / Mistral hd=128 path 용 RoPE-only QKV launcher. qk-norm
    // 없고 Q는 f32 out, K/V는 f16 bits out. grid = (seq_len, num_heads +
    // num_kv_heads, 1), block = (64, 1, 1).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qk_rope_neox_hd128_f16kv(
        &mut self,
        q_in_dev: u64,
        k_in_dev: u64,
        v_in_dev: u64,
        rope_sin_dev: u64,
        rope_cos_dev: u64,
        q_out_dev: u64,
        k_out_dev: u64,
        v_out_dev: u64,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<(), String> {
        let mut q_in_arg = q_in_dev;
        let mut k_in_arg = k_in_dev;
        let mut v_in_arg = v_in_dev;
        let mut rope_sin_arg = rope_sin_dev;
        let mut rope_cos_arg = rope_cos_dev;
        let mut q_out_arg = q_out_dev;
        let mut k_out_arg = k_out_dev;
        let mut v_out_arg = v_out_dev;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("qk rope hd128 seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("qk rope hd128 num_heads exceeds u32: {num_heads}"))?;
        let mut kv_heads_arg = u32::try_from(num_kv_heads)
            .map_err(|_| format!("qk rope hd128 num_kv_heads exceeds u32: {num_kv_heads}"))?;
        self.launch_cached_gemv(
            "rnb_qk_rope_neox_hd128_f16kv",
            &[
                (&mut q_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_sin_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_cos_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg + kv_heads_arg, 1),
            (64, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_qk_norm_rope_neox_hd512_f16kv(
        &mut self,
        q_in_dev: u64,
        k_in_dev: u64,
        v_in_dev: u64,
        q_norm_dev: u64,
        k_norm_dev: u64,
        rope_sin_dev: u64,
        rope_cos_dev: u64,
        q_out_dev: u64,
        k_out_dev: u64,
        v_out_dev: u64,
        eps: f32,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        pos_start: usize,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
    ) -> Result<(), String> {
        let mut q_in_arg = q_in_dev;
        let mut k_in_arg = k_in_dev;
        let mut v_in_arg = v_in_dev;
        let mut q_norm_arg = q_norm_dev;
        let mut k_norm_arg = k_norm_dev;
        let mut rope_sin_arg = rope_sin_dev;
        let mut rope_cos_arg = rope_cos_dev;
        let mut q_out_arg = q_out_dev;
        let mut k_out_arg = k_out_dev;
        let mut v_out_arg = v_out_dev;
        let mut eps_arg = eps;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("qk norm rope seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("qk norm rope num_heads exceeds u32: {num_heads}"))?;
        let mut kv_heads_arg = u32::try_from(num_kv_heads)
            .map_err(|_| format!("qk norm rope num_kv_heads exceeds u32: {num_kv_heads}"))?;
        let mut pos_start_arg = u32::try_from(pos_start)
            .map_err(|_| format!("qk norm rope pos_start exceeds u32: {pos_start}"))?;
        let mut q_unit_arg = u32::from(q_unit_offset);
        let mut k_unit_arg = u32::from(k_unit_offset);
        let mut v_norm_arg = u32::from(v_no_scale_norm);
        self.launch_cached_gemv(
            "rnb_qk_norm_rope_neox_hd512_f16kv",
            &[
                (&mut q_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_sin_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_cos_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut q_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut k_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut v_norm_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg + kv_heads_arg, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q_norm_rope_neox_hd512(
        &mut self,
        q_in_dev: u64,
        q_norm_dev: u64,
        rope_sin_dev: u64,
        rope_cos_dev: u64,
        q_out_dev: u64,
        eps: f32,
        seq_len: usize,
        num_heads: usize,
        pos_start: usize,
        q_unit_offset: bool,
    ) -> Result<(), String> {
        let mut q_in_arg = q_in_dev;
        let mut q_norm_arg = q_norm_dev;
        let mut rope_sin_arg = rope_sin_dev;
        let mut rope_cos_arg = rope_cos_dev;
        let mut q_out_arg = q_out_dev;
        let mut eps_arg = eps;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("q norm rope seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("q norm rope num_heads exceeds u32: {num_heads}"))?;
        let mut pos_start_arg = u32::try_from(pos_start)
            .map_err(|_| format!("q norm rope pos_start exceeds u32: {pos_start}"))?;
        let mut q_unit_arg = u32::from(q_unit_offset);
        self.launch_cached_gemv(
            "rnb_q_norm_rope_neox_hd512",
            &[
                (&mut q_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_sin_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_cos_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut q_unit_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q_norm_rope_neox_hd256(
        &mut self,
        q_in_dev: u64,
        q_norm_dev: u64,
        rope_sin_dev: u64,
        rope_cos_dev: u64,
        q_out_dev: u64,
        eps: f32,
        seq_len: usize,
        num_heads: usize,
        pos_start: usize,
        q_unit_offset: bool,
    ) -> Result<(), String> {
        let mut q_in_arg = q_in_dev;
        let mut q_norm_arg = q_norm_dev;
        let mut rope_sin_arg = rope_sin_dev;
        let mut rope_cos_arg = rope_cos_dev;
        let mut q_out_arg = q_out_dev;
        let mut eps_arg = eps;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("q norm rope hd256 seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("q norm rope hd256 num_heads exceeds u32: {num_heads}"))?;
        let mut pos_start_arg = u32::try_from(pos_start)
            .map_err(|_| format!("q norm rope hd256 pos_start exceeds u32: {pos_start}"))?;
        let mut q_unit_arg = u32::from(q_unit_offset);
        self.launch_cached_gemv(
            "rnb_q_norm_rope_neox_hd256",
            &[
                (&mut q_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_sin_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_cos_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut q_unit_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg, 1),
            (128, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qk_norm_rope_neox_hd256_f16kv(
        &mut self,
        q_in_dev: u64,
        k_in_dev: u64,
        v_in_dev: u64,
        q_norm_dev: u64,
        k_norm_dev: u64,
        rope_sin_dev: u64,
        rope_cos_dev: u64,
        q_out_dev: u64,
        k_out_dev: u64,
        v_out_dev: u64,
        eps: f32,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        pos_start: usize,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
    ) -> Result<(), String> {
        let mut q_in_arg = q_in_dev;
        let mut k_in_arg = k_in_dev;
        let mut v_in_arg = v_in_dev;
        let mut q_norm_arg = q_norm_dev;
        let mut k_norm_arg = k_norm_dev;
        let mut rope_sin_arg = rope_sin_dev;
        let mut rope_cos_arg = rope_cos_dev;
        let mut q_out_arg = q_out_dev;
        let mut k_out_arg = k_out_dev;
        let mut v_out_arg = v_out_dev;
        let mut eps_arg = eps;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("qk norm rope hd256 seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("qk norm rope hd256 num_heads exceeds u32: {num_heads}"))?;
        let mut kv_heads_arg = u32::try_from(num_kv_heads)
            .map_err(|_| format!("qk norm rope hd256 num_kv_heads exceeds u32: {num_kv_heads}"))?;
        let mut pos_start_arg = u32::try_from(pos_start)
            .map_err(|_| format!("qk norm rope hd256 pos_start exceeds u32: {pos_start}"))?;
        let mut q_unit_arg = u32::from(q_unit_offset);
        let mut k_unit_arg = u32::from(k_unit_offset);
        let mut v_norm_arg = u32::from(v_no_scale_norm);
        self.launch_cached_gemv(
            "rnb_qk_norm_rope_neox_hd256_f16kv",
            &[
                (&mut q_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_sin_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_cos_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut q_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut k_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut v_norm_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg + kv_heads_arg, 1),
            (128, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qk_norm_rope_select_hd256_f16kv(
        &mut self,
        q_in_dev: u64,
        k_in_dev: u64,
        v_in_dev: u64,
        q_norm_dev: u64,
        k_norm_dev: u64,
        rope_sin_dev: u64,
        rope_cos_dev: u64,
        q_out_dev: u64,
        k_out_dev: u64,
        v_out_dev: u64,
        eps: f32,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        pos_start: usize,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
        rope_dim: usize,
        rope_neox: bool,
    ) -> Result<(), String> {
        let mut q_in_arg = q_in_dev;
        let mut k_in_arg = k_in_dev;
        let mut v_in_arg = v_in_dev;
        let mut q_norm_arg = q_norm_dev;
        let mut k_norm_arg = k_norm_dev;
        let mut rope_sin_arg = rope_sin_dev;
        let mut rope_cos_arg = rope_cos_dev;
        let mut q_out_arg = q_out_dev;
        let mut k_out_arg = k_out_dev;
        let mut v_out_arg = v_out_dev;
        let mut eps_arg = eps;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("qk norm rope select hd256 seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("qk norm rope select hd256 num_heads exceeds u32: {num_heads}"))?;
        let mut kv_heads_arg = u32::try_from(num_kv_heads).map_err(|_| {
            format!("qk norm rope select hd256 num_kv_heads exceeds u32: {num_kv_heads}")
        })?;
        let mut pos_start_arg = u32::try_from(pos_start)
            .map_err(|_| format!("qk norm rope select hd256 pos_start exceeds u32: {pos_start}"))?;
        let mut q_unit_arg = u32::from(q_unit_offset);
        let mut k_unit_arg = u32::from(k_unit_offset);
        let mut v_norm_arg = u32::from(v_no_scale_norm);
        let mut rope_dim_arg = u32::try_from(rope_dim)
            .map_err(|_| format!("qk norm rope select hd256 rope_dim exceeds u32: {rope_dim}"))?;
        let mut rope_neox_arg = u32::from(rope_neox);
        self.launch_cached_gemv(
            "rnb_qk_norm_rope_select_hd256_f16kv",
            &[
                (&mut q_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_sin_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_cos_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut q_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut k_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut v_norm_arg as *mut u32).cast::<libc::c_void>(),
                (&mut rope_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut rope_neox_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg + kv_heads_arg, 1),
            (128, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qk_norm_rope_select_hd512_f16kv(
        &mut self,
        q_in_dev: u64,
        k_in_dev: u64,
        v_in_dev: u64,
        q_norm_dev: u64,
        k_norm_dev: u64,
        rope_sin_dev: u64,
        rope_cos_dev: u64,
        q_out_dev: u64,
        k_out_dev: u64,
        v_out_dev: u64,
        eps: f32,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        pos_start: usize,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
        rope_dim: usize,
        rope_neox: bool,
    ) -> Result<(), String> {
        let mut q_in_arg = q_in_dev;
        let mut k_in_arg = k_in_dev;
        let mut v_in_arg = v_in_dev;
        let mut q_norm_arg = q_norm_dev;
        let mut k_norm_arg = k_norm_dev;
        let mut rope_sin_arg = rope_sin_dev;
        let mut rope_cos_arg = rope_cos_dev;
        let mut q_out_arg = q_out_dev;
        let mut k_out_arg = k_out_dev;
        let mut v_out_arg = v_out_dev;
        let mut eps_arg = eps;
        let mut seq_arg = u32::try_from(seq_len)
            .map_err(|_| format!("qk norm rope select hd512 seq_len exceeds u32: {seq_len}"))?;
        let mut heads_arg = u32::try_from(num_heads)
            .map_err(|_| format!("qk norm rope select hd512 num_heads exceeds u32: {num_heads}"))?;
        let mut kv_heads_arg = u32::try_from(num_kv_heads).map_err(|_| {
            format!("qk norm rope select hd512 num_kv_heads exceeds u32: {num_kv_heads}")
        })?;
        let mut pos_start_arg = u32::try_from(pos_start)
            .map_err(|_| format!("qk norm rope select hd512 pos_start exceeds u32: {pos_start}"))?;
        let mut q_unit_arg = u32::from(q_unit_offset);
        let mut k_unit_arg = u32::from(k_unit_offset);
        let mut v_norm_arg = u32::from(v_no_scale_norm);
        let mut rope_dim_arg = u32::try_from(rope_dim)
            .map_err(|_| format!("qk norm rope select hd512 rope_dim exceeds u32: {rope_dim}"))?;
        let mut rope_neox_arg = u32::from(rope_neox);
        self.launch_cached_gemv(
            "rnb_qk_norm_rope_select_hd512_f16kv",
            &[
                (&mut q_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_in_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_sin_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rope_cos_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut q_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut k_unit_arg as *mut u32).cast::<libc::c_void>(),
                (&mut v_norm_arg as *mut u32).cast::<libc::c_void>(),
                (&mut rope_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut rope_neox_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_arg, heads_arg + kv_heads_arg, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_gemv_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        rows: usize,
        selected: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_selected_gate_up_gemv",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, selected as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q2k_gate_up_gemv_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        rows: usize,
        selected: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q2k_selected_gate_up_gemv_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, selected as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_gemv_by_token_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp8 = tuning::mtp_verify_selected_gate_warp8_enabled();
        let warp_reduce = tuning::mtp_verify_selected_gate_warp_reduce_enabled();
        self.launch_cached_gemv(
            if warp8 {
                "rnb_q4k_selected_gate_up_gemv_by_token_warp8"
            } else if warp_reduce {
                "rnb_q4k_selected_gate_up_gemv_by_token_warp_reduce"
            } else {
                "rnb_q4k_selected_gate_up_gemv_by_token"
            },
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            if warp8 {
                ((rows as u32).saturating_add(7) / 8, slots as u32, 1)
            } else {
                (rows as u32, slots as u32, 1)
            },
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_gemv_pair2_by_token_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        expert_ids_dev: u64,
        pair_slots_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots_per_token: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
        fuse_silu: bool,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut pair_slots_arg = pair_slots_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows as u32;
        let mut slots_per_token_arg = slots_per_token as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut fuse_silu_arg = u32::from(fuse_silu);
        self.launch_cached_gemv(
            "rnb_q4k_selected_gate_up_gemv_by_token_pair2_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pair_slots_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut slots_per_token_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut fuse_silu_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                rows.div_ceil(8) as u32,
                slots_per_token.saturating_mul(2) as u32,
                1,
            ),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_q8dot_by_token_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_selected_gate_up_q8dot_by_token_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((rows as u32).saturating_add(7) / 8, slots as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_iq4_xs_gate_up_gemv_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        rows: usize,
        selected: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_iq4_xs_selected_gate_up_gemv",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, selected as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_iq2_xxs_gate_up_gemv_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        rows: usize,
        selected: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_iq2_xxs_selected_gate_up_gemv",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, selected as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_glm_iq_gate_up_gemv_by_token_to_dev(
        &mut self,
        kernel_name: &'static str,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, slots as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_glm_iq_gate_up_gemv_by_token_group4(
        &mut self,
        kernel_name: &'static str,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((rows as u32).div_ceil(4), groups as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_glm_iq_down_accum_by_token_group4(
        &mut self,
        kernel_name: &'static str,
        down_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        activation_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut down_weights_arg = down_ptrs_dev;
        let mut activation_arg = activation_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut down_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut activation_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((rows as u32).div_ceil(4), groups as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_gemv_by_token_group4_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp4 = std::env::var("RNB_CUDA_GROUP4_GATE_UP_WARP4")
            .ok()
            .as_deref()
            != Some("0");
        let group16 = warp4
            && std::env::var("RNB_CUDA_GROUP16_GATE_UP_WARP4")
                .ok()
                .as_deref()
                == Some("1");
        let group8 = warp4
            && std::env::var("RNB_CUDA_GROUP8_GATE_UP_WARP4")
                .ok()
                .as_deref()
                != Some("0");
        let kernel_name = if group16 {
            "rnb_q4k_selected_gate_up_gemv_by_token_group16_warp4"
        } else if group8 {
            "rnb_q4k_selected_gate_up_gemv_by_token_group8_warp4"
        } else if warp4 {
            "rnb_q4k_selected_gate_up_gemv_by_token_group4_warp4"
        } else {
            "rnb_q4k_selected_gate_up_gemv_by_token_group4"
        };
        let row8 = group8
            && !group16
            && std::env::var("RNB_CUDA_GROUP8_GATE_UP_ROW8")
                .ok()
                .as_deref()
                == Some("1");
        let rows_per_block = if row8 { 8u32 } else { 4u32 };
        let grid = if warp4 {
            (
                ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                groups as u32,
                1,
            )
        } else {
            (rows as u32, groups as u32, 1)
        };
        let block = if warp4 {
            (32, rows_per_block, 1)
        } else {
            (256, 1, 1)
        };
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_silu_by_token_group8_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let row8 = std::env::var("RNB_CUDA_GROUP8_GATE_UP_ROW8")
            .ok()
            .as_deref()
            == Some("1");
        let rows_per_block = if row8 { 8u32 } else { 4u32 };
        self.launch_cached_gemv(
            "rnb_q4k_selected_gate_up_silu_by_token_group8_warp4",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                groups as u32,
                1,
            ),
            (32, rows_per_block, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_silu_q8dot_by_token_group8_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        pack_group_offsets_dev: u64,
        rows: usize,
        groups: usize,
        input_blocks_per_row: usize,
        pack_blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        packed_dev: u64,
        q8_output: Option<(u64, u64)>,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        if rows > u32::MAX as usize
            || input_blocks_per_row > u32::MAX as usize
            || pack_blocks_per_row > u32::MAX as usize
        {
            return Err(format!(
                "Qwen35 Q4 gate/up group8 q8dot launcher dims exceed CUDA u32: rows={rows} input_blocks_per_row={input_blocks_per_row} pack_blocks_per_row={pack_blocks_per_row}"
            ));
        }

        let mut packed_arg = packed_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut pack_group_offsets_arg = pack_group_offsets_dev;
        let mut rows_arg = rows as u32;
        let mut input_blocks_per_row_arg = input_blocks_per_row as u32;
        let mut pack_blocks_per_row_arg = pack_blocks_per_row as u32;
        let (mut output_qs_arg, mut output_ds_arg) = q8_output.unwrap_or((0, 0));
        let row8 = std::env::var("RNB_CUDA_GROUP8_GATE_UP_ROW8")
            .ok()
            .as_deref()
            == Some("1");
        let rows_per_block = if row8 { 8u32 } else { 4u32 };
        let mmq = tuning::qwen35_q4_gate_up_q8dot_mmq_enabled();
        let group32 = tuning::qwen35_q4_gate_up_q8dot_mmq_group32_enabled();
        let group16 = tuning::qwen35_q4_gate_up_q8dot_mmq_group16_enabled()
            || std::env::var("RNB_CUDA_GROUP16_GATE_UP_WARP4")
                .ok()
                .as_deref()
                == Some("1");
        let (kernel, grid, block) = if mmq {
            (
                if group32 {
                    "rnb_q4k_selected_gate_up_silu_q8dot_mmq_group32"
                } else if group16 {
                    "rnb_q4k_selected_gate_up_silu_q8dot_mmq_group16"
                } else {
                    "rnb_q4k_selected_gate_up_silu_q8dot_mmq_group8"
                },
                (((rows as u32).saturating_add(31)) / 32, groups as u32, 1),
                (if group32 { 256 } else { 64 }, 1, 1),
            )
        } else {
            (
                "rnb_q4k_selected_gate_up_silu_q8dot_by_token_group8_warp4",
                (
                    ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                    groups as u32,
                    1,
                ),
                (32, rows_per_block, 1),
            )
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut packed_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pack_group_offsets_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut input_blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pack_blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut output_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_ds_arg as *mut u64).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group4_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        input_blocks_per_row: usize,
        pack_blocks_per_row: usize,
        input_dev: u64,
        packed_dev: u64,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        if rows > u32::MAX as usize
            || input_blocks_per_row > u32::MAX as usize
            || pack_blocks_per_row > u32::MAX as usize
        {
            return Err(format!(
                "Qwen35 Q4 gate/up pack4 launcher dims exceed CUDA u32: rows={rows} input_blocks_per_row={input_blocks_per_row} pack_blocks_per_row={pack_blocks_per_row}"
            ));
        }

        let mut packed_arg = packed_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut input_blocks_per_row_arg = input_blocks_per_row as u32;
        let mut pack_blocks_per_row_arg = pack_blocks_per_row as u32;
        let rows_per_block = 4u32;
        self.launch_cached_gemv(
            "rnb_q4k_selected_gate_up_silu_pack4_f32_by_token_group4_warp4",
            &[
                (&mut packed_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut input_blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pack_blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                groups as u32,
                1,
            ),
            (32, rows_per_block, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group8_to_dev(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        pack_group_offsets_dev: u64,
        rows: usize,
        groups: usize,
        input_blocks_per_row: usize,
        pack_blocks_per_row: usize,
        input_dev: u64,
        packed_dev: u64,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        if rows > u32::MAX as usize
            || input_blocks_per_row > u32::MAX as usize
            || pack_blocks_per_row > u32::MAX as usize
        {
            return Err(format!(
                "Qwen35 Q4 gate/up group8 pack4 launcher dims exceed CUDA u32: rows={rows} input_blocks_per_row={input_blocks_per_row} pack_blocks_per_row={pack_blocks_per_row}"
            ));
        }

        let mut packed_arg = packed_dev;
        let mut gate_weights_arg = gate_ptrs_dev;
        let mut up_weights_arg = up_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut pack_group_offsets_arg = pack_group_offsets_dev;
        let mut rows_arg = rows as u32;
        let mut input_blocks_per_row_arg = input_blocks_per_row as u32;
        let mut pack_blocks_per_row_arg = pack_blocks_per_row as u32;
        let row8 = std::env::var("RNB_CUDA_GROUP8_GATE_UP_ROW8")
            .ok()
            .as_deref()
            == Some("1");
        let rows_per_block = if row8 { 8u32 } else { 4u32 };
        self.launch_cached_gemv(
            "rnb_q4k_selected_gate_up_silu_pack4_f32_by_token_group8_warp4",
            &[
                (&mut packed_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pack_group_offsets_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut input_blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pack_blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                groups as u32,
                1,
            ),
            (32, rows_per_block, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_down_silu_rowreduce(
        &mut self,
        kernel_name: &str,
        weight_ptrs_dev: u64,
        rows: usize,
        selected: usize,
        blocks_per_row: usize,
        gate_dev: u64,
        up_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut route_arg = route_dev;
        let mut rows_arg = rows as u32;
        let mut selected_arg = selected as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp8 = tuning::mtp_verify_selected_down_warp8_enabled()
            && matches!(
                kernel_name,
                "rnb_q4k_selected_down_silu_rowreduce" | "rnb_q6k_selected_down_silu_rowreduce"
            );
        let warp8_kernel = match kernel_name {
            "rnb_q4k_selected_down_silu_rowreduce" => "rnb_q4k_selected_down_silu_rowreduce_warp8",
            "rnb_q6k_selected_down_silu_rowreduce" => "rnb_q6k_selected_down_silu_rowreduce_warp8",
            _ => kernel_name,
        };
        self.launch_cached_gemv(
            if warp8 { warp8_kernel } else { kernel_name },
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut selected_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            if warp8 {
                ((rows as u32).saturating_add(7) / 8, 1, 1)
            } else {
                (rows as u32, 1, 1)
            },
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_down_silu_rowreduce_by_token(
        &mut self,
        kernel_name: &str,
        weight_ptrs_dev: u64,
        rows: usize,
        slots_per_token: usize,
        token_count: usize,
        blocks_per_row: usize,
        gate_dev: u64,
        up_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut route_arg = route_dev;
        let mut rows_arg = rows as u32;
        let mut slots_per_token_arg = slots_per_token as u32;
        let mut token_count_arg = token_count as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut slots_per_token_arg as *mut u32).cast::<libc::c_void>(),
                (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, token_count as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q3k_selected_down_silu_per_slot(
        &mut self,
        weight_ptrs_dev: u64,
        rows: usize,
        selected: usize,
        blocks_per_row: usize,
        gate_dev: u64,
        up_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q3k_selected_down_silu_per_slot",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, selected as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_down_accum_by_token(
        &mut self,
        kernel_name: &str,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, slots as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_q5_1_selected_down_accum(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if tuning::nemotron_q5_down_warp4_enabled() {
            self.launch_selected_down_accum_by_token_warp4(
                "rnb_q5_1_selected_down_accum_by_token_warp4",
                weight_ptrs_dev,
                token_ids_dev,
                rows,
                slots,
                blocks_per_row,
                input_dev,
                route_dev,
                output_dev,
            )
        } else {
            self.launch_selected_down_accum_by_token(
                "rnb_q5_1_selected_down_accum_by_token",
                weight_ptrs_dev,
                token_ids_dev,
                rows,
                slots,
                blocks_per_row,
                input_dev,
                route_dev,
                output_dev,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_q8_0_selected_down_accum(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if tuning::nemotron_q8_down_warp4_enabled() {
            self.launch_selected_down_accum_by_token_warp4(
                "rnb_q8_0_selected_down_accum_by_token_warp4",
                weight_ptrs_dev,
                token_ids_dev,
                rows,
                slots,
                blocks_per_row,
                input_dev,
                route_dev,
                output_dev,
            )
        } else {
            self.launch_selected_down_accum_by_token(
                "rnb_q8_0_selected_down_accum_by_token",
                weight_ptrs_dev,
                token_ids_dev,
                rows,
                slots,
                blocks_per_row,
                input_dev,
                route_dev,
                output_dev,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_q5_0_selected_relu_sqr(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q5_0_selected_relu_sqr_by_token",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, slots as u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_q5_0_selected_relu_sqr_group4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let rows_per_block = 4u32;
        self.launch_cached_nemotron_selected(
            "rnb_q5_0_selected_relu_sqr_by_token_group4_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                groups as u32,
                1,
            ),
            (32, rows_per_block, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_q5_1_selected_down_accum_group4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_nemotron_selected_down_accum_group4(
            "rnb_q5_1_selected_down_accum_by_token_group4_warp4",
            weight_ptrs_dev,
            token_ids_dev,
            group_meta_dev,
            rows,
            groups,
            blocks_per_row,
            input_dev,
            route_dev,
            output_dev,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_q8_0_selected_down_accum_group4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_nemotron_selected_down_accum_group4(
            "rnb_q8_0_selected_down_accum_by_token_group4_warp4",
            weight_ptrs_dev,
            token_ids_dev,
            group_meta_dev,
            rows,
            groups,
            blocks_per_row,
            input_dev,
            route_dev,
            output_dev,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_nemotron_selected_down_accum_group4(
        &mut self,
        kernel_name: &str,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let rows_per_block = 4u32;
        self.launch_cached_nemotron_selected(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                groups as u32,
                1,
            ),
            (32, rows_per_block, 1),
        )
    }

    pub(in crate::runtime) fn launch_basic_gemv_to_dev(
        &mut self,
        kernel_name: &str,
        weights_dev: u64,
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_nemotron_q8_shared_gemv_to_dev(
        &mut self,
        kernel_name: &str,
        warp4_kernel_name: &str,
        weights_dev: u64,
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if tuning::nemotron_q8_shared_warp4_enabled() {
            let mut output_arg = output_dev;
            let mut weights_arg = weights_dev;
            let mut input_arg = input_dev;
            let mut rows_arg = rows as u32;
            let mut blocks_per_row_arg = blocks_per_row as u32;
            self.launch_cached_gemv(
                warp4_kernel_name,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (rows.div_ceil(4) as u32, 1, 1),
                (32, 4, 1),
            )
        } else {
            self.launch_basic_gemv_to_dev(
                kernel_name,
                weights_dev,
                rows,
                blocks_per_row,
                input_dev,
                output_dev,
            )
        }
    }

    pub(in crate::runtime) fn launch_q8_0_dequant_f32_to_dev(
        &mut self,
        weights_dev: u64,
        output_dev: u64,
        rows: usize,
        blocks_per_row: usize,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q8_0_dequant_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, blocks_per_row as u32, 1),
            (32, 1, 1),
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn launch_q8_0_gemv_batch_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if crate::tuning::q8_0_mmq_tile32_enabled(seq_len, rows, blocks_per_row) {
            let input_len = seq_len
                .checked_mul(blocks_per_row)
                .and_then(|value| value.checked_mul(32))
                .ok_or_else(|| "Q8_0 device-input MMQ input length overflow".to_string())?;
            let qs_bytes = input_len;
            let ds_bytes = (input_len / 32)
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "Q8_0 device-input MMQ scale byte size overflow".to_string())?;
            let ds_offset = qs_bytes.next_multiple_of(256);
            let slab_bytes = (ds_offset + ds_bytes).next_multiple_of(256);
            let slab = self.compute_temp_slab_ptr(slab_bytes)?;
            let input_qs_dev = slab;
            let input_ds_dev = slab + ds_offset as u64;
            self.launch_quantize_q8_1_by_32(input_dev, input_qs_dev, input_ds_dev, input_len)?;
            return self.launch_q8_0_q8_1_matmul_mmq_tile32(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_qs_dev,
                input_ds_dev,
                output_dev,
            );
        }
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| format!("Q8_0 batch rows exceeds CUDA u32 limit: {rows}"))?;
        let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            format!("Q8_0 batch blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}")
        })?;
        let seq_u32 = u32::try_from(seq_len)
            .map_err(|_| format!("Q8_0 batch seq_len exceeds CUDA u32 limit: {seq_len}"))?;
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows_u32;
        let mut blocks_per_row_arg = blocks_u32;
        let mut seq_arg = seq_u32;
        let token2 = crate::tuning::q8_0_gemv_batch_token2_enabled(seq_len);
        let kernel_name = if token2 {
            "rnb_q8_0_gemv_batch_token2"
        } else {
            "rnb_q8_0_gemv_batch"
        };
        let grid = if token2 {
            (rows_u32, 1, 1)
        } else {
            (rows_u32, seq_u32, 1)
        };
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            (if token2 { 128 } else { 256 }, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q8_0_gemv_batch_token2_multi3_to_dev(
        &mut self,
        weights: [&[u8]; 3],
        rows: [usize; 3],
        blocks_per_row: usize,
        input_dev: u64,
        outputs: [u64; 3],
    ) -> Result<(), String> {
        let total_rows = rows
            .into_iter()
            .try_fold(0usize, |total, rows| total.checked_add(rows))
            .ok_or_else(|| "Q8_0 token2 multi-projection row count overflow".to_string())?;
        let total_rows_u32 = u32::try_from(total_rows).map_err(|_| {
            format!("Q8_0 token2 multi-projection rows exceeds CUDA u32 limit: {total_rows}")
        })?;
        let rows_u32 = rows.map(|rows| rows as u32);
        if rows_u32
            .iter()
            .zip(rows.iter())
            .any(|(&narrow, &wide)| narrow as usize != wide)
        {
            return Err(
                "Q8_0 token2 multi-projection matrix rows exceeds CUDA u32 limit".to_string(),
            );
        }
        let mut weight_ptrs = [0u64; 3];
        for index in 0..3 {
            weight_ptrs[index] = if rows[index] == 0 {
                weight_ptrs[0]
            } else {
                self.resident_q4k_weights_ptr(weights[index])?
            };
        }
        let mut output0 = outputs[0];
        let mut weights0 = weight_ptrs[0];
        let mut rows0 = rows_u32[0];
        let mut output1 = outputs[1];
        let mut weights1 = weight_ptrs[1];
        let mut rows1 = rows_u32[1];
        let mut output2 = outputs[2];
        let mut weights2 = weight_ptrs[2];
        let mut rows2 = rows_u32[2];
        let mut input = input_dev;
        let mut blocks = u32::try_from(blocks_per_row).map_err(|_| {
            format!(
                "Q8_0 token2 multi-projection blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}"
            )
        })?;
        let mut seq_len = 2u32;
        self.launch_cached_gemv(
            "rnb_q8_0_gemv_batch_token2_multi3",
            &[
                (&mut output0 as *mut u64).cast::<libc::c_void>(),
                (&mut weights0 as *mut u64).cast::<libc::c_void>(),
                (&mut rows0 as *mut u32).cast::<libc::c_void>(),
                (&mut output1 as *mut u64).cast::<libc::c_void>(),
                (&mut weights1 as *mut u64).cast::<libc::c_void>(),
                (&mut rows1 as *mut u32).cast::<libc::c_void>(),
                (&mut output2 as *mut u64).cast::<libc::c_void>(),
                (&mut weights2 as *mut u64).cast::<libc::c_void>(),
                (&mut rows2 as *mut u32).cast::<libc::c_void>(),
                (&mut input as *mut u64).cast::<libc::c_void>(),
                (&mut blocks as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len as *mut u32).cast::<libc::c_void>(),
            ],
            (total_rows_u32, 1, 1),
            (128, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_f32_gemv_batch_token2_multi2_to_dev(
        &mut self,
        weights: [&[f32]; 2],
        rows: [usize; 2],
        cols: usize,
        input_dev: u64,
        outputs: [u64; 2],
    ) -> Result<(), String> {
        let total_rows = rows[0]
            .checked_add(rows[1])
            .ok_or_else(|| "F32 token2 multi-projection row count overflow".to_string())?;
        let total_rows_u32 = u32::try_from(total_rows).map_err(|_| {
            format!("F32 token2 multi-projection rows exceeds CUDA u32 limit: {total_rows}")
        })?;
        let mut rows0 = u32::try_from(rows[0]).map_err(|_| {
            format!(
                "F32 token2 multi-projection first matrix rows exceeds CUDA u32 limit: {}",
                rows[0]
            )
        })?;
        let mut rows1 = u32::try_from(rows[1]).map_err(|_| {
            format!(
                "F32 token2 multi-projection second matrix rows exceeds CUDA u32 limit: {}",
                rows[1]
            )
        })?;
        let mut cols = u32::try_from(cols).map_err(|_| {
            format!("F32 token2 multi-projection columns exceeds CUDA u32 limit: {cols}")
        })?;
        let mut weights0 = self.resident_f32_ptr(weights[0])?;
        let mut weights1 = self.resident_f32_ptr(weights[1])?;
        let mut output0 = outputs[0];
        let mut output1 = outputs[1];
        let mut input = input_dev;
        let mut seq_len = 2u32;
        self.launch_cached_gemv(
            "rnb_f32_gemv_batch_token2_multi2",
            &[
                (&mut output0 as *mut u64).cast::<libc::c_void>(),
                (&mut weights0 as *mut u64).cast::<libc::c_void>(),
                (&mut rows0 as *mut u32).cast::<libc::c_void>(),
                (&mut output1 as *mut u64).cast::<libc::c_void>(),
                (&mut weights1 as *mut u64).cast::<libc::c_void>(),
                (&mut rows1 as *mut u32).cast::<libc::c_void>(),
                (&mut input as *mut u64).cast::<libc::c_void>(),
                (&mut cols as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len as *mut u32).cast::<libc::c_void>(),
            ],
            (total_rows_u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_relu_sqr_inplace(
        &mut self,
        values_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut values_arg = values_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_relu_sqr_inplace",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (len.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_f32_gemv_to_dev(
        &mut self,
        weights_dev: u64,
        rows: usize,
        cols: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let stream = self.stream;
        let cublas = self.cublas_state_mut()?;
        unsafe {
            cublas.api.set_stream(cublas.handle, stream)?;
            cublas.api.sgemm(
                cublas.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                rows as i32,
                1,
                cols as i32,
                1.0,
                weights_dev,
                cols as i32,
                input_dev,
                cols as i32,
                0.0,
                output_dev,
                rows as i32,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_down_accum_by_token_group4(
        &mut self,
        kernel_name: &str,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp4 = std::env::var("RNB_CUDA_GROUP4_DOWN_WARP4").ok().as_deref() != Some("0");
        let kernel_name = if warp4 {
            let group8 = std::env::var("RNB_CUDA_GROUP8_DOWN_WARP4").ok().as_deref() == Some("1");
            match kernel_name {
                "rnb_q4k_selected_down_accum_by_token_group4" if group8 => {
                    "rnb_q4k_selected_down_accum_by_token_group8_warp4"
                }
                "rnb_q5k_selected_down_accum_by_token_group4" if group8 => {
                    "rnb_q5k_selected_down_accum_by_token_group8_warp4"
                }
                "rnb_q6k_selected_down_accum_by_token_group4" if group8 => {
                    "rnb_q6k_selected_down_accum_by_token_group8_warp4"
                }
                "rnb_q4k_selected_down_accum_by_token_group4"
                    if tuning::group2_down_warp4_enabled() =>
                {
                    "rnb_q4k_selected_down_accum_by_token_group2_warp4"
                }
                "rnb_q5k_selected_down_accum_by_token_group4"
                    if tuning::group2_down_warp4_enabled() =>
                {
                    "rnb_q5k_selected_down_accum_by_token_group2_warp4"
                }
                "rnb_q6k_selected_down_accum_by_token_group4"
                    if tuning::group2_down_warp4_enabled() =>
                {
                    "rnb_q6k_selected_down_accum_by_token_group2_warp4"
                }
                "rnb_q6k_selected_down_accum_by_token_group4"
                    if tuning::q6k_group4_down_lowreg_enabled() =>
                {
                    "rnb_q6k_selected_down_accum_by_token_group4_lowreg_warp4"
                }
                "rnb_q6k_selected_down_accum_by_token_group4"
                    if tuning::qwen35_q6_down_full4_fastpath_enabled() =>
                {
                    "rnb_q6k_selected_down_accum_by_token_group4_fast4_warp4"
                }
                "rnb_q4k_selected_down_accum_by_token_group4" => {
                    "rnb_q4k_selected_down_accum_by_token_group4_warp4"
                }
                "rnb_q5k_selected_down_accum_by_token_group4" => {
                    "rnb_q5k_selected_down_accum_by_token_group4_warp4"
                }
                "rnb_q6k_selected_down_accum_by_token_group4" => {
                    "rnb_q6k_selected_down_accum_by_token_group4_warp4"
                }
                other => other,
            }
        } else {
            kernel_name
        };
        let row8 = warp4 && tuning::group4_down_row8_enabled();
        let rows_per_block = if row8 { 8u32 } else { 4u32 };
        let grid = if warp4 {
            (
                ((rows as u32).saturating_add(rows_per_block - 1)) / rows_per_block,
                groups as u32,
                1,
            )
        } else {
            (rows as u32, groups as u32, 1)
        };
        let block = if warp4 {
            (32, rows_per_block, 1)
        } else {
            (256, 1, 1)
        };
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q6k_down_accum_by_token_group4_pack4_f32_warp4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let kernel = if tuning::qwen35_q6_down_pack4_f32_vec4_enabled() {
            "rnb_q6k_selected_down_accum_by_token_group4_pack4_f32_vec4_warp4"
        } else {
            "rnb_q6k_selected_down_accum_by_token_group4_pack4_f32_warp4"
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, groups as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_down_accum_by_token_warp4(
        &mut self,
        kernel_name: &str,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        rows: usize,
        slots: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, slots as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q5k_down_accum_by_token_pair2_warp4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        expert_ids_dev: u64,
        pair_slots_dev: u64,
        rows: usize,
        slots_per_token: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut pair_slots_arg = pair_slots_dev;
        let mut rows_arg = rows as u32;
        let mut slots_per_token_arg = slots_per_token as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q5k_selected_down_accum_by_token_pair2_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pair_slots_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut slots_per_token_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (
                ((rows as u32).saturating_add(3)) / 4,
                (slots_per_token * 2) as u32,
                1,
            ),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q6k_down_accum_run_batched_ref_warp4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        run_tile_meta_dev: u64,
        rows: usize,
        run_tiles: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if run_tiles == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut run_tile_meta_arg = run_tile_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_selected_down_accum_run_batched_ref_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut run_tile_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, run_tiles as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q6k_down_accum_run_batched8_warp4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        run_tile_meta_dev: u64,
        rows: usize,
        run_tiles: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if run_tiles == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut run_tile_meta_arg = run_tile_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_selected_down_accum_by_token_group8_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut run_tile_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, run_tiles as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q6k_down_accum_run_tiled4_warp4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        run_tile_meta_dev: u64,
        rows: usize,
        run_tiles: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if run_tiles == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut run_tile_meta_arg = run_tile_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_selected_down_accum_run_tiled4_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut run_tile_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, run_tiles as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q6k_down_accum_token_major_warp4(
        &mut self,
        weight_ptrs_dev: u64,
        token_offsets_dev: u64,
        slot_indices_dev: u64,
        rows: usize,
        tokens: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if tokens == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_offsets_arg = token_offsets_dev;
        let mut slot_indices_arg = slot_indices_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_selected_down_accum_token_major_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_offsets_arg as *mut u64).cast::<libc::c_void>(),
                (&mut slot_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, tokens as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q6k_down_accum_by_token_group4_full_warp4(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_arg = input_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_selected_down_accum_by_token_group4_full_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, groups as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q6k_down_accum_by_token_group4_q8dot(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_selected_down_accum_by_token_group4_q8dot_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, groups as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q4k_down_accum_by_token_group4_q8dot(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_selected_down_accum_by_token_group4_q8dot_warp4",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(3)) / 4, groups as u32, 1),
            (32, 4, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_selected_q5k_down_accum_q8dot_mmq_group16(
        &mut self,
        weight_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        rows: usize,
        groups: usize,
        token_count: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        route_dev: u64,
        slot_outputs_dev: u64,
        token_offsets_dev: u64,
        slot_indices_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        let mut slot_outputs_arg = slot_outputs_dev;
        let mut weights_arg = weight_ptrs_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut route_arg = route_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let group32 = tuning::qwen35_q5_down_q8dot_mmq_group32_enabled();
        let group64 = group32 && tuning::qwen35_q5_down_q8dot_mmq_group64_enabled();
        self.launch_cached_gemv(
            if group32 {
                "rnb_q5k_selected_down_accum_q8dot_mmq_group32"
            } else {
                "rnb_q5k_selected_down_accum_q8dot_mmq_group16"
            },
            &[
                (&mut slot_outputs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((rows as u32).saturating_add(31)) / 32, groups as u32, 1),
            (
                if group64 {
                    512
                } else if group32 {
                    256
                } else {
                    64
                },
                1,
                1,
            ),
        )?;
        let mut output_arg = output_dev;
        let mut token_offsets_arg = token_offsets_dev;
        let mut slot_indices_arg = slot_indices_dev;
        self.launch_cached_gemv(
            "rnb_q5k_selected_down_reduce_slots_deterministic",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut slot_outputs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_offsets_arg as *mut u64).cast::<libc::c_void>(),
                (&mut slot_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(256) as u32, token_count as u32, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q2k_gemv_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q2k_gemv_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q3k_gemv_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q3k_gemv_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_gemv_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let (kernel, grid_x) = if tuning::q4k_gemv_warp8_enabled() {
            ("rnb_q4k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q4k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_dequant_f16_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_dequant_f16",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_dequant_f32_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_dequant_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q6k_dequant_f16_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        // Q6_K raw weight goes through the same Q4_K resident cache (key hashed
        // on host ptr+len), so prewarm_all_dense_q4k_raw_weights doesn't help
        // here — Q6_K weights aren't in that prewarm list. Cache may miss the
        // first time; resident_q4k_weights_ptr handles the upload + insert.
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_dequant_f16",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q6k_dequant_f32_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        // cu26 phase A: GPU dequant for Q6_K → f32 cache enrollment, matching
        // the cu22 q4_f32_cache pattern. Same Q4_K resident cache hosts the
        // raw Q6_K bytes via resident_q4k_weights_ptr.
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_dequant_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn launch_q4k_embedding_gather_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_q4k_embedding_gather_to_dev_inner(
            weights,
            rows,
            blocks_per_row,
            token_ids_dev,
            token_count,
            output_dev,
            false,
        )
    }

    pub(in crate::runtime) fn launch_q4k_embedding_gather_to_dev_pinned(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_q4k_embedding_gather_to_dev_inner(
            weights,
            rows,
            blocks_per_row,
            token_ids_dev,
            token_count,
            output_dev,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_q4k_embedding_gather_to_dev_inner(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
        pinned: bool,
    ) -> Result<(), String> {
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| format!("Q4_K embedding rows exceeds CUDA u32 limit: {rows}"))?;
        let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            format!("Q4_K embedding blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}")
        })?;
        let token_count_u32 = u32::try_from(token_count).map_err(|_| {
            format!("Q4_K embedding token_count exceeds CUDA u32 limit: {token_count}")
        })?;
        let weights_dev = if pinned {
            self.resident_q4k_weights_ptr_pinned(weights)?
        } else {
            self.resident_q4k_weights_ptr(weights)?
        };
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows_u32;
        let mut blocks_per_row_arg = blocks_u32;
        let mut token_count_arg = token_count_u32;
        self.launch_cached_gemv(
            "rnb_q4k_embedding_gather_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (token_count_u32, blocks_u32, 1),
            (256, 1, 1),
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn launch_q6k_embedding_gather_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_q6k_embedding_gather_to_dev_inner(
            weights,
            rows,
            blocks_per_row,
            token_ids_dev,
            token_count,
            output_dev,
            false,
        )
    }

    pub(in crate::runtime) fn launch_q6k_embedding_gather_to_dev_pinned(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_q6k_embedding_gather_to_dev_inner(
            weights,
            rows,
            blocks_per_row,
            token_ids_dev,
            token_count,
            output_dev,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_q6k_embedding_gather_to_dev_inner(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
        pinned: bool,
    ) -> Result<(), String> {
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| format!("Q6_K embedding rows exceeds CUDA u32 limit: {rows}"))?;
        let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            format!("Q6_K embedding blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}")
        })?;
        let token_count_u32 = u32::try_from(token_count).map_err(|_| {
            format!("Q6_K embedding token_count exceeds CUDA u32 limit: {token_count}")
        })?;
        let weights_dev = if pinned {
            self.resident_q4k_weights_ptr_pinned(weights)?
        } else {
            self.resident_q4k_weights_ptr(weights)?
        };
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows_u32;
        let mut blocks_per_row_arg = blocks_u32;
        let mut token_count_arg = token_count_u32;
        self.launch_cached_gemv(
            "rnb_q6k_embedding_gather_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (token_count_u32, blocks_u32, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q8_0_embedding_gather_to_dev_pinned(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| format!("Q8_0 embedding rows exceeds CUDA u32 limit: {rows}"))?;
        let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            format!("Q8_0 embedding blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}")
        })?;
        let token_count_u32 = u32::try_from(token_count).map_err(|_| {
            format!("Q8_0 embedding token_count exceeds CUDA u32 limit: {token_count}")
        })?;
        let weights_dev = self.resident_q4k_weights_ptr_pinned(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows_u32;
        let mut blocks_per_row_arg = blocks_u32;
        let mut token_count_arg = token_count_u32;
        self.launch_cached_gemv(
            "rnb_q8_0_embedding_gather_f32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (token_count_u32, blocks_u32, 1),
            (32, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_quant_embedding_gather_to_dev(
        &mut self,
        kernel_name: &str,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        block_elems: usize,
        token_ids_dev: u64,
        token_count: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| format!("embedding rows exceeds CUDA u32 limit: {rows}"))?;
        let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            format!("embedding blocks_per_row exceeds CUDA u32 limit: {blocks_per_row}")
        })?;
        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| format!("embedding token_count exceeds CUDA u32 limit: {token_count}"))?;
        let threads_u32 = u32::try_from(block_elems)
            .map_err(|_| format!("embedding block width exceeds CUDA u32 limit: {block_elems}"))?;
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut token_ids_arg = token_ids_dev;
        let mut rows_arg = rows_u32;
        let mut blocks_per_row_arg = blocks_u32;
        let mut token_count_arg = token_count_u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut token_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (token_count_u32, blocks_u32, 1),
            (threads_u32, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_gemv_gelu_mul_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        mul_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut mul_arg = mul_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_gemv_gelu_mul_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut mul_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_gemv_q8dot_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_gemv_q8dot_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_packed_gemv_q8dot_to_dev(
        &mut self,
        weights_dev: u64,
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let use_warp4 = blocks_per_row >= 16 && tuning::q4k_packed_gemv_warp4_enabled();
        let (kernel, grid, block) = if use_warp4 {
            (
                "rnb_q4k_packed_gemv_q8dot_warp4",
                (rows as u32, 1, 1),
                (32, 4, 1),
            )
        } else {
            (
                "rnb_q4k_packed_gemv_q8dot_warp8",
                (rows.div_ceil(8) as u32, 1, 1),
                (256, 1, 1),
            )
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    pub(in crate::runtime) fn launch_q4k_gate_up_gemv_to_dev(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let gate_weights_dev = self.resident_q4k_weights_ptr(gate_weights)?;
        let up_weights_dev = self.resident_q4k_weights_ptr(up_weights)?;
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_weights_dev;
        let mut up_weights_arg = up_weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_gate_up_gemv_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q4k_gate_up_gemv_q8dot_to_dev(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let gate_weights_dev = self.resident_q4k_weights_ptr(gate_weights)?;
        let up_weights_dev = self.resident_q4k_weights_ptr(up_weights)?;
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_weights_dev;
        let mut up_weights_arg = up_weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_gate_up_gemv_q8dot_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q4k_gate_up_gemv_batch_seq2_q8dot_to_dev(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let gate_weights_dev = self.resident_q4k_weights_ptr(gate_weights)?;
        let up_weights_dev = self.resident_q4k_weights_ptr(up_weights)?;
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_weights_dev;
        let mut up_weights_arg = up_weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_gate_up_gemv_batch_seq2_q8dot_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q4k_packed_gate_up_gemv_q8dot_to_dev(
        &mut self,
        gate_weights_dev: u64,
        up_weights_dev: u64,
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_weights_dev;
        let mut up_weights_arg = up_weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_packed_gate_up_gemv_q8dot_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q4k_packed_gate_up_gemv_batch_seq2_q8dot_to_dev(
        &mut self,
        gate_weights_dev: u64,
        up_weights_dev: u64,
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        gate_dev: u64,
        up_dev: u64,
    ) -> Result<(), String> {
        let mut gate_out_arg = gate_dev;
        let mut up_out_arg = up_dev;
        let mut gate_weights_arg = gate_weights_dev;
        let mut up_weights_arg = up_weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q4k_packed_gate_up_gemv_batch_seq2_q8dot_warp8",
            &[
                (&mut gate_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q4k_qkv_gemv_to_dev(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        q_dev: u64,
        k_dev: u64,
        v_dev: u64,
    ) -> Result<(), String> {
        // cu27: 다중 register OOM race 보호 (q8dot path와 동일).
        let q_weights_dev = self.resident_q4k_weights_ptr_pinned(q_weights)?;
        let k_weights_dev = self.resident_q4k_weights_ptr_pinned(k_weights)?;
        let v_weights_dev = self.resident_q4k_weights_ptr_pinned(v_weights)?;
        let mut q_out_arg = q_dev;
        let mut k_out_arg = k_dev;
        let mut v_out_arg = v_dev;
        let mut q_weights_arg = q_weights_dev;
        let mut k_weights_arg = k_weights_dev;
        let mut v_weights_arg = v_weights_dev;
        let mut input_arg = input_dev;
        let mut q_rows_arg = q_rows as u32;
        let mut kv_rows_arg = kv_rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let total_rows = q_rows + kv_rows * 2;
        let result = self.launch_cached_gemv(
            "rnb_q4k_qkv_gemv_warp8",
            &[
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (total_rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        );
        self.unpin_resident_q4k(q_weights);
        self.unpin_resident_q4k(k_weights);
        self.unpin_resident_q4k(v_weights);
        result
    }

    #[allow(clippy::too_many_arguments)]
    // cu38 Phase 3: 4-warp 32x32 output tile (occupancy 향상). block = 128
    // thread. grid = (rows/32, seq_len/32, 1).
    pub(in crate::runtime) fn launch_q4k_sgemm_fused_wmma_4warp(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_f16_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_f16_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(32) as u32;
        let grid_y = seq_len.div_ceil(32) as u32;
        self.launch_cached_gemv(
            "rnb_q4k_sgemm_fused_wmma_4warp",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (128, 1, 1),
        )
    }

    // cu38 Phase 2: Q4K + cuBLAS fused matmul launcher (wmma tensor core).
    // Ampere SM 80+ mma.m16n8k16.f16.f16. grid = (rows/16, seq_len/16, 1),
    // block = (32, 1, 1) (1 warp).
    pub(in crate::runtime) fn launch_q4k_sgemm_fused_wmma(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_f16_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_f16_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(16) as u32;
        let grid_y = seq_len.div_ceil(16) as u32;
        self.launch_cached_gemv(
            "rnb_q4k_sgemm_fused_wmma",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (32, 1, 1),
        )
    }

    // cu38 Phase 1: Q4K + cuBLAS fused matmul launcher (naive). Phase 3 dispatcher
    // 미연결 (dead_code). Phase 2-3 shared mem tile + tensor core + dispatcher
    // 후 활성화.
    #[allow(dead_code)]
    pub(in crate::runtime) fn launch_q4k_sgemm_fused_naive(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        self.launch_cached_gemv(
            "rnb_q4k_sgemm_fused_naive",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, seq_len as u32, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_qkv_gemv_q8dot_to_dev(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        q_dev: u64,
        k_dev: u64,
        v_dev: u64,
    ) -> Result<(), String> {
        // cu27: q→k→v 등록 시퀀스 안에서 OOM이 나면 offload_non_pinned_resident_q4k가
        // 호출되어 모든 non-pinned 가 free됨. 직전 등록한 Q/K weight도 함께 사라져서
        // launch 시 dangling ptr → cuLaunchKernel CUDA 700. 임시 pin으로 보호.
        let q_weights_dev = self.resident_q4k_weights_ptr_pinned(q_weights)?;
        let k_weights_dev = self.resident_q4k_weights_ptr_pinned(k_weights)?;
        let v_weights_dev = self.resident_q4k_weights_ptr_pinned(v_weights)?;
        let mut q_out_arg = q_dev;
        let mut k_out_arg = k_dev;
        let mut v_out_arg = v_dev;
        let mut q_weights_arg = q_weights_dev;
        let mut k_weights_arg = k_weights_dev;
        let mut v_weights_arg = v_weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut q_rows_arg = q_rows as u32;
        let mut kv_rows_arg = kv_rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let total_rows = q_rows + kv_rows * 2;
        let result = self.launch_cached_gemv(
            "rnb_q4k_qkv_gemv_q8dot_warp8",
            &[
                (&mut q_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_out_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (total_rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        );
        // cu27: launch enqueue 후 즉시 unpin. 다음 OOM offload가 이 weight를
        // free할 수 있게 됨 (stream sync로 in-flight kernel 보장됨).
        self.unpin_resident_q4k(q_weights);
        self.unpin_resident_q4k(k_weights);
        self.unpin_resident_q4k(v_weights);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_quant_gemv_batch_to_dev(
        &mut self,
        kernel: &'static str,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_count: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut token_count_arg = token_count as u32;
        let (kernel, grid_x) =
            if kernel == "rnb_q6k_gemv_batch" && tuning::q6k_gemv_batch_warp8_enabled() {
                ("rnb_q6k_gemv_batch_warp8", rows.div_ceil(8) as u32)
            } else {
                (kernel, rows as u32)
            };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut token_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, token_count_arg, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q5k_gemv_batch_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_count: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_quant_gemv_batch_to_dev(
            "rnb_q5k_gemv_batch",
            weights,
            rows,
            blocks_per_row,
            token_count,
            input_dev,
            output_dev,
        )
    }

    pub(in crate::runtime) fn launch_q6k_gemv_batch_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        token_count: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.launch_quant_gemv_batch_to_dev(
            "rnb_q6k_gemv_batch",
            weights,
            rows,
            blocks_per_row,
            token_count,
            input_dev,
            output_dev,
        )
    }

    pub(in crate::runtime) fn launch_q6k_gemv_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp8_output = tuning::q6k_output_warp8_enabled();
        let (kernel, grid_x) = if warp8_output {
            ("rnb_q6k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q6k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q6k_gemv_q8dot_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_gemv_q8dot_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q6k_gemv_batch_q8dot_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        self.launch_cached_gemv(
            "rnb_q6k_gemv_batch_q8dot_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, seq_len as u32, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q6k_gemv_batch_seq2_warp8_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_gemv_batch_seq2_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q6k_packed_q8dot_to_dev(
        &mut self,
        packed_qs_dev: u64,
        packed_d_super_dev: u64,
        packed_sub_scale_dev: u64,
        rows: usize,
        blocks_per_row: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut packed_qs_arg = packed_qs_dev;
        let mut packed_d_super_arg = packed_d_super_dev;
        let mut packed_sub_scale_arg = packed_sub_scale_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let use_warp4 = blocks_per_row >= 16 && tuning::q6k_packed_gemv_warp4_enabled();
        let (kernel, grid, block) = if use_warp4 {
            (
                "rnb_q6k_packed_q8dot_warp4",
                (rows as u32, 1, 1),
                (32, 4, 1),
            )
        } else {
            (
                "rnb_q6k_packed_q8dot_warp8",
                (rows.div_ceil(8) as u32, 1, 1),
                (256, 1, 1),
            )
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut packed_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut packed_d_super_arg as *mut u64).cast::<libc::c_void>(),
                (&mut packed_sub_scale_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    pub(in crate::runtime) fn launch_q6k_packed_batch_q8dot_to_dev(
        &mut self,
        packed_qs_dev: u64,
        packed_d_super_dev: u64,
        packed_sub_scale_dev: u64,
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut packed_qs_arg = packed_qs_dev;
        let mut packed_d_super_arg = packed_d_super_dev;
        let mut packed_sub_scale_arg = packed_sub_scale_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let plan = q6k_packed_batch_kernel_plan(seq_len, blocks_per_row);
        let (kernel, grid, block) = match plan {
            Q6PackedBatchKernelPlan::Seq8Warp4 => (
                "rnb_q6k_packed_batch_q8dot_seq8_warp4",
                (rows as u32, seq_len.div_ceil(8) as u32, 1),
                (32, 4, 1),
            ),
            Q6PackedBatchKernelPlan::Seq4Warp4 => (
                "rnb_q6k_packed_batch_q8dot_seq4_warp4",
                (rows as u32, seq_len.div_ceil(4) as u32, 1),
                (32, 4, 1),
            ),
            Q6PackedBatchKernelPlan::Seq4Warp8 => (
                "rnb_q6k_packed_batch_q8dot_seq4_warp8",
                (rows.div_ceil(8) as u32, seq_len.div_ceil(4) as u32, 1),
                (256, 1, 1),
            ),
            Q6PackedBatchKernelPlan::Warp8 => (
                "rnb_q6k_packed_batch_q8dot_warp8",
                (rows.div_ceil(8) as u32, seq_len as u32, 1),
                (256, 1, 1),
            ),
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut packed_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut packed_d_super_arg as *mut u64).cast::<libc::c_void>(),
                (&mut packed_sub_scale_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    pub(in crate::runtime) fn launch_bf16_gemv_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut cols_arg = cols as u32;
        self.launch_cached_gemv(
            "rnb_bf16_gemv_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut cols_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_f16_gemv_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut cols_arg = cols as u32;
        self.launch_cached_gemv(
            "rnb_f16_gemv_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut cols_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q5k_gemv_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q5k_gemv",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_silu_mul(
        &mut self,
        gate_dev: u64,
        up_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_silu_mul_inplace",
            &[
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_silu_mul_group4_pack_f32(
        &mut self,
        gate_dev: u64,
        up_dev: u64,
        packed_dev: u64,
        group_meta_dev: u64,
        groups: usize,
        blocks_per_row: usize,
    ) -> Result<(), String> {
        if groups == 0 {
            return Ok(());
        }
        let total = groups
            .checked_mul(blocks_per_row)
            .and_then(|value| value.checked_mul(256))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                format!(
                    "Qwen35 pack4 F32 activation size overflow: groups={groups} blocks_per_row={blocks_per_row}"
                )
            })?;
        if total > u32::MAX as usize {
            return Err(format!(
                "Qwen35 pack4 F32 activation too large for CUDA launcher: elements={total}"
            ));
        }

        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut packed_arg = packed_dev;
        let mut group_meta_arg = group_meta_dev;
        let mut groups_arg = groups as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_silu_mul_group4_pack_f32",
            &[
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut packed_arg as *mut u64).cast::<libc::c_void>(),
                (&mut group_meta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut groups_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((total as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_gelu_mul(
        &mut self,
        gate_dev: u64,
        up_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_gelu_mul_inplace",
            &[
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_gelu_mul_q8_1(
        &mut self,
        gate_dev: u64,
        up_dev: u64,
        q8_qs_dev: u64,
        q8_ds_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut qs_arg = q8_qs_dev;
        let mut ds_arg = q8_ds_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_gelu_mul_q8_1",
            &[
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (len.div_ceil(32) as u32, 1, 1),
            (32, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_silu_mul_q8_1(
        &mut self,
        gate_dev: u64,
        up_dev: u64,
        q8_qs_dev: u64,
        q8_ds_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut gate_arg = gate_dev;
        let mut up_arg = up_dev;
        let mut qs_arg = q8_qs_dev;
        let mut ds_arg = q8_ds_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_silu_mul_q8_1",
            &[
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut up_arg as *mut u64).cast::<libc::c_void>(),
                (&mut qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (len.div_ceil(32) as u32, 1, 1),
            (32, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_scale_rows_inplace(
        &mut self,
        output_dev: u64,
        scale_dev: u64,
        rows: usize,
        row_count: usize,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut scale_arg = scale_dev;
        let mut rows_arg = rows as u32;
        let mut row_count_arg = row_count as u32;
        let len = rows.saturating_mul(row_count);
        self.launch_cached_gemv(
            "rnb_scale_rows_inplace",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut scale_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut row_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_scale_rows_add_f32_inplace(
        &mut self,
        dst_dev: u64,
        src_dev: u64,
        scale_dev: u64,
        rows: usize,
        row_count: usize,
    ) -> Result<(), String> {
        let len = rows
            .checked_mul(row_count)
            .ok_or_else(|| "scale-add element count overflow".to_string())?;
        let mut dst_arg = dst_dev;
        let mut src_arg = src_dev;
        let mut scale_arg = scale_dev;
        let mut rows_arg = u32::try_from(rows)
            .map_err(|_| format!("scale-add rows exceeds CUDA u32 limit: {rows}"))?;
        let mut row_count_arg = u32::try_from(row_count)
            .map_err(|_| format!("scale-add row count exceeds CUDA u32 limit: {row_count}"))?;
        let len = u32::try_from(len)
            .map_err(|_| format!("scale-add element count exceeds CUDA u32 limit: {len}"))?;
        self.launch_cached_gemv(
            "rnb_scale_rows_add_f32_inplace",
            &[
                (&mut dst_arg as *mut u64).cast::<libc::c_void>(),
                (&mut src_arg as *mut u64).cast::<libc::c_void>(),
                (&mut scale_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut row_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_add_f32_inplace(
        &mut self,
        dst_dev: u64,
        src_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut dst_arg = dst_dev;
        let mut src_arg = src_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_add_f32_inplace",
            &[
                (&mut dst_arg as *mut u64).cast::<libc::c_void>(),
                (&mut src_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_rows_f32_inplace(
        &mut self,
        kernel: &str,
        dst_dev: u64,
        src_dev: u64,
        len: usize,
        row_width: usize,
    ) -> Result<(), String> {
        let mut dst_arg = dst_dev;
        let mut src_arg = src_dev;
        let mut len_arg = u32::try_from(len)
            .map_err(|_| format!("{kernel} element count exceeds CUDA u32 limit: {len}"))?;
        let mut row_width_arg = u32::try_from(row_width)
            .map_err(|_| format!("{kernel} row width exceeds CUDA u32 limit: {row_width}"))?;
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut dst_arg as *mut u64).cast::<libc::c_void>(),
                (&mut src_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut row_width_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_arg.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_sigmoid_f32_inplace(
        &mut self,
        values_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut values_arg = values_dev;
        let mut len_arg = u32::try_from(len)
            .map_err(|_| format!("sigmoid element count exceeds CUDA u32 limit: {len}"))?;
        self.launch_cached_gemv(
            "rnb_sigmoid_f32_inplace",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len_arg.saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_l2_norm_rows_f32(
        &mut self,
        input_dev: u64,
        output_dev: u64,
        row_width: usize,
        row_count: usize,
        eps: f32,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut output_arg = output_dev;
        let mut row_width_arg = u32::try_from(row_width)
            .map_err(|_| format!("L2 norm row width exceeds CUDA u32 limit: {row_width}"))?;
        let mut row_count_arg = u32::try_from(row_count)
            .map_err(|_| format!("L2 norm row count exceeds CUDA u32 limit: {row_count}"))?;
        let mut eps_arg = eps;
        self.launch_cached_gemv(
            "rnb_l2_norm_rows_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut row_width_arg as *mut u32).cast::<libc::c_void>(),
                (&mut row_count_arg as *mut u32).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (row_count_arg, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_axpby_f32_inplace(
        &mut self,
        dst_dev: u64,
        src_dev: u64,
        alpha: f32,
        beta: f32,
        len: usize,
    ) -> Result<(), String> {
        let mut dst_arg = dst_dev;
        let mut src_arg = src_dev;
        let mut alpha_arg = alpha;
        let mut beta_arg = beta;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_axpby_f32_inplace",
            &[
                (&mut dst_arg as *mut u64).cast::<libc::c_void>(),
                (&mut src_arg as *mut u64).cast::<libc::c_void>(),
                (&mut alpha_arg as *mut f32).cast::<libc::c_void>(),
                (&mut beta_arg as *mut f32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_relu_sqr_f32_inplace(
        &mut self,
        values_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut values_arg = values_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_relu_sqr_f32_inplace",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_normalize_selected_softmax_f32(
        &mut self,
        selected_logits_dev: u64,
        seq_len: usize,
        selected_len: usize,
    ) -> Result<(), String> {
        let mut selected_logits_arg = selected_logits_dev;
        let mut seq_len_arg = seq_len as u32;
        let mut selected_len_arg = selected_len as u32;
        self.launch_cached_gemv(
            "rnb_normalize_selected_softmax_f32",
            &[
                (&mut selected_logits_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut selected_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_len as u32, 1, 1),
            (1, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_moe_route_topk_f32(
        &mut self,
        logits_dev: u64,
        selection_bias_dev: u64,
        expert_ids_dev: u64,
        route_weights_dev: u64,
        retained_counts_dev: u64,
        seq_len: usize,
        n_expert: usize,
        top_k: usize,
        sigmoid_mode: bool,
        normalize_selected: bool,
        scale: f32,
        adaptive_top_p: Option<f32>,
    ) -> Result<(), String> {
        if seq_len == 0 || n_expert == 0 || n_expert > 256 || top_k == 0 || top_k > n_expert {
            return Err(format!(
                "invalid CUDA MoE route shape: seq_len={seq_len} n_expert={n_expert} top_k={top_k}"
            ));
        }
        let mut logits_arg = logits_dev;
        let mut selection_bias_arg = selection_bias_dev;
        let mut expert_ids_arg = expert_ids_dev;
        let mut route_weights_arg = route_weights_dev;
        let mut retained_counts_arg = retained_counts_dev;
        let mut seq_len_arg = seq_len as u32;
        let mut n_expert_arg = n_expert as u32;
        let mut top_k_arg = top_k as u32;
        let mut sigmoid_mode_arg = u32::from(sigmoid_mode);
        let mut normalize_selected_arg = u32::from(normalize_selected);
        let mut scale_arg = scale;
        let mut adaptive_top_p_arg = adaptive_top_p.unwrap_or(-1.0);
        self.launch_cached_gemv(
            "rnb_moe_route_topk_f32",
            &[
                (&mut logits_arg as *mut u64).cast::<libc::c_void>(),
                (&mut selection_bias_arg as *mut u64).cast::<libc::c_void>(),
                (&mut expert_ids_arg as *mut u64).cast::<libc::c_void>(),
                (&mut route_weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut retained_counts_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_expert_arg as *mut u32).cast::<libc::c_void>(),
                (&mut top_k_arg as *mut u32).cast::<libc::c_void>(),
                (&mut sigmoid_mode_arg as *mut u32).cast::<libc::c_void>(),
                (&mut normalize_selected_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                (&mut adaptive_top_p_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (seq_len as u32, 1, 1),
            (1, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_hadamard_f32_inplace(
        &mut self,
        values_dev: u64,
        chunk_len: usize,
        chunk_count: usize,
    ) -> Result<(), String> {
        if chunk_len == 0 || !chunk_len.is_power_of_two() || chunk_len > 1024 || chunk_count == 0 {
            return Err(format!(
                "invalid CUDA Hadamard shape: chunk_len={chunk_len} chunk_count={chunk_count}"
            ));
        }
        let mut values_arg = values_dev;
        let mut chunk_len_arg = chunk_len as u32;
        let mut chunk_count_arg = chunk_count as u32;
        self.launch_cached_gemv(
            "rnb_hadamard_f32_inplace",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut chunk_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut chunk_count_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (chunk_count as u32, 1, 1),
            (chunk_len as u32, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_rope_f32_inplace(
        &mut self,
        values_dev: u64,
        factors_dev: u64,
        pair_total: usize,
        dim: usize,
        head_dim: usize,
        n_rot: usize,
        pos_start: usize,
        theta: f32,
        mode: u32,
    ) -> Result<(), String> {
        if pair_total == 0 {
            return Ok(());
        }
        for (label, value) in [
            ("pair_total", pair_total),
            ("dim", dim),
            ("head_dim", head_dim),
            ("n_rot", n_rot),
            ("pos_start", pos_start),
        ] {
            if value > u32::MAX as usize {
                return Err(format!("CUDA RoPE {label} exceeds u32: {value}"));
            }
        }
        let mut values_arg = values_dev;
        let mut factors_arg = factors_dev;
        let mut pair_total_arg = pair_total as u32;
        let mut dim_arg = dim as u32;
        let mut head_dim_arg = head_dim as u32;
        let mut n_rot_arg = n_rot as u32;
        let mut pos_start_arg = pos_start as u32;
        let mut theta_arg = theta;
        let mut mode_arg = mode;
        self.launch_cached_gemv(
            "rnb_rope_f32_inplace",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut factors_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pair_total_arg as *mut u32).cast::<libc::c_void>(),
                (&mut dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut n_rot_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut theta_arg as *mut f32).cast::<libc::c_void>(),
                (&mut mode_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((pair_total as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    // cu51 step 42: f32 → f16 pack (K/V projection result → KV cache).
    pub(in crate::runtime) fn launch_f32_to_f16_pack(
        &mut self,
        src_dev: u64,
        dst_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        let mut src_arg = src_dev;
        let mut dst_arg = dst_dev;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_f32_to_f16_pack",
            &[
                (&mut src_arg as *mut u64).cast::<libc::c_void>(),
                (&mut dst_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    // cu44 step 20: scalar f32 multiply inplace (Gemma4 layer_output_scale).
    pub(in crate::runtime) fn launch_scale_f32_inplace(
        &mut self,
        dst_dev: u64,
        scale: f32,
        len: usize,
    ) -> Result<(), String> {
        let mut dst_arg = dst_dev;
        let mut scale_arg = scale;
        let mut len_arg = len as u32;
        self.launch_cached_gemv(
            "rnb_scale_f32_inplace",
            &[
                (&mut dst_arg as *mut u64).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((len as u32).saturating_add(255)) / 256, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_quantize_q8_1_by_32(
        &mut self,
        input_dev: u64,
        qs_dev: u64,
        ds_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        if len % 32 != 0 {
            return Err(format!(
                "Q8_1 quantize len must be divisible by 32, got {len}"
            ));
        }
        let mut input_arg = input_dev;
        let mut qs_arg = qs_dev;
        let mut ds_arg = ds_dev;
        let mut chunks_arg = (len / 32) as u32;
        self.launch_cached_gemv(
            "rnb_quantize_q8_1_by_32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut chunks_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len / 32) as u32, 1, 1),
            (32, 1, 1),
        )
    }

    // cu39: Q8_1 quantize (32-elem chunks) + per-chunk sum 저장. dp4a matmul 의
    // m-term (sum_q8 * mn) 계산에 필요. existing `launch_quantize_q8_1_by_32` 는
    // d 만 저장 — Q4_K m offset path 없는 곳에서 쓰임.
    pub(in crate::runtime) fn launch_quantize_q8_1_with_sum_by_32(
        &mut self,
        input_dev: u64,
        qs_dev: u64,
        ds_dev: u64,
        sums_dev: u64,
        len: usize,
    ) -> Result<(), String> {
        if len % 32 != 0 {
            return Err(format!(
                "Q8_1 quantize len must be divisible by 32, got {len}"
            ));
        }
        let mut input_arg = input_dev;
        let mut qs_arg = qs_dev;
        let mut ds_arg = ds_dev;
        let mut sums_arg = sums_dev;
        let mut chunks_arg = (len / 32) as u32;
        self.launch_cached_gemv(
            "rnb_quantize_q8_1_with_sum_by_32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut sums_arg as *mut u64).cast::<libc::c_void>(),
                (&mut chunks_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((len / 32) as u32, 1, 1),
            (32, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q4k_q8_1_matmul_mmq_tile32(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        self.launch_cached_gemv(
            "rnb_q4k_q8_1_matmul_mmq_tile32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(32) as u32, seq_len.div_ceil(32) as u32, 1),
            (256, 1, 1),
        )
    }
    pub(in crate::runtime) fn launch_q8_0_q8_1_matmul_mmq_tile32(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        self.launch_cached_gemv(
            "rnb_q8_0_q8_1_matmul_mmq_tile32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(32) as u32, seq_len.div_ceil(32) as u32, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_q6k_q8_1_matmul_mmq_tile32(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        self.launch_cached_gemv(
            "rnb_q6k_q8_1_matmul_mmq_tile32",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(32) as u32, seq_len.div_ceil(32) as u32, 1),
            (256, 1, 1),
        )
    }

    // cu39 Phase 6 v3: packed nibble unpack (mmq pattern) — 16-iter byte loop →
    // 4-int load + bit shift.
    pub(in crate::runtime) fn launch_q4k_q8_1_matmul_mma_4warp_v3(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(64) as u32;
        let grid_y = seq_len.div_ceil(8) as u32;
        self.launch_cached_gemv(
            "rnb_q4k_q8_1_matmul_mma_4warp_v3",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (128, 1, 1),
        )
    }

    // cu39 Phase 5 v2: mma 4-warp variant for dense.rs q4k_batch_q8dot_to_dev integration.
    // Same as mma_4warp but no input_sums (inline sum_qy via dp4a).
    pub(in crate::runtime) fn launch_q4k_q8_1_matmul_mma_4warp_v2(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(64) as u32;
        let grid_y = seq_len.div_ceil(8) as u32;
        self.launch_cached_gemv(
            "rnb_q4k_q8_1_matmul_mma_4warp_v2",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (128, 1, 1),
        )
    }

    // cu39 Phase 4: Q4_K × Q8_1 mma 4-warp expansion (mmq_y=64).
    // grid = (rows.div_ceil(64), seq_len.div_ceil(8), 1), block = (128, 1, 1) = 4 warps.
    pub(in crate::runtime) fn launch_q4k_q8_1_matmul_mma_4warp(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        input_sums_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut input_sums_arg = input_sums_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(64) as u32;
        let grid_y = seq_len.div_ceil(8) as u32;
        self.launch_cached_gemv(
            "rnb_q4k_q8_1_matmul_mma_4warp",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_sums_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (128, 1, 1),
        )
    }

    // cu39 Phase 3: Q4_K × Q8_1 mma.m16n8k32.s8.s8 tensor core matmul.
    // grid = (rows.div_ceil(16), seq_len.div_ceil(8), 1), block = (32, 1, 1) = 1 warp.
    // Ampere SM 80+ tensor core m16n8k32 = 4096 int8 MAC per instruction.
    pub(in crate::runtime) fn launch_q4k_q8_1_matmul_mma(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        input_sums_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut input_sums_arg = input_sums_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(16) as u32;
        let grid_y = seq_len.div_ceil(8) as u32;
        self.launch_cached_gemv(
            "rnb_q4k_q8_1_matmul_mma",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_sums_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (32, 1, 1),
        )
    }

    // cu39 Phase 2: Q4_K × Q8_1 DP4A matmul with shared mem weight tile staging.
    // grid = (rows.div_ceil(8), seq_len.div_ceil(8), 1), block = (32, 1, 1) = 1 warp.
    // CTA 당 8 row × 8 seq output tile. weight 를 shared mem 에 1회 로드 + 8 seq 재사용.
    pub(in crate::runtime) fn launch_q4k_q8_1_matmul_dp4a_tile(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        input_sums_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut input_sums_arg = input_sums_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(8) as u32;
        let grid_y = seq_len.div_ceil(8) as u32;
        self.launch_cached_gemv(
            "rnb_q4k_q8_1_matmul_dp4a_tile",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_sums_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (32, 1, 1),
        )
    }

    // cu39 Phase 1: Q4_K × Q8_1 DP4A matmul launcher.
    // grid = (rows.div_ceil(8), seq_len, 1), block = (256, 1, 1) = 8 warp.
    pub(in crate::runtime) fn launch_q4k_q8_1_matmul_dp4a(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        input_sums_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut input_sums_arg = input_sums_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid_x = rows.div_ceil(8) as u32;
        let grid_y = seq_len as u32;
        self.launch_cached_gemv(
            "rnb_q4k_q8_1_matmul_dp4a",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_sums_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, grid_y, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_rms_norm_f32(
        &mut self,
        input_dev: u64,
        weight_dev: u64,
        output_dev: u64,
        eps: f32,
        len: usize,
        unit_offset: bool,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut weight_arg = weight_dev;
        let mut output_arg = output_dev;
        let mut eps_arg = eps;
        let mut len_arg = len as u32;
        let mut unit_offset_arg = u32::from(unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_rms_norm_add_f32_inplace(
        &mut self,
        input_dev: u64,
        weight_dev: u64,
        residual_dev: u64,
        eps: f32,
        len: usize,
        unit_offset: bool,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut weight_arg = weight_dev;
        let mut residual_arg = residual_dev;
        let mut eps_arg = eps;
        let mut len_arg = len as u32;
        let mut unit_offset_arg = u32::from(unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_add_f32_inplace",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_rms_norm_add_then_rms_norm_f32(
        &mut self,
        input_dev: u64,
        post_weight_dev: u64,
        residual_dev: u64,
        pre_weight_dev: u64,
        output_dev: u64,
        eps: f32,
        len: usize,
        post_unit_offset: bool,
        pre_unit_offset: bool,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut post_weight_arg = post_weight_dev;
        let mut residual_arg = residual_dev;
        let mut pre_weight_arg = pre_weight_dev;
        let mut output_arg = output_dev;
        let mut eps_arg = eps;
        let mut len_arg = len as u32;
        let mut post_unit_offset_arg = u32::from(post_unit_offset);
        let mut pre_unit_offset_arg = u32::from(pre_unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_add_then_rms_norm_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut post_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pre_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut post_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pre_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_rms_norm_add_then_rms_norm_q8_1_f32(
        &mut self,
        input_dev: u64,
        post_weight_dev: u64,
        residual_dev: u64,
        pre_weight_dev: u64,
        output_dev: u64,
        qs_dev: u64,
        ds_dev: u64,
        eps: f32,
        len: usize,
        post_unit_offset: bool,
        pre_unit_offset: bool,
    ) -> Result<(), String> {
        if len % 32 != 0 {
            return Err(format!(
                "combined norm Q8 len must be divisible by 32, got {len}"
            ));
        }
        let mut input_arg = input_dev;
        let mut post_weight_arg = post_weight_dev;
        let mut residual_arg = residual_dev;
        let mut pre_weight_arg = pre_weight_dev;
        let mut output_arg = output_dev;
        let mut qs_arg = qs_dev;
        let mut ds_arg = ds_dev;
        let mut eps_arg = eps;
        let mut len_arg = len as u32;
        let mut post_unit_offset_arg = u32::from(post_unit_offset);
        let mut pre_unit_offset_arg = u32::from(pre_unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_add_then_rms_norm_q8_1_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut post_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pre_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut post_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pre_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (1, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_rms_norm_rows_f32(
        &mut self,
        input_dev: u64,
        weight_dev: u64,
        output_dev: u64,
        eps: f32,
        rows: usize,
        len: usize,
        unit_offset: bool,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut weight_arg = weight_dev;
        let mut output_arg = output_dev;
        let mut eps_arg = eps;
        let mut rows_arg = rows as u32;
        let mut len_arg = len as u32;
        let mut unit_offset_arg = u32::from(unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_rows_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_rms_norm_rows_f32_serial(
        &mut self,
        input_dev: u64,
        weight_dev: u64,
        output_dev: u64,
        eps: f32,
        rows: usize,
        len: usize,
        unit_offset: bool,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut weight_arg = weight_dev;
        let mut output_arg = output_dev;
        let mut eps_arg = eps;
        let mut rows_arg = rows as u32;
        let mut len_arg = len as u32;
        let mut unit_offset_arg = u32::from(unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_rows_f32_serial",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn launch_mtp_build_eh_input_f32(
        &mut self,
        token_rows_dev: u64,
        target_hidden_rows_dev: u64,
        enorm_dev: u64,
        hnorm_dev: u64,
        output_dev: u64,
        eps: f32,
        rows: usize,
        hidden_dim: usize,
    ) -> Result<(), String> {
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| format!("MTP EH input rows exceeds CUDA u32 limit: {rows}"))?;
        let hidden_u32 = u32::try_from(hidden_dim)
            .map_err(|_| format!("MTP EH input hidden_dim exceeds CUDA u32 limit: {hidden_dim}"))?;
        let mut token_arg = token_rows_dev;
        let mut hidden_arg = target_hidden_rows_dev;
        let mut enorm_arg = enorm_dev;
        let mut hnorm_arg = hnorm_dev;
        let mut output_arg = output_dev;
        let mut eps_arg = eps;
        let mut rows_arg = rows_u32;
        let mut hidden_dim_arg = hidden_u32;
        self.launch_cached_gemv(
            "rnb_mtp_build_eh_input_f32",
            &[
                (&mut token_arg as *mut u64).cast::<libc::c_void>(),
                (&mut hidden_arg as *mut u64).cast::<libc::c_void>(),
                (&mut enorm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut hnorm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut hidden_dim_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows_u32, 1, 1),
            (256, 1, 1),
        )
    }

    pub(in crate::runtime) fn launch_rms_norm_add_rows_f32_inplace(
        &mut self,
        input_dev: u64,
        weight_dev: u64,
        residual_dev: u64,
        eps: f32,
        rows: usize,
        len: usize,
        unit_offset: bool,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut weight_arg = weight_dev;
        let mut residual_arg = residual_dev;
        let mut eps_arg = eps;
        let mut rows_arg = rows as u32;
        let mut len_arg = len as u32;
        let mut unit_offset_arg = u32::from(unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_add_rows_f32_inplace",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_rms_norm_add_then_rms_norm_rows_f32(
        &mut self,
        input_dev: u64,
        post_weight_dev: u64,
        residual_dev: u64,
        pre_weight_dev: u64,
        output_dev: u64,
        eps: f32,
        rows: usize,
        len: usize,
        post_unit_offset: bool,
        pre_unit_offset: bool,
    ) -> Result<(), String> {
        let mut input_arg = input_dev;
        let mut post_weight_arg = post_weight_dev;
        let mut residual_arg = residual_dev;
        let mut pre_weight_arg = pre_weight_dev;
        let mut output_arg = output_dev;
        let mut eps_arg = eps;
        let mut rows_arg = rows as u32;
        let mut len_arg = len as u32;
        let mut post_unit_offset_arg = u32::from(post_unit_offset);
        let mut pre_unit_offset_arg = u32::from(pre_unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_add_then_rms_norm_rows_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut post_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pre_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut post_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pre_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_rms_norm_add_then_rms_norm_rows_q8_1_f32(
        &mut self,
        input_dev: u64,
        post_weight_dev: u64,
        residual_dev: u64,
        pre_weight_dev: u64,
        output_dev: u64,
        qs_dev: u64,
        ds_dev: u64,
        eps: f32,
        rows: usize,
        len: usize,
        post_unit_offset: bool,
        pre_unit_offset: bool,
    ) -> Result<(), String> {
        if len % 32 != 0 {
            return Err(format!(
                "combined norm rows Q8 len must be divisible by 32, got {len}"
            ));
        }
        let mut input_arg = input_dev;
        let mut post_weight_arg = post_weight_dev;
        let mut residual_arg = residual_dev;
        let mut pre_weight_arg = pre_weight_dev;
        let mut output_arg = output_dev;
        let mut qs_arg = qs_dev;
        let mut ds_arg = ds_dev;
        let mut eps_arg = eps;
        let mut rows_arg = rows as u32;
        let mut len_arg = len as u32;
        let mut post_unit_offset_arg = u32::from(post_unit_offset);
        let mut pre_unit_offset_arg = u32::from(pre_unit_offset);
        self.launch_cached_gemv(
            "rnb_rms_norm_add_then_rms_norm_rows_q8_1_f32",
            &[
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut post_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pre_weight_arg as *mut u64).cast::<libc::c_void>(),
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut post_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
                (&mut pre_unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )
    }

    #[cfg(test)]
    pub(in crate::runtime) fn launch_ptx_kernel(
        &self,
        ptx: &str,
        kernel_name: &str,
        params: &[*mut libc::c_void],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), String> {
        self.set_current()?;
        let module = unsafe { self.api.module_load_data(ptx)? };
        let result = (|| {
            let function = unsafe { self.api.module_get_function(module, kernel_name)? };
            unsafe {
                self.api.launch_kernel(
                    function,
                    grid,
                    block,
                    0,
                    self.stream,
                    params.as_ptr() as *mut *mut libc::c_void,
                )
            }
        })();
        let unload_result = unsafe { self.api.module_unload(module) };
        result?;
        unload_result
    }

    fn load_q4k_gemv_module_image(&self) -> Result<*mut libc::c_void, String> {
        unsafe {
            if tuning::cubin_modules_enabled() {
                self.api
                    .module_load_cubin_or_ptx(Q4K_GEMV_CUBIN, Q4K_GEMV_PARALLEL_PTX)
            } else {
                self.api.module_load_data(Q4K_GEMV_PARALLEL_PTX)
            }
        }
    }

    pub(in crate::runtime) fn launch_cached_gemv(
        &mut self,
        kernel_name: &str,
        params: &[*mut libc::c_void],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), String> {
        self.set_current()?;
        if self.q4k_gemv_module.is_none() {
            let module = self.load_q4k_gemv_module_image()?;
            self.q4k_gemv_module = Some(module as usize);
        }
        let module = self
            .q4k_gemv_module
            .ok_or_else(|| "missing cached gemv module".to_string())?;
        let function = unsafe {
            self.api
                .module_get_function(module as *mut libc::c_void, kernel_name)?
        };
        unsafe {
            self.api.launch_kernel(
                function,
                grid,
                block,
                0,
                self.stream,
                params.as_ptr() as *mut *mut libc::c_void,
            )
        }
    }

    pub(in crate::runtime) fn launch_cached_nemotron_selected(
        &mut self,
        kernel_name: &str,
        params: &[*mut libc::c_void],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), String> {
        self.set_current()?;
        if self.nemotron_selected_module.is_none() {
            let module = unsafe {
                if tuning::cubin_modules_enabled() {
                    self.api
                        .module_load_cubin_or_ptx(NEMOTRON_SELECTED_CUBIN, NEMOTRON_SELECTED_PTX)?
                } else {
                    self.api.module_load_data(NEMOTRON_SELECTED_PTX)?
                }
            };
            self.nemotron_selected_module = Some(module as usize);
        }
        let module = self
            .nemotron_selected_module
            .ok_or_else(|| "missing cached Nemotron selected module".to_string())?;
        let function = unsafe {
            self.api
                .module_get_function(module as *mut libc::c_void, kernel_name)?
        };
        unsafe {
            self.api.launch_kernel(
                function,
                grid,
                block,
                0,
                self.stream,
                params.as_ptr() as *mut *mut libc::c_void,
            )
        }
    }

    pub(in crate::runtime) fn ensure_q4k_gemv_module(&mut self) -> Result<usize, String> {
        self.set_current()?;
        if self.q4k_gemv_module.is_none() {
            let module = self.load_q4k_gemv_module_image()?;
            self.q4k_gemv_module = Some(module as usize);
        }
        self.q4k_gemv_module
            .ok_or_else(|| "missing cached gemv module".to_string())
    }

    fn ensure_cu62_counter(&mut self) -> Result<u64, String> {
        if let Some(ptr) = self.cu62_counter_dev {
            return Ok(ptr);
        }
        self.set_current()?;
        let ptr = unsafe { self.api.mem_alloc(8) }?;
        self.cu62_counter_dev = Some(ptr);
        Ok(ptr)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_q4k_ple_megakernel_m1(
        &mut self,
        residual_dev: u64,
        out_scratch_dev: u64,
        weights_dev: u64,
        input_qs_dev: u64,
        input_ds_dev: u64,
        norm_weight_dev: u64,
        eps: f32,
        len: u32,
        blocks_per_row: u32,
        unit_offset: bool,
    ) -> Result<(), String> {
        let rows_per_block: u32 = 16;
        let block_threads: u32 = 256;
        let grid_blocks: u32 = len.div_ceil(rows_per_block);

        self.set_current()?;

        let counter_dev = self.ensure_cu62_counter()?;
        unsafe {
            self.api.memset_d32_async(counter_dev, 0, 2, self.stream)?;
        }

        use std::sync::OnceLock;
        static FUNCTION_PTR: OnceLock<usize> = OnceLock::new();

        let function = if let Some(p) = FUNCTION_PTR.get() {
            *p as *mut libc::c_void
        } else {
            let module = self.ensure_q4k_gemv_module()?;
            let f = unsafe {
                self.api
                    .module_get_function(module as *mut libc::c_void, "rnb_q4k_ple_megakernel_m1")?
            };
            FUNCTION_PTR.set(f as usize).ok();
            f
        };

        let mut residual_arg = residual_dev;
        let mut out_scratch_arg = out_scratch_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut norm_weight_arg = norm_weight_dev;
        let mut eps_arg: f32 = eps;
        let mut counter_arg = counter_dev;
        let mut len_arg: u32 = len;
        let mut blocks_per_row_arg: u32 = blocks_per_row;
        let mut unit_offset_arg: u32 = if unit_offset { 1 } else { 0 };
        let mut rows_per_block_arg: u32 = rows_per_block;

        let mut params: [*mut libc::c_void; 12] = [
            (&mut residual_arg as *mut u64).cast::<libc::c_void>(),
            (&mut out_scratch_arg as *mut u64).cast::<libc::c_void>(),
            (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
            (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
            (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
            (&mut norm_weight_arg as *mut u64).cast::<libc::c_void>(),
            (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
            (&mut counter_arg as *mut u64).cast::<libc::c_void>(),
            (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            (&mut unit_offset_arg as *mut u32).cast::<libc::c_void>(),
            (&mut rows_per_block_arg as *mut u32).cast::<libc::c_void>(),
        ];

        unsafe {
            self.api.launch_kernel(
                function,
                (grid_blocks, 1, 1),
                (block_threads, 1, 1),
                0,
                self.stream,
                params.as_mut_ptr(),
            )
        }
    }
}
