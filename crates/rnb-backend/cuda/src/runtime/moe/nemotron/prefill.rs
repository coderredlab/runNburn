use super::super::super::*;
use super::NemotronDeviceRoutePackOutput;

#[derive(Clone, Copy)]
enum NemotronDeviceSparseRoute<'a> {
    Host {
        route_weights: &'a [f32],
        token_ids: &'a [u32],
    },
    DevicePack(NemotronDeviceRoutePackOutput),
}

impl NemotronDeviceSparseRoute<'_> {
    fn slots(&self) -> usize {
        match self {
            Self::Host { route_weights, .. } => route_weights.len(),
            Self::DevicePack(route) => route.slots,
        }
    }

    fn validate(&self, token_count: usize) -> Result<(), String> {
        match self {
            Self::Host {
                route_weights,
                token_ids,
            } => {
                if route_weights.len() != token_ids.len() {
                    return Err(
                        "Nemotron prefill Q8 shared + Q5 sparse route length mismatch".to_string(),
                    );
                }
                if token_ids.iter().any(|&token| token as usize >= token_count) {
                    return Err(
                        "Nemotron prefill Q8 shared + Q5 sparse token id out of range".to_string(),
                    );
                }
                Ok(())
            }
            Self::DevicePack(route) => {
                if route.seq_len != token_count {
                    return Err(format!(
                        "Nemotron device route pack token count mismatch: got {}, expected {token_count}",
                        route.seq_len
                    ));
                }
                Ok(())
            }
        }
    }
}

impl CudaState {
    pub(in crate::runtime) fn nemotron_q8_shared_prefill(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        shared_ff: usize,
        n_embd: usize,
        token_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let input_bytes = std::mem::size_of_val(input);
        let input_dev = self.compute_input_ptr(input_bytes)?;
        let mid_dev = self.compute_mid_b_ptr(
            token_count
                .checked_mul(shared_ff)
                .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    format!(
                        "Nemotron Q8 shared prefill mid byte overflow: tokens={token_count} shared_ff={shared_ff}"
                    )
                })?,
        )?;
        let output_len = token_count.checked_mul(n_embd).ok_or_else(|| {
            format!(
                "Nemotron Q8 shared prefill output overflow: tokens={token_count} n_embd={n_embd}"
            )
        })?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Nemotron Q8 shared prefill output byte overflow: tokens={token_count} n_embd={n_embd}"
                )
            })?;
        let output_dev = self.compute_output_ptr(output_bytes)?;

        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
        }

        let shared_up_dev = self.compute_weights_ptr(shared_up.len())?;
        unsafe {
            self.api.memcpy_htod_async(
                shared_up_dev,
                shared_up.as_ptr().cast::<libc::c_void>(),
                shared_up.len(),
                self.stream,
            )?;
        }
        let shared_up_f32_dev =
            self.resident_q8_0_f32_ptr(shared_up, shared_up_dev, shared_ff, n_embd / 32)?;
        self.sgemm_device(
            shared_up_f32_dev,
            shared_ff,
            n_embd,
            input_dev,
            token_count,
            mid_dev,
        )?;
        self.launch_relu_sqr_inplace(mid_dev, token_count * shared_ff)?;

        let shared_down_dev = self.compute_weights_ptr(shared_down.len())?;
        unsafe {
            self.api.memcpy_htod_async(
                shared_down_dev,
                shared_down.as_ptr().cast::<libc::c_void>(),
                shared_down.len(),
                self.stream,
            )?;
        }
        let shared_down_f32_dev =
            self.resident_q8_0_f32_ptr(shared_down, shared_down_dev, n_embd, shared_ff / 32)?;
        self.sgemm_device(
            shared_down_f32_dev,
            n_embd,
            shared_ff,
            mid_dev,
            token_count,
            output_dev,
        )?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q5_sparse_relu_sqr_by_token(
        &mut self,
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.nemotron_q5_sparse_relu_sqr_by_token_with_down(
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q5_q8_sparse_relu_sqr_by_token(
        &mut self,
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.nemotron_q5_sparse_relu_sqr_by_token_with_down(
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q5_sparse_relu_sqr_by_token_with_down(
        &mut self,
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        down_q8: bool,
    ) -> Result<Vec<f32>, String> {
        let trace = std::env::var("RNB_CUDA_NEMOTRON_PREFILL_SPARSE_TRACE")
            .ok()
            .as_deref()
            == Some("1");
        let total_start = trace.then(Instant::now);
        let slots = up_weights.len();
        let setup_start = trace.then(Instant::now);
        let input_bytes = std::mem::size_of_val(input);
        let input_dev = self.compute_input_ptr(input_bytes)?;
        let mid_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = token_count * n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
        let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
        let group_meta = if tuning::nemotron_prefill_group4_enabled(token_count, slots) {
            build_group_meta(up_weights, down_weights, 4)
        } else {
            Vec::new()
        };
        let group_meta_dev = if group_meta.is_empty() {
            0
        } else {
            self.compute_group_meta_ptr(std::mem::size_of_val(group_meta.as_slice()))?
        };

        let resident_batch_bytes = unique_q4k_slot_bytes(up_weights.iter().chain(down_weights));
        let prefill_resident_enabled = std::env::var("RNB_CUDA_NEMOTRON_PREFILL_RESIDENT_SPARSE")
            .ok()
            .as_deref()
            == Some("1");
        let use_resident =
            prefill_resident_enabled && resident_batch_bytes <= self.resident_q4k_limit;
        let mut used_prefetch = false;
        let mut temp_slab_ptrs_len = 0usize;
        let (up_ptrs, down_ptrs) = if use_resident {
            let mut local_ptrs = HashMap::new();
            (
                self.resident_q4k_slot_ptrs(up_weights, &mut local_ptrs)?,
                self.resident_q4k_slot_ptrs(down_weights, &mut local_ptrs)?,
            )
        } else if let Some((up_ptrs, down_ptrs)) =
            self.pending_nemotron_prefill_sparse_ptrs(up_weights, down_weights)?
        {
            used_prefetch = true;
            (up_ptrs, down_ptrs)
        } else {
            let (_, up_ptrs, down_ptrs, temp_slab_ptrs) =
                self.temp_q4k_slot_ptrs_3(&[], up_weights, down_weights)?;
            temp_slab_ptrs_len = temp_slab_ptrs.len();
            (up_ptrs, down_ptrs)
        };
        let setup_ms = setup_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let h2d_start = trace.then(Instant::now);
        let input_src = if tuning::nemotron_prefill_sparse_input_pinned_enabled()
            && input_bytes >= (1 << 20)
        {
            let host_ptr = self.host_sparse_input_slab_ptr(input_bytes)?;
            unsafe {
                std::ptr::copy_nonoverlapping(input.as_ptr().cast::<u8>(), host_ptr, input_bytes);
            }
            host_ptr.cast::<libc::c_void>() as *const libc::c_void
        } else {
            input.as_ptr().cast::<libc::c_void>()
        };
        unsafe {
            self.api
                .memcpy_htod_async(input_dev, input_src, input_bytes, self.stream)?;
            self.api.memcpy_htod_async(
                up_ptrs_dev,
                up_ptrs.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(up_ptrs.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                down_ptrs_dev,
                down_ptrs.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(down_ptrs.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                route_dev,
                route_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(route_weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                token_ids_dev,
                token_ids.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(token_ids),
                self.stream,
            )?;
            if !group_meta.is_empty() {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    group_meta.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(group_meta.as_slice()),
                    self.stream,
                )?;
            }
        }
        if trace {
            self.stream_synchronize()?;
        }
        let h2d_ms = h2d_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let zero_start = trace.then(Instant::now);
        self.launch_zero_f32(output_dev, token_count * n_embd)?;
        if trace {
            self.stream_synchronize()?;
        }
        let zero_ms = zero_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let up_start = trace.then(Instant::now);
        if group_meta.is_empty() {
            self.launch_nemotron_q5_0_selected_relu_sqr(
                up_ptrs_dev,
                token_ids_dev,
                n_ff,
                slots,
                n_embd / 32,
                input_dev,
                mid_dev,
            )?;
        } else {
            self.launch_nemotron_q5_0_selected_relu_sqr_group4(
                up_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_ff,
                group_meta.len() / 2,
                n_embd / 32,
                input_dev,
                mid_dev,
            )?;
        }
        if trace {
            self.stream_synchronize()?;
        }
        let up_ms = up_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let down_copy_sync_start = trace.then(Instant::now);
        if temp_slab_ptrs_len > 0 && tuning::prefill_down_copy_overlap_enabled() {
            unsafe {
                self.api.stream_synchronize(self.copy_stream)?;
            }
        }
        let down_copy_sync_ms =
            down_copy_sync_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let down_start = trace.then(Instant::now);
        if down_q8 {
            if group_meta.is_empty() {
                self.launch_nemotron_q8_0_selected_down_accum(
                    down_ptrs_dev,
                    token_ids_dev,
                    n_embd,
                    slots,
                    n_ff / 32,
                    mid_dev,
                    route_dev,
                    output_dev,
                )?;
            } else {
                self.launch_nemotron_q8_0_selected_down_accum_group4(
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    group_meta.len() / 2,
                    n_ff / 32,
                    mid_dev,
                    route_dev,
                    output_dev,
                )?;
            }
        } else if group_meta.is_empty() {
            self.launch_nemotron_q5_1_selected_down_accum(
                down_ptrs_dev,
                token_ids_dev,
                n_embd,
                slots,
                n_ff / 32,
                mid_dev,
                route_dev,
                output_dev,
            )?;
        } else {
            self.launch_nemotron_q5_1_selected_down_accum_group4(
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                group_meta.len() / 2,
                n_ff / 32,
                mid_dev,
                route_dev,
                output_dev,
            )?;
        }
        if trace {
            self.stream_synchronize()?;
        }
        let down_ms = down_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);

        let mut output = vec![0.0f32; token_count * n_embd];
        let dtoh_start = trace.then(Instant::now);
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        if used_prefetch {
            self.clear_pending_nemotron_prefill_sparse()?;
        }
        if let Some(total_start) = total_start {
            eprintln!(
                "[cuda:nemotron-prefill-sparse] slots={} tokens={} resident={} prefetch={} down_overlap={} setup_ms={:.3} h2d_ms={:.3} zero_ms={:.3} up_ms={:.3} copy_sync_ms={:.3} down_ms={:.3} dtoh_ms={:.3} total_ms={:.3}",
                slots,
                token_count,
                use_resident,
                used_prefetch,
                temp_slab_ptrs_len > 0 && tuning::prefill_down_copy_overlap_enabled(),
                setup_ms.unwrap_or(0.0),
                h2d_ms.unwrap_or(0.0),
                zero_ms.unwrap_or(0.0),
                up_ms.unwrap_or(0.0),
                down_copy_sync_ms.unwrap_or(0.0),
                down_ms.unwrap_or(0.0),
                dtoh_start
                    .map(|start| start.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0),
                total_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_ptrs(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input_dev: u64,
        residual_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.nemotron_q8_shared_q5_sparse_prefill_moe_device_ptrs_with_route(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            NemotronDeviceSparseRoute::Host {
                route_weights,
                token_ids,
            },
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_dev,
            residual_dev,
            output_dev,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack_ptrs(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_pack: NemotronDeviceRoutePackOutput,
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input_dev: u64,
        residual_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.nemotron_q8_shared_q5_sparse_prefill_moe_device_ptrs_with_route(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            NemotronDeviceSparseRoute::DevicePack(route_pack),
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_dev,
            residual_dev,
            output_dev,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn nemotron_q8_shared_q5_sparse_prefill_moe_device_ptrs_with_route(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_source: NemotronDeviceSparseRoute<'_>,
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input_dev: u64,
        residual_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let slots = route_source.slots();
        route_source.validate(token_count)?;
        if up_weights.len() != slots || down_weights.len() != slots {
            return Err("Nemotron prefill Q8 shared + Q5 sparse batch length mismatch".to_string());
        }
        let output_len = token_count.checked_mul(n_embd).ok_or_else(|| {
            format!(
                "Nemotron Q8 shared + Q5 sparse prefill output overflow: tokens={token_count} n_embd={n_embd}"
            )
        })?;
        let shared_mid_bytes = token_count
            .checked_mul(shared_ff)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "Nemotron Q8 shared + Q5 sparse prefill shared mid byte overflow: tokens={token_count} shared_ff={shared_ff}"
                )
            })?;
        let sparse_mid_bytes = slots
            .max(1)
            .checked_mul(n_ff)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "Nemotron Q8 shared + Q5 sparse prefill sparse mid byte overflow: slots={slots} n_ff={n_ff}"
                )
            })?;

        let (shared_mid_dev, sparse_mid_dev) = if let Some(ptrs) =
            self.nemotron_workspace_moe_mid_ptrs(shared_mid_bytes, sparse_mid_bytes)
        {
            ptrs
        } else {
            (
                self.compute_mid_b_ptr(shared_mid_bytes)?,
                self.compute_mid_a_ptr(sparse_mid_bytes)?,
            )
        };

        let mut used_prefetch = false;
        let result = (|| -> Result<(), String> {
            let shared_up_dev = self.compute_weights_ptr(shared_up.len())?;
            unsafe {
                self.api.memcpy_htod_async(
                    shared_up_dev,
                    shared_up.as_ptr().cast::<libc::c_void>(),
                    shared_up.len(),
                    self.stream,
                )?;
            }
            let shared_up_f32_dev =
                self.resident_q8_0_f32_ptr(shared_up, shared_up_dev, shared_ff, n_embd / 32)?;
            self.sgemm_device(
                shared_up_f32_dev,
                shared_ff,
                n_embd,
                input_dev,
                token_count,
                shared_mid_dev,
            )?;
            self.launch_relu_sqr_inplace(shared_mid_dev, token_count * shared_ff)?;

            let shared_down_dev = self.compute_weights_ptr(shared_down.len())?;
            unsafe {
                self.api.memcpy_htod_async(
                    shared_down_dev,
                    shared_down.as_ptr().cast::<libc::c_void>(),
                    shared_down.len(),
                    self.stream,
                )?;
            }
            let shared_down_f32_dev =
                self.resident_q8_0_f32_ptr(shared_down, shared_down_dev, n_embd, shared_ff / 32)?;
            self.sgemm_device(
                shared_down_f32_dev,
                n_embd,
                shared_ff,
                shared_mid_dev,
                token_count,
                output_dev,
            )?;

            if slots > 0 {
                let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
                let down_ptrs_dev =
                    self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
                let (route_dev, token_ids_dev) = match route_source {
                    NemotronDeviceSparseRoute::Host {
                        route_weights,
                        token_ids,
                    } => (
                        self.compute_route_ptr(std::mem::size_of_val(route_weights))?,
                        self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?,
                    ),
                    NemotronDeviceSparseRoute::DevicePack(route) => {
                        (route.route_weights_dev, route.token_ids_dev)
                    }
                };
                let group_meta = if tuning::nemotron_prefill_group4_enabled(token_count, slots) {
                    build_group_meta(up_weights, down_weights, 4)
                } else {
                    Vec::new()
                };
                let group_meta_dev = if group_meta.is_empty() {
                    0
                } else {
                    self.compute_group_meta_ptr(std::mem::size_of_val(group_meta.as_slice()))?
                };

                let resident_batch_bytes =
                    unique_q4k_slot_bytes(up_weights.iter().chain(down_weights));
                let prefill_resident_enabled =
                    std::env::var("RNB_CUDA_NEMOTRON_PREFILL_RESIDENT_SPARSE")
                        .ok()
                        .as_deref()
                        == Some("1");
                let use_resident =
                    prefill_resident_enabled && resident_batch_bytes <= self.resident_q4k_limit;
                let mut temp_slab_ptrs_len = 0usize;
                let (up_ptrs, down_ptrs) = if use_resident {
                    let mut local_ptrs = HashMap::new();
                    (
                        self.resident_q4k_slot_ptrs(up_weights, &mut local_ptrs)?,
                        self.resident_q4k_slot_ptrs(down_weights, &mut local_ptrs)?,
                    )
                } else if let Some((up_ptrs, down_ptrs)) =
                    self.pending_nemotron_prefill_sparse_ptrs(up_weights, down_weights)?
                {
                    used_prefetch = true;
                    (up_ptrs, down_ptrs)
                } else {
                    let (_, up_ptrs, down_ptrs, temp_slab_ptrs) =
                        self.temp_q4k_slot_ptrs_3(&[], up_weights, down_weights)?;
                    temp_slab_ptrs_len = temp_slab_ptrs.len();
                    (up_ptrs, down_ptrs)
                };

                unsafe {
                    self.api.memcpy_htod_async(
                        up_ptrs_dev,
                        up_ptrs.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(up_ptrs.as_slice()),
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        down_ptrs_dev,
                        down_ptrs.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(down_ptrs.as_slice()),
                        self.stream,
                    )?;
                    if let NemotronDeviceSparseRoute::Host {
                        route_weights,
                        token_ids,
                    } = route_source
                    {
                        self.api.memcpy_htod_async(
                            route_dev,
                            route_weights.as_ptr().cast::<libc::c_void>(),
                            std::mem::size_of_val(route_weights),
                            self.stream,
                        )?;
                        self.api.memcpy_htod_async(
                            token_ids_dev,
                            token_ids.as_ptr().cast::<libc::c_void>(),
                            std::mem::size_of_val(token_ids),
                            self.stream,
                        )?;
                    }
                    if !group_meta.is_empty() {
                        self.api.memcpy_htod_async(
                            group_meta_dev,
                            group_meta.as_ptr().cast::<libc::c_void>(),
                            std::mem::size_of_val(group_meta.as_slice()),
                            self.stream,
                        )?;
                    }
                }

                if group_meta.is_empty() {
                    self.launch_nemotron_q5_0_selected_relu_sqr(
                        up_ptrs_dev,
                        token_ids_dev,
                        n_ff,
                        slots,
                        n_embd / 32,
                        input_dev,
                        sparse_mid_dev,
                    )?;
                } else {
                    self.launch_nemotron_q5_0_selected_relu_sqr_group4(
                        up_ptrs_dev,
                        token_ids_dev,
                        group_meta_dev,
                        n_ff,
                        group_meta.len() / 2,
                        n_embd / 32,
                        input_dev,
                        sparse_mid_dev,
                    )?;
                }
                if temp_slab_ptrs_len > 0 && tuning::prefill_down_copy_overlap_enabled() {
                    unsafe {
                        self.api.stream_synchronize(self.copy_stream)?;
                    }
                }
                if group_meta.is_empty() {
                    self.launch_nemotron_q5_1_selected_down_accum(
                        down_ptrs_dev,
                        token_ids_dev,
                        n_embd,
                        slots,
                        n_ff / 32,
                        sparse_mid_dev,
                        route_dev,
                        output_dev,
                    )?;
                } else {
                    self.launch_nemotron_q5_1_selected_down_accum_group4(
                        down_ptrs_dev,
                        token_ids_dev,
                        group_meta_dev,
                        n_embd,
                        group_meta.len() / 2,
                        n_ff / 32,
                        sparse_mid_dev,
                        route_dev,
                        output_dev,
                    )?;
                }
            }

            self.launch_add_f32_inplace(output_dev, residual_dev, output_len)?;
            Ok(())
        })();

        if used_prefetch {
            match self
                .stream_synchronize()
                .and_then(|_| self.clear_pending_nemotron_prefill_sparse())
            {
                Ok(()) => {}
                Err(cleanup_err) if result.is_ok() => return Err(cleanup_err),
                Err(_) => {}
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_prefill_moe(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input: &[f32],
        residual: &[f32],
    ) -> Result<Vec<f32>, String> {
        let input_bytes = std::mem::size_of_val(input);
        let output_len = token_count.checked_mul(n_embd).ok_or_else(|| {
            format!(
                "Nemotron Q8 shared + Q5 sparse prefill output overflow: tokens={token_count} n_embd={n_embd}"
            )
        })?;
        let output_bytes = output_len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            format!(
                "Nemotron Q8 shared + Q5 sparse prefill output byte overflow: tokens={token_count} n_embd={n_embd}"
            )
        })?;

        let input_dev = self.compute_input_ptr(input_bytes)?;
        let residual_dev = self.compute_aux_output_ptr(output_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;

        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                residual_dev,
                residual.as_ptr().cast::<libc::c_void>(),
                output_bytes,
                self.stream,
            )?;
        }

        self.nemotron_q8_shared_q5_sparse_prefill_moe_device_ptrs(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_dev,
            residual_dev,
            output_dev,
        )?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q5_sparse_relu_sqr_full_layer_by_token(
        &mut self,
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        _n_expert: usize,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let slots = expert_ids.len();
        let up_expert_bytes = n_ff * (n_embd / 32) * 22;
        let down_expert_bytes = n_embd * (n_ff / 32) * 24;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let mid_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = token_count * n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let full_up_dev = self.compute_full_up_ptr(std::mem::size_of_val(up_all))?;
        let full_down_dev = self.compute_full_down_ptr(std::mem::size_of_val(down_all))?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
        let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
        let mut up_ptrs = Vec::with_capacity(slots);
        let mut down_ptrs = Vec::with_capacity(slots);
        for &expert in expert_ids {
            up_ptrs.push(full_up_dev + expert as u64 * up_expert_bytes as u64);
            down_ptrs.push(full_down_dev + expert as u64 * down_expert_bytes as u64);
        }
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                full_up_dev,
                up_all.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(up_all),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                full_down_dev,
                down_all.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(down_all),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                up_ptrs_dev,
                up_ptrs.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(up_ptrs.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                down_ptrs_dev,
                down_ptrs.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(down_ptrs.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                route_dev,
                route_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(route_weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                token_ids_dev,
                token_ids.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(token_ids),
                self.stream,
            )?;
        }

        self.launch_zero_f32(output_dev, token_count * n_embd)?;
        self.launch_nemotron_q5_0_selected_relu_sqr(
            up_ptrs_dev,
            token_ids_dev,
            n_ff,
            slots,
            n_embd / 32,
            input_dev,
            mid_dev,
        )?;
        self.launch_nemotron_q5_1_selected_down_accum(
            down_ptrs_dev,
            token_ids_dev,
            n_embd,
            slots,
            n_ff / 32,
            mid_dev,
            route_dev,
            output_dev,
        )?;

        let mut output = vec![0.0f32; token_count * n_embd];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }
}
