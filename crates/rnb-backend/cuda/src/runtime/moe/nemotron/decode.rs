use super::super::super::*;

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q5_decode_moe_shared_sparse(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.raise_resident_q4k_limit_for_nemotron_decode()?;
        let sparse_slots = up_weights.len();
        let slots = sparse_slots + 1;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let mid_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let route_dev = self.compute_route_ptr(slots * std::mem::size_of::<f32>())?;
        let token_ids_dev = self.compute_token_ids_ptr(slots * std::mem::size_of::<u32>())?;

        let mut combined_up = Vec::with_capacity(slots);
        let mut combined_down = Vec::with_capacity(slots);
        combined_up.push(shared_up);
        combined_down.push(shared_down);
        combined_up.extend_from_slice(up_weights);
        combined_down.extend_from_slice(down_weights);

        let requested_resident = slots <= 16;
        let resident_batch_bytes =
            unique_q4k_slot_bytes(combined_up.iter().chain(combined_down.iter()));
        let use_resident = requested_resident && resident_batch_bytes <= self.resident_q4k_limit;
        let (up_ptrs, down_ptrs) = if use_resident {
            let mut local_ptrs = HashMap::new();
            (
                self.resident_q4k_slot_ptrs_touch_hits(&combined_up, &mut local_ptrs)?,
                self.resident_q4k_slot_ptrs_touch_hits(&combined_down, &mut local_ptrs)?,
            )
        } else {
            let (_, up_ptrs, down_ptrs, _) =
                self.temp_q4k_slot_ptrs_3(&[], &combined_up, &combined_down)?;
            (up_ptrs, down_ptrs)
        };

        let mut combined_routes = Vec::with_capacity(slots);
        combined_routes.push(1.0f32);
        combined_routes.extend_from_slice(route_weights);
        let token_ids = vec![0u32; slots];

        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
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
                combined_routes.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(combined_routes.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                token_ids_dev,
                token_ids.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(token_ids.as_slice()),
                self.stream,
            )?;
        }

        self.launch_zero_f32(output_dev, n_embd)?;
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

        let mut output = vec![0.0f32; n_embd];
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
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_decode_moe(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.raise_resident_q4k_limit_for_nemotron_decode()?;
        let trace = std::env::var("RNB_CUDA_NEMOTRON_DECODE_MOE_TRACE")
            .ok()
            .as_deref()
            == Some("1");
        let total_start = trace.then(std::time::Instant::now);
        let sparse_slots = up_weights.len();
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let shared_mid_dev = self.compute_mid_b_ptr(shared_ff * std::mem::size_of::<f32>())?;
        let sparse_mid_dev =
            self.compute_mid_a_ptr(sparse_slots.max(1) * n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let q8_quant_cache = std::env::var("RNB_CUDA_NEMOTRON_Q8_QUANT_CACHE")
            .ok()
            .as_deref()
            == Some("1");
        let shared_up_dev = if q8_quant_cache {
            self.resident_q8_quant_ptr(shared_up, shared_ff, n_embd)?
        } else {
            self.resident_q4k_weights_ptr_touch_hit(shared_up)?
        };
        let shared_down_dev = if q8_quant_cache {
            self.resident_q8_quant_ptr(shared_down, n_embd, shared_ff)?
        } else {
            self.resident_q4k_weights_ptr_touch_hit(shared_down)?
        };
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let h2d_ms = if trace {
            let start = std::time::Instant::now();
            self.stream_synchronize()?;
            start.elapsed().as_micros() as f64 / 1000.0
        } else {
            0.0
        };
        let mut sparse_used_resident = false;
        let mut sparse_copy_prefetch = false;
        let mut resident_hits_before = 0usize;
        let mut resident_slots_before = 0usize;
        if sparse_slots > 0 {
            let requested_resident = sparse_slots <= 16;
            let resident_batch_bytes =
                unique_q4k_slot_bytes(up_weights.iter().chain(down_weights.iter()));
            let resident_possible =
                requested_resident && resident_batch_bytes <= self.resident_q4k_limit;
            if trace && resident_possible {
                resident_hits_before = q4k_resident_hit_count(&self.resident_q4k, up_weights)
                    + q4k_resident_hit_count(&self.resident_q4k, down_weights);
                resident_slots_before = up_weights.len().saturating_add(down_weights.len());
            }
            let decode_resident_enabled = std::env::var("RNB_CUDA_NEMOTRON_DECODE_RESIDENT_SPARSE")
                .ok()
                .as_deref()
                != Some("0");
            let warmup_temp_calls = std::env::var("RNB_CUDA_NEMOTRON_DECODE_TEMP_WARMUP_CALLS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let call_index = self.nemotron_decode_sparse_calls;
            self.nemotron_decode_sparse_calls = self.nemotron_decode_sparse_calls.saturating_add(1);
            sparse_used_resident =
                resident_possible && decode_resident_enabled && call_index >= warmup_temp_calls;
            if sparse_used_resident && tuning::nemotron_decode_sparse_copy_prefetch_enabled() {
                let local_ptrs = HashMap::new();
                sparse_copy_prefetch = self.batch_resident_q4k_slot_misses_many_on_stream(
                    &[up_weights, down_weights],
                    &local_ptrs,
                    self.copy_stream,
                )?;
            }
        }
        let shared_start = trace.then(std::time::Instant::now);
        if tuning::nemotron_q8_shared_cublas_enabled() {
            let shared_up_f32_dev =
                self.resident_q8_0_f32_ptr(shared_up, shared_up_dev, shared_ff, n_embd / 32)?;
            self.launch_f32_gemv_to_dev(
                shared_up_f32_dev,
                shared_ff,
                n_embd,
                input_dev,
                shared_mid_dev,
            )?;
            self.launch_relu_sqr_inplace(shared_mid_dev, shared_ff)?;
            let shared_down_f32_dev =
                self.resident_q8_0_f32_ptr(shared_down, shared_down_dev, n_embd, shared_ff / 32)?;
            self.launch_f32_gemv_to_dev(
                shared_down_f32_dev,
                n_embd,
                shared_ff,
                shared_mid_dev,
                output_dev,
            )?;
        } else {
            self.launch_nemotron_q8_shared_gemv_to_dev(
                "rnb_q8_0_gemv",
                "rnb_q8_0_gemv_warp4",
                shared_up_dev,
                shared_ff,
                n_embd / 32,
                input_dev,
                shared_mid_dev,
            )?;
            self.launch_nemotron_q8_shared_gemv_to_dev(
                "rnb_q8_0_gemv_relu_sqr_input",
                "rnb_q8_0_gemv_relu_sqr_input_warp4",
                shared_down_dev,
                n_embd,
                shared_ff / 32,
                shared_mid_dev,
                output_dev,
            )?;
        }
        let shared_ms = if let Some(start) = shared_start {
            self.stream_synchronize()?;
            start.elapsed().as_micros() as f64 / 1000.0
        } else {
            0.0
        };

        let mut sparse_setup_ms = 0.0;
        let mut sparse_up_ms = 0.0;
        let mut sparse_down_ms = 0.0;
        if sparse_slots > 0 {
            let setup_start = trace.then(std::time::Instant::now);
            let up_ptrs_dev =
                self.compute_up_ptrs_ptr(sparse_slots * std::mem::size_of::<u64>())?;
            let down_ptrs_dev =
                self.compute_down_ptrs_ptr(sparse_slots * std::mem::size_of::<u64>())?;
            let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
            let token_ids_dev =
                self.compute_token_ids_ptr(sparse_slots * std::mem::size_of::<u32>())?;
            if sparse_copy_prefetch {
                unsafe {
                    self.api.stream_synchronize(self.copy_stream)?;
                }
            }
            let (up_ptrs, down_ptrs) = if sparse_used_resident {
                let mut local_ptrs = HashMap::new();
                if !sparse_copy_prefetch {
                    self.batch_resident_q4k_slot_misses_many(
                        &[up_weights, down_weights],
                        &local_ptrs,
                    )?;
                }
                (
                    self.resident_q4k_slot_ptrs_touch_hits(up_weights, &mut local_ptrs)?,
                    self.resident_q4k_slot_ptrs_touch_hits(down_weights, &mut local_ptrs)?,
                )
            } else {
                let (_, up_ptrs, down_ptrs, _) =
                    self.temp_q4k_slot_ptrs_3(&[], up_weights, down_weights)?;
                (up_ptrs, down_ptrs)
            };
            let token_ids = vec![0u32; sparse_slots];
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
                self.api.memcpy_htod_async(
                    route_dev,
                    route_weights.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(route_weights),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    token_ids_dev,
                    token_ids.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(token_ids.as_slice()),
                    self.stream,
                )?;
            }
            if let Some(start) = setup_start {
                self.stream_synchronize()?;
                sparse_setup_ms = start.elapsed().as_micros() as f64 / 1000.0;
            }
            let up_start = trace.then(std::time::Instant::now);
            self.launch_nemotron_q5_0_selected_relu_sqr(
                up_ptrs_dev,
                token_ids_dev,
                n_ff,
                sparse_slots,
                n_embd / 32,
                input_dev,
                sparse_mid_dev,
            )?;
            if let Some(start) = up_start {
                self.stream_synchronize()?;
                sparse_up_ms = start.elapsed().as_micros() as f64 / 1000.0;
            }
            let down_start = trace.then(std::time::Instant::now);
            self.launch_nemotron_q5_1_selected_down_accum(
                down_ptrs_dev,
                token_ids_dev,
                n_embd,
                sparse_slots,
                n_ff / 32,
                sparse_mid_dev,
                route_dev,
                output_dev,
            )?;
            if let Some(start) = down_start {
                self.stream_synchronize()?;
                sparse_down_ms = start.elapsed().as_micros() as f64 / 1000.0;
            }
        }

        let mut output = vec![0.0f32; n_embd];
        let dtoh_start = trace.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        if let Some(start) = total_start {
            let dtoh_ms = dtoh_start
                .map(|s| s.elapsed().as_micros() as f64 / 1000.0)
                .unwrap_or(0.0);
            let (resident_hits_after, resident_slots_after) = if sparse_used_resident {
                (
                    q4k_resident_hit_count(&self.resident_q4k, up_weights)
                        + q4k_resident_hit_count(&self.resident_q4k, down_weights),
                    up_weights.len().saturating_add(down_weights.len()),
                )
            } else {
                (0, 0)
            };
            eprintln!(
                "[cuda:nemotron-decode-moe] slots={} resident={}/{} before={}/{} h2d={:.3} shared={:.3} sparse_setup={:.3} sparse_up={:.3} sparse_down={:.3} dtoh={:.3} total={:.3}",
                sparse_slots,
                resident_hits_after,
                resident_slots_after,
                resident_hits_before,
                resident_slots_before,
                h2d_ms,
                shared_ms,
                sparse_setup_ms,
                sparse_up_ms,
                sparse_down_ms,
                dtoh_ms,
                start.elapsed().as_micros() as f64 / 1000.0
            );
        }
        Ok(output)
    }
}
