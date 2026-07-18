use super::super::super::*;

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        self.raise_resident_q4k_limit_for_nemotron_decode()?;
        let key = nemotron_q5_layer_key(up_all, down_all, n_ff, n_embd);
        let Some((up_base, down_base)) = self
            .resident_moe_layers
            .get(&key)
            .map(|entry| (entry.up_base, entry.down_base))
        else {
            if std::env::var("RNB_CUDA_NEMOTRON_CACHED_LAYER_TRACE")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[cuda:nemotron-cached-layer] decode miss quant=q5 entries={} up_len={} down_len={} n_ff={} n_embd={}",
                    self.resident_moe_layers.len(),
                    up_all.len(),
                    down_all.len(),
                    n_ff,
                    n_embd
                );
            }
            return Ok(None);
        };
        if std::env::var("RNB_CUDA_NEMOTRON_CACHED_LAYER_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cuda:nemotron-cached-layer] decode hit quant=q5 entries={} up_len={} down_len={} n_ff={} n_embd={}",
                self.resident_moe_layers.len(),
                up_all.len(),
                down_all.len(),
                n_ff,
                n_embd
            );
        }
        self.touch_resident_moe_layer(key);

        let sparse_slots = expert_ids.len();
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

        if sparse_slots > 0 {
            let up_expert_bytes = n_ff * (n_embd / 32) * 22;
            let down_expert_bytes = n_embd * (n_ff / 32) * 24;
            let up_ptrs_dev =
                self.compute_up_ptrs_ptr(sparse_slots * std::mem::size_of::<u64>())?;
            let down_ptrs_dev =
                self.compute_down_ptrs_ptr(sparse_slots * std::mem::size_of::<u64>())?;
            let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
            let token_ids_dev =
                self.compute_token_ids_ptr(sparse_slots * std::mem::size_of::<u32>())?;
            let mut up_ptrs = Vec::with_capacity(sparse_slots);
            let mut down_ptrs = Vec::with_capacity(sparse_slots);
            for &expert in expert_ids {
                up_ptrs.push(up_base + expert as u64 * up_expert_bytes as u64);
                down_ptrs.push(down_base + expert as u64 * down_expert_bytes as u64);
            }
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
            self.launch_nemotron_q5_0_selected_relu_sqr(
                up_ptrs_dev,
                token_ids_dev,
                n_ff,
                sparse_slots,
                n_embd / 32,
                input_dev,
                sparse_mid_dev,
            )?;
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
        }

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
        Ok(Some(output))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input: &[f32],
        residual: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        let key = nemotron_q5_layer_key(up_all, down_all, n_ff, n_embd);
        let Some((up_base, down_base)) = self
            .resident_moe_layers
            .get(&key)
            .map(|entry| (entry.up_base, entry.down_base))
        else {
            return Ok(None);
        };
        self.touch_resident_moe_layer(key);

        let slots = expert_ids.len();
        let input_bytes = std::mem::size_of_val(input);
        let output_len = token_count.checked_mul(n_embd).ok_or_else(|| {
            format!(
                "Nemotron cached Q8 shared + Q5 sparse prefill output overflow: tokens={token_count} n_embd={n_embd}"
            )
        })?;
        let output_bytes = output_len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            format!(
                "Nemotron cached Q8 shared + Q5 sparse prefill output byte overflow: tokens={token_count} n_embd={n_embd}"
            )
        })?;
        let shared_mid_bytes = token_count
            .checked_mul(shared_ff)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "Nemotron cached Q8 shared + Q5 sparse prefill shared mid byte overflow: tokens={token_count} shared_ff={shared_ff}"
                )
            })?;
        let sparse_mid_bytes = slots
            .max(1)
            .checked_mul(n_ff)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                format!(
                    "Nemotron cached Q8 shared + Q5 sparse prefill sparse mid byte overflow: slots={slots} n_ff={n_ff}"
                )
            })?;

        let input_dev = self.compute_input_ptr(input_bytes)?;
        let residual_dev = self.compute_aux_output_ptr(output_bytes)?;
        let shared_mid_dev = self.compute_mid_b_ptr(shared_mid_bytes)?;
        let sparse_mid_dev = self.compute_mid_a_ptr(sparse_mid_bytes)?;
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
            let up_expert_bytes = n_ff * (n_embd / 32) * 22;
            let down_expert_bytes = n_embd * (n_ff / 32) * 24;
            let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
            let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
            let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
            let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
            let group_meta = if tuning::nemotron_prefill_group4_enabled(token_count, slots) {
                build_group_meta_from_ids(expert_ids, 4)
            } else {
                Vec::new()
            };
            let group_meta_dev = if group_meta.is_empty() {
                0
            } else {
                self.compute_group_meta_ptr(std::mem::size_of_val(group_meta.as_slice()))?
            };
            let mut up_ptrs = Vec::with_capacity(slots);
            let mut down_ptrs = Vec::with_capacity(slots);
            for &expert in expert_ids {
                up_ptrs.push(up_base + expert as u64 * up_expert_bytes as u64);
                down_ptrs.push(down_base + expert as u64 * down_expert_bytes as u64);
            }
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
        Ok(Some(output))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q5_sparse_relu_sqr_cached_layer_by_token(
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
    ) -> Result<Option<Vec<f32>>, String> {
        let key = nemotron_q5_layer_key(up_all, down_all, n_ff, n_embd);
        let Some((up_base, down_base)) = self
            .resident_moe_layers
            .get(&key)
            .map(|entry| (entry.up_base, entry.down_base))
        else {
            return Ok(None);
        };
        self.touch_resident_moe_layer(key);

        let slots = expert_ids.len();
        let up_expert_bytes = n_ff * (n_embd / 32) * 22;
        let down_expert_bytes = n_embd * (n_ff / 32) * 24;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let mid_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = token_count * n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
        let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
        let group_meta = if tuning::nemotron_prefill_group4_enabled(token_count, slots) {
            build_group_meta_from_ids(expert_ids, 4)
        } else {
            Vec::new()
        };
        let group_meta_dev = if group_meta.is_empty() {
            0
        } else {
            self.compute_group_meta_ptr(std::mem::size_of_val(group_meta.as_slice()))?
        };
        let mut up_ptrs = Vec::with_capacity(slots);
        let mut down_ptrs = Vec::with_capacity(slots);
        for &expert in expert_ids {
            up_ptrs.push(up_base + expert as u64 * up_expert_bytes as u64);
            down_ptrs.push(down_base + expert as u64 * down_expert_bytes as u64);
        }
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

        self.launch_zero_f32(output_dev, token_count * n_embd)?;
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
        Ok(Some(output))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
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
    ) -> Result<Option<Vec<f32>>, String> {
        let key = nemotron_q5_q8_layer_key(up_all, down_all, n_ff, n_embd);
        let Some((up_base, down_base)) = self
            .resident_moe_layers
            .get(&key)
            .map(|entry| (entry.up_base, entry.down_base))
        else {
            return Ok(None);
        };
        self.touch_resident_moe_layer(key);

        let slots = expert_ids.len();
        let up_expert_bytes = n_ff * (n_embd / 32) * 22;
        let down_expert_bytes = n_embd * (n_ff / 32) * 34;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let mid_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = token_count * n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
        let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
        let group_meta = if tuning::nemotron_prefill_group4_enabled(token_count, slots) {
            build_group_meta_from_ids(expert_ids, 4)
        } else {
            Vec::new()
        };
        let group_meta_dev = if group_meta.is_empty() {
            0
        } else {
            self.compute_group_meta_ptr(std::mem::size_of_val(group_meta.as_slice()))?
        };
        let mut up_ptrs = Vec::with_capacity(slots);
        let mut down_ptrs = Vec::with_capacity(slots);
        for &expert in expert_ids {
            up_ptrs.push(up_base + expert as u64 * up_expert_bytes as u64);
            down_ptrs.push(down_base + expert as u64 * down_expert_bytes as u64);
        }
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

        self.launch_zero_f32(output_dev, token_count * n_embd)?;
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
        Ok(Some(output))
    }
}
