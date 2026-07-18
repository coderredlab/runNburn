use super::super::*;

pub(super) struct GlmGroupedSlots<'a> {
    pub(super) gate_weights: Vec<&'a [u8]>,
    pub(super) up_weights: Vec<&'a [u8]>,
    pub(super) down_weights: Vec<&'a [u8]>,
    pub(super) route_weights: Vec<f32>,
    pub(super) token_ids: Vec<u32>,
    pub(super) group_meta: Vec<u32>,
}

fn glm_slot_identity(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
) -> (usize, usize, usize, usize, usize, usize) {
    (
        gate.as_ptr() as usize,
        gate.len(),
        up.as_ptr() as usize,
        up.len(),
        down.as_ptr() as usize,
        down.len(),
    )
}

pub(super) fn group_glm_slots<'a>(
    gate_weights: &[&'a [u8]],
    up_weights: &[&'a [u8]],
    down_weights: &[&'a [u8]],
    route_weights: &[f32],
    token_ids: &[u32],
) -> GlmGroupedSlots<'a> {
    debug_assert_eq!(gate_weights.len(), up_weights.len());
    debug_assert_eq!(gate_weights.len(), down_weights.len());
    debug_assert_eq!(gate_weights.len(), route_weights.len());
    debug_assert_eq!(gate_weights.len(), token_ids.len());

    let mut order = (0..gate_weights.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&slot| {
        glm_slot_identity(gate_weights[slot], up_weights[slot], down_weights[slot])
    });

    let mut grouped = GlmGroupedSlots {
        gate_weights: Vec::with_capacity(order.len()),
        up_weights: Vec::with_capacity(order.len()),
        down_weights: Vec::with_capacity(order.len()),
        route_weights: Vec::with_capacity(order.len()),
        token_ids: Vec::with_capacity(order.len()),
        group_meta: Vec::new(),
    };
    let mut start = 0usize;
    while start < order.len() {
        let first = order[start];
        let identity =
            glm_slot_identity(gate_weights[first], up_weights[first], down_weights[first]);
        let mut len = 1usize;
        while start + len < order.len()
            && len < 4
            && glm_slot_identity(
                gate_weights[order[start + len]],
                up_weights[order[start + len]],
                down_weights[order[start + len]],
            ) == identity
        {
            len += 1;
        }
        grouped.group_meta.push(start as u32);
        grouped.group_meta.push(len as u32);
        for &slot in &order[start..start + len] {
            grouped.gate_weights.push(gate_weights[slot]);
            grouped.up_weights.push(up_weights[slot]);
            grouped.down_weights.push(down_weights[slot]);
            grouped.route_weights.push(route_weights[slot]);
            grouped.token_ids.push(token_ids[slot]);
        }
        start += len;
    }
    grouped
}

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn glm_sparse_experts_iq2xxs_iq3xxs(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.glm_sparse_experts_iq2xxs_iq3xxs_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            n_ff,
            n_embd,
            input_dev,
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
    pub(in crate::runtime) fn glm_sparse_experts_iq2xxs_iq3xxs_to_dev(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let selected = gate_weights.len();
        if selected == 0 {
            return Err(
                "GLM device sparse IQ2_XXS/IQ3_XXS MoE requires selected experts".to_string(),
            );
        }
        if selected != up_weights.len()
            || selected != down_weights.len()
            || selected != route_weights.len()
        {
            return Err(format!(
                "GLM device sparse IQ2_XXS/IQ3_XXS MoE selection mismatch: gate={} up={} down={} route={}",
                selected,
                up_weights.len(),
                down_weights.len(),
                route_weights.len()
            ));
        }
        if selected > 32 {
            return Err(format!(
                "GLM device sparse IQ2_XXS/IQ3_XXS MoE supports up to 32 slots, got {selected}"
            ));
        }
        if n_embd % 256 != 0 || n_ff % 256 != 0 {
            return Err(format!(
                "GLM device sparse IQ2_XXS/IQ3_XXS MoE dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
            ));
        }

        let gate_row_bytes = (n_embd / 256) * 66;
        let down_row_bytes = (n_ff / 256) * 98;
        for (slot, weights) in gate_weights.iter().enumerate() {
            if weights.len() != n_ff * gate_row_bytes {
                return Err(format!(
                    "GLM sparse IQ2_XXS gate[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_ff * gate_row_bytes
                ));
            }
        }
        for (slot, weights) in up_weights.iter().enumerate() {
            if weights.len() != n_ff * gate_row_bytes {
                return Err(format!(
                    "GLM sparse IQ2_XXS up[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_ff * gate_row_bytes
                ));
            }
        }
        for (slot, weights) in down_weights.iter().enumerate() {
            if weights.len() != n_embd * down_row_bytes {
                return Err(format!(
                    "GLM sparse IQ3_XXS down[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_embd * down_row_bytes
                ));
            }
        }

        let gate_dev = self.compute_mid_a_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        self.set_current()?;
        let (gate_ptrs, up_ptrs, down_ptrs, temp_slab_ptrs) =
            self.temp_q4k_slot_ptrs_3(gate_weights, up_weights, down_weights)?;

        let ptr_bytes = selected * std::mem::size_of::<u64>();
        let route_bytes = std::mem::size_of_val(route_weights);
        let meta_bytes = ptr_bytes * 3 + route_bytes;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(meta_bytes)?;
        let up_ptrs_dev = gate_ptrs_dev + ptr_bytes as u64;
        let down_ptrs_dev = gate_ptrs_dev + (ptr_bytes * 2) as u64;
        let route_dev = gate_ptrs_dev + (ptr_bytes * 3) as u64;
        let mut meta = vec![0u8; meta_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gate_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr(),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                up_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                down_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes * 2),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes * 3),
                route_bytes,
            );
            self.api.memcpy_htod_async(
                gate_ptrs_dev,
                meta.as_ptr().cast::<libc::c_void>(),
                meta_bytes,
                self.stream,
            )?;
        }

        self.launch_selected_iq2_xxs_gate_up_gemv_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            n_ff,
            selected,
            n_embd / 256,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        if !temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled() {
            unsafe { self.api.stream_synchronize(self.copy_stream)? };
        }
        self.launch_selected_down_silu_rowreduce(
            "rnb_iq3_xxs_selected_down_silu_rowreduce",
            down_ptrs_dev,
            n_embd,
            selected,
            n_ff / 256,
            gate_dev,
            up_dev,
            route_dev,
            output_dev,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn glm_sparse_experts_iq2xxs_iq3xxs_by_token(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        file_regions: Option<&[rnb_core::tensor::FileBackedRegion; 3]>,
        direct_file: bool,
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let slots = gate_weights.len();
        if token_count == 0 || slots == 0 || slots % token_count != 0 {
            return Err(format!(
                "GLM batched sparse slots must be non-zero and divisible by token_count: slots={slots} token_count={token_count}"
            ));
        }
        if up_weights.len() != slots
            || down_weights.len() != slots
            || route_weights.len() != slots
            || token_ids.len() != slots
        {
            return Err(format!(
                "GLM batched sparse slot mismatch: gate={} up={} down={} route={} token_ids={}",
                slots,
                up_weights.len(),
                down_weights.len(),
                route_weights.len(),
                token_ids.len()
            ));
        }
        if input.len() != token_count.saturating_mul(n_embd) {
            return Err(format!(
                "GLM batched sparse input mismatch: got={} expected={}",
                input.len(),
                token_count.saturating_mul(n_embd)
            ));
        }
        if token_ids.iter().any(|&token| token as usize >= token_count) {
            return Err("GLM batched sparse token id is out of range".to_string());
        }
        if n_embd % 256 != 0 || n_ff % 256 != 0 {
            return Err(format!(
                "GLM batched sparse dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
            ));
        }

        let gate_row_bytes = (n_embd / 256) * 66;
        let down_row_bytes = (n_ff / 256) * 98;
        for (slot, weights) in gate_weights.iter().enumerate() {
            if weights.len() != n_ff * gate_row_bytes {
                return Err(format!(
                    "GLM batched sparse IQ2_XXS gate[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_ff * gate_row_bytes
                ));
            }
        }
        for (slot, weights) in up_weights.iter().enumerate() {
            if weights.len() != n_ff * gate_row_bytes {
                return Err(format!(
                    "GLM batched sparse IQ2_XXS up[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_ff * gate_row_bytes
                ));
            }
        }
        for (slot, weights) in down_weights.iter().enumerate() {
            if weights.len() != n_embd * down_row_bytes {
                return Err(format!(
                    "GLM batched sparse IQ3_XXS down[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_embd * down_row_bytes
                ));
            }
        }

        let grouped_slots = tuning::glm_expert_grouped_enabled(token_count, slots)
            .then(|| {
                group_glm_slots(
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                    token_ids,
                )
            })
            .filter(|grouped| grouped.group_meta.len() / 2 < slots);
        let (gate_weights, up_weights, down_weights, route_weights, token_ids, group_meta) =
            if let Some(grouped) = grouped_slots.as_ref() {
                (
                    grouped.gate_weights.as_slice(),
                    grouped.up_weights.as_slice(),
                    grouped.down_weights.as_slice(),
                    grouped.route_weights.as_slice(),
                    grouped.token_ids.as_slice(),
                    grouped.group_meta.as_slice(),
                )
            } else {
                (
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                    token_ids,
                    &[][..],
                )
            };
        if !group_meta.is_empty()
            && std::env::var("RNB_CUDA_GLM_EXPERT_GROUPED_TRACE")
                .ok()
                .as_deref()
                == Some("1")
        {
            let mut groups_by_len = [0usize; 4];
            for group in group_meta.chunks_exact(2) {
                groups_by_len[group[1] as usize - 1] += 1;
            }
            eprintln!(
                "[cuda-glm-expert-grouped] slots={slots} groups={} len1={} len2={} len3={} len4={}",
                group_meta.len() / 2,
                groups_by_len[0],
                groups_by_len[1],
                groups_by_len[2],
                groups_by_len[3],
            );
        }

        self.set_current()?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = token_count * n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let gate_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }

        let direct_file_pipeline = direct_file && tuning::glm_direct_file_pipeline_enabled();
        let (gate_ptrs, up_ptrs, down_ptrs, temp_slab_ptrs) = if direct_file {
            let file_regions = file_regions.ok_or_else(|| {
                "direct file GLM prefill requires file-backed GGUF expert tensors".to_string()
            })?;
            self.temp_q4k_slot_ptrs_3_direct_file(
                gate_weights,
                up_weights,
                down_weights,
                file_regions,
            )?
        } else {
            self.temp_q4k_slot_ptrs_3(gate_weights, up_weights, down_weights)?
        };
        let ptr_bytes = slots * std::mem::size_of::<u64>();
        let route_bytes = std::mem::size_of_val(route_weights);
        let token_bytes = std::mem::size_of_val(token_ids);
        let group_meta_bytes = std::mem::size_of_val(group_meta);
        let meta_bytes = ptr_bytes * 3 + route_bytes + token_bytes + group_meta_bytes;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(meta_bytes)?;
        let up_ptrs_dev = gate_ptrs_dev + ptr_bytes as u64;
        let down_ptrs_dev = gate_ptrs_dev + (ptr_bytes * 2) as u64;
        let route_dev = gate_ptrs_dev + (ptr_bytes * 3) as u64;
        let token_ids_dev = route_dev + route_bytes as u64;
        let group_meta_dev = token_ids_dev + token_bytes as u64;
        let mut meta = vec![0u8; meta_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gate_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr(),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                up_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                down_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes * 2),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes * 3),
                route_bytes,
            );
            std::ptr::copy_nonoverlapping(
                token_ids.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes * 3 + route_bytes),
                token_bytes,
            );
            std::ptr::copy_nonoverlapping(
                group_meta.as_ptr().cast::<u8>(),
                meta.as_mut_ptr()
                    .add(ptr_bytes * 3 + route_bytes + token_bytes),
                group_meta_bytes,
            );
            self.api.memcpy_htod_async(
                gate_ptrs_dev,
                meta.as_ptr().cast::<libc::c_void>(),
                meta_bytes,
                self.stream,
            )?;
        }

        if group_meta.is_empty() {
            self.launch_selected_iq2_xxs_gate_up_gemv_by_token_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                token_ids_dev,
                n_ff,
                slots,
                n_embd / 256,
                input_dev,
                gate_dev,
                up_dev,
            )?;
        } else {
            self.launch_selected_iq2_xxs_gate_up_gemv_by_token_group4(
                gate_ptrs_dev,
                up_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_ff,
                group_meta.len() / 2,
                n_embd / 256,
                input_dev,
                gate_dev,
                up_dev,
            )?;
            self.launch_silu_mul(gate_dev, up_dev, slots * n_ff)?;
        }
        if direct_file_pipeline
            || (!temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled())
        {
            unsafe { self.api.stream_synchronize(self.copy_stream)? };
        }
        if group_meta.is_empty() {
            self.launch_selected_down_silu_rowreduce_by_token(
                "rnb_iq3_xxs_selected_down_silu_rowreduce_by_token",
                down_ptrs_dev,
                n_embd,
                slots / token_count,
                token_count,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            )?;
        } else {
            unsafe {
                self.api
                    .memset_d32_async(output_dev, 0, token_count * n_embd, self.stream)?;
            }
            self.launch_selected_iq3_xxs_down_accum_by_token_group4(
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                group_meta.len() / 2,
                n_ff / 256,
                gate_dev,
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
        self.release_compute_temp_slab()?;
        Ok(output)
    }

    pub(in crate::runtime) fn glm_shared_expert_q5k_q6k(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let gate_dev = self.compute_mid_a_ptr(n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.launch_q5k_gemv_to_dev(gate_weights, n_ff, n_embd / 256, input_dev, gate_dev)?;
        self.launch_q5k_gemv_to_dev(up_weights, n_ff, n_embd / 256, input_dev, up_dev)?;
        self.launch_silu_mul(gate_dev, up_dev, n_ff)?;
        self.launch_q6k_gemv_to_dev(down_weights, n_embd, n_ff / 256, gate_dev, output_dev)?;
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
}
