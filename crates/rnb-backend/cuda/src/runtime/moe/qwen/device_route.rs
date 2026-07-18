use super::super::super::*;

#[allow(dead_code)]
pub(in crate::runtime) struct Qwen35HostRoutePack {
    pub(in crate::runtime) expert_ids: Vec<u32>,
    pub(in crate::runtime) route_weights: Vec<f32>,
    pub(in crate::runtime) token_ids: Vec<u32>,
    pub(in crate::runtime) seq_len: usize,
    pub(in crate::runtime) n_expert: usize,
    pub(in crate::runtime) n_expert_used: usize,
}

impl Qwen35HostRoutePack {
    pub(in crate::runtime) fn sort_by_expert_token(&mut self) -> Result<(), String> {
        qwen35_sort_route_arrays_by_expert_token(
            &mut self.expert_ids,
            &mut self.route_weights,
            &mut self.token_ids,
            self.seq_len,
            self.n_expert,
            self.n_expert_used,
        )
    }

    #[cfg(test)]
    pub(in crate::runtime) fn to_route_slots(&self) -> Vec<rnb_backend_api::MoeRouteSlot> {
        self.expert_ids
            .iter()
            .zip(self.route_weights.iter())
            .zip(self.token_ids.iter())
            .map(|((&expert, &weight), &token)| {
                rnb_backend_api::MoeRouteSlot::new(expert as usize, token, weight)
            })
            .collect()
    }
}

impl CudaState {
    pub(in crate::runtime) fn qwen35_prefill_device_topk_route_pack(
        &mut self,
        router_w: &[f32],
        n_expert: usize,
        hidden_dim: usize,
        norm_all: &[f32],
        seq_len: usize,
        n_expert_used: usize,
    ) -> Result<Qwen35HostRoutePack, String> {
        self.validate_qwen35_prefill_device_topk_route_shape(
            router_w,
            n_expert,
            hidden_dim,
            seq_len,
            n_expert_used,
        )?;
        let expected_input = seq_len
            .checked_mul(hidden_dim)
            .ok_or_else(|| "Qwen35 prefill device top-k input shape overflow".to_string())?;
        if norm_all.len() != expected_input {
            return Err(format!(
                "Qwen35 prefill device top-k input len mismatch: got {}, expected {expected_input}",
                norm_all.len()
            ));
        }
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(norm_all))?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                norm_all.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(norm_all),
                self.stream,
            )?;
        }
        self.qwen35_prefill_device_topk_route_pack_from_ptr(
            router_w,
            n_expert,
            hidden_dim,
            input_dev,
            seq_len,
            n_expert_used,
        )
    }

    pub(in crate::runtime) fn qwen35_prefill_device_topk_route_pack_device_input(
        &mut self,
        router_w: &[f32],
        n_expert: usize,
        hidden_dim: usize,
        input_id: rnb_backend_api::DeviceTensorId,
        input_desc: rnb_backend_api::DeviceTensorDesc,
        seq_len: usize,
        n_expert_used: usize,
    ) -> Result<Qwen35HostRoutePack, String> {
        self.validate_qwen35_prefill_device_topk_route_shape(
            router_w,
            n_expert,
            hidden_dim,
            seq_len,
            n_expert_used,
        )?;
        let expected_desc = rnb_backend_api::DeviceTensorDesc::new(
            seq_len,
            hidden_dim,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::Normalized,
        );
        if input_desc != expected_desc {
            return Err(format!(
                "Qwen35 prefill device top-k input desc mismatch: got {:?}, expected {:?}",
                input_desc, expected_desc
            ));
        }
        let input_dev = self.device_tensor_ptr(input_id, input_desc)?;
        self.qwen35_prefill_device_topk_route_pack_from_ptr(
            router_w,
            n_expert,
            hidden_dim,
            input_dev,
            seq_len,
            n_expert_used,
        )
    }

    fn validate_qwen35_prefill_device_topk_route_shape(
        &self,
        router_w: &[f32],
        n_expert: usize,
        hidden_dim: usize,
        seq_len: usize,
        n_expert_used: usize,
    ) -> Result<(), String> {
        if seq_len == 0 || hidden_dim == 0 || n_expert == 0 || n_expert_used == 0 {
            return Err(format!(
                "Qwen35 prefill device top-k requires non-zero dims: seq_len={seq_len} hidden_dim={hidden_dim} n_expert={n_expert} used={n_expert_used}"
            ));
        }
        if n_expert_used > n_expert || n_expert_used > 32 {
            return Err(format!(
                "Qwen35 prefill device top-k selected expert count invalid: used={n_expert_used} experts={n_expert}"
            ));
        }

        let expected_router = n_expert
            .checked_mul(hidden_dim)
            .ok_or_else(|| "Qwen35 prefill device top-k router shape overflow".to_string())?;
        if router_w.len() != expected_router {
            return Err(format!(
                "Qwen35 prefill device top-k router len mismatch: got {}, expected {expected_router}",
                router_w.len()
            ));
        }
        Ok(())
    }

    fn qwen35_prefill_device_topk_route_pack_from_ptr(
        &mut self,
        router_w: &[f32],
        n_expert: usize,
        hidden_dim: usize,
        input_dev: u64,
        seq_len: usize,
        n_expert_used: usize,
    ) -> Result<Qwen35HostRoutePack, String> {
        let slots = seq_len
            .checked_mul(n_expert_used)
            .ok_or_else(|| "Qwen35 prefill device top-k slot overflow".to_string())?;
        let logits_bytes = seq_len
            .checked_mul(n_expert)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Qwen35 prefill device top-k logits byte overflow".to_string())?;
        let route_bytes = slots
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Qwen35 prefill device top-k route byte overflow".to_string())?;
        let ids_bytes = slots
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "Qwen35 prefill device top-k id byte overflow".to_string())?;

        let logits_dev = self.compute_aux_output_ptr(logits_bytes)?;
        let expert_ids_dev = self.compute_gate_ptrs_ptr(ids_bytes)?;
        let route_weights_dev = self.compute_route_ptr(route_bytes)?;
        let token_ids_dev = self.compute_token_ids_ptr(ids_bytes)?;
        let router_dev = self.resident_f32_ptr(router_w)?;

        self.sgemm_device(
            router_dev, n_expert, hidden_dim, input_dev, seq_len, logits_dev,
        )?;
        self.launch_qwen35_router_topk_logits_f32(
            logits_dev,
            expert_ids_dev,
            route_weights_dev,
            token_ids_dev,
            seq_len,
            n_expert,
            n_expert_used,
        )?;

        let mut expert_ids = vec![0_u32; slots];
        let mut route_weights = vec![0.0_f32; slots];
        let mut token_ids = vec![0_u32; slots];
        unsafe {
            self.api.memcpy_dtoh_async(
                expert_ids.as_mut_ptr().cast::<libc::c_void>(),
                expert_ids_dev,
                std::mem::size_of_val(expert_ids.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                route_weights.as_mut_ptr().cast::<libc::c_void>(),
                route_weights_dev,
                std::mem::size_of_val(route_weights.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                token_ids.as_mut_ptr().cast::<libc::c_void>(),
                token_ids_dev,
                std::mem::size_of_val(token_ids.as_slice()),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        normalize_qwen35_route_logits_host(&mut route_weights, seq_len, n_expert_used);

        Ok(Qwen35HostRoutePack {
            expert_ids,
            route_weights,
            token_ids,
            seq_len,
            n_expert,
            n_expert_used,
        })
    }
}

fn normalize_qwen35_route_logits_host(
    route_weights: &mut [f32],
    seq_len: usize,
    n_expert_used: usize,
) {
    for token in 0..seq_len {
        let start = token * n_expert_used;
        let selected = &mut route_weights[start..start + n_expert_used];
        let max_selected = selected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut selected_sum = 0.0f32;
        for value in selected.iter_mut() {
            *value = (*value - max_selected).exp();
            selected_sum += *value;
        }
        for value in selected.iter_mut() {
            *value /= selected_sum;
        }
    }
}
