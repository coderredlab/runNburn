use crate::runtime::CudaState;

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn gemma4_moe_gelu_selected(
        &mut self,
        gate_up_experts: &[u8],
        down_experts: &[u8],
        down_quant: u32,
        n_expert: usize,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        expert_ids: &[u32],
        token_ids: &[u32],
        route_weights: &[f32],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let slots = expert_ids.len();
        if slots == 0 || token_ids.len() != slots || route_weights.len() != slots {
            return Err(format!(
                "Gemma4 grouped MoE route length mismatch: experts={} tokens={} weights={}",
                slots,
                token_ids.len(),
                route_weights.len()
            ));
        }
        if input.len() != seq_len.saturating_mul(n_embd) {
            return Err(format!(
                "Gemma4 grouped MoE input length mismatch: got {}, expected {}",
                input.len(),
                seq_len.saturating_mul(n_embd)
            ));
        }
        if n_embd % 256 != 0 || n_ff % 32 != 0 {
            return Err(format!(
                "Gemma4 grouped MoE unsupported dimensions: n_embd={n_embd} n_ff={n_ff}"
            ));
        }
        let down_block_bytes = match down_quant {
            7 => 24usize,
            8 => 34usize,
            other => {
                return Err(format!(
                    "Gemma4 grouped MoE unsupported down quant code {other}"
                ))
            }
        };
        let gate_matrix_bytes = n_ff
            .checked_mul(n_embd / 256)
            .and_then(|blocks| blocks.checked_mul(144))
            .ok_or_else(|| "Gemma4 grouped MoE gate byte size overflow".to_string())?;
        let per_gate_up = gate_matrix_bytes
            .checked_mul(2)
            .ok_or_else(|| "Gemma4 grouped MoE gate/up byte size overflow".to_string())?;
        let expected_gate_up = n_expert
            .checked_mul(per_gate_up)
            .ok_or_else(|| "Gemma4 grouped MoE total gate/up byte size overflow".to_string())?;
        if gate_up_experts.len() != expected_gate_up {
            return Err(format!(
                "Gemma4 grouped MoE gate/up byte mismatch: got {}, expected {expected_gate_up}",
                gate_up_experts.len()
            ));
        }
        let per_down = n_embd
            .checked_mul(n_ff / 32)
            .and_then(|blocks| blocks.checked_mul(down_block_bytes))
            .ok_or_else(|| "Gemma4 grouped MoE down byte size overflow".to_string())?;
        let expected_down = n_expert
            .checked_mul(per_down)
            .ok_or_else(|| "Gemma4 grouped MoE total down byte size overflow".to_string())?;
        if down_experts.len() != expected_down {
            return Err(format!(
                "Gemma4 grouped MoE down byte mismatch: got {}, expected {expected_down}",
                down_experts.len()
            ));
        }
        for (index, &expert) in expert_ids.iter().enumerate() {
            if expert as usize >= n_expert {
                return Err(format!(
                    "Gemma4 grouped MoE expert index {expert} exceeds {n_expert}"
                ));
            }
            if index > 0 && expert < expert_ids[index - 1] {
                return Err("Gemma4 grouped MoE expert ids must be sorted".to_string());
            }
        }
        for &token in token_ids {
            if token as usize >= seq_len {
                return Err(format!(
                    "Gemma4 grouped MoE token index {token} exceeds {seq_len}"
                ));
            }
        }

        let mut max_group = 0usize;
        let mut start = 0usize;
        while start < slots {
            let expert = expert_ids[start];
            let mut end = start + 1;
            while end < slots && expert_ids[end] == expert {
                end += 1;
            }
            max_group = max_group.max(end - start);
            start = end;
        }

        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
        let group_input_dev = self.compute_full_gate_ptr(
            max_group
                .checked_mul(n_embd)
                .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "Gemma4 grouped MoE input scratch overflow".to_string())?,
        )?;
        let group_output_dev = self.compute_full_down_ptr(
            max_group
                .checked_mul(n_embd)
                .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "Gemma4 grouped MoE output scratch overflow".to_string())?,
        )?;
        let output_len = seq_len
            .checked_mul(n_embd)
            .ok_or_else(|| "Gemma4 grouped MoE output length overflow".to_string())?;
        let output_dev = self.compute_output_ptr(
            output_len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "Gemma4 grouped MoE output byte size overflow".to_string())?,
        )?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                token_ids_dev,
                token_ids.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(token_ids),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                route_dev,
                route_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(route_weights),
                self.stream,
            )?;
        }
        self.launch_zero_f32(output_dev, output_len)?;

        start = 0;
        while start < slots {
            let expert = expert_ids[start] as usize;
            let mut end = start + 1;
            while end < slots && expert_ids[end] as usize == expert {
                end += 1;
            }
            let group_len = end - start;
            let token_group_dev = token_ids_dev + (start * std::mem::size_of::<u32>()) as u64;
            let route_group_dev = route_dev + (start * std::mem::size_of::<f32>()) as u64;
            self.launch_gather_rows_f32(
                input_dev,
                token_group_dev,
                group_input_dev,
                n_embd,
                group_len,
            )?;
            let gate_up_base = expert * per_gate_up;
            let down_base = expert * per_down;
            self.gemma4_moe_gelu_ffn_batch_dev_input_to_dev(
                &gate_up_experts[gate_up_base..gate_up_base + per_gate_up],
                &down_experts[down_base..down_base + per_down],
                down_quant,
                n_ff,
                n_embd,
                group_len,
                group_input_dev,
                group_output_dev,
            )?;
            self.launch_scatter_add_weighted_rows_f32(
                group_output_dev,
                token_group_dev,
                route_group_dev,
                output_dev,
                n_embd,
                group_len,
            )?;
            start = end;
        }

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                std::mem::size_of_val(output.as_slice()),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }
}
