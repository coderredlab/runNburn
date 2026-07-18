use super::super::*;

macro_rules! ensure_with_oom_retry {
    ($self:expr, $ptr:ident, $cap:ident, $bytes:expr) => {{
        let bytes = $bytes;
        $self.set_current()?;
        let first = ensure_device_buffer(&$self.api, &mut $self.$ptr, &mut $self.$cap, bytes);
        match first {
            Ok(p) => Ok(p),
            Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                let _ = $self.offload_non_pinned_resident_q4k();
                $self.set_current()?;
                let second =
                    ensure_device_buffer(&$self.api, &mut $self.$ptr, &mut $self.$cap, bytes);
                match second {
                    Ok(p) => Ok(p),
                    Err(err2) if cuda_mem_alloc_oom(&err2) => {
                        $self.clear_resident_moe_layer_cache()?;
                        $self.set_current()?;
                        ensure_device_buffer(&$self.api, &mut $self.$ptr, &mut $self.$cap, bytes)
                    }
                    Err(err2) => Err(err2),
                }
            }
            Err(err) => Err(err),
        }
    }};
}

impl CudaState {
    pub(in crate::runtime) fn compute_weights_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_weights, compute_weights_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_input_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_input, compute_input_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_output_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_output, compute_output_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_aux_output_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_aux_output, compute_aux_output_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_mid_a_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_mid_a, compute_mid_a_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_mid_b_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_mid_b, compute_mid_b_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_gate_ptrs_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_gate_ptrs, compute_gate_ptrs_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_up_ptrs_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_up_ptrs, compute_up_ptrs_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_down_ptrs_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_down_ptrs, compute_down_ptrs_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_full_gate_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_full_gate, compute_full_gate_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_full_up_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_full_up, compute_full_up_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_full_down_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_full_down, compute_full_down_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_temp_slab_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_temp_slab, compute_temp_slab_capacity, bytes)
    }

    pub(in crate::runtime) fn release_compute_temp_slab(&mut self) -> Result<(), String> {
        self.set_current()?;
        if let Some(ptr) = self.compute_temp_slab.take() {
            unsafe { self.api.mem_free(ptr)? };
        }
        self.compute_temp_slab_capacity = 0;
        Ok(())
    }

    pub(in crate::runtime) fn qwen35_packed_act_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, qwen35_packed_act, qwen35_packed_act_capacity, bytes)
    }

    pub(in crate::runtime) fn host_temp_slab_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<*mut u8, String> {
        self.set_current()?;
        if self.host_temp_slab_capacity >= bytes {
            return self
                .host_temp_slab
                .map(|ptr| ptr as *mut u8)
                .ok_or_else(|| "missing CUDA pinned host temp slab".to_string());
        }
        if let Some(ptr) = self.host_temp_slab.take() {
            unsafe { self.api.mem_free_host(ptr as *mut libc::c_void)? };
        }
        let ptr = unsafe { self.api.mem_host_alloc(bytes)? };
        self.host_temp_slab = Some(ptr as usize);
        self.host_temp_slab_capacity = bytes;
        Ok(ptr.cast::<u8>())
    }

    pub(in crate::runtime) fn host_sparse_input_slab_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<*mut u8, String> {
        self.set_current()?;
        if self.host_sparse_input_slab_capacity >= bytes {
            return self
                .host_sparse_input_slab
                .map(|ptr| ptr as *mut u8)
                .ok_or_else(|| "missing CUDA pinned sparse input slab".to_string());
        }
        if let Some(ptr) = self.host_sparse_input_slab.take() {
            unsafe { self.api.mem_free_host(ptr as *mut libc::c_void)? };
        }
        let ptr = unsafe { self.api.mem_host_alloc(bytes)? };
        self.host_sparse_input_slab = Some(ptr as usize);
        self.host_sparse_input_slab_capacity = bytes;
        Ok(ptr.cast::<u8>())
    }

    pub(in crate::runtime) fn compute_route_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_route, compute_route_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_token_ids_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_token_ids, compute_token_ids_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_group_meta_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_group_meta, compute_group_meta_capacity, bytes)
    }

    pub(in crate::runtime) fn gemma_ple_base_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, gemma_ple_base, gemma_ple_base_capacity, bytes)
    }

    // cu29 Phase 2: hd128 QKV+RoPE pipeline 의 GPU output buffers.
    pub(in crate::runtime) fn compute_q_rope_out_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_q_rope_out, compute_q_rope_out_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_k_bits_out_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_k_bits_out, compute_k_bits_out_capacity, bytes)
    }

    pub(in crate::runtime) fn compute_v_bits_out_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(self, compute_v_bits_out, compute_v_bits_out_capacity, bytes)
    }

    // cu41 Phase 1: dedicated decode hidden carrier — attention/gdn 와 분리.
    pub(in crate::runtime) fn decode_hidden_carrier_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(
            self,
            decode_hidden_carrier,
            decode_hidden_carrier_capacity,
            bytes
        )
    }

    // cu41 step 8: RMS norm input 전용 buffer — compute_input cache race 회피.
    pub(in crate::runtime) fn decode_rms_input_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, decode_rms_input, decode_rms_input_capacity, bytes)
    }

    // cu42 step 12: RMS norm 결과 (norm_buf) 용 carrier.
    pub(in crate::runtime) fn decode_norm_buf_carrier_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(
            self,
            decode_norm_buf_carrier,
            decode_norm_buf_carrier_capacity,
            bytes
        )
    }

    // cu47 step 32: attention forward 의 device output buffer (attn_out).
    pub(in crate::runtime) fn decode_attn_out_carrier_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(
            self,
            decode_attn_out_carrier,
            decode_attn_out_carrier_capacity,
            bytes
        )
    }

    // cu49 step 38: K/V projection 의 device output buffer.
    pub(in crate::runtime) fn decode_k_carrier_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, decode_k_carrier, decode_k_carrier_capacity, bytes)
    }

    pub(in crate::runtime) fn decode_v_carrier_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, decode_v_carrier, decode_v_carrier_capacity, bytes)
    }

    // cu52 step 47: K/V f16 carrier (attention compute 의 input 호환).
    pub(in crate::runtime) fn decode_k_f16_carrier_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(
            self,
            decode_k_f16_carrier,
            decode_k_f16_carrier_capacity,
            bytes
        )
    }

    pub(in crate::runtime) fn decode_v_f16_carrier_ptr(
        &mut self,
        bytes: usize,
    ) -> Result<u64, String> {
        ensure_with_oom_retry!(
            self,
            decode_v_f16_carrier,
            decode_v_f16_carrier_capacity,
            bytes
        )
    }

    pub(in crate::runtime) fn decode_q_carrier_ptr(&mut self, bytes: usize) -> Result<u64, String> {
        ensure_with_oom_retry!(self, decode_q_carrier, decode_q_carrier_capacity, bytes)
    }

    pub(in crate::runtime) fn cu65_graph_pos_ptr(&mut self) -> Result<u64, String> {
        ensure_with_oom_retry!(self, cu65_graph_pos, cu65_graph_pos_capacity, 4)
    }

    pub(in crate::runtime) fn cu68_graph_kv_len_ptr(&mut self) -> Result<u64, String> {
        ensure_with_oom_retry!(self, cu68_graph_kv_len, cu68_graph_kv_len_capacity, 4)
    }
}
