use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn resident_delta_state_ptr(
        &mut self,
        state: &[f32],
    ) -> Result<u64, String> {
        self.set_current()?;
        let key = (state.as_ptr() as usize, state.len());
        if let Some(entry) = self.resident_delta_states.get(&key) {
            return Ok(entry.ptr);
        }
        let bytes = std::mem::size_of_val(state);
        self.reclaim_residency_for_transient(bytes)?;
        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                state.as_ptr().cast::<libc::c_void>(),
                bytes,
                self.stream,
            )
        }?;
        self.resident_delta_states
            .insert(key, ResidentDeltaState { ptr });
        Ok(ptr)
    }

    pub(in crate::runtime) fn clear_resident_delta_states(&mut self) -> Result<(), String> {
        self.set_current()?;
        self.stream_synchronize()?;
        for (_, entry) in self.resident_delta_states.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        Ok(())
    }

    pub(in crate::runtime) fn sync_resident_delta_state(
        &mut self,
        state: &mut [f32],
    ) -> Result<bool, String> {
        self.set_current()?;
        let key = (state.as_ptr() as usize, state.len());
        let Some(entry) = self.resident_delta_states.get(&key) else {
            return Ok(false);
        };
        unsafe {
            self.api.memcpy_dtoh_async(
                state.as_mut_ptr().cast::<libc::c_void>(),
                entry.ptr,
                std::mem::size_of_val(state),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(true)
    }

    pub(in crate::runtime) fn allocate_delta_state_snapshot(
        &mut self,
        bytes: usize,
    ) -> Result<DeltaStateSnapshot, String> {
        self.set_current()?;
        if bytes == 0 {
            return Err("delta state snapshot requires non-zero bytes".to_string());
        }
        if !crate::tuning::mtp_verify_snapshot_pool_enabled() {
            self.reclaim_residency_for_transient(bytes)?;
            return Ok(DeltaStateSnapshot {
                ptr: unsafe { self.api.mem_alloc(bytes) }?,
                bytes,
                pool_slot: None,
            });
        }
        if let Some((slot, entry)) = self
            .mtp_verify_snapshot_pool
            .iter_mut()
            .enumerate()
            .filter(|(_, entry)| !entry.in_use && entry.capacity >= bytes)
            .min_by_key(|(_, entry)| entry.capacity)
        {
            entry.in_use = true;
            return Ok(DeltaStateSnapshot {
                ptr: entry.ptr,
                bytes,
                pool_slot: Some(slot),
            });
        }
        self.reclaim_residency_for_transient(bytes)?;
        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        let slot = self.mtp_verify_snapshot_pool.len();
        self.mtp_verify_snapshot_pool
            .push(MtpVerifySnapshotPoolEntry {
                ptr,
                capacity: bytes,
                in_use: true,
            });
        Ok(DeltaStateSnapshot {
            ptr,
            bytes,
            pool_slot: Some(slot),
        })
    }

    pub(in crate::runtime) fn snapshot_resident_delta_state(
        &mut self,
        state: &mut [f32],
    ) -> Result<Option<DeltaStateSnapshot>, String> {
        self.set_current()?;
        let key = (state.as_ptr() as usize, state.len());
        let Some(state_ptr) = self.resident_delta_states.get(&key).map(|entry| entry.ptr) else {
            return Ok(None);
        };
        let bytes = std::mem::size_of_val(state);
        let snapshot = self.allocate_delta_state_snapshot(bytes)?;
        unsafe {
            self.api
                .memcpy_dtod_async(snapshot.ptr, state_ptr, bytes, self.stream)?;
        }
        self.stream_synchronize()?;
        Ok(Some(snapshot))
    }

    pub(in crate::runtime) fn restore_resident_delta_state(
        &mut self,
        state: &mut [f32],
        snapshot: &DeltaStateSnapshot,
    ) -> Result<bool, String> {
        self.set_current()?;
        let key = (state.as_ptr() as usize, state.len());
        let Some(entry) = self.resident_delta_states.get(&key) else {
            return Ok(false);
        };
        let bytes = std::mem::size_of_val(state);
        if snapshot.bytes != bytes {
            return Err(format!(
                "delta state snapshot byte mismatch: snapshot={}, state={}",
                snapshot.bytes, bytes
            ));
        }
        unsafe {
            self.api
                .memcpy_dtod_async(entry.ptr, snapshot.ptr, bytes, self.stream)?;
        }
        self.stream_synchronize()?;
        Ok(true)
    }

    pub(in crate::runtime) fn free_delta_state_snapshot(
        &mut self,
        snapshot: DeltaStateSnapshot,
    ) -> Result<(), String> {
        self.set_current()?;
        if let Some(slot) = snapshot.pool_slot {
            let entry = self
                .mtp_verify_snapshot_pool
                .get_mut(slot)
                .ok_or_else(|| format!("delta state snapshot pool slot out of range: {slot}"))?;
            if entry.ptr != snapshot.ptr || !entry.in_use {
                return Err("delta state snapshot pool ownership mismatch".to_string());
            }
            entry.in_use = false;
            return Ok(());
        }
        self.stream_synchronize()?;
        unsafe { self.api.mem_free(snapshot.ptr) }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn delta_net_decode(
        &mut self,
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        sync_state_to_host: bool,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let gate_bytes = std::mem::size_of_val(gate);
        let beta_bytes = std::mem::size_of_val(beta);
        let output_len = num_heads * head_v_dim;
        let output_bytes = output_len * std::mem::size_of::<f32>();

        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let gate_dev = self.compute_gate_ptrs_ptr(gate_bytes)?;
        let beta_dev = self.compute_up_ptrs_ptr(beta_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let state_dev = self.resident_delta_state_ptr(state)?;

        unsafe {
            self.api.memcpy_htod_async(
                q_dev,
                q.as_ptr().cast::<libc::c_void>(),
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                k_dev,
                k.as_ptr().cast::<libc::c_void>(),
                k_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                v_dev,
                v.as_ptr().cast::<libc::c_void>(),
                v_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                gate_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                gate_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                beta_dev,
                beta.as_ptr().cast::<libc::c_void>(),
                beta_bytes,
                self.stream,
            )?;
        }

        let mut output_arg = output_dev;
        let mut state_arg = state_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut gate_arg = gate_dev;
        let mut beta_arg = beta_dev;
        let mut heads_arg = num_heads as u32;
        let mut head_k_arg = head_k_dim as u32;
        let mut head_v_arg = head_v_dim as u32;
        self.launch_cached_gemv(
            "rnb_delta_net_decode",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_k_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (head_v_dim as u32, num_heads as u32, 1),
            (256, 1, 1),
        )?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
            if sync_state_to_host {
                self.api.memcpy_dtoh_async(
                    state.as_mut_ptr().cast::<libc::c_void>(),
                    state_dev,
                    std::mem::size_of_val(state),
                    self.stream,
                )?;
            }
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_mamba2_decode_scan(
        &mut self,
        state: &mut [f32],
        x: &[f32],
        b: &[f32],
        c: &[f32],
        dt: &[f32],
        a: &[f32],
        d: &[f32],
        num_heads: usize,
        head_dim: usize,
        state_dim: usize,
        n_group: usize,
    ) -> Result<Vec<f32>, String> {
        if state_dim > 256 {
            return Err(format!(
                "unsupported_state_dim: state_dim={state_dim} exceeds decode scan block width 256"
            ));
        }
        let output_len = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Nemotron Mamba2 decode output len overflow".to_string())?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Nemotron Mamba2 decode output byte overflow".to_string())?;
        let x_dev = self.compute_input_ptr(std::mem::size_of_val(x))?;
        let b_dev = self.compute_mid_a_ptr(std::mem::size_of_val(b))?;
        let c_dev = self.compute_mid_b_ptr(std::mem::size_of_val(c))?;
        let dt_dev = self.compute_route_ptr(std::mem::size_of_val(dt))?;
        let a_dev = self.compute_gate_ptrs_ptr(std::mem::size_of_val(a))?;
        let d_dev = self.compute_up_ptrs_ptr(std::mem::size_of_val(d))?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let state_dev = self.resident_delta_state_ptr(state)?;
        unsafe {
            self.api.memcpy_htod_async(
                x_dev,
                x.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(x),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                b_dev,
                b.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(b),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                c_dev,
                c.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(c),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                dt_dev,
                dt.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(dt),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                a_dev,
                a.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(a),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                d_dev,
                d.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(d),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut state_arg = state_dev;
        let mut x_arg = x_dev;
        let mut b_arg = b_dev;
        let mut c_arg = c_dev;
        let mut dt_arg = dt_dev;
        let mut a_arg = a_dev;
        let mut d_arg = d_dev;
        let mut heads_arg = u32::try_from(num_heads).map_err(|_| {
            format!("Nemotron Mamba2 decode num_heads exceeds CUDA u32 limit: {num_heads}")
        })?;
        let mut head_dim_arg = u32::try_from(head_dim).map_err(|_| {
            format!("Nemotron Mamba2 decode head_dim exceeds CUDA u32 limit: {head_dim}")
        })?;
        let mut state_dim_arg = u32::try_from(state_dim).map_err(|_| {
            format!("Nemotron Mamba2 decode state_dim exceeds CUDA u32 limit: {state_dim}")
        })?;
        let mut groups_arg = u32::try_from(n_group).map_err(|_| {
            format!("Nemotron Mamba2 decode n_group exceeds CUDA u32 limit: {n_group}")
        })?;
        self.launch_cached_gemv(
            "rnb_nemotron_mamba2_decode_scan",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                (&mut x_arg as *mut u64).cast::<libc::c_void>(),
                (&mut b_arg as *mut u64).cast::<libc::c_void>(),
                (&mut c_arg as *mut u64).cast::<libc::c_void>(),
                (&mut dt_arg as *mut u64).cast::<libc::c_void>(),
                (&mut a_arg as *mut u64).cast::<libc::c_void>(),
                (&mut d_arg as *mut u64).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut state_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut groups_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (head_dim_arg, heads_arg, 1),
            (256, 1, 1),
        )?;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                state.as_mut_ptr().cast::<libc::c_void>(),
                state_dev,
                std::mem::size_of_val(state),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_mamba2_prefill_scan(
        &mut self,
        state: &mut [f32],
        conv_activated: &[f32],
        dt_data: &[f32],
        a: &[f32],
        d: &[f32],
        seq_len: usize,
        conv_channels: usize,
        bc_dim: usize,
        num_heads: usize,
        head_dim: usize,
        state_dim: usize,
        n_group: usize,
    ) -> Result<Vec<f32>, String> {
        if state_dim > 256 {
            return Err(format!(
                "unsupported_state_dim: state_dim={state_dim} exceeds prefill scan block width 256"
            ));
        }
        let d_inner = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Nemotron Mamba2 prefill d_inner overflow".to_string())?;
        let output_len = seq_len
            .checked_mul(d_inner)
            .ok_or_else(|| "Nemotron Mamba2 prefill output len overflow".to_string())?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Nemotron Mamba2 prefill output byte overflow".to_string())?;
        let conv_dev = self.compute_input_ptr(std::mem::size_of_val(conv_activated))?;
        let dt_dev = self.compute_route_ptr(std::mem::size_of_val(dt_data))?;
        let a_dev = self.compute_gate_ptrs_ptr(std::mem::size_of_val(a))?;
        let d_dev = self.compute_up_ptrs_ptr(std::mem::size_of_val(d))?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let state_dev = self.resident_delta_state_ptr(state)?;
        unsafe {
            self.api.memcpy_htod_async(
                conv_dev,
                conv_activated.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(conv_activated),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                dt_dev,
                dt_data.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(dt_data),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                a_dev,
                a.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(a),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                d_dev,
                d.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(d),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut state_arg = state_dev;
        let mut conv_arg = conv_dev;
        let mut dt_arg = dt_dev;
        let mut a_arg = a_dev;
        let mut d_arg = d_dev;
        let mut seq_arg = u32::try_from(seq_len).map_err(|_| {
            format!("Nemotron Mamba2 prefill seq_len exceeds CUDA u32 limit: {seq_len}")
        })?;
        let mut conv_channels_arg = u32::try_from(conv_channels).map_err(|_| {
            format!("Nemotron Mamba2 prefill conv_channels exceeds CUDA u32 limit: {conv_channels}")
        })?;
        let mut bc_dim_arg = u32::try_from(bc_dim).map_err(|_| {
            format!("Nemotron Mamba2 prefill bc_dim exceeds CUDA u32 limit: {bc_dim}")
        })?;
        let mut heads_arg = u32::try_from(num_heads).map_err(|_| {
            format!("Nemotron Mamba2 prefill num_heads exceeds CUDA u32 limit: {num_heads}")
        })?;
        let mut head_dim_arg = u32::try_from(head_dim).map_err(|_| {
            format!("Nemotron Mamba2 prefill head_dim exceeds CUDA u32 limit: {head_dim}")
        })?;
        let mut state_dim_arg = u32::try_from(state_dim).map_err(|_| {
            format!("Nemotron Mamba2 prefill state_dim exceeds CUDA u32 limit: {state_dim}")
        })?;
        let mut groups_arg = u32::try_from(n_group).map_err(|_| {
            format!("Nemotron Mamba2 prefill n_group exceeds CUDA u32 limit: {n_group}")
        })?;
        self.launch_cached_gemv(
            "rnb_nemotron_mamba2_prefill_scan",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                (&mut conv_arg as *mut u64).cast::<libc::c_void>(),
                (&mut dt_arg as *mut u64).cast::<libc::c_void>(),
                (&mut a_arg as *mut u64).cast::<libc::c_void>(),
                (&mut d_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut conv_channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut bc_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut state_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut groups_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (head_dim_arg, heads_arg, 1),
            (256, 1, 1),
        )?;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                state.as_mut_ptr().cast::<libc::c_void>(),
                state_dev,
                std::mem::size_of_val(state),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_nemotron_mamba2_prefill_scan_dev(
        &mut self,
        output_dev: u64,
        state: &mut [f32],
        conv_dev: u64,
        dt_dev: u64,
        a_dev: u64,
        d_dev: u64,
        seq_len: usize,
        conv_channels: usize,
        bc_dim: usize,
        num_heads: usize,
        head_dim: usize,
        state_dim: usize,
        n_group: usize,
    ) -> Result<(), String> {
        if state_dim > 256 {
            return Err(format!(
                "unsupported_state_dim: state_dim={state_dim} exceeds prefill scan block width 256"
            ));
        }
        let state_dev = self.resident_delta_state_ptr(state)?;
        let mut output_arg = output_dev;
        let mut state_arg = state_dev;
        let mut conv_arg = conv_dev;
        let mut dt_arg = dt_dev;
        let mut a_arg = a_dev;
        let mut d_arg = d_dev;
        let mut seq_arg = u32::try_from(seq_len).map_err(|_| {
            format!("Nemotron Mamba2 prefill scan seq_len exceeds CUDA u32 limit: {seq_len}")
        })?;
        let mut conv_channels_arg = u32::try_from(conv_channels).map_err(|_| {
            format!(
                "Nemotron Mamba2 prefill scan conv_channels exceeds CUDA u32 limit: {conv_channels}"
            )
        })?;
        let mut bc_dim_arg = u32::try_from(bc_dim).map_err(|_| {
            format!("Nemotron Mamba2 prefill scan bc_dim exceeds CUDA u32 limit: {bc_dim}")
        })?;
        let mut heads_arg = u32::try_from(num_heads).map_err(|_| {
            format!("Nemotron Mamba2 prefill scan num_heads exceeds CUDA u32 limit: {num_heads}")
        })?;
        let mut head_dim_arg = u32::try_from(head_dim).map_err(|_| {
            format!("Nemotron Mamba2 prefill scan head_dim exceeds CUDA u32 limit: {head_dim}")
        })?;
        let mut state_dim_arg = u32::try_from(state_dim).map_err(|_| {
            format!("Nemotron Mamba2 prefill scan state_dim exceeds CUDA u32 limit: {state_dim}")
        })?;
        let mut groups_arg = u32::try_from(n_group).map_err(|_| {
            format!("Nemotron Mamba2 prefill scan n_group exceeds CUDA u32 limit: {n_group}")
        })?;
        self.launch_cached_gemv(
            "rnb_nemotron_mamba2_prefill_scan",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                (&mut conv_arg as *mut u64).cast::<libc::c_void>(),
                (&mut dt_arg as *mut u64).cast::<libc::c_void>(),
                (&mut a_arg as *mut u64).cast::<libc::c_void>(),
                (&mut d_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut conv_channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut bc_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut state_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut groups_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (head_dim_arg, heads_arg, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    pub(in crate::runtime) fn delta_net_prefill(
        &mut self,
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        sync_state_to_host: bool,
    ) -> Result<Vec<f32>, String> {
        self.delta_net_prefill_with_snapshot(
            state,
            q,
            k,
            v,
            gate,
            beta,
            seq_len,
            num_heads,
            head_k_dim,
            head_v_dim,
            sync_state_to_host,
            None,
        )
        .map(|(output, _)| output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn delta_net_prefill_with_snapshot(
        &mut self,
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        sync_state_to_host: bool,
        snapshot_after_tokens: Option<usize>,
    ) -> Result<(Vec<f32>, Option<DeltaStateSnapshot>), String> {
        if let Some(tokens) = snapshot_after_tokens {
            if tokens == 0 || tokens > seq_len {
                return Err(format!(
                    "delta prefill snapshot token count {tokens} out of range for seq_len {seq_len}"
                ));
            }
        }
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let gate_bytes = std::mem::size_of_val(gate);
        let beta_bytes = std::mem::size_of_val(beta);
        let output_len = seq_len * num_heads * head_v_dim;
        let output_bytes = output_len * std::mem::size_of::<f32>();

        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let gate_dev = self.compute_gate_ptrs_ptr(gate_bytes)?;
        let beta_dev = self.compute_up_ptrs_ptr(beta_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let state_dev = self.resident_delta_state_ptr(state)?;
        let snapshot = if snapshot_after_tokens.is_some() {
            let bytes = std::mem::size_of_val(state);
            Some(self.allocate_delta_state_snapshot(bytes)?)
        } else {
            None
        };

        unsafe {
            self.api.memcpy_htod_async(
                q_dev,
                q.as_ptr().cast::<libc::c_void>(),
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                k_dev,
                k.as_ptr().cast::<libc::c_void>(),
                k_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                v_dev,
                v.as_ptr().cast::<libc::c_void>(),
                v_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                gate_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                gate_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                beta_dev,
                beta.as_ptr().cast::<libc::c_void>(),
                beta_bytes,
                self.stream,
            )?;
        }

        let mut output_arg = output_dev;
        let mut state_arg = state_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut gate_arg = gate_dev;
        let mut beta_arg = beta_dev;
        let mut seq_arg = seq_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut snapshot_arg = snapshot.as_ref().map(|s| s.ptr).unwrap_or(0);
        let mut snapshot_after_arg = snapshot_after_tokens.unwrap_or(0) as u32;
        if head_k_dim == 128 && tuning::prefill_delta_k128_warp4_enabled() {
            let mut head_v_arg = head_v_dim as u32;
            self.launch_cached_gemv(
                "rnb_delta_net_prefill_k128_warp4",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut snapshot_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut snapshot_after_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (
                    ((head_v_dim as u32).saturating_add(3)) / 4,
                    num_heads as u32,
                    1,
                ),
                (32, 4, 1),
            )?;
        } else if head_k_dim == 128 {
            let mut head_v_arg = head_v_dim as u32;
            self.launch_cached_gemv(
                "rnb_delta_net_prefill_k128",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut snapshot_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut snapshot_after_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (head_v_dim as u32, num_heads as u32, 1),
                (128, 1, 1),
            )?;
        } else {
            let mut head_k_arg = head_k_dim as u32;
            let mut head_v_arg = head_v_dim as u32;
            self.launch_cached_gemv(
                "rnb_delta_net_prefill",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut snapshot_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_k_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut snapshot_after_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (head_v_dim as u32, num_heads as u32, 1),
                (256, 1, 1),
            )?;
        }

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
            if sync_state_to_host {
                self.api.memcpy_dtoh_async(
                    state.as_mut_ptr().cast::<libc::c_void>(),
                    state_dev,
                    std::mem::size_of_val(state),
                    self.stream,
                )?;
            }
        }
        self.stream_synchronize()?;
        Ok((output, snapshot))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn delta_net_prefill_with_snapshots(
        &mut self,
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        sync_state_to_host: bool,
        snapshot_after_tokens: &[usize],
    ) -> Result<(Vec<f32>, Vec<DeltaStateSnapshot>), String> {
        if snapshot_after_tokens.is_empty() {
            return Err("delta prefill multi-snapshot requires at least one prefix".to_string());
        }
        for &tokens in snapshot_after_tokens {
            if tokens == 0 || tokens > seq_len {
                return Err(format!(
                    "delta prefill snapshot token count {tokens} out of range for seq_len {seq_len}"
                ));
            }
        }

        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let gate_bytes = std::mem::size_of_val(gate);
        let beta_bytes = std::mem::size_of_val(beta);
        let output_len = seq_len * num_heads * head_v_dim;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let state_bytes = std::mem::size_of_val(state);

        let mut snapshots = Vec::with_capacity(snapshot_after_tokens.len());
        let result = (|| {
            let q_dev = self.compute_input_ptr(q_bytes)?;
            let k_dev = self.compute_mid_a_ptr(k_bytes)?;
            let v_dev = self.compute_mid_b_ptr(v_bytes)?;
            let gate_dev = self.compute_gate_ptrs_ptr(gate_bytes)?;
            let beta_dev = self.compute_up_ptrs_ptr(beta_bytes)?;
            let output_dev = self.compute_output_ptr(output_bytes)?;
            let state_dev = self.resident_delta_state_ptr(state)?;

            let snapshot_bytes = state_bytes
                .checked_mul(snapshot_after_tokens.len())
                .ok_or_else(|| "delta snapshot allocation size overflow".to_string())?;
            self.reclaim_residency_for_transient(snapshot_bytes)?;
            for _ in snapshot_after_tokens {
                snapshots.push(self.allocate_delta_state_snapshot(state_bytes)?);
            }
            let snapshot_ptrs = snapshots
                .iter()
                .map(|snapshot| snapshot.ptr)
                .collect::<Vec<_>>();
            let snapshot_tokens = snapshot_after_tokens
                .iter()
                .map(|&tokens| tokens as u32)
                .collect::<Vec<_>>();
            let snapshot_ptrs_dev =
                self.compute_down_ptrs_ptr(std::mem::size_of_val(snapshot_ptrs.as_slice()))?;
            let snapshot_tokens_dev =
                self.compute_token_ids_ptr(std::mem::size_of_val(snapshot_tokens.as_slice()))?;

            unsafe {
                self.api.memcpy_htod_async(
                    q_dev,
                    q.as_ptr().cast::<libc::c_void>(),
                    q_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    k_dev,
                    k.as_ptr().cast::<libc::c_void>(),
                    k_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    v_dev,
                    v.as_ptr().cast::<libc::c_void>(),
                    v_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    gate_dev,
                    gate.as_ptr().cast::<libc::c_void>(),
                    gate_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    beta_dev,
                    beta.as_ptr().cast::<libc::c_void>(),
                    beta_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    snapshot_ptrs_dev,
                    snapshot_ptrs.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(snapshot_ptrs.as_slice()),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    snapshot_tokens_dev,
                    snapshot_tokens.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(snapshot_tokens.as_slice()),
                    self.stream,
                )?;
            }

            let mut output_arg = output_dev;
            let mut state_arg = state_dev;
            let mut q_arg = q_dev;
            let mut k_arg = k_dev;
            let mut v_arg = v_dev;
            let mut gate_arg = gate_dev;
            let mut beta_arg = beta_dev;
            let mut snapshot_ptrs_arg = snapshot_ptrs_dev;
            let mut snapshot_tokens_arg = snapshot_tokens_dev;
            let mut snapshot_count_arg = snapshot_after_tokens.len() as u32;
            let mut seq_arg = seq_len as u32;
            let mut heads_arg = num_heads as u32;
            if head_k_dim == 128 && tuning::prefill_delta_k128_warp4_enabled() {
                let mut head_v_arg = head_v_dim as u32;
                self.launch_cached_gemv(
                    "rnb_delta_net_prefill_k128_warp4_multi_snapshot",
                    &[
                        (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_tokens_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_count_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    ],
                    (
                        ((head_v_dim as u32).saturating_add(3)) / 4,
                        num_heads as u32,
                        1,
                    ),
                    (32, 4, 1),
                )?;
            } else if head_k_dim == 128 {
                let mut head_v_arg = head_v_dim as u32;
                self.launch_cached_gemv(
                    "rnb_delta_net_prefill_k128_multi_snapshot",
                    &[
                        (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_tokens_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_count_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    ],
                    (head_v_dim as u32, num_heads as u32, 1),
                    (128, 1, 1),
                )?;
            } else {
                let mut head_k_arg = head_k_dim as u32;
                let mut head_v_arg = head_v_dim as u32;
                self.launch_cached_gemv(
                    "rnb_delta_net_prefill_multi_snapshot",
                    &[
                        (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut state_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut gate_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut beta_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_ptrs_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_tokens_arg as *mut u64).cast::<libc::c_void>(),
                        (&mut snapshot_count_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_k_arg as *mut u32).cast::<libc::c_void>(),
                        (&mut head_v_arg as *mut u32).cast::<libc::c_void>(),
                    ],
                    (head_v_dim as u32, num_heads as u32, 1),
                    (256, 1, 1),
                )?;
            }

            let mut output = vec![0.0f32; output_len];
            unsafe {
                self.api.memcpy_dtoh_async(
                    output.as_mut_ptr().cast::<libc::c_void>(),
                    output_dev,
                    output_bytes,
                    self.stream,
                )?;
                if sync_state_to_host {
                    self.api.memcpy_dtoh_async(
                        state.as_mut_ptr().cast::<libc::c_void>(),
                        state_dev,
                        state_bytes,
                        self.stream,
                    )?;
                }
            }
            self.stream_synchronize()?;
            Ok(output)
        })();

        match result {
            Ok(output) => Ok((output, snapshots)),
            Err(err) => {
                self.stream_synchronize().ok();
                for snapshot in snapshots {
                    unsafe {
                        let _ = self.api.mem_free(snapshot.ptr);
                    }
                }
                Err(err)
            }
        }
    }
}
