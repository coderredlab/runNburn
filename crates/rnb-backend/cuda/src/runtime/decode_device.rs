use super::*;

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_full_layer_device_resident(
        &mut self,
        layer_idx: usize,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        attn_norm: &[f32],
        ffn_norm: &[f32],
        n_embd: usize,
        n_ff: usize,
        num_heads: usize,
        num_kv_heads: usize,
        _head_dim: usize,
        kv_dim: usize,
        q_rows: usize,
        q_norm_weight: Option<&[f32]>,
        k_norm_weight: Option<&[f32]>,
        out_scale: f32,
        rope_theta: f32,
        rope_pos: usize,
        kv_len: usize,
        norm_eps: f32,
        hidden_dev: u64,
    ) -> Result<(), String> {
        self.set_current()?;
        if layer_idx <= 4 && crate::tuning::cu63_sync_diag() {
            eprintln!(
                "[cu63-entry] L{layer_idx} q_rows={q_rows} kv={kv_dim} v_w={} k_w={}",
                v_weights.len(),
                k_weights.len()
            );
        }
        let q_blocks = n_embd / 256;
        let kv_blocks = kv_dim / 256;
        let ff_blocks = n_embd / 256;
        let down_blocks = n_ff / 256;
        let kv_rows = kv_dim;
        let f32_size = std::mem::size_of::<f32>();

        // Allocate intermediate buffers. Each compute_*_ptr returns a single
        // reusable buffer — calling it twice with different sizes can realloc and
        // invalidate the first pointer. Use max(all sizes) per buffer slot.
        let norm_dev = self.decode_rms_input_ptr(n_embd * f32_size)?;
        let q_dev = self.compute_input_ptr(q_rows * f32_size)?;
        let mid_a_size = std::cmp::max(kv_rows, n_embd) * f32_size;
        let mid_a_dev = self.compute_mid_a_ptr(mid_a_size)?;
        let k_dev = mid_a_dev;
        let v_dev = self.compute_mid_b_ptr(kv_rows * f32_size)?;
        let attn_out_dev = self.compute_output_ptr(q_rows * f32_size)?;
        let gate_dev = self.compute_gate_ptrs_ptr(n_ff * f32_size)?;
        let up_dev = self.compute_up_ptrs_ptr(n_ff * f32_size)?;
        let down_dev = mid_a_dev;

        // 1. Attention norm: hidden → norm_dev
        self.launch_rms_norm_device(hidden_dev, attn_norm, n_embd, norm_eps, norm_dev)?;

        if layer_idx == 0 && crate::tuning::cu63_sync_diag() {
            self.stream_synchronize()?;
            let mut dbg = vec![0.0f32; n_embd.min(8)];
            unsafe {
                self.api.memcpy_dtoh_async(
                    dbg.as_mut_ptr().cast::<libc::c_void>(),
                    hidden_dev,
                    dbg.len() * 4,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            eprintln!("[cu63-dbg] L0 hidden_dev[0..8]: {:?}", dbg);
            let mut dbg2 = vec![0.0f32; n_embd.min(8)];
            unsafe {
                self.api.memcpy_dtoh_async(
                    dbg2.as_mut_ptr().cast::<libc::c_void>(),
                    norm_dev,
                    dbg2.len() * 4,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            eprintln!("[cu63-dbg] L0 norm_dev[0..8]: {:?}", dbg2);
        }

        // 2. QKV GEMV (device→device, no sync)
        self.q4k_gemv_device_to_device(q_weights, q_rows, q_blocks, norm_dev, q_dev)?;
        self.q4k_gemv_device_to_device(k_weights, kv_rows, kv_blocks, norm_dev, k_dev)?;
        let expected_v_q4k_bytes = kv_rows * kv_blocks * 144;
        if v_weights.len() == expected_v_q4k_bytes {
            self.q4k_gemv_device_to_device(v_weights, kv_rows, kv_blocks, norm_dev, v_dev)?;
        } else {
            self.q6k_gemv_device_to_device(v_weights, kv_rows, kv_blocks, norm_dev, v_dev)?;
        }

        if layer_idx == 0 && crate::tuning::cu63_sync_diag() {
            self.stream_synchronize()?;
            let mut q_dbg = vec![0.0f32; 8];
            unsafe {
                self.api.memcpy_dtoh_async(
                    q_dbg.as_mut_ptr().cast::<libc::c_void>(),
                    q_dev,
                    q_dbg.len() * 4,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            eprintln!("[cu63-dbg] L0 q_dev[0..8]: {:?}", q_dbg);
        }

        // 2b. QK norm (per-head RMS norm) — required for Gemma4 to stabilize attention
        let actual_head_dim = kv_dim / num_kv_heads.max(1);
        if let Some(qn) = q_norm_weight {
            self.launch_qk_norm_device(q_dev, qn, num_heads, actual_head_dim, norm_eps)?;
        }
        if let Some(kn) = k_norm_weight {
            self.launch_qk_norm_device(k_dev, kn, num_kv_heads, actual_head_dim, norm_eps)?;
        }

        // 3. RoPE in-place
        self.launch_rope_decode(
            q_dev,
            k_dev,
            num_heads,
            num_kv_heads,
            actual_head_dim,
            rope_theta,
            rope_pos,
        )?;

        // 4. KV cache f16 write
        self.launch_kv_f16_write(layer_idx, k_dev, v_dev, kv_dim, kv_len)?;

        // 5. Attention decode (device Q + device KV cache → attn_out_dev)
        self.launch_attention_decode_device(
            layer_idx,
            q_dev,
            attn_out_dev,
            num_heads,
            num_kv_heads,
            actual_head_dim,
            kv_len,
        )?;

        // 6. Output projection + residual
        self.q4k_gemv_device_to_device(o_weights, n_embd, q_rows / 256, attn_out_dev, down_dev)?;
        self.launch_add_f32_inplace(hidden_dev, down_dev, n_embd)?;

        if layer_idx <= 4 && crate::tuning::cu63_sync_diag() {
            self.stream_synchronize()?;
            let mut chk = vec![0.0f32; 4];
            unsafe {
                self.api.memcpy_dtoh_async(
                    chk.as_mut_ptr().cast::<libc::c_void>(),
                    hidden_dev,
                    16,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            eprintln!("[cu63-step6] L{layer_idx} after attn+res: {:?}", chk);
        }

        // 7. FFN norm: hidden → norm_dev
        self.launch_rms_norm_device(hidden_dev, ffn_norm, n_embd, norm_eps, norm_dev)?;

        // 8. FFN gate + up (device→device)
        self.q4k_gemv_device_to_device(gate_weights, n_ff, ff_blocks, norm_dev, gate_dev)?;
        self.q4k_gemv_device_to_device(up_weights, n_ff, ff_blocks, norm_dev, up_dev)?;

        if layer_idx == 0 && crate::tuning::cu63_sync_diag() {
            self.stream_synchronize()?;
            let mut chk = vec![0.0f32; 4];
            unsafe {
                self.api.memcpy_dtoh_async(
                    chk.as_mut_ptr().cast::<libc::c_void>(),
                    gate_dev,
                    16,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            eprintln!("[cu63-step8] L0 gate[0..4]: {:?}", chk);
        }

        // 9. GELU(gate) × up → gate_dev in-place (Gemma4 uses GELU, not SiLU)
        self.launch_gelu_mul(gate_dev, up_dev, n_ff)?;

        if layer_idx == 0 && crate::tuning::cu63_sync_diag() {
            self.stream_synchronize()?;
            let mut chk = vec![0.0f32; 4];
            unsafe {
                self.api.memcpy_dtoh_async(
                    chk.as_mut_ptr().cast::<libc::c_void>(),
                    gate_dev,
                    16,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            eprintln!("[cu63-step9] L0 silu_mul[0..4]: {:?}", chk);
        }

        // 10. FFN down + residual. Quant type varies per layer (Q6K for sliding window,
        // Q4K for full attention in Gemma4 E2B Q4_K_M). Detect from byte size.
        let expected_q4k_down_bytes = n_embd * down_blocks * 144;
        if down_weights.len() == expected_q4k_down_bytes {
            self.q4k_gemv_device_to_device(down_weights, n_embd, down_blocks, gate_dev, down_dev)?;
        } else {
            self.q6k_gemv_device_to_device(down_weights, n_embd, down_blocks, gate_dev, down_dev)?;
        }
        self.launch_add_f32_inplace(hidden_dev, down_dev, n_embd)?;

        // 11. Layer output scale (Gemma4: dampens residual accumulation)
        if out_scale != 1.0 && out_scale != 0.0 {
            self.launch_scale_f32_inplace(hidden_dev, out_scale, n_embd)?;
        }
        if layer_idx <= 4 && crate::tuning::cu63_sync_diag() {
            eprintln!("[cu63-scale] L{layer_idx} out_scale={out_scale}");
        }

        if crate::tuning::cu63_sync_diag() {
            self.stream_synchronize()?;
            let mut chk = vec![0.0f32; 4];
            unsafe {
                self.api.memcpy_dtoh_async(
                    chk.as_mut_ptr().cast::<libc::c_void>(),
                    hidden_dev,
                    chk.len() * 4,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            if chk.iter().any(|v| v.is_nan() || v.is_infinite()) {
                eprintln!("[cu63-nan] NaN/Inf at layer {layer_idx} end: {:?}", chk);
            }
        }

        Ok(())
    }

    // --- Launch helpers (cu63 Tasks 4+5) ---

    /// Upload host f32 norm weight to device scratch, then launch RMS norm kernel.
    /// Uses `decode_norm_buf_carrier_ptr` which is not used by the GEMV compute pipeline.
    fn launch_rms_norm_device(
        &mut self,
        input_dev: u64,
        weight: &[f32],
        dim: usize,
        eps: f32,
        output_dev: u64,
    ) -> Result<(), String> {
        let weight_bytes = dim * std::mem::size_of::<f32>();
        let weight_dev = self.decode_norm_buf_carrier_ptr(weight_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                weight_dev,
                weight.as_ptr().cast::<libc::c_void>(),
                weight_bytes,
                self.stream,
            )?;
        }
        self.launch_rms_norm_f32(input_dev, weight_dev, output_dev, eps, dim, false)
    }

    fn launch_qk_norm_device(
        &mut self,
        data_dev: u64,
        norm_weight: &[f32],
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let weight_bytes = head_dim * std::mem::size_of::<f32>();
        let weight_dev = self.decode_norm_buf_carrier_ptr(weight_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                weight_dev,
                norm_weight.as_ptr().cast::<libc::c_void>(),
                weight_bytes,
                self.stream,
            )?;
        }
        let mut input_arg = data_dev;
        let mut weight_arg = weight_dev;
        let mut output_arg = data_dev;
        let mut eps_arg = eps;
        let mut rows_arg = num_heads as u32;
        let mut len_arg = head_dim as u32;
        let mut unit_offset_arg = 0u32; // standard RMS norm (weight * x), NOT unit_offset
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
            (num_heads as u32, 1, 1),
            (256, 1, 1),
        )
    }

    /// RoPE NeoX decode (single token, any head_dim) — in-place on device Q and K.
    /// grid = (num_heads + num_kv_heads), block = (256).
    fn launch_rope_decode(
        &mut self,
        q_dev: u64,
        k_dev: u64,
        num_heads: usize,
        num_kv_heads: usize,
        actual_head_dim: usize,
        theta: f32,
        pos: usize,
    ) -> Result<(), String> {
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut hd_arg = actual_head_dim as u32;
        let mut theta_arg = theta;
        let mut pos_arg = pos as u32;
        self.launch_cached_gemv(
            "rnb_rope_neox_decode",
            &[
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut hd_arg as *mut u32).cast::<libc::c_void>(),
                (&mut theta_arg as *mut f32).cast::<libc::c_void>(),
                (&mut pos_arg as *mut u32).cast::<libc::c_void>(),
            ],
            ((num_heads + num_kv_heads) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    fn launch_rope_decode_pos_dev(
        &mut self,
        q_dev: u64,
        k_dev: u64,
        num_heads: usize,
        num_kv_heads: usize,
        actual_head_dim: usize,
        theta: f32,
        pos_dev: u64,
    ) -> Result<(), String> {
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut hd_arg = actual_head_dim as u32;
        let mut theta_arg = theta;
        let mut pos_arg = pos_dev;
        self.launch_cached_gemv(
            "rnb_rope_neox_decode_pos_dev",
            &[
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut hd_arg as *mut u32).cast::<libc::c_void>(),
                (&mut theta_arg as *mut f32).cast::<libc::c_void>(),
                (&mut pos_arg as *mut u64).cast::<libc::c_void>(),
            ],
            ((num_heads + num_kv_heads) as u32, 1, 1),
            (256, 1, 1),
        )
    }

    /// Convert f32 K/V on device to f16 and append to the per-layer KV cache.
    /// `kv_len` is the write position (0-indexed, the slot for the new token).
    /// After this call the cache holds `kv_len + 1` tokens.
    fn launch_kv_f16_write(
        &mut self,
        layer_idx: usize,
        k_dev: u64,
        v_dev: u64,
        kv_dim: usize,
        kv_len: usize,
    ) -> Result<(), String> {
        // Total tokens after this write.
        let total_tokens = kv_len + 1;
        let required_bytes = total_tokens
            .checked_mul(kv_dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "cu63 kv_f16_write: capacity overflow".to_string())?;

        let mut cache = self
            .decode_attention_kv
            .remove(&layer_idx)
            .unwrap_or_default();

        // Reset if kv_rows changed or cached_tokens went backwards.
        if cache.kv_rows != kv_dim || cache.cached_tokens > total_tokens {
            if let Some(ptr) = cache.k_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            if let Some(ptr) = cache.v_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            cache = DecodeAttentionKvCache {
                kv_rows: kv_dim,
                ..Default::default()
            };
        }

        // Grow if needed.
        if cache.k_bits_capacity < required_bytes || cache.v_bits_capacity < required_bytes {
            if let Some(ptr) = cache.k_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            if let Some(ptr) = cache.v_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            let capacity = align_up(required_bytes, 1024 * 1024);
            let k_ptr = match unsafe { self.api.mem_alloc(capacity) } {
                Ok(p) => p,
                Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                    let _ = self.offload_non_pinned_resident_q4k();
                    match unsafe { self.api.mem_alloc(capacity) } {
                        Ok(p) => p,
                        Err(err2) if cuda_mem_alloc_oom(&err2) => {
                            self.clear_resident_moe_layer_cache()?;
                            unsafe { self.api.mem_alloc(capacity)? }
                        }
                        Err(err2) => return Err(err2),
                    }
                }
                Err(err) => return Err(err),
            };
            let v_ptr = match unsafe { self.api.mem_alloc(capacity) } {
                Ok(p) => p,
                Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                    let _ = self.offload_non_pinned_resident_q4k();
                    match unsafe { self.api.mem_alloc(capacity) } {
                        Ok(p) => p,
                        Err(err2) if cuda_mem_alloc_oom(&err2) => {
                            self.clear_resident_moe_layer_cache()?;
                            unsafe { self.api.mem_alloc(capacity)? }
                        }
                        Err(err2) => return Err(err2),
                    }
                }
                Err(err) => return Err(err),
            };
            cache.k_bits_dev = Some(k_ptr);
            cache.v_bits_dev = Some(v_ptr);
            cache.k_bits_capacity = capacity;
            cache.v_bits_capacity = capacity;
            cache.cached_tokens = 0;
        }

        let k_cache_dev = cache
            .k_bits_dev
            .ok_or_else(|| "cu63 kv_f16_write: missing K buffer".to_string())?;
        let v_cache_dev = cache
            .v_bits_dev
            .ok_or_else(|| "cu63 kv_f16_write: missing V buffer".to_string())?;

        // Launch rnb_f32_to_f16_kv_write for K, then V.
        // kernel signature: (kv_cache: *half, src: *f32, dim: u32, pos: u32, max_seq: u32)
        let dim_u32 = kv_dim as u32;
        let pos_u32 = kv_len as u32;
        let max_seq_u32 = total_tokens as u32;

        // K write
        {
            let mut cache_arg = k_cache_dev;
            let mut src_arg = k_dev;
            let mut dim_arg = dim_u32;
            let mut pos_arg = pos_u32;
            let mut max_seq_arg = max_seq_u32;
            self.launch_cached_gemv(
                "rnb_f32_to_f16_kv_write",
                &[
                    (&mut cache_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut src_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut dim_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut pos_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut max_seq_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (1, 1, 1),
                (256, 1, 1),
            )?;
        }
        // V write
        {
            let mut cache_arg = v_cache_dev;
            let mut src_arg = v_dev;
            let mut dim_arg = dim_u32;
            let mut pos_arg = pos_u32;
            let mut max_seq_arg = max_seq_u32;
            self.launch_cached_gemv(
                "rnb_f32_to_f16_kv_write",
                &[
                    (&mut cache_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut src_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut dim_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut pos_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut max_seq_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (1, 1, 1),
                (256, 1, 1),
            )?;
        }

        cache.cached_tokens = total_tokens;
        self.decode_attention_kv.insert(layer_idx, cache);
        Ok(())
    }

    /// Attention decode with Q and KV cache already on device. No H2D for Q, no D2H for output.
    /// Dispatches the standard attention decode kernel (hd128/256/512).
    pub(super) fn launch_attention_decode_device(
        &mut self,
        layer_idx: usize,
        q_dev: u64,
        output_dev: u64,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
    ) -> Result<(), String> {
        if !matches!(head_dim, 128 | 256 | 512) {
            return Err(format!(
                "cu63 attention_decode_device: unsupported head_dim={head_dim}"
            ));
        }
        // kv_len here is the write position of the current token. The attention
        // kernel needs to attend over all tokens including the one just written,
        // so the actual sequence length is kv_len + 1.
        let attn_kv_len = kv_len + 1;

        let cache = self.decode_attention_kv.get(&layer_idx).ok_or_else(|| {
            format!("cu63 attention_decode_device: no KV cache for layer {layer_idx}")
        })?;
        let k_cache_dev = cache
            .k_bits_dev
            .ok_or_else(|| "cu63 attention_decode_device: missing K buffer".to_string())?;
        let v_cache_dev = cache
            .v_bits_dev
            .ok_or_else(|| "cu63 attention_decode_device: missing V buffer".to_string())?;

        // Gemma4: attention scale = 1.0 (Q is pre-scaled by query_pre_attn_scalar
        // during projection, not during attention score computation).
        // TODO: pass attn_scale as parameter for non-Gemma4 models.
        let scale = 1.0f32;

        let kernel = match head_dim {
            128 => "rnb_attention_decode_hd128",
            256 => "rnb_attention_decode_hd256",
            512 => "rnb_attention_decode_hd512",
            _ => unreachable!("validated head_dim"),
        };

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_cache_dev;
        let mut v_arg = v_cache_dev;
        let mut kv_len_arg = attn_kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (num_heads as u32, 1, 1),
            (head_dim as u32, 1, 1),
        )
    }

    pub(super) fn launch_attention_decode_device_len_device(
        &mut self,
        layer_idx: usize,
        q_dev: u64,
        output_dev: u64,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
    ) -> Result<(), String> {
        if head_dim != 512 {
            return Err(format!(
                "cu68 attention_decode_device_len_device: unsupported head_dim={head_dim}"
            ));
        }
        let attn_kv_len = kv_len + 1;

        let cache = self.decode_attention_kv.get(&layer_idx).ok_or_else(|| {
            format!("cu68 attention_decode_device_len_device: no KV cache for layer {layer_idx}")
        })?;
        let k_cache_dev = cache.k_bits_dev.ok_or_else(|| {
            "cu68 attention_decode_device_len_device: missing K buffer".to_string()
        })?;
        let v_cache_dev = cache.v_bits_dev.ok_or_else(|| {
            "cu68 attention_decode_device_len_device: missing V buffer".to_string()
        })?;

        let kv_len_dev = self.cu68_graph_kv_len_ptr()?;
        let kv_len_value = attn_kv_len as u32;
        unsafe {
            self.api.memcpy_htod_async(
                kv_len_dev,
                (&kv_len_value as *const u32).cast::<libc::c_void>(),
                std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }

        let scale = 1.0f32;
        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_cache_dev;
        let mut v_arg = v_cache_dev;
        let mut kv_len_dev_arg = kv_len_dev;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        self.launch_cached_gemv(
            "rnb_attention_decode_hd512_len_device",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut kv_len_dev_arg as *mut u64).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (num_heads as u32, 1, 1),
            (512, 1, 1),
        )
    }

    pub(super) fn populate_device_kv_cache_f16(
        &mut self,
        layer_idx: usize,
        k_bits: &[u16],
        v_bits: &[u16],
        kv_dim: usize,
        num_tokens: usize,
    ) -> Result<(), String> {
        self.set_current()?;
        let required_bytes = num_tokens * kv_dim * std::mem::size_of::<u16>();
        if k_bits.len() < num_tokens * kv_dim || v_bits.len() < num_tokens * kv_dim {
            return Err(format!(
                "cu63 populate_kv: bits too short — k={} v={} need={}",
                k_bits.len(),
                v_bits.len(),
                num_tokens * kv_dim,
            ));
        }

        let mut cache = self
            .decode_attention_kv
            .remove(&layer_idx)
            .unwrap_or_default();

        if cache.kv_rows != kv_dim || cache.k_bits_capacity < required_bytes {
            if let Some(ptr) = cache.k_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            if let Some(ptr) = cache.v_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            let capacity = align_up(required_bytes, 1024 * 1024);
            let k_ptr = unsafe { self.api.mem_alloc(capacity)? };
            let v_ptr = unsafe { self.api.mem_alloc(capacity)? };
            cache.k_bits_dev = Some(k_ptr);
            cache.v_bits_dev = Some(v_ptr);
            cache.k_bits_capacity = capacity;
            cache.v_bits_capacity = capacity;
            cache.kv_rows = kv_dim;
        }

        let k_dev = cache.k_bits_dev.unwrap();
        let v_dev = cache.v_bits_dev.unwrap();
        unsafe {
            self.api.memcpy_htod_async(
                k_dev,
                k_bits.as_ptr().cast::<libc::c_void>(),
                required_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                v_dev,
                v_bits.as_ptr().cast::<libc::c_void>(),
                required_bytes,
                self.stream,
            )?;
        }
        cache.cached_tokens = num_tokens;
        self.decode_attention_kv.insert(layer_idx, cache);
        Ok(())
    }

    /// cu66: device QKV + QK norm + RoPE + f16 K/V pack on device.
    /// K/V f16 packed into existing carriers (skips cu52 host H2D + pack).
    /// Q is copied to a dedicated carrier; K/V f16 carriers feed the existing attention path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_device_qkv_rope_kv(
        &mut self,
        layer_idx: usize,
        norm_carrier_dev: u64,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_norm_weight: Option<&[f32]>,
        k_norm_weight: Option<&[f32]>,
        q_rows: usize,
        kv_dim: usize,
        n_embd: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
        rope_pos: usize,
        kv_len: usize,
        norm_eps: f32,
        q_host_out: &mut [f32],
        k_host_out: &mut [f32],
        v_host_out: &mut [f32],
    ) -> Result<u64, String> {
        self.decode_device_qkv_rope_kv_inner(
            layer_idx,
            norm_carrier_dev,
            q_weights,
            k_weights,
            v_weights,
            q_norm_weight,
            k_norm_weight,
            q_rows,
            kv_dim,
            n_embd,
            num_heads,
            num_kv_heads,
            rope_theta,
            rope_pos,
            kv_len,
            norm_eps,
            q_host_out,
            k_host_out,
            v_host_out,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_device_qkv_rope_kv_graph(
        &mut self,
        layer_idx: usize,
        norm_carrier_dev: u64,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_norm_weight: Option<&[f32]>,
        k_norm_weight: Option<&[f32]>,
        q_rows: usize,
        kv_dim: usize,
        n_embd: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
        rope_pos: usize,
        kv_len: usize,
        norm_eps: f32,
        q_host_out: &mut [f32],
        k_host_out: &mut [f32],
        v_host_out: &mut [f32],
    ) -> Result<u64, String> {
        self.set_current()?;
        let f32_size = std::mem::size_of::<f32>();
        let f16_size = std::mem::size_of::<u16>();
        let actual_head_dim = kv_dim / num_kv_heads.max(1);

        let q_dev = self.compute_input_ptr(q_rows * f32_size)?;
        let k_dev = self.compute_mid_a_ptr(kv_dim * f32_size)?;
        let v_dev = self.compute_mid_b_ptr(kv_dim * f32_size)?;
        let q_carrier_dev = self.decode_q_carrier_ptr(q_rows * f32_size)?;
        let k_f16_dev = self.decode_k_f16_carrier_ptr(kv_dim * f16_size)?;
        let v_f16_dev = self.decode_v_f16_carrier_ptr(kv_dim * f16_size)?;
        let pos_dev = self.cu65_graph_pos_ptr()?;

        let pos_arg = rope_pos as u32;
        unsafe {
            self.api.memcpy_htod_async(
                pos_dev,
                (&pos_arg as *const u32).cast::<libc::c_void>(),
                std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }

        let q_norm_key = q_norm_weight.map(f32_key);
        let k_norm_key = k_norm_weight.map(f32_key);
        let key = Cu65QkvGraphKey {
            layer_idx,
            q_rows,
            kv_dim,
            n_embd,
            num_heads,
            num_kv_heads,
            actual_head_dim,
            rope_theta_bits: rope_theta.to_bits(),
            norm_eps_bits: norm_eps.to_bits(),
            norm_carrier_dev,
            q_dev,
            k_dev,
            v_dev,
            q_carrier_dev,
            k_f16_dev,
            v_f16_dev,
            pos_dev,
            q_weight_ptr: q_weights.as_ptr() as usize,
            q_weight_len: q_weights.len(),
            k_weight_ptr: k_weights.as_ptr() as usize,
            k_weight_len: k_weights.len(),
            v_weight_ptr: v_weights.as_ptr() as usize,
            v_weight_len: v_weights.len(),
            q_norm_ptr: q_norm_key.map(|key| key.ptr).unwrap_or(0),
            q_norm_len: q_norm_key.map(|key| key.len).unwrap_or(0),
            q_norm_hash: q_norm_key.map(|key| key.bit_hash).unwrap_or(0),
            k_norm_ptr: k_norm_key.map(|key| key.ptr).unwrap_or(0),
            k_norm_len: k_norm_key.map(|key| key.len).unwrap_or(0),
            k_norm_hash: k_norm_key.map(|key| key.bit_hash).unwrap_or(0),
        };

        if let Some(graph) = self.cu65_qkv_graphs.get(&key) {
            unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?
            };
            return Ok(q_carrier_dev);
        }

        if !self.cu65_qkv_graph_warmed.contains(&key) {
            self.cu65_qkv_graph_warmed.insert(key);
            return self.decode_device_qkv_rope_kv_inner(
                layer_idx,
                norm_carrier_dev,
                q_weights,
                k_weights,
                v_weights,
                q_norm_weight,
                k_norm_weight,
                q_rows,
                kv_dim,
                n_embd,
                num_heads,
                num_kv_heads,
                rope_theta,
                rope_pos,
                kv_len,
                norm_eps,
                q_host_out,
                k_host_out,
                v_host_out,
                Some(pos_dev),
            );
        }

        self.ensure_q4k_gemv_module()?;
        let _ = self.resident_q4k_weights_ptr(q_weights)?;
        let _ = self.resident_q4k_weights_ptr(k_weights)?;
        let _ = self.resident_q4k_weights_ptr(v_weights)?;
        if let Some(weight) = q_norm_weight {
            let _ = self.resident_f32_ptr(weight)?;
        }
        if let Some(weight) = k_norm_weight {
            let _ = self.resident_f32_ptr(weight)?;
        }

        unsafe { self.api.stream_begin_capture(self.stream)? };
        let capture_result = self.decode_device_qkv_rope_kv_inner(
            layer_idx,
            norm_carrier_dev,
            q_weights,
            k_weights,
            v_weights,
            q_norm_weight,
            k_norm_weight,
            q_rows,
            kv_dim,
            n_embd,
            num_heads,
            num_kv_heads,
            rope_theta,
            rope_pos,
            kv_len,
            norm_eps,
            q_host_out,
            k_host_out,
            v_host_out,
            Some(pos_dev),
        );
        let captured_q = match capture_result {
            Ok(ptr) => ptr,
            Err(err) => {
                unsafe {
                    let _ = self.api.stream_end_capture(self.stream);
                }
                return Err(err);
            }
        };
        if captured_q != q_carrier_dev {
            unsafe {
                let _ = self.api.stream_end_capture(self.stream);
            }
            return Err("cu65 QKV CUDA graph captured unexpected Q carrier".to_string());
        }
        let graph = unsafe { self.api.stream_end_capture(self.stream)? };
        let exec = unsafe { self.api.graph_instantiate(graph)? };
        self.cu65_qkv_graphs.insert(
            key,
            SparseMoeGraph {
                graph: graph as usize,
                exec: exec as usize,
            },
        );
        let graph = self
            .cu65_qkv_graphs
            .get(&key)
            .ok_or_else(|| "missing cu65 QKV CUDA graph".to_string())?;
        unsafe {
            self.api
                .graph_launch(graph.exec as *mut libc::c_void, self.stream)?
        };
        Ok(q_carrier_dev)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_device_qkv_rope_kv_inner(
        &mut self,
        _layer_idx: usize,
        norm_carrier_dev: u64,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_norm_weight: Option<&[f32]>,
        k_norm_weight: Option<&[f32]>,
        q_rows: usize,
        kv_dim: usize,
        n_embd: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
        rope_pos: usize,
        _kv_len: usize,
        norm_eps: f32,
        _q_host_out: &mut [f32],
        _k_host_out: &mut [f32],
        _v_host_out: &mut [f32],
        rope_pos_dev: Option<u64>,
    ) -> Result<u64, String> {
        self.set_current()?;
        let f32_size = std::mem::size_of::<f32>();
        let q_blocks = n_embd / 256;
        let kv_blocks = n_embd / 256;
        let actual_head_dim = kv_dim / num_kv_heads.max(1);

        // Allocate device buffers for Q, K, V
        let q_dev = self.compute_input_ptr(q_rows * f32_size)?;
        let k_dev = self.compute_mid_a_ptr(kv_dim * f32_size)?;
        let v_dev = self.compute_mid_b_ptr(kv_dim * f32_size)?;

        // QKV GEMV (device→device, no sync)
        self.q4k_gemv_device_to_device(q_weights, q_rows, q_blocks, norm_carrier_dev, q_dev)?;
        self.q4k_gemv_device_to_device(k_weights, kv_dim, kv_blocks, norm_carrier_dev, k_dev)?;
        let expected_v_q4k = kv_dim * kv_blocks * 144;
        if v_weights.len() == expected_v_q4k {
            self.q4k_gemv_device_to_device(v_weights, kv_dim, kv_blocks, norm_carrier_dev, v_dev)?;
        } else {
            self.q6k_gemv_device_to_device(v_weights, kv_dim, kv_blocks, norm_carrier_dev, v_dev)?;
        }

        // QK norm (per-head RMS norm)
        if let Some(qn) = q_norm_weight {
            self.launch_qk_norm_device(q_dev, qn, num_heads, actual_head_dim, norm_eps)?;
        }
        if let Some(kn) = k_norm_weight {
            self.launch_qk_norm_device(k_dev, kn, num_kv_heads, actual_head_dim, norm_eps)?;
        }

        // RoPE (device in-place)
        if let Some(pos_dev) = rope_pos_dev {
            self.launch_rope_decode_pos_dev(
                q_dev,
                k_dev,
                num_heads,
                num_kv_heads,
                actual_head_dim,
                rope_theta,
                pos_dev,
            )?;
        } else {
            self.launch_rope_decode(
                q_dev,
                k_dev,
                num_heads,
                num_kv_heads,
                actual_head_dim,
                rope_theta,
                rope_pos,
            )?;
        }

        // cu66: Q to dedicated carrier (compute_input_ptr is shared).
        let q_carrier = self.decode_q_carrier_ptr(q_rows * f32_size)?;
        unsafe {
            self.api
                .memcpy_dtod_async(q_carrier, q_dev, q_rows * f32_size, self.stream)?;
        }

        // cu66: pack K/V to f16 on device for last_token_k/v_dev.
        let f16_size = std::mem::size_of::<u16>();
        let k_f16_dev = self.decode_k_f16_carrier_ptr(kv_dim * f16_size)?;
        let v_f16_dev = self.decode_v_f16_carrier_ptr(kv_dim * f16_size)?;
        self.launch_f32_to_f16_pack(k_dev, k_f16_dev, kv_dim)?;
        self.launch_f32_to_f16_pack(v_dev, v_f16_dev, kv_dim)?;

        // cu66: no D2H, no sync. Caller must gate to non-sliding-window layers.

        Ok(q_carrier)
    }
}
