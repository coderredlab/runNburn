use super::*;
fn checked_quant_weight_bytes(
    label: &str,
    rows: usize,
    blocks_per_row: usize,
    block_bytes: usize,
) -> Result<usize, String> {
    rows.checked_mul(blocks_per_row)
        .and_then(|bytes| bytes.checked_mul(block_bytes))
        .ok_or_else(|| format!("device decode {label} byte size overflow"))
}

impl CudaState {
    fn cooperative_norms_supported(&mut self) -> bool {
        if let Some(supported) = self.cooperative_launch_supported {
            return supported;
        }
        let supported = unsafe { self.api.device_supports_cooperative_launch() }.unwrap_or(false);
        self.cooperative_launch_supported = Some(supported);
        supported
    }

    fn pin_full_device_layer_weights(
        &mut self,
        weights: &[&[u8]],
    ) -> Result<Vec<(usize, usize)>, String> {
        let mut leases = Vec::with_capacity(weights.len());
        for &weight in weights {
            match self.resident_q4k_weights_ptr_pinned_with_lease(weight) {
                Ok((_ptr, Some(key))) => leases.push(key),
                Ok((_ptr, None)) => {}
                Err(err) => {
                    for key in leases {
                        self.unpin_resident_q4k_key(key);
                    }
                    return Err(err);
                }
            }
        }
        Ok(leases)
    }
}

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_full_layer_device_resident(
        &mut self,
        layer_idx: usize,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        o_weights: &[u8],
        attention_gate_weights: Option<&[u8]>,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        attn_norm: &[f32],
        post_attn_norm: Option<&[f32]>,
        ffn_norm: &[f32],
        post_ffn_norm: Option<&[f32]>,
        n_embd: usize,
        n_ff: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        q_rows: usize,
        q_norm_weight: Option<&[f32]>,
        k_norm_weight: Option<&[f32]>,
        attention_scale: f32,
        apply_rope: bool,
        rope_neox: bool,
        sliding_window: Option<usize>,
        ffn_uses_gelu: bool,
        out_scale: f32,
        rope_theta: f32,
        rope_pos: usize,
        kv_len: usize,
        norm_eps: f32,
        post_norm_eps: f32,
        hidden_dev: u64,
    ) -> Result<(), String> {
        self.set_current()?;
        if n_embd == 0
            || n_ff == 0
            || !n_embd.is_multiple_of(256)
            || !n_ff.is_multiple_of(256)
            || !q_rows.is_multiple_of(256)
            || num_heads == 0
            || num_kv_heads == 0
            || num_heads % num_kv_heads != 0
            || q_rows != num_heads.saturating_mul(head_dim)
            || kv_dim != num_kv_heads.saturating_mul(head_dim)
            || [
                n_embd,
                n_ff,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_dim,
                q_rows,
                rope_pos,
            ]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
            || kv_len >= u32::MAX as usize
            || sliding_window.is_some_and(|window| window == 0 || window > u32::MAX as usize)
        {
            return Err(format!(
                "device decode layer geometry mismatch: hidden={n_embd} ff={n_ff} q_rows={q_rows} kv_dim={kv_dim} heads={num_heads}/{num_kv_heads} head_dim={head_dim} rope_pos={rope_pos} kv_len={kv_len} sliding_window={sliding_window:?}"
            ));
        }
        let input_blocks = n_embd / 256;
        let q4_q_bytes = checked_quant_weight_bytes("Q", q_rows, input_blocks, 144)?;
        let q4_kv_bytes = checked_quant_weight_bytes("K/V", kv_dim, input_blocks, 144)?;
        let q6_v_bytes = checked_quant_weight_bytes("V Q6_K", kv_dim, input_blocks, 210)?;
        let q4_o_bytes = checked_quant_weight_bytes("O", n_embd, q_rows / 256, 144)?;
        let q4_ffn_bytes = checked_quant_weight_bytes("gate/up", n_ff, input_blocks, 144)?;
        let q4_down_bytes = checked_quant_weight_bytes("down Q4_K", n_embd, n_ff / 256, 144)?;
        let q6_down_bytes = checked_quant_weight_bytes("down Q6_K", n_embd, n_ff / 256, 210)?;
        let v_q6 = match v_weights.len() {
            len if len == q4_kv_bytes => false,
            len if len == q6_v_bytes => true,
            len => {
                return Err(format!(
                    "device decode V weight bytes mismatch: got {len}, expected Q4_K {q4_kv_bytes} or Q6_K {q6_v_bytes}"
                ));
            }
        };
        let down_quant = match down_weights.len() {
            len if len == q4_down_bytes => 12,
            len if len == q6_down_bytes => 14,
            len => {
                return Err(format!(
                    "device decode down weight bytes mismatch: got {len}, expected Q4_K {q4_down_bytes} or Q6_K {q6_down_bytes}"
                ));
            }
        };
        let attention_gate_len_valid =
            attention_gate_weights.is_none_or(|weight| weight.len() == q4_q_bytes);
        let norm_lengths_valid = attn_norm.len() == n_embd
            && ffn_norm.len() == n_embd
            && post_attn_norm.is_none_or(|weight| weight.len() == n_embd)
            && post_ffn_norm.is_none_or(|weight| weight.len() == n_embd)
            && q_norm_weight.is_none_or(|weight| weight.len() == head_dim)
            && k_norm_weight.is_none_or(|weight| weight.len() == head_dim);
        if q_weights.len() != q4_q_bytes
            || k_weights.len() != q4_kv_bytes
            || o_weights.len() != q4_o_bytes
            || gate_weights.len() != q4_ffn_bytes
            || up_weights.len() != q4_ffn_bytes
            || !attention_gate_len_valid
            || !norm_lengths_valid
        {
            return Err(format!(
                "device decode layer weight/norm shape mismatch: q={}/{} k={}/{} o={}/{} gate={}/{} up={}/{} attention_gate={:?}/{} norms=({},{},{:?},{:?},{:?},{:?}) hidden={n_embd} head_dim={head_dim}",
                q_weights.len(),
                q4_q_bytes,
                k_weights.len(),
                q4_kv_bytes,
                o_weights.len(),
                q4_o_bytes,
                gate_weights.len(),
                q4_ffn_bytes,
                up_weights.len(),
                q4_ffn_bytes,
                attention_gate_weights.map(<[u8]>::len),
                q4_q_bytes,
                attn_norm.len(),
                ffn_norm.len(),
                post_attn_norm.map(<[f32]>::len),
                post_ffn_norm.map(<[f32]>::len),
                q_norm_weight.map(<[f32]>::len),
                k_norm_weight.map(<[f32]>::len),
            ));
        }
        let f32_size = std::mem::size_of::<f32>();
        let qkv_q8dot = tuning::full_device_decode_qkv_q8dot_enabled();
        let cooperative_norms = tuning::full_device_decode_cooperative_norms_enabled()
            && self.cooperative_norms_supported();
        let q8_qs_bytes = input_blocks * 256;
        let q8_ds_bytes = input_blocks * 8 * f32_size;
        let norm_bytes = n_embd * f32_size;
        let norm_dev = self.decode_rms_input_ptr(
            norm_bytes
                + if qkv_q8dot {
                    q8_qs_bytes + q8_ds_bytes
                } else {
                    0
                },
        )?;
        let q_dev = self.compute_input_ptr(q_rows * f32_size)?;
        let mid_a_dev = self.compute_mid_a_ptr(kv_dim.max(n_embd) * f32_size)?;
        let k_dev = mid_a_dev;
        let v_dev = self.compute_mid_b_ptr(kv_dim * f32_size)?;
        let attn_out_dev = self.compute_output_ptr(q_rows * f32_size)?;
        let attention_gate_dev = attention_gate_weights
            .map(|_| self.compute_aux_output_ptr(q_rows * f32_size))
            .transpose()?;
        let down_dev = mid_a_dev;
        self.ensure_decode_kv_f16_capacity(layer_idx, kv_dim, kv_len + 1)?;
        self.prepare_muse_decode_layer_streaming_admission()?;
        let mut layer_weights = Vec::with_capacity(8);
        layer_weights.extend_from_slice(&[
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
        ]);
        if let Some(attention_gate) = attention_gate_weights {
            layer_weights.push(attention_gate);
        }
        let weight_leases = self.pin_full_device_layer_weights(&layer_weights)?;
        let result = (|| -> Result<(), String> {
            self.launch_rms_norm_device(
                hidden_dev,
                attn_norm,
                n_embd,
                norm_eps,
                norm_dev,
                cooperative_norms,
            )?;
            if qkv_q8dot {
                let q8_qs_dev = norm_dev + norm_bytes as u64;
                let q8_ds_dev = q8_qs_dev + q8_qs_bytes as u64;
                self.launch_quantize_q8_1_by_32(
                    norm_dev,
                    q8_qs_dev,
                    q8_ds_dev,
                    input_blocks * 256,
                )?;
                let fused_qkv_gate = if tuning::q4k_q8dot_row4_enabled()
                    && (!v_q6 || tuning::q6k_q8dot_row4_enabled())
                {
                    if let (Some(gate_weights), Some(gate_dev)) =
                        (attention_gate_weights, attention_gate_dev)
                    {
                        self.launch_qkv_gate_gemv_q8dot_to_dev(
                            q_weights,
                            k_weights,
                            v_weights,
                            gate_weights,
                            q_rows,
                            kv_dim,
                            v_q6,
                            input_blocks,
                            q8_qs_dev,
                            q8_ds_dev,
                            q_dev,
                            k_dev,
                            v_dev,
                            gate_dev,
                        )?;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !fused_qkv_gate {
                    self.launch_q4k_gemv_q8dot_to_dev(
                        q_weights,
                        q_rows,
                        input_blocks,
                        q8_qs_dev,
                        q8_ds_dev,
                        q_dev,
                    )?;
                    self.launch_q4k_gemv_q8dot_to_dev(
                        k_weights,
                        kv_dim,
                        input_blocks,
                        q8_qs_dev,
                        q8_ds_dev,
                        k_dev,
                    )?;
                    if !v_q6 {
                        self.launch_q4k_gemv_q8dot_to_dev(
                            v_weights,
                            kv_dim,
                            input_blocks,
                            q8_qs_dev,
                            q8_ds_dev,
                            v_dev,
                        )?;
                    } else {
                        self.launch_q6k_gemv_q8dot_to_dev(
                            v_weights,
                            kv_dim,
                            input_blocks,
                            q8_qs_dev,
                            q8_ds_dev,
                            v_dev,
                        )?;
                    }
                    if let (Some(weights), Some(output_dev)) =
                        (attention_gate_weights, attention_gate_dev)
                    {
                        self.launch_q4k_gemv_q8dot_to_dev(
                            weights,
                            q_rows,
                            input_blocks,
                            q8_qs_dev,
                            q8_ds_dev,
                            output_dev,
                        )?;
                    }
                }
            } else {
                self.q4k_gemv_device_to_device(q_weights, q_rows, input_blocks, norm_dev, q_dev)?;
                self.q4k_gemv_device_to_device(k_weights, kv_dim, input_blocks, norm_dev, k_dev)?;
                if !v_q6 {
                    self.q4k_gemv_device_to_device(
                        v_weights,
                        kv_dim,
                        input_blocks,
                        norm_dev,
                        v_dev,
                    )?;
                } else {
                    self.q6k_gemv_device_to_device(
                        v_weights,
                        kv_dim,
                        input_blocks,
                        norm_dev,
                        v_dev,
                    )?;
                }
                if let (Some(weights), Some(output_dev)) =
                    (attention_gate_weights, attention_gate_dev)
                {
                    self.q4k_gemv_device_to_device(
                        weights,
                        q_rows,
                        input_blocks,
                        norm_dev,
                        output_dev,
                    )?;
                }
            }

            if let Some(qn) = q_norm_weight {
                self.launch_qk_norm_device(q_dev, qn, num_heads, head_dim, norm_eps)?;
            }
            if let Some(kn) = k_norm_weight {
                self.launch_qk_norm_device(k_dev, kn, num_kv_heads, head_dim, norm_eps)?;
            }
            if apply_rope {
                if rope_neox {
                    self.launch_rope_decode(
                        q_dev,
                        k_dev,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        rope_theta,
                        rope_pos,
                    )?;
                } else {
                    self.launch_rope_f32_inplace(
                        q_dev,
                        0,
                        num_heads * head_dim / 2,
                        q_rows,
                        head_dim,
                        head_dim,
                        rope_pos,
                        rope_theta,
                        0,
                    )?;
                    self.launch_rope_f32_inplace(
                        k_dev,
                        0,
                        num_kv_heads * head_dim / 2,
                        kv_dim,
                        head_dim,
                        head_dim,
                        rope_pos,
                        rope_theta,
                        0,
                    )?;
                }
            }

            self.launch_kv_f16_write(layer_idx, k_dev, v_dev, kv_dim, kv_len)?;
            self.launch_attention_decode_device_with_policy(
                layer_idx,
                q_dev,
                attn_out_dev,
                num_heads,
                num_kv_heads,
                head_dim,
                kv_len,
                attention_scale,
                sliding_window,
            )?;
            if let Some(gate_dev) = attention_gate_dev {
                self.launch_sigmoid_mul_inplace(attn_out_dev, gate_dev, q_rows)?;
            }

            let mut trace_stage = std::time::Instant::now();
            self.launch_dense_chain_graph_ops(
                o_weights,
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                post_attn_norm,
                ffn_norm,
                post_ffn_norm,
                q_rows,
                n_ff,
                n_embd,
                norm_eps,
                post_norm_eps,
                false,
                false,
                hidden_dev,
                attn_out_dev,
                norm_dev,
                down_dev,
                ffn_uses_gelu,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0,
                false,
                (out_scale != 1.0).then_some(out_scale),
                None,
                &mut trace_stage,
                cooperative_norms,
            )?;
            Ok(())
        })();
        for key in weight_leases {
            self.unpin_resident_q4k_key(key);
        }
        result
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
        cooperative: bool,
    ) -> Result<(), String> {
        let weight_dev = self.resident_f32_ptr_stable_source(weight)?;
        self.launch_rms_norm_f32(
            input_dev,
            weight_dev,
            output_dev,
            eps,
            dim,
            false,
            cooperative,
        )
    }

    fn launch_qk_norm_device(
        &mut self,
        data_dev: u64,
        norm_weight: &[f32],
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let weight_dev = self.resident_f32_ptr_stable_source(norm_weight)?;
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
    fn alloc_decode_kv_cache_buffer(&mut self, bytes: usize) -> Result<u64, String> {
        match unsafe { self.api.mem_alloc(bytes) } {
            Ok(ptr) => Ok(ptr),
            Err(err) if cuda_mem_alloc_oom(&err) => {
                self.release_muse_decode_tail_residency_for_oom()?;
                match unsafe { self.api.mem_alloc(bytes) } {
                    Ok(ptr) => Ok(ptr),
                    Err(err2) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err2) => {
                        let _ = self.offload_non_pinned_resident_q4k_releasing(bytes);
                        match unsafe { self.api.mem_alloc(bytes) } {
                            Ok(ptr) => Ok(ptr),
                            Err(err3) if cuda_mem_alloc_oom(&err3) => {
                                self.clear_resident_moe_layer_cache()?;
                                unsafe { self.api.mem_alloc(bytes) }
                            }
                            Err(err3) => Err(err3),
                        }
                    }
                    Err(err2) => Err(err2),
                }
            }
            Err(err) => Err(err),
        }
    }

    fn ensure_decode_kv_f16_capacity(
        &mut self,
        layer_idx: usize,
        kv_dim: usize,
        total_tokens: usize,
    ) -> Result<(), String> {
        let required_bytes = total_tokens
            .checked_mul(kv_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "cu63 kv_f16 capacity overflow".to_string())?;
        let mut cache = self
            .decode_attention_kv
            .remove(&layer_idx)
            .unwrap_or_default();
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
        if cache.k_bits_capacity < required_bytes || cache.v_bits_capacity < required_bytes {
            let capacity = align_up(required_bytes, 1024 * 1024);
            let copy_bytes = match cache
                .cached_tokens
                .checked_mul(kv_dim)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            {
                Some(bytes) => bytes,
                None => {
                    self.decode_attention_kv.insert(layer_idx, cache);
                    return Err("cu63 kv_f16 migration byte size overflow".to_string());
                }
            };
            let new_k = match self.alloc_decode_kv_cache_buffer(capacity) {
                Ok(ptr) => ptr,
                Err(err) => {
                    self.decode_attention_kv.insert(layer_idx, cache);
                    return Err(err);
                }
            };
            let new_v = match self.alloc_decode_kv_cache_buffer(capacity) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = unsafe { self.api.mem_free(new_k) };
                    self.decode_attention_kv.insert(layer_idx, cache);
                    return Err(err);
                }
            };
            let migrate = (|| {
                if copy_bytes != 0 {
                    let old_k = cache
                        .k_bits_dev
                        .ok_or_else(|| "cu63 kv_f16 growth missing old K buffer".to_string())?;
                    let old_v = cache
                        .v_bits_dev
                        .ok_or_else(|| "cu63 kv_f16 growth missing old V buffer".to_string())?;
                    unsafe {
                        self.api
                            .memcpy_dtod_async(new_k, old_k, copy_bytes, self.stream)?;
                        self.api
                            .memcpy_dtod_async(new_v, old_v, copy_bytes, self.stream)?;
                    }
                }
                self.stream_synchronize()
            })();
            if let Err(err) = migrate {
                let _ = unsafe { self.api.mem_free(new_k) };
                let _ = unsafe { self.api.mem_free(new_v) };
                self.decode_attention_kv.insert(layer_idx, cache);
                return Err(err);
            }
            let old_k = cache.k_bits_dev.replace(new_k);
            let old_v = cache.v_bits_dev.replace(new_v);
            cache.k_bits_capacity = capacity;
            cache.v_bits_capacity = capacity;
            let release_old = (|| {
                if let Some(ptr) = old_k {
                    unsafe { self.api.mem_free(ptr)? };
                }
                if let Some(ptr) = old_v {
                    unsafe { self.api.mem_free(ptr)? };
                }
                Ok::<(), String>(())
            })();
            if let Err(err) = release_old {
                self.decode_attention_kv.insert(layer_idx, cache);
                return Err(err);
            }
        }
        self.decode_attention_kv.insert(layer_idx, cache);
        Ok(())
    }

    /// Convert f32 K/V on device to f16 and append to the per-layer KV cache.
    /// `kv_len` is the write position (0-indexed, the slot for the new token).
    /// After this call the cache holds `kv_len + 1` tokens.
    pub(super) fn launch_kv_f16_write(
        &mut self,
        layer_idx: usize,
        k_dev: u64,
        v_dev: u64,
        kv_dim: usize,
        kv_len: usize,
    ) -> Result<(), String> {
        let total_tokens = kv_len + 1;
        self.ensure_decode_kv_f16_capacity(layer_idx, kv_dim, total_tokens)?;
        let mut cache = self
            .decode_attention_kv
            .remove(&layer_idx)
            .expect("decode KV capacity initialized");

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
        self.launch_attention_decode_device_with_policy(
            layer_idx,
            q_dev,
            output_dev,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            1.0,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_attention_decode_device_with_policy(
        &mut self,
        layer_idx: usize,
        q_dev: u64,
        output_dev: u64,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        scale: f32,
        sliding_window: Option<usize>,
    ) -> Result<(), String> {
        if !matches!(head_dim, 128 | 256 | 512) {
            return Err(format!(
                "device attention decode: unsupported head_dim={head_dim}"
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("device attention decode: invalid scale={scale}"));
        }
        if sliding_window.is_some() && head_dim != 128 {
            return Err(format!(
                "device attention decode: sliding window requires head_dim=128, got {head_dim}"
            ));
        }
        let attn_kv_len = kv_len + 1;
        let window_len = sliding_window
            .map(|window| window.min(attn_kv_len))
            .unwrap_or(attn_kv_len);
        if window_len == 0 {
            return Err("device attention decode: sliding window must be non-zero".to_string());
        }
        let window_start = attn_kv_len - window_len;
        let cache = self
            .decode_attention_kv
            .get(&layer_idx)
            .ok_or_else(|| format!("device attention decode: no KV cache for layer {layer_idx}"))?;
        let k_cache_dev = cache
            .k_bits_dev
            .ok_or_else(|| "device attention decode: missing K buffer".to_string())?;
        let v_cache_dev = cache
            .v_bits_dev
            .ok_or_else(|| "device attention decode: missing V buffer".to_string())?;
        let kv_rows = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "device attention decode: KV row width overflow".to_string())?;
        let window_offset_bytes = window_start
            .checked_mul(kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "device attention decode: window offset overflow".to_string())?;
        let k_window_dev = k_cache_dev + window_offset_bytes as u64;
        let v_window_dev = v_cache_dev + window_offset_bytes as u64;

        if Self::cached_decode_split_preferred(head_dim, window_len) {
            return self.launch_attention_decode_split_device(
                output_dev,
                q_dev,
                k_window_dev,
                v_window_dev,
                head_dim,
                window_len,
                num_heads,
                num_kv_heads,
                scale,
            );
        }

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_window_dev;
        let mut v_arg = v_window_dev;
        let mut kv_len_arg = window_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        let (kernel, block) = match head_dim {
            128 if crate::tuning::attention_decode_hd128_warp_enabled() => {
                ("rnb_attention_decode_hd128_warp", (32, 1, 1))
            }
            128 => ("rnb_attention_decode_hd128", (128, 1, 1)),
            256 => ("rnb_attention_decode_hd256", (256, 1, 1)),
            512 => ("rnb_attention_decode_hd512", (512, 1, 1)),
            _ => unreachable!("validated head_dim"),
        };
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
            block,
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
        if num_tokens == 0 || kv_dim == 0 {
            return Err(format!(
                "cu63 populate_kv requires non-zero tokens and kv_dim: tokens={num_tokens} kv_dim={kv_dim}"
            ));
        }
        let required_values = num_tokens
            .checked_mul(kv_dim)
            .ok_or_else(|| "cu63 populate_kv: value count overflow".to_string())?;
        let required_bytes = required_values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "cu63 populate_kv: byte count overflow".to_string())?;
        if k_bits.len() < required_values || v_bits.len() < required_values {
            return Err(format!(
                "cu63 populate_kv: bits too short — k={} v={} need={}",
                k_bits.len(),
                v_bits.len(),
                required_values,
            ));
        }

        self.ensure_decode_kv_f16_capacity(layer_idx, kv_dim, num_tokens)?;
        let mut cache = self
            .decode_attention_kv
            .remove(&layer_idx)
            .expect("decode KV capacity initialized");

        let k_dev = cache.k_bits_dev.unwrap();
        let v_dev = cache.v_bits_dev.unwrap();
        let upload = unsafe {
            self.api
                .memcpy_htod_async(
                    k_dev,
                    k_bits.as_ptr().cast::<libc::c_void>(),
                    required_bytes,
                    self.stream,
                )
                .and_then(|()| {
                    self.api.memcpy_htod_async(
                        v_dev,
                        v_bits.as_ptr().cast::<libc::c_void>(),
                        required_bytes,
                        self.stream,
                    )
                })
        };
        if let Err(err) = upload {
            self.decode_attention_kv.insert(layer_idx, cache);
            return Err(err);
        }
        cache.cached_tokens = num_tokens;
        self.decode_attention_kv.insert(layer_idx, cache);
        Ok(())
    }

    pub(super) fn device_kv_cache_f16_matches(
        &self,
        layer_idx: usize,
        kv_dim: usize,
        num_tokens: usize,
    ) -> bool {
        let required_bytes = num_tokens
            .saturating_mul(kv_dim)
            .saturating_mul(std::mem::size_of::<u16>());
        self.decode_attention_kv
            .get(&layer_idx)
            .is_some_and(|cache| {
                cache.kv_rows == kv_dim
                    && cache.cached_tokens == num_tokens
                    && cache.k_bits_dev.is_some()
                    && cache.v_bits_dev.is_some()
                    && cache.k_bits_capacity >= required_bytes
                    && cache.v_bits_capacity >= required_bytes
            })
    }

    pub(super) fn sync_device_kv_cache_f16_to_host(
        &mut self,
        layer_idx: usize,
        k_bits: &mut [u16],
        v_bits: &mut [u16],
        kv_dim: usize,
        num_tokens: usize,
    ) -> Result<bool, String> {
        self.set_current()?;
        let required_elements = num_tokens
            .checked_mul(kv_dim)
            .ok_or_else(|| "CUDA sequence KV materialization size overflow".to_string())?;
        if k_bits.len() != required_elements || v_bits.len() != required_elements {
            return Err(format!(
                "CUDA sequence KV materialization layout mismatch: k={} v={} expected={required_elements}",
                k_bits.len(),
                v_bits.len()
            ));
        }
        let Some(cache) = self.decode_attention_kv.get(&layer_idx) else {
            return Ok(false);
        };
        if cache.kv_rows != kv_dim || cache.cached_tokens < num_tokens {
            return Ok(false);
        }
        let Some(k_dev) = cache.k_bits_dev else {
            return Ok(false);
        };
        let Some(v_dev) = cache.v_bits_dev else {
            return Ok(false);
        };
        let required_bytes = required_elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "CUDA sequence KV materialization byte size overflow".to_string())?;
        if cache.k_bits_capacity < required_bytes || cache.v_bits_capacity < required_bytes {
            return Err("CUDA sequence KV materialization exceeds resident capacity".to_string());
        }
        unsafe {
            self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                k_dev,
                required_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                v_dev,
                required_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(true)
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
