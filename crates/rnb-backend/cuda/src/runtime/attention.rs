use super::gemv::{Q4kF16DenseChainOutput, Q4kF16QkvInput};
use super::*;

impl CudaState {
    pub(super) fn clear_decode_attention_kv_cache(&mut self) -> Result<(), String> {
        self.set_current()?;
        self.stream_synchronize()?;
        for (_, cache) in self.decode_attention_kv.drain() {
            if let Some(ptr) = cache.k_bits_dev {
                unsafe { self.api.mem_free(ptr)? };
            }
            if let Some(ptr) = cache.v_bits_dev {
                unsafe { self.api.mem_free(ptr)? };
            }
        }
        for (_, cache) in self.decode_attention_kvarn.drain() {
            if let Some(ptr) = cache.records_dev {
                unsafe { self.api.mem_free(ptr)? };
            }
            if let Some(ptr) = cache.f16_dev {
                unsafe { self.api.mem_free(ptr)? };
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_attention_prefill_flash_hd512_kernel(
        &mut self,
        output_dev: u64,
        q_dev: u64,
        k_dev: u64,
        v_dev: u64,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        window: Option<usize>,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut seq_arg = seq_len as u32;
        let mut kv_len_arg = kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        if let Some(window) = window {
            let mut window_arg = window as u32;
            return self.launch_cached_gemv(
                "rnb_attention_prefill_flash_hd512_window_w256",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (seq_len as u32, num_heads as u32, 1),
                (256, 1, 1),
            );
        }

        let use_w256 = tuning::prefill_flash_attention_hd512_w256_enabled();
        // cu38 Phase 4 + cu39 Phase 7: j-batch=4 kernel (4 K position 동시 처리).
        // cu39 Phase 7 default ON: 5 모델 ABAB 검증 (Gemma -2.4% 단독 / -9% v3 결합,
        // 다른 모델 회귀 없음). head_dim 256+ 만 영향. env="0" 으로 disable.
        let use_jbatch4 = use_w256
            && std::env::var("RNB_CUDA_ATTN_FLASH_HD512_W256_JBATCH4")
                .ok()
                .as_deref()
                != Some("0");
        let kernel_name = if use_jbatch4 {
            "rnb_attention_prefill_flash_hd512_w256_jbatch4"
        } else if use_w256 {
            "rnb_attention_prefill_flash_hd512_w256"
        } else {
            "rnb_attention_prefill_flash_hd512"
        };
        let block_threads = if use_w256 { 256 } else { 512 };
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (seq_len as u32, num_heads as u32, 1),
            (block_threads, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_flash_hd512(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let output_len = seq_len * num_heads * 512;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }

        self.launch_attention_prefill_flash_hd512_kernel(
            output_dev,
            q_dev,
            k_dev,
            v_dev,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            None,
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
    pub(super) fn attention_prefill_flash_hd512_f16kv(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let k_f32_bytes = k
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("CUDA attention f16kv k f32 byte overflow: len={}", k.len()))?;
        let v_f32_bytes = v
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("CUDA attention f16kv v f32 byte overflow: len={}", v.len()))?;
        let output_len = seq_len * num_heads * 512;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let k_f32_dev = self.compute_aux_output_ptr(k_f32_bytes)?;
        let v_f32_dev = self.compute_weights_ptr(v_f32_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }
        self.launch_f16_to_f32(k_dev, k_f32_dev, k.len())?;
        self.launch_f16_to_f32(v_dev, v_f32_dev, v.len())?;

        self.launch_attention_prefill_flash_hd512_kernel(
            output_dev,
            q_dev,
            k_f32_dev,
            v_f32_dev,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            None,
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
    pub(super) fn attention_prefill_flash_hd512_f16kv_window(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        window: usize,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let k_f32_bytes = k
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention f16kv window k f32 byte overflow: len={}",
                    k.len()
                )
            })?;
        let v_f32_bytes = v
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention f16kv window v f32 byte overflow: len={}",
                    v.len()
                )
            })?;
        let output_len = seq_len * num_heads * 512;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let k_f32_dev = self.compute_aux_output_ptr(k_f32_bytes)?;
        let v_f32_dev = self.compute_weights_ptr(v_f32_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }
        self.launch_f16_to_f32(k_dev, k_f32_dev, k.len())?;
        self.launch_f16_to_f32(v_dev, v_f32_dev, v.len())?;
        self.launch_attention_prefill_flash_hd512_kernel(
            output_dev,
            q_dev,
            k_f32_dev,
            v_f32_dev,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            Some(window),
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
    pub(super) fn attention_prefill_flash_hd256_f16kv_window(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        window: usize,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let k_f32_bytes = k
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention hd256 f16kv window k f32 byte overflow: len={}",
                    k.len()
                )
            })?;
        let v_f32_bytes = v
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention hd256 f16kv window v f32 byte overflow: len={}",
                    v.len()
                )
            })?;
        let output_len = seq_len * num_heads * 256;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let k_f32_dev = self.compute_aux_output_ptr(k_f32_bytes)?;
        let v_f32_dev = self.compute_weights_ptr(v_f32_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }
        // cu108: hd256 SWA mma flash (opt-in). f16 K/V 직접 (launch_f16_to_f32 skip), q f32.
        // cu109: reuse_q/standalone site 격리용 env (REUSE_Q=0 이면 이 경로만 jbatch4 유지).
        let use_mma_flash = std::env::var("RNB_CUDA_PREFILL_MMA_FLASH").ok().as_deref()
            == Some("1")
            && std::env::var("RNB_CUDA_PREFILL_MMA_REUSE_Q")
                .ok()
                .as_deref()
                != Some("0");
        if use_mma_flash {
            let mut output_arg = output_dev;
            let mut q_arg = q_dev;
            let mut k_arg = k_dev; // f16 직접
            let mut v_arg = v_dev; // f16 직접
            let mut seq_arg = seq_len as u32;
            let mut kv_len_arg = kv_len as u32;
            let mut heads_arg = num_heads as u32;
            let mut kv_heads_arg = num_kv_heads as u32;
            let mut scale_arg = scale;
            let mut window_arg = window as u32;
            self.launch_cached_gemv(
                "rnb_attention_prefill_flash_hd256_window_mma",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                ((seq_len as u32 + 63) / 64, num_heads as u32, 1),
                (128, 1, 1),
            )?;
        } else {
            self.launch_f16_to_f32(k_dev, k_f32_dev, k.len())?;
            self.launch_f16_to_f32(v_dev, v_f32_dev, v.len())?;

            let mut output_arg = output_dev;
            let mut q_arg = q_dev;
            let mut k_arg = k_f32_dev;
            let mut v_arg = v_f32_dev;
            let mut seq_arg = seq_len as u32;
            let mut kv_len_arg = kv_len as u32;
            let mut heads_arg = num_heads as u32;
            let mut kv_heads_arg = num_kv_heads as u32;
            let mut scale_arg = scale;
            let mut window_arg = window as u32;
            // cu38 Phase 4 + cu39 Phase 7 default ON: j-batch=4 kernel (4 K position
            // 동시 처리, syncthreads 절감). 5 모델 ABAB 검증, 회귀 없음. env="0" 으로 disable.
            let kernel_hd256_window = if std::env::var("RNB_CUDA_ATTN_FLASH_HD256_JBATCH4")
                .ok()
                .as_deref()
                != Some("0")
            {
                "rnb_attention_prefill_flash_hd256_window_jbatch4"
            } else {
                "rnb_attention_prefill_flash_hd256_window"
            };
            self.launch_cached_gemv(
                kernel_hd256_window,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (seq_len as u32, num_heads as u32, 1),
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
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_flash_hd512_f16kv_dense_chain(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<(), String> {
        self.attention_prefill_flash_hd512_f16kv_dense_chain_impl(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            None,
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_flash_hd512_f16kv_window_dense_chain(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        window: usize,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<(), String> {
        self.attention_prefill_flash_hd512_f16kv_dense_chain_impl(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            Some(window),
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_flash_hd256_f16kv_window_dense_chain(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        window: usize,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<(), String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let k_f32_bytes = k
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention hd256 window chain k f32 byte overflow: len={}",
                    k.len()
                )
            })?;
        let v_f32_bytes = v
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention hd256 window chain v f32 byte overflow: len={}",
                    v.len()
                )
            })?;
        let output_len = seq_len * num_heads * 256;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let k_f32_dev = self.compute_aux_output_ptr(k_f32_bytes)?;
        let v_f32_dev = self.compute_weights_ptr(v_f32_bytes)?;
        let output_dev = self.compute_full_down_ptr(output_bytes)?;
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
        }
        // cu108: hd256 SWA mma flash (opt-in). f16 K/V 직접 (launch_f16_to_f32 skip), q f32.
        // cu109: reuse_q/standalone site 격리용 env (REUSE_Q=0 이면 이 경로만 jbatch4 유지).
        let use_mma_flash = std::env::var("RNB_CUDA_PREFILL_MMA_FLASH").ok().as_deref()
            == Some("1")
            && std::env::var("RNB_CUDA_PREFILL_MMA_REUSE_Q")
                .ok()
                .as_deref()
                != Some("0");
        if use_mma_flash {
            let mut output_arg = output_dev;
            let mut q_arg = q_dev;
            let mut k_arg = k_dev; // f16 직접
            let mut v_arg = v_dev; // f16 직접
            let mut seq_arg = seq_len as u32;
            let mut kv_len_arg = kv_len as u32;
            let mut heads_arg = num_heads as u32;
            let mut kv_heads_arg = num_kv_heads as u32;
            let mut scale_arg = scale;
            let mut window_arg = window as u32;
            self.launch_cached_gemv(
                "rnb_attention_prefill_flash_hd256_window_mma",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                ((seq_len as u32 + 63) / 64, num_heads as u32, 1),
                (128, 1, 1),
            )?;
        } else {
            self.launch_f16_to_f32(k_dev, k_f32_dev, k.len())?;
            self.launch_f16_to_f32(v_dev, v_f32_dev, v.len())?;

            let mut output_arg = output_dev;
            let mut q_arg = q_dev;
            let mut k_arg = k_f32_dev;
            let mut v_arg = v_f32_dev;
            let mut seq_arg = seq_len as u32;
            let mut kv_len_arg = kv_len as u32;
            let mut heads_arg = num_heads as u32;
            let mut kv_heads_arg = num_kv_heads as u32;
            let mut scale_arg = scale;
            let mut window_arg = window as u32;
            // cu38 Phase 4 + cu39 Phase 7 default ON: j-batch=4 kernel (4 K position
            // 동시 처리, syncthreads 절감). 5 모델 ABAB 검증, 회귀 없음. env="0" 으로 disable.
            let kernel_hd256_window = if std::env::var("RNB_CUDA_ATTN_FLASH_HD256_JBATCH4")
                .ok()
                .as_deref()
                != Some("0")
            {
                "rnb_attention_prefill_flash_hd256_window_jbatch4"
            } else {
                "rnb_attention_prefill_flash_hd256_window"
            };
            self.launch_cached_gemv(
                kernel_hd256_window,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (seq_len as u32, num_heads as u32, 1),
                (256, 1, 1),
            )?;
        }

        self.dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            None,
            None,
            None,
            None,
            0,
            o_cols,
            n_ff,
            n_embd,
            seq_len,
            hidden,
            None,
            output_dev,
            None,
            None,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn q4k_f16_qkv_postprocess_hd256_window_dense_chain(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        v_quant: u32,
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        hidden_input: &[f32],
        hidden_input_device: Option<(
            rnb_backend_api::DeviceTensorId,
            rnb_backend_api::DeviceTensorDesc,
        )>,
        attn_norm_weight: &[f32],
        q_norm: &[f32],
        k_norm: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
        window: usize,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        ple_gate_weights: Option<&[u8]>,
        ple_proj_weights: Option<&[u8]>,
        ple_post_norm_weight: Option<&[f32]>,
        ple_input: Option<&[f32]>,
        ple_dim: usize,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        layer_out_scale: Option<&[f32]>,
        device_output_desc: Option<rnb_backend_api::DeviceTensorDesc>,
        unit_offset_attn_norm: bool,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<Option<(Vec<u16>, Vec<u16>, Q4kF16DenseChainOutput)>, String> {
        if window == 0 || q_rows != o_cols || q_rows != num_heads.saturating_mul(256) {
            return Ok(None);
        }
        let expected_hidden = seq_len.saturating_mul(n_embd);
        if kv_rows != num_kv_heads.saturating_mul(256) || hidden.len() != expected_hidden {
            return Ok(None);
        }
        if let Some((_, desc)) = hidden_input_device.as_ref() {
            if desc.rows() != seq_len
                || desc.cols() != n_embd
                || desc.dtype() != rnb_backend_api::ScalarType::F32
            {
                return Ok(None);
            }
        } else if hidden_input.len() != expected_hidden {
            return Ok(None);
        }
        let qkv_input = match hidden_input_device {
            Some((id, desc)) => Q4kF16QkvInput::DeviceHidden {
                id,
                desc,
                attn_norm: attn_norm_weight,
                unit_offset: unit_offset_attn_norm,
            },
            None => Q4kF16QkvInput::Hidden {
                values: hidden_input,
                attn_norm: attn_norm_weight,
                unit_offset: unit_offset_attn_norm,
            },
        };
        let Some(device) = self.q4k_f16_qkv_postprocess_hd256_to_device(
            q_weights,
            k_weights,
            v_weights,
            v_quant,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            qkv_input,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )?
        else {
            return Ok(None);
        };

        // cu108: hd256 SWA mma tensor core flash (opt-in). f16 K/V 직접 (launch_f16_to_f32 skip,
        // f32 왕복 제거). arch<sm_80 빌드 + 이 env 조합은 미지원 (커널이 빈 스텁 → fallback 권장).
        let use_mma_flash =
            std::env::var("RNB_CUDA_PREFILL_MMA_FLASH").ok().as_deref() == Some("1");
        if use_mma_flash {
            let mut output_arg = device.attn_out_scratch_dev;
            let mut q_arg = device.q_post_dev;
            let mut k_arg = device.k_bits_dev; // f16 직접
            let mut v_arg = device.v_bits_dev; // f16 직접
            let mut seq_arg = seq_len as u32;
            let mut kv_len_arg = seq_len as u32;
            let mut heads_arg = num_heads as u32;
            let mut kv_heads_arg = num_kv_heads as u32;
            let mut scale_arg = scale;
            let mut window_arg = window as u32;
            self.launch_cached_gemv(
                "rnb_attention_prefill_flash_hd256_window_mma",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                ((seq_len as u32 + 63) / 64, num_heads as u32, 1),
                (128, 1, 1),
            )?;
        } else {
            self.launch_f16_to_f32(device.k_bits_dev, device.k_f32_scratch_dev, device.kv_len)?;
            self.launch_f16_to_f32(device.v_bits_dev, device.v_f32_scratch_dev, device.kv_len)?;

            let mut output_arg = device.attn_out_scratch_dev;
            let mut q_arg = device.q_post_dev;
            let mut k_arg = device.k_f32_scratch_dev;
            let mut v_arg = device.v_f32_scratch_dev;
            let mut seq_arg = seq_len as u32;
            let mut kv_len_arg = seq_len as u32;
            let mut heads_arg = num_heads as u32;
            let mut kv_heads_arg = num_kv_heads as u32;
            let mut scale_arg = scale;
            let mut window_arg = window as u32;
            // cu38 Phase 4 + cu39 Phase 7 default ON: j-batch=4 kernel (4 K position
            // 동시 처리, syncthreads 절감). 5 모델 ABAB 검증, 회귀 없음. env="0" 으로 disable.
            let kernel_hd256_window = if std::env::var("RNB_CUDA_ATTN_FLASH_HD256_JBATCH4")
                .ok()
                .as_deref()
                != Some("0")
            {
                "rnb_attention_prefill_flash_hd256_window_jbatch4"
            } else {
                "rnb_attention_prefill_flash_hd256_window"
            };
            self.launch_cached_gemv(
                kernel_hd256_window,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (seq_len as u32, num_heads as u32, 1),
                (256, 1, 1),
            )?;
        }

        let mut k_bits = vec![0u16; device.kv_len];
        let mut v_bits = vec![0u16; device.kv_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.k_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.v_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
        }
        // The dense tail reuses and may grow the same scratch slabs that hold
        // k_bits_dev/v_bits_dev. Close the async D2H lifetime before those
        // slabs can be reallocated or overwritten.
        self.stream_synchronize()?;

        let output_id = self
            .dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
                o_weights,
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                post_attn_norm_weight,
                ffn_norm_weight,
                post_ffn_norm_weight,
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
                ple_dim,
                o_cols,
                n_ff,
                n_embd,
                seq_len,
                hidden,
                device.hidden_dev,
                device.attn_out_scratch_dev,
                device_output_desc,
                layer_out_scale,
                norm_eps,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )?;
        Ok(Some((
            k_bits,
            v_bits,
            match output_id {
                Some(id) => Q4kF16DenseChainOutput::Device(id),
                None => Q4kF16DenseChainOutput::Host,
            },
        )))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_flash_hd512_f16kv_dense_chain_impl(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        window: Option<usize>,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<(), String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let k_f32_bytes = k
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention chain f16kv k f32 byte overflow: len={}",
                    k.len()
                )
            })?;
        let v_f32_bytes = v
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "CUDA attention chain f16kv v f32 byte overflow: len={}",
                    v.len()
                )
            })?;
        let output_len = seq_len * num_heads * 512;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let k_f32_dev = self.compute_aux_output_ptr(k_f32_bytes)?;
        let v_f32_dev = self.compute_weights_ptr(v_f32_bytes)?;
        let output_dev = self.compute_full_down_ptr(output_bytes)?;
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
        }
        // cu113: hd512 FULL mma flash (opt-in). f16 K/V 직접(launch_f16_to_f32 skip), q f32,
        // window 없음(causal only). prefill 24배 격차 주범인 FULL hd512 attention 을 tensor
        // core 로. window Some(현재 hd512 경로는 FULL=None) 이면 scalar fallback.
        let use_mma_flash =
            std::env::var("RNB_CUDA_PREFILL_MMA_FLASH").ok().as_deref() == Some("1");
        if use_mma_flash && window.is_none() {
            let mut output_arg = output_dev;
            let mut q_arg = q_dev;
            let mut k_arg = k_dev; // f16 직접
            let mut v_arg = v_dev; // f16 직접
            let mut seq_arg = seq_len as u32;
            let mut kv_len_arg = kv_len as u32;
            let mut heads_arg = num_heads as u32;
            let mut kv_heads_arg = num_kv_heads as u32;
            let mut scale_arg = scale;
            self.launch_cached_gemv(
                "rnb_attention_prefill_flash_hd512_mma",
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                ],
                ((seq_len as u32 + 63) / 64, num_heads as u32, 1),
                (128, 1, 1),
            )?;
        } else {
            self.launch_f16_to_f32(k_dev, k_f32_dev, k.len())?;
            self.launch_f16_to_f32(v_dev, v_f32_dev, v.len())?;

            self.launch_attention_prefill_flash_hd512_kernel(
                output_dev,
                q_dev,
                k_f32_dev,
                v_f32_dev,
                seq_len,
                kv_len,
                num_heads,
                num_kv_heads,
                scale,
                window,
            )?;
        }

        self.dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            None,
            None,
            None,
            None,
            0,
            o_cols,
            n_ff,
            n_embd,
            seq_len,
            hidden,
            None,
            output_dev,
            None,
            None,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_flash_hd256(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        sliding_window: Option<usize>,
        softcap: Option<f32>,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let output_len = seq_len * num_heads * head_dim;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut seq_arg = seq_len as u32;
        let mut kv_len_arg = kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut head_dim_arg = head_dim as u32;
        let mut scale_arg = scale;
        let mut window_arg = sliding_window.unwrap_or(0) as u32;
        let mut softcap_arg = softcap.unwrap_or(0.0);
        self.launch_cached_gemv(
            "rnb_attention_prefill_flash_hd256",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                (&mut window_arg as *mut u32).cast::<libc::c_void>(),
                (&mut softcap_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (seq_len as u32, num_heads as u32, 1),
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
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_flash_hd128(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let output_len = seq_len * num_heads * 128;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut seq_arg = seq_len as u32;
        let mut kv_len_arg = kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        self.launch_cached_gemv(
            "rnb_attention_prefill_flash_hd128",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (seq_len as u32, num_heads as u32, 1),
            (128, 1, 1),
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

    pub(super) fn attention_decode_hd256(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let output_len = num_heads * 256;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut kv_len_arg = kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        self.launch_cached_gemv(
            "rnb_attention_decode_hd256",
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
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn attention_decode_hd512(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let output_len = num_heads * 512;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut kv_len_arg = kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        self.launch_cached_gemv(
            "rnb_attention_decode_hd512",
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
            (512, 1, 1),
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

    pub(super) fn attention_decode_hd512_len_device(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let output_len = num_heads * 512;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let kv_len_dev = self.cu68_graph_kv_len_ptr()?;
        let kv_len_value = kv_len as u32;
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
                kv_len_dev,
                (&kv_len_value as *const u32).cast::<libc::c_void>(),
                std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
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

    pub(super) fn attention_decode_hd128(
        &mut self,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let q_bytes = std::mem::size_of_val(q);
        let k_bytes = std::mem::size_of_val(k);
        let v_bytes = std::mem::size_of_val(v);
        let output_len = num_heads * 128;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let q_dev = self.compute_input_ptr(q_bytes)?;
        let k_dev = self.compute_mid_a_ptr(k_bytes)?;
        let v_dev = self.compute_mid_b_ptr(v_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
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
        }

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut kv_len_arg = kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        self.launch_cached_gemv(
            "rnb_attention_decode_hd128",
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
            (128, 1, 1),
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
    pub(super) fn attention_decode_cached(
        &mut self,
        layer_index: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        self.attention_decode_cached_range(
            layer_index,
            q,
            k,
            v,
            kv_len,
            0,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .map(|opt| opt.expect("host return path"))
    }

    // cu47 step 32: attention_decode_cached 의 device output variant. output_dev_target
    // (caller-provided device buffer) 에 결과 write, D2H + sync 안 함.
    // chain function 의 attn_out H2D 제거 위해 caller 가 carrier 에 직접 받음.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_decode_cached_to_device(
        &mut self,
        layer_index: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        output_dev_target: u64,
        last_token_k_dev: Option<u64>,
        last_token_v_dev: Option<u64>,
        q_dev_override: Option<u64>,
    ) -> Result<(), String> {
        let _ = self.attention_decode_cached_range(
            layer_index,
            q,
            k,
            v,
            kv_len,
            0,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            Some(output_dev_target),
            last_token_k_dev,
            last_token_v_dev,
            q_dev_override,
            false,
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_decode_cached_to_device_len_device(
        &mut self,
        layer_index: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        output_dev_target: u64,
        last_token_k_dev: Option<u64>,
        last_token_v_dev: Option<u64>,
        q_dev_override: Option<u64>,
    ) -> Result<(), String> {
        let _ = self.attention_decode_cached_range(
            layer_index,
            q,
            k,
            v,
            kv_len,
            0,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            Some(output_dev_target),
            last_token_k_dev,
            last_token_v_dev,
            q_dev_override,
            true,
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_decode_cached_to_device_len_device_graph(
        &mut self,
        layer_index: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        output_dev_target: u64,
        last_token_k_dev: Option<u64>,
        last_token_v_dev: Option<u64>,
        q_dev_override: Option<u64>,
    ) -> Result<(), String> {
        let _ = self.attention_decode_cached_range(
            layer_index,
            q,
            k,
            v,
            kv_len,
            0,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            Some(output_dev_target),
            last_token_k_dev,
            last_token_v_dev,
            q_dev_override,
            true,
            true,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_decode_cached_window(
        &mut self,
        layer_index: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        window_start: usize,
        window_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        self.attention_decode_cached_range(
            layer_index,
            q,
            k,
            v,
            kv_len,
            window_start,
            window_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .map(|opt| opt.expect("host return path"))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_attention_decode_hd512_len_device_kernel(
        &mut self,
        output_dev: u64,
        q_dev: u64,
        k_dev: u64,
        v_dev: u64,
        kv_len_dev: u64,
        num_heads: u32,
        num_kv_heads: u32,
        scale: f32,
    ) -> Result<(), String> {
        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let mut k_arg = k_dev;
        let mut v_arg = v_dev;
        let mut kv_len_dev_arg = kv_len_dev;
        let mut heads_arg = num_heads;
        let mut kv_heads_arg = num_kv_heads;
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
            (num_heads, 1, 1),
            (512, 1, 1),
        )
    }

    fn cached_decode_split_preferred(head_dim: usize, window_len: usize) -> bool {
        match head_dim {
            256 => window_len >= 512 && crate::tuning::decode_attention_hd256_split_enabled(),
            512 => window_len >= 256 && crate::tuning::decode_attention_hd512_split_enabled(),
            _ => false,
        }
    }

    fn attention_kvarn_to_device_impl(
        &mut self,
        request: rnb_backend_api::KvarnDecodeRequest<'_>,
        output_dev: u64,
        query_dev_override: Option<u64>,
        query_rows: usize,
        current_k_bits_dev: u64,
        current_v_bits_dev: u64,
        current_tokens: usize,
    ) -> Result<(), String> {
        if query_dev_override.is_some() {
            request.validate_device_query(query_rows)?;
            if current_tokens != query_rows {
                return Err(format!(
                    "CUDA KVarN device query/current row mismatch: query={query_rows} current={current_tokens}"
                ));
            }
            if current_tokens > 0 && (current_k_bits_dev == 0 || current_v_bits_dev == 0) {
                return Err(
                    "CUDA KVarN device query requires non-null current K/V pointers".to_string(),
                );
            }
        } else {
            request.validate()?;
        }
        self.set_current()?;

        let head_dim = request.head_dim();
        if !matches!(head_dim, 128 | 256 | 512) {
            return Err(format!(
                "CUDA KVarN decode attention unsupported head_dim={head_dim}"
            ));
        }
        let num_heads = u32::try_from(request.num_heads())
            .map_err(|_| "CUDA KVarN num_heads exceeds u32".to_string())?;
        let num_kv_heads = u32::try_from(request.num_kv_heads())
            .map_err(|_| "CUDA KVarN num_kv_heads exceeds u32".to_string())?;
        let kv_len = u32::try_from(request.kv_len())
            .map_err(|_| "CUDA KVarN kv_len exceeds u32".to_string())?;
        let tail_start = u32::try_from(request.tail_start())
            .map_err(|_| "CUDA KVarN tail_start exceeds u32".to_string())?;
        let group = u32::try_from(request.group())
            .map_err(|_| "CUDA KVarN group exceeds u32".to_string())?;
        let block_bytes = u32::try_from(request.block_bytes())
            .map_err(|_| "CUDA KVarN block size exceeds u32".to_string())?;
        let row_width = request
            .num_kv_heads()
            .checked_mul(head_dim)
            .ok_or_else(|| "CUDA KVarN row width overflow".to_string())?;
        let sink_len = request.sink_key().len() / row_width;
        let tail_len = request.tail_key().len() / row_width;
        let num_blocks = request.packed_blocks().len() / request.block_bytes();
        let sink_len_u32 = u32::try_from(sink_len)
            .map_err(|_| "CUDA KVarN sink length exceeds u32".to_string())?;
        let tail_len_u32 = u32::try_from(tail_len)
            .map_err(|_| "CUDA KVarN tail length exceeds u32".to_string())?;
        let num_blocks_u32 = u32::try_from(num_blocks)
            .map_err(|_| "CUDA KVarN block count exceeds u32".to_string())?;
        let channels = row_width;
        let token_rows = request
            .num_kv_heads()
            .checked_mul(request.group())
            .ok_or_else(|| "CUDA KVarN token-row count overflow".to_string())?;
        let key_packed_bytes = channels
            .checked_mul(request.group())
            .map(|bytes| bytes / 2)
            .ok_or_else(|| "CUDA KVarN packed-key size overflow".to_string())?;
        let value_pack = 8usize / request.value_bits() as usize;
        let value_packed_bytes = token_rows
            .checked_mul(head_dim)
            .map(|bytes| bytes / value_pack)
            .ok_or_else(|| "CUDA KVarN packed-value size overflow".to_string())?;
        let channel_scale_bytes = channels
            .checked_mul(2)
            .ok_or_else(|| "CUDA KVarN channel-scale size overflow".to_string())?;
        let token_scale_bytes = token_rows
            .checked_mul(2)
            .ok_or_else(|| "CUDA KVarN token-scale size overflow".to_string())?;
        let key_packed_offset = 0usize;
        let key_scale_offset = key_packed_offset
            .checked_add(key_packed_bytes)
            .ok_or_else(|| "CUDA KVarN key-scale offset overflow".to_string())?;
        let key_zero_offset = key_scale_offset
            .checked_add(channel_scale_bytes)
            .ok_or_else(|| "CUDA KVarN key-zero offset overflow".to_string())?;
        let key_token_scale_offset = key_zero_offset
            .checked_add(channel_scale_bytes)
            .ok_or_else(|| "CUDA KVarN key-token-scale offset overflow".to_string())?;
        let value_packed_offset = key_token_scale_offset
            .checked_add(token_scale_bytes)
            .ok_or_else(|| "CUDA KVarN packed-value offset overflow".to_string())?;
        let value_channel_scale_offset = value_packed_offset
            .checked_add(value_packed_bytes)
            .ok_or_else(|| "CUDA KVarN value-channel-scale offset overflow".to_string())?;
        let value_token_scale_offset = value_channel_scale_offset
            .checked_add(channel_scale_bytes)
            .ok_or_else(|| "CUDA KVarN value-token-scale offset overflow".to_string())?;
        let value_zero_offset = value_token_scale_offset
            .checked_add(token_scale_bytes)
            .ok_or_else(|| "CUDA KVarN value-zero offset overflow".to_string())?;
        let expected_block_bytes = value_zero_offset
            .checked_add(token_scale_bytes)
            .ok_or_else(|| "CUDA KVarN block size overflow".to_string())?;
        if request.block_bytes() != expected_block_bytes {
            return Err(format!(
                "CUDA KVarN block size mismatch: got {} expected {expected_block_bytes}",
                request.block_bytes()
            ));
        }
        let key_packed_offset = u32::try_from(key_packed_offset)
            .map_err(|_| "CUDA KVarN packed-key offset exceeds u32".to_string())?;
        let key_scale_offset = u32::try_from(key_scale_offset)
            .map_err(|_| "CUDA KVarN key-scale offset exceeds u32".to_string())?;
        let key_zero_offset = u32::try_from(key_zero_offset)
            .map_err(|_| "CUDA KVarN key-zero offset exceeds u32".to_string())?;
        let key_token_scale_offset = u32::try_from(key_token_scale_offset)
            .map_err(|_| "CUDA KVarN key-token-scale offset exceeds u32".to_string())?;
        let value_packed_offset = u32::try_from(value_packed_offset)
            .map_err(|_| "CUDA KVarN packed-value offset exceeds u32".to_string())?;
        let value_channel_scale_offset = u32::try_from(value_channel_scale_offset)
            .map_err(|_| "CUDA KVarN value-channel-scale offset exceeds u32".to_string())?;
        let value_token_scale_offset = u32::try_from(value_token_scale_offset)
            .map_err(|_| "CUDA KVarN value-token-scale offset exceeds u32".to_string())?;
        let value_zero_offset = u32::try_from(value_zero_offset)
            .map_err(|_| "CUDA KVarN value-zero offset exceeds u32".to_string())?;

        let f16_capacity_elements = request
            .sink_tokens()
            .checked_add(request.group())
            .and_then(|tokens| tokens.checked_mul(row_width))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| "CUDA KVarN F16 staging capacity overflow".to_string())?;
        let f16_bytes = f16_capacity_elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "CUDA KVarN F16 staging byte size overflow".to_string())?;
        let records_bytes = request.packed_blocks().len();
        let q_values_per_row = request
            .num_heads()
            .checked_mul(head_dim)
            .ok_or_else(|| "CUDA KVarN query row size overflow".to_string())?;
        let q_bytes = q_values_per_row
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "CUDA KVarN query byte size overflow".to_string())?;

        let layer_idx = request.layer_idx();
        let mut cache = self
            .decode_attention_kvarn
            .remove(&layer_idx)
            .unwrap_or_default();
        let result = (|| {
            if cache.kv_rows != row_width
                || cache.key_bits != request.key_bits()
                || cache.value_bits != request.value_bits()
                || cache.group != request.group()
                || cache.sink_tokens != request.sink_tokens()
                || cache.block_bytes != request.block_bytes()
            {
                if let Some(ptr) = cache.records_dev.take() {
                    unsafe { self.api.mem_free(ptr)? };
                }
                if let Some(ptr) = cache.f16_dev.take() {
                    unsafe { self.api.mem_free(ptr)? };
                }
                cache = KvarnDecodeAttentionCache {
                    kv_rows: row_width,
                    key_bits: request.key_bits(),
                    value_bits: request.value_bits(),
                    group: request.group(),
                    sink_tokens: request.sink_tokens(),
                    block_bytes: request.block_bytes(),
                    ..KvarnDecodeAttentionCache::default()
                };
            }

            let old_records_dev = cache.records_dev;
            let records_dev = if records_bytes == 0 {
                cache.records_dev.unwrap_or(0)
            } else {
                ensure_device_buffer(
                    &self.api,
                    &mut cache.records_dev,
                    &mut cache.records_capacity,
                    records_bytes,
                )?
            };
            if cache.records_dev != old_records_dev
                || cache.host_records_base != request.packed_blocks().as_ptr() as usize
                || records_bytes < cache.uploaded_record_bytes
            {
                cache.uploaded_record_bytes = 0;
            }
            if records_bytes > cache.uploaded_record_bytes {
                let offset = cache.uploaded_record_bytes;
                unsafe {
                    self.api.memcpy_htod_async(
                        records_dev + offset as u64,
                        request.packed_blocks().as_ptr().add(offset).cast(),
                        records_bytes - offset,
                        self.stream,
                    )?;
                }
                cache.uploaded_record_bytes = records_bytes;
                cache.host_records_base = request.packed_blocks().as_ptr() as usize;
            }

            let old_f16_dev = cache.f16_dev;
            let f16_dev = ensure_device_buffer(
                &self.api,
                &mut cache.f16_dev,
                &mut cache.f16_capacity,
                f16_bytes,
            )?;
            if cache.f16_dev != old_f16_dev {
                cache.uploaded_sink_keys = 0;
                cache.uploaded_sink_values = 0;
                cache.host_sink_k_base = 0;
                cache.host_sink_v_base = 0;
            }

            let sink_capacity_elements = request
                .sink_tokens()
                .checked_mul(row_width)
                .ok_or_else(|| "CUDA KVarN sink capacity overflow".to_string())?;
            let tail_capacity_elements = request
                .group()
                .checked_mul(row_width)
                .ok_or_else(|| "CUDA KVarN tail capacity overflow".to_string())?;
            let sink_v_dev = f16_dev + (sink_capacity_elements * 2) as u64;
            let tail_k_dev = sink_v_dev + (sink_capacity_elements * 2) as u64;
            let tail_v_dev = tail_k_dev + (tail_capacity_elements * 2) as u64;

            let sink_key_elements = request.sink_key().len();
            if cache.host_sink_k_base != request.sink_key().as_ptr() as usize
                || sink_key_elements < cache.uploaded_sink_keys
            {
                cache.uploaded_sink_keys = 0;
            }
            if sink_key_elements > cache.uploaded_sink_keys {
                let offset = cache.uploaded_sink_keys;
                unsafe {
                    self.api.memcpy_htod_async(
                        f16_dev + (offset * 2) as u64,
                        request.sink_key().as_ptr().add(offset).cast(),
                        (sink_key_elements - offset) * 2,
                        self.stream,
                    )?;
                }
                cache.uploaded_sink_keys = sink_key_elements;
                cache.host_sink_k_base = request.sink_key().as_ptr() as usize;
            }

            let sink_value_elements = request.sink_value().len();
            if cache.host_sink_v_base != request.sink_value().as_ptr() as usize
                || sink_value_elements < cache.uploaded_sink_values
            {
                cache.uploaded_sink_values = 0;
            }
            if sink_value_elements > cache.uploaded_sink_values {
                let offset = cache.uploaded_sink_values;
                unsafe {
                    self.api.memcpy_htod_async(
                        sink_v_dev + (offset * 2) as u64,
                        request.sink_value().as_ptr().add(offset).cast(),
                        (sink_value_elements - offset) * 2,
                        self.stream,
                    )?;
                }
                cache.uploaded_sink_values = sink_value_elements;
                cache.host_sink_v_base = request.sink_value().as_ptr() as usize;
            }

            if !request.tail_key().is_empty() {
                unsafe {
                    self.api.memcpy_htod_async(
                        tail_k_dev,
                        request.tail_key().as_ptr().cast(),
                        request.tail_key().len() * 2,
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        tail_v_dev,
                        request.tail_value().as_ptr().cast(),
                        request.tail_value().len() * 2,
                        self.stream,
                    )?;
                }
            }

            let q_dev = if let Some(query_dev) = query_dev_override {
                query_dev
            } else {
                let q_dev = self.compute_input_ptr(q_bytes)?;
                unsafe {
                    self.api.memcpy_htod_async(
                        q_dev,
                        request.query().as_ptr().cast(),
                        q_bytes,
                        self.stream,
                    )?;
                }
                q_dev
            };

            let query_rows_u32 = u32::try_from(query_rows)
                .map_err(|_| "CUDA KVarN query row count exceeds u32".to_string())?;
            let mut output_arg = output_dev;
            let mut query_arg = q_dev;
            let mut records_arg = records_dev;
            let mut sink_key_arg = f16_dev;
            let mut sink_value_arg = sink_v_dev;
            let mut tail_key_arg = tail_k_dev;
            let mut tail_value_arg = tail_v_dev;
            let mut current_key_arg = current_k_bits_dev;
            let mut current_value_arg = current_v_bits_dev;
            let mut kv_len_arg = kv_len;
            let mut tail_start_arg = tail_start;
            let mut sink_len_arg = sink_len_u32;
            let mut tail_len_arg = tail_len_u32;
            let mut num_blocks_arg = num_blocks_u32;
            let mut num_heads_arg = num_heads;
            let mut num_kv_heads_arg = num_kv_heads;
            let mut group_arg = group;
            let mut value_bits_arg = request.value_bits() as u32;
            let mut block_bytes_arg = block_bytes;
            let mut key_packed_offset_arg = key_packed_offset;
            let mut key_scale_offset_arg = key_scale_offset;
            let mut key_zero_offset_arg = key_zero_offset;
            let mut key_token_scale_offset_arg = key_token_scale_offset;
            let mut value_packed_offset_arg = value_packed_offset;
            let mut value_channel_scale_offset_arg = value_channel_scale_offset;
            let mut value_token_scale_offset_arg = value_token_scale_offset;
            let mut value_zero_offset_arg = value_zero_offset;
            let mut current_tokens_arg = u32::try_from(current_tokens)
                .map_err(|_| "CUDA KVarN current token count exceeds u32".to_string())?;
            let mut sliding_window_arg = u32::try_from(request.sliding_window().unwrap_or(0))
                .map_err(|_| "CUDA KVarN sliding window exceeds u32".to_string())?;
            let mut scale_arg = request.scale();
            let mut softcap_arg = request.softcap().unwrap_or(0.0);
            let kernel = match head_dim {
                128 => "rnb_kvarn_attention_decode_hd128",
                256 => "rnb_kvarn_attention_decode_hd256",
                512 => "rnb_kvarn_attention_decode_hd512",
                _ => unreachable!(),
            };
            self.launch_cached_gemv(
                kernel,
                &[
                    (&mut output_arg as *mut u64).cast(),
                    (&mut query_arg as *mut u64).cast(),
                    (&mut records_arg as *mut u64).cast(),
                    (&mut sink_key_arg as *mut u64).cast(),
                    (&mut sink_value_arg as *mut u64).cast(),
                    (&mut tail_key_arg as *mut u64).cast(),
                    (&mut tail_value_arg as *mut u64).cast(),
                    (&mut current_key_arg as *mut u64).cast(),
                    (&mut current_value_arg as *mut u64).cast(),
                    (&mut kv_len_arg as *mut u32).cast(),
                    (&mut tail_start_arg as *mut u32).cast(),
                    (&mut sink_len_arg as *mut u32).cast(),
                    (&mut tail_len_arg as *mut u32).cast(),
                    (&mut num_blocks_arg as *mut u32).cast(),
                    (&mut num_heads_arg as *mut u32).cast(),
                    (&mut num_kv_heads_arg as *mut u32).cast(),
                    (&mut group_arg as *mut u32).cast(),
                    (&mut value_bits_arg as *mut u32).cast(),
                    (&mut block_bytes_arg as *mut u32).cast(),
                    (&mut key_packed_offset_arg as *mut u32).cast(),
                    (&mut key_scale_offset_arg as *mut u32).cast(),
                    (&mut key_zero_offset_arg as *mut u32).cast(),
                    (&mut key_token_scale_offset_arg as *mut u32).cast(),
                    (&mut value_packed_offset_arg as *mut u32).cast(),
                    (&mut value_channel_scale_offset_arg as *mut u32).cast(),
                    (&mut value_token_scale_offset_arg as *mut u32).cast(),
                    (&mut value_zero_offset_arg as *mut u32).cast(),
                    (&mut scale_arg as *mut f32).cast(),
                    (&mut softcap_arg as *mut f32).cast(),
                    (&mut current_tokens_arg as *mut u32).cast(),
                    (&mut sliding_window_arg as *mut u32).cast(),
                ],
                (num_heads, query_rows_u32, 1),
                (head_dim as u32, 1, 1),
            )
        })();
        self.decode_attention_kvarn.insert(layer_idx, cache);
        result
    }

    pub(super) fn attention_decode_kvarn_to_device(
        &mut self,
        request: rnb_backend_api::KvarnDecodeRequest<'_>,
        output_dev: u64,
    ) -> Result<(), String> {
        self.attention_kvarn_to_device_impl(request, output_dev, None, 1, 0, 0, 0)
    }

    pub(in crate::runtime) fn attention_verify_kvarn_to_device(
        &mut self,
        request: rnb_backend_api::KvarnDecodeRequest<'_>,
        query_dev: u64,
        current_k_bits_dev: u64,
        current_v_bits_dev: u64,
        current_tokens: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        self.attention_kvarn_to_device_impl(
            request,
            output_dev,
            Some(query_dev),
            current_tokens,
            current_k_bits_dev,
            current_v_bits_dev,
            current_tokens,
        )
    }

    pub(super) fn attention_decode_kvarn(
        &mut self,
        request: rnb_backend_api::KvarnDecodeRequest<'_>,
    ) -> Result<Vec<f32>, String> {
        let output_len = request
            .num_heads()
            .checked_mul(request.head_dim())
            .ok_or_else(|| "CUDA KVarN output length overflow".to_string())?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "CUDA KVarN output byte size overflow".to_string())?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        self.attention_decode_kvarn_to_device(request, output_dev)?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_decode_cached_range(
        &mut self,
        layer_index: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        kv_len: usize,
        window_start: usize,
        window_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        // cu47 step 32: caller-provided device output target. Some 시 D2H skip.
        output_dev_target: Option<u64>,
        // cu50 step 40: caller-provided K/V device source (KV cache device-resident).
        // Some 시 host k/v slice 무시 + device copy 만. host f16 변환 + H2D 우회.
        // 정확한 size: kv_rows × 2 bytes (마지막 1 token row 만).
        last_token_k_dev: Option<u64>,
        last_token_v_dev: Option<u64>,
        // cu65: device Q override. Some 이면 host Q H2D skip — Q 이미 device에 있음.
        q_dev_override: Option<u64>,
        use_device_len: bool,
        use_graph: bool,
    ) -> Result<Option<Vec<f32>>, String> {
        self.set_current()?;
        if !matches!(head_dim, 128 | 256 | 512) {
            return Err(format!(
                "CUDA cached decode attention unsupported head_dim={head_dim}"
            ));
        }
        if use_device_len && (head_dim != 512 || window_start != 0) {
            return Err(format!(
                "CUDA cached decode device-length attention requires hd512 full window: head_dim={head_dim} window_start={window_start}"
            ));
        }
        let window_end = window_start
            .checked_add(window_len)
            .ok_or_else(|| "CUDA cached decode attention window overflow".to_string())?;
        if window_len == 0 || window_start > kv_len || window_end > kv_len {
            return Err(format!(
                "CUDA cached decode attention invalid window: kv_len={kv_len} start={window_start} len={window_len}"
            ));
        }
        let kv_rows = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "CUDA cached decode attention kv rows overflow".to_string())?;
        let required_values = kv_len
            .checked_mul(kv_rows)
            .ok_or_else(|| "CUDA cached decode attention kv value overflow".to_string())?;
        let required_bytes = required_values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "CUDA cached decode attention kv byte overflow".to_string())?;
        let host_k_base = k.as_ptr() as usize;
        let host_v_base = v.as_ptr() as usize;

        let mut cache = self
            .decode_attention_kv
            .remove(&layer_index)
            .unwrap_or_default();
        let reset_cache = cache.kv_rows != kv_rows
            || cache.host_k_base != host_k_base
            || cache.host_v_base != host_v_base
            || cache.cached_tokens > kv_len;
        if reset_cache {
            if let Some(ptr) = cache.k_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            if let Some(ptr) = cache.v_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            cache = DecodeAttentionKvCache {
                kv_rows,
                host_k_base,
                host_v_base,
                ..Default::default()
            };
        }
        if cache.k_bits_capacity < required_bytes || cache.v_bits_capacity < required_bytes {
            if let Some(ptr) = cache.k_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            if let Some(ptr) = cache.v_bits_dev.take() {
                unsafe { self.api.mem_free(ptr)? };
            }
            let capacity = align_up(required_bytes, 1024 * 1024);
            // cu26: OOM retry pattern (cu20 generic) — offload q4k resident
            // / clear moe layer cache and retry. Llama-3.1 8B decode attention
            // hit cuMemAlloc panic at ~2.3 GB free remaining; this lets the
            // resident weight cache make room before failing.
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
            cache.kv_rows = kv_rows;
            cache.host_k_base = host_k_base;
            cache.host_v_base = host_v_base;
        }

        let k_dev = cache
            .k_bits_dev
            .ok_or_else(|| "CUDA cached decode attention missing K buffer".to_string())?;
        let v_dev = cache
            .v_bits_dev
            .ok_or_else(|| "CUDA cached decode attention missing V buffer".to_string())?;
        if cache.cached_tokens < kv_len {
            let offset_values = cache
                .cached_tokens
                .checked_mul(kv_rows)
                .ok_or_else(|| "CUDA cached decode attention offset overflow".to_string())?;
            let suffix_values = (kv_len - cache.cached_tokens)
                .checked_mul(kv_rows)
                .ok_or_else(|| "CUDA cached decode attention suffix overflow".to_string())?;
            let offset_bytes = offset_values
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| "CUDA cached decode attention offset byte overflow".to_string())?;
            let suffix_bytes = suffix_values
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| "CUDA cached decode attention suffix byte overflow".to_string())?;
            // cu50 step 40: device source (last_token_k_dev/v_dev) 가 있고 incremental
            // single-token append (suffix_values == kv_rows) 시 host H2D 대신 device copy.
            // 그 외 (prefill 또는 multi-token suffix) 시 기존 host H2D path.
            let use_device_source =
                matches!((last_token_k_dev, last_token_v_dev), (Some(_), Some(_)))
                    && suffix_values == kv_rows;
            if use_device_source {
                let k_src = last_token_k_dev.expect("checked");
                let v_src = last_token_v_dev.expect("checked");
                unsafe {
                    self.api.memcpy_dtod_async(
                        k_dev + offset_bytes as u64,
                        k_src,
                        suffix_bytes,
                        self.stream,
                    )?;
                    self.api.memcpy_dtod_async(
                        v_dev + offset_bytes as u64,
                        v_src,
                        suffix_bytes,
                        self.stream,
                    )?;
                }
            } else {
                unsafe {
                    self.api.memcpy_htod_async(
                        k_dev + offset_bytes as u64,
                        k[offset_values..offset_values + suffix_values]
                            .as_ptr()
                            .cast::<libc::c_void>(),
                        suffix_bytes,
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        v_dev + offset_bytes as u64,
                        v[offset_values..offset_values + suffix_values]
                            .as_ptr()
                            .cast::<libc::c_void>(),
                        suffix_bytes,
                        self.stream,
                    )?;
                }
            }
            cache.cached_tokens = kv_len;
        }
        self.decode_attention_kv.insert(layer_index, cache);

        let q_bytes = std::mem::size_of_val(q);
        let output_len = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "CUDA cached decode attention output len overflow".to_string())?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "CUDA cached decode attention output byte overflow".to_string())?;
        let q_dev = if let Some(dev_ptr) = q_dev_override {
            dev_ptr
        } else {
            let ptr = self.compute_input_ptr(q_bytes)?;
            unsafe {
                self.api.memcpy_htod_async(
                    ptr,
                    q.as_ptr().cast::<libc::c_void>(),
                    q_bytes,
                    self.stream,
                )?;
            }
            ptr
        };
        let output_dev = match output_dev_target {
            Some(ptr) => ptr,
            None => self.compute_output_ptr(output_bytes)?,
        };

        let mut output_arg = output_dev;
        let mut q_arg = q_dev;
        let window_offset_bytes = window_start
            .checked_mul(kv_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "CUDA cached decode attention window offset overflow".to_string())?;
        let mut k_arg = k_dev + window_offset_bytes as u64;
        let mut v_arg = v_dev + window_offset_bytes as u64;
        let mut kv_len_arg = window_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        let prefer_split = Self::cached_decode_split_preferred(head_dim, window_len);
        if use_device_len && !prefer_split {
            let kv_len_dev = self.cu68_graph_kv_len_ptr()?;
            unsafe {
                self.api.memcpy_htod_async(
                    kv_len_dev,
                    (&kv_len_arg as *const u32).cast::<libc::c_void>(),
                    std::mem::size_of::<u32>(),
                    self.stream,
                )?;
            }
            if use_graph {
                let key = Cu68AttentionGraphKey {
                    layer_idx: layer_index,
                    num_heads,
                    num_kv_heads,
                    q_dev: q_arg,
                    k_dev: k_arg,
                    v_dev: v_arg,
                    output_dev: output_arg,
                    kv_len_dev,
                    scale_bits: scale_arg.to_bits(),
                };
                if let Some(graph) = self.cu68_attention_graphs.get(&key) {
                    unsafe {
                        self.api
                            .graph_launch(graph.exec as *mut libc::c_void, self.stream)?
                    };
                } else if self.cu68_attention_graph_warmed.contains(&key) {
                    unsafe { self.api.stream_begin_capture(self.stream)? };
                    let capture_result = self.launch_attention_decode_hd512_len_device_kernel(
                        output_arg,
                        q_arg,
                        k_arg,
                        v_arg,
                        kv_len_dev,
                        heads_arg,
                        kv_heads_arg,
                        scale_arg,
                    );
                    if let Err(err) = capture_result {
                        unsafe {
                            let _ = self.api.stream_end_capture(self.stream);
                        }
                        return Err(err);
                    }
                    let graph = unsafe { self.api.stream_end_capture(self.stream)? };
                    let exec = unsafe { self.api.graph_instantiate(graph)? };
                    self.cu68_attention_graphs.insert(
                        key,
                        SparseMoeGraph {
                            graph: graph as usize,
                            exec: exec as usize,
                        },
                    );
                    let graph = self
                        .cu68_attention_graphs
                        .get(&key)
                        .ok_or_else(|| "missing cu68 attention CUDA graph".to_string())?;
                    unsafe {
                        self.api
                            .graph_launch(graph.exec as *mut libc::c_void, self.stream)?
                    };
                } else {
                    self.cu68_attention_graph_warmed.insert(key);
                    self.launch_attention_decode_hd512_len_device_kernel(
                        output_arg,
                        q_arg,
                        k_arg,
                        v_arg,
                        kv_len_dev,
                        heads_arg,
                        kv_heads_arg,
                        scale_arg,
                    )?;
                }
            } else {
                self.launch_attention_decode_hd512_len_device_kernel(
                    output_arg,
                    q_arg,
                    k_arg,
                    v_arg,
                    kv_len_dev,
                    heads_arg,
                    kv_heads_arg,
                    scale_arg,
                )?;
            }
        } else if prefer_split {
            let chunk_size = match head_dim {
                256 => crate::tuning::decode_attention_hd256_split_chunk_size(),
                512 => crate::tuning::decode_attention_hd512_split_chunk_size(),
                _ => unreachable!("split preference validates head_dim"),
            };
            let num_chunks = window_len.div_ceil(chunk_size);
            let partial_values = num_heads
                .checked_mul(num_chunks)
                .and_then(|n| n.checked_mul(head_dim))
                .ok_or_else(|| "CUDA cached decode attention split partial overflow".to_string())?;
            let partial_bytes = partial_values
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    "CUDA cached decode attention split partial byte overflow".to_string()
                })?;
            let meta_values = num_heads
                .checked_mul(num_chunks)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| "CUDA cached decode attention split meta overflow".to_string())?;
            let meta_bytes = meta_values
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    "CUDA cached decode attention split meta byte overflow".to_string()
                })?;
            let partial_dev = self.compute_mid_a_ptr(partial_bytes)?;
            let meta_dev = self.compute_mid_b_ptr(meta_bytes)?;
            let mut partial_arg = partial_dev;
            let mut meta_arg = meta_dev;
            let mut chunk_size_arg = chunk_size as u32;
            self.launch_cached_gemv(
                match head_dim {
                    256 => "rnb_attention_decode_hd256_split_partials",
                    512 => "rnb_attention_decode_hd512_split_partials",
                    _ => unreachable!("split preference validates head_dim"),
                },
                &[
                    (&mut partial_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut meta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                    (&mut chunk_size_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (num_heads as u32, num_chunks as u32, 1),
                (head_dim as u32, 1, 1),
            )?;
            let mut num_chunks_arg = num_chunks as u32;
            self.launch_cached_gemv(
                match head_dim {
                    256 => "rnb_attention_decode_hd256_split_reduce",
                    512 => "rnb_attention_decode_hd512_split_reduce",
                    _ => unreachable!("split preference validates head_dim"),
                },
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut partial_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut meta_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut num_chunks_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (num_heads as u32, 1, 1),
                (head_dim as u32, 1, 1),
            )?;
        } else {
            let kernel = match head_dim {
                128 => "rnb_attention_decode_hd128",
                256 => "rnb_attention_decode_hd256",
                512 => "rnb_attention_decode_hd512",
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
                (head_dim as u32, 1, 1),
            )?;
        }

        // cu47 step 32: caller 가 device target 받은 경우 D2H + sync 안 함.
        // host return 안 됨 — caller 가 carrier 통해 device 에서 직접 사용.
        if output_dev_target.is_some() {
            return Ok(None);
        }
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
    pub(super) fn glm_mla_prefill_attention_f16(
        &mut self,
        q_absorbed: &[f32],
        q_pe: &[f32],
        cache: &[u16],
        pos_start: usize,
        seq_len: usize,
        num_heads: usize,
        kv_len: usize,
        kv_rank: usize,
        rope_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        self.set_current()?;
        let query_count = seq_len
            .checked_mul(num_heads)
            .ok_or_else(|| "GLM MLA query count overflow".to_string())?;
        let kv_width = kv_rank
            .checked_add(rope_dim)
            .ok_or_else(|| "GLM MLA KV width overflow".to_string())?;
        let output_len = query_count
            .checked_mul(kv_rank)
            .ok_or_else(|| "GLM MLA output length overflow".to_string())?;
        let score_len = query_count
            .checked_mul(kv_len)
            .ok_or_else(|| "GLM MLA score length overflow".to_string())?;

        let q_absorbed_bytes = std::mem::size_of_val(q_absorbed);
        let q_pe_bytes = std::mem::size_of_val(q_pe);
        let cache_bytes = std::mem::size_of_val(cache);
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "GLM MLA output bytes overflow".to_string())?;
        let score_bytes = score_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "GLM MLA score bytes overflow".to_string())?;
        let q_absorbed_dev = self.compute_input_ptr(q_absorbed_bytes)?;
        let q_pe_dev = self.compute_mid_a_ptr(q_pe_bytes)?;
        let cache_dev = self.compute_weights_ptr(cache_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let scores_dev = self.compute_aux_output_ptr(score_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                q_absorbed_dev,
                q_absorbed.as_ptr().cast::<libc::c_void>(),
                q_absorbed_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                q_pe_dev,
                q_pe.as_ptr().cast::<libc::c_void>(),
                q_pe_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                cache_dev,
                cache.as_ptr().cast::<libc::c_void>(),
                cache_bytes,
                self.stream,
            )?;
        }

        let mut scores_arg = scores_dev;
        let mut q_absorbed_arg = q_absorbed_dev;
        let mut q_pe_arg = q_pe_dev;
        let mut cache_arg = cache_dev;
        let mut pos_start_arg = pos_start as u32;
        let mut seq_len_arg = seq_len as u32;
        let mut num_heads_arg = num_heads as u32;
        let mut kv_len_arg = kv_len as u32;
        let mut kv_rank_arg = kv_rank as u32;
        let mut rope_dim_arg = rope_dim as u32;
        let mut kv_width_arg = kv_width as u32;
        let mut scale_arg = scale;
        self.launch_cached_gemv(
            "rnb_glm_mla_prefill_scores_f16",
            &[
                (&mut scores_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_absorbed_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_pe_arg as *mut u64).cast::<libc::c_void>(),
                (&mut cache_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_rank_arg as *mut u32).cast::<libc::c_void>(),
                (&mut rope_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_width_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (query_count as u32, kv_len.div_ceil(4) as u32, 1),
            (128, 1, 1),
        )?;
        self.launch_cached_gemv(
            "rnb_glm_mla_prefill_softmax",
            &[
                (&mut scores_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (query_count as u32, 1, 1),
            (256, 1, 1),
        )?;

        let mut output_arg = output_dev;
        self.launch_cached_gemv(
            "rnb_glm_mla_prefill_weighted_f16",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut scores_arg as *mut u64).cast::<libc::c_void>(),
                (&mut cache_arg as *mut u64).cast::<libc::c_void>(),
                (&mut pos_start_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut num_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_rank_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_width_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (query_count as u32, 1, 1),
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
        }
        self.stream_synchronize()?;
        Ok(output)
    }
}
