use super::*;

impl CudaState {
    pub(super) fn ssm_conv1d_silu(
        &mut self,
        input: &[f32],
        kernel: &[f32],
        seq_len: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, String> {
        let input_bytes = std::mem::size_of_val(input);
        let kernel_bytes = std::mem::size_of_val(kernel);
        let output_len = seq_len * channels;
        let output_bytes = output_len * std::mem::size_of::<f32>();

        let input_dev = self.compute_input_ptr(input_bytes)?;
        let kernel_dev = self.compute_weights_ptr(kernel_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                kernel_dev,
                kernel.as_ptr().cast::<libc::c_void>(),
                kernel_bytes,
                self.stream,
            )?;
        }

        let mut output_arg = output_dev;
        let mut input_arg = input_dev;
        let mut kernel_arg = kernel_dev;
        let mut seq_arg = seq_len as u32;
        let mut channels_arg = channels as u32;
        let mut kernel_size_arg = kernel_size as u32;
        self.launch_cached_gemv(
            "rnb_ssm_conv1d_silu",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut kernel_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut channels_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kernel_size_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (((output_len as u32).saturating_add(255)) / 256, 1, 1),
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

    pub(super) fn gdn_gated_norm_silu(
        &mut self,
        delta_out: &[f32],
        z: &[f32],
        norm_weight: &[f32],
        rows: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let len = rows * head_dim;
        let data_bytes = len * std::mem::size_of::<f32>();
        let norm_bytes = std::mem::size_of_val(norm_weight);

        let delta_dev = self.compute_input_ptr(data_bytes)?;
        let z_dev = self.compute_mid_a_ptr(data_bytes)?;
        let norm_dev = self.compute_mid_b_ptr(norm_bytes)?;
        let output_dev = self.compute_output_ptr(data_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                delta_dev,
                delta_out.as_ptr().cast::<libc::c_void>(),
                data_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                z_dev,
                z.as_ptr().cast::<libc::c_void>(),
                data_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                norm_dev,
                norm_weight.as_ptr().cast::<libc::c_void>(),
                norm_bytes,
                self.stream,
            )?;
        }

        let mut output_arg = output_dev;
        let mut delta_arg = delta_dev;
        let mut z_arg = z_dev;
        let mut norm_arg = norm_dev;
        let mut rows_arg = rows as u32;
        let mut head_dim_arg = head_dim as u32;
        let mut eps_arg = eps;
        self.launch_cached_gemv(
            "rnb_gdn_gated_norm_silu",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut delta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut z_arg as *mut u64).cast::<libc::c_void>(),
                (&mut norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )?;

        let mut output = vec![0.0f32; len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                data_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_gated_norm_silu_f32_gemm(
        &mut self,
        delta_out: &[f32],
        z: &[f32],
        norm_weight: &[f32],
        proj_weights: &[f32],
        seq_len: usize,
        head_dim: usize,
        proj_rows: usize,
        proj_cols: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let norm_rows = seq_len * (proj_cols / head_dim);
        let gated_len = seq_len * proj_cols;
        let gated_bytes = gated_len * std::mem::size_of::<f32>();
        let delta_bytes = delta_out.len() * std::mem::size_of::<f32>();
        let z_bytes = z.len() * std::mem::size_of::<f32>();
        let norm_bytes = std::mem::size_of_val(norm_weight);
        let weights_bytes = std::mem::size_of_val(proj_weights);
        let output_len = seq_len * proj_rows;
        let output_bytes = output_len * std::mem::size_of::<f32>();

        let delta_dev = self.compute_input_ptr(delta_bytes)?;
        let z_dev = self.compute_mid_a_ptr(z_bytes)?;
        let norm_dev = self.compute_mid_b_ptr(norm_bytes)?;
        let gated_dev = self.compute_output_ptr(gated_bytes)?;
        let weights_dev = self.compute_weights_ptr(weights_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                delta_dev,
                delta_out.as_ptr().cast::<libc::c_void>(),
                delta_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                z_dev,
                z.as_ptr().cast::<libc::c_void>(),
                z_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                norm_dev,
                norm_weight.as_ptr().cast::<libc::c_void>(),
                norm_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                weights_dev,
                proj_weights.as_ptr().cast::<libc::c_void>(),
                weights_bytes,
                self.stream,
            )?;
        }

        let mut gated_arg = gated_dev;
        let mut delta_arg = delta_dev;
        let mut z_arg = z_dev;
        let mut norm_arg = norm_dev;
        let mut rows_arg = norm_rows as u32;
        let mut head_dim_arg = head_dim as u32;
        let mut eps_arg = eps;
        self.launch_cached_gemv(
            "rnb_gdn_gated_norm_silu",
            &[
                (&mut gated_arg as *mut u64).cast::<libc::c_void>(),
                (&mut delta_arg as *mut u64).cast::<libc::c_void>(),
                (&mut z_arg as *mut u64).cast::<libc::c_void>(),
                (&mut norm_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut head_dim_arg as *mut u32).cast::<libc::c_void>(),
                (&mut eps_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (norm_rows as u32, 1, 1),
            (256, 1, 1),
        )?;

        let proj_dev = self.compute_full_gate_ptr(output_bytes)?;
        let stream = self.stream;
        let cublas = self.cublas_state_mut()?;
        unsafe {
            cublas.api.set_stream(cublas.handle, stream)?;
            cublas.api.sgemm(
                cublas.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                proj_rows as i32,
                seq_len as i32,
                proj_cols as i32,
                1.0,
                weights_dev,
                proj_cols as i32,
                gated_dev,
                proj_cols as i32,
                0.0,
                proj_dev,
                proj_rows as i32,
            )?;
        }

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                proj_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }
}
