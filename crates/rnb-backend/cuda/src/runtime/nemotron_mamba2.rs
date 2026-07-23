use super::mtp_verify::GGML_Q8_0;
use super::*;
use rnb_backend_api::{DeviceTensorDesc, DeviceTensorId, DeviceTensorRole, ScalarType};

#[derive(Debug)]
pub struct NemotronMamba2DeviceOutput {
    pub output_id: DeviceTensorId,
    pub output_desc: DeviceTensorDesc,
    pub conv_state_d2h_bytes: usize,
    pub delta_state_d2h_bytes: usize,
}

fn checked_q8_0_bytes(label: &str, rows: usize, cols: usize) -> Result<usize, String> {
    rows.checked_mul(cols / 32)
        .and_then(|v| v.checked_mul(34))
        .ok_or_else(|| format!("state_shape_mismatch:{label}_q8_bytes_overflow"))
}

fn checked_f32_bytes(label: &str, len: usize) -> Result<usize, String> {
    len.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("state_shape_mismatch:{label}_bytes_overflow"))
}

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_mamba2_prefill_device(
        &mut self,
        input_id: DeviceTensorId,
        input_desc: DeviceTensorDesc,
        ssm_in_quant: u32,
        ssm_in: &[u8],
        ssm_in_rows: usize,
        ssm_in_cols: usize,
        ssm_out_quant: u32,
        ssm_out: &[u8],
        ssm_out_rows: usize,
        ssm_out_cols: usize,
        input_norm: &[f32],
        conv_kernel: &[f32],
        conv_bias: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        ssm_d: &[f32],
        ssm_norm: &[f32],
        conv_state: &mut [f32],
        delta_state: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
        d_inner: usize,
        conv_channels: usize,
        bc_dim: usize,
        num_heads: usize,
        head_dim: usize,
        n_group: usize,
        d_state: usize,
        conv_kernel_size: usize,
        norm_eps: f32,
    ) -> Result<NemotronMamba2DeviceOutput, String> {
        if ssm_in_quant != GGML_Q8_0 {
            return Err(format!("unsupported_ssm_in_quant:{ssm_in_quant}"));
        }
        if ssm_out_quant != GGML_Q8_0 {
            return Err(format!("unsupported_ssm_out_quant:{ssm_out_quant}"));
        }
        if seq_len == 0
            || hidden_dim == 0
            || d_inner == 0
            || conv_channels == 0
            || num_heads == 0
            || head_dim == 0
            || n_group == 0
            || d_state == 0
            || conv_kernel_size == 0
        {
            return Err(format!(
                "state_shape_mismatch: zero dimension seq_len={seq_len} hidden_dim={hidden_dim} d_inner={d_inner} conv_channels={conv_channels} heads={num_heads} head_dim={head_dim} groups={n_group} d_state={d_state} kernel={conv_kernel_size}"
            ));
        }
        if conv_kernel_size < 2 {
            return Err(format!(
                "unsupported_conv_shape: conv_kernel_size={conv_kernel_size} must be >= 2"
            ));
        }
        if d_state > 256 {
            return Err(format!(
                "unsupported_state_dim: state_dim={d_state} exceeds prefill scan block width 256"
            ));
        }
        if input_desc.rows() != seq_len
            || input_desc.cols() != hidden_dim
            || input_desc.dtype() != ScalarType::F32
        {
            return Err(format!(
                "device_tensor_shape_mismatch: input_desc={input_desc:?} seq_len={seq_len} hidden_dim={hidden_dim}"
            ));
        }
        if ssm_in_cols != hidden_dim {
            return Err(format!(
                "state_shape_mismatch: ssm_in_cols={ssm_in_cols} hidden_dim={hidden_dim}"
            ));
        }
        let bc_twice = bc_dim
            .checked_mul(2)
            .ok_or_else(|| "state_shape_mismatch:bc_twice_overflow".to_string())?;
        let expected_conv_channels = d_inner
            .checked_add(bc_twice)
            .ok_or_else(|| "state_shape_mismatch:conv_channels_overflow".to_string())?;
        if conv_channels != expected_conv_channels {
            return Err(format!(
                "state_shape_mismatch: conv_channels={conv_channels} expected={expected_conv_channels} d_inner={d_inner} bc_dim={bc_dim}"
            ));
        }
        let expected_rows = d_inner
            .checked_add(conv_channels)
            .and_then(|v| v.checked_add(num_heads))
            .ok_or_else(|| "state_shape_mismatch:ssm_in_rows_overflow".to_string())?;
        if ssm_in_rows != expected_rows {
            return Err(format!(
                "state_shape_mismatch: ssm_in_rows={ssm_in_rows} expected={expected_rows}"
            ));
        }
        if ssm_out_rows != hidden_dim || ssm_out_cols != d_inner {
            return Err(format!(
                "state_shape_mismatch: ssm_out=[{ssm_out_rows},{ssm_out_cols}] expected=[{hidden_dim},{d_inner}]"
            ));
        }
        let expected_d_inner = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "state_shape_mismatch:d_inner_overflow".to_string())?;
        let expected_bc_dim = n_group
            .checked_mul(d_state)
            .ok_or_else(|| "state_shape_mismatch:bc_dim_overflow".to_string())?;
        if d_inner != expected_d_inner || bc_dim != expected_bc_dim || d_inner % n_group != 0 {
            return Err(format!(
                "state_shape_mismatch: d_inner={d_inner} heads={num_heads} head_dim={head_dim} bc_dim={bc_dim} groups={n_group} d_state={d_state}"
            ));
        }
        if num_heads % n_group != 0 {
            return Err(format!(
                "state_shape_mismatch: heads={num_heads} groups={n_group}"
            ));
        }
        if ssm_in_cols % 32 != 0 || ssm_out_cols % 32 != 0 {
            return Err(format!(
                "state_shape_mismatch: q8 cols must be divisible by 32: ssm_in_cols={ssm_in_cols} ssm_out_cols={ssm_out_cols}"
            ));
        }
        let expected_ssm_in_bytes = checked_q8_0_bytes("ssm_in", ssm_in_rows, ssm_in_cols)?;
        if ssm_in.len() != expected_ssm_in_bytes {
            return Err(format!(
                "state_shape_mismatch: ssm_in bytes={} expected={}",
                ssm_in.len(),
                expected_ssm_in_bytes
            ));
        }
        let expected_ssm_out_bytes = checked_q8_0_bytes("ssm_out", ssm_out_rows, ssm_out_cols)?;
        if ssm_out.len() != expected_ssm_out_bytes {
            return Err(format!(
                "state_shape_mismatch: ssm_out bytes={} expected={}",
                ssm_out.len(),
                expected_ssm_out_bytes
            ));
        }
        let expected_conv_kernel_len = conv_kernel_size
            .checked_mul(conv_channels)
            .ok_or_else(|| "unsupported_conv_shape:conv_kernel_len_overflow".to_string())?;
        if conv_kernel.len() != expected_conv_kernel_len || conv_bias.len() != conv_channels {
            return Err(format!(
                "unsupported_conv_shape: kernel={} bias={} expected_kernel={} expected_bias={conv_channels}",
                conv_kernel.len(),
                conv_bias.len(),
                expected_conv_kernel_len
            ));
        }
        if dt_bias.len() < num_heads || ssm_a.len() < num_heads || ssm_d.len() < num_heads {
            return Err(format!(
                "state_shape_mismatch: dt_bias={} a={} d={} heads={num_heads}",
                dt_bias.len(),
                ssm_a.len(),
                ssm_d.len()
            ));
        }
        if input_norm.len() != hidden_dim {
            return Err(format!(
                "state_shape_mismatch: input_norm={} hidden_dim={hidden_dim}",
                input_norm.len()
            ));
        }
        if ssm_norm.len() != d_inner {
            return Err(format!(
                "state_shape_mismatch: ssm_norm={} d_inner={d_inner}",
                ssm_norm.len()
            ));
        }
        let conv_state_len = conv_kernel_size
            .saturating_sub(1)
            .checked_mul(conv_channels)
            .ok_or_else(|| "state_shape_mismatch:conv_state_len_overflow".to_string())?;
        if conv_state.len() != conv_state_len {
            return Err(format!(
                "state_shape_mismatch: conv_state={} expected={conv_state_len}",
                conv_state.len()
            ));
        }
        let delta_state_len = d_inner
            .checked_mul(d_state)
            .ok_or_else(|| "state_shape_mismatch:delta_state_len_overflow".to_string())?;
        if delta_state.len() != delta_state_len {
            return Err(format!(
                "state_shape_mismatch: delta_state={} expected={delta_state_len}",
                delta_state.len()
            ));
        }

        self.set_current()?;
        let input_dev = self.device_tensor_ptr(input_id, input_desc)?;
        let input_norm_dev = self.resident_f32_ptr(input_norm)?;
        let ssm_norm_dev = self.resident_f32_ptr(ssm_norm)?;
        let conv_kernel_dev = self.resident_f32_ptr(conv_kernel)?;
        let conv_bias_dev = self.resident_f32_ptr(conv_bias)?;
        let dt_bias_dev = self.resident_f32_ptr(dt_bias)?;
        let a_dev = self.resident_f32_ptr(ssm_a)?;
        let d_dev = self.resident_f32_ptr(ssm_d)?;

        let hidden_len = seq_len
            .checked_mul(hidden_dim)
            .ok_or_else(|| "state_shape_mismatch:hidden_len_overflow".to_string())?;
        let projected_len = seq_len
            .checked_mul(expected_rows)
            .ok_or_else(|| "state_shape_mismatch:projected_len_overflow".to_string())?;
        let z_len = seq_len
            .checked_mul(d_inner)
            .ok_or_else(|| "state_shape_mismatch:z_len_overflow".to_string())?;
        let conv_seed_len = seq_len
            .checked_mul(conv_channels)
            .ok_or_else(|| "state_shape_mismatch:conv_seed_len_overflow".to_string())?;
        let dt_len = seq_len
            .checked_mul(num_heads)
            .ok_or_else(|| "state_shape_mismatch:dt_len_overflow".to_string())?;
        let conv_input_rows = seq_len
            .checked_add(conv_kernel_size - 1)
            .ok_or_else(|| "state_shape_mismatch:conv_input_rows_overflow".to_string())?;
        let conv_input_len = conv_input_rows
            .checked_mul(conv_channels)
            .ok_or_else(|| "state_shape_mismatch:conv_input_len_overflow".to_string())?;
        let hidden_bytes = checked_f32_bytes("hidden", hidden_len)?;
        let projected_bytes = checked_f32_bytes("projected", projected_len)?;
        let z_bytes = checked_f32_bytes("z", z_len)?;
        let conv_seed_bytes = checked_f32_bytes("conv_seed", conv_seed_len)?;
        let dt_bytes = checked_f32_bytes("dt", dt_len)?;
        let conv_state_bytes = checked_f32_bytes("conv_state", conv_state_len)?;
        let conv_input_bytes = checked_f32_bytes("conv_input", conv_input_len)?;
        let gated_rows = seq_len
            .checked_mul(n_group)
            .ok_or_else(|| "state_shape_mismatch:gated_rows_overflow".to_string())?;
        let normed_dev = self.compute_input_ptr(hidden_bytes)?;
        let projected_dev = self.compute_mid_a_ptr(projected_bytes)?;
        let z_dev = self.compute_aux_output_ptr(z_bytes)?;
        let conv_seed_dev = self.compute_route_ptr(conv_seed_bytes)?;
        let dt_dev = self.compute_gate_ptrs_ptr(dt_bytes)?;
        let conv_state_dev = self.compute_up_ptrs_ptr(conv_state_bytes)?;
        let conv_input_dev = self.compute_down_ptrs_ptr(conv_input_bytes)?;
        let conv_out_dev = self.compute_full_gate_ptr(conv_seed_bytes)?;
        let scan_out_dev = self.compute_full_up_ptr(z_bytes)?;
        let gated_dev = self.compute_full_down_ptr(z_bytes)?;
        let proj_dev = self.compute_output_ptr(hidden_bytes)?;
        let output_desc = DeviceTensorDesc::new(
            seq_len,
            hidden_dim,
            ScalarType::F32,
            DeviceTensorRole::MambaOutput,
        );
        let output_bytes = output_desc
            .byte_len()
            .ok_or_else(|| "Mamba2 output byte overflow".to_string())?;
        let (output_dev, output_storage) =
            if let Some((ptr, storage)) = self.nemotron_workspace_hidden_output_ptr(output_bytes) {
                (ptr, storage)
            } else {
                (
                    unsafe { self.api.mem_alloc(output_bytes)? },
                    DeviceTensorStorage::Owned,
                )
            };

        let run = (|| -> Result<(), String> {
            self.launch_rms_norm_rows_f32(
                input_dev,
                input_norm_dev,
                normed_dev,
                norm_eps,
                seq_len,
                hidden_dim,
                false,
            )?;
            let in_blocks = ssm_in_cols / 32;
            self.launch_q8_0_gemv_batch_to_dev(
                ssm_in,
                ssm_in_rows,
                in_blocks,
                seq_len,
                normed_dev,
                projected_dev,
            )?;
            self.launch_nemotron_mamba2_split_projection(
                projected_dev,
                dt_bias_dev,
                z_dev,
                conv_seed_dev,
                dt_dev,
                seq_len,
                d_inner,
                conv_channels,
                num_heads,
            )?;
            unsafe {
                self.api.memcpy_htod_async(
                    conv_state_dev,
                    conv_state.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(conv_state),
                    self.stream,
                )?;
            }
            self.launch_gdn_build_conv_input_f32(
                conv_input_dev,
                conv_state_dev,
                conv_seed_dev,
                seq_len,
                conv_channels,
                conv_kernel_size,
            )?;
            self.launch_nemotron_mamba2_conv1d_bias_silu_dev(
                conv_input_dev,
                conv_kernel_dev,
                conv_bias_dev,
                conv_out_dev,
                seq_len,
                conv_channels,
                conv_kernel_size,
            )?;
            self.launch_nemotron_mamba2_prefill_scan_dev(
                scan_out_dev,
                delta_state,
                conv_out_dev,
                dt_dev,
                a_dev,
                d_dev,
                seq_len,
                conv_channels,
                bc_dim,
                num_heads,
                head_dim,
                d_state,
                n_group,
            )?;
            self.launch_gdn_gated_norm_silu_dev(
                gated_dev,
                scan_out_dev,
                z_dev,
                ssm_norm_dev,
                gated_rows,
                d_inner / n_group,
                norm_eps,
            )?;
            let out_blocks = ssm_out_cols / 32;
            self.launch_q8_0_gemv_batch_to_dev(
                ssm_out,
                ssm_out_rows,
                out_blocks,
                seq_len,
                gated_dev,
                proj_dev,
            )?;
            self.launch_nemotron_mamba2_add_residual(output_dev, proj_dev, input_dev, hidden_len)?;
            if conv_state_len > 0 {
                let offset_bytes = seq_len
                    .checked_mul(conv_channels)
                    .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| "state_shape_mismatch:conv_state_offset_overflow".to_string())?;
                unsafe {
                    self.api.memcpy_dtoh_async(
                        conv_state.as_mut_ptr().cast::<libc::c_void>(),
                        conv_input_dev + offset_bytes as u64,
                        std::mem::size_of_val(conv_state),
                        self.stream,
                    )?;
                }
            }
            let delta_state_dev = self.resident_delta_state_ptr(delta_state)?;
            unsafe {
                self.api.memcpy_dtoh_async(
                    delta_state.as_mut_ptr().cast::<libc::c_void>(),
                    delta_state_dev,
                    std::mem::size_of_val(delta_state),
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            Ok(())
        })();
        if let Err(err) = run {
            if let Err(cleanup_err) =
                self.cleanup_device_tensor_storage_allocation(output_dev, output_storage)
            {
                return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
            }
            return Err(err);
        }

        let output_id = match self.insert_device_tensor_slot_with_storage(
            output_dev,
            output_bytes,
            output_desc,
            output_storage,
        ) {
            Ok(output_id) => output_id,
            Err(err) => {
                if let Err(cleanup_err) =
                    self.release_workspace_storage_after_insert_failure(output_storage)
                {
                    return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
                }
                return Err(err);
            }
        };
        Ok(NemotronMamba2DeviceOutput {
            output_id,
            output_desc,
            conv_state_d2h_bytes: std::mem::size_of_val(conv_state),
            delta_state_d2h_bytes: std::mem::size_of_val(delta_state),
        })
    }
}
