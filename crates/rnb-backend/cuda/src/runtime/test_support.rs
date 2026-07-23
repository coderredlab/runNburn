use super::*;
use crate::runtime::state::{pack_q4k_for_q8dot, pack_q6k_for_q8dot};

#[cfg(test)]
pub fn launch_smoke_add_one_for_test(input: f32) -> Result<f32, String> {
    let state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of::<f32>())?;
    let output_dev = state.mem_alloc(std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            (&input as *const f32).cast::<libc::c_void>(),
            std::mem::size_of::<f32>(),
            state.stream,
        )?;
    }
    let mut output_arg = output_dev;
    let mut input_arg = input_dev;
    state.launch_ptx_kernel(
        SMOKE_ADD_ONE_PTX,
        "rnb_smoke_add_one",
        &[
            (&mut output_arg as *mut u64).cast::<libc::c_void>(),
            (&mut input_arg as *mut u64).cast::<libc::c_void>(),
        ],
        (1, 1, 1),
        (1, 1, 1),
    )?;
    let mut output = 0.0f32;
    unsafe {
        state.api.memcpy_dtoh_async(
            (&mut output as *mut f32).cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of::<f32>(),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn nemotron_mamba2_split_projection_for_test(
    projected: &[f32],
    dt_bias: &[f32],
    seq_len: usize,
    d_inner: usize,
    conv_channels: usize,
    num_heads: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let rows = d_inner + conv_channels + num_heads;
    if projected.len() != seq_len * rows {
        return Err(format!(
            "Mamba2 split test projected len mismatch: got {}, expected {}",
            projected.len(),
            seq_len * rows
        ));
    }
    if dt_bias.len() != num_heads {
        return Err(format!(
            "Mamba2 split test dt_bias len mismatch: got {}, expected {num_heads}",
            dt_bias.len()
        ));
    }
    let mut state = CudaState::open()?;
    let projected_dev = state.mem_alloc(std::mem::size_of_val(projected))?;
    let dt_bias_dev = state.mem_alloc(std::mem::size_of_val(dt_bias))?;
    let z_len = seq_len * d_inner;
    let conv_len = seq_len * conv_channels;
    let dt_len = seq_len * num_heads;
    let z_dev = state.mem_alloc(z_len * std::mem::size_of::<f32>())?;
    let conv_dev = state.mem_alloc(conv_len * std::mem::size_of::<f32>())?;
    let dt_dev = state.mem_alloc(dt_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            projected_dev,
            projected.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(projected),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            dt_bias_dev,
            dt_bias.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(dt_bias),
            state.stream,
        )?;
    }
    state.launch_nemotron_mamba2_split_projection(
        projected_dev,
        dt_bias_dev,
        z_dev,
        conv_dev,
        dt_dev,
        seq_len,
        d_inner,
        conv_channels,
        num_heads,
    )?;
    let mut z = vec![0.0f32; z_len];
    let mut conv = vec![0.0f32; conv_len];
    let mut dt = vec![0.0f32; dt_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            z.as_mut_ptr().cast::<libc::c_void>(),
            z_dev,
            std::mem::size_of_val(&z[..]),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            conv.as_mut_ptr().cast::<libc::c_void>(),
            conv_dev,
            std::mem::size_of_val(&conv[..]),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            dt.as_mut_ptr().cast::<libc::c_void>(),
            dt_dev,
            std::mem::size_of_val(&dt[..]),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(projected_dev)?;
        state.api.mem_free(dt_bias_dev)?;
        state.api.mem_free(z_dev)?;
        state.api.mem_free(conv_dev)?;
        state.api.mem_free(dt_dev)?;
    }
    Ok((z, conv, dt))
}

#[cfg(test)]
pub fn nemotron_mamba2_conv1d_bias_silu_for_test(
    input: &[f32],
    kernel: &[f32],
    bias: &[f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, String> {
    let input_rows = seq_len
        .checked_add(kernel_size)
        .ok_or_else(|| "Mamba2 conv test input rows overflow".to_string())?;
    let expected_input_len = input_rows
        .checked_mul(channels)
        .ok_or_else(|| "Mamba2 conv test input len overflow".to_string())?;
    if input.len() != expected_input_len {
        return Err(format!(
            "Mamba2 conv test input len mismatch: got {}, expected {}",
            input.len(),
            expected_input_len
        ));
    }
    let expected_kernel_len = kernel_size
        .checked_mul(channels)
        .ok_or_else(|| "Mamba2 conv test kernel len overflow".to_string())?;
    if kernel.len() != expected_kernel_len {
        return Err(format!(
            "Mamba2 conv test kernel len mismatch: got {}, expected {}",
            kernel.len(),
            expected_kernel_len
        ));
    }
    if bias.len() != channels {
        return Err(format!(
            "Mamba2 conv test bias len mismatch: got {}, expected {channels}",
            bias.len()
        ));
    }
    let output_len = seq_len
        .checked_mul(channels)
        .ok_or_else(|| "Mamba2 conv test output len overflow".to_string())?;
    if output_len == 0 {
        return Ok(Vec::new());
    }
    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let kernel_dev = state.mem_alloc(std::mem::size_of_val(kernel))?;
    let bias_dev = state.mem_alloc(std::mem::size_of_val(bias))?;
    let output_dev = state.mem_alloc(output_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            kernel_dev,
            kernel.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(kernel),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            bias_dev,
            bias.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(bias),
            state.stream,
        )?;
    }
    state.launch_nemotron_mamba2_conv1d_bias_silu_dev(
        input_dev,
        kernel_dev,
        bias_dev,
        output_dev,
        seq_len,
        channels,
        kernel_size,
    )?;
    let mut output = vec![0.0f32; output_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(&output[..]),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(kernel_dev)?;
        state.api.mem_free(bias_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn nemotron_mamba2_add_residual_for_test(
    proj: &[f32],
    residual: &[f32],
) -> Result<Vec<f32>, String> {
    if proj.len() != residual.len() {
        return Err(format!(
            "Mamba2 residual test len mismatch: proj {}, residual {}",
            proj.len(),
            residual.len()
        ));
    }
    if proj.is_empty() {
        return Ok(Vec::new());
    }
    let mut state = CudaState::open()?;
    let proj_dev = state.mem_alloc(std::mem::size_of_val(proj))?;
    let residual_dev = state.mem_alloc(std::mem::size_of_val(residual))?;
    let output_dev = state.mem_alloc(std::mem::size_of_val(proj))?;
    unsafe {
        state.api.memcpy_htod_async(
            proj_dev,
            proj.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(proj),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            residual_dev,
            residual.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(residual),
            state.stream,
        )?;
    }
    state.launch_nemotron_mamba2_add_residual(output_dev, proj_dev, residual_dev, proj.len())?;
    let mut output = vec![0.0f32; proj.len()];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(&output[..]),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(proj_dev)?;
        state.api.mem_free(residual_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q4k_f32_gemm_batch_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4 F32 test cols must be divisible by 256, got {cols}"
        ));
    }
    let seq_len = input
        .len()
        .checked_div(cols)
        .ok_or_else(|| "Q4 F32 test cols must be non-zero".to_string())?;
    if input.len() != seq_len * cols {
        return Err(format!(
            "Q4 F32 test input length mismatch: got {}, expected multiple of {cols}",
            input.len()
        ));
    }
    let mut state = CudaState::open()?;
    state.resident_q4_f32_limit = usize::MAX;
    let weights_dev = state
        .resident_q4k_f32_ptr(weights, rows, cols / 256)?
        .ok_or_else(|| "Q4 F32 test cache unexpectedly disabled".to_string())?;
    let input_dev = state.compute_input_ptr(std::mem::size_of_val(input))?;
    let output_len = seq_len * rows;
    let output_dev = state.compute_output_ptr(output_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
    }
    state.sgemm_device(weights_dev, rows, cols, input_dev, seq_len, output_dev)?;
    let mut output = vec![0.0f32; output_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    Ok(output)
}

#[cfg(test)]
pub fn q6k_f32_gemm_batch_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q6 F32 test cols must be divisible by 256, got {cols}"
        ));
    }
    let seq_len = input
        .len()
        .checked_div(cols)
        .ok_or_else(|| "Q6 F32 test cols must be non-zero".to_string())?;
    if input.len() != seq_len * cols {
        return Err(format!(
            "Q6 F32 test input length mismatch: got {}, expected multiple of {cols}",
            input.len()
        ));
    }
    let mut state = CudaState::open()?;
    state.resident_q6_f32_limit = usize::MAX;
    let weights_dev = state
        .resident_q6k_f32_ptr(weights, rows, cols / 256)?
        .ok_or_else(|| "Q6 F32 test cache unexpectedly disabled".to_string())?;
    let input_dev = state.compute_input_ptr(std::mem::size_of_val(input))?;
    let output_len = seq_len * rows;
    let output_dev = state.compute_output_ptr(output_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
    }
    state.sgemm_device(weights_dev, rows, cols, input_dev, seq_len, output_dev)?;
    let mut output = vec![0.0f32; output_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    Ok(output)
}

#[cfg(test)]
pub fn launch_smoke_graph_add_one_for_test(input: f32) -> Result<f32, String> {
    let state = CudaState::open()?;
    let value_dev = state.mem_alloc(std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            value_dev,
            (&input as *const f32).cast::<libc::c_void>(),
            std::mem::size_of::<f32>(),
            state.stream,
        )?;
    }

    let module = unsafe { state.api.module_load_data(SMOKE_ADD_ONE_PTX)? };
    let result = (|| {
        let function = unsafe { state.api.module_get_function(module, "rnb_smoke_add_one")? };
        let mut output_arg = value_dev;
        let mut input_arg = value_dev;
        let params = [
            (&mut output_arg as *mut u64).cast::<libc::c_void>(),
            (&mut input_arg as *mut u64).cast::<libc::c_void>(),
        ];
        unsafe {
            state.api.stream_begin_capture(state.stream)?;
            state.api.launch_kernel(
                function,
                (1, 1, 1),
                (1, 1, 1),
                0,
                state.stream,
                params.as_ptr() as *mut *mut libc::c_void,
            )?;
            let graph = state.api.stream_end_capture(state.stream)?;
            let exec = state.api.graph_instantiate(graph)?;
            let launch_result = (|| {
                state.api.graph_launch(exec, state.stream)?;
                state.api.graph_launch(exec, state.stream)?;
                state.api.stream_synchronize(state.stream)
            })();
            let destroy_exec = state.api.graph_exec_destroy(exec);
            let destroy_graph = state.api.graph_destroy(graph);
            launch_result?;
            destroy_exec?;
            destroy_graph
        }
    })();
    let unload_result = unsafe { state.api.module_unload(module) };
    result?;
    unload_result?;

    let mut output = 0.0f32;
    unsafe {
        state.api.memcpy_dtoh_async(
            (&mut output as *mut f32).cast::<libc::c_void>(),
            value_dev,
            std::mem::size_of::<f32>(),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(value_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q4k_block_dot_for_test(block: &[u8; 144], input: &[f32; 256]) -> Result<f32, String> {
    let state = CudaState::open()?;
    let block_dev = state.mem_alloc(block.len())?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let output_dev = state.mem_alloc(std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            block_dev,
            block.as_ptr().cast::<libc::c_void>(),
            block.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
    }
    let mut output_arg = output_dev;
    let mut block_arg = block_dev;
    let mut input_arg = input_dev;
    state.launch_ptx_kernel(
        Q4K_BLOCK_DOT_PTX,
        "rnb_q4k_block_dot",
        &[
            (&mut output_arg as *mut u64).cast::<libc::c_void>(),
            (&mut block_arg as *mut u64).cast::<libc::c_void>(),
            (&mut input_arg as *mut u64).cast::<libc::c_void>(),
        ],
        (1, 1, 1),
        (1, 1, 1),
    )?;
    let mut output = 0.0f32;
    unsafe {
        state.api.memcpy_dtoh_async(
            (&mut output as *mut f32).cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of::<f32>(),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(block_dev)?;
        state.api.mem_free(input_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q4k_row_dot_for_test(row: &[u8], input: &[f32]) -> Result<f32, String> {
    if row.len() % 144 != 0 {
        return Err(format!(
            "Q4_K row bytes must be multiple of 144, got {}",
            row.len()
        ));
    }
    if input.len() != (row.len() / 144) * 256 {
        return Err(format!(
            "Q4_K input length mismatch: row blocks={}, input={}",
            row.len() / 144,
            input.len()
        ));
    }
    let state = CudaState::open()?;
    let row_dev = state.mem_alloc(row.len())?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let output_dev = state.mem_alloc(std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            row_dev,
            row.as_ptr().cast::<libc::c_void>(),
            row.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
    }
    let mut output_arg = output_dev;
    let mut row_arg = row_dev;
    let mut input_arg = input_dev;
    let mut blocks_arg = (row.len() / 144) as u32;
    state.launch_ptx_kernel(
        Q4K_ROW_DOT_PTX,
        "rnb_q4k_row_dot",
        &[
            (&mut output_arg as *mut u64).cast::<libc::c_void>(),
            (&mut row_arg as *mut u64).cast::<libc::c_void>(),
            (&mut input_arg as *mut u64).cast::<libc::c_void>(),
            (&mut blocks_arg as *mut u32).cast::<libc::c_void>(),
        ],
        (1, 1, 1),
        (1, 1, 1),
    )?;
    let mut output = 0.0f32;
    unsafe {
        state.api.memcpy_dtoh_async(
            (&mut output as *mut f32).cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of::<f32>(),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(row_dev)?;
        state.api.mem_free(input_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q4k_gemv_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q4k_gemv(weights, rows, cols, input)
}

#[cfg(test)]
pub fn q4k_embedding_gather_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    if cols == 0 || cols % 256 != 0 {
        return Err(format!(
            "Q4_K embedding cols must be non-zero and divisible by 256, got {cols}"
        ));
    }
    if token_ids.is_empty() {
        return Err("Q4_K embedding gather requires at least one token".to_string());
    }
    let blocks_per_row = cols / 256;
    let expected_weights = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(144))
        .ok_or_else(|| {
            format!(
                "Q4_K embedding weight size overflow: rows={rows} blocks_per_row={blocks_per_row}"
            )
        })?;
    if weights.len() != expected_weights {
        return Err(format!(
            "Q4_K embedding weight byte mismatch: got {}, expected {expected_weights}",
            weights.len()
        ));
    }
    for &token_id in token_ids {
        let token_idx = usize::try_from(token_id)
            .map_err(|_| format!("Q4_K embedding token id exceeds usize: {token_id}"))?;
        if token_idx >= rows {
            return Err(format!(
                "Q4_K embedding token id out of range: token={token_idx}, rows={rows}"
            ));
        }
    }

    let mut state = CudaState::open()?;
    let token_bytes = std::mem::size_of_val(token_ids);
    let output_len = token_ids.len().checked_mul(cols).ok_or_else(|| {
        format!(
            "Q4_K embedding output length overflow: tokens={} cols={cols}",
            token_ids.len()
        )
    })?;
    let output_bytes = output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("Q4_K embedding output byte overflow: len={output_len}"))?;
    let token_ids_dev = state.mem_alloc(token_bytes)?;
    let output_dev = state.mem_alloc(output_bytes)?;
    unsafe {
        state.api.memcpy_htod_async(
            token_ids_dev,
            token_ids.as_ptr().cast::<libc::c_void>(),
            token_bytes,
            state.stream,
        )?;
    }
    state.launch_q4k_embedding_gather_to_dev(
        weights,
        rows,
        blocks_per_row,
        token_ids_dev,
        token_ids.len(),
        output_dev,
    )?;
    let mut output = vec![0.0f32; output_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            output_bytes,
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(token_ids_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q5k_gemv_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q5k_gemv(weights, rows, cols, input)
}

#[cfg(test)]
pub fn iq4_xs_gemv_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    iq4_xs_gemv(weights, rows, cols, input)
}

#[cfg(test)]
pub fn iq4_xs_gemv_batch_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    iq4_xs_gemv_batch(weights, rows, cols, input)
}

#[cfg(test)]
pub fn q8_0_gemv_batch_device_input_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols == 0 || cols % 32 != 0 {
        return Err(format!(
            "Q8_0 device-input batch cols must be non-zero and divisible by 32, got {cols}"
        ));
    }
    if input.is_empty() || input.len() % cols != 0 {
        return Err(format!(
            "Q8_0 device-input batch input length {} is not a non-zero multiple of cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 32;
    let expected_weights = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(34))
        .ok_or_else(|| {
            format!("Q8_0 device-input batch weight size overflow: rows={rows} cols={cols}")
        })?;
    if weights.len() != expected_weights {
        return Err(format!(
            "Q8_0 device-input batch weight byte mismatch: got {}, expected {expected_weights}",
            weights.len()
        ));
    }

    let mut state = CudaState::open()?;
    let input_bytes = std::mem::size_of_val(input);
    let output_len = seq_len.checked_mul(rows).ok_or_else(|| {
        format!("Q8_0 device-input batch output length overflow: seq={seq_len} rows={rows}")
    })?;
    let output_bytes = output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("Q8_0 device-input batch output byte overflow: len={output_len}"))?;
    let input_dev = state.mem_alloc(input_bytes)?;
    let output_dev = state.mem_alloc(output_bytes)?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            input_bytes,
            state.stream,
        )?;
    }
    state.launch_q8_0_gemv_batch_to_dev(
        weights,
        rows,
        blocks_per_row,
        seq_len,
        input_dev,
        output_dev,
    )?;
    let mut output = vec![0.0f32; output_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            output_bytes,
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q8_0_f32_gemm_batch_cached_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols == 0 || cols % 32 != 0 {
        return Err(format!(
            "Q8_0 F32 GEMM batch cols must be non-zero and divisible by 32, got {cols}"
        ));
    }
    if input.is_empty() || input.len() % cols != 0 {
        return Err(format!(
            "Q8_0 F32 GEMM batch input length {} is not a non-zero multiple of cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 32;
    let expected_weights = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(34))
        .ok_or_else(|| {
            format!("Q8_0 F32 GEMM batch weight size overflow: rows={rows} cols={cols}")
        })?;
    if weights.len() != expected_weights {
        return Err(format!(
            "Q8_0 F32 GEMM batch weight byte mismatch: got {}, expected {expected_weights}",
            weights.len()
        ));
    }

    let mut state = CudaState::open()?;
    state.resident_q8_f32_limit = usize::MAX;
    state
        .q8_0_f32_gemm_batch_cached(weights, rows, blocks_per_row, seq_len, input)?
        .ok_or_else(|| "Q8_0 F32 GEMM batch unexpectedly unavailable".to_string())
}

#[cfg(test)]
pub fn q4k_gate_up_q8_for_test(
    gate_weights: &[u8],
    up_weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != cols {
        return Err(format!(
            "Q4_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = quantize_q8_1_by_32_for_test(input, blocks_per_row);
    let mut state = CudaState::open()?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let gate_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q4k_gate_up_gemv_q8dot_to_dev(
        gate_weights,
        up_weights,
        rows,
        blocks_per_row,
        qs_dev,
        ds_dev,
        gate_dev,
        up_dev,
    )?;
    let mut gate = vec![0.0f32; rows];
    let mut up = vec![0.0f32; rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            gate.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(gate.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            up.as_mut_ptr().cast::<libc::c_void>(),
            up_dev,
            std::mem::size_of_val(up.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
    }
    Ok((gate, up))
}

#[cfg(test)]
fn selected_pair_slots_for_test(expert_ids: &[u32]) -> Vec<u32> {
    const INVALID_SLOT: u32 = u32::MAX;
    const SKIP_SLOT: u32 = u32::MAX - 1;

    let slots_per_token = expert_ids.len() / 2;
    expert_ids
        .iter()
        .enumerate()
        .map(|(slot, expert)| {
            if slot < slots_per_token {
                expert_ids[slots_per_token..]
                    .iter()
                    .position(|second| second == expert)
                    .map(|second| (slots_per_token + second) as u32)
                    .unwrap_or(INVALID_SLOT)
            } else if expert_ids[..slots_per_token]
                .iter()
                .any(|first| first == expert)
            {
                SKIP_SLOT
            } else {
                INVALID_SLOT
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn q4k_selected_gate_up_pair2_for_test(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    expert_ids: &[u32],
    rows: usize,
    cols: usize,
    input: &[f32],
    paired: bool,
    fuse_silu: bool,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    let slots = expert_ids.len();
    if slots == 0 || slots % 2 != 0 {
        return Err(format!(
            "Q4_K pair2 selected gate/up requires two equal non-empty slot groups, got {slots}"
        ));
    }
    if gate_weights.len() != slots || up_weights.len() != slots {
        return Err(format!(
            "Q4_K pair2 selected gate/up slot mismatch: gate={} up={} expert={slots}",
            gate_weights.len(),
            up_weights.len()
        ));
    }
    if input.len() != 2 * cols {
        return Err(format!(
            "Q4_K pair2 selected gate/up input length mismatch: got {}, expected {}",
            input.len(),
            2 * cols
        ));
    }
    let slots_per_token = slots / 2;
    let blocks_per_row = cols / 256;
    let expected_weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|value| value.checked_mul(144))
        .ok_or_else(|| format!("Q4_K pair2 weight size overflow: rows={rows} cols={cols}"))?;
    for (slot, (gate, up)) in gate_weights.iter().zip(up_weights.iter()).enumerate() {
        if gate.len() != expected_weight_bytes || up.len() != expected_weight_bytes {
            return Err(format!(
                "Q4_K pair2 weight byte mismatch at slot {slot}: gate={} up={} expected={expected_weight_bytes}",
                gate.len(),
                up.len()
            ));
        }
    }
    for first in 0..slots_per_token {
        for second in slots_per_token..slots {
            if expert_ids[first] == expert_ids[second]
                && (gate_weights[first] != gate_weights[second]
                    || up_weights[first] != up_weights[second])
            {
                return Err(format!(
                    "Q4_K pair2 expert {} has inconsistent slot weights",
                    expert_ids[first]
                ));
            }
        }
    }
    let token_ids = (0..slots)
        .map(|slot| u32::from(slot >= slots_per_token))
        .collect::<Vec<_>>();
    let pair_slots = paired.then(|| selected_pair_slots_for_test(expert_ids));

    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let expert_ids_dev = state.mem_alloc(std::mem::size_of_val(expert_ids))?;
    let token_ids_dev = state.mem_alloc(std::mem::size_of_val(token_ids.as_slice()))?;
    let pair_slots_dev = if let Some(pair_slots) = pair_slots.as_ref() {
        state.mem_alloc(std::mem::size_of_val(pair_slots.as_slice()))?
    } else {
        0
    };
    let gate_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let up_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let gate_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let mut gate_weight_devs = Vec::with_capacity(slots);
    let mut up_weight_devs = Vec::with_capacity(slots);
    for (gate, up) in gate_weights.iter().zip(up_weights.iter()) {
        let gate_weight_dev = state.mem_alloc(gate.len())?;
        let up_weight_dev = state.mem_alloc(up.len())?;
        unsafe {
            state.api.memcpy_htod_async(
                gate_weight_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                gate.len(),
                state.stream,
            )?;
            state.api.memcpy_htod_async(
                up_weight_dev,
                up.as_ptr().cast::<libc::c_void>(),
                up.len(),
                state.stream,
            )?;
        }
        gate_weight_devs.push(gate_weight_dev);
        up_weight_devs.push(up_weight_dev);
    }
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            expert_ids_dev,
            expert_ids.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(expert_ids),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            token_ids_dev,
            token_ids.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(token_ids.as_slice()),
            state.stream,
        )?;
        if let Some(pair_slots) = pair_slots.as_ref() {
            state.api.memcpy_htod_async(
                pair_slots_dev,
                pair_slots.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(pair_slots.as_slice()),
                state.stream,
            )?;
        }
        state.api.memcpy_htod_async(
            gate_ptrs_dev,
            gate_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(gate_weight_devs.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            up_ptrs_dev,
            up_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(up_weight_devs.as_slice()),
            state.stream,
        )?;
    }

    if paired {
        state.launch_selected_q4k_gate_up_gemv_pair2_by_token_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            expert_ids_dev,
            pair_slots_dev,
            token_ids_dev,
            rows,
            slots_per_token,
            blocks_per_row,
            input_dev,
            gate_dev,
            up_dev,
            fuse_silu,
        )?;
    } else {
        state.launch_selected_q4k_gate_up_gemv_by_token_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            rows,
            slots,
            blocks_per_row,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        if fuse_silu {
            state.launch_silu_mul(gate_dev, up_dev, slots * rows)?;
        }
    }

    let mut gate = vec![0.0f32; slots * rows];
    let mut up = vec![0.0f32; slots * rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            gate.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(gate.as_slice()),
            state.stream,
        )?;
        if !fuse_silu {
            state.api.memcpy_dtoh_async(
                up.as_mut_ptr().cast::<libc::c_void>(),
                up_dev,
                std::mem::size_of_val(up.as_slice()),
                state.stream,
            )?;
        }
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(expert_ids_dev)?;
        state.api.mem_free(token_ids_dev)?;
        if pair_slots_dev != 0 {
            state.api.mem_free(pair_slots_dev)?;
        }
        state.api.mem_free(gate_ptrs_dev)?;
        state.api.mem_free(up_ptrs_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
        for ptr in gate_weight_devs {
            state.api.mem_free(ptr)?;
        }
        for ptr in up_weight_devs {
            state.api.mem_free(ptr)?;
        }
    }
    Ok((gate, up))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn q5k_selected_down_pair2_for_test(
    down_weights: &[&[u8]],
    expert_ids: &[u32],
    route: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
    paired: bool,
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q5_K cols must be multiple of 256, got {cols}"));
    }
    let slots = expert_ids.len();
    if slots == 0 || slots % 2 != 0 {
        return Err(format!(
            "Q5_K pair2 selected down requires two equal non-empty slot groups, got {slots}"
        ));
    }
    if down_weights.len() != slots || route.len() != slots {
        return Err(format!(
            "Q5_K pair2 selected down slot mismatch: down={} route={} expert={slots}",
            down_weights.len(),
            route.len()
        ));
    }
    if input.len() != slots * cols {
        return Err(format!(
            "Q5_K pair2 selected down input length mismatch: got {}, expected {}",
            input.len(),
            slots * cols
        ));
    }
    let slots_per_token = slots / 2;
    let blocks_per_row = cols / 256;
    let expected_weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|value| value.checked_mul(176))
        .ok_or_else(|| format!("Q5_K pair2 weight size overflow: rows={rows} cols={cols}"))?;
    for (slot, down) in down_weights.iter().enumerate() {
        if down.len() != expected_weight_bytes {
            return Err(format!(
                "Q5_K pair2 weight byte mismatch at slot {slot}: down={} expected={expected_weight_bytes}",
                down.len()
            ));
        }
    }
    for first in 0..slots_per_token {
        for second in slots_per_token..slots {
            if expert_ids[first] == expert_ids[second]
                && down_weights[first] != down_weights[second]
            {
                return Err(format!(
                    "Q5_K pair2 expert {} has inconsistent slot weights",
                    expert_ids[first]
                ));
            }
        }
    }
    let token_ids = (0..slots)
        .map(|slot| u32::from(slot >= slots_per_token))
        .collect::<Vec<_>>();
    let pair_slots = paired.then(|| selected_pair_slots_for_test(expert_ids));

    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let route_dev = state.mem_alloc(std::mem::size_of_val(route))?;
    let expert_ids_dev = state.mem_alloc(std::mem::size_of_val(expert_ids))?;
    let token_ids_dev = state.mem_alloc(std::mem::size_of_val(token_ids.as_slice()))?;
    let pair_slots_dev = if let Some(pair_slots) = pair_slots.as_ref() {
        state.mem_alloc(std::mem::size_of_val(pair_slots.as_slice()))?
    } else {
        0
    };
    let down_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let output_dev = state.mem_alloc(2 * rows * std::mem::size_of::<f32>())?;
    let mut down_weight_devs = Vec::with_capacity(slots);
    for down in down_weights {
        let down_weight_dev = state.mem_alloc(down.len())?;
        unsafe {
            state.api.memcpy_htod_async(
                down_weight_dev,
                down.as_ptr().cast::<libc::c_void>(),
                down.len(),
                state.stream,
            )?;
        }
        down_weight_devs.push(down_weight_dev);
    }
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            route_dev,
            route.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(route),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            expert_ids_dev,
            expert_ids.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(expert_ids),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            token_ids_dev,
            token_ids.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(token_ids.as_slice()),
            state.stream,
        )?;
        if let Some(pair_slots) = pair_slots.as_ref() {
            state.api.memcpy_htod_async(
                pair_slots_dev,
                pair_slots.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(pair_slots.as_slice()),
                state.stream,
            )?;
        }
        state.api.memcpy_htod_async(
            down_ptrs_dev,
            down_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(down_weight_devs.as_slice()),
            state.stream,
        )?;
    }
    state.launch_zero_f32(output_dev, 2 * rows)?;
    if paired {
        state.launch_selected_q5k_down_accum_by_token_pair2_warp4(
            down_ptrs_dev,
            token_ids_dev,
            expert_ids_dev,
            pair_slots_dev,
            rows,
            slots_per_token,
            blocks_per_row,
            input_dev,
            route_dev,
            output_dev,
        )?;
    } else {
        state.launch_selected_down_accum_by_token_warp4(
            "rnb_q5k_selected_down_accum_by_token_warp4",
            down_ptrs_dev,
            token_ids_dev,
            rows,
            slots,
            blocks_per_row,
            input_dev,
            route_dev,
            output_dev,
        )?;
    }

    let mut output = vec![0.0f32; 2 * rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(route_dev)?;
        state.api.mem_free(expert_ids_dev)?;
        state.api.mem_free(token_ids_dev)?;
        if pair_slots_dev != 0 {
            state.api.mem_free(pair_slots_dev)?;
        }
        state.api.mem_free(down_ptrs_dev)?;
        state.api.mem_free(output_dev)?;
        for ptr in down_weight_devs {
            state.api.mem_free(ptr)?;
        }
    }
    Ok(output)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn q4k_selected_gate_up_silu_group8_for_test(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    token_ids: &[u32],
    group_meta: &[u32],
    rows: usize,
    cols: usize,
    input: &[f32],
    fused: bool,
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    let slots = token_ids.len();
    if slots == 0 || gate_weights.len() != slots || up_weights.len() != slots {
        return Err(format!(
            "Q4_K selected gate/up slot mismatch: gate={} up={} token={slots}",
            gate_weights.len(),
            up_weights.len()
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K selected gate/up input length {} is not a multiple of cols {cols}",
            input.len()
        ));
    }
    if group_meta.len() % 2 != 0 {
        return Err(format!(
            "Q4_K selected gate/up group meta length must be even, got {}",
            group_meta.len()
        ));
    }
    let token_count = input.len() / cols;
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err(format!(
            "Q4_K selected gate/up token id exceeds token count {token_count}"
        ));
    }
    let blocks_per_row = cols / 256;
    let expected_weight_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(144))
        .ok_or_else(|| {
            format!("Q4_K selected gate/up weight size overflow: rows={rows} cols={cols}")
        })?;
    for (slot, (gate, up)) in gate_weights.iter().zip(up_weights.iter()).enumerate() {
        if gate.len() != expected_weight_bytes || up.len() != expected_weight_bytes {
            return Err(format!(
                "Q4_K selected gate/up weight byte mismatch at slot {slot}: gate={} up={} expected={expected_weight_bytes}",
                gate.len(),
                up.len()
            ));
        }
    }

    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let token_ids_dev = state.mem_alloc(std::mem::size_of_val(token_ids))?;
    let group_meta_dev = state.mem_alloc(std::mem::size_of_val(group_meta))?;
    let gate_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let up_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let gate_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let mut gate_weight_devs = Vec::with_capacity(slots);
    let mut up_weight_devs = Vec::with_capacity(slots);
    for (gate, up) in gate_weights.iter().zip(up_weights.iter()) {
        let gate_dev = state.mem_alloc(gate.len())?;
        let up_dev = state.mem_alloc(up.len())?;
        unsafe {
            state.api.memcpy_htod_async(
                gate_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                gate.len(),
                state.stream,
            )?;
            state.api.memcpy_htod_async(
                up_dev,
                up.as_ptr().cast::<libc::c_void>(),
                up.len(),
                state.stream,
            )?;
        }
        gate_weight_devs.push(gate_dev);
        up_weight_devs.push(up_dev);
    }

    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            token_ids_dev,
            token_ids.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(token_ids),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            group_meta_dev,
            group_meta.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(group_meta),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            gate_ptrs_dev,
            gate_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(gate_weight_devs.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            up_ptrs_dev,
            up_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(up_weight_devs.as_slice()),
            state.stream,
        )?;
    }

    if fused {
        state.launch_selected_q4k_gate_up_silu_by_token_group8_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            group_meta_dev,
            rows,
            group_meta.len() / 2,
            blocks_per_row,
            input_dev,
            gate_dev,
            up_dev,
        )?;
    } else {
        state.launch_selected_q4k_gate_up_gemv_by_token_group4_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            group_meta_dev,
            rows,
            group_meta.len() / 2,
            blocks_per_row,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        state.launch_silu_mul(gate_dev, up_dev, slots * rows)?;
    }

    let mut gate = vec![0.0f32; slots * rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            gate.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(gate.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(token_ids_dev)?;
        state.api.mem_free(group_meta_dev)?;
        state.api.mem_free(gate_ptrs_dev)?;
        state.api.mem_free(up_ptrs_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
        for ptr in gate_weight_devs {
            state.api.mem_free(ptr)?;
        }
        for ptr in up_weight_devs {
            state.api.mem_free(ptr)?;
        }
    }
    Ok(gate)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn q4k_selected_gate_up_silu_pack4_for_test(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    token_ids: &[u32],
    group_meta: &[u32],
    rows: usize,
    cols: usize,
    input: &[f32],
    fused: bool,
) -> Result<Vec<f32>, String> {
    if rows % 256 != 0 || cols % 256 != 0 {
        return Err(format!(
            "Q4_K pack4 gate/up dims must be multiples of 256, got rows={rows} cols={cols}"
        ));
    }
    let slots = token_ids.len();
    if slots == 0 || gate_weights.len() != slots || up_weights.len() != slots {
        return Err(format!(
            "Q4_K pack4 gate/up slot mismatch: gate={} up={} token={slots}",
            gate_weights.len(),
            up_weights.len()
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K pack4 gate/up input length {} is not a multiple of cols {cols}",
            input.len()
        ));
    }
    if group_meta.len() % 2 != 0 {
        return Err(format!(
            "Q4_K pack4 gate/up group meta length must be even, got {}",
            group_meta.len()
        ));
    }
    if group_meta
        .chunks_exact(2)
        .any(|chunk| chunk[1] == 0 || chunk[1] > 4)
    {
        return Err("Q4_K pack4 gate/up group meta requires group lengths 1..=4".to_string());
    }
    let token_count = input.len() / cols;
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err(format!(
            "Q4_K pack4 gate/up token id exceeds token count {token_count}"
        ));
    }
    let input_blocks_per_row = cols / 256;
    let pack_blocks_per_row = rows / 256;
    let expected_weight_bytes = rows
        .checked_mul(input_blocks_per_row)
        .and_then(|v| v.checked_mul(144))
        .ok_or_else(|| {
            format!("Q4_K pack4 gate/up weight size overflow: rows={rows} cols={cols}")
        })?;
    for (slot, (gate, up)) in gate_weights.iter().zip(up_weights.iter()).enumerate() {
        if gate.len() != expected_weight_bytes || up.len() != expected_weight_bytes {
            return Err(format!(
                "Q4_K pack4 gate/up weight byte mismatch at slot {slot}: gate={} up={} expected={expected_weight_bytes}",
                gate.len(),
                up.len()
            ));
        }
    }

    let groups = group_meta.len() / 2;
    let packed_len = groups
        .checked_mul(4)
        .and_then(|value| value.checked_mul(rows))
        .ok_or_else(|| {
            format!("Q4_K pack4 gate/up activation size overflow: groups={groups} rows={rows}")
        })?;

    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let token_ids_dev = state.mem_alloc(std::mem::size_of_val(token_ids))?;
    let group_meta_dev = state.mem_alloc(std::mem::size_of_val(group_meta))?;
    let gate_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let up_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let gate_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let packed_dev = state.mem_alloc(packed_len * std::mem::size_of::<f32>())?;
    let mut gate_weight_devs = Vec::with_capacity(slots);
    let mut up_weight_devs = Vec::with_capacity(slots);
    for (gate, up) in gate_weights.iter().zip(up_weights.iter()) {
        let gate_dev = state.mem_alloc(gate.len())?;
        let up_dev = state.mem_alloc(up.len())?;
        unsafe {
            state.api.memcpy_htod_async(
                gate_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                gate.len(),
                state.stream,
            )?;
            state.api.memcpy_htod_async(
                up_dev,
                up.as_ptr().cast::<libc::c_void>(),
                up.len(),
                state.stream,
            )?;
        }
        gate_weight_devs.push(gate_dev);
        up_weight_devs.push(up_dev);
    }

    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            token_ids_dev,
            token_ids.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(token_ids),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            group_meta_dev,
            group_meta.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(group_meta),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            gate_ptrs_dev,
            gate_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(gate_weight_devs.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            up_ptrs_dev,
            up_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(up_weight_devs.as_slice()),
            state.stream,
        )?;
    }

    state.launch_zero_f32(packed_dev, packed_len)?;
    if fused {
        state.launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group4_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            group_meta_dev,
            rows,
            groups,
            input_blocks_per_row,
            pack_blocks_per_row,
            input_dev,
            packed_dev,
        )?;
    } else {
        state.launch_selected_q4k_gate_up_gemv_by_token_group4_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            group_meta_dev,
            rows,
            groups,
            input_blocks_per_row,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        state.launch_silu_mul_group4_pack_f32(
            gate_dev,
            up_dev,
            packed_dev,
            group_meta_dev,
            groups,
            pack_blocks_per_row,
        )?;
    }

    let mut packed = vec![0.0f32; packed_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            packed.as_mut_ptr().cast::<libc::c_void>(),
            packed_dev,
            std::mem::size_of_val(packed.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(token_ids_dev)?;
        state.api.mem_free(group_meta_dev)?;
        state.api.mem_free(gate_ptrs_dev)?;
        state.api.mem_free(up_ptrs_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
        state.api.mem_free(packed_dev)?;
        for ptr in gate_weight_devs {
            state.api.mem_free(ptr)?;
        }
        for ptr in up_weight_devs {
            state.api.mem_free(ptr)?;
        }
    }
    Ok(packed)
}

#[cfg(test)]
fn pack4_group_offsets_from_group_meta_for_test(group_meta: &[u32]) -> Result<Vec<u32>, String> {
    if group_meta.len() % 2 != 0 {
        return Err(format!(
            "Q4_K group8 pack4 group meta length must be even, got {}",
            group_meta.len()
        ));
    }
    let mut offsets = Vec::with_capacity(group_meta.len() / 2 + 1);
    let mut next = 0u32;
    offsets.push(next);
    for chunk in group_meta.chunks_exact(2) {
        let len = chunk[1];
        if len == 0 || len > 8 {
            return Err(format!(
                "Q4_K group8 pack4 group length must be 1..=8, got {len}"
            ));
        }
        next = next
            .checked_add((len + 3) / 4)
            .ok_or_else(|| "Q4_K group8 pack4 group offset overflow".to_string())?;
        offsets.push(next);
    }
    Ok(offsets)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn q4k_selected_gate_up_silu_pack4_group8_for_test(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    token_ids: &[u32],
    gate_up_group_meta: &[u32],
    down_group_meta: &[u32],
    rows: usize,
    cols: usize,
    input: &[f32],
    fused: bool,
) -> Result<Vec<f32>, String> {
    if rows % 256 != 0 || cols % 256 != 0 {
        return Err(format!(
            "Q4_K group8 pack4 gate/up dims must be multiples of 256, got rows={rows} cols={cols}"
        ));
    }
    let slots = token_ids.len();
    if slots == 0 || gate_weights.len() != slots || up_weights.len() != slots {
        return Err(format!(
            "Q4_K group8 pack4 gate/up slot mismatch: gate={} up={} token={slots}",
            gate_weights.len(),
            up_weights.len()
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K group8 pack4 gate/up input length {} is not a multiple of cols {cols}",
            input.len()
        ));
    }
    if down_group_meta.len() % 2 != 0 {
        return Err(format!(
            "Q4_K group8 pack4 down meta length must be even, got {}",
            down_group_meta.len()
        ));
    }
    if down_group_meta
        .chunks_exact(2)
        .any(|chunk| chunk[1] == 0 || chunk[1] > 4)
    {
        return Err("Q4_K group8 pack4 down meta requires group lengths 1..=4".to_string());
    }
    let pack_group_offsets = pack4_group_offsets_from_group_meta_for_test(gate_up_group_meta)?;
    let down_groups = down_group_meta.len() / 2;
    if pack_group_offsets.last().copied().unwrap_or(0) as usize != down_groups {
        return Err(format!(
            "Q4_K group8 pack4 offset/down group mismatch: offsets={} down_groups={down_groups}",
            pack_group_offsets.last().copied().unwrap_or(0)
        ));
    }
    let token_count = input.len() / cols;
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err(format!(
            "Q4_K group8 pack4 gate/up token id exceeds token count {token_count}"
        ));
    }
    let input_blocks_per_row = cols / 256;
    let pack_blocks_per_row = rows / 256;
    let expected_weight_bytes = rows
        .checked_mul(input_blocks_per_row)
        .and_then(|v| v.checked_mul(144))
        .ok_or_else(|| {
            format!("Q4_K group8 pack4 gate/up weight size overflow: rows={rows} cols={cols}")
        })?;
    for (slot, (gate, up)) in gate_weights.iter().zip(up_weights.iter()).enumerate() {
        if gate.len() != expected_weight_bytes || up.len() != expected_weight_bytes {
            return Err(format!(
                "Q4_K group8 pack4 gate/up weight byte mismatch at slot {slot}: gate={} up={} expected={expected_weight_bytes}",
                gate.len(),
                up.len()
            ));
        }
    }

    let packed_len = down_groups
        .checked_mul(4)
        .and_then(|value| value.checked_mul(rows))
        .ok_or_else(|| {
            format!("Q4_K group8 pack4 activation size overflow: groups={down_groups} rows={rows}")
        })?;

    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let token_ids_dev = state.mem_alloc(std::mem::size_of_val(token_ids))?;
    let gate_up_group_meta_dev = state.mem_alloc(std::mem::size_of_val(gate_up_group_meta))?;
    let down_group_meta_dev = state.mem_alloc(std::mem::size_of_val(down_group_meta))?;
    let pack_group_offsets_dev =
        state.mem_alloc(std::mem::size_of_val(pack_group_offsets.as_slice()))?;
    let gate_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let up_ptrs_dev = state.mem_alloc(slots * std::mem::size_of::<u64>())?;
    let gate_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(slots * rows * std::mem::size_of::<f32>())?;
    let packed_dev = state.mem_alloc(packed_len * std::mem::size_of::<f32>())?;
    let mut gate_weight_devs = Vec::with_capacity(slots);
    let mut up_weight_devs = Vec::with_capacity(slots);
    for (gate, up) in gate_weights.iter().zip(up_weights.iter()) {
        let gate_dev = state.mem_alloc(gate.len())?;
        let up_dev = state.mem_alloc(up.len())?;
        unsafe {
            state.api.memcpy_htod_async(
                gate_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                gate.len(),
                state.stream,
            )?;
            state.api.memcpy_htod_async(
                up_dev,
                up.as_ptr().cast::<libc::c_void>(),
                up.len(),
                state.stream,
            )?;
        }
        gate_weight_devs.push(gate_dev);
        up_weight_devs.push(up_dev);
    }

    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            token_ids_dev,
            token_ids.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(token_ids),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            gate_up_group_meta_dev,
            gate_up_group_meta.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(gate_up_group_meta),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            down_group_meta_dev,
            down_group_meta.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(down_group_meta),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            pack_group_offsets_dev,
            pack_group_offsets.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(pack_group_offsets.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            gate_ptrs_dev,
            gate_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(gate_weight_devs.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            up_ptrs_dev,
            up_weight_devs.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(up_weight_devs.as_slice()),
            state.stream,
        )?;
    }

    state.launch_zero_f32(packed_dev, packed_len)?;
    if fused {
        state.launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group8_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            gate_up_group_meta_dev,
            pack_group_offsets_dev,
            rows,
            gate_up_group_meta.len() / 2,
            input_blocks_per_row,
            pack_blocks_per_row,
            input_dev,
            packed_dev,
        )?;
    } else {
        state.launch_selected_q4k_gate_up_gemv_by_token_group4_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            gate_up_group_meta_dev,
            rows,
            gate_up_group_meta.len() / 2,
            input_blocks_per_row,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        state.launch_silu_mul_group4_pack_f32(
            gate_dev,
            up_dev,
            packed_dev,
            down_group_meta_dev,
            down_groups,
            pack_blocks_per_row,
        )?;
    }

    let mut packed = vec![0.0f32; packed_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            packed.as_mut_ptr().cast::<libc::c_void>(),
            packed_dev,
            std::mem::size_of_val(packed.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(token_ids_dev)?;
        state.api.mem_free(gate_up_group_meta_dev)?;
        state.api.mem_free(down_group_meta_dev)?;
        state.api.mem_free(pack_group_offsets_dev)?;
        state.api.mem_free(gate_ptrs_dev)?;
        state.api.mem_free(up_ptrs_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
        state.api.mem_free(packed_dev)?;
        for ptr in gate_weight_devs {
            state.api.mem_free(ptr)?;
        }
        for ptr in up_weight_devs {
            state.api.mem_free(ptr)?;
        }
    }
    Ok(packed)
}

#[cfg(test)]
pub fn q4k_gate_up_batch_seq2_q8_for_test(
    gate_weights: &[u8],
    up_weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != 2 * cols {
        return Err(format!(
            "Q4_K seq2 input length mismatch: got {}, expected {}",
            input.len(),
            2 * cols
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = super::gemv::quantize_q8_1_batch_by_32(input, blocks_per_row, 2);
    let mut state = CudaState::open()?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let gate_dev = state.mem_alloc(2 * rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(2 * rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q4k_gate_up_gemv_batch_seq2_q8dot_to_dev(
        gate_weights,
        up_weights,
        rows,
        blocks_per_row,
        qs_dev,
        ds_dev,
        gate_dev,
        up_dev,
    )?;
    let mut gate = vec![0.0f32; 2 * rows];
    let mut up = vec![0.0f32; 2 * rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            gate.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(gate.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            up.as_mut_ptr().cast::<libc::c_void>(),
            up_dev,
            std::mem::size_of_val(up.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
    }
    Ok((gate, up))
}

#[cfg(test)]
pub fn q4k_packed_gate_up_q8_for_test(
    gate_weights: &[u8],
    up_weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != cols {
        return Err(format!(
            "Q4_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = quantize_q8_1_by_32_for_test(input, blocks_per_row);
    let mut state = CudaState::open()?;
    state.resident_q4_packed_limit = usize::MAX;
    let gate_packed_dev = state
        .resident_q4k_packed_ptrs(gate_weights, rows, blocks_per_row)?
        .ok_or_else(|| "Q4 packed gate cache disabled in test".to_string())?;
    let up_packed_dev = state
        .resident_q4k_packed_ptrs(up_weights, rows, blocks_per_row)?
        .ok_or_else(|| "Q4 packed up cache disabled in test".to_string())?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let gate_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q4k_packed_gate_up_gemv_q8dot_to_dev(
        gate_packed_dev,
        up_packed_dev,
        rows,
        blocks_per_row,
        qs_dev,
        ds_dev,
        gate_dev,
        up_dev,
    )?;
    let mut gate = vec![0.0f32; rows];
    let mut up = vec![0.0f32; rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            gate.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(gate.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            up.as_mut_ptr().cast::<libc::c_void>(),
            up_dev,
            std::mem::size_of_val(up.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
    }
    Ok((gate, up))
}

#[cfg(test)]
pub fn q4k_packed_gate_up_batch_seq2_q8_for_test(
    gate_weights: &[u8],
    up_weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != 2 * cols {
        return Err(format!(
            "Q4_K seq2 input length mismatch: got {}, expected {}",
            input.len(),
            2 * cols
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = super::gemv::quantize_q8_1_batch_by_32(input, blocks_per_row, 2);
    let mut state = CudaState::open()?;
    state.resident_q4_packed_limit = usize::MAX;
    let gate_packed_dev = state
        .resident_q4k_packed_ptrs(gate_weights, rows, blocks_per_row)?
        .ok_or_else(|| "Q4 packed gate cache disabled in test".to_string())?;
    let up_packed_dev = state
        .resident_q4k_packed_ptrs(up_weights, rows, blocks_per_row)?
        .ok_or_else(|| "Q4 packed up cache disabled in test".to_string())?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let gate_dev = state.mem_alloc(2 * rows * std::mem::size_of::<f32>())?;
    let up_dev = state.mem_alloc(2 * rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q4k_packed_gate_up_gemv_batch_seq2_q8dot_to_dev(
        gate_packed_dev,
        up_packed_dev,
        rows,
        blocks_per_row,
        qs_dev,
        ds_dev,
        gate_dev,
        up_dev,
    )?;
    let mut gate = vec![0.0f32; 2 * rows];
    let mut up = vec![0.0f32; 2 * rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            gate.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(gate.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            up.as_mut_ptr().cast::<libc::c_void>(),
            up_dev,
            std::mem::size_of_val(up.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
    }
    Ok((gate, up))
}

#[cfg(test)]
pub fn rms_norm_add_then_rms_norm_for_test(
    input: &[f32],
    post_weight: &[f32],
    residual: &[f32],
    pre_weight: &[f32],
    eps: f32,
    post_unit_offset: bool,
    pre_unit_offset: bool,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if input.len() != residual.len()
        || input.len() != post_weight.len()
        || input.len() != pre_weight.len()
    {
        return Err(format!(
            "combined norm length mismatch: input={} post={} residual={} pre={}",
            input.len(),
            post_weight.len(),
            residual.len(),
            pre_weight.len()
        ));
    }
    let len = input.len();
    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let post_weight_dev = state.mem_alloc(std::mem::size_of_val(post_weight))?;
    let residual_dev = state.mem_alloc(std::mem::size_of_val(residual))?;
    let pre_weight_dev = state.mem_alloc(std::mem::size_of_val(pre_weight))?;
    let output_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            post_weight_dev,
            post_weight.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(post_weight),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            residual_dev,
            residual.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(residual),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            pre_weight_dev,
            pre_weight.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(pre_weight),
            state.stream,
        )?;
    }
    state.launch_rms_norm_add_then_rms_norm_f32(
        input_dev,
        post_weight_dev,
        residual_dev,
        pre_weight_dev,
        output_dev,
        eps,
        len,
        post_unit_offset,
        pre_unit_offset,
    )?;
    let mut updated = vec![0.0f32; len];
    let mut output = vec![0.0f32; len];
    unsafe {
        state.api.memcpy_dtoh_async(
            updated.as_mut_ptr().cast::<libc::c_void>(),
            residual_dev,
            std::mem::size_of_val(updated.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(post_weight_dev)?;
        state.api.mem_free(residual_dev)?;
        state.api.mem_free(pre_weight_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok((updated, output))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn mtp_build_eh_input_for_test(
    token_rows: &[f32],
    target_hidden_rows: &[f32],
    enorm: &[f32],
    hnorm: &[f32],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let expected = rows
        .checked_mul(hidden_dim)
        .ok_or_else(|| format!("MTP EH input length overflow: rows={rows} hidden={hidden_dim}"))?;
    if token_rows.len() != expected || target_hidden_rows.len() != expected {
        return Err(format!(
            "MTP EH input row length mismatch: token={} hidden={} expected={expected}",
            token_rows.len(),
            target_hidden_rows.len()
        ));
    }
    if enorm.len() != hidden_dim || hnorm.len() != hidden_dim {
        return Err(format!(
            "MTP EH norm length mismatch: enorm={} hnorm={} expected={hidden_dim}",
            enorm.len(),
            hnorm.len()
        ));
    }
    let output_len = expected
        .checked_mul(2)
        .ok_or_else(|| format!("MTP EH output length overflow: input={expected}"))?;

    let mut state = CudaState::open()?;
    let token_dev = state.mem_alloc(std::mem::size_of_val(token_rows))?;
    let hidden_dev = state.mem_alloc(std::mem::size_of_val(target_hidden_rows))?;
    let enorm_dev = state.mem_alloc(std::mem::size_of_val(enorm))?;
    let hnorm_dev = state.mem_alloc(std::mem::size_of_val(hnorm))?;
    let output_dev = state.mem_alloc(output_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            token_dev,
            token_rows.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(token_rows),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            hidden_dev,
            target_hidden_rows.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(target_hidden_rows),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            enorm_dev,
            enorm.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(enorm),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            hnorm_dev,
            hnorm.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(hnorm),
            state.stream,
        )?;
    }
    state.launch_mtp_build_eh_input_f32(
        token_dev, hidden_dev, enorm_dev, hnorm_dev, output_dev, eps, rows, hidden_dim,
    )?;
    let mut output = vec![0.0f32; output_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(token_dev)?;
        state.api.mem_free(hidden_dev)?;
        state.api.mem_free(enorm_dev)?;
        state.api.mem_free(hnorm_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
#[allow(clippy::type_complexity)]
pub fn rms_norm_add_then_rms_norm_q8_for_test(
    input: &[f32],
    post_weight: &[f32],
    residual: &[f32],
    pre_weight: &[f32],
    eps: f32,
    post_unit_offset: bool,
    pre_unit_offset: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<i8>, Vec<f32>), String> {
    if input.len() != residual.len()
        || input.len() != post_weight.len()
        || input.len() != pre_weight.len()
    {
        return Err(format!(
            "combined norm Q8 length mismatch: input={} post={} residual={} pre={}",
            input.len(),
            post_weight.len(),
            residual.len(),
            pre_weight.len()
        ));
    }
    if input.len() % 32 != 0 {
        return Err(format!(
            "combined norm Q8 length must be divisible by 32, got {}",
            input.len()
        ));
    }
    let len = input.len();
    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let post_weight_dev = state.mem_alloc(std::mem::size_of_val(post_weight))?;
    let residual_dev = state.mem_alloc(std::mem::size_of_val(residual))?;
    let pre_weight_dev = state.mem_alloc(std::mem::size_of_val(pre_weight))?;
    let output_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let qs_dev = state.mem_alloc(len)?;
    let ds_len = len / 32;
    let ds_dev = state.mem_alloc(ds_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            post_weight_dev,
            post_weight.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(post_weight),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            residual_dev,
            residual.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(residual),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            pre_weight_dev,
            pre_weight.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(pre_weight),
            state.stream,
        )?;
    }
    state.launch_rms_norm_add_then_rms_norm_q8_1_f32(
        input_dev,
        post_weight_dev,
        residual_dev,
        pre_weight_dev,
        output_dev,
        qs_dev,
        ds_dev,
        eps,
        len,
        post_unit_offset,
        pre_unit_offset,
    )?;
    let mut updated = vec![0.0f32; len];
    let mut output = vec![0.0f32; len];
    let mut qs = vec![0i8; len];
    let mut ds = vec![0.0f32; ds_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            updated.as_mut_ptr().cast::<libc::c_void>(),
            residual_dev,
            std::mem::size_of_val(updated.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            qs.as_mut_ptr().cast::<libc::c_void>(),
            qs_dev,
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            ds.as_mut_ptr().cast::<libc::c_void>(),
            ds_dev,
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(post_weight_dev)?;
        state.api.mem_free(residual_dev)?;
        state.api.mem_free(pre_weight_dev)?;
        state.api.mem_free(output_dev)?;
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
    }
    Ok((updated, output, qs, ds))
}

#[cfg(test)]
pub fn gemma_ple_base_slice_for_test(
    base: &[f32],
    offset: usize,
    len: usize,
) -> Result<Vec<f32>, String> {
    let mut state = CudaState::open()?;
    state.upload_gemma_ple_base(base)?;
    let slice_dev = state.gemma_ple_base_slice_ptr(offset, len)?;
    let mut output = vec![0.0f32; len];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            slice_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    Ok(output)
}

#[cfg(test)]
pub fn q4k_gemv_q8_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    qk_gemv_q8_for_test(weights, rows, cols, input, false)
}

#[cfg(test)]
pub fn q4k_gemv_gelu_mul_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    mul: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != cols {
        return Err(format!(
            "Q4_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    if mul.len() != rows {
        return Err(format!(
            "Q4_K GELU mul length mismatch: got {}, expected {rows}",
            mul.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let mut state = CudaState::open()?;
    let input_dev = state.mem_alloc(std::mem::size_of_val(input))?;
    let mul_dev = state.mem_alloc(std::mem::size_of_val(mul))?;
    let output_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(input),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            mul_dev,
            mul.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(mul),
            state.stream,
        )?;
    }
    state.launch_q4k_gemv_gelu_mul_to_dev(
        weights,
        rows,
        blocks_per_row,
        input_dev,
        mul_dev,
        output_dev,
    )?;
    let mut output = vec![0.0f32; rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(input_dev)?;
        state.api.mem_free(mul_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
fn q4k_packed_q8_with_resident_weight_for_test<F>(
    rows: usize,
    cols: usize,
    input: &[f32],
    resident_weight: F,
) -> Result<Vec<f32>, String>
where
    F: FnOnce(&mut CudaState, usize) -> Result<Option<u64>, String>,
{
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != cols {
        return Err(format!(
            "Q4_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = quantize_q8_1_by_32_for_test(input, blocks_per_row);
    let mut state = CudaState::open()?;
    state.resident_q4_packed_limit = usize::MAX;
    let weights_dev = resident_weight(&mut state, blocks_per_row)?
        .ok_or_else(|| "Q4 packed cache disabled in test".to_string())?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let output_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q4k_packed_gemv_q8dot_to_dev(
        weights_dev,
        rows,
        blocks_per_row,
        qs_dev,
        ds_dev,
        output_dev,
    )?;
    let mut output = vec![0.0f32; rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q4k_packed_q8_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q4k_packed_q8_with_resident_weight_for_test(rows, cols, input, |state, blocks_per_row| {
        state.resident_q4k_packed_ptrs(weights, rows, blocks_per_row)
    })
}

#[cfg(test)]
pub fn q4k_packed_q8_view_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let view = rnb_backend_api::TransformedWeightView::new(
        rnb_backend_api::TransformedWeightLayout::Q4kCompactMetadata,
        rnb_backend_api::TransformedSourceQuant::DenseQ4kRowPair,
        rows,
        cols,
        0x5134_4B51_344B_0001,
        1,
        256,
        0x5134_5041_434B_0001,
        weights,
    )
    .map_err(|error| format!("{error:?}"))?;

    q4k_packed_q8_with_resident_weight_for_test(rows, cols, input, |state, _blocks_per_row| {
        state.resident_q4k_transformed_view_ptr(view)
    })
}

#[cfg(test)]
pub fn q4k_packed_q8_payload_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let payload = pack_q4k_for_q8dot(weights, rows, blocks_per_row)?;
    q4k_packed_q8_with_resident_weight_for_test(rows, cols, input, |state, blocks_per_row| {
        state.resident_q4k_packed_payload_ptr(weights, &payload, rows, blocks_per_row)
    })
}

#[cfg(test)]
pub fn q4k_qkv_q8_for_test(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != cols {
        return Err(format!(
            "Q4_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = quantize_q8_1_by_32_for_test(input, blocks_per_row);
    let mut state = CudaState::open()?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let q_dev = state.mem_alloc(q_rows * std::mem::size_of::<f32>())?;
    let k_dev = state.mem_alloc(kv_rows * std::mem::size_of::<f32>())?;
    let v_dev = state.mem_alloc(kv_rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q4k_qkv_gemv_q8dot_to_dev(
        q_weights,
        k_weights,
        v_weights,
        q_rows,
        kv_rows,
        blocks_per_row,
        qs_dev,
        ds_dev,
        q_dev,
        k_dev,
        v_dev,
    )?;
    let mut q = vec![0.0f32; q_rows];
    let mut k = vec![0.0f32; kv_rows];
    let mut v = vec![0.0f32; kv_rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            q.as_mut_ptr().cast::<libc::c_void>(),
            q_dev,
            std::mem::size_of_val(q.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            k.as_mut_ptr().cast::<libc::c_void>(),
            k_dev,
            std::mem::size_of_val(k.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            v.as_mut_ptr().cast::<libc::c_void>(),
            v_dev,
            std::mem::size_of_val(v.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(q_dev)?;
        state.api.mem_free(k_dev)?;
        state.api.mem_free(v_dev)?;
    }
    Ok((q, k, v))
}

#[cfg(test)]
pub fn q6k_gemv_q8_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    qk_gemv_q8_for_test(weights, rows, cols, input, true)
}

#[cfg(test)]
fn q6k_packed_q8_with_resident_weight_for_test<F>(
    rows: usize,
    cols: usize,
    input: &[f32],
    resident_weight: F,
) -> Result<Vec<f32>, String>
where
    F: FnOnce(&mut CudaState, usize) -> Result<Option<(u64, u64, u64)>, String>,
{
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != cols {
        return Err(format!(
            "Q6_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = quantize_q8_1_by_32_for_test(input, blocks_per_row);
    let mut state = CudaState::open()?;
    state.resident_q6_packed_limit = usize::MAX;
    let (packed_qs_dev, packed_d_super_dev, packed_sub_scale_dev) =
        resident_weight(&mut state, blocks_per_row)?
            .ok_or_else(|| "Q6 packed cache disabled in test".to_string())?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let output_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q6k_packed_q8dot_to_dev(
        packed_qs_dev,
        packed_d_super_dev,
        packed_sub_scale_dev,
        rows,
        blocks_per_row,
        qs_dev,
        ds_dev,
        output_dev,
    )?;
    let mut output = vec![0.0f32; rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn q6k_packed_q8_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q6k_packed_q8_with_resident_weight_for_test(rows, cols, input, |state, blocks_per_row| {
        state.resident_q6k_packed_ptrs(weights, rows, blocks_per_row)
    })
}

#[cfg(test)]
pub fn q6k_packed_q8_view_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let view = rnb_backend_api::TransformedWeightView::new(
        rnb_backend_api::TransformedWeightLayout::Q6kPackedQ8dot,
        rnb_backend_api::TransformedSourceQuant::DenseQ6k,
        rows,
        cols,
        0x5136_4B51_364B_0001,
        1,
        256,
        0x5136_5041_434B_0001,
        weights,
    )
    .map_err(|error| format!("{error:?}"))?;

    q6k_packed_q8_with_resident_weight_for_test(rows, cols, input, |state, _blocks_per_row| {
        state.resident_q6k_transformed_view_ptrs(view)
    })
}

#[cfg(test)]
pub fn q6k_packed_q8_payload_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be multiple of 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let (qs, d_super, sub_scale) = pack_q6k_for_q8dot(weights, rows, blocks_per_row)?;
    let mut payload = Vec::with_capacity(
        qs.len()
            + d_super.len() * std::mem::size_of::<u16>()
            + sub_scale.len() * std::mem::size_of::<i8>(),
    );
    payload.extend(qs.iter().map(|v| *v as u8));
    for value in d_super {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    payload.extend(sub_scale.iter().map(|v| *v as u8));

    q6k_packed_q8_with_resident_weight_for_test(rows, cols, input, |state, blocks_per_row| {
        state.resident_q6k_packed_payload_ptrs(weights, &payload, rows, blocks_per_row)
    })
}

#[cfg(test)]
pub fn q6k_packed_batch_q8_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != seq_len.saturating_mul(cols) {
        return Err(format!(
            "Q6_K input length mismatch: got {}, expected {}",
            input.len(),
            seq_len.saturating_mul(cols)
        ));
    }
    let blocks_per_row = cols / 256;
    let mut qs = Vec::with_capacity(seq_len * blocks_per_row * 256);
    let mut ds = Vec::with_capacity(seq_len * blocks_per_row * 8);
    for seq in 0..seq_len {
        let (seq_qs, seq_ds) =
            quantize_q8_1_by_32_for_test(&input[seq * cols..(seq + 1) * cols], blocks_per_row);
        qs.extend(seq_qs);
        ds.extend(seq_ds);
    }
    let mut state = CudaState::open()?;
    state.resident_q6_packed_limit = usize::MAX;
    let (packed_qs_dev, packed_d_super_dev, packed_sub_scale_dev) = state
        .resident_q6k_packed_ptrs(weights, rows, blocks_per_row)?
        .ok_or_else(|| "Q6 packed cache disabled in test".to_string())?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let output_dev = state.mem_alloc(seq_len * rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.launch_q6k_packed_batch_q8dot_to_dev(
        packed_qs_dev,
        packed_d_super_dev,
        packed_sub_scale_dev,
        rows,
        blocks_per_row,
        seq_len,
        qs_dev,
        ds_dev,
        output_dev,
    )?;
    let mut output = vec![0.0f32; seq_len * rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn gelu_mul_q8_1_for_test(gate: &[f32], up: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
    if gate.len() != up.len() {
        return Err(format!(
            "GELU Q8 input length mismatch: gate={} up={}",
            gate.len(),
            up.len()
        ));
    }
    if gate.len() % 32 != 0 {
        return Err(format!(
            "GELU Q8 input length must be multiple of 32, got {}",
            gate.len()
        ));
    }
    let mut state = CudaState::open()?;
    let gate_dev = state.mem_alloc(std::mem::size_of_val(gate))?;
    let up_dev = state.mem_alloc(std::mem::size_of_val(up))?;
    let qs_dev = state.mem_alloc(gate.len())?;
    let ds_len = gate.len() / 32;
    let ds_dev = state.mem_alloc(ds_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            gate_dev,
            gate.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(gate),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            up_dev,
            up.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(up),
            state.stream,
        )?;
    }
    state.launch_gelu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, gate.len())?;
    let mut f32_out = vec![0.0f32; gate.len()];
    let mut qs = vec![0i8; gate.len()];
    let mut ds = vec![0.0f32; ds_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            f32_out.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(f32_out.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            qs.as_mut_ptr().cast::<libc::c_void>(),
            qs_dev,
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            ds.as_mut_ptr().cast::<libc::c_void>(),
            ds_dev,
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
    }
    let mut dequant = vec![0.0f32; gate.len()];
    for block in 0..ds_len {
        for lane in 0..32 {
            let idx = block * 32 + lane;
            dequant[idx] = f32::from(qs[idx]) * ds[block];
        }
    }
    Ok((f32_out, dequant))
}

#[cfg(test)]
pub fn silu_mul_q8_1_for_test(gate: &[f32], up: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
    if gate.len() != up.len() {
        return Err(format!(
            "SiLU Q8 input length mismatch: gate={} up={}",
            gate.len(),
            up.len()
        ));
    }
    if gate.len() % 32 != 0 {
        return Err(format!(
            "SiLU Q8 input length must be multiple of 32, got {}",
            gate.len()
        ));
    }
    let mut state = CudaState::open()?;
    let gate_dev = state.mem_alloc(std::mem::size_of_val(gate))?;
    let up_dev = state.mem_alloc(std::mem::size_of_val(up))?;
    let qs_dev = state.mem_alloc(gate.len())?;
    let ds_len = gate.len() / 32;
    let ds_dev = state.mem_alloc(ds_len * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            gate_dev,
            gate.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(gate),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            up_dev,
            up.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(up),
            state.stream,
        )?;
    }
    state.launch_silu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, gate.len())?;
    let mut f32_out = vec![0.0f32; gate.len()];
    let mut qs = vec![0i8; gate.len()];
    let mut ds = vec![0.0f32; ds_len];
    unsafe {
        state.api.memcpy_dtoh_async(
            f32_out.as_mut_ptr().cast::<libc::c_void>(),
            gate_dev,
            std::mem::size_of_val(f32_out.as_slice()),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            qs.as_mut_ptr().cast::<libc::c_void>(),
            qs_dev,
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_dtoh_async(
            ds.as_mut_ptr().cast::<libc::c_void>(),
            ds_dev,
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(gate_dev)?;
        state.api.mem_free(up_dev)?;
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
    }
    let mut dequant = vec![0.0f32; gate.len()];
    for block in 0..ds_len {
        for lane in 0..32 {
            let idx = block * 32 + lane;
            dequant[idx] = f32::from(qs[idx]) * ds[block];
        }
    }
    Ok((f32_out, dequant))
}

#[cfg(test)]
fn qk_gemv_q8_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    q6k: bool,
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q_K cols must be multiple of 256, got {cols}"));
    }
    if input.len() != cols {
        return Err(format!(
            "Q_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let (qs, ds) = quantize_q8_1_by_32_for_test(input, blocks_per_row);
    let mut state = CudaState::open()?;
    let qs_dev = state.mem_alloc(qs.len())?;
    let ds_dev = state.mem_alloc(std::mem::size_of_val(ds.as_slice()))?;
    let output_dev = state.mem_alloc(rows * std::mem::size_of::<f32>())?;
    unsafe {
        state.api.memcpy_htod_async(
            qs_dev,
            qs.as_ptr().cast::<libc::c_void>(),
            qs.len(),
            state.stream,
        )?;
        state.api.memcpy_htod_async(
            ds_dev,
            ds.as_ptr().cast::<libc::c_void>(),
            std::mem::size_of_val(ds.as_slice()),
            state.stream,
        )?;
    }
    if q6k {
        state.launch_q6k_gemv_q8dot_to_dev(
            weights,
            rows,
            blocks_per_row,
            qs_dev,
            ds_dev,
            output_dev,
        )?;
    } else {
        state.launch_q4k_gemv_q8dot_to_dev(
            weights,
            rows,
            blocks_per_row,
            qs_dev,
            ds_dev,
            output_dev,
        )?;
    }
    let mut output = vec![0.0f32; rows];
    unsafe {
        state.api.memcpy_dtoh_async(
            output.as_mut_ptr().cast::<libc::c_void>(),
            output_dev,
            std::mem::size_of_val(output.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    unsafe {
        state.api.mem_free(qs_dev)?;
        state.api.mem_free(ds_dev)?;
        state.api.mem_free(output_dev)?;
    }
    Ok(output)
}

#[cfg(test)]
fn quantize_q8_1_by_32_for_test(input: &[f32], blocks_per_row: usize) -> (Vec<i8>, Vec<f32>) {
    let mut qs = vec![0i8; blocks_per_row * 256];
    let mut ds = vec![0.0f32; blocks_per_row * 8];
    for b in 0..blocks_per_row {
        for j in 0..8 {
            let off = b * 256 + j * 32;
            let chunk = &input[off..off + 32];
            let max_abs = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if max_abs == 0.0 {
                continue;
            }
            let d = max_abs / 127.0;
            let inv_d = 1.0 / d;
            ds[b * 8 + j] = d;
            for (idx, &value) in chunk.iter().enumerate() {
                qs[off + idx] = (value * inv_d).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (qs, ds)
}

#[cfg(test)]
pub fn q6k_gemv_for_test(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q6k_gemv(weights, rows, cols, input)
}
