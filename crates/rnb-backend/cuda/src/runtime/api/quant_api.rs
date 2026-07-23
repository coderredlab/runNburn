use super::super::state::q4_f32_cache_bytes;
use super::super::*;
use crate::runtime::gemv::Q4kF16DenseChainOutput;
use rnb_backend_api::QuantFormat;

fn env_enabled_or(name: &str, default: bool) -> bool {
    env_enabled_value(name).unwrap_or(default)
}

fn env_enabled_value(name: &str) -> Option<bool> {
    std::env::var(name)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .ok()
}

struct Q4F32GateUpCandidate<'a> {
    gate: &'a [u8],
    up: &'a [u8],
    rows: usize,
    blocks_per_row: usize,
    pair_bytes: usize,
    order: usize,
}

fn q4k_f32_gate_up_candidates<'a>(
    weights: &[(&'a [u8], &'a [u8], usize, usize)],
) -> Result<Vec<Q4F32GateUpCandidate<'a>>, String> {
    let mut candidates = Vec::with_capacity(weights.len());
    for (order, &(gate, up, rows, cols)) in weights.iter().enumerate() {
        if cols % 256 != 0 {
            return Err(format!(
                "Q4_K F32 gate/up cols must be divisible by 256, got {cols}"
            ));
        }
        let blocks_per_row = cols / 256;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| {
                format!("Q4_K F32 gate/up prewarm size overflow: rows={rows} cols={cols}")
            })?;
        if gate.len() != expected || up.len() != expected {
            return Err(format!(
                "Q4_K F32 gate/up prewarm byte mismatch: gate={} up={} expected={expected}",
                gate.len(),
                up.len()
            ));
        }
        let single_bytes = q4_f32_cache_bytes(rows, blocks_per_row)?;
        let pair_bytes = single_bytes.checked_mul(2).ok_or_else(|| {
            format!("Q4_K F32 gate/up pair byte overflow: rows={rows} cols={cols}")
        })?;
        candidates.push(Q4F32GateUpCandidate {
            gate,
            up,
            rows,
            blocks_per_row,
            pair_bytes,
            order,
        });
    }
    candidates.sort_by(|a, b| {
        b.pair_bytes
            .cmp(&a.pair_bytes)
            .then_with(|| a.order.cmp(&b.order))
    });
    Ok(candidates)
}

pub fn q2k_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q2_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 84;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q2_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q2_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q2k_gemv(weights, rows, blocks_per_row, input)
}

pub fn q3k_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q3_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 110;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q3_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q3_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q3k_gemv(weights, rows, blocks_per_row, input)
}

pub fn q2k_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "Q2_K",
        256,
        84,
        "rnb_q2k_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q3k_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "Q3_K",
        256,
        110,
        "rnb_q3k_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q4k_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q4_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q4_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_gemv(weights, rows, blocks_per_row, input)
}

pub fn q4k_gemv_into(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    q4k_gemv_into_with_touch(weights, rows, cols, input, output, false)
}

pub fn q4k_gemv_into_touch_hit(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    q4k_gemv_into_with_touch(weights, rows, cols, input, output, true)
}

// cu45 step 23: 단일 q6k_gemv 의 device input variant. Gemma4 V weight (Q6K) 의
// device input chain. q4k_gemv_with_device_input 와 같은 패턴.
pub fn q6k_gemv_with_device_input(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input_dev: u64,
    output: &mut [f32],
) -> Result<(), String> {
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 210;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q6_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if output.len() < rows {
        return Err(format!(
            "Q6_K output too small: got {}, expected >= {rows}",
            output.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q6k_gemv_with_device_input(
            weights,
            rows,
            blocks_per_row,
            input_dev,
            &mut output[..rows],
        )
}

// cu42 step 9: 단일 q4k_gemv 의 device input variant. caller (decode_attention_qkv
// 의 reuse_q_only path) 가 RMS norm carrier 를 input 으로 직접 전달.
pub fn q4k_gemv_with_device_input(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input_dev: u64,
    output: &mut [f32],
) -> Result<(), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q4_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if output.len() < rows {
        return Err(format!(
            "Q4_K output too small: got {}, expected >= {rows}",
            output.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_gemv_with_device_input(
            weights,
            rows,
            blocks_per_row,
            input_dev,
            &mut output[..rows],
        )
}

fn q4k_gemv_into_with_touch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
    touch_hit: bool,
) -> Result<(), String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q4_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols || output.len() < rows {
        return Err(format!(
            "Q4_K shape mismatch: input={} output={} expected input={cols} output>={rows}",
            input.len(),
            output.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    if touch_hit {
        state.q4k_gemv_into_touch_hit(weights, rows, blocks_per_row, input, &mut output[..rows])
    } else {
        state.q4k_gemv_into(weights, rows, blocks_per_row, input, &mut output[..rows])
    }
}

pub fn q4_0_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv(
        "Q4_0",
        32,
        18,
        "rnb_q4_0_gemv_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q4_1_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv(
        "Q4_1",
        32,
        20,
        "rnb_q4_1_gemv_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q4_0_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "Q4_0",
        32,
        18,
        "rnb_q4_0_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q4_1_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "Q4_1",
        32,
        20,
        "rnb_q4_1_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q5_0_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q5_basic_gemv("Q5_0", 22, "rnb_q5_0_gemv", weights, rows, cols, input)
}

pub fn q5_1_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q5_basic_gemv("Q5_1", 24, "rnb_q5_1_gemv", weights, rows, cols, input)
}

pub fn q5_0_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q5_basic_gemv_batch(
        "Q5_0",
        22,
        "rnb_q5_0_gemv_batch",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q5_1_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q5_basic_gemv_batch(
        "Q5_1",
        24,
        "rnb_q5_1_gemv_batch",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q8_0_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q5_basic_gemv("Q8_0", 34, "rnb_q8_0_gemv", weights, rows, cols, input)
}

pub fn q8_1_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv(
        "Q8_1",
        32,
        36,
        "rnb_q8_1_gemv_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q8_1_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "Q8_1",
        32,
        36,
        "rnb_q8_1_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn iq2_xxs_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv(
        "IQ2_XXS",
        256,
        66,
        "rnb_iq2_xxs_gemv_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn iq2_xxs_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "IQ2_XXS",
        256,
        66,
        "rnb_iq2_xxs_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn iq2_s_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv(
        "IQ2_S",
        256,
        82,
        "rnb_iq2_s_gemv_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn iq2_s_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "IQ2_S",
        256,
        82,
        "rnb_iq2_s_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn iq3_xxs_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv(
        "IQ3_XXS",
        256,
        98,
        "rnb_iq3_xxs_gemv_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn iq3_xxs_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    quant_block_gemv_batch(
        "IQ3_XXS",
        256,
        98,
        "rnb_iq3_xxs_gemv_batch_warp8",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn iq4_xs_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("IQ4_XS cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 136;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "IQ4_XS weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "IQ4_XS input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .iq4_xs_gemv(weights, rows, blocks_per_row, input)
}

pub fn iq4_xs_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("IQ4_XS cols must be divisible by 256, got {cols}"));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "IQ4_XS batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 136;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "IQ4_XS weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .gemv_batch(
            "rnb_iq4_xs_gemv_batch_warp8",
            weights,
            rows,
            blocks_per_row,
            seq_len,
            input,
        )
}

pub fn q8_0_gemv_argmax(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(u32, f32), String> {
    if cols % 32 != 0 {
        return Err(format!("Q8_0 cols must be divisible by 32, got {cols}"));
    }
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * 34;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q8_0 weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q8_0 input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q8_0_gemv_argmax(weights, rows, blocks_per_row, input)
}

pub fn q8_0_gemv_argmax_q8dot(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(u32, f32), String> {
    if cols % 32 != 0 {
        return Err(format!("Q8_0 cols must be divisible by 32, got {cols}"));
    }
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * 34;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q8_0 weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q8_0 input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q8_0_gemv_argmax_q8dot(weights, rows, blocks_per_row, input)
}

pub fn q8_0_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    q5_basic_gemv_batch(
        "Q8_0",
        34,
        "rnb_q8_0_gemv_batch",
        weights,
        rows,
        cols,
        input,
    )
}

pub fn q8_0_head_gemv_batch(
    weights: &[u8],
    head_count: usize,
    rows_per_head: usize,
    cols: usize,
    token_count: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if head_count == 0 || rows_per_head == 0 || token_count == 0 {
        return Err(format!(
            "Q8_0 head batch dimensions must be non-zero: heads={head_count} rows_per_head={rows_per_head} tokens={token_count}"
        ));
    }
    if cols == 0 || cols % 32 != 0 {
        return Err(format!(
            "Q8_0 head batch cols must be non-zero and divisible by 32, got {cols}"
        ));
    }
    let blocks_per_row = cols / 32;
    let expected_weights = head_count
        .checked_mul(rows_per_head)
        .and_then(|value| value.checked_mul(blocks_per_row))
        .and_then(|value| value.checked_mul(34))
        .ok_or_else(|| "Q8_0 head batch weight byte size overflow".to_string())?;
    if weights.len() != expected_weights {
        return Err(format!(
            "Q8_0 head batch weight byte mismatch: got {}, expected {expected_weights}",
            weights.len()
        ));
    }
    let expected_input = token_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(cols))
        .ok_or_else(|| "Q8_0 head batch input length overflow".to_string())?;
    if input.len() != expected_input {
        return Err(format!(
            "Q8_0 head batch input length mismatch: got {}, expected {expected_input}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q8_0_head_gemv_batch(
            weights,
            head_count,
            rows_per_head,
            blocks_per_row,
            token_count,
            input,
        )
}

pub fn q8_0_f32_gemm_batch_cached(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !env_enabled_or("RNB_CUDA_Q8_0_PREFILL_F32_GEMM", false) {
        return Ok(None);
    }
    if cols == 0 || cols % 32 != 0 {
        return Err(format!(
            "Q8_0 F32 GEMM cols must be non-zero and divisible by 32, got {cols}"
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q8_0 F32 GEMM batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * 34;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q8_0 F32 GEMM weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q8_0_f32_gemm_batch_cached(weights, rows, blocks_per_row, seq_len, input)
}

pub fn prewarm_q8_0_weight(weights: &[u8], rows: usize, cols: usize) -> Result<(), String> {
    if cols % 32 != 0 {
        return Err(format!("Q8_0 cols must be divisible by 32, got {cols}"));
    }
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * 34;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q8_0 weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    state.resident_q8_quant_ptr(weights, rows, cols)?;
    state.stream_synchronize()
}

pub fn prewarm_q4k_weights(weights: &[&[u8]]) -> Result<usize, String> {
    if weights.is_empty() {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for weight in weights {
        state.resident_q4k_weights_ptr(weight)?;
        warmed += 1;
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

fn div_ceil_usize(value: usize, divisor: usize) -> usize {
    value.saturating_add(divisor.saturating_sub(1)) / divisor
}

pub fn prewarm_quant_resident_q4k_weights(weights: &[&[u8]]) -> Result<usize, String> {
    if !super::super::state::quant_resident_policy_requested()? {
        return Ok(0);
    }
    if weights.is_empty() {
        return Ok(0);
    }

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");

    let (free_bytes, total_bytes) = unsafe { state.api.mem_get_info() }?;
    let mib = 1024 * 1024;
    let model_quant_bytes = weights.iter().try_fold(0usize, |total, weight| {
        total
            .checked_add(weight.len())
            .ok_or_else(|| "Q4_K quant resident prewarm byte count overflowed usize".to_string())
    })?;
    let model_quant_mib = div_ceil_usize(model_quant_bytes, mib);
    let plan = super::super::state::quant_resident_budget_plan(
        total_bytes / mib,
        free_bytes / mib,
        model_quant_mib,
        0,
    )?;
    if !plan.enabled || plan.raw_quant_target_mib == 0 {
        return Ok(0);
    }

    let mut attempted = Vec::new();
    let target_bytes = plan.raw_quant_target_mib.saturating_mul(mib);
    if state.resident_q4k_limit < target_bytes {
        state.resident_q4k_limit = target_bytes;
    }
    for weight in weights {
        if state.resident_q4k_bytes >= target_bytes
            || weight.len() > target_bytes.saturating_sub(state.resident_q4k_bytes)
        {
            break;
        }
        attempted.push(*weight);
        let _ = state.preload_resident_q4k_weight_slice(weight)?;
    }
    state.stream_synchronize()?;
    let warmed = attempted
        .iter()
        .filter(|weight| state.q4k_weight_slice_is_resident(weight))
        .count();
    Ok(warmed)
}

pub fn prewarm_q4k_weights_pinned(weights: &[&[u8]]) -> Result<usize, String> {
    if weights.is_empty() {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for weight in weights {
        state.resident_q4k_weights_ptr_pinned(weight)?;
        warmed += 1;
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

pub fn prewarm_q4k_packed_gate_up_weights(
    weights: &[(&[u8], &[u8], usize, usize)],
) -> Result<usize, String> {
    if weights.is_empty() || !env_enabled_or("RNB_CUDA_DENSE_Q4_PACKED_Q8DOT", true) {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for (gate, up, rows, cols) in weights {
        if cols % 256 != 0 {
            return Err(format!(
                "Q4_K packed gate/up cols must be divisible by 256, got {cols}"
            ));
        }
        let blocks_per_row = cols / 256;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| {
                format!("Q4_K packed gate/up prewarm size overflow: rows={rows} cols={cols}")
            })?;
        if gate.len() != expected || up.len() != expected {
            return Err(format!(
                "Q4_K packed gate/up prewarm byte mismatch: gate={} up={} expected={expected}",
                gate.len(),
                up.len()
            ));
        }
        let gate = state.resident_q4k_packed_ptrs(gate, *rows, blocks_per_row)?;
        let up = state.resident_q4k_packed_ptrs(up, *rows, blocks_per_row)?;
        if gate.is_some() && up.is_some() {
            warmed += 1;
        }
    }
    if warmed > 0 {
        let _ = state.cublas_state_mut()?;
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

pub fn prewarm_q4k_f32_gate_up_weights(
    weights: &[(&[u8], &[u8], usize, usize)],
) -> Result<usize, String> {
    if weights.is_empty() || !crate::tuning::expanded_weight_cache_allowed() {
        return Ok(0);
    }
    let f32_enabled = env_enabled_or("RNB_CUDA_Q4K_BATCH_F32_GATE_UP", false);
    let f16_enabled = env_enabled_value("RNB_CUDA_Q4K_BATCH_F16_GATE_UP").unwrap_or(!f32_enabled);
    if !f16_enabled && !f32_enabled {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for candidate in q4k_f32_gate_up_candidates(weights)? {
        let resident = if f16_enabled {
            state.resident_q4k_f16_pair_ptrs(
                candidate.gate,
                candidate.up,
                candidate.rows,
                candidate.blocks_per_row,
            )?
        } else {
            state.resident_q4k_f32_pair_ptrs(
                candidate.gate,
                candidate.up,
                candidate.rows,
                candidate.blocks_per_row,
            )?
        };
        if resident.is_some() {
            warmed += 1;
        }
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

pub fn prewarm_q4k_f16_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize, String> {
    if weights.is_empty()
        || !crate::tuning::expanded_weight_cache_allowed()
        || !env_enabled_or("RNB_CUDA_Q4K_BATCH_F16_DOWN", false)
    {
        return Ok(0);
    }
    prewarm_q4k_f16_weights_impl(weights, true)
}

pub fn prewarm_q4k_prefill_f16_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize, String> {
    if weights.is_empty()
        || !crate::tuning::expanded_weight_cache_allowed()
        || !env_enabled_or("RNB_CUDA_Q4K_PREFILL_F16_GEMM", false)
    {
        return Ok(0);
    }
    prewarm_q4k_f16_weights_impl(weights, false)
}

fn prewarm_q4k_f16_weights_impl(
    weights: &[(&[u8], usize, usize)],
    sort_by_size: bool,
) -> Result<usize, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut candidates = Vec::with_capacity(weights.len());
    for (raw, rows, cols) in weights {
        if cols % 256 != 0 {
            return Err(format!(
                "Q4_K F16 prewarm cols must be divisible by 256, got {cols}"
            ));
        }
        let blocks_per_row = cols / 256;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| format!("Q4_K F16 prewarm size overflow: rows={rows} cols={cols}"))?;
        if raw.len() != expected {
            return Err(format!(
                "Q4_K F16 prewarm byte mismatch: got {}, expected={expected}",
                raw.len()
            ));
        }
        let values = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| format!("Q4_K F16 prewarm score overflow: rows={rows} cols={cols}"))?;
        candidates.push((*raw, *rows, blocks_per_row, values));
    }
    if sort_by_size {
        candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.3));
    }

    let mut warmed = 0usize;
    for (raw, rows, blocks_per_row, _) in candidates {
        if state
            .resident_q4k_f16_ptr(raw, rows, blocks_per_row)?
            .is_some()
        {
            warmed += 1;
        }
    }
    if warmed > 0 {
        let _ = state.cublas_state_mut()?;
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

pub fn prewarm_q4k_f32_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize, String> {
    if weights.is_empty()
        || !crate::tuning::expanded_weight_cache_allowed()
        || !env_enabled_or("RNB_CUDA_Q4K_PREFILL_F32_GEMM", false)
    {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for (raw, rows, cols) in weights {
        if cols % 256 != 0 {
            return Err(format!(
                "Q4_K F32 prewarm cols must be divisible by 256, got {cols}"
            ));
        }
        let blocks_per_row = cols / 256;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| format!("Q4_K F32 prewarm size overflow: rows={rows} cols={cols}"))?;
        if raw.len() != expected {
            return Err(format!(
                "Q4_K F32 prewarm byte mismatch: got {}, expected={expected}",
                raw.len()
            ));
        }
        if state
            .resident_q4k_f32_ptr(raw, *rows, blocks_per_row)?
            .is_some()
        {
            warmed += 1;
        }
    }
    if warmed > 0 {
        let _ = state.cublas_state_mut()?;
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static Q4_F32_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn restore_env(name: &str, previous: Option<String>) {
        unsafe {
            if let Some(previous) = previous {
                std::env::set_var(name, previous);
            } else {
                std::env::remove_var(name);
            }
        }
    }

    #[test]
    fn q4k_f32_gemm_batch_cached_requires_expanded_gate() {
        let _guard = Q4_F32_ENV_LOCK.lock().expect("Q4 F32 env lock");
        let prev_allow = std::env::var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE").ok();
        let prev_q4_f32 = std::env::var("RNB_CUDA_Q4K_PREFILL_F32_GEMM").ok();
        unsafe {
            std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
            std::env::set_var("RNB_CUDA_Q4K_PREFILL_F32_GEMM", "1");
        }

        assert_eq!(
            q4k_f32_gemm_batch_cached(&[], 1, 0, &[]).expect("gated off"),
            None
        );

        unsafe {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
        }
        let err = q4k_f32_gemm_batch_cached(&[], 1, 0, &[])
            .expect_err("allow gate should reach shape validation");
        assert!(err.contains("cols must be non-zero"));

        restore_env("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", prev_allow);
        restore_env("RNB_CUDA_Q4K_PREFILL_F32_GEMM", prev_q4_f32);
    }

    #[test]
    fn q4_f32_gate_up_candidates_are_shape_ordered() {
        let small_gate = vec![1u8; 2 * 2 * 144];
        let small_up = vec![2u8; 2 * 2 * 144];
        let large_gate = vec![3u8; 4 * 3 * 144];
        let large_up = vec![4u8; 4 * 3 * 144];
        let weights = [
            (&small_gate[..], &small_up[..], 2usize, 512usize),
            (&large_gate[..], &large_up[..], 4usize, 768usize),
        ];

        let candidates = q4k_f32_gate_up_candidates(&weights).expect("valid candidates");

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].rows, 4);
        assert_eq!(candidates[0].blocks_per_row, 3);
        assert_eq!(candidates[1].rows, 2);
        assert_eq!(candidates[1].blocks_per_row, 2);
    }

    #[test]
    fn q4_f32_gate_up_candidates_keep_original_order_for_same_shape() {
        let first_gate = vec![1u8; 2 * 2 * 144];
        let first_up = vec![2u8; 2 * 2 * 144];
        let second_gate = vec![3u8; 2 * 2 * 144];
        let second_up = vec![4u8; 2 * 2 * 144];
        let weights = [
            (&first_gate[..], &first_up[..], 2usize, 512usize),
            (&second_gate[..], &second_up[..], 2usize, 512usize),
        ];

        let candidates = q4k_f32_gate_up_candidates(&weights).expect("valid candidates");

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].gate.as_ptr(), first_gate.as_ptr());
        assert_eq!(candidates[1].gate.as_ptr(), second_gate.as_ptr());
    }
}

pub fn prewarm_q4k_packed_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize, String> {
    if weights.is_empty() || !env_enabled_or("RNB_CUDA_DENSE_Q4_PACKED_Q8DOT", true) {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for (raw, rows, cols) in weights {
        if cols % 256 != 0 {
            return Err(format!(
                "Q4_K packed cols must be divisible by 256, got {cols}"
            ));
        }
        let blocks_per_row = cols / 256;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| format!("Q4_K packed prewarm size overflow: rows={rows} cols={cols}"))?;
        if raw.len() != expected {
            return Err(format!(
                "Q4_K packed prewarm byte mismatch: got {}, expected {expected}",
                raw.len()
            ));
        }
        if state
            .resident_q4k_packed_ptrs(raw, *rows, blocks_per_row)?
            .is_some()
        {
            warmed += 1;
        }
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

pub fn prewarm_q6k_packed_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize, String> {
    if weights.is_empty() || !env_enabled_or("RNB_CUDA_DENSE_Q6_PACKED_Q8DOT", true) {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for (raw, rows, cols) in weights {
        if cols % 256 != 0 {
            return Err(format!(
                "Q6_K packed cols must be divisible by 256, got {cols}"
            ));
        }
        let blocks_per_row = cols / 256;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(210))
            .ok_or_else(|| format!("Q6_K packed prewarm size overflow: rows={rows} cols={cols}"))?;
        if raw.len() != expected {
            return Err(format!(
                "Q6_K packed prewarm byte mismatch: got {}, expected {expected}",
                raw.len()
            ));
        }
        if state
            .resident_q6k_packed_ptrs(raw, *rows, blocks_per_row)?
            .is_some()
        {
            warmed += 1;
        }
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

pub fn q6_f16_prewarm_enabled() -> bool {
    if !crate::tuning::expanded_weight_cache_allowed() {
        return false;
    }
    if env_enabled_or("RNB_CUDA_Q6K_BATCH_F16_PREWARM", false) {
        return true;
    }
    std::env::var("RNB_CUDA_Q6K_BATCH_F16_DOWN")
        .map(|value| value.eq_ignore_ascii_case("force"))
        .unwrap_or(false)
}

pub fn prewarm_q6k_f32_weights(weights: &[(&[u8], usize, usize)]) -> Result<usize, String> {
    if weights.is_empty() || !crate::tuning::expanded_weight_cache_allowed() {
        return Ok(0);
    }
    let f16_enabled = q6_f16_prewarm_enabled();
    let f32_enabled = env_enabled_or("RNB_CUDA_Q6K_BATCH_F32_DOWN", false);
    if !f16_enabled && !f32_enabled {
        return Ok(0);
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let mut warmed = 0usize;
    for (raw, rows, cols) in weights {
        if cols % 256 != 0 {
            return Err(format!(
                "Q6_K F32 cols must be divisible by 256, got {cols}"
            ));
        }
        let blocks_per_row = cols / 256;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(210))
            .ok_or_else(|| format!("Q6_K F32 prewarm size overflow: rows={rows} cols={cols}"))?;
        if raw.len() != expected {
            return Err(format!(
                "Q6_K F32 prewarm byte mismatch: got {}, expected {expected}",
                raw.len()
            ));
        }
        let resident = if f32_enabled {
            state.resident_q6k_f32_ptr(raw, *rows, blocks_per_row)?
        } else {
            state.resident_q6k_f16_ptr(raw, *rows, blocks_per_row)?
        };
        if resident.is_some() {
            warmed += 1;
        }
    }
    if warmed > 0 {
        let _ = state.cublas_state_mut()?;
    }
    state.stream_synchronize()?;
    Ok(warmed)
}

fn quant_block_gemv(
    label: &str,
    block_elems: usize,
    block_bytes: usize,
    kernel: &'static str,
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % block_elems != 0 {
        return Err(format!(
            "{label} cols must be divisible by {block_elems}, got {cols}"
        ));
    }
    let blocks_per_row = cols / block_elems;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|value| value.checked_mul(block_bytes))
        .ok_or_else(|| format!("{label} weight byte size overflow"))?;
    if weights.len() != expected {
        return Err(format!(
            "{label} weight byte mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "{label} input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q5_basic_gemv(weights, rows, blocks_per_row, input, kernel, false)
}

fn quant_block_gemv_batch(
    label: &str,
    block_elems: usize,
    block_bytes: usize,
    kernel: &'static str,
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % block_elems != 0 {
        return Err(format!(
            "{label} cols must be divisible by {block_elems}, got {cols}"
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "{label} batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / block_elems;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|value| value.checked_mul(block_bytes))
        .ok_or_else(|| format!("{label} weight byte size overflow"))?;
    if weights.len() != expected {
        return Err(format!(
            "{label} weight byte mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q5_basic_gemv_batch(weights, rows, blocks_per_row, seq_len, input, kernel)
}

pub fn quant_embedding_gather(
    quant: QuantFormat,
    weights: &[u8],
    rows: usize,
    cols: usize,
    token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    let (kernel, block_elems, block_bytes) = match quant {
        QuantFormat::F32 => ("rnb_quant_f32_embedding_gather", 32usize, 128usize),
        QuantFormat::F16 => ("rnb_quant_f16_embedding_gather", 32, 64),
        QuantFormat::BF16 => ("rnb_quant_bf16_embedding_gather", 32, 64),
        QuantFormat::Q40 => ("rnb_quant_q4_0_embedding_gather", 32, 18),
        QuantFormat::Q41 => ("rnb_quant_q4_1_embedding_gather", 32, 20),
        QuantFormat::Q50 => ("rnb_quant_q5_0_embedding_gather", 32, 22),
        QuantFormat::Q51 => ("rnb_quant_q5_1_embedding_gather", 32, 24),
        QuantFormat::Q80 => ("rnb_quant_q8_0_embedding_gather", 32, 34),
        QuantFormat::Q81 => ("rnb_quant_q8_1_embedding_gather", 32, 36),
        QuantFormat::Q2K => ("rnb_quant_q2k_embedding_gather", 256, 84),
        QuantFormat::Q3K => ("rnb_quant_q3k_embedding_gather", 256, 110),
        QuantFormat::Q4K => ("rnb_quant_q4k_embedding_gather", 256, 144),
        QuantFormat::Q5K => ("rnb_quant_q5k_embedding_gather", 256, 176),
        QuantFormat::Q6K => ("rnb_quant_q6k_embedding_gather", 256, 210),
        QuantFormat::IQ2XXS => ("rnb_quant_iq2_xxs_embedding_gather", 256, 66),
        QuantFormat::IQ2S => ("rnb_quant_iq2_s_embedding_gather", 256, 82),
        QuantFormat::IQ3XXS => ("rnb_quant_iq3_xxs_embedding_gather", 256, 98),
        QuantFormat::IQ4XS => ("rnb_quant_iq4_xs_embedding_gather", 256, 136),
    };
    if cols == 0 || cols % block_elems != 0 {
        return Err(format!(
            "{quant:?} embedding cols must be non-zero and divisible by {block_elems}, got {cols}"
        ));
    }
    if token_ids.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(token) = token_ids
        .iter()
        .copied()
        .find(|&token| token as usize >= rows)
    {
        return Err(format!(
            "{quant:?} embedding token id {token} is out of range for {rows} rows"
        ));
    }
    let blocks_per_row = cols / block_elems;
    let expected_weights = rows
        .checked_mul(blocks_per_row)
        .and_then(|value| value.checked_mul(block_bytes))
        .ok_or_else(|| format!("{quant:?} embedding weight byte size overflow"))?;
    if weights.len() != expected_weights {
        return Err(format!(
            "{quant:?} embedding weight byte mismatch: got {}, expected {expected_weights}",
            weights.len()
        ));
    }
    let output_len = token_ids
        .len()
        .checked_mul(cols)
        .ok_or_else(|| format!("{quant:?} embedding output length overflow"))?;
    let output_bytes = output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{quant:?} embedding output byte size overflow"))?;
    let token_bytes = std::mem::size_of_val(token_ids);

    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    state.set_current()?;
    let token_ids_dev = unsafe { state.api.mem_alloc(token_bytes)? };
    let output_dev = match unsafe { state.api.mem_alloc(output_bytes) } {
        Ok(output) => output,
        Err(err) => {
            unsafe {
                state.api.mem_free(token_ids_dev)?;
            }
            return Err(err);
        }
    };
    let mut output = vec![0.0f32; output_len];
    let operation = (|| {
        unsafe {
            state.api.memcpy_htod_async(
                token_ids_dev,
                token_ids.as_ptr().cast::<libc::c_void>(),
                token_bytes,
                state.stream,
            )?;
        }
        state.launch_quant_embedding_gather_to_dev(
            kernel,
            weights,
            rows,
            blocks_per_row,
            block_elems,
            token_ids_dev,
            token_ids.len(),
            output_dev,
        )?;
        unsafe {
            state.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                state.stream,
            )?;
        }
        state.stream_synchronize()
    })();
    let free_result = unsafe {
        state
            .api
            .mem_free(output_dev)
            .and_then(|_| state.api.mem_free(token_ids_dev))
    };
    operation?;
    free_result?;
    Ok(output)
}

fn q5_basic_gemv(
    label: &str,
    block_bytes: usize,
    kernel: &'static str,
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 32 != 0 {
        return Err(format!("{label} cols must be divisible by 32, got {cols}"));
    }
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * block_bytes;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "{label} weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "{label} input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q5_basic_gemv(
            weights,
            rows,
            blocks_per_row,
            input,
            kernel,
            label == "Q8_0",
        )
}

fn q5_basic_gemv_batch(
    label: &str,
    block_bytes: usize,
    kernel: &'static str,
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 32 != 0 {
        return Err(format!("{label} cols must be divisible by 32, got {cols}"));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "{label} batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 32;
    let row_bytes = blocks_per_row * block_bytes;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "{label} weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q5_basic_gemv_batch(weights, rows, blocks_per_row, seq_len, input, kernel)
}

pub fn q6k_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 210;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q6_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q6_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q6k_gemv(weights, rows, blocks_per_row, input)
}

pub fn q6k_gemv_into(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    q6k_gemv_into_with_touch(weights, rows, cols, input, output, false)
}

pub fn q6k_gemv_into_touch_hit(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    q6k_gemv_into_with_touch(weights, rows, cols, input, output, true)
}

fn q6k_gemv_into_with_touch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
    touch_hit: bool,
) -> Result<(), String> {
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 210;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q6_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols || output.len() < rows {
        return Err(format!(
            "Q6_K shape mismatch: input={} output={} expected input={cols} output>={rows}",
            input.len(),
            output.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    if touch_hit {
        state.q6k_gemv_into_touch_hit(weights, rows, blocks_per_row, input, &mut output[..rows])
    } else {
        state.q6k_gemv_into(weights, rows, blocks_per_row, input, &mut output[..rows])
    }
}

pub fn q6k_gemv_argmax(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(u32, f32), String> {
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 210;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q6_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q6_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q6k_gemv_argmax(weights, rows, blocks_per_row, input)
}

pub fn q5k_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q5_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 176;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q5_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols {
        return Err(format!(
            "Q5_K input length mismatch: got {}, expected {cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q5k_gemv(weights, rows, blocks_per_row, input)
}

pub fn q5k_gemv_into(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    if cols % 256 != 0 {
        return Err(format!("Q5_K cols must be divisible by 256, got {cols}"));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 176;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q5_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    if input.len() != cols || output.len() < rows {
        return Err(format!(
            "Q5_K shape mismatch: input={} output={} expected input={cols} output>={rows}",
            input.len(),
            output.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q5k_gemv_into(weights, rows, blocks_per_row, input, &mut output[..rows])
}

pub fn q4k_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q4_K cols must be divisible by 256, got {cols}"));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q4_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_gemv_batch(weights, rows, blocks_per_row, seq_len, input)
}

pub fn q4k_f32_gemm_batch_cached(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !crate::tuning::q4k_prefill_f32_gemm_enabled() {
        return Ok(None);
    }
    if cols == 0 || cols % 256 != 0 {
        return Err(format!(
            "Q4_K F32 GEMM cols must be non-zero and divisible by 256, got {cols}"
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K F32 GEMM batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q4_K F32 GEMM weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f32_gemm_batch_cached(weights, rows, blocks_per_row, seq_len, input)
}

pub fn q4k_f16_gemm_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 GEMM cols must be divisible by 256, got {cols}"
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K F16 GEMM batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q4_K F16 GEMM weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_gemm_batch(weights, rows, blocks_per_row, seq_len, input)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_f16_qkv_gemm_batch(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 QKV GEMM cols must be divisible by 256, got {cols}"
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K F16 QKV GEMM batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 QKV GEMM q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let kv_expected = kv_rows * row_bytes;
    if k_weights.len() != kv_expected || v_weights.len() != kv_expected {
        return Err(format!(
            "Q4_K F16 QKV GEMM k/v weight byte mismatch: k={} v={} expected {kv_expected}",
            k_weights.len(),
            v_weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_qkv_gemm_batch(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            input,
        )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_f16_qkv_postprocess_hd256(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    input: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
) -> Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 QKV postprocess cols must be divisible by 256, got {cols}"
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K F16 QKV postprocess batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 QKV postprocess q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let kv_expected = kv_rows * row_bytes;
    if k_weights.len() != kv_expected || v_weights.len() != kv_expected {
        return Err(format!(
            "Q4_K F16 QKV postprocess k/v weight byte mismatch: k={} v={} expected {kv_expected}",
            k_weights.len(),
            v_weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_qkv_postprocess_hd256(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            input,
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
        )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_f16_qkv_postprocess_hd256_window_dense_chain(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    v_quant: u32,
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
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
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<(Vec<u16>, Vec<u16>)>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain cols must be divisible by 256, got {cols}"
        ));
    }
    if hidden_input.len() % cols != 0 {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain input length {} is not divisible by cols {cols}",
            hidden_input.len()
        ));
    }
    let seq_len = hidden_input.len() / cols;
    if attn_norm_weight.len() != cols {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain attn norm length mismatch: got {}, expected {cols}",
            attn_norm_weight.len()
        ));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let k_expected = kv_rows * row_bytes;
    let v_row_bytes = match v_quant {
        12 => row_bytes,
        14 => blocks_per_row * 210,
        other => {
            return Err(format!(
                "unsupported Q4_K F16 QKV hd256 window chain V quant code {other}"
            ))
        }
    };
    let v_expected = kv_rows * v_row_bytes;
    if k_weights.len() != k_expected {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain k weight byte mismatch: got {}, expected {k_expected}",
            k_weights.len()
        ));
    }
    if v_weights.len() != v_expected {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain v weight byte mismatch: got {}, expected {v_expected}",
            v_weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let output = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            q_weights,
            k_weights,
            v_weights,
            v_quant,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            hidden_input,
            None,
            attn_norm_weight,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
            window,
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
            hidden,
            None,
            None,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    match output {
        Some((k_bits, v_bits, Q4kF16DenseChainOutput::Host)) => Ok(Some((k_bits, v_bits))),
        Some((_, _, Q4kF16DenseChainOutput::Device(_))) => Err(
            "Q4_K F16 QKV hd256 window dense chain returned device output for host request"
                .to_string(),
        ),
        None => Ok(None),
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_f16_qkv_postprocess_hd256_window_dense_chain_device_output(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    v_quant: u32,
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
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
    device_output_desc: rnb_backend_api::DeviceTensorDesc,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<(Vec<u16>, Vec<u16>, rnb_backend_api::DeviceTensorId)>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain device-output cols must be divisible by 256, got {cols}"
        ));
    }
    let seq_len = if let Some((_, desc)) = hidden_input_device.as_ref() {
        if desc.cols() != cols || desc.dtype() != rnb_backend_api::ScalarType::F32 {
            return Ok(None);
        }
        desc.rows()
    } else {
        if hidden_input.len() % cols != 0 {
            return Err(format!(
                "Q4_K F16 QKV hd256 window chain device-output input length {} is not divisible by cols {cols}",
                hidden_input.len()
            ));
        }
        hidden_input.len() / cols
    };
    let expected_hidden = seq_len.saturating_mul(n_embd);
    if attn_norm_weight.len() != cols {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain device-output attn norm length mismatch: got {}, expected {cols}",
            attn_norm_weight.len()
        ));
    }
    if hidden.len() != expected_hidden {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain device-output hidden length mismatch: got {}, expected {}",
            hidden.len(),
            expected_hidden
        ));
    }
    if device_output_desc.rows() != seq_len
        || device_output_desc.cols() != n_embd
        || device_output_desc.dtype() != rnb_backend_api::ScalarType::F32
    {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain device output desc mismatch: got rows={} cols={} dtype={:?}, expected rows={seq_len} cols={n_embd} dtype=F32",
            device_output_desc.rows(),
            device_output_desc.cols(),
            device_output_desc.dtype()
        ));
    }
    if let Some(scale) = layer_out_scale {
        if scale.is_empty() {
            return Err(
                "Q4_K F16 QKV hd256 window chain device-output out_scale must be non-empty"
                    .to_string(),
            );
        }
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain device-output q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let k_expected = kv_rows * row_bytes;
    let v_row_bytes = match v_quant {
        12 => row_bytes,
        14 => blocks_per_row * 210,
        other => {
            return Err(format!(
                "unsupported Q4_K F16 QKV hd256 window chain device-output V quant code {other}"
            ))
        }
    };
    let v_expected = kv_rows * v_row_bytes;
    if k_weights.len() != k_expected {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain device-output k weight byte mismatch: got {}, expected {k_expected}",
            k_weights.len()
        ));
    }
    if v_weights.len() != v_expected {
        return Err(format!(
            "Q4_K F16 QKV hd256 window chain device-output v weight byte mismatch: got {}, expected {v_expected}",
            v_weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let output = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_qkv_postprocess_hd256_window_dense_chain(
            q_weights,
            k_weights,
            v_weights,
            v_quant,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            hidden_input,
            hidden_input_device,
            attn_norm_weight,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
            window,
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
            hidden,
            layer_out_scale,
            Some(device_output_desc),
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    match output {
        Some((k_bits, v_bits, Q4kF16DenseChainOutput::Device(id))) => {
            Ok(Some((k_bits, v_bits, id)))
        }
        Some((_, _, Q4kF16DenseChainOutput::Host)) => Err(
            "Q4_K F16 QKV hd256 window dense chain returned host output for device-output request"
                .to_string(),
        ),
        None => Ok(None),
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_f16_qkv_prefill_attention_hd512(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    input: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
) -> Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 QKV attention cols must be divisible by 256, got {cols}"
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q4_K F16 QKV attention batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 QKV attention q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let kv_expected = kv_rows * row_bytes;
    if k_weights.len() != kv_expected || v_weights.len() != kv_expected {
        return Err(format!(
            "Q4_K F16 QKV attention k/v weight byte mismatch: k={} v={} expected {kv_expected}",
            k_weights.len(),
            v_weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_qkv_prefill_attention_hd512(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            input,
            q_norm,
            k_norm,
            freq_factors,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_f16_qkv_prefill_attention_hd512_dense_chain(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
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
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<(Vec<u16>, Vec<u16>)>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain cols must be divisible by 256, got {cols}"
        ));
    }
    if hidden_input.len() % cols != 0 {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain input length {} is not divisible by cols {cols}",
            hidden_input.len()
        ));
    }
    let seq_len = hidden_input.len() / cols;
    if attn_norm_weight.len() != cols {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain attn norm length mismatch: got {}, expected {cols}",
            attn_norm_weight.len()
        ));
    }
    if hidden.len() != seq_len.saturating_mul(n_embd) {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain hidden length mismatch: got {}, expected {}",
            hidden.len(),
            seq_len.saturating_mul(n_embd)
        ));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let kv_expected = kv_rows * row_bytes;
    if k_weights.len() != kv_expected || v_weights.len() != kv_expected {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain k/v weight byte mismatch: k={} v={} expected {kv_expected}",
            k_weights.len(),
            v_weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let output = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_qkv_prefill_attention_hd512_dense_chain(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            hidden_input,
            None,
            attn_norm_weight,
            q_norm,
            k_norm,
            freq_factors,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
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
            hidden,
            None,
            None,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    match output {
        Some((k_bits, v_bits, Q4kF16DenseChainOutput::Host)) => Ok(Some((k_bits, v_bits))),
        Some((_, _, Q4kF16DenseChainOutput::Device(_))) => Err(
            "Q4_K F16 QKV hd512 dense chain returned device output for host request".to_string(),
        ),
        None => Ok(None),
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_f16_qkv_prefill_attention_hd512_dense_chain_device_output(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    hidden_input_device: Option<(
        rnb_backend_api::DeviceTensorId,
        rnb_backend_api::DeviceTensorDesc,
    )>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    freq_factors: Option<&[f32]>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
    k_unit_offset: bool,
    v_no_scale_norm: bool,
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
    device_output_desc: rnb_backend_api::DeviceTensorDesc,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<(Vec<u16>, Vec<u16>, rnb_backend_api::DeviceTensorId)>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain device-output cols must be divisible by 256, got {cols}"
        ));
    }
    let seq_len = if let Some((_, desc)) = hidden_input_device.as_ref() {
        if desc.cols() != cols || desc.dtype() != rnb_backend_api::ScalarType::F32 {
            return Ok(None);
        }
        desc.rows()
    } else {
        if hidden_input.len() % cols != 0 {
            return Err(format!(
                "Q4_K F16 QKV hd512 dense chain device-output input length {} is not divisible by cols {cols}",
                hidden_input.len()
            ));
        }
        hidden_input.len() / cols
    };
    let expected_hidden = seq_len.saturating_mul(n_embd);
    if attn_norm_weight.len() != cols {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain device-output attn norm length mismatch: got {}, expected {cols}",
            attn_norm_weight.len()
        ));
    }
    if hidden.len() != expected_hidden {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain device-output hidden length mismatch: got {}, expected {}",
            hidden.len(),
            expected_hidden
        ));
    }
    if device_output_desc.rows() != seq_len
        || device_output_desc.cols() != n_embd
        || device_output_desc.dtype() != rnb_backend_api::ScalarType::F32
    {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain device output desc mismatch: got rows={} cols={} dtype={:?}, expected rows={seq_len} cols={n_embd} dtype=F32",
            device_output_desc.rows(),
            device_output_desc.cols(),
            device_output_desc.dtype()
        ));
    }
    if let Some(scale) = layer_out_scale {
        if scale.is_empty() {
            return Err(
                "Q4_K F16 QKV hd512 dense chain device-output out_scale must be non-empty"
                    .to_string(),
            );
        }
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain device-output q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let kv_expected = kv_rows * row_bytes;
    if k_weights.len() != kv_expected || v_weights.len() != kv_expected {
        return Err(format!(
            "Q4_K F16 QKV hd512 dense chain device-output k/v weight byte mismatch: k={} v={} expected {kv_expected}",
            k_weights.len(),
            v_weights.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let output = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_qkv_prefill_attention_hd512_dense_chain(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            hidden_input,
            hidden_input_device,
            attn_norm_weight,
            q_norm,
            k_norm,
            freq_factors,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
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
            hidden,
            layer_out_scale,
            Some(device_output_desc),
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    match output {
        Some((k_bits, v_bits, Q4kF16DenseChainOutput::Device(id))) => {
            Ok(Some((k_bits, v_bits, id)))
        }
        Some((_, _, Q4kF16DenseChainOutput::Host)) => Err(
            "Q4_K F16 QKV hd512 dense chain returned host output for device-output request"
                .to_string(),
        ),
        None => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    seq_len: usize,
    kv_len: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    freq_factors: Option<&[f32]>,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
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
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<bool, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain cols must be divisible by 256, got {cols}"
        ));
    }
    if hidden_input.len() != seq_len.saturating_mul(cols) {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain input length mismatch: got {}, expected {}",
            hidden_input.len(),
            seq_len.saturating_mul(cols)
        ));
    }
    if hidden.len() != seq_len.saturating_mul(n_embd) {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain hidden length mismatch: got {}, expected {}",
            hidden.len(),
            seq_len.saturating_mul(n_embd)
        ));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain(
            q_weights,
            q_rows,
            blocks_per_row,
            seq_len,
            kv_len,
            hidden_input,
            None,
            attn_norm_weight,
            q_norm,
            freq_factors,
            cached_k_f16,
            cached_v_f16,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
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
            hidden,
            None,
            None,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map(|result| result.is_some())
}

#[allow(clippy::too_many_arguments)]
pub fn q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain_device_output(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    seq_len: usize,
    kv_len: usize,
    hidden_input: &[f32],
    hidden_input_device: Option<(
        rnb_backend_api::DeviceTensorId,
        rnb_backend_api::DeviceTensorDesc,
    )>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    freq_factors: Option<&[f32]>,
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
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
    device_output_desc: rnb_backend_api::DeviceTensorDesc,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<rnb_backend_api::DeviceTensorId>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain device-output cols must be divisible by 256, got {cols}"
        ));
    }
    let expected_hidden = seq_len.saturating_mul(n_embd);
    if let Some((_, desc)) = hidden_input_device {
        if desc.rows() != seq_len
            || desc.cols() != cols
            || desc.dtype() != rnb_backend_api::ScalarType::F32
        {
            return Err(format!(
                "Q4_K F16 Q cached hd512 dense chain device input desc mismatch: got rows={} cols={} dtype={:?}, expected rows={seq_len} cols={cols} dtype=F32",
                desc.rows(),
                desc.cols(),
                desc.dtype()
            ));
        }
    } else if hidden_input.len() != seq_len.saturating_mul(cols) {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain device-output input length mismatch: got {}, expected {}",
            hidden_input.len(),
            seq_len.saturating_mul(cols)
        ));
    }
    if hidden.len() != expected_hidden {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain device-output hidden length mismatch: got {}, expected {}",
            hidden.len(),
            expected_hidden
        ));
    }
    if device_output_desc.rows() != seq_len
        || device_output_desc.cols() != n_embd
        || device_output_desc.dtype() != rnb_backend_api::ScalarType::F32
    {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain device output desc mismatch: got rows={} cols={} dtype={:?}, expected rows={seq_len} cols={n_embd} dtype=F32",
            device_output_desc.rows(),
            device_output_desc.cols(),
            device_output_desc.dtype()
        ));
    }
    if let Some(scale) = layer_out_scale {
        if scale.is_empty() {
            return Err(
                "Q4_K F16 Q cached hd512 dense chain device-output out_scale must be non-empty"
                    .to_string(),
            );
        }
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 Q cached hd512 dense chain device-output q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let output = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain(
            q_weights,
            q_rows,
            blocks_per_row,
            seq_len,
            kv_len,
            hidden_input,
            hidden_input_device,
            attn_norm_weight,
            q_norm,
            freq_factors,
            cached_k_f16,
            cached_v_f16,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
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
            hidden,
            layer_out_scale,
            Some(device_output_desc),
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    match output {
        Some(Q4kF16DenseChainOutput::Device(id)) => Ok(Some(id)),
        Some(Q4kF16DenseChainOutput::Host) => Err(
            "Q4_K F16 Q cached hd512 dense chain returned host output for device-output request"
                .to_string(),
        ),
        None => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    seq_len: usize,
    kv_len: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
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
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<bool, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain cols must be divisible by 256, got {cols}"
        ));
    }
    if hidden_input.len() != seq_len.saturating_mul(cols) {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain input length mismatch: got {}, expected {}",
            hidden_input.len(),
            seq_len.saturating_mul(cols)
        ));
    }
    if hidden.len() != seq_len.saturating_mul(n_embd) {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain hidden length mismatch: got {}, expected {}",
            hidden.len(),
            seq_len.saturating_mul(n_embd)
        ));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain(
            q_weights,
            q_rows,
            blocks_per_row,
            seq_len,
            kv_len,
            hidden_input,
            None,
            attn_norm_weight,
            q_norm,
            cached_k_f16,
            cached_v_f16,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            window,
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
            hidden,
            None,
            None,
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
        .map(|result| result.is_some())
}

#[allow(clippy::too_many_arguments)]
pub fn q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain_device_output(
    q_weights: &[u8],
    q_rows: usize,
    cols: usize,
    seq_len: usize,
    kv_len: usize,
    hidden_input: &[f32],
    hidden_input_device: Option<(
        rnb_backend_api::DeviceTensorId,
        rnb_backend_api::DeviceTensorDesc,
    )>,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    cached_k_f16: &[u16],
    cached_v_f16: &[u16],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    norm_eps: f32,
    q_unit_offset: bool,
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
    device_output_desc: rnb_backend_api::DeviceTensorDesc,
    unit_offset_attn_norm: bool,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<Option<rnb_backend_api::DeviceTensorId>, String> {
    if cols % 256 != 0 {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain device-output cols must be divisible by 256, got {cols}"
        ));
    }
    let expected_hidden = seq_len.saturating_mul(n_embd);
    if let Some((_, desc)) = hidden_input_device {
        if desc.rows() != seq_len
            || desc.cols() != cols
            || desc.dtype() != rnb_backend_api::ScalarType::F32
        {
            return Err(format!(
                "Q4_K F16 Q hd256 cached dense chain device input desc mismatch: got rows={} cols={} dtype={:?}, expected rows={seq_len} cols={cols} dtype=F32",
                desc.rows(),
                desc.cols(),
                desc.dtype()
            ));
        }
    } else if hidden_input.len() != seq_len.saturating_mul(cols) {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain device-output input length mismatch: got {}, expected {}",
            hidden_input.len(),
            seq_len.saturating_mul(cols)
        ));
    }
    if hidden.len() != expected_hidden {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain device-output hidden length mismatch: got {}, expected {}",
            hidden.len(),
            expected_hidden
        ));
    }
    if device_output_desc.rows() != seq_len
        || device_output_desc.cols() != n_embd
        || device_output_desc.dtype() != rnb_backend_api::ScalarType::F32
    {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain device output desc mismatch: got rows={} cols={} dtype={:?}, expected rows={seq_len} cols={n_embd} dtype=F32",
            device_output_desc.rows(),
            device_output_desc.cols(),
            device_output_desc.dtype()
        ));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "Q4_K F16 Q hd256 cached dense chain device-output q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let output = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain(
            q_weights,
            q_rows,
            blocks_per_row,
            seq_len,
            kv_len,
            hidden_input,
            hidden_input_device,
            attn_norm_weight,
            q_norm,
            cached_k_f16,
            cached_v_f16,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            window,
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
            hidden,
            layer_out_scale,
            Some(device_output_desc),
            unit_offset_attn_norm,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )?;
    match output {
        Some(Q4kF16DenseChainOutput::Device(id)) => Ok(Some(id)),
        Some(Q4kF16DenseChainOutput::Host) => Err(
            "Q4_K F16 Q hd256 cached dense chain returned host output for device-output request"
                .to_string(),
        ),
        None => Ok(None),
    }
}

pub fn q6k_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q6_K cols must be divisible by 256, got {cols}"));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q6_K batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 210;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q6_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q6k_gemv_batch(weights, rows, blocks_per_row, seq_len, input)
}

pub fn q5k_gemv_batch(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols % 256 != 0 {
        return Err(format!("Q5_K cols must be divisible by 256, got {cols}"));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "Q5_K batch input length {} is not divisible by cols {cols}",
            input.len()
        ));
    }
    let seq_len = input.len() / cols;
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row * 176;
    if weights.len() != rows * row_bytes {
        return Err(format!(
            "Q5_K weight byte mismatch: got {}, expected {}",
            weights.len(),
            rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q5k_gemv_batch(weights, rows, blocks_per_row, seq_len, input)
}
