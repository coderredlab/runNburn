use super::types::{GGML_Q4_K, GGML_Q6_K, GGML_Q8_0};

pub(in crate::runtime) fn validate_mtp_verify_prefix_tokens(
    window_tokens: usize,
    prefix_tokens: &[usize],
) -> Result<(), String> {
    for &prefix in prefix_tokens {
        if prefix == 0 {
            return Err("MTP verify prefix token index must be > 0".to_string());
        }
        if prefix > window_tokens {
            return Err(format!(
                "MTP verify prefix token index must be <= window_tokens: prefix={prefix}, window_tokens={window_tokens}"
            ));
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn validate_mtp_verify_q4k_matrix(
    label: &str,
    weights: &[u8],
    rows: usize,
    cols: usize,
    expected_cols: usize,
) -> Result<usize, String> {
    if rows == 0 {
        return Err(format!("MTP verify {label} rows must be non-zero"));
    }
    if cols != expected_cols {
        return Err(format!(
            "MTP verify {label} cols must match hidden_dim: cols={cols}, hidden_dim={expected_cols}"
        ));
    }
    if cols == 0 || cols % 256 != 0 {
        return Err(format!(
            "MTP verify {label} Q4_K cols must be non-zero and divisible by 256, got {cols}"
        ));
    }
    let blocks_per_row = cols / 256;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(144))
        .ok_or_else(|| {
            format!("MTP verify {label} Q4_K byte size overflow: rows={rows} cols={cols}")
        })?;
    if weights.len() != expected {
        return Err(format!(
            "MTP verify {label} Q4_K byte mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }
    Ok(blocks_per_row)
}

pub(super) fn validate_mtp_verify_q6k_matrix(
    label: &str,
    weights: &[u8],
    rows: usize,
    cols: usize,
    expected_cols: usize,
) -> Result<usize, String> {
    if rows == 0 {
        return Err(format!("MTP verify {label} rows must be non-zero"));
    }
    if cols != expected_cols {
        return Err(format!(
            "MTP verify {label} cols must match hidden_dim: cols={cols}, hidden_dim={expected_cols}"
        ));
    }
    if cols == 0 || cols % 256 != 0 {
        return Err(format!(
            "MTP verify {label} Q6_K cols must be non-zero and divisible by 256, got {cols}"
        ));
    }
    let blocks_per_row = cols / 256;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(210))
        .ok_or_else(|| {
            format!("MTP verify {label} Q6_K byte size overflow: rows={rows} cols={cols}")
        })?;
    if weights.len() != expected {
        return Err(format!(
            "MTP verify {label} Q6_K byte mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }
    Ok(blocks_per_row)
}

pub(super) fn validate_mtp_verify_q8_0_matrix(
    label: &str,
    weights: &[u8],
    rows: usize,
    cols: usize,
    expected_cols: usize,
) -> Result<usize, String> {
    if rows == 0 {
        return Err(format!("MTP verify {label} rows must be non-zero"));
    }
    if cols != expected_cols {
        return Err(format!(
            "MTP verify {label} cols must match hidden_dim: cols={cols}, hidden_dim={expected_cols}"
        ));
    }
    if cols == 0 || cols % 32 != 0 {
        return Err(format!(
            "MTP verify {label} Q8_0 cols must be non-zero and divisible by 32, got {cols}"
        ));
    }
    let blocks_per_row = cols / 32;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(34))
        .ok_or_else(|| {
            format!("MTP verify {label} Q8_0 byte size overflow: rows={rows} cols={cols}")
        })?;
    if weights.len() != expected {
        return Err(format!(
            "MTP verify {label} Q8_0 byte mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }
    Ok(blocks_per_row)
}

pub(super) fn validate_mtp_verify_f32_matrix(
    label: &str,
    weights: &[f32],
    rows: usize,
    cols: usize,
    expected_cols: usize,
) -> Result<(), String> {
    if rows == 0 {
        return Err(format!("MTP verify {label} rows must be non-zero"));
    }
    if cols != expected_cols {
        return Err(format!(
            "MTP verify {label} cols must match hidden_dim: cols={cols}, hidden_dim={expected_cols}"
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("MTP verify {label} F32 size overflow: rows={rows} cols={cols}"))?;
    if weights.len() != expected {
        return Err(format!(
            "MTP verify {label} F32 len mismatch: got {}, expected {expected}",
            weights.len()
        ));
    }
    Ok(())
}

pub(in crate::runtime) fn validate_mtp_verify_k_quant_matrix(
    label: &str,
    quant: u32,
    weights: &[u8],
    rows: usize,
    cols: usize,
    expected_cols: usize,
) -> Result<usize, String> {
    match quant {
        GGML_Q4_K => validate_mtp_verify_q4k_matrix(label, weights, rows, cols, expected_cols),
        GGML_Q6_K => validate_mtp_verify_q6k_matrix(label, weights, rows, cols, expected_cols),
        GGML_Q8_0 => validate_mtp_verify_q8_0_matrix(label, weights, rows, cols, expected_cols),
        other => Err(format!(
            "MTP verify {label} quant must be Q4_K, Q6_K or Q8_0, got {other}"
        )),
    }
}
