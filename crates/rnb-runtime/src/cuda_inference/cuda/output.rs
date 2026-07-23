use rnb_loader::GGMLType;

use super::backend;

pub fn prefill_output_logits(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    normed_data: &[f32],
) -> Option<Vec<f32>> {
    if !backend::tuning::prefill_output_logits_requested()
        || !backend::tuning::output_logits_enabled()
    {
        return None;
    }
    output_logits_impl(ggml_type, rows, cols, raw, normed_data, false)
}

pub fn try_output_logits(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    normed_data: &[f32],
) -> Option<Vec<f32>> {
    output_logits_impl(ggml_type, rows, cols, raw, normed_data, true)
}

pub fn try_output_logits_if_enabled(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    normed_data: &[f32],
    token_embedding_output: bool,
) -> Option<Vec<f32>> {
    if token_embedding_output || !output_logits_enabled() {
        return None;
    }
    try_output_logits(ggml_type, rows, cols, raw, normed_data)
}

pub fn try_output_logits_into_if_enabled(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    normed_data: &[f32],
    token_embedding_output: bool,
    out: &mut [f32],
    write_logits: bool,
) -> Option<u32> {
    if token_embedding_output || !output_logits_enabled() || out.len() < rows {
        return None;
    }
    if backend::tuning::output_argmax_enabled() && ggml_type == GGMLType::Q8_0 {
        let q8dot = backend::tuning::q8_0_output_q8dot_argmax_enabled();
        let result = if q8dot {
            backend::q8_0_gemv_argmax_q8dot(raw, rows, cols, normed_data)
        } else {
            backend::q8_0_gemv_argmax(raw, rows, cols, normed_data)
        };
        let Ok((token, value)) = result else {
            return None;
        };
        let token = token as usize;
        if token >= rows {
            return None;
        }
        if write_logits {
            out[..rows].fill(f32::NEG_INFINITY);
            out[token] = value;
        }
        return Some(token as u32);
    }
    if backend::tuning::output_argmax_enabled() && ggml_type == GGMLType::Q6_K {
        let Ok((token, value)) = backend::q6k_gemv_argmax(raw, rows, cols, normed_data) else {
            return None;
        };
        let token = token as usize;
        if token >= rows {
            return None;
        }
        if write_logits {
            out[..rows].fill(f32::NEG_INFINITY);
            out[token] = value;
        }
        return Some(token as u32);
    }
    None
}

pub fn try_output_argmax_token(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    normed_data: &[f32],
) -> Option<u32> {
    let result = match ggml_type {
        GGMLType::Q8_0 => {
            let q8dot = backend::tuning::q8_0_output_q8dot_argmax_enabled();
            if q8dot {
                backend::q8_0_gemv_argmax_q8dot(raw, rows, cols, normed_data)
            } else {
                backend::q8_0_gemv_argmax(raw, rows, cols, normed_data)
            }
        }
        GGMLType::Q6_K => backend::q6k_gemv_argmax(raw, rows, cols, normed_data),
        _ => {
            let logits =
                super::prefill::decode_gemv(ggml_type, raw, rows, cols, normed_data)?.ok()?;
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                .map(|(token, _)| token as u32);
        }
    };
    let Ok((token, _)) = result else {
        return None;
    };
    ((token as usize) < rows).then_some(token)
}

fn output_logits_impl(
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    normed_data: &[f32],
    argmax_requires_valid_token: bool,
) -> Option<Vec<f32>> {
    if backend::tuning::output_argmax_enabled() && ggml_type == GGMLType::Q8_0 {
        let q8dot = backend::tuning::q8_0_output_q8dot_argmax_enabled();
        let result = if q8dot {
            backend::q8_0_gemv_argmax_q8dot(raw, rows, cols, normed_data)
        } else {
            backend::q8_0_gemv_argmax(raw, rows, cols, normed_data)
        };
        if let Some(logits) = result.ok().and_then(|(token, value)| {
            sparse_argmax_logits(rows, token, value, argmax_requires_valid_token)
        }) {
            return Some(logits);
        }
    }
    if backend::tuning::output_argmax_enabled() && ggml_type == GGMLType::Q6_K {
        if let Some(logits) = backend::q6k_gemv_argmax(raw, rows, cols, normed_data)
            .ok()
            .and_then(|(token, value)| {
                sparse_argmax_logits(rows, token, value, argmax_requires_valid_token)
            })
        {
            return Some(logits);
        }
    }
    super::prefill::decode_gemv(ggml_type, raw, rows, cols, normed_data)?.ok()
}

fn sparse_argmax_logits(
    rows: usize,
    token: u32,
    value: f32,
    argmax_requires_valid_token: bool,
) -> Option<Vec<f32>> {
    let mut out = vec![f32::NEG_INFINITY; rows];
    if let Some(logit) = out.get_mut(token as usize) {
        *logit = value;
        return Some(out);
    }
    (!argmax_requires_valid_token).then_some(out)
}

pub fn output_logits_enabled() -> bool {
    backend::tuning::output_logits_enabled()
}

pub fn prewarm_output_weight(ggml_type: GGMLType, rows: usize, cols: usize, raw: &[u8]) -> bool {
    if ggml_type != GGMLType::Q8_0
        || rows == 0
        || cols == 0
        || !backend::tuning::output_logits_enabled()
    {
        return false;
    }
    backend::prewarm_q4k_weights(&[raw]).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .get_or_init(Default::default)
            .lock()
            .expect("CUDA output env lock poisoned")
    }

    fn q8_0_rows_i8(rows: &[&[i8; 32]]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(rows.len() * 34);
        for row in rows {
            raw.extend_from_slice(&[0x00, 0x3c]);
            raw.extend(row.iter().map(|&value| value as u8));
        }
        raw
    }

    #[test]
    fn q8_0_output_argmax_returns_sparse_logits() {
        let _guard = env_lock();
        let old_argmax = std::env::var("RNB_CUDA_OUTPUT_ARGMAX").ok();
        let old_q8dot = std::env::var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX").ok();
        std::env::set_var("RNB_CUDA_OUTPUT_ARGMAX", "1");
        std::env::set_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", "1");

        let row0 = [1i8; 32];
        let row1 = [2i8; 32];
        let row2 = [-1i8; 32];
        let weights = q8_0_rows_i8(&[&row0, &row1, &row2]);
        let input = vec![1.0f32; 32];
        let logits = try_output_logits(GGMLType::Q8_0, 3, 32, &weights, &input)
            .expect("CUDA Q8_0 sparse output logits");

        let finite = logits
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        assert_eq!(finite.len(), 1);
        assert_eq!(finite[0].0, 1);

        match old_argmax {
            Some(value) => std::env::set_var("RNB_CUDA_OUTPUT_ARGMAX", value),
            None => std::env::remove_var("RNB_CUDA_OUTPUT_ARGMAX"),
        }
        match old_q8dot {
            Some(value) => std::env::set_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", value),
            None => std::env::remove_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX"),
        }
    }

    #[test]
    fn q8_0_output_argmax_token_returns_best_row() {
        let _guard = env_lock();
        let old_q8dot = std::env::var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX").ok();
        std::env::set_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", "1");

        let row0 = [1i8; 32];
        let row1 = [3i8; 32];
        let row2 = [-1i8; 32];
        let weights = q8_0_rows_i8(&[&row0, &row1, &row2]);
        let input = vec![1.0f32; 32];
        let token = try_output_argmax_token(GGMLType::Q8_0, 3, 32, &weights, &input)
            .expect("CUDA Q8_0 output argmax token");

        assert_eq!(token, 1);

        match old_q8dot {
            Some(value) => std::env::set_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX", value),
            None => std::env::remove_var("RNB_CUDA_Q8_0_OUTPUT_Q8DOT_ARGMAX"),
        }
    }
}
