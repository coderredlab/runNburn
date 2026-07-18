#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum PrefillProjectionPath {
    QuantizedGemv,
    F32Gemm,
}

const DEFAULT_QUANTIZED_MAX_SEQ: usize = 64;

pub(in crate::engine) fn prefill_projection_path(
    seq_len: usize,
    legacy_q_env_name: &str,
) -> PrefillProjectionPath {
    let unified = crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_PROJECTION");
    let legacy = crate::engine::policy::env_string(legacy_q_env_name);
    let max_seq = crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_PREFILL_Q_GEMV_MAX_SEQ");
    prefill_projection_path_from_env(
        seq_len,
        unified.as_deref(),
        legacy.as_deref(),
        max_seq.as_deref(),
    )
}

pub(in crate::engine) fn prefill_projection_path_from_env(
    seq_len: usize,
    unified_override: Option<&str>,
    legacy_q_override: Option<&str>,
    q_max_seq_override: Option<&str>,
) -> PrefillProjectionPath {
    if let Some(path) = unified_override.and_then(parse_projection_override) {
        return path;
    }
    if let Some(path) = legacy_q_override.map(parse_legacy_q_override) {
        return path;
    }

    let max_quantized_seq = q_max_seq_override
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_QUANTIZED_MAX_SEQ);
    if seq_len > 1 && seq_len <= max_quantized_seq {
        PrefillProjectionPath::QuantizedGemv
    } else {
        PrefillProjectionPath::F32Gemm
    }
}

fn parse_projection_override(raw: &str) -> Option<PrefillProjectionPath> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "default" => None,
        "q" | "quant" | "quantized" | "qgemv" | "quantized_gemv" => {
            Some(PrefillProjectionPath::QuantizedGemv)
        }
        "f32" | "gemm" | "f32gemm" | "f32_gemm" | "0" | "false" | "off" | "no" => {
            Some(PrefillProjectionPath::F32Gemm)
        }
        _ => None,
    }
}

fn parse_legacy_q_override(raw: &str) -> PrefillProjectionPath {
    match raw.trim().to_ascii_lowercase().as_str() {
        "0" | "false" | "off" | "no" => PrefillProjectionPath::F32Gemm,
        _ => PrefillProjectionPath::QuantizedGemv,
    }
}

#[cfg(test)]
mod tests {
    use super::{prefill_projection_path_from_env, PrefillProjectionPath};

    #[test]
    fn nemotron_projection_policy_keeps_short_prefill_on_quantized_path() {
        assert_eq!(
            prefill_projection_path_from_env(29, None, None, None),
            PrefillProjectionPath::QuantizedGemv
        );
    }

    #[test]
    fn nemotron_projection_policy_moves_long_prefill_to_f32_path() {
        assert_eq!(
            prefill_projection_path_from_env(128, None, None, None),
            PrefillProjectionPath::F32Gemm
        );
    }

    #[test]
    fn nemotron_projection_policy_preserves_legacy_explicit_q_override() {
        assert_eq!(
            prefill_projection_path_from_env(128, None, Some("1"), None),
            PrefillProjectionPath::QuantizedGemv
        );
    }

    #[test]
    fn nemotron_projection_policy_preserves_legacy_explicit_f32_override() {
        assert_eq!(
            prefill_projection_path_from_env(29, None, Some("0"), None),
            PrefillProjectionPath::F32Gemm
        );
    }

    #[test]
    fn nemotron_projection_policy_allows_global_projection_override() {
        assert_eq!(
            prefill_projection_path_from_env(128, Some("q"), None, None),
            PrefillProjectionPath::QuantizedGemv
        );
        assert_eq!(
            prefill_projection_path_from_env(29, Some("f32"), None, None),
            PrefillProjectionPath::F32Gemm
        );
    }

    #[test]
    fn nemotron_projection_policy_allows_seq_threshold_override() {
        assert_eq!(
            prefill_projection_path_from_env(96, None, None, Some("128")),
            PrefillProjectionPath::QuantizedGemv
        );
        assert_eq!(
            prefill_projection_path_from_env(129, None, None, Some("128")),
            PrefillProjectionPath::F32Gemm
        );
    }
}
