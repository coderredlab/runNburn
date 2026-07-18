use crate::gemm::QuantType;

pub fn force_generic_gemv() -> bool {
    std::env::var("RNB_FORCE_GENERIC_GEMV").is_ok()
}

pub fn force_generic_rows_ge() -> Option<usize> {
    std::env::var("RNB_FORCE_GENERIC_GEMV_ROWS_GE")
        .ok()?
        .parse()
        .ok()
}

pub fn force_generic_cols_ge() -> Option<usize> {
    std::env::var("RNB_FORCE_GENERIC_GEMV_COLS_GE")
        .ok()?
        .parse()
        .ok()
}

pub fn rawmeta_repack_cache_enabled() -> bool {
    std::env::var("RNB_RAWMETA_REPACK_CACHE").ok().as_deref() != Some("0")
}

pub fn q80_packed_i8mm_enabled() -> bool {
    std::env::var("RNB_Q80_PACKED_I8MM").is_ok()
}

pub fn q80_f32_scales_requested() -> bool {
    std::env::var("RNB_Q80_F32_SCALES").is_ok()
}

pub fn aarch64_dotprod_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn aarch64_i8mm_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("i8mm")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn q80_pair_i8mm_supported(cols: usize) -> bool {
    q80_packed_i8mm_enabled() && aarch64_i8mm_available() && cols % 32 == 0
}

pub fn fast_gemv_enabled() -> bool {
    std::env::var("RNB_DISABLE_FAST_GEMV").is_err()
}

pub fn fast_dotprod_enabled() -> bool {
    aarch64_dotprod_available() && fast_gemv_enabled()
}

pub fn force_generic_gemv_for_shape(rows: usize, cols: usize) -> bool {
    if force_generic_gemv() {
        return true;
    }
    if let Some(threshold) = force_generic_rows_ge() {
        if rows >= threshold {
            return true;
        }
    }
    if let Some(threshold) = force_generic_cols_ge() {
        if cols >= threshold {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4KKernelBackend {
    Builtin,
}

pub fn q4k_kernel_backend_from_env(explicit: Option<&str>) -> Option<Q4KKernelBackend> {
    match explicit.map(str::trim).map(str::to_ascii_lowercase) {
        Some(value) if value == "builtin" => Some(Q4KKernelBackend::Builtin),
        Some(_) | None => None,
    }
}

pub fn gemv_q8k_profile_method(packed_quant_type: Option<QuantType>) -> &'static str {
    match packed_quant_type {
        Some(QuantType::Q4KCompact) => "gemv_vec_q8k_compact",
        Some(QuantType::Q4K) | Some(QuantType::Q5K) | Some(QuantType::Q6K) => "gemv_vec_q8k_packed",
        _ => "gemv_vec_q8k",
    }
}

pub fn rawmeta_runtime_repack_enabled(
    _packed_quant_type: Option<QuantType>,
    _is_q4k_weight: bool,
    _seq_len: usize,
    _rows: usize,
    _cols: usize,
) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_kernel_backend_defaults_to_none() {
        assert_eq!(q4k_kernel_backend_from_env(None), None);
    }

    #[test]
    fn q4k_kernel_backend_parses_builtin_only() {
        assert_eq!(
            q4k_kernel_backend_from_env(Some("builtin")),
            Some(Q4KKernelBackend::Builtin)
        );
        assert_eq!(q4k_kernel_backend_from_env(Some("external")), None);
        assert_eq!(q4k_kernel_backend_from_env(Some("external-prefill")), None);
    }

    #[test]
    fn q80_support_helpers_are_false_off_aarch64() {
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!aarch64_dotprod_available());
            assert!(!aarch64_i8mm_available());
            assert!(!q80_pair_i8mm_supported(32));
            assert!(!fast_dotprod_enabled());
        }
    }

    #[test]
    fn gemv_q8k_profile_method_names_packed_variants() {
        assert_eq!(
            gemv_q8k_profile_method(Some(QuantType::Q4KCompact)),
            "gemv_vec_q8k_compact"
        );
        assert_eq!(
            gemv_q8k_profile_method(Some(QuantType::Q4K)),
            "gemv_vec_q8k_packed"
        );
        assert_eq!(gemv_q8k_profile_method(None), "gemv_vec_q8k");
    }

    #[test]
    fn rawmeta_runtime_repack_is_disabled() {
        assert!(!rawmeta_runtime_repack_enabled(
            Some(QuantType::Q4K),
            true,
            2,
            8192,
            1536
        ));
        assert!(!rawmeta_runtime_repack_enabled(None, false, 1, 0, 0));
    }
}
