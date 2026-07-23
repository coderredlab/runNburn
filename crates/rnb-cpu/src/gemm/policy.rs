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
}
