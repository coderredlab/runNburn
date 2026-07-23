//! Runtime platform description and low-level platform policy.
//!
//! This crate is the destination for decisions that depend on operating
//! system, CPU architecture, and device form factor. Higher-level runtime
//! code can ask for a `RuntimeTarget` instead of spreading `cfg!` checks or
//! Android CPU-affinity details through session/model logic.

pub mod android;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatingSystem {
    Linux,
    Android,
    Windows,
    Macos,
    Ios,
    Unknown,
}

impl OperatingSystem {
    pub fn current() -> Self {
        if cfg!(target_os = "linux") {
            Self::Linux
        } else if cfg!(target_os = "android") {
            Self::Android
        } else if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::Macos
        } else if cfg!(target_os = "ios") {
            Self::Ios
        } else {
            Self::Unknown
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuArch {
    X86_64,
    Aarch64,
    Unknown,
}

impl CpuArch {
    pub fn current() -> Self {
        if cfg!(target_arch = "x86_64") {
            Self::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Self::Aarch64
        } else {
            Self::Unknown
        }
    }
}

pub fn aarch64_has_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn aarch64_has_i8mm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("i8mm")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Returns physical host RAM visible to the current operating system.
///
/// Linux and Android expose this through `sysconf`; container or application
/// memory limits remain separate runtime policy inputs.
pub fn host_physical_memory_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages <= 0 || page_size <= 0 {
            return None;
        }
        (pages as u64).checked_mul(page_size as u64)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn parse_host_available_memory_bytes(meminfo: &str) -> Option<u64> {
    let kib = meminfo.lines().find_map(|line| {
        let mut fields = line.strip_prefix("MemAvailable:")?.split_ascii_whitespace();
        let kib = fields.next()?.parse::<u64>().ok()?;
        (fields.next() == Some("kB") && fields.next().is_none()).then_some(kib)
    })?;
    kib.checked_mul(1024)
}

/// Returns memory the operating system can currently provide without swapping.
pub fn host_available_memory_bytes() -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|meminfo| parse_host_available_memory_bytes(&meminfo))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormFactor {
    Desktop,
    Mobile,
    Server,
    Unknown,
}

impl FormFactor {
    pub fn inferred(os: OperatingSystem) -> Self {
        match os {
            OperatingSystem::Android | OperatingSystem::Ios => Self::Mobile,
            OperatingSystem::Linux
            | OperatingSystem::Windows
            | OperatingSystem::Macos
            | OperatingSystem::Unknown => Self::Desktop,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeTarget {
    pub os: OperatingSystem,
    pub arch: CpuArch,
    pub form_factor: FormFactor,
}

impl RuntimeTarget {
    pub fn current() -> Self {
        let os = OperatingSystem::current();
        Self {
            os,
            arch: CpuArch::current(),
            form_factor: FormFactor::inferred(os),
        }
    }

    pub const fn new(os: OperatingSystem, arch: CpuArch, form_factor: FormFactor) -> Self {
        Self {
            os,
            arch,
            form_factor,
        }
    }

    pub const fn is_android(self) -> bool {
        matches!(self.os, OperatingSystem::Android)
    }

    pub const fn is_aarch64(self) -> bool {
        matches!(self.arch, CpuArch::Aarch64)
    }

    pub const fn is_mobile(self) -> bool {
        matches!(self.form_factor, FormFactor::Mobile)
    }

    pub const fn is_desktop(self) -> bool {
        matches!(self.form_factor, FormFactor::Desktop)
    }
}

/// Vulkan execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanExecutionPath {
    /// Single-stage GPU offload (output projection / embed lookup / batched
    /// prefill GEMV) with CPU fallback for the rest. Vendor-agnostic safe.
    Partial,
    /// All transformer stages execute on the GPU end-to-end, no host roundtrip
    /// per layer. Requires vendor-level fp ops parity with the CPU reference;
    /// only desktop discrete GPUs (NVIDIA, AMD, Intel) qualify.
    Fullpath,
}

/// Default Vulkan execution path for the current build target.
///
/// mv31 (2026-05-06) 결론:
/// - mobile-class GPU (Adreno / Mali): fp ops 정밀도가 ARM CPU와 비트 일치 X.
///   24-layer fullpath 누적 fp drift → token divergence. partial path 강제.
/// - desktop GPU: NVIDIA/AMD/Intel 모두 vendor 기본 fp 정밀도가 mobile-class
///   보다 robust. fullpath default ON. vendor별 추가 검증은 PC Vulkan 트랙.
///
/// Policy owner: `rnb-platform`. backend crate (`rnb-backend/vulkan/`) 는
/// vendor-agnostic SPIR-V emit / dispatch 인프라만 제공하고, runtime path
/// 선택은 platform / runtime 정책 layer가 결정한다 —
/// `docs/superpowers/specs/2026-04-28-runtime-crate-boundaries-design.md` 참조.
pub fn vulkan_default_path() -> VulkanExecutionPath {
    vulkan_default_path_for(RuntimeTarget::current())
}

/// Same as `vulkan_default_path` but takes an explicit target so tests /
/// cross-build smoke can simulate a different platform.
pub const fn vulkan_default_path_for(target: RuntimeTarget) -> VulkanExecutionPath {
    if target.is_mobile() {
        VulkanExecutionPath::Partial
    } else {
        VulkanExecutionPath::Fullpath
    }
}

/// Whether `RNB_GPU_FULLPATH=1` env override should be honored. mobile은
/// vendor 검증 안 됐으니 env override 차단 (warn 권장). desktop은 default
/// fullpath이지만 env로 partial 강제 가능.
pub const fn vulkan_fullpath_env_honored(target: RuntimeTarget) -> bool {
    !target.is_mobile()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_target_has_known_os_and_arch() {
        let target = RuntimeTarget::current();
        assert_ne!(target.os, OperatingSystem::Unknown);
        assert_ne!(target.arch, CpuArch::Unknown);
    }

    #[cfg(unix)]
    #[test]
    fn physical_host_memory_is_detectable() {
        assert!(host_physical_memory_bytes().is_some_and(|bytes| bytes > 0));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn available_host_memory_is_detectable() {
        assert!(host_available_memory_bytes().is_some_and(|bytes| bytes > 0));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn parses_mem_available_without_accepting_other_fields() {
        let meminfo =
            "MemTotal:       65536 kB\nMemFree:         1024 kB\nMemAvailable:   32768 kB\n";
        assert_eq!(
            parse_host_available_memory_bytes(meminfo),
            Some(32 * 1024 * 1024)
        );
        assert_eq!(
            parse_host_available_memory_bytes("MemFree: 32768 kB\n"),
            None
        );
    }
    #[test]
    fn form_factor_infers_mobile_for_mobile_operating_systems() {
        assert_eq!(
            FormFactor::inferred(OperatingSystem::Android),
            FormFactor::Mobile
        );
        assert_eq!(
            FormFactor::inferred(OperatingSystem::Ios),
            FormFactor::Mobile
        );
    }

    #[test]
    fn runtime_target_exposes_platform_axes() {
        let target = RuntimeTarget::new(
            OperatingSystem::Android,
            CpuArch::Aarch64,
            FormFactor::Mobile,
        );

        assert!(target.is_android());
        assert!(target.is_aarch64());
        assert!(target.is_mobile());
        assert!(!target.is_desktop());
    }

    #[test]
    fn aarch64_feature_helpers_are_false_off_aarch64() {
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!aarch64_has_dotprod());
            assert!(!aarch64_has_i8mm());
        }

        #[cfg(target_arch = "aarch64")]
        {
            let _ = aarch64_has_dotprod();
            let _ = aarch64_has_i8mm();
        }
    }
}
