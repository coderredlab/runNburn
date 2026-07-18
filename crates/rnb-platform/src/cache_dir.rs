//! Platform-aware resolver for the runNburn user cache directory.
//!
//! The cache holds runtime artifacts (e.g. packed `.rnb` sidecars built from
//! GGUF models). Picking the right location depends on the host OS and on
//! environment overrides — both of which are platform concerns, not loader
//! or memory-policy concerns. Higher layers ask `resolve_cache_dir()` for the
//! base directory and then organise files inside it.
//!
//! Priority:
//!
//! 1. `RNB_CACHE_DIR` environment variable (any OS, opt-in override)
//! 2. OS-standard cache root:
//!    - Linux: `$XDG_CACHE_HOME/runNburn` or `$HOME/.cache/runNburn`
//!    - macOS: `$HOME/Library/Caches/runNburn`
//!    - Windows: `%LOCALAPPDATA%\runNburn\cache`
//!    - Android: there is no portable default — caller (FFI host) MUST set
//!      `RNB_CACHE_DIR` to the app's private cache (`Context.cacheDir`) or
//!      `/data/local/tmp/rnb/cache/` for ADB harnesses.
//!    - Other targets: not supported.

use std::path::PathBuf;

use crate::OperatingSystem;

const ENV_OVERRIDE: &str = "RNB_CACHE_DIR";
const APP_DIR_NAME: &str = "runNburn";

/// Resolve the runNburn cache root directory for the current platform.
pub fn resolve_cache_dir() -> Result<PathBuf, String> {
    if let Ok(env_dir) = std::env::var(ENV_OVERRIDE) {
        if !env_dir.is_empty() {
            return Ok(PathBuf::from(env_dir));
        }
    }
    resolve_for(OperatingSystem::current())
}

/// Resolve the cache root for an explicitly given OS, ignoring `RNB_CACHE_DIR`.
/// Exposed for testing and for callers that want to drive the policy from a
/// known [`OperatingSystem`] (e.g. cross-platform diagnostics).
pub fn resolve_for(os: OperatingSystem) -> Result<PathBuf, String> {
    match os {
        OperatingSystem::Linux => linux_xdg_cache().map(|base| base.join(APP_DIR_NAME)),
        OperatingSystem::Macos => {
            home_dir().map(|home| home.join("Library/Caches").join(APP_DIR_NAME))
        }
        OperatingSystem::Windows => {
            env_dir("LOCALAPPDATA").map(|base| base.join(APP_DIR_NAME).join("cache"))
        }
        OperatingSystem::Android => Err(format!(
            "{ENV_OVERRIDE} must be set on Android (no portable default)"
        )),
        OperatingSystem::Ios => Err(format!(
            "{ENV_OVERRIDE} must be set on iOS (no portable default)"
        )),
        OperatingSystem::Unknown => Err("unsupported OS for cache resolution".into()),
    }
}

fn linux_xdg_cache() -> Result<PathBuf, String> {
    if let Some(xdg) = env_dir("XDG_CACHE_HOME").ok() {
        return Ok(xdg);
    }
    home_dir().map(|home| home.join(".cache"))
}

fn home_dir() -> Result<PathBuf, String> {
    env_dir("HOME")
}

fn env_dir(name: &str) -> Result<PathBuf, String> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|_| format!("environment variable {name} is not set"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Cache-dir tests mutate process-global env vars, so they must not run in
    /// parallel against each other. We serialise them with a `Mutex`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce()>(set: &[(&str, &str)], unset: &[&str], f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: tests are serialised by `ENV_LOCK`; there are no other
        // threads racing on these vars during the closure.
        unsafe {
            for name in unset {
                std::env::remove_var(name);
            }
            for (k, v) in set {
                std::env::set_var(k, v);
            }
        }
        f();
        unsafe {
            for (k, _) in set {
                std::env::remove_var(k);
            }
        }
    }

    #[test]
    fn env_override_takes_priority_over_os_default() {
        with_env(
            &[(ENV_OVERRIDE, "/tmp/test-cache-runnburn-12345")],
            &[],
            || {
                let loc = resolve_cache_dir().expect("override must resolve");
                assert_eq!(loc, PathBuf::from("/tmp/test-cache-runnburn-12345"));
            },
        );
    }

    #[test]
    fn empty_env_override_falls_through_to_os_default() {
        with_env(
            &[
                (ENV_OVERRIDE, ""),
                ("XDG_CACHE_HOME", "/tmp/xdg-fallthrough"),
            ],
            &[],
            || {
                let loc = resolve_for(OperatingSystem::Linux).unwrap();
                assert_eq!(loc, PathBuf::from("/tmp/xdg-fallthrough/runNburn"));
            },
        );
    }

    #[test]
    fn linux_uses_xdg_cache_home_when_set() {
        with_env(&[("XDG_CACHE_HOME", "/tmp/xdg-cache")], &[], || {
            let loc = resolve_for(OperatingSystem::Linux).unwrap();
            assert_eq!(loc, PathBuf::from("/tmp/xdg-cache/runNburn"));
        });
    }

    #[test]
    fn linux_falls_back_to_home_dot_cache_when_xdg_missing() {
        with_env(
            &[("HOME", "/tmp/home-runnburn")],
            &["XDG_CACHE_HOME"],
            || {
                let loc = resolve_for(OperatingSystem::Linux).unwrap();
                assert_eq!(loc, PathBuf::from("/tmp/home-runnburn/.cache/runNburn"));
            },
        );
    }

    #[test]
    fn macos_uses_library_caches() {
        with_env(&[("HOME", "/Users/runnburn")], &["XDG_CACHE_HOME"], || {
            let loc = resolve_for(OperatingSystem::Macos).unwrap();
            assert_eq!(
                loc,
                PathBuf::from("/Users/runnburn/Library/Caches/runNburn")
            );
        });
    }

    #[test]
    fn windows_uses_localappdata() {
        with_env(
            &[("LOCALAPPDATA", "C:\\Users\\Runnburn\\AppData\\Local")],
            &[],
            || {
                let loc = resolve_for(OperatingSystem::Windows).unwrap();
                assert_eq!(
                    loc,
                    PathBuf::from("C:\\Users\\Runnburn\\AppData\\Local")
                        .join("runNburn")
                        .join("cache")
                );
            },
        );
    }

    #[test]
    fn android_requires_explicit_override() {
        with_env(&[], &[ENV_OVERRIDE], || {
            let err = resolve_for(OperatingSystem::Android).unwrap_err();
            assert!(err.contains(ENV_OVERRIDE));
        });
    }

    #[test]
    fn unknown_os_is_unsupported() {
        with_env(&[], &[ENV_OVERRIDE], || {
            let err = resolve_for(OperatingSystem::Unknown).unwrap_err();
            assert!(err.contains("unsupported"));
        });
    }
}
