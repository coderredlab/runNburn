pub fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}

pub fn env_present_os(name: &str) -> bool {
    std::env::var_os(name).is_some()
}
pub fn env_os_string(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name)
}

pub fn env_flag_default_on(name: &str) -> bool {
    std::env::var(name).map(|v| v != "0").unwrap_or(true)
}

pub fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

pub fn env_truthy_override(name: &str) -> Option<bool> {
    env_string(name).map(|value| {
        let value = value.to_ascii_lowercase();
        !matches!(value.as_str(), "0" | "false" | "off" | "no")
    })
}

pub fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

pub fn env_isize(name: &str) -> Option<isize> {
    std::env::var(name).ok()?.parse().ok()
}

pub fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.parse().ok()
}

pub fn env_terms(name: &str) -> Vec<String> {
    env_string(name)
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|term| !term.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

pub fn layer_matches_spec(raw: &str, layer_idx: usize) -> bool {
    for term in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((start, end)) = term.split_once('-') {
            if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                if start <= layer_idx && layer_idx <= end {
                    return true;
                }
            }
        } else if let Ok(want) = term.parse::<usize>() {
            if want == layer_idx {
                return true;
            }
        }
    }
    false
}

pub fn env_layer_matches(name: &str, layer_idx: usize) -> bool {
    env_string(name)
        .as_deref()
        .is_some_and(|raw| layer_matches_spec(raw, layer_idx))
}
