//! Sibling drafter file auto-detect for external MTP wiring.

use std::path::{Path, PathBuf};

/// Look for an external drafter GGUF next to the target model.
///
/// Patterns checked in order:
/// 1. `{target_dir}/{target_stem}-assistant.Q4_K_M.gguf`
/// 2. `{target_dir}/{target_stem}-assistant.gguf`
/// 3. `{target_dir}/{target_stem}-mtp/{target_stem}-assistant.*.gguf`
/// 4. `{target_dir_parent}/{target_dir_name}-mtp/{target_stem}-assistant.*.gguf`
///    (sibling-of-parent layout used in this repo, e.g.
///    `models/gemma-4-E4B-mtp/` next to `models/gemma-4-E4B/`)
pub fn find_sibling_drafter(target_path: &Path) -> Option<PathBuf> {
    let dir = target_path.parent()?;
    let stem = target_path.file_stem()?.to_str()?;
    let model_stems = candidate_model_stems(stem);

    for model_stem in &model_stems {
        for name in [
            format!("{model_stem}-assistant.Q4_K_M.gguf"),
            format!("{model_stem}-assistant.gguf"),
        ] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    for model_stem in &model_stems {
        let prefix = format!("{model_stem}-assistant.");
        if let Some(path) = scan_prefixed_gguf(dir, &prefix) {
            return Some(path);
        }
    }

    for model_stem in &model_stems {
        let prefix = format!("{model_stem}-assistant.");
        if let Some(path) = scan_prefixed_gguf(&dir.join(format!("{model_stem}-mtp")), &prefix) {
            return Some(path);
        }
    }

    if let (Some(parent_dir), Some(dir_name)) =
        (dir.parent(), dir.file_name().and_then(|name| name.to_str()))
    {
        for model_stem in &model_stems {
            let prefix = format!("{model_stem}-assistant.");
            if let Some(path) =
                scan_prefixed_gguf(&parent_dir.join(format!("{dir_name}-mtp")), &prefix)
            {
                return Some(path);
            }
        }
    }

    None
}

fn candidate_model_stems(stem: &str) -> Vec<&str> {
    let mut stems = vec![stem];
    if let Some((base, suffix)) = stem.rsplit_once('-') {
        let suffix = suffix.to_ascii_uppercase();
        if suffix.starts_with('Q')
            || suffix.starts_with("IQ")
            || matches!(suffix.as_str(), "F16" | "F32" | "BF16")
        {
            stems.push(base);
        }
    }
    stems
}

fn scan_prefixed_gguf(dir: &Path, prefix: &str) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    let mut matches = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".gguf"))
        })
        .collect::<Vec<_>>();
    matches.sort_unstable();
    matches.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_target_quant_suffix_for_assistant_lookup() {
        assert_eq!(
            candidate_model_stems("gemma-4-E4B-it-Q4_K_M"),
            vec!["gemma-4-E4B-it-Q4_K_M", "gemma-4-E4B-it"]
        );
        assert_eq!(
            candidate_model_stems("gemma-4-E4B-it-BF16"),
            vec!["gemma-4-E4B-it-BF16", "gemma-4-E4B-it"]
        );
        assert_eq!(
            candidate_model_stems("gemma-4-E4B-it"),
            vec!["gemma-4-E4B-it"]
        );
    }

    #[test]
    fn finds_flat_assistant_for_quantized_target() {
        let root = std::env::temp_dir().join(format!(
            "rnb-auto-drafter-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("gemma-4-E4B-it-Q4_K_M.gguf");
        let assistant = root.join("gemma-4-E4B-it-assistant.Q4_K_M.gguf");
        std::fs::write(&target, []).unwrap();
        std::fs::write(&assistant, []).unwrap();

        assert_eq!(find_sibling_drafter(&target), Some(assistant));

        std::fs::remove_dir_all(root).unwrap();
    }
}
