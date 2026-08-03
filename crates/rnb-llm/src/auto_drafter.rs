//! Sibling drafter file auto-detect for external MTP wiring.

use std::path::{Path, PathBuf};

/// Look for an external drafter GGUF next to the target model.
///
/// 1. `{target_dir}/{target_stem}-assistant.Q4_K_M.gguf`
/// 2. `{target_dir}/{target_stem}-assistant.gguf`
/// 3. `{target_dir}/MTP/mtp-{target_stem}-Q8_0.gguf`
/// 4. `{target_dir}/{target_stem}-mtp/{target_stem}-assistant.*.gguf`
/// 5. `{target_dir_parent}/{target_dir_name}-mtp/{target_stem}-assistant.*.gguf`
///    (sibling-of-parent layout used in this repo, e.g.
///    `models/gemma-4-E4B-mtp/` next to `models/gemma-4-E4B/`)
pub fn find_sibling_drafter(target_path: &Path) -> Option<PathBuf> {
    find_sibling_drafter_candidates(target_path)
        .into_iter()
        .next()
}

pub(crate) fn find_sibling_drafter_candidates(target_path: &Path) -> Vec<PathBuf> {
    let Some(dir) = target_path.parent() else {
        return Vec::new();
    };
    let Some(stem) = target_path.file_stem().and_then(|stem| stem.to_str()) else {
        return Vec::new();
    };
    let model_stems = candidate_model_stems(stem);
    let mut candidates = Vec::new();

    for candidate_dir in [dir.to_path_buf(), dir.join("DSpark"), dir.join("dspark")] {
        for model_stem in &model_stems {
            for name in [
                format!("{model_stem}-DSpark.gguf"),
                format!("dspark-{model_stem}.gguf"),
            ] {
                push_candidate(&mut candidates, candidate_dir.join(name));
            }
            for prefix in [
                format!("{model_stem}-DSpark-"),
                format!("dspark-{model_stem}-"),
            ] {
                extend_prefixed_ggufs(&mut candidates, &candidate_dir, &prefix);
            }
        }
    }

    for model_stem in &model_stems {
        for name in [
            format!("{model_stem}-assistant.Q4_K_M.gguf"),
            format!("{model_stem}-assistant.gguf"),
        ] {
            push_candidate(&mut candidates, dir.join(name));
        }
    }

    for model_stem in &model_stems {
        let name = format!("mtp-{model_stem}-Q8_0.gguf");
        for subdir in ["MTP", "mtp"] {
            push_candidate(&mut candidates, dir.join(subdir).join(&name));
        }
    }

    for model_stem in &model_stems {
        let prefix = format!("{model_stem}-assistant.");
        extend_prefixed_ggufs(&mut candidates, dir, &prefix);
    }

    for model_stem in &model_stems {
        let prefix = format!("{model_stem}-assistant.");
        extend_prefixed_ggufs(
            &mut candidates,
            &dir.join(format!("{model_stem}-mtp")),
            &prefix,
        );
    }

    if let (Some(parent_dir), Some(dir_name)) =
        (dir.parent(), dir.file_name().and_then(|name| name.to_str()))
    {
        for model_stem in &model_stems {
            let prefix = format!("{model_stem}-assistant.");
            extend_prefixed_ggufs(
                &mut candidates,
                &parent_dir.join(format!("{dir_name}-mtp")),
                &prefix,
            );
        }
    }

    candidates
}

fn candidate_model_stems(stem: &str) -> Vec<String> {
    let stem = strip_shard_suffix(stem);
    let mut stems = vec![stem.to_string()];
    if let Some((base, suffix)) = stem.rsplit_once('-') {
        let suffix = suffix.to_ascii_uppercase();
        if suffix.starts_with('Q')
            || suffix.starts_with("IQ")
            || matches!(suffix.as_str(), "F16" | "F32" | "BF16")
        {
            stems.push(base.to_string());
            if let Some(base_without_ud) = base.strip_suffix("-UD") {
                stems.push(base_without_ud.to_string());
            }
        }
    }
    stems
}

fn strip_shard_suffix(stem: &str) -> &str {
    let mut parts = stem.rsplitn(4, '-');
    let Some(total) = parts.next() else {
        return stem;
    };
    let Some(marker) = parts.next() else {
        return stem;
    };
    let Some(index) = parts.next() else {
        return stem;
    };
    let Some(base) = parts.next() else {
        return stem;
    };
    if marker.eq_ignore_ascii_case("of")
        && !index.is_empty()
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && !total.is_empty()
        && total.bytes().all(|byte| byte.is_ascii_digit())
    {
        base
    } else {
        stem
    }
}

fn push_candidate(candidates: &mut Vec<PathBuf>, candidate: PathBuf) {
    if candidate.is_file() && !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

fn extend_prefixed_ggufs(candidates: &mut Vec<PathBuf>, dir: &Path, prefix: &str) {
    if !dir.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut matches = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".gguf"))
        })
        .collect::<Vec<_>>();
    matches.sort_unstable();
    for candidate in matches {
        push_candidate(candidates, candidate);
    }
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
            candidate_model_stems("gemma-4-26B-A4B-it-UD-Q4_K_M"),
            vec![
                "gemma-4-26B-A4B-it-UD-Q4_K_M",
                "gemma-4-26B-A4B-it-UD",
                "gemma-4-26B-A4B-it"
            ]
        );
        assert_eq!(
            candidate_model_stems("gemma-4-E4B-it-BF16"),
            vec!["gemma-4-E4B-it-BF16", "gemma-4-E4B-it"]
        );
        assert_eq!(
            candidate_model_stems("gemma-4-E4B-it"),
            vec!["gemma-4-E4B-it"]
        );
        assert_eq!(
            candidate_model_stems("DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003"),
            vec![
                "DeepSeek-V4-Flash-0731-UD-IQ2_M",
                "DeepSeek-V4-Flash-0731-UD",
                "DeepSeek-V4-Flash-0731"
            ]
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

    #[test]
    fn finds_dspark_sidecar_without_runtime_flags() {
        let root = std::env::temp_dir().join(format!(
            "rnb-auto-dspark-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf");
        let dspark = root.join("DeepSeek-V4-Flash-0731-DSpark.gguf");
        std::fs::write(&target, []).unwrap();
        std::fs::write(&dspark, []).unwrap();

        assert_eq!(find_sibling_drafter(&target), Some(dspark));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn finds_quantized_or_sharded_dspark_sidecar_for_related_target() {
        let root = std::env::temp_dir().join(format!(
            "rnb-auto-dspark-suffix-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf");
        let dspark = root.join("DeepSeek-V4-Flash-0731-DSpark-Q4_K_M.gguf");
        let shard = root.join("DeepSeek-V4-Flash-0731-DSpark-00001-of-00003.gguf");
        std::fs::write(&target, []).unwrap();
        std::fs::write(&dspark, []).unwrap();
        std::fs::write(&shard, []).unwrap();

        assert_eq!(find_sibling_drafter(&target), Some(shard.clone()));
        std::fs::remove_file(shard).unwrap();
        assert_eq!(find_sibling_drafter(&target), Some(dspark));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ignores_unrelated_dspark_before_valid_assistant() {
        let root = std::env::temp_dir().join(format!(
            "rnb-auto-unrelated-dspark-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("gemma-4-E4B-it-Q4_K_M.gguf");
        let unrelated = root.join("DeepSeek-V4-Flash-0731-DSpark.gguf");
        let assistant = root.join("gemma-4-E4B-it-assistant.Q4_K_M.gguf");
        std::fs::write(&target, []).unwrap();
        std::fs::write(&unrelated, []).unwrap();
        std::fs::write(&assistant, []).unwrap();

        assert_eq!(find_sibling_drafter(&target), Some(assistant));

        std::fs::remove_dir_all(root).unwrap();
    }
}
