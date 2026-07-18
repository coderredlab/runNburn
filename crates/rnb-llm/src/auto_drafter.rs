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

    let direct_candidates = [
        format!("{stem}-assistant.Q4_K_M.gguf"),
        format!("{stem}-assistant.gguf"),
    ];
    for name in &direct_candidates {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let prefix = format!("{stem}-assistant.");
    let scan_mtp_dir = |mtp_dir: &Path| -> Option<PathBuf> {
        if !mtp_dir.is_dir() {
            return None;
        }
        std::fs::read_dir(mtp_dir).ok().and_then(|entries| {
            entries.flatten().find_map(|entry| {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str())?;
                if name.starts_with(&prefix) && name.ends_with(".gguf") {
                    Some(path)
                } else {
                    None
                }
            })
        })
    };

    if let Some(p) = scan_mtp_dir(&dir.join(format!("{stem}-mtp"))) {
        return Some(p);
    }

    if let (Some(parent_dir), Some(dir_name)) =
        (dir.parent(), dir.file_name().and_then(|n| n.to_str()))
    {
        if let Some(p) = scan_mtp_dir(&parent_dir.join(format!("{dir_name}-mtp"))) {
            return Some(p);
        }
    }

    None
}
