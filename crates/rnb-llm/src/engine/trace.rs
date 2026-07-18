//! Debug / trace helpers used across prefill and decode paths.
//!
//! Most of these are env-guarded (`RNB_DEBUG_*` / `RNB_DUMP_BIN_*`) and return
//! immediately when the corresponding knob isn't set.

pub(super) fn kv_trace_enabled() -> bool {
    super::policy::debug_kv_trace_enabled()
}

pub(super) fn layer_trace_enabled() -> bool {
    super::policy::debug_layer_trace_enabled()
}

pub(super) fn emit_layer_trace(tag: &str, layer_idx: usize, hidden: &[f32]) {
    if !layer_trace_enabled() {
        return;
    }
    let n = hidden.len().max(1) as f32;
    let mean = hidden.iter().sum::<f32>() / n;
    let l2 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
    let head = hidden.iter().take(4).copied().collect::<Vec<_>>();
    eprintln!(
        "[layer-trace][{}] layer={} mean={:.6} l2={:.6} head={:?}",
        tag, layer_idx, mean, l2, head
    );
}

pub(super) fn attn_trace_enabled() -> bool {
    super::policy::debug_attn_trace_enabled()
}

pub(super) fn mtp_finite_trace_enabled() -> bool {
    super::policy::env_flag("RNB_MTP_FINITE_TRACE")
}

pub(super) fn emit_mtp_finite_trace(tag: &str, layer_idx: usize, name: &str, data: &[f32]) {
    if !mtp_finite_trace_enabled() {
        return;
    }
    let mut nonfinite = 0usize;
    let mut first_bad = None;
    let mut max_abs = 0.0f32;
    for (idx, &value) in data.iter().enumerate() {
        if value.is_finite() {
            max_abs = max_abs.max(value.abs());
        } else {
            nonfinite += 1;
            if first_bad.is_none() {
                first_bad = Some(idx);
            }
        }
    }
    let head = data.iter().take(4).copied().collect::<Vec<_>>();
    eprintln!(
        "[MTP_FINITE] tag={} layer={} name={} len={} nonfinite={} first_bad={:?} max_abs={:.6e} head={:?}",
        tag,
        layer_idx,
        name,
        data.len(),
        nonfinite,
        first_bad,
        max_abs,
        head
    );
}

pub(super) fn targeted_attn_trace_enabled(layer_idx: usize) -> bool {
    super::policy::debug_targeted_attn_trace_enabled(layer_idx)
}

pub(super) fn ple_trace_enabled() -> bool {
    super::policy::debug_ple_trace_enabled()
}

pub(super) fn emit_vec_trace(tag: &str, layer_idx: usize, name: &str, data: &[f32]) {
    dump_bin(tag, layer_idx, name, data);
    if !attn_trace_enabled() {
        return;
    }
    let n = data.len().max(1) as f32;
    let mean = data.iter().sum::<f32>() / n;
    let l2 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let head = data.iter().take(4).copied().collect::<Vec<_>>();
    eprintln!(
        "[attn-trace][{}] layer={} {} mean={:.6} l2={:.6} head={:?}",
        tag, layer_idx, name, mean, l2, head
    );
}

pub(super) fn dump_bin_dir() -> Option<std::path::PathBuf> {
    super::policy::dump_bin_dir()
}

pub(super) fn dump_bin(tag: &str, layer_idx: usize, name: &str, data: &[f32]) {
    let Some(dir) = dump_bin_dir() else { return };
    if !super::policy::dump_bin_layer_enabled(layer_idx) {
        return;
    }
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let filename = if layer_idx == usize::MAX {
        format!("{}_{}.bin", tag, name)
    } else {
        format!("{}_L{}_{}.bin", tag, layer_idx, name)
    };
    let path = dir.join(filename);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let _ = std::fs::write(&path, bytes);
    if super::policy::dump_bin_log_enabled() {
        eprintln!(
            "[dump-bin] {} layer={} {} shape=({} f32) path={}",
            tag,
            layer_idx,
            name,
            data.len(),
            path.display()
        );
    }
}

pub(super) fn emit_ple_trace(tag: &str, layer_idx: usize, name: &str, data: &[f32]) {
    if !ple_trace_enabled() {
        return;
    }
    if layer_idx != usize::MAX && !super::policy::debug_ple_layer_matches(layer_idx) {
        return;
    }
    let n = data.len().max(1) as f32;
    let mean = data.iter().sum::<f32>() / n;
    let l2 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let head = data.iter().take(4).copied().collect::<Vec<_>>();
    eprintln!(
        "[ple-trace][{}] layer={} {} mean={:.6} l2={:.6} head={:?}",
        tag, layer_idx, name, mean, l2, head
    );
}

pub(super) fn emit_ple_dump(layer_idx: usize, name: &str, data: &[f32]) {
    let Some(want) = super::policy::debug_ple_dump_layer() else {
        return;
    };
    if layer_idx != want {
        return;
    }
    let take = super::policy::debug_ple_dump_take().min(data.len());
    eprintln!(
        "[ple-dump] layer={} {} first{}={:?}",
        layer_idx,
        name,
        take,
        &data[..take]
    );
}

pub(super) fn gemma4_ple_replay_dump_enabled(layer_idx: usize) -> bool {
    super::policy::debug_gemma4_ple_replay_dir().is_some()
        && super::policy::debug_gemma4_ple_replay_layer().is_some_and(|want| want == layer_idx)
}

pub(super) fn dump_gemma4_ple_replay_f32(
    tag: &str,
    layer_idx: usize,
    name: &str,
    data: &[f32],
) -> crate::error::Result<()> {
    if !gemma4_ple_replay_dump_enabled(layer_idx) {
        return Ok(());
    }
    let Some(dir) = super::policy::debug_gemma4_ple_replay_dir() else {
        return Ok(());
    };
    std::fs::create_dir_all(&dir).map_err(|err| {
        crate::error::LlmError::Forward(format!(
            "Gemma4 PLE replay dump create dir failed: {}: {err}",
            dir.display()
        ))
    })?;
    let path = dir.join(format!("{tag}_L{layer_idx}_{name}.bin"));
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    std::fs::write(&path, bytes).map_err(|err| {
        crate::error::LlmError::Forward(format!(
            "Gemma4 PLE replay dump write failed: {}: {err}",
            path.display()
        ))
    })?;
    eprintln!(
        "[gemma4-ple-replay] tag={tag} layer={layer_idx} name={name} len={} path={}",
        data.len(),
        path.display()
    );
    Ok(())
}

pub(super) fn emit_final_dump(name: &str, data: &[f32]) {
    if !super::policy::debug_final_dump_enabled() {
        return;
    }
    let take = super::policy::debug_final_dump_take().min(data.len());
    eprintln!("[final-dump] {} first{}={:?}", name, take, &data[..take]);
}

pub(super) fn decode_trace_targets() -> Option<Vec<String>> {
    super::policy::debug_decode_targets()
}

pub(super) fn decode_trace_layer_enabled(layer_idx: usize) -> bool {
    super::policy::debug_decode_layer_enabled(layer_idx)
}
