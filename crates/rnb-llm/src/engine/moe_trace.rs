//! MoE expert selection tracing.
//!
//! Global atomic histogram indexed by `(layer_idx, expert_idx)`. Used by
//! `rnb-moe-profile` to collect expert popularity over a corpus without
//! modifying the hot path for non-profiling runs.
//!
//! Overhead when inactive: one `OnceLock::get()` + one bool check per MoE
//! layer per token (≈ 30 × ~30 tok = ~1 000 check/prompt for 26B-A4B).
//! When active: one `fetch_add(1, Relaxed)` per picked expert (~8 × 30 =
//! 240/token). Negligible vs. the GEMV cost.

use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

struct MoeTraceState {
    n_layer: usize,
    n_expert: usize,
    // Flat layout: `counts[layer * n_expert + expert]`.
    counts: Vec<AtomicU64>,
    active: AtomicBool,
}

static STATE: OnceLock<MoeTraceState> = OnceLock::new();
static ROUTE_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static ROUTE_TRACE_WRITER: OnceLock<Mutex<Option<BufWriter<std::fs::File>>>> = OnceLock::new();
static PREDICTOR_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static PREDICTOR_TRACE_WRITER: OnceLock<Mutex<Option<BufWriter<std::fs::File>>>> = OnceLock::new();
static PREDICTOR_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Initialize the trace state. Must be called exactly once before any
/// `record` / `is_active` call in the profiling session. Subsequent calls
/// are no-ops (the underlying `OnceLock` pins shape permanently).
pub fn init(n_layer: usize, n_expert: usize) {
    STATE.get_or_init(|| MoeTraceState {
        n_layer,
        n_expert,
        counts: (0..n_layer * n_expert).map(|_| AtomicU64::new(0)).collect(),
        active: AtomicBool::new(false),
    });
}

/// Turn recording on. `init` must have been called first.
pub fn enable() {
    if let Some(s) = STATE.get() {
        s.active.store(true, Ordering::Release);
    }
}

/// Turn recording off. Existing counts are preserved.
pub fn disable() {
    if let Some(s) = STATE.get() {
        s.active.store(false, Ordering::Release);
    }
}

/// Fast path query used by `MoeLayerView::forward_with_logits`.
#[inline]
pub fn is_active() -> bool {
    match STATE.get() {
        Some(s) => s.active.load(Ordering::Relaxed),
        None => false,
    }
}

/// Increment the hit counter for one picked expert.
#[inline]
pub fn record(layer_idx: usize, expert_idx: usize) {
    let Some(s) = STATE.get() else { return };
    if layer_idx >= s.n_layer || expert_idx >= s.n_expert {
        return;
    }
    s.counts[layer_idx * s.n_expert + expert_idx].fetch_add(1, Ordering::Relaxed);
}

/// Record one complete selected-expert set for a MoE layer invocation.
///
/// This preserves the old histogram behavior when `enable()` was called, and
/// optionally writes an ordered CSV-like route trace when
/// `RNB_MOE_ROUTE_TRACE_FILE` is set. Trace lines are:
///
/// ```text
/// layer,expert0,expert1,...
/// ```
pub fn record_selection(layer_idx: usize, expert_ids: &[usize]) {
    if let Some(s) = STATE.get() {
        if s.active.load(Ordering::Relaxed) && layer_idx < s.n_layer {
            for &expert_idx in expert_ids {
                if expert_idx < s.n_expert {
                    s.counts[layer_idx * s.n_expert + expert_idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    if route_trace_enabled() {
        write_route_trace_line(layer_idx, expert_ids);
    }
}

/// Write predictor-training trace when `RNB_MOE_PREDICTOR_TRACE_FILE` is set.
///
/// JSONL schema:
/// `{"seq":N,"layer":L,"selected":[...],"top":[{"expert":E,"prob":P},...]}`
///
/// `top` length defaults to 16 and can be overridden with
/// `RNB_MOE_PREDICTOR_TRACE_TOP_N`.
pub fn record_predictor(layer_idx: usize, selected: &[usize], probs: &[f32]) {
    if !predictor_trace_enabled() {
        return;
    }
    let seq = PREDICTOR_TRACE_SEQ.fetch_add(1, Ordering::Relaxed);
    write_predictor_trace_line(seq, layer_idx, selected, probs);
}

#[inline]
fn route_trace_enabled() -> bool {
    *ROUTE_TRACE_ENABLED.get_or_init(super::policy::moe_route_trace_enabled)
}

#[inline]
pub(in crate::engine) fn route_trace_is_active() -> bool {
    route_trace_enabled()
}

fn route_trace_writer() -> &'static Mutex<Option<BufWriter<std::fs::File>>> {
    ROUTE_TRACE_WRITER.get_or_init(|| {
        let writer = super::policy::moe_route_trace_file()
            .and_then(|path| std::fs::File::create(path).ok().map(BufWriter::new));
        Mutex::new(writer)
    })
}

#[inline]
fn predictor_trace_enabled() -> bool {
    *PREDICTOR_TRACE_ENABLED.get_or_init(super::policy::moe_predictor_trace_enabled)
}

#[inline]
pub(in crate::engine) fn predictor_trace_is_active() -> bool {
    predictor_trace_enabled()
}

fn predictor_trace_writer() -> &'static Mutex<Option<BufWriter<std::fs::File>>> {
    PREDICTOR_TRACE_WRITER.get_or_init(|| {
        let writer = super::policy::moe_predictor_trace_file()
            .and_then(|path| std::fs::File::create(path).ok().map(BufWriter::new));
        Mutex::new(writer)
    })
}

fn predictor_trace_top_n() -> usize {
    super::policy::moe_predictor_trace_top_n_limit()
}

fn write_route_trace_line(layer_idx: usize, expert_ids: &[usize]) {
    let lock = route_trace_writer();
    let Ok(mut guard) = lock.lock() else {
        return;
    };
    let Some(writer) = guard.as_mut() else {
        return;
    };

    let mut line = String::new();
    format_selection_line(&mut line, layer_idx, expert_ids);
    let _ = writer.write_all(line.as_bytes());
    let _ = writer.flush();
}

fn write_predictor_trace_line(seq: u64, layer_idx: usize, selected: &[usize], probs: &[f32]) {
    let lock = predictor_trace_writer();
    let Ok(mut guard) = lock.lock() else {
        return;
    };
    let Some(writer) = guard.as_mut() else {
        return;
    };

    let mut line = String::new();
    format_predictor_line(
        &mut line,
        seq,
        layer_idx,
        selected,
        probs,
        predictor_trace_top_n(),
    );
    let _ = writer.write_all(line.as_bytes());
    let _ = writer.flush();
}

fn format_selection_line(line: &mut String, layer_idx: usize, expert_ids: &[usize]) {
    line.clear();
    line.push_str(&layer_idx.to_string());
    for &expert_idx in expert_ids {
        line.push(',');
        line.push_str(&expert_idx.to_string());
    }
    line.push('\n');
}

fn format_predictor_line(
    line: &mut String,
    seq: u64,
    layer_idx: usize,
    selected: &[usize],
    probs: &[f32],
    top_n: usize,
) {
    line.clear();
    line.push_str("{\"seq\":");
    line.push_str(&seq.to_string());
    line.push_str(",\"layer\":");
    line.push_str(&layer_idx.to_string());
    line.push_str(",\"selected\":[");
    for (idx, &expert_idx) in selected.iter().enumerate() {
        if idx > 0 {
            line.push(',');
        }
        line.push_str(&expert_idx.to_string());
    }
    line.push_str("],\"top\":[");

    let mut ranked = (0..probs.len()).collect::<Vec<_>>();
    ranked.sort_by(|&left, &right| {
        probs[right]
            .partial_cmp(&probs[left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    for (idx, expert_idx) in ranked.into_iter().take(top_n).enumerate() {
        if idx > 0 {
            line.push(',');
        }
        line.push_str("{\"expert\":");
        line.push_str(&expert_idx.to_string());
        line.push_str(",\"prob\":");
        line.push_str(&format!("{:.6}", probs[expert_idx]));
        trim_json_float_zeros(line);
        line.push('}');
    }
    line.push_str("]}\n");
}

fn trim_json_float_zeros(line: &mut String) {
    while line.ends_with('0') {
        line.pop();
    }
    if line.ends_with('.') {
        line.push('0');
    }
}

/// Snapshot all counts as `[layer][expert]`. Atomic loads; safe under
/// concurrent `record`.
pub fn snapshot() -> Option<Vec<Vec<u64>>> {
    let s = STATE.get()?;
    let mut out = Vec::with_capacity(s.n_layer);
    for l in 0..s.n_layer {
        let base = l * s.n_expert;
        let row: Vec<u64> = (0..s.n_expert)
            .map(|e| s.counts[base + e].load(Ordering::Relaxed))
            .collect();
        out.push(row);
    }
    Some(out)
}

/// Reset all counts to zero (does not change shape or active flag).
pub fn reset() {
    if let Some(s) = STATE.get() {
        for c in &s.counts {
            c.store(0, Ordering::Relaxed);
        }
    }
}

/// Shape `(n_layer, n_expert)` if initialized.
pub fn shape() -> Option<(usize, usize)> {
    STATE.get().map(|s| (s.n_layer, s.n_expert))
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: OnceLock makes these tests order-sensitive when run in the same
    // process. We only verify the shape-once semantics and basic counting.

    #[test]
    fn init_record_snapshot_roundtrip() {
        init(4, 8);
        enable();
        record(0, 3);
        record(2, 7);
        record(0, 3);
        let snap = snapshot().unwrap();
        assert_eq!(snap.len(), 4);
        assert_eq!(snap[0].len(), 8);
        // counts may accumulate across test ordering; just assert >=.
        assert!(snap[0][3] >= 2);
        assert!(snap[2][7] >= 1);
        disable();
    }

    #[test]
    fn out_of_bounds_is_silent_noop() {
        init(4, 8);
        // Should not panic.
        record(99, 99);
    }

    #[test]
    fn format_selection_line_writes_layer_and_selected_experts() {
        let mut line = String::new();
        format_selection_line(&mut line, 3, &[9, 2, 5]);
        assert_eq!(line, "3,9,2,5\n");
    }

    #[test]
    fn format_predictor_line_writes_jsonl_with_top_candidates() {
        let mut line = String::new();
        let probs = [0.1, 0.4, 0.2, 0.3];

        format_predictor_line(&mut line, 7, 3, &[1, 3], &probs, 3);

        assert_eq!(
            line,
            "{\"seq\":7,\"layer\":3,\"selected\":[1,3],\"top\":[{\"expert\":1,\"prob\":0.4},{\"expert\":3,\"prob\":0.3},{\"expert\":2,\"prob\":0.2}]}\n"
        );
    }
}
