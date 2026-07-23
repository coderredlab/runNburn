//! Background page-warming of dense attention projection weights before the
//! first decode step (D1).
//!
//! On large offloaded MoE models (e.g. Hy3 295B on ~38 GiB RAM) the long
//! prefill reads ~100 GiB of expert bytes, which evicts the early layers'
//! attention Q/K/V/O quant-weight pages from the page cache. The first decode
//! step then re-faults every layer's attention projection from disk — measured
//! at ~1.3 s of the D1 cold transition on Hy3-Q2_K (`qkv_gemv+norm 827->25 ms`,
//! `o_weight 555->14 ms` vs the warm D2 step).
//!
//! Those projections are used on *every* decode token (route-independent,
//! deterministic) and total only ~2 GiB, so warming their pages during the
//! final prefill layers — overlapped with prefill compute — removes the D1
//! fault without stealing steady-state page-cache room: the pages age out
//! naturally after decode, exactly as they would if D1 faulted them itself.
//! This is deliberately *not* anonymous-RAM residency, which would permanently
//! shrink the expert page-cache and reproduce the discarded hot-expert
//! replication regression.
//!
//! Diagnostic, opt-in via `RNB_HY3_DECODE_ATTN_PREWARM`; default off until an
//! ABAB run proves a win on the target device.

use super::layer_weights::{LayerType, ModelWeights};
use std::sync::atomic::{AtomicBool, Ordering};

/// Armed once per prefill (reset when the range covering layer 0 is entered),
/// consumed once when the trailing-layer trigger fires.
static PREWARM_ARMED: AtomicBool = AtomicBool::new(false);

/// Minimum page granularity used as the touch stride. 4 KiB is the smallest
/// page size on the supported targets; striding by it touches every page at
/// least once even where the OS uses larger pages.
const MIN_PAGE_BYTES: usize = 4096;

/// A byte range into a live mmap-backed weight tensor.
///
/// The raw pointer is valid for the model's lifetime: the owning `Engine` (and
/// its mmap) outlives the warming thread, which finishes during prefill.
struct PageRange {
    ptr: usize,
    len: usize,
}

// SAFETY: the pointer addresses the model mmap, which outlives the spawned
// thread (see module docs); no aliasing mutation occurs — reads only.
unsafe impl Send for PageRange {}

fn collect_attention_pages(weights: &ModelWeights) -> Vec<PageRange> {
    let mut ranges = Vec::new();
    let mut push = |bytes: Option<&[u8]>| {
        if let Some(b) = bytes {
            if !b.is_empty() {
                ranges.push(PageRange {
                    ptr: b.as_ptr() as usize,
                    len: b.len(),
                });
            }
        }
    };
    for layer in &weights.layers {
        if let LayerType::Attention(w) = layer {
            push(w.q_weight.data.as_bytes());
            push(w.k_weight.data.as_bytes());
            push(w.v_weight.data.as_bytes());
            push(w.o_weight.data.as_bytes());
        }
    }
    ranges
}

/// Number of trailing prefill layers over which the warm is overlapped, derived
/// from model depth rather than a fixed constant so it shrinks on shallow
/// models. The warm must finish before D1 yet start late enough that the warmed
/// pages are not evicted by the remaining layers' expert reads.
fn prewarm_lead_layers(num_layers: usize) -> usize {
    (num_layers / 8).max(1).min(num_layers)
}

/// Called at the top of each prefill layer iteration. Resets the arm state when
/// a fresh prefill starts (layer 0) and spawns the warming thread exactly once,
/// when the trailing-layer window is reached.
pub(super) fn maybe_spawn_prefill_attention_prewarm(weights: &ModelWeights, layer_idx: usize) {
    if !super::policy::hy3_decode_attn_prewarm_enabled() {
        return;
    }
    if layer_idx == 0 {
        PREWARM_ARMED.store(true, Ordering::Release);
    }
    let num_layers = weights.layers.len();
    if num_layers == 0 {
        return;
    }
    let trigger = num_layers.saturating_sub(prewarm_lead_layers(num_layers));
    if layer_idx < trigger {
        return;
    }
    // Consume the arm exactly once for this prefill.
    if !PREWARM_ARMED.swap(false, Ordering::AcqRel) {
        return;
    }
    let ranges = collect_attention_pages(weights);
    if ranges.is_empty() {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("rnb-attn-prewarm".to_string())
        .spawn(move || warm_pages(&ranges));
}

fn warm_pages(ranges: &[PageRange]) {
    let mut acc: u8 = 0;
    for range in ranges {
        let ptr = range.ptr as *const u8;
        let mut off = 0usize;
        while off < range.len {
            // SAFETY: `[ptr, ptr + len)` is a live mmap range for the model
            // lifetime; this thread finishes during prefill (module docs).
            acc ^= unsafe { std::ptr::read_volatile(ptr.add(off)) };
            off += MIN_PAGE_BYTES;
        }
    }
    std::hint::black_box(acc);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prewarm_lead_scales_with_depth_and_clamps() {
        assert_eq!(prewarm_lead_layers(80), 10);
        assert_eq!(prewarm_lead_layers(8), 1);
        assert_eq!(prewarm_lead_layers(1), 1);
        assert_eq!(prewarm_lead_layers(0), 0);
    }

    #[test]
    fn warm_pages_reads_every_page_without_panicking() {
        // Two pages plus a byte; touching must stay in bounds.
        let buf = vec![7u8; MIN_PAGE_BYTES * 2 + 1];
        let ranges = vec![PageRange {
            ptr: buf.as_ptr() as usize,
            len: buf.len(),
        }];
        warm_pages(&ranges);
    }
}
