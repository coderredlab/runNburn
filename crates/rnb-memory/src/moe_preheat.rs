//! MoE hot-pool pagecache preheat.
//!
//! This module owns the page residency action for hot MoE tensors. It parses
//! the `.rnb` metadata needed to locate the hot rank prefix, then touches the
//! corresponding bytes supplied by the caller's model container lookup.
//!
//! Session 64 axis C.
//!
//! Hot-sorted `.rnb` (produced by `rnb-convert --moe-popularity`) stores
//! each layer's MoE expert bytes sorted by popularity — the first
//! `moe_hot_expert_count` experts are the ones that get selected most often
//! in offline profiling. This module walks those hot regions at model-load
//! time and issues one `read_volatile` per 4 KB page, which forces the
//! kernel to populate `pagecache` with the hot weights *before* decode
//! starts.
//!
//! Without preheat, the first decode step triggers a storm of page faults
//! (every unique expert the router picks has to page in from storage), and
//! the LRU evicts hot experts as cold ones come in. With preheat, the hot
//! tier is already resident when decoding begins, so decode-time storage
//! IO approaches zero and the steady-state token rate jumps past the
//! Session 55 theoretical ceiling (1.62 tok/s, derived from "every token
//! re-reads 893 MB from storage").
//!
//! Gated by `RNB_MOE_PREHEAT=1`.

#[derive(Debug, Clone, Copy)]
pub struct PreheatWeightView<'a> {
    pub rows: usize,
    pub data: &'a [u8],
}

/// Parse the `moe_hot_expert_count` field out of the `.rnb` metadata JSON.
/// Hand-rolled because we refuse to take a serde_json dep for a single
/// usize.
pub fn parse_hot_count(metadata: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(metadata).ok()?;
    let key_pos = text.find("\"moe_hot_expert_count\"")?;
    let tail = &text[key_pos..];
    let colon = tail.find(':')?;
    let tail = &tail[colon + 1..];
    let start = tail.find(|c: char| c.is_ascii_digit())?;
    let tail = &tail[start..];
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end].parse::<usize>().ok()
}

/// Parse the `moe_expert_permutation` 2D array from `.rnb` metadata JSON.
///
/// Returns `Vec<Vec<u32>>` where `result[layer][rank] = original_expert_id`.
/// The inverse maps rank-space (what the engine sees after hot-sort) back to
/// popularity-space (what the profiler recorded). Session 68 Phase 0 uses
/// this so memtrace can log both the rank (for hot/cold-tier reasoning) and
/// the original id (so downstream analyzers can filter by the offline
/// popularity top-N).
///
/// Hand-rolled, tolerant of whitespace / newlines / trailing commas. Does
/// not accept nested objects, scientific notation, or non-u32 integers.
pub fn parse_permutation(metadata: &[u8]) -> Option<Vec<Vec<u32>>> {
    let text = std::str::from_utf8(metadata).ok()?;
    let key_pos = text.find("\"moe_expert_permutation\"")?;
    let tail = &text[key_pos..];
    let colon = tail.find(':')?;
    let tail = &tail[colon + 1..];
    let lbracket = tail.find('[')?;
    let bytes = tail.as_bytes();
    let mut i = lbracket + 1;

    let mut result: Vec<Vec<u32>> = Vec::new();
    loop {
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return None;
        }
        match bytes[i] {
            b']' => return Some(result),
            b',' => {
                i += 1;
                continue;
            }
            b'[' => {
                i += 1;
                let mut inner: Vec<u32> = Vec::new();
                loop {
                    while i < bytes.len() && !bytes[i].is_ascii_digit() && bytes[i] != b']' {
                        i += 1;
                    }
                    if i >= bytes.len() {
                        return None;
                    }
                    if bytes[i] == b']' {
                        i += 1;
                        break;
                    }
                    let start = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    let num_str = std::str::from_utf8(&bytes[start..i]).ok()?;
                    inner.push(num_str.parse::<u32>().ok()?);
                }
                result.push(inner);
            }
            _ => return None,
        }
    }
}

/// Walk the hot region of each MoE tensor (gate_up_exps, down_exps) and
/// touch one byte per 4 KB page. The router (ffn_gate_inp, F32 ~1.4 MB
/// per layer) is cheap enough to preheat in full.
///
/// Returns `(pages_touched, bytes_covered)` so callers can log. The caller
/// supplies storage lookups so this crate owns residency policy without
/// depending on a concrete model container format.
pub fn preheat_moe_hot_tensors<'a>(
    n_layer: usize,
    hot_count: usize,
    mut get_weight: impl FnMut(&str) -> Option<PreheatWeightView<'a>>,
) -> (u64, u64) {
    const PAGE: usize = 4096;
    let mut pages = 0u64;
    let mut bytes = 0u64;

    for i in 0..n_layer {
        let gu_name = format!("blk.{i}.ffn_gate_up_exps.weight");
        let dn_name = format!("blk.{i}.ffn_down_exps.weight");
        let rt_name = format!("blk.{i}.ffn_gate_inp.weight");

        if let Some(w) = get_weight(&gu_name) {
            let data = w.data;
            // data layout: [n_expert, per_expert_bytes] contiguous, sorted
            // with hot experts first (rank 0..hot_count).
            let per_expert = if w.rows > 0 { data.len() / w.rows } else { 0 };
            let hot_end = (hot_count * per_expert).min(data.len());
            let (p, b) = touch_range(data, 0..hot_end, PAGE);
            pages += p;
            bytes += b;
        }
        if let Some(w) = get_weight(&dn_name) {
            let data = w.data;
            let per_expert = if w.rows > 0 { data.len() / w.rows } else { 0 };
            let hot_end = (hot_count * per_expert).min(data.len());
            let (p, b) = touch_range(data, 0..hot_end, PAGE);
            pages += p;
            bytes += b;
        }
        // Router is small (~1.4 MB / layer × 30 layer = 42 MB total),
        // preheat the whole thing.
        if let Some(w) = get_weight(&rt_name) {
            let data = w.data;
            let (p, b) = touch_range(data, 0..data.len(), PAGE);
            pages += p;
            bytes += b;
        }
    }
    (pages, bytes)
}

pub fn run_configured_preheat<'a>(
    metadata: &[u8],
    n_layer: usize,
    get_weight: impl FnMut(&str) -> Option<PreheatWeightView<'a>>,
) {
    if !crate::moe_policy::preheat_enabled() {
        return;
    }
    let Some(hot_count) = parse_hot_count(metadata) else {
        eprintln!(
            "[WARN] RNB_MOE_PREHEAT set but .rnb metadata has no moe_hot_expert_count; skipping"
        );
        return;
    };
    let t0 = std::time::Instant::now();
    let (pages, bytes) = preheat_moe_hot_tensors(n_layer, hot_count, get_weight);
    eprintln!(
        "[INFO] MoE preheat: hot_count={} touched {} pages ({:.1} MB) in {:.2}s",
        hot_count,
        pages,
        bytes as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f64()
    );
}

#[inline]
fn touch_range(buf: &[u8], range: std::ops::Range<usize>, _stride: usize) -> (u64, u64) {
    if range.is_empty() {
        return (0, 0);
    }
    let end = range.end.min(buf.len());
    let slice = &buf[range.start..end];
    // Byte-by-byte sequential sum forces the kernel into a "sequential
    // access" readahead pattern. With 4 KB stride + one read_volatile per
    // page, Linux treats each access as random and only pulls in that
    // single page; with contiguous reads, the readahead window grows up
    // to 128 KB per fault and pulls in neighbouring bytes that the decode
    // loop will actually touch. Compiler vectorizes the sum loop, so CPU
    // cost is bounded by RAM/storage bandwidth, not per-byte instruction
    // count. `black_box` keeps the compiler from eliding the read entirely.
    let mut acc: u64 = 0;
    for &b in slice {
        acc = acc.wrapping_add(b as u64);
    }
    std::hint::black_box(acc);
    let pages = ((end - range.start + 4095) / 4096) as u64;
    (pages, (end - range.start) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hot_count_from_emitted_metadata() {
        let text = br#"{
  "moe_hot_expert_count": 39,
  "moe_expert_permutation": [
    [0, 1, 2]
  ]
}"#;
        assert_eq!(parse_hot_count(text), Some(39));
    }

    #[test]
    fn returns_none_when_key_missing() {
        let text = br#"{"something_else": 42}"#;
        assert_eq!(parse_hot_count(text), None);
    }

    #[test]
    fn touch_range_counts_pages_exactly() {
        let buf = vec![0u8; 16384]; // 4 pages
        let (pages, bytes) = touch_range(&buf, 0..16384, 4096);
        assert_eq!(pages, 4);
        assert_eq!(bytes, 16384);
    }

    #[test]
    fn touch_range_handles_partial_final_page() {
        let buf = vec![0u8; 16384];
        let (pages, bytes) = touch_range(&buf, 0..10000, 4096);
        // offsets 0, 4096, 8192 → 3 pages
        assert_eq!(pages, 3);
        assert_eq!(bytes, 10000);
    }

    #[test]
    fn parses_permutation_from_emitted_metadata() {
        let text = br#"{
  "moe_hot_expert_count": 39,
  "moe_expert_permutation": [
    [0, 1, 2],
    [5, 3, 4],
    [127, 0, 63]
  ]
}"#;
        let perm = parse_permutation(text).expect("permutation should parse");
        assert_eq!(perm.len(), 3);
        assert_eq!(perm[0], vec![0, 1, 2]);
        assert_eq!(perm[1], vec![5, 3, 4]);
        assert_eq!(perm[2], vec![127, 0, 63]);
    }

    #[test]
    fn permutation_returns_none_when_key_missing() {
        let text = br#"{"moe_hot_expert_count": 39}"#;
        assert!(parse_permutation(text).is_none());
    }

    #[test]
    fn parses_permutation_with_irregular_whitespace() {
        // Single-line, no spaces between elements.
        let text = br#"{"moe_expert_permutation":[[1,2,3],[4,5,6]]}"#;
        let perm = parse_permutation(text).expect("parse should succeed");
        assert_eq!(perm, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn parses_empty_permutation_list() {
        let text = br#"{"moe_expert_permutation": []}"#;
        let perm = parse_permutation(text).expect("empty list should parse");
        assert!(perm.is_empty());
    }
}
