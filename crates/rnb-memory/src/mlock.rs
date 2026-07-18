//! Session 67 axis P: small-pool `mlock(2)` for router + hottest experts.
//!
//! Session 66 H3 measurement showed baseline mmap LRU evicts popular top-10
//! experts (mincore resident rate median 0% over 1406 samples). Axis P pins
//! the router tensor (30 × 1.38 MB ≈ 41 MB on gemma4-26B-A4B) plus an
//! optional per-layer top-N hottest experts into physical RAM via `mlock`,
//! so the kernel cannot steal them.
//!
//! Enabled through env:
//!   * `RNB_MOE_MLOCK_ROUTER=1` — mlock every MoE layer's router projection.
//!   * `RNB_MOE_MLOCK_TOP=N`    — mlock the per-layer top-N hottest experts
//!     (requires `RNB_MOE_POPULARITY=<path>` pointing at a
//!     `rnb-moe-profile` JSON).
//!
//! All failures are logged and non-fatal — Android's `RLIMIT_MEMLOCK`
//! default is 64 MB, so on a 600-expert pin request most calls will
//! return `ENOMEM` and the engine continues without pinning.

use std::io;
use std::path::Path;

/// Pin `len` bytes starting at `ptr` into RAM. `len == 0` is a no-op.
///
/// The kernel rounds `ptr` down to the nearest page and `ptr + len` up,
/// so the locked region may slightly exceed `len`. Returns the byte count
/// passed to `mlock(2)` (not the kernel-rounded count).
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn mlock_region(ptr: *const u8, len: usize) -> io::Result<usize> {
    if len == 0 {
        return Ok(0);
    }
    let r = unsafe { libc::mlock(ptr as *const libc::c_void, len) };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(len)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn mlock_region(_ptr: *const u8, _len: usize) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "mlock is Linux/Android-only",
    ))
}

/// Parsed contents of `docs/superpowers/results/2026-04-21-expert_popularity.json`:
/// a `[n_layer][n_expert]` matrix of hit counts from `rnb-moe-profile`.
pub struct Popularity {
    pub n_layer: usize,
    pub n_expert: usize,
    pub hit_counts: Vec<Vec<u32>>,
}

impl Popularity {
    /// Parse the popularity JSON. Accepts the schema produced by
    /// `rnb-moe-profile` (and used by `scripts/memtrace_analyze.py`):
    /// `{"n_layer": .., "n_expert": .., "hit_counts": [[..n_expert..], ..]}`.
    /// Hand-rolled parser so rnb-llm doesn't pick up a serde_json dep for one
    /// file read (same pattern as `moe_preheat::parse_hot_count`).
    pub fn load(path: &Path) -> io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let n_layer = extract_usize(&s, "n_layer")?;
        let n_expert = extract_usize(&s, "n_expert")?;
        let hit_counts = parse_hit_counts(&s)?;
        Ok(Self {
            n_layer,
            n_expert,
            hit_counts,
        })
    }

    /// Expert IDs of the top-`n` hottest experts for `layer`, ordered by
    /// descending hit count. Returns an empty vec for out-of-range layers.
    /// Ties are broken by expert id (stable sort), so successive calls with
    /// the same input produce the same order.
    pub fn top_n(&self, layer: usize, n: usize) -> Vec<usize> {
        if layer >= self.hit_counts.len() || n == 0 {
            return Vec::new();
        }
        let row = &self.hit_counts[layer];
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| row[b].cmp(&row[a]).then(a.cmp(&b)));
        idx.truncate(n.min(row.len()));
        idx
    }
}

pub fn expert_pin_ids(
    popularity: &Popularity,
    layer_idx: usize,
    top_n: usize,
    max_slots: usize,
    hot_sorted_layout: bool,
) -> Vec<usize> {
    if top_n == 0 || max_slots == 0 {
        return Vec::new();
    }
    if hot_sorted_layout {
        return (0..top_n.min(max_slots)).collect();
    }
    popularity
        .top_n(layer_idx, top_n)
        .into_iter()
        .filter(|&id| id < max_slots)
        .collect()
}

fn extract_usize(text: &str, key: &str) -> io::Result<usize> {
    let pat = format!("\"{key}\"");
    let key_pos = text.find(&pat).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing \"{key}\" in popularity JSON"),
        )
    })?;
    let tail = &text[key_pos + pat.len()..];
    let colon = tail
        .find(':')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no colon after key"))?;
    let tail = &tail[colon + 1..];
    let start = tail
        .find(|c: char| c.is_ascii_digit())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no digit after key"))?;
    let tail = &tail[start..];
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end]
        .parse::<usize>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Scan for `"hit_counts": [ [row], [row], ... ]` and return the row vectors.
/// Robust to surrounding whitespace and sibling fields. Rows must contain
/// only non-negative integers separated by commas.
fn parse_hit_counts(text: &str) -> io::Result<Vec<Vec<u32>>> {
    let key_pos = text
        .find("\"hit_counts\"")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing \"hit_counts\""))?;
    let after_key = &text[key_pos + "\"hit_counts\"".len()..];
    let outer_open = after_key
        .find('[')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no opening [ for hit_counts"))?;
    let inside_outer = &after_key[outer_open + 1..];

    let mut out: Vec<Vec<u32>> = Vec::new();
    let mut cursor = 0usize;
    let bytes = inside_outer.as_bytes();
    while cursor < bytes.len() {
        // skip whitespace and commas between rows
        while cursor < bytes.len() && matches!(bytes[cursor], b' ' | b'\n' | b'\r' | b'\t' | b',') {
            cursor += 1;
        }
        if cursor >= bytes.len() {
            break;
        }
        match bytes[cursor] {
            b'[' => {
                let row_start = cursor + 1;
                let rel_end = inside_outer[row_start..].find(']').ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unterminated row in hit_counts")
                })?;
                let row_text = &inside_outer[row_start..row_start + rel_end];
                let mut row: Vec<u32> = Vec::new();
                for tok in row_text.split(',') {
                    let t = tok.trim();
                    if t.is_empty() {
                        continue;
                    }
                    let n = t.parse::<u32>().map_err(|e| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("bad hit_counts integer {t:?}: {e}"),
                        )
                    })?;
                    row.push(n);
                }
                out.push(row);
                cursor = row_start + rel_end + 1;
            }
            b']' => break, // outer array closed
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected byte {other:#x} in hit_counts"),
                ));
            }
        }
    }
    Ok(out)
}

/// Axis P configuration parsed from environment.
#[derive(Clone, Debug, Default)]
pub struct AxisPConfig {
    pub mlock_router: bool,
    pub mlock_top_n: usize,
    pub popularity_path: Option<std::path::PathBuf>,
}

impl AxisPConfig {
    pub fn from_env() -> Self {
        let mlock_router = std::env::var("RNB_MOE_MLOCK_ROUTER")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mlock_top_n = std::env::var("RNB_MOE_MLOCK_TOP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0usize);
        let popularity_path = std::env::var("RNB_MOE_POPULARITY")
            .ok()
            .map(std::path::PathBuf::from);
        Self {
            mlock_router,
            mlock_top_n,
            popularity_path,
        }
    }

    pub fn is_active(&self) -> bool {
        self.mlock_router || self.mlock_top_n > 0
    }
}

pub fn load_popularity_for_config(cfg: &AxisPConfig) -> Option<Popularity> {
    if cfg.mlock_top_n == 0 {
        return None;
    }
    match cfg.popularity_path.as_deref() {
        Some(p) => match Popularity::load(p) {
            Ok(pop) => Some(pop),
            Err(e) => {
                eprintln!(
                    "[axis-P] popularity load from {:?} failed ({e}); \
                     RNB_MOE_MLOCK_TOP disabled",
                    p
                );
                None
            }
        },
        None => {
            eprintln!(
                "[axis-P] RNB_MOE_MLOCK_TOP={} requires RNB_MOE_POPULARITY=<path>; \
                 disabled",
                cfg.mlock_top_n
            );
            None
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AxisPMlockReport {
    pub router_bytes: usize,
    pub expert_bytes: usize,
    pub locked_experts: usize,
    pub router_failures: usize,
    pub expert_failures: usize,
}

impl AxisPMlockReport {
    pub fn lock_router(&mut self, layer_idx: usize, ptr: *const u8, bytes: usize) {
        match mlock_region(ptr, bytes) {
            Ok(n) => self.router_bytes += n,
            Err(e) => {
                eprintln!(
                    "[axis-P] layer {layer_idx} router mlock failed \
                     ({bytes} bytes): {e}"
                );
                self.router_failures += 1;
            }
        }
    }

    pub fn lock_expert_pair(
        &mut self,
        layer_idx: usize,
        expert_id: usize,
        gate_up_ptr: *const u8,
        gate_up_bytes: usize,
        down_ptr: *const u8,
        down_bytes: usize,
    ) {
        let gate_up = mlock_region(gate_up_ptr, gate_up_bytes);
        let down = mlock_region(down_ptr, down_bytes);
        match (gate_up, down) {
            (Ok(g), Ok(d)) => {
                self.expert_bytes += g + d;
                self.locked_experts += 1;
            }
            (gate_up, down) => {
                if let Err(e) = gate_up {
                    eprintln!(
                        "[axis-P] layer {layer_idx} expert {expert_id} gate_up \
                         mlock failed ({gate_up_bytes} bytes): {e}"
                    );
                }
                if let Err(e) = down {
                    eprintln!(
                        "[axis-P] layer {layer_idx} expert {expert_id} down \
                         mlock failed ({down_bytes} bytes): {e}"
                    );
                }
                self.expert_failures += 1;
            }
        }
    }

    pub fn lock_expert_pair_by_id(
        &mut self,
        layer_idx: usize,
        expert_id: usize,
        gate_up: &[u8],
        gate_up_bytes_per_expert: usize,
        down: &[u8],
        down_bytes_per_expert: usize,
    ) {
        let gate_up_start = expert_id
            .checked_mul(gate_up_bytes_per_expert)
            .unwrap_or(usize::MAX);
        let down_start = expert_id
            .checked_mul(down_bytes_per_expert)
            .unwrap_or(usize::MAX);
        if gate_up_start.saturating_add(gate_up_bytes_per_expert) > gate_up.len()
            || down_start.saturating_add(down_bytes_per_expert) > down.len()
        {
            self.record_expert_out_of_range(
                layer_idx,
                expert_id,
                gate_up_start,
                gate_up_bytes_per_expert,
                gate_up.len(),
                down_start,
                down_bytes_per_expert,
                down.len(),
            );
            return;
        }

        let gate_up_ptr = unsafe { gate_up.as_ptr().add(gate_up_start) };
        let down_ptr = unsafe { down.as_ptr().add(down_start) };
        self.lock_expert_pair(
            layer_idx,
            expert_id,
            gate_up_ptr,
            gate_up_bytes_per_expert,
            down_ptr,
            down_bytes_per_expert,
        );
    }

    pub fn record_expert_out_of_range(
        &mut self,
        layer_idx: usize,
        expert_id: usize,
        gate_up_start: usize,
        gate_up_bytes: usize,
        gate_up_len: usize,
        down_start: usize,
        down_bytes: usize,
        down_len: usize,
    ) {
        eprintln!(
            "[axis-P] layer {layer_idx} expert {expert_id} out of range \
             (gu {}+{} > {}, dn {}+{} > {})",
            gate_up_start, gate_up_bytes, gate_up_len, down_start, down_bytes, down_len
        );
        self.expert_failures += 1;
    }

    pub fn log_summary(&self, cfg: &AxisPConfig) {
        eprintln!(
            "[axis-P] mlocked router {}KB + {} experts {}KB (router_fail={}, \
             expert_fail={}); RNB_MOE_MLOCK_ROUTER={} RNB_MOE_MLOCK_TOP={}",
            self.router_bytes / 1024,
            self.locked_experts,
            self.expert_bytes / 1024,
            self.router_failures,
            self.expert_failures,
            cfg.mlock_router as u8,
            cfg.mlock_top_n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_n_orders_by_hit_then_id() {
        let pop = Popularity {
            n_layer: 1,
            n_expert: 5,
            hit_counts: vec![vec![10, 50, 50, 30, 0]],
        };
        // 50 ties broken by ascending id → [1, 2], then 30, then 10, then 0.
        assert_eq!(pop.top_n(0, 3), vec![1, 2, 3]);
        assert_eq!(pop.top_n(0, 10), vec![1, 2, 3, 0, 4]);
    }

    #[test]
    fn top_n_out_of_range_layer_is_empty() {
        let pop = Popularity {
            n_layer: 1,
            n_expert: 3,
            hit_counts: vec![vec![1, 2, 3]],
        };
        assert!(pop.top_n(1, 3).is_empty());
        assert!(pop.top_n(0, 0).is_empty());
    }

    #[test]
    fn expert_pin_ids_use_slots_for_hot_sorted_layout() {
        let pop = Popularity {
            n_layer: 1,
            n_expert: 5,
            hit_counts: vec![vec![0, 100, 10, 90, 5]],
        };

        assert_eq!(expert_pin_ids(&pop, 0, 3, 5, true), vec![0, 1, 2]);
        assert_eq!(expert_pin_ids(&pop, 0, 3, 5, false), vec![1, 3, 2]);
        assert_eq!(expert_pin_ids(&pop, 0, 10, 2, true), vec![0, 1]);
    }

    #[test]
    fn axis_p_report_records_zero_length_successes() {
        let mut report = AxisPMlockReport::default();

        report.lock_router(0, std::ptr::null(), 0);
        report.lock_expert_pair(0, 1, std::ptr::null(), 0, std::ptr::null(), 0);

        assert_eq!(report.router_bytes, 0);
        assert_eq!(report.expert_bytes, 0);
        assert_eq!(report.locked_experts, 1);
        assert_eq!(report.router_failures, 0);
        assert_eq!(report.expert_failures, 0);
    }

    #[test]
    fn axis_p_report_resolves_expert_id_bounds_before_locking() {
        let mut report = AxisPMlockReport::default();
        let gate_up = vec![0u8; 8];
        let down = vec![0u8; 8];

        report.lock_expert_pair_by_id(0, 2, &gate_up, 4, &down, 4);

        assert_eq!(report.expert_failures, 1);
        assert_eq!(report.locked_experts, 0);
        assert_eq!(report.expert_bytes, 0);
    }

    #[test]
    fn env_default_is_inactive() {
        // Clear envs defensively — the test process may carry them over.
        unsafe {
            std::env::remove_var("RNB_MOE_MLOCK_ROUTER");
            std::env::remove_var("RNB_MOE_MLOCK_TOP");
            std::env::remove_var("RNB_MOE_POPULARITY");
        }
        let cfg = AxisPConfig::from_env();
        assert!(!cfg.is_active());
        assert_eq!(cfg.mlock_top_n, 0);
        assert!(!cfg.mlock_router);
    }

    #[test]
    fn load_parses_inline_json() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            r#"{
  "gguf_path": "foo.gguf",
  "n_layer": 2,
  "n_expert": 4,
  "hit_counts": [
    [10, 5, 50, 7],
    [0, 100, 1, 99]
  ]
}"#,
        )
        .unwrap();
        let pop = Popularity::load(tmp.path()).unwrap();
        assert_eq!(pop.n_layer, 2);
        assert_eq!(pop.n_expert, 4);
        assert_eq!(
            pop.hit_counts,
            vec![vec![10, 5, 50, 7], vec![0, 100, 1, 99]]
        );
        assert_eq!(pop.top_n(0, 2), vec![2, 0]); // 50@2 then 10@0
        assert_eq!(pop.top_n(1, 2), vec![1, 3]); // 100@1 then 99@3
    }

    #[test]
    fn load_matches_session64_popularity_shape() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // Smaller facsimile of the Session 64 `expert_popularity.json` schema.
        // 3 layers × 5 experts with realistic-looking variation.
        std::fs::write(
            tmp.path(),
            r#"{"n_layer": 3, "n_expert": 5, "n_expert_used": 2,
               "hit_counts": [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [0, 0, 10, 0, 0]]}"#,
        )
        .unwrap();
        let pop = Popularity::load(tmp.path()).unwrap();
        assert_eq!(pop.top_n(0, 3), vec![4, 3, 2]);
        assert_eq!(pop.top_n(1, 3), vec![0, 1, 2]);
        assert_eq!(pop.top_n(2, 1), vec![2]);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn mlock_small_region() {
        // A 4 KB page should be well within RLIMIT_MEMLOCK on any dev host.
        let buf = vec![0u8; 4096];
        let ok = mlock_region(buf.as_ptr(), buf.len());
        // Tolerate the rare sandbox that has RLIMIT_MEMLOCK=0 — we only
        // require the call to compile + not panic.
        assert!(ok.is_ok() || ok.err().map(|e| e.raw_os_error()).is_some());
    }
}
