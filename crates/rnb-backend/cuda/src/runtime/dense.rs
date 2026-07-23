use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static DENSE_CHAIN_TRACE_CALLS: AtomicUsize = AtomicUsize::new(0);
static DENSE_EXPERT_TRACE_CALLS: AtomicUsize = AtomicUsize::new(0);
#[derive(Clone, Copy)]
enum Qwen35DeviceSharedWeights<'a> {
    F32 {
        gate: &'a [f32],
        up: &'a [f32],
        down: &'a [f32],
    },
    Quant {
        gate: &'a [u8],
        gate_quant: u32,
        up: &'a [u8],
        up_quant: u32,
        down: &'a [u8],
        down_quant: u32,
    },
}

fn env_flag(name: &str) -> bool {
    env_flag_or(name, false)
}

fn env_flag_value(name: &str) -> Option<bool> {
    std::env::var(name)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .ok()
}

fn env_flag_or(name: &str, default: bool) -> bool {
    env_flag_value(name).unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn log_qwen35_device_moe_compute_phase(
    label: &str,
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    elapsed: std::time::Duration,
) {
    if tuning::qwen35_device_moe_phase_profile_enabled() {
        eprintln!(
            "  [CUDA-QWEN35-MOE compute] {:24} {:.3}ms tokens={} n_ff={} n_embd={}",
            label,
            elapsed.as_secs_f64() * 1000.0,
            token_count,
            n_ff,
            n_embd
        );
    }
}

fn dense_chain_trace_call() -> Option<usize> {
    if !env_flag("RNB_CUDA_DENSE_CHAIN_TRACE") {
        return None;
    }
    let call = DENSE_CHAIN_TRACE_CALLS.fetch_add(1, Ordering::Relaxed);
    (call < env_usize("RNB_CUDA_DENSE_CHAIN_TRACE_LIMIT", 64)).then_some(call)
}

fn dense_expert_trace_call() -> Option<usize> {
    if !env_flag("RNB_CUDA_DENSE_EXPERT_TRACE") {
        return None;
    }
    let call = DENSE_EXPERT_TRACE_CALLS.fetch_add(1, Ordering::Relaxed);
    (call < env_usize("RNB_CUDA_DENSE_EXPERT_TRACE_LIMIT", 64)).then_some(call)
}

fn dense_expert_trace_enabled() -> bool {
    env_flag("RNB_CUDA_DENSE_EXPERT_TRACE")
}

/// cu60 axis A — PLE branch env gate.
///
/// `RNB_CU60_NO_PLE=1` 이면 chain function body 진입 직후 PLE args 5개
/// (ple_gate_weights / ple_proj_weights / ple_post_norm_weight / ple_input /
/// ple_input_device_offset) 를 강제 None 처리. 기존 if let 분기 (line 4256,
/// 4513) 가 자동 skip 돼서 PLE compute / weight kind 판별 비용 0 측정.
/// Gemma 정확성 transient 깨짐 OK (측정 용도). Llama 등 PLE 미사용 arch
/// 는 build_*_args 단계에서 이미 None 이라 무영향 (대조군).
///
/// OnceLock cache — chain function 진입마다 syscall 회피
/// (chain_diag_bridge::is_active 와 동일 패턴).
fn cu60_no_ple() -> bool {
    use std::sync::OnceLock;
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| std::env::var("RNB_CU60_NO_PLE").as_deref() == Ok("1"))
}

fn cu62_ple_megakernel() -> bool {
    use std::sync::OnceLock;
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| std::env::var("RNB_CU62_PLE_MEGAKERNEL").as_deref() == Ok("1"))
}

fn quantize_q8_1_by_32(input: &[f32], blocks_per_row: usize) -> (Vec<i8>, Vec<f32>) {
    let mut qs = vec![0i8; blocks_per_row * 256];
    let mut ds = vec![0.0f32; blocks_per_row * 8];
    for b in 0..blocks_per_row {
        for j in 0..8 {
            let off = b * 256 + j * 32;
            let chunk = &input[off..off + 32];
            let max_abs = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if max_abs == 0.0 {
                continue;
            }
            let d = max_abs / 127.0;
            let inv_d = 1.0 / d;
            ds[b * 8 + j] = d;
            for (idx, &value) in chunk.iter().enumerate() {
                qs[off + idx] = (value * inv_d).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (qs, ds)
}

fn f32_to_f16_bits(input: &[f32]) -> Vec<u16> {
    input
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DensePleWeightKind {
    Q4K,
    F32,
}

#[cfg(test)]
pub(super) struct Gemma4PleF32DebugStages {
    pub gate: Vec<f32>,
    pub gated: Vec<f32>,
    pub projected: Vec<f32>,
    pub final_hidden: Vec<f32>,
}

#[derive(Clone, Debug)]
struct DenseGemma4PleReplayDump {
    call: usize,
    dir: std::path::PathBuf,
}

static DENSE_GEMMA4_PLE_REPLAY_CALL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn dense_gemma4_ple_replay_request() -> Result<Option<DenseGemma4PleReplayDump>, String> {
    let Some(dir) = std::env::var_os("RNB_DEBUG_GEMMA4_PLE_REPLAY_DIR") else {
        return Ok(None);
    };
    let target = std::env::var("RNB_DEBUG_GEMMA4_PLE_REPLAY_BACKEND_CALL")
        .ok()
        .map(|raw| {
            raw.parse::<usize>().map_err(|err| {
                format!("RNB_DEBUG_GEMMA4_PLE_REPLAY_BACKEND_CALL must be usize, got {raw}: {err}")
            })
        })
        .transpose()?
        .unwrap_or(0);
    let call = DENSE_GEMMA4_PLE_REPLAY_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if call != target {
        return Ok(None);
    }
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).map_err(|err| {
        format!(
            "Gemma4 PLE replay backend dump create dir failed: {}: {err}",
            dir.display()
        )
    })?;
    Ok(Some(DenseGemma4PleReplayDump { call, dir }))
}

fn dense_dump_gemma4_ple_replay_f32(
    dump: &DenseGemma4PleReplayDump,
    name: &str,
    data: &[f32],
) -> Result<(), String> {
    let path = dump
        .dir
        .join(format!("cuda_call{}_{}.bin", dump.call, name));
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    std::fs::write(&path, bytes).map_err(|err| {
        format!(
            "Gemma4 PLE replay backend dump write failed: {}: {err}",
            path.display()
        )
    })?;
    eprintln!(
        "[gemma4-ple-replay-backend] call={} name={} len={} path={}",
        dump.call,
        name,
        data.len(),
        path.display()
    );
    Ok(())
}

fn dense_q8dot_gate_up_enabled(default: bool) -> bool {
    match std::env::var("RNB_CUDA_DENSE_Q8DOT_GATE_UP") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

fn dense_q8dot_down_enabled(default: bool) -> bool {
    match std::env::var("RNB_CUDA_DENSE_Q8DOT_DOWN") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

fn dense_q8dot_qkv_enabled(default: bool) -> bool {
    match std::env::var("RNB_CUDA_DENSE_Q8DOT_QKV") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

fn dense_q6_packed_q8dot_enabled(default: bool) -> bool {
    env_flag_or("RNB_CUDA_DENSE_Q6_PACKED_Q8DOT", default)
}

fn dense_q4_packed_q8dot_enabled(default: bool) -> bool {
    env_flag_or("RNB_CUDA_DENSE_Q4_PACKED_Q8DOT", default)
}

fn expanded_weight_cache_allowed() -> bool {
    crate::tuning::expanded_weight_cache_allowed()
}

fn dense_q4_batch_f32_gate_up_enabled(default: bool) -> bool {
    expanded_weight_cache_allowed() && env_flag_or("RNB_CUDA_Q4K_BATCH_F32_GATE_UP", default)
}

fn dense_q4_batch_f16_gate_up_default(seq_len: usize, n_ff: usize) -> bool {
    let min_activations = env_usize("RNB_CUDA_Q4K_BATCH_F16_GATE_UP_MIN_ACTS", 4 * 1024 * 1024);
    seq_len
        .checked_mul(n_ff)
        .is_some_and(|activations| activations >= min_activations)
}

fn dense_q4_batch_f16_gate_up_enabled_for(
    seq_len: usize,
    n_ff: usize,
    f32_gate_up_enabled: bool,
) -> bool {
    expanded_weight_cache_allowed()
        && env_flag_value("RNB_CUDA_Q4K_BATCH_F16_GATE_UP").unwrap_or_else(|| {
            !f32_gate_up_enabled && dense_q4_batch_f16_gate_up_default(seq_len, n_ff)
        })
}

fn dense_q4_batch_f16_down_enabled(default: bool) -> bool {
    expanded_weight_cache_allowed() && env_flag_or("RNB_CUDA_Q4K_BATCH_F16_DOWN", default)
}

fn dense_q6_batch_f32_down_enabled(default: bool) -> bool {
    expanded_weight_cache_allowed() && env_flag_or("RNB_CUDA_Q6K_BATCH_F32_DOWN", default)
}

pub(in crate::runtime) fn dense_q6_batch_f16_down_enabled_for(
    default: bool,
    seq_len: usize,
    n_ff: usize,
) -> bool {
    if !expanded_weight_cache_allowed() {
        return false;
    }
    let requested = match std::env::var("RNB_CUDA_Q6K_BATCH_F16_DOWN") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            if value == "force" {
                return true;
            }
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    };
    if !requested {
        return false;
    }

    // cu19: max_acts default raised to 128M activations (~256 MiB f16 buffer)
    // to cover long-prompt prefills on 16k FFN models without forcing. The Q6 F16
    // cache path now has a GPU-dequant transient fallback (q6_f16_cache.rs),
    // so the ceiling is bounded by the transient pool size, not by this check.
    // Smaller VRAM still safe: transient pool rotates 8 slots, slot grows to
    // fit largest weight, so steady-state VRAM = 8 * max_weight_f16_bytes.
    let max_activations = env_usize("RNB_CUDA_Q6K_BATCH_F16_DOWN_MAX_ACTS", 128 * 1024 * 1024);
    seq_len
        .checked_mul(n_ff)
        .is_some_and(|activations| activations <= max_activations)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DenseGateUpDispatchPlan {
    PackedQ8Dot,
    RawQuant,
    ExpandedF16,
    ExpandedF32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DenseDownDispatchPlan {
    PackedQ8Dot,
    RawQuant,
    ExpandedF16,
    ExpandedF32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DenseQ4ProjectionKind {
    #[cfg(test)]
    Qkv,
    #[cfg(test)]
    Ple,
    Output,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DenseQ4ProjectionDispatchPlan {
    RawQuant,
    ExpandedF16,
}

fn dense_q4_gate_up_dispatch_plan(
    seq_len: usize,
    n_ff: usize,
    n_embd: usize,
    q8dot_supported: bool,
    packed_can_admit: bool,
) -> DenseGateUpDispatchPlan {
    if q8dot_supported && packed_can_admit && dense_q4_packed_q8dot_enabled(true) {
        return DenseGateUpDispatchPlan::PackedQ8Dot;
    }
    if q8dot_supported {
        return DenseGateUpDispatchPlan::RawQuant;
    }
    let f32_enabled = dense_q4_batch_f32_gate_up_enabled(false);
    if dense_q4_batch_f16_gate_up_enabled_for(seq_len, n_ff, f32_enabled) {
        return DenseGateUpDispatchPlan::ExpandedF16;
    }
    if f32_enabled {
        return DenseGateUpDispatchPlan::ExpandedF32;
    }
    let _ = n_embd;
    DenseGateUpDispatchPlan::RawQuant
}

fn dense_q4_down_dispatch_plan(
    seq_len: usize,
    n_ff: usize,
    n_embd: usize,
    raw_supported: bool,
) -> DenseDownDispatchPlan {
    if raw_supported {
        return DenseDownDispatchPlan::RawQuant;
    }
    if dense_q4_batch_f16_down_enabled(false) {
        return DenseDownDispatchPlan::ExpandedF16;
    }
    let _ = (seq_len, n_ff, n_embd);
    DenseDownDispatchPlan::RawQuant
}

fn dense_q6_down_dispatch_plan(
    gelu: bool,
    down_quant: u32,
    seq_len: usize,
    n_ff: usize,
    n_embd: usize,
    packed_can_admit: bool,
) -> DenseDownDispatchPlan {
    let q8dot_supported = dense_q8dot_down_enabled(gelu && down_quant == 14 && n_ff >= 1024);
    if q8dot_supported && packed_can_admit && dense_q6_packed_q8dot_enabled(true) {
        return DenseDownDispatchPlan::PackedQ8Dot;
    }
    if q8dot_supported {
        return DenseDownDispatchPlan::RawQuant;
    }
    if down_quant == 14 && dense_q6_batch_f32_down_enabled(false) {
        return DenseDownDispatchPlan::ExpandedF32;
    }
    if down_quant == 14 && dense_q6_batch_f16_down_enabled_for(true, seq_len, n_ff) {
        return DenseDownDispatchPlan::ExpandedF16;
    }
    let _ = n_embd;
    DenseDownDispatchPlan::RawQuant
}

fn dense_q4_projection_dispatch_plan(
    kind: DenseQ4ProjectionKind,
    seq_len: usize,
    rows: usize,
    cols: usize,
    raw_supported: bool,
) -> DenseQ4ProjectionDispatchPlan {
    if matches!(kind, DenseQ4ProjectionKind::Output)
        && tuning::prefill_q4k_f16_o_proj_force_enabled()
    {
        return DenseQ4ProjectionDispatchPlan::ExpandedF16;
    }
    if raw_supported {
        return DenseQ4ProjectionDispatchPlan::RawQuant;
    }
    if matches!(kind, DenseQ4ProjectionKind::Output) && tuning::prefill_q4k_f16_o_proj_enabled() {
        return DenseQ4ProjectionDispatchPlan::ExpandedF16;
    }
    let _ = (seq_len, rows, cols);
    DenseQ4ProjectionDispatchPlan::RawQuant
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseGateUpDispatchPlanForTest {
    PackedQ8Dot,
    RawQuant,
    ExpandedF16,
    ExpandedF32,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseDownDispatchPlanForTest {
    PackedQ8Dot,
    RawQuant,
    ExpandedF16,
    ExpandedF32,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseQ4ProjectionKindForTest {
    Qkv,
    Ple,
    Output,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseQ4ProjectionDispatchPlanForTest {
    RawQuant,
    ExpandedF16,
}

#[cfg(test)]
pub fn dense_q4_gate_up_dispatch_plan_for_test(
    seq_len: usize,
    n_ff: usize,
    n_embd: usize,
    q8dot_supported: bool,
    packed_can_admit: bool,
) -> DenseGateUpDispatchPlanForTest {
    match dense_q4_gate_up_dispatch_plan(seq_len, n_ff, n_embd, q8dot_supported, packed_can_admit) {
        DenseGateUpDispatchPlan::PackedQ8Dot => DenseGateUpDispatchPlanForTest::PackedQ8Dot,
        DenseGateUpDispatchPlan::RawQuant => DenseGateUpDispatchPlanForTest::RawQuant,
        DenseGateUpDispatchPlan::ExpandedF16 => DenseGateUpDispatchPlanForTest::ExpandedF16,
        DenseGateUpDispatchPlan::ExpandedF32 => DenseGateUpDispatchPlanForTest::ExpandedF32,
    }
}

#[cfg(test)]
pub fn dense_q4_down_dispatch_plan_for_test(
    seq_len: usize,
    n_ff: usize,
    n_embd: usize,
    raw_supported: bool,
) -> DenseDownDispatchPlanForTest {
    match dense_q4_down_dispatch_plan(seq_len, n_ff, n_embd, raw_supported) {
        DenseDownDispatchPlan::PackedQ8Dot => DenseDownDispatchPlanForTest::PackedQ8Dot,
        DenseDownDispatchPlan::RawQuant => DenseDownDispatchPlanForTest::RawQuant,
        DenseDownDispatchPlan::ExpandedF16 => DenseDownDispatchPlanForTest::ExpandedF16,
        DenseDownDispatchPlan::ExpandedF32 => DenseDownDispatchPlanForTest::ExpandedF32,
    }
}

#[cfg(test)]
pub fn dense_q4_projection_dispatch_plan_for_test(
    kind: DenseQ4ProjectionKindForTest,
    seq_len: usize,
    rows: usize,
    cols: usize,
    raw_supported: bool,
) -> DenseQ4ProjectionDispatchPlanForTest {
    let kind = match kind {
        DenseQ4ProjectionKindForTest::Qkv => DenseQ4ProjectionKind::Qkv,
        DenseQ4ProjectionKindForTest::Ple => DenseQ4ProjectionKind::Ple,
        DenseQ4ProjectionKindForTest::Output => DenseQ4ProjectionKind::Output,
    };
    match dense_q4_projection_dispatch_plan(kind, seq_len, rows, cols, raw_supported) {
        DenseQ4ProjectionDispatchPlan::RawQuant => DenseQ4ProjectionDispatchPlanForTest::RawQuant,
        DenseQ4ProjectionDispatchPlan::ExpandedF16 => {
            DenseQ4ProjectionDispatchPlanForTest::ExpandedF16
        }
    }
}

#[cfg(test)]
pub fn dense_q6_down_dispatch_plan_for_test(
    gelu: bool,
    down_quant: u32,
    seq_len: usize,
    n_ff: usize,
    n_embd: usize,
    packed_can_admit: bool,
) -> DenseDownDispatchPlanForTest {
    match dense_q6_down_dispatch_plan(gelu, down_quant, seq_len, n_ff, n_embd, packed_can_admit) {
        DenseDownDispatchPlan::PackedQ8Dot => DenseDownDispatchPlanForTest::PackedQ8Dot,
        DenseDownDispatchPlan::RawQuant => DenseDownDispatchPlanForTest::RawQuant,
        DenseDownDispatchPlan::ExpandedF16 => DenseDownDispatchPlanForTest::ExpandedF16,
        DenseDownDispatchPlan::ExpandedF32 => DenseDownDispatchPlanForTest::ExpandedF32,
    }
}

#[cfg(test)]
pub(in crate::runtime) fn dense_q4_batch_f16_gate_up_enabled_for_test(
    seq_len: usize,
    n_ff: usize,
    f32_gate_up_enabled: bool,
) -> bool {
    dense_q4_batch_f16_gate_up_enabled_for(seq_len, n_ff, f32_gate_up_enabled)
}

#[cfg(test)]
pub(in crate::runtime) fn dense_q4_batch_f16_down_enabled_for_test(default: bool) -> bool {
    dense_q4_batch_f16_down_enabled(default)
}

fn dense_q4_batch_q8dot_seq4_enabled(default: bool) -> bool {
    env_flag_or("RNB_CUDA_Q4K_BATCH_Q8DOT_SEQ4", default)
}

fn dense_q4_batch_dev_input_q8dot_enabled(
    seq_len: usize,
    rows: usize,
    blocks_per_row: usize,
) -> bool {
    let shape_supported = seq_len > 0 && rows > 0 && blocks_per_row > 0;
    if !shape_supported {
        return false;
    }
    let default = (2..=4).contains(&seq_len) && rows >= 1024 && blocks_per_row >= 4;
    env_flag_value("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV")
        .or_else(|| env_flag_value("RNB_CUDA_Q4K_BATCH_Q8DOT"))
        .unwrap_or(default)
}

fn dense_combined_norms_enabled(default: bool) -> bool {
    env_flag_or("RNB_CUDA_DENSE_COMBINED_NORMS", default)
}

fn dense_ple_gate_gelu_enabled(default: bool) -> bool {
    env_flag_or("RNB_CUDA_DENSE_PLE_GATE_GELU", default)
}

fn dense_q4k_gemv_q8dot_enabled(default: bool) -> bool {
    match std::env::var("RNB_CUDA_Q4K_GEMV_Q8DOT") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests {
    use super::{dense_q4_batch_dev_input_q8dot_enabled, dense_q4_batch_f16_gate_up_enabled_for};
    use std::sync::Mutex;

    static GATE_UP_ENV_LOCK: Mutex<()> = Mutex::new(());
    static Q4_BATCH_DEV_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_clean_gate_up_env(test: impl FnOnce()) {
        let _guard = GATE_UP_ENV_LOCK.lock().expect("gate/up env lock");
        let prev_allow = std::env::var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE").ok();
        let prev = std::env::var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP").ok();
        let prev_mtp = std::env::var("RNB_MTP_DEVICE_VERIFY").ok();
        std::env::remove_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE");
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_F16_GATE_UP");
        std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        test();
        restore_env("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", prev_allow);
        restore_env("RNB_CUDA_Q4K_BATCH_F16_GATE_UP", prev);
        restore_env("RNB_MTP_DEVICE_VERIFY", prev_mtp);
    }

    fn restore_env(name: &str, previous: Option<String>) {
        if let Some(previous) = previous {
            std::env::set_var(name, previous);
        } else {
            std::env::remove_var(name);
        }
    }

    fn with_clean_q4_batch_dev_env(test: impl FnOnce()) {
        let _guard = Q4_BATCH_DEV_ENV_LOCK.lock().expect("Q4 batch dev env lock");
        let prev_dev = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV").ok();
        let prev_batch = std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT").ok();
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV");
        std::env::remove_var("RNB_CUDA_Q4K_BATCH_Q8DOT");
        test();
        restore_env("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV", prev_dev);
        restore_env("RNB_CUDA_Q4K_BATCH_Q8DOT", prev_batch);
    }

    #[test]
    fn q4_f16_gate_up_default_policy_requires_large_prefill_work() {
        with_clean_gate_up_env(|| {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
            assert!(!dense_q4_batch_f16_gate_up_enabled_for(128, 4096, false));
            assert!(dense_q4_batch_f16_gate_up_enabled_for(1024, 6144, false));
        });
    }

    #[test]
    fn q4_f16_gate_up_default_policy_does_not_enable_small_mtp_verify_windows() {
        with_clean_gate_up_env(|| {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
            std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
            assert!(!dense_q4_batch_f16_gate_up_enabled_for(2, 3584, false));
            assert!(!dense_q4_batch_f16_gate_up_enabled_for(4, 3584, false));
            assert!(!dense_q4_batch_f16_gate_up_enabled_for(5, 3584, false));
        });
    }

    #[test]
    fn q4_f16_gate_up_default_policy_yields_to_f32_opt_in() {
        with_clean_gate_up_env(|| {
            std::env::set_var("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
            assert!(!dense_q4_batch_f16_gate_up_enabled_for(1024, 6144, true));
        });
    }

    #[test]
    fn q4_batch_dev_input_q8dot_policy_targets_small_verify_windows() {
        with_clean_q4_batch_dev_env(|| {
            assert!(dense_q4_batch_dev_input_q8dot_enabled(2, 1024, 4));
            assert!(dense_q4_batch_dev_input_q8dot_enabled(4, 1024, 4));
            assert!(!dense_q4_batch_dev_input_q8dot_enabled(1, 1024, 4));
            assert!(!dense_q4_batch_dev_input_q8dot_enabled(5, 1024, 4));
            assert!(!dense_q4_batch_dev_input_q8dot_enabled(2, 512, 4));
            assert!(!dense_q4_batch_dev_input_q8dot_enabled(2, 1024, 3));
        });
    }

    #[test]
    fn q4_batch_dev_input_q8dot_policy_allows_explicit_override() {
        with_clean_q4_batch_dev_env(|| {
            std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV", "0");
            assert!(!dense_q4_batch_dev_input_q8dot_enabled(2, 1024, 4));
            std::env::set_var("RNB_CUDA_Q4K_BATCH_Q8DOT_DEV", "1");
            assert!(dense_q4_batch_dev_input_q8dot_enabled(8, 1024, 4));
        });
    }
}

impl CudaState {
    pub(in crate::runtime) fn resident_f32_weights_ptr_from_le_bytes(
        &mut self,
        bytes: &[u8],
        label: &str,
    ) -> Result<u64, String> {
        if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
            return Err(format!(
                "{label} byte length must be a multiple of 4, got {}",
                bytes.len()
            ));
        }
        let mut bit_hash = 0xcbf29ce484222325_u64;
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            bit_hash ^= bits as u64;
            bit_hash = bit_hash.wrapping_mul(0x100000001b3);
        }
        let key = F32Key {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len() / std::mem::size_of::<f32>(),
            bit_hash,
        };
        if let Some(entry) = self.resident_f32.get(&key) {
            return Ok(entry.ptr);
        }
        // cu111: F32 PLE weight(Gemma4 gate/proj 등) resident 직접 alloc 도 OOM retry
        // 경로로 전환. 이전엔 retry 없는 직접 mem_alloc 이라 VRAM 천장 근처(예: E4B
        // 10GB)에서 50MiB F32 PLE alloc 이 offload 없이 즉시 panic 했다. attention.rs
        // 의 cu26 generic OOM retry 와 동일 패턴: q4k resident offload → MoE cache clear.
        self.reclaim_residency_for_transient(bytes.len())?;
        let ptr = match unsafe { self.api.mem_alloc(bytes.len()) } {
            Ok(p) => p,
            Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                let _ = self.offload_non_pinned_resident_q4k();
                match unsafe { self.api.mem_alloc(bytes.len()) } {
                    Ok(p) => p,
                    Err(err2) if cuda_mem_alloc_oom(&err2) => {
                        self.clear_resident_moe_layer_cache()?;
                        unsafe { self.api.mem_alloc(bytes.len())? }
                    }
                    Err(err2) => return Err(err2),
                }
            }
            Err(err) => return Err(err),
        };
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                bytes.as_ptr().cast::<libc::c_void>(),
                bytes.len(),
                self.stream,
            )?;
        }
        self.resident_f32.insert(key, ResidentF32 { ptr });
        self.record_native_f32_residency(bytes.len());
        Ok(ptr)
    }

    fn debug_copy_device_f32(
        &mut self,
        ptr: u64,
        len: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0f32; len];
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} byte length overflow: len={len}"))?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                ptr,
                bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn q4k_batch_dev_input_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let mmq_tile32 = tuning::q4k_mmq_tile32_enabled(seq_len, rows, blocks_per_row);
        if mmq_tile32
            || (!tuning::q4k_batch_raw_seq4_enabled(seq_len, rows, blocks_per_row)
                && dense_q4_batch_dev_input_q8dot_enabled(seq_len, rows, blocks_per_row))
        {
            let qs_dev = self.compute_gate_ptrs_ptr(seq_len * blocks_per_row * 256)?;
            let ds_dev = self
                .compute_up_ptrs_ptr(seq_len * blocks_per_row * 8 * std::mem::size_of::<f32>())?;
            self.launch_quantize_q8_1_by_32(
                input_dev,
                qs_dev,
                ds_dev,
                seq_len * blocks_per_row * 256,
            )?;
            return self.q4k_batch_q8dot_to_dev(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                qs_dev,
                ds_dev,
                output_dev,
            );
        }

        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let (kernel, grid, block) =
            if tuning::q4k_batch_raw_seq4_enabled(seq_len, rows, blocks_per_row) {
                (
                    "rnb_q4k_gemv_batch_seq4_warp8",
                    (rows.div_ceil(8) as u32, seq_len.div_ceil(4) as u32, 1),
                    (256, 1, 1),
                )
            } else if tuning::q4k_gemv_batch_warp8_enabled() {
                (
                    "rnb_q4k_gemv_batch_warp8",
                    (rows.div_ceil(8) as u32, seq_len as u32, 1),
                    (256, 1, 1),
                )
            } else {
                (
                    "rnb_q4k_gemv_batch",
                    (rows as u32, seq_len as u32, 1),
                    (256, 1, 1),
                )
            };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn q4k_batch_q8dot_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_qs_dev: u64,
        input_ds_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if tuning::q4k_mmq_tile32_enabled(seq_len, rows, blocks_per_row) {
            return self.launch_q4k_q8_1_matmul_mmq_tile32(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_qs_dev,
                input_ds_dev,
                output_dev,
            );
        }
        // cu39 Phase 5/6/7: mma 4-warp 변형 (default seq4_warp8 대체).
        // - V2 (inline sum_qy, 16-iter unpack): +1.2% 회귀 측정
        // - V3 (packed unpack): 16-iter byte loop → 4-int + bit shift
        // cu39 Phase 7 (default ON): 5 모델 (Gemma E2B/E4B, Llama 3.1 8B/3.2 3B,
        // Mistral 7B, Qwen 3.6 35B, Nemotron-H 30B) ABAB 검증 — Gemma -5.7~6.0%,
        // 다른 모델 회귀 ≤+0.3% ε. default ON 안전. env="0" 으로 disable 가능.
        if std::env::var("RNB_CUDA_Q4K_MMA_V3").ok().as_deref() != Some("0")
            && seq_len >= 8
            && rows >= 64
        {
            return self.launch_q4k_q8_1_matmul_mma_4warp_v3(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_qs_dev,
                input_ds_dev,
                output_dev,
            );
        }
        if std::env::var("RNB_CUDA_Q4K_MMA_V2").ok().as_deref() == Some("1")
            && seq_len >= 8
            && rows >= 64
        {
            return self.launch_q4k_q8_1_matmul_mma_4warp_v2(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_qs_dev,
                input_ds_dev,
                output_dev,
            );
        }

        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let use_seq4 = dense_q4_batch_q8dot_seq4_enabled(seq_len > 1);
        let kernel = if use_seq4 {
            "rnb_q4k_gemv_batch_q8dot_seq4_warp8"
        } else {
            "rnb_q4k_gemv_batch_q8dot_warp8"
        };
        let grid_y = if use_seq4 {
            seq_len.div_ceil(4) as u32
        } else {
            seq_len as u32
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, grid_y, 1),
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn q5k_batch_dev_input_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.k_quant_batch_dev_input_to_dev(
            "rnb_q5k_gemv_batch",
            weights,
            rows,
            blocks_per_row,
            seq_len,
            input_dev,
            output_dev,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn q6k_batch_dev_input_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if tuning::q6k_mmq_tile32_enabled(seq_len, rows, blocks_per_row) {
            let qs_dev = self.compute_gate_ptrs_ptr(seq_len * blocks_per_row * 256)?;
            let ds_dev = self
                .compute_up_ptrs_ptr(seq_len * blocks_per_row * 8 * std::mem::size_of::<f32>())?;
            self.launch_quantize_q8_1_by_32(
                input_dev,
                qs_dev,
                ds_dev,
                seq_len * blocks_per_row * 256,
            )?;
            return self.launch_q6k_q8_1_matmul_mmq_tile32(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                qs_dev,
                ds_dev,
                output_dev,
            );
        }
        if seq_len == 2 && tuning::q6k_gemv_batch_seq2_warp8_enabled() {
            return self.launch_q6k_gemv_batch_seq2_warp8_to_dev(
                weights,
                rows,
                blocks_per_row,
                input_dev,
                output_dev,
            );
        }
        let kernel = if tuning::q6k_gemv_batch_warp8_enabled() {
            "rnb_q6k_gemv_batch_warp8"
        } else {
            "rnb_q6k_gemv_batch"
        };
        self.k_quant_batch_dev_input_to_dev(
            kernel,
            weights,
            rows,
            blocks_per_row,
            seq_len,
            input_dev,
            output_dev,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn k_quant_batch_dev_input_to_dev(
        &mut self,
        kernel: &str,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let grid = if kernel.ends_with("_warp8") {
            (rows.div_ceil(8) as u32, seq_len as u32, 1)
        } else {
            (rows as u32, seq_len as u32, 1)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            (256, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_gelu_ffn_batch(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.dense_q4k_ffn_batch(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            input,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_silu_ffn_batch(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.dense_q4k_ffn_batch(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            input,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_q4k_ffn_batch(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        input: &[f32],
        gelu: bool,
    ) -> Result<Vec<f32>, String> {
        let activation_name = if gelu { "GELU" } else { "SiLU" };
        if seq_len <= 1 {
            return Err(format!(
                "dense {activation_name} FFN batch requires seq_len > 1, got {seq_len}"
            ));
        }
        if input.len() != seq_len * n_embd {
            return Err(format!(
                "dense {activation_name} FFN batch input length mismatch: got {}, expected {}",
                input.len(),
                seq_len * n_embd
            ));
        }
        if n_embd % 256 != 0 || n_ff % 256 != 0 {
            return Err(format!(
                "dense {activation_name} FFN batch dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        let gate_blocks = n_embd / 256;
        let down_blocks = n_ff / 256;
        let gate_row_bytes = gate_blocks * 144;
        let down_row_bytes = match down_quant {
            12 => down_blocks * 144,
            13 => down_blocks * 176,
            14 => down_blocks * 210,
            other => {
                return Err(format!(
                    "unsupported dense {activation_name} FFN batch down quant code {other}"
                ))
            }
        };
        if gate_weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "dense {activation_name} FFN batch gate byte mismatch: got {}, expected {}",
                gate_weights.len(),
                n_ff * gate_row_bytes
            ));
        }
        if up_weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "dense {activation_name} FFN batch up byte mismatch: got {}, expected {}",
                up_weights.len(),
                n_ff * gate_row_bytes
            ));
        }
        if down_weights.len() != n_embd * down_row_bytes {
            return Err(format!(
                "dense {activation_name} FFN batch down byte mismatch: got {}, expected {}",
                down_weights.len(),
                n_embd * down_row_bytes
            ));
        }

        let trace_call = dense_chain_trace_call();
        let trace_total = trace_call.map(|_| std::time::Instant::now());
        let mut trace_stage = std::time::Instant::now();
        let gate_dev = self.compute_mid_a_ptr(seq_len * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(seq_len * n_ff * std::mem::size_of::<f32>())?;
        let output_len = seq_len * n_embd;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;

        let q8dot_gate_up = dense_q8dot_gate_up_enabled(n_ff >= 1024 && gate_blocks >= 4);
        let packed_gate_up_supported =
            q8dot_gate_up && seq_len == 2 && tuning::q4k_gate_up_batch_seq2_q8dot_enabled();
        let gate_up_plan = dense_q4_gate_up_dispatch_plan(
            seq_len,
            n_ff,
            n_embd,
            q8dot_gate_up,
            packed_gate_up_supported,
        );
        let packed_gate_up = if matches!(gate_up_plan, DenseGateUpDispatchPlan::PackedQ8Dot) {
            let gate = self.resident_q4k_packed_ptrs(gate_weights, n_ff, gate_blocks)?;
            let up = self.resident_q4k_packed_ptrs(up_weights, n_ff, gate_blocks)?;
            match (gate, up) {
                (Some(gate), Some(up)) => Some((gate, up)),
                _ => None,
            }
        } else {
            None
        };
        let gate_up_plan = if packed_gate_up.is_none()
            && matches!(gate_up_plan, DenseGateUpDispatchPlan::PackedQ8Dot)
        {
            DenseGateUpDispatchPlan::RawQuant
        } else {
            gate_up_plan
        };
        let f16_gate_up = if matches!(gate_up_plan, DenseGateUpDispatchPlan::ExpandedF16) {
            self.resident_q4k_f16_pair_ptrs(gate_weights, up_weights, n_ff, gate_blocks)?
        } else {
            None
        };
        let f32_gate_up = if matches!(gate_up_plan, DenseGateUpDispatchPlan::ExpandedF32) {
            self.resident_q4k_f32_pair_ptrs(gate_weights, up_weights, n_ff, gate_blocks)?
        } else {
            None
        };
        if let Some((gate_w_dev, up_w_dev)) = f16_gate_up {
            let input_f16 = f32_to_f16_bits(input);
            let input_dev = self.compute_input_ptr(std::mem::size_of_val(input_f16.as_slice()))?;
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input_f16.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input_f16.as_slice()),
                    self.stream,
                )?;
            }
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-batch",
                "input_h2d_f16",
                &mut trace_stage,
            )?;
            self.hgemm_to_f32_device(gate_w_dev, n_ff, n_embd, input_dev, seq_len, gate_dev)?;
            self.hgemm_to_f32_device(up_w_dev, n_ff, n_embd, input_dev, seq_len, up_dev)?;
        } else if let Some((gate_w_dev, up_w_dev)) = f32_gate_up {
            let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input),
                    self.stream,
                )?;
            }
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-batch",
                "input_h2d",
                &mut trace_stage,
            )?;
            self.sgemm_device(gate_w_dev, n_ff, n_embd, input_dev, seq_len, gate_dev)?;
            self.sgemm_device(up_w_dev, n_ff, n_embd, input_dev, seq_len, up_dev)?;
        } else if q8dot_gate_up {
            let (qs, ds) = super::gemv::quantize_q8_1_batch_by_32(input, gate_blocks, seq_len);
            let qs_dev = self.compute_gate_ptrs_ptr(qs.len())?;
            let ds_dev = self.compute_up_ptrs_ptr(std::mem::size_of_val(ds.as_slice()))?;
            unsafe {
                self.api.memcpy_htod_async(
                    qs_dev,
                    qs.as_ptr().cast::<libc::c_void>(),
                    qs.len(),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    ds_dev,
                    ds.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(ds.as_slice()),
                    self.stream,
                )?;
            }
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-batch",
                "input_q8_quant+h2d",
                &mut trace_stage,
            )?;
            if let Some((gate_packed, up_packed)) = packed_gate_up {
                self.launch_q4k_packed_gate_up_gemv_batch_seq2_q8dot_to_dev(
                    gate_packed,
                    up_packed,
                    n_ff,
                    gate_blocks,
                    qs_dev,
                    ds_dev,
                    gate_dev,
                    up_dev,
                )?;
            } else if seq_len == 2 && tuning::q4k_gate_up_batch_seq2_q8dot_enabled() {
                self.launch_q4k_gate_up_gemv_batch_seq2_q8dot_to_dev(
                    gate_weights,
                    up_weights,
                    n_ff,
                    gate_blocks,
                    qs_dev,
                    ds_dev,
                    gate_dev,
                    up_dev,
                )?;
            } else {
                self.q4k_batch_q8dot_to_dev(
                    gate_weights,
                    n_ff,
                    gate_blocks,
                    seq_len,
                    qs_dev,
                    ds_dev,
                    gate_dev,
                )?;
                self.q4k_batch_q8dot_to_dev(
                    up_weights,
                    n_ff,
                    gate_blocks,
                    seq_len,
                    qs_dev,
                    ds_dev,
                    up_dev,
                )?;
            }
        } else {
            let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input),
                    self.stream,
                )?;
            }
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-batch",
                "input_h2d",
                &mut trace_stage,
            )?;
            self.q4k_batch_dev_input_to_dev(
                gate_weights,
                n_ff,
                gate_blocks,
                seq_len,
                input_dev,
                gate_dev,
            )?;
            self.q4k_batch_dev_input_to_dev(
                up_weights,
                n_ff,
                gate_blocks,
                seq_len,
                input_dev,
                up_dev,
            )?;
        }
        self.trace_dense_stage(trace_call, "cuda-dense-batch", "gate_up", &mut trace_stage)?;

        let q8dot_down =
            dense_q8dot_down_enabled(gelu && matches!(down_quant, 12 | 14) && n_ff >= 1024);
        let q6_down_plan = if down_quant == 14 {
            dense_q6_down_dispatch_plan(gelu, down_quant, seq_len, n_ff, n_embd, true)
        } else {
            DenseDownDispatchPlan::RawQuant
        };
        let packed_q6_down = if matches!(q6_down_plan, DenseDownDispatchPlan::PackedQ8Dot) {
            self.resident_q6k_packed_ptrs(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let q6_down_plan = if packed_q6_down.is_none()
            && matches!(q6_down_plan, DenseDownDispatchPlan::PackedQ8Dot)
        {
            DenseDownDispatchPlan::RawQuant
        } else {
            q6_down_plan
        };
        let f32_q6_down = if matches!(q6_down_plan, DenseDownDispatchPlan::ExpandedF32) {
            self.resident_q6k_f32_ptr(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let f16_q6_down = if matches!(q6_down_plan, DenseDownDispatchPlan::ExpandedF16) {
            self.resident_q6k_f16_ptr(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let q4_down_plan = if down_quant == 12 {
            dense_q4_down_dispatch_plan(seq_len, n_ff, n_embd, true)
        } else {
            DenseDownDispatchPlan::RawQuant
        };
        let f16_q4_down = if matches!(q4_down_plan, DenseDownDispatchPlan::ExpandedF16) {
            self.resident_q4k_f16_ptr(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let down_q8 = if q8dot_down
            && f32_q6_down.is_none()
            && f16_q6_down.is_none()
            && f16_q4_down.is_none()
        {
            // cu107: down q8 qs must NOT reuse compute_input. In the dense
            // attention+FFN chain `compute_input` doubles as the residual hidden
            // buffer (attention `input_dev` = raw hidden = residual source), so
            // writing q8 quant here corrupts the residual and overflows the
            // post-FFN norm (FLT_MAX → NaN) on FULL-attention layers. The q8 down
            // path never touches compute_down_ptrs (only the f16 down path does),
            // so it is a safe scratch that never aliases the residual.
            let qs_dev = self.compute_down_ptrs_ptr(seq_len * n_ff)?;
            let ds_dev =
                self.compute_aux_output_ptr(seq_len * (n_ff / 32) * std::mem::size_of::<f32>())?;
            if gelu {
                self.launch_gelu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, seq_len * n_ff)?;
            } else {
                self.launch_silu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, seq_len * n_ff)?;
            }
            Some((qs_dev, ds_dev))
        } else if gelu {
            self.launch_gelu_mul(gate_dev, up_dev, seq_len * n_ff)?;
            None
        } else {
            self.launch_silu_mul(gate_dev, up_dev, seq_len * n_ff)?;
            None
        };
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-batch",
            "activation",
            &mut trace_stage,
        )?;

        match down_quant {
            12 => {
                if let Some(down_w_dev) = f16_q4_down {
                    let down_input_f16_dev =
                        self.compute_down_ptrs_ptr(seq_len * n_ff * std::mem::size_of::<u16>())?;
                    self.launch_f32_to_f16(gate_dev, down_input_f16_dev, seq_len * n_ff)?;
                    self.hgemm_to_f32_device(
                        down_w_dev,
                        n_embd,
                        n_ff,
                        down_input_f16_dev,
                        seq_len,
                        output_dev,
                    )
                } else if let Some((qs_dev, ds_dev)) = down_q8 {
                    self.q4k_batch_q8dot_to_dev(
                        down_weights,
                        n_embd,
                        down_blocks,
                        seq_len,
                        qs_dev,
                        ds_dev,
                        output_dev,
                    )
                } else {
                    self.q4k_batch_dev_input_to_dev(
                        down_weights,
                        n_embd,
                        down_blocks,
                        seq_len,
                        gate_dev,
                        output_dev,
                    )
                }
            }
            13 => self.q5k_batch_dev_input_to_dev(
                down_weights,
                n_embd,
                down_blocks,
                seq_len,
                gate_dev,
                output_dev,
            ),
            14 => {
                if let Some(down_w_dev) = f32_q6_down {
                    self.sgemm_device(down_w_dev, n_embd, n_ff, gate_dev, seq_len, output_dev)
                } else if let Some(down_w_dev) = f16_q6_down {
                    let down_input_f16_dev =
                        self.compute_down_ptrs_ptr(seq_len * n_ff * std::mem::size_of::<u16>())?;
                    self.launch_f32_to_f16(gate_dev, down_input_f16_dev, seq_len * n_ff)?;
                    self.hgemm_to_f32_device(
                        down_w_dev,
                        n_embd,
                        n_ff,
                        down_input_f16_dev,
                        seq_len,
                        output_dev,
                    )
                } else if let Some((qs_dev, ds_dev)) = down_q8 {
                    if let Some((packed_qs_dev, packed_d_super_dev, packed_sub_scale_dev)) =
                        packed_q6_down
                    {
                        self.launch_q6k_packed_batch_q8dot_to_dev(
                            packed_qs_dev,
                            packed_d_super_dev,
                            packed_sub_scale_dev,
                            n_embd,
                            down_blocks,
                            seq_len,
                            qs_dev,
                            ds_dev,
                            output_dev,
                        )
                    } else {
                        self.launch_q6k_gemv_batch_q8dot_to_dev(
                            down_weights,
                            n_embd,
                            down_blocks,
                            seq_len,
                            qs_dev,
                            ds_dev,
                            output_dev,
                        )
                    }
                } else {
                    self.q6k_batch_dev_input_to_dev(
                        down_weights,
                        n_embd,
                        down_blocks,
                        seq_len,
                        gate_dev,
                        output_dev,
                    )
                }
            }
            _ => unreachable!("validated dense FFN batch down quant"),
        }?;
        self.trace_dense_stage(trace_call, "cuda-dense-batch", "down", &mut trace_stage)?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        self.trace_dense_stage(trace_call, "cuda-dense-batch", "dtoh", &mut trace_stage)?;
        if let (Some(call), Some(total)) = (trace_call, trace_total) {
            eprintln!(
                "[cuda-dense-batch] call={call} stage=total ms={:.3}",
                total.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_gelu_ffn_batch_dev_input_to_dev(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
        prequantized_q8: Option<(u64, u64)>,
        trace_call: Option<usize>,
        trace_stage: &mut std::time::Instant,
    ) -> Result<(), String> {
        self.dense_q4k_ffn_batch_dev_input_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            input_dev,
            output_dev,
            prequantized_q8,
            trace_call,
            trace_stage,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_silu_ffn_batch_dev_input_to_dev(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
        prequantized_q8: Option<(u64, u64)>,
        trace_call: Option<usize>,
        trace_stage: &mut std::time::Instant,
    ) -> Result<(), String> {
        self.dense_q4k_ffn_batch_dev_input_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            input_dev,
            output_dev,
            prequantized_q8,
            trace_call,
            trace_stage,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_q4k_ffn_batch_dev_input_to_dev(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        input_dev: u64,
        output_dev: u64,
        prequantized_q8: Option<(u64, u64)>,
        trace_call: Option<usize>,
        trace_stage: &mut std::time::Instant,
        gelu: bool,
    ) -> Result<(), String> {
        let trace_call = trace_call.or_else(dense_chain_trace_call);
        let gate_blocks = n_embd / 256;
        let down_blocks = n_ff / 256;
        let gate_dev = self.compute_mid_a_ptr(seq_len * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(seq_len * n_ff * std::mem::size_of::<f32>())?;

        let q8dot_gate_up = dense_q8dot_gate_up_enabled(n_ff >= 1024 && gate_blocks >= 4);
        let packed_gate_up_supported =
            q8dot_gate_up && seq_len == 2 && tuning::q4k_gate_up_batch_seq2_q8dot_enabled();
        let gate_up_plan = dense_q4_gate_up_dispatch_plan(
            seq_len,
            n_ff,
            n_embd,
            q8dot_gate_up,
            packed_gate_up_supported,
        );
        let packed_gate_up = if matches!(gate_up_plan, DenseGateUpDispatchPlan::PackedQ8Dot) {
            let gate = self.resident_q4k_packed_ptrs(gate_weights, n_ff, gate_blocks)?;
            let up = self.resident_q4k_packed_ptrs(up_weights, n_ff, gate_blocks)?;
            match (gate, up) {
                (Some(gate), Some(up)) => Some((gate, up)),
                _ => None,
            }
        } else {
            None
        };
        let gate_up_plan = if packed_gate_up.is_none()
            && matches!(gate_up_plan, DenseGateUpDispatchPlan::PackedQ8Dot)
        {
            DenseGateUpDispatchPlan::RawQuant
        } else {
            gate_up_plan
        };
        let f16_gate_up = if matches!(gate_up_plan, DenseGateUpDispatchPlan::ExpandedF16) {
            self.resident_q4k_f16_pair_ptrs(gate_weights, up_weights, n_ff, gate_blocks)?
        } else {
            None
        };
        let f32_gate_up = if matches!(gate_up_plan, DenseGateUpDispatchPlan::ExpandedF32) {
            self.resident_q4k_f32_pair_ptrs(gate_weights, up_weights, n_ff, gate_blocks)?
        } else {
            None
        };
        if let Some((gate_w_dev, up_w_dev)) = f16_gate_up {
            // cu107: f16 gate/up input must NOT reuse compute_input — it doubles
            // as the residual hidden carrier in the attention+FFN chain, so the
            // f32→f16 write here corrupts the residual (f16 bits read back as f32
            // → ~2.6e9) on FULL-attention layers. compute_down_ptrs is unused on
            // the gate/up f16 path, safe scratch that never aliases the residual.
            let input_f16_dev =
                self.compute_down_ptrs_ptr(seq_len * n_embd * std::mem::size_of::<u16>())?;
            self.launch_f32_to_f16(input_dev, input_f16_dev, seq_len * n_embd)?;
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-batch-dev",
                "input_f32_to_f16",
                trace_stage,
            )?;
            self.hgemm_to_f32_device(gate_w_dev, n_ff, n_embd, input_f16_dev, seq_len, gate_dev)?;
            self.hgemm_to_f32_device(up_w_dev, n_ff, n_embd, input_f16_dev, seq_len, up_dev)?;
        } else if let Some((gate_w_dev, up_w_dev)) = f32_gate_up {
            self.sgemm_device(gate_w_dev, n_ff, n_embd, input_dev, seq_len, gate_dev)?;
            self.sgemm_device(up_w_dev, n_ff, n_embd, input_dev, seq_len, up_dev)?;
        } else if q8dot_gate_up {
            let (qs_dev, ds_dev) = if let Some(prequantized_q8) = prequantized_q8 {
                prequantized_q8
            } else {
                let qs_dev = self.compute_gate_ptrs_ptr(seq_len * n_embd)?;
                let ds_dev =
                    self.compute_up_ptrs_ptr(seq_len * (n_embd / 32) * std::mem::size_of::<f32>())?;
                self.launch_quantize_q8_1_by_32(input_dev, qs_dev, ds_dev, seq_len * n_embd)?;
                self.trace_dense_stage(
                    trace_call,
                    "cuda-dense-batch-dev",
                    "input_q8_quant",
                    trace_stage,
                )?;
                (qs_dev, ds_dev)
            };
            if let Some((gate_packed, up_packed)) = packed_gate_up {
                self.launch_q4k_packed_gate_up_gemv_batch_seq2_q8dot_to_dev(
                    gate_packed,
                    up_packed,
                    n_ff,
                    gate_blocks,
                    qs_dev,
                    ds_dev,
                    gate_dev,
                    up_dev,
                )?;
            } else if seq_len == 2 && tuning::q4k_gate_up_batch_seq2_q8dot_enabled() {
                self.launch_q4k_gate_up_gemv_batch_seq2_q8dot_to_dev(
                    gate_weights,
                    up_weights,
                    n_ff,
                    gate_blocks,
                    qs_dev,
                    ds_dev,
                    gate_dev,
                    up_dev,
                )?;
            } else {
                self.q4k_batch_q8dot_to_dev(
                    gate_weights,
                    n_ff,
                    gate_blocks,
                    seq_len,
                    qs_dev,
                    ds_dev,
                    gate_dev,
                )?;
                self.q4k_batch_q8dot_to_dev(
                    up_weights,
                    n_ff,
                    gate_blocks,
                    seq_len,
                    qs_dev,
                    ds_dev,
                    up_dev,
                )?;
            }
        } else {
            self.q4k_batch_dev_input_to_dev(
                gate_weights,
                n_ff,
                gate_blocks,
                seq_len,
                input_dev,
                gate_dev,
            )?;
            self.q4k_batch_dev_input_to_dev(
                up_weights,
                n_ff,
                gate_blocks,
                seq_len,
                input_dev,
                up_dev,
            )?;
        }
        self.trace_dense_stage(trace_call, "cuda-dense-batch-dev", "gate_up", trace_stage)?;

        let q8dot_down =
            dense_q8dot_down_enabled(gelu && matches!(down_quant, 12 | 14) && n_ff >= 1024);
        let q6_down_plan = if down_quant == 14 {
            dense_q6_down_dispatch_plan(gelu, down_quant, seq_len, n_ff, n_embd, true)
        } else {
            DenseDownDispatchPlan::RawQuant
        };
        let packed_q6_down = if matches!(q6_down_plan, DenseDownDispatchPlan::PackedQ8Dot) {
            self.resident_q6k_packed_ptrs(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let q6_down_plan = if packed_q6_down.is_none()
            && matches!(q6_down_plan, DenseDownDispatchPlan::PackedQ8Dot)
        {
            DenseDownDispatchPlan::RawQuant
        } else {
            q6_down_plan
        };
        let f32_q6_down = if matches!(q6_down_plan, DenseDownDispatchPlan::ExpandedF32) {
            self.resident_q6k_f32_ptr(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let f16_q6_down = if matches!(q6_down_plan, DenseDownDispatchPlan::ExpandedF16) {
            self.resident_q6k_f16_ptr(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let q4_down_plan = if down_quant == 12 {
            dense_q4_down_dispatch_plan(seq_len, n_ff, n_embd, true)
        } else {
            DenseDownDispatchPlan::RawQuant
        };
        let f16_q4_down = if matches!(q4_down_plan, DenseDownDispatchPlan::ExpandedF16) {
            self.resident_q4k_f16_ptr(down_weights, n_embd, down_blocks)?
        } else {
            None
        };
        let down_q8 = if q8dot_down
            && f32_q6_down.is_none()
            && f16_q6_down.is_none()
            && f16_q4_down.is_none()
        {
            // cu107: down q8 qs must NOT reuse compute_input. In the dense
            // attention+FFN chain `compute_input` doubles as the residual hidden
            // buffer (attention `input_dev` = raw hidden = residual source), so
            // writing q8 quant here corrupts the residual and overflows the
            // post-FFN norm (FLT_MAX → NaN) on FULL-attention layers. The q8 down
            // path never touches compute_down_ptrs (only the f16 down path does),
            // so it is a safe scratch that never aliases the residual.
            let qs_dev = self.compute_down_ptrs_ptr(seq_len * n_ff)?;
            let ds_dev =
                self.compute_aux_output_ptr(seq_len * (n_ff / 32) * std::mem::size_of::<f32>())?;
            if gelu {
                self.launch_gelu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, seq_len * n_ff)?;
            } else {
                self.launch_silu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, seq_len * n_ff)?;
            }
            Some((qs_dev, ds_dev))
        } else if gelu {
            self.launch_gelu_mul(gate_dev, up_dev, seq_len * n_ff)?;
            None
        } else {
            self.launch_silu_mul(gate_dev, up_dev, seq_len * n_ff)?;
            None
        };
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-batch-dev",
            "activation",
            trace_stage,
        )?;

        match down_quant {
            12 => {
                if let Some(down_w_dev) = f16_q4_down {
                    let down_input_f16_dev =
                        self.compute_down_ptrs_ptr(seq_len * n_ff * std::mem::size_of::<u16>())?;
                    self.launch_f32_to_f16(gate_dev, down_input_f16_dev, seq_len * n_ff)?;
                    self.hgemm_to_f32_device(
                        down_w_dev,
                        n_embd,
                        n_ff,
                        down_input_f16_dev,
                        seq_len,
                        output_dev,
                    )
                } else if let Some((qs_dev, ds_dev)) = down_q8 {
                    self.q4k_batch_q8dot_to_dev(
                        down_weights,
                        n_embd,
                        down_blocks,
                        seq_len,
                        qs_dev,
                        ds_dev,
                        output_dev,
                    )
                } else {
                    self.q4k_batch_dev_input_to_dev(
                        down_weights,
                        n_embd,
                        down_blocks,
                        seq_len,
                        gate_dev,
                        output_dev,
                    )
                }
            }
            13 => self.q5k_batch_dev_input_to_dev(
                down_weights,
                n_embd,
                down_blocks,
                seq_len,
                gate_dev,
                output_dev,
            ),
            14 => {
                if let Some(down_w_dev) = f32_q6_down {
                    self.sgemm_device(down_w_dev, n_embd, n_ff, gate_dev, seq_len, output_dev)
                } else if let Some(down_w_dev) = f16_q6_down {
                    let down_input_f16_dev =
                        self.compute_down_ptrs_ptr(seq_len * n_ff * std::mem::size_of::<u16>())?;
                    self.launch_f32_to_f16(gate_dev, down_input_f16_dev, seq_len * n_ff)?;
                    self.hgemm_to_f32_device(
                        down_w_dev,
                        n_embd,
                        n_ff,
                        down_input_f16_dev,
                        seq_len,
                        output_dev,
                    )
                } else if let Some((qs_dev, ds_dev)) = down_q8 {
                    if let Some((packed_qs_dev, packed_d_super_dev, packed_sub_scale_dev)) =
                        packed_q6_down
                    {
                        self.launch_q6k_packed_batch_q8dot_to_dev(
                            packed_qs_dev,
                            packed_d_super_dev,
                            packed_sub_scale_dev,
                            n_embd,
                            down_blocks,
                            seq_len,
                            qs_dev,
                            ds_dev,
                            output_dev,
                        )
                    } else {
                        self.launch_q6k_gemv_batch_q8dot_to_dev(
                            down_weights,
                            n_embd,
                            down_blocks,
                            seq_len,
                            qs_dev,
                            ds_dev,
                            output_dev,
                        )
                    }
                } else {
                    self.q6k_batch_dev_input_to_dev(
                        down_weights,
                        n_embd,
                        down_blocks,
                        seq_len,
                        gate_dev,
                        output_dev,
                    )
                }
            }
            _ => unreachable!("validated dense GELU FFN batch down quant"),
        }?;
        self.trace_dense_stage(trace_call, "cuda-dense-batch-dev", "down", trace_stage)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
        &mut self,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<(), String> {
        if seq_len <= 1 {
            return Err(format!(
                "dense attention+FFN batch chain requires seq_len > 1, got {seq_len}"
            ));
        }
        if hidden.len() != seq_len * n_embd {
            return Err(format!(
                "dense attention+FFN batch hidden length mismatch: got {}, expected {}",
                hidden.len(),
                seq_len * n_embd
            ));
        }
        if attn_out.len() != seq_len * o_cols {
            return Err(format!(
                "dense attention+FFN batch attn_out length mismatch: got {}, expected {}",
                attn_out.len(),
                seq_len * o_cols
            ));
        }
        if o_cols % 256 != 0 || n_embd % 256 != 0 || n_ff % 256 != 0 {
            return Err(format!(
                "dense attention+FFN batch dims must be divisible by 256, got o_cols={o_cols} n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        if ffn_norm_weight.len() != n_embd {
            return Err(format!(
                "dense attention+FFN batch ffn norm length mismatch: got {}, expected {n_embd}",
                ffn_norm_weight.len()
            ));
        }
        if let Some(weight) = post_attn_norm_weight {
            if weight.len() != n_embd {
                return Err(format!(
                    "dense attention+FFN batch post attention norm length mismatch: got {}, expected {n_embd}",
                    weight.len()
                ));
            }
        }
        if let Some(weight) = post_ffn_norm_weight {
            if weight.len() != n_embd {
                return Err(format!(
                    "dense attention+FFN batch post FFN norm length mismatch: got {}, expected {n_embd}",
                    weight.len()
                ));
            }
        }
        let o_row_bytes = (o_cols / 256) * 144;
        let gate_row_bytes = (n_embd / 256) * 144;
        let down_row_bytes = match down_quant {
            12 => (n_ff / 256) * 144,
            13 => (n_ff / 256) * 176,
            14 => (n_ff / 256) * 210,
            other => {
                return Err(format!(
                    "unsupported dense attention+FFN batch down quant code {other}"
                ))
            }
        };
        if o_weights.len() != n_embd * o_row_bytes {
            return Err(format!(
                "dense attention+FFN batch o_proj byte mismatch: got {}, expected {}",
                o_weights.len(),
                n_embd * o_row_bytes
            ));
        }
        if gate_weights.len() != n_ff * gate_row_bytes || up_weights.len() != n_ff * gate_row_bytes
        {
            return Err(format!(
                "dense attention+FFN batch gate/up byte mismatch: gate={} up={} expected {}",
                gate_weights.len(),
                up_weights.len(),
                n_ff * gate_row_bytes
            ));
        }
        if down_weights.len() != n_embd * down_row_bytes {
            return Err(format!(
                "dense attention+FFN batch down byte mismatch: got {}, expected {}",
                down_weights.len(),
                n_embd * down_row_bytes
            ));
        }

        let trace_call = dense_chain_trace_call();
        let trace_total = trace_call.map(|_| std::time::Instant::now());
        let mut trace_stage = std::time::Instant::now();
        let hidden_bytes = std::mem::size_of_val(hidden);
        let attn_out_bytes = std::mem::size_of_val(attn_out);
        let hidden_dev = self.compute_full_gate_ptr(hidden_bytes)?;
        let attn_out_dev = self.compute_full_down_ptr(attn_out_bytes)?;
        let normed_dev = self.compute_full_up_ptr(hidden_bytes)?;
        let proj_dev = self.compute_output_ptr(hidden_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                hidden_dev,
                hidden.as_ptr().cast::<libc::c_void>(),
                hidden_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                attn_out_dev,
                attn_out.as_ptr().cast::<libc::c_void>(),
                attn_out_bytes,
                self.stream,
            )?;
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-batch-chain",
            "h2d_hidden_attn",
            &mut trace_stage,
        )?;

        let o_blocks = o_cols / 256;
        let o_f16 = if matches!(
            dense_q4_projection_dispatch_plan(
                DenseQ4ProjectionKind::Output,
                seq_len,
                n_embd,
                o_cols,
                true,
            ),
            DenseQ4ProjectionDispatchPlan::ExpandedF16
        ) {
            match self.resident_q4k_f16_ptr(o_weights, n_embd, o_blocks)? {
                Some(ptr) => Some(ptr),
                None => Some(self.transient_q4k_f16_ptr(o_weights, n_embd, o_blocks)?),
            }
        } else {
            None
        };
        if let Some(o_weights_dev) = o_f16 {
            let attn_out_f16_dev =
                self.compute_aux_output_ptr(seq_len * o_cols * std::mem::size_of::<u16>())?;
            self.launch_f32_to_f16(attn_out_dev, attn_out_f16_dev, seq_len * o_cols)?;
            self.hgemm_to_f32_device(
                o_weights_dev,
                n_embd,
                o_cols,
                attn_out_f16_dev,
                seq_len,
                proj_dev,
            )?;
        } else {
            self.q4k_batch_dev_input_to_dev(
                o_weights,
                n_embd,
                o_blocks,
                seq_len,
                attn_out_dev,
                proj_dev,
            )?;
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-batch-chain",
            "o_proj",
            &mut trace_stage,
        )?;

        let ffn_norm_dev = self.resident_f32_ptr(ffn_norm_weight)?;
        let mut prequantized_q8 = None;
        if let Some(post_weight) = post_attn_norm_weight {
            let post_weight_dev = self.resident_f32_ptr(post_weight)?;
            if dense_combined_norms_enabled(true) && dense_q8dot_gate_up_enabled(true) {
                let q8_qs_dev = self.compute_gate_ptrs_ptr(seq_len * n_embd)?;
                let q8_ds_dev =
                    self.compute_up_ptrs_ptr(seq_len * (n_embd / 32) * std::mem::size_of::<f32>())?;
                self.launch_rms_norm_add_then_rms_norm_rows_q8_1_f32(
                    proj_dev,
                    post_weight_dev,
                    hidden_dev,
                    ffn_norm_dev,
                    normed_dev,
                    q8_qs_dev,
                    q8_ds_dev,
                    norm_eps,
                    seq_len,
                    n_embd,
                    unit_offset_post_attn_norm,
                    unit_offset_ffn_norm,
                )?;
                prequantized_q8 = Some((q8_qs_dev, q8_ds_dev));
                self.trace_dense_stage(
                    trace_call,
                    "cuda-dense-batch-chain",
                    "post_attn_resid_norm+ffn_pre_norm+input_q8_quant",
                    &mut trace_stage,
                )?;
            } else {
                self.launch_rms_norm_add_then_rms_norm_rows_f32(
                    proj_dev,
                    post_weight_dev,
                    hidden_dev,
                    ffn_norm_dev,
                    normed_dev,
                    norm_eps,
                    seq_len,
                    n_embd,
                    unit_offset_post_attn_norm,
                    unit_offset_ffn_norm,
                )?;
                self.trace_dense_stage(
                    trace_call,
                    "cuda-dense-batch-chain",
                    "post_attn_resid_norm+ffn_pre_norm",
                    &mut trace_stage,
                )?;
            }
        } else {
            self.launch_add_f32_inplace(hidden_dev, proj_dev, seq_len * n_embd)?;
            self.launch_rms_norm_rows_f32(
                hidden_dev,
                ffn_norm_dev,
                normed_dev,
                norm_eps,
                seq_len,
                n_embd,
                unit_offset_ffn_norm,
            )?;
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-batch-chain",
                "attn_resid+ffn_pre_norm",
                &mut trace_stage,
            )?;
        }

        self.dense_q4k_gelu_ffn_batch_dev_input_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            normed_dev,
            proj_dev,
            prequantized_q8,
            trace_call,
            &mut trace_stage,
        )?;
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-batch-chain",
            "ffn",
            &mut trace_stage,
        )?;

        if let Some(weight) = post_ffn_norm_weight {
            let weight_dev = self.resident_f32_ptr(weight)?;
            self.launch_rms_norm_add_rows_f32_inplace(
                proj_dev,
                weight_dev,
                hidden_dev,
                norm_eps,
                seq_len,
                n_embd,
                unit_offset_post_ffn_norm,
            )?;
        } else {
            self.launch_add_f32_inplace(hidden_dev, proj_dev, seq_len * n_embd)?;
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-batch-chain",
            "post_ffn_resid_norm",
            &mut trace_stage,
        )?;

        unsafe {
            self.api.memcpy_dtoh_async(
                hidden.as_mut_ptr().cast::<libc::c_void>(),
                hidden_dev,
                hidden_bytes,
                self.stream,
            )?;
        }
        if trace_call.is_some() {
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-batch-chain",
                "dtoh_hidden",
                &mut trace_stage,
            )?;
            if let (Some(call), Some(total)) = (trace_call, trace_total) {
                eprintln!(
                    "[cuda-dense-batch-chain] call={call} stage=total ms={:.3}",
                    total.elapsed().as_secs_f64() * 1000.0
                );
            }
        } else {
            self.stream_synchronize()?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
        &mut self,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        ple_gate_weights: Option<&[u8]>,
        ple_proj_weights: Option<&[u8]>,
        ple_post_norm_weight: Option<&[f32]>,
        ple_input: Option<&[f32]>,
        ple_dim: usize,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        seq_len: usize,
        hidden: &mut [f32],
        hidden_dev_override: Option<u64>,
        attn_out_dev: u64,
        device_output_desc: Option<rnb_backend_api::DeviceTensorDesc>,
        layer_out_scale: Option<&[f32]>,
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<Option<rnb_backend_api::DeviceTensorId>, String> {
        let trace_call = dense_chain_trace_call();
        let trace_total = trace_call.map(|_| std::time::Instant::now());
        let mut trace_stage = std::time::Instant::now();
        let hidden_bytes = std::mem::size_of_val(hidden);
        let hidden_dev = if let Some(hidden_dev) = hidden_dev_override {
            hidden_dev
        } else {
            self.compute_full_gate_ptr(hidden_bytes)?
        };
        let normed_dev = self.compute_full_up_ptr(hidden_bytes)?;
        let proj_dev = self.compute_output_ptr(hidden_bytes)?;
        let ple_weight_kind = if ple_gate_weights.is_some()
            || ple_proj_weights.is_some()
            || ple_post_norm_weight.is_some()
            || ple_input.is_some()
        {
            let (
                Some(ple_gate_weights),
                Some(ple_proj_weights),
                Some(ple_post_norm_weight),
                Some(ple_input),
            ) = (
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
            )
            else {
                return Err(
                    "dense attention+FFN batch PLE parameters must be all present or all absent"
                        .to_string(),
                );
            };
            if ple_dim == 0 || !ple_dim.is_multiple_of(256) {
                return Err(format!(
                    "dense attention+FFN batch PLE dim must be non-zero and divisible by 256, got {ple_dim}"
                ));
            }
            if ple_input.len() != seq_len.saturating_mul(ple_dim) {
                return Err(format!(
                    "dense attention+FFN batch PLE input length mismatch: got {}, expected {}",
                    ple_input.len(),
                    seq_len.saturating_mul(ple_dim)
                ));
            }
            if ple_post_norm_weight.len() != n_embd {
                return Err(format!(
                    "dense attention+FFN batch PLE post norm length mismatch: got {}, expected {n_embd}",
                    ple_post_norm_weight.len()
                ));
            }
            let ple_gate_row_bytes = (n_embd / 256) * 144;
            let ple_proj_row_bytes = (ple_dim / 256) * 144;
            let q4k_gate_bytes = ple_dim.saturating_mul(ple_gate_row_bytes);
            let q4k_proj_bytes = n_embd.saturating_mul(ple_proj_row_bytes);
            let f32_gate_bytes = ple_dim
                .checked_mul(n_embd)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    "dense attention+FFN batch PLE F32 gate byte overflow".to_string()
                })?;
            let f32_proj_bytes = n_embd
                .checked_mul(ple_dim)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    "dense attention+FFN batch PLE F32 proj byte overflow".to_string()
                })?;
            if ple_gate_weights.len() == q4k_gate_bytes && ple_proj_weights.len() == q4k_proj_bytes
            {
                Some(DensePleWeightKind::Q4K)
            } else if ple_gate_weights.len() == f32_gate_bytes
                && ple_proj_weights.len() == f32_proj_bytes
            {
                Some(DensePleWeightKind::F32)
            } else {
                return Err(format!(
                    "dense attention+FFN batch PLE byte mismatch: gate got {}, expected q4k {} or f32 {}; proj got {}, expected q4k {} or f32 {}",
                    ple_gate_weights.len(),
                    q4k_gate_bytes,
                    f32_gate_bytes,
                    ple_proj_weights.len(),
                    q4k_proj_bytes,
                    f32_proj_bytes
                ));
            }
        } else {
            None
        };
        if let Some(scale) = layer_out_scale {
            if scale.is_empty() {
                return Err("dense attention+FFN layer out_scale must be non-empty".to_string());
            }
        }
        if hidden_dev_override.is_none() {
            unsafe {
                self.api.memcpy_htod_async(
                    hidden_dev,
                    hidden.as_ptr().cast::<libc::c_void>(),
                    hidden_bytes,
                    self.stream,
                )?;
            }
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-attn-dense-batch-chain",
            if hidden_dev_override.is_some() {
                "reuse_hidden_dev"
            } else {
                "h2d_hidden"
            },
            &mut trace_stage,
        )?;

        let o_blocks = o_cols / 256;
        let o_f16 = if matches!(
            dense_q4_projection_dispatch_plan(
                DenseQ4ProjectionKind::Output,
                seq_len,
                n_embd,
                o_cols,
                true,
            ),
            DenseQ4ProjectionDispatchPlan::ExpandedF16
        ) {
            match self.resident_q4k_f16_ptr(o_weights, n_embd, o_blocks)? {
                Some(ptr) => Some(ptr),
                None => Some(self.transient_q4k_f16_ptr(o_weights, n_embd, o_blocks)?),
            }
        } else {
            None
        };
        if let Some(o_weights_dev) = o_f16 {
            let attn_out_f16_dev =
                self.compute_aux_output_ptr(seq_len * o_cols * std::mem::size_of::<u16>())?;
            self.launch_f32_to_f16(attn_out_dev, attn_out_f16_dev, seq_len * o_cols)?;
            self.hgemm_to_f32_device(
                o_weights_dev,
                n_embd,
                o_cols,
                attn_out_f16_dev,
                seq_len,
                proj_dev,
            )?;
        } else {
            // cu19: avoid the alias between `attn_out_dev` and `proj_dev`. The
            // caller's `attn_out_dev` is `compute_output_ptr(q_bytes)` and
            // `proj_dev` above is `compute_output_ptr(hidden_bytes)`. When
            // `q_rows == n_embd` (typical Gemma4 shape) both reservations
            // share the same underlying slab, so a non-aliasing GEMV would read
            // its own output. Raw Q4 projection is now the product default, so
            // keep the alias-safe aux staging instead of falling back to an
            // expanded F16 O-proj buffer.
            let proj_tmp =
                self.compute_aux_output_ptr(seq_len * n_embd * std::mem::size_of::<f32>())?;
            self.q4k_batch_dev_input_to_dev(
                o_weights,
                n_embd,
                o_blocks,
                seq_len,
                attn_out_dev,
                proj_tmp,
            )?;
            unsafe {
                self.api.memcpy_dtod_async(
                    proj_dev,
                    proj_tmp,
                    seq_len * n_embd * std::mem::size_of::<f32>(),
                    self.stream,
                )?;
            }
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-attn-dense-batch-chain",
            "o_proj",
            &mut trace_stage,
        )?;

        let ffn_norm_dev = self.resident_f32_ptr(ffn_norm_weight)?;
        let mut prequantized_q8 = None;
        if let Some(post_weight) = post_attn_norm_weight {
            let post_weight_dev = self.resident_f32_ptr(post_weight)?;
            if dense_combined_norms_enabled(true) && dense_q8dot_gate_up_enabled(true) {
                let q8_qs_dev = self.compute_gate_ptrs_ptr(seq_len * n_embd)?;
                let q8_ds_dev =
                    self.compute_up_ptrs_ptr(seq_len * (n_embd / 32) * std::mem::size_of::<f32>())?;
                self.launch_rms_norm_add_then_rms_norm_rows_q8_1_f32(
                    proj_dev,
                    post_weight_dev,
                    hidden_dev,
                    ffn_norm_dev,
                    normed_dev,
                    q8_qs_dev,
                    q8_ds_dev,
                    norm_eps,
                    seq_len,
                    n_embd,
                    unit_offset_post_attn_norm,
                    unit_offset_ffn_norm,
                )?;
                prequantized_q8 = Some((q8_qs_dev, q8_ds_dev));
                self.trace_dense_stage(
                    trace_call,
                    "cuda-attn-dense-batch-chain",
                    "post_attn_resid_norm+ffn_pre_norm+input_q8_quant",
                    &mut trace_stage,
                )?;
            } else {
                self.launch_rms_norm_add_then_rms_norm_rows_f32(
                    proj_dev,
                    post_weight_dev,
                    hidden_dev,
                    ffn_norm_dev,
                    normed_dev,
                    norm_eps,
                    seq_len,
                    n_embd,
                    unit_offset_post_attn_norm,
                    unit_offset_ffn_norm,
                )?;
                self.trace_dense_stage(
                    trace_call,
                    "cuda-attn-dense-batch-chain",
                    "post_attn_resid_norm+ffn_pre_norm",
                    &mut trace_stage,
                )?;
            }
        } else {
            self.launch_add_f32_inplace(hidden_dev, proj_dev, seq_len * n_embd)?;
            self.launch_rms_norm_rows_f32(
                hidden_dev,
                ffn_norm_dev,
                normed_dev,
                norm_eps,
                seq_len,
                n_embd,
                unit_offset_ffn_norm,
            )?;
            self.trace_dense_stage(
                trace_call,
                "cuda-attn-dense-batch-chain",
                "attn_resid+ffn_pre_norm",
                &mut trace_stage,
            )?;
        }

        self.dense_q4k_gelu_ffn_batch_dev_input_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            normed_dev,
            proj_dev,
            prequantized_q8,
            trace_call,
            &mut trace_stage,
        )?;
        self.trace_dense_stage(
            trace_call,
            "cuda-attn-dense-batch-chain",
            "ffn",
            &mut trace_stage,
        )?;

        if let Some(weight) = post_ffn_norm_weight {
            let weight_dev = self.resident_f32_ptr(weight)?;
            self.launch_rms_norm_add_rows_f32_inplace(
                proj_dev,
                weight_dev,
                hidden_dev,
                norm_eps,
                seq_len,
                n_embd,
                unit_offset_post_ffn_norm,
            )?;
        } else {
            self.launch_add_f32_inplace(hidden_dev, proj_dev, seq_len * n_embd)?;
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-attn-dense-batch-chain",
            "post_ffn_resid_norm",
            &mut trace_stage,
        )?;

        let mut ple_replay_dump = None;
        if let (
            Some(ple_gate_weights),
            Some(ple_proj_weights),
            Some(ple_post_norm_weight),
            Some(ple_input),
        ) = (
            ple_gate_weights,
            ple_proj_weights,
            ple_post_norm_weight,
            ple_input,
        ) {
            let ple_bytes = std::mem::size_of_val(ple_input);
            let ple_input_dev = self.compute_full_down_ptr(ple_bytes)?;
            let ple_gate_dev = self.compute_mid_a_ptr(ple_bytes)?;
            unsafe {
                self.api.memcpy_htod_async(
                    ple_input_dev,
                    ple_input.as_ptr().cast::<libc::c_void>(),
                    ple_bytes,
                    self.stream,
                )?;
            }
            self.trace_dense_stage(
                trace_call,
                "cuda-attn-dense-batch-chain",
                "ple_h2d",
                &mut trace_stage,
            )?;
            let ple_weight_kind = ple_weight_kind
                .ok_or_else(|| "dense attention+FFN batch PLE weight kind missing".to_string())?;
            if ple_weight_kind == DensePleWeightKind::F32 {
                ple_replay_dump = dense_gemma4_ple_replay_request()?;
                if let Some(dump) = &ple_replay_dump {
                    let pre_ple = self.debug_copy_device_f32(
                        hidden_dev,
                        seq_len * n_embd,
                        "Gemma4 PLE replay pre hidden",
                    )?;
                    dense_dump_gemma4_ple_replay_f32(dump, "ple_hidden", &pre_ple)?;
                }
            }
            match ple_weight_kind {
                DensePleWeightKind::Q4K => {
                    self.q4k_batch_dev_input_to_dev(
                        ple_gate_weights,
                        ple_dim,
                        n_embd / 256,
                        seq_len,
                        hidden_dev,
                        ple_gate_dev,
                    )?;
                    self.launch_gelu_mul(ple_gate_dev, ple_input_dev, seq_len * ple_dim)?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-attn-dense-batch-chain",
                        "ple_gate+gelu",
                        &mut trace_stage,
                    )?;
                    self.q4k_batch_dev_input_to_dev(
                        ple_proj_weights,
                        n_embd,
                        ple_dim / 256,
                        seq_len,
                        ple_gate_dev,
                        proj_dev,
                    )?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-attn-dense-batch-chain",
                        "ple_proj",
                        &mut trace_stage,
                    )?;
                }
                DensePleWeightKind::F32 => {
                    let gate_dev = self.resident_f32_weights_ptr_from_le_bytes(
                        ple_gate_weights,
                        "dense attention+FFN PLE gate",
                    )?;
                    self.sgemm_device(
                        gate_dev,
                        ple_dim,
                        n_embd,
                        hidden_dev,
                        seq_len,
                        ple_gate_dev,
                    )?;
                    if let Some(dump) = &ple_replay_dump {
                        let gate = self.debug_copy_device_f32(
                            ple_gate_dev,
                            seq_len * ple_dim,
                            "Gemma4 PLE replay gate",
                        )?;
                        dense_dump_gemma4_ple_replay_f32(dump, "ple_gate", &gate)?;
                    }
                    self.launch_gelu_mul(ple_gate_dev, ple_input_dev, seq_len * ple_dim)?;
                    if let Some(dump) = &ple_replay_dump {
                        let gated = self.debug_copy_device_f32(
                            ple_gate_dev,
                            seq_len * ple_dim,
                            "Gemma4 PLE replay gated",
                        )?;
                        dense_dump_gemma4_ple_replay_f32(dump, "ple_gated", &gated)?;
                    }
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-attn-dense-batch-chain",
                        "ple_gate_f32+gelu",
                        &mut trace_stage,
                    )?;
                    let proj_weights_dev = self.resident_f32_weights_ptr_from_le_bytes(
                        ple_proj_weights,
                        "dense attention+FFN PLE proj",
                    )?;
                    self.sgemm_device(
                        proj_weights_dev,
                        n_embd,
                        ple_dim,
                        ple_gate_dev,
                        seq_len,
                        proj_dev,
                    )?;
                    if let Some(dump) = &ple_replay_dump {
                        let projected = self.debug_copy_device_f32(
                            proj_dev,
                            seq_len * n_embd,
                            "Gemma4 PLE replay projected",
                        )?;
                        dense_dump_gemma4_ple_replay_f32(dump, "ple_projected", &projected)?;
                    }
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-attn-dense-batch-chain",
                        "ple_proj_f32",
                        &mut trace_stage,
                    )?;
                }
            }
            let post_norm_dev = self.resident_f32_ptr(ple_post_norm_weight)?;
            self.launch_rms_norm_add_rows_f32_inplace(
                proj_dev,
                post_norm_dev,
                hidden_dev,
                norm_eps,
                seq_len,
                n_embd,
                false,
            )?;
            self.trace_dense_stage(
                trace_call,
                "cuda-attn-dense-batch-chain",
                "ple_post_norm",
                &mut trace_stage,
            )?;
            if let Some(dump) = &ple_replay_dump {
                let final_hidden = self.debug_copy_device_f32(
                    hidden_dev,
                    seq_len * n_embd,
                    "Gemma4 PLE replay final",
                )?;
                dense_dump_gemma4_ple_replay_f32(dump, "ple_final", &final_hidden)?;
            }
        }

        if let Some(scale) = layer_out_scale {
            let scale_dev = self.resident_f32_ptr(&scale[..1])?;
            self.launch_scale_rows_inplace(hidden_dev, scale_dev, seq_len * n_embd, 1)?;
            self.trace_dense_stage(
                trace_call,
                "cuda-attn-dense-batch-chain",
                "out_scale",
                &mut trace_stage,
            )?;
            if let Some(dump) = &ple_replay_dump {
                let scaled = self.debug_copy_device_f32(
                    hidden_dev,
                    seq_len * n_embd,
                    "Gemma4 PLE replay final scaled",
                )?;
                dense_dump_gemma4_ple_replay_f32(dump, "ple_final_scaled", &scaled)?;
            }
        }

        if let Some(desc) = device_output_desc {
            let desc_bytes = desc.byte_len().ok_or_else(|| {
                "dense attention+FFN device output byte length overflow".to_string()
            })?;
            if desc_bytes != hidden_bytes {
                return Err(format!(
                    "dense attention+FFN device output bytes mismatch: desc={desc_bytes} hidden={hidden_bytes}"
                ));
            }
            let output_dev = unsafe { self.api.mem_alloc(hidden_bytes)? };
            let copy_result = unsafe {
                self.api
                    .memcpy_dtod_async(output_dev, hidden_dev, hidden_bytes, self.stream)
            }
            .and_then(|_| self.stream_synchronize());
            if let Err(err) = copy_result {
                let _ = unsafe { self.api.mem_free(output_dev) };
                return Err(err);
            }
            if trace_call.is_some() {
                self.trace_dense_stage(
                    trace_call,
                    "cuda-attn-dense-batch-chain",
                    "dtod_hidden",
                    &mut trace_stage,
                )?;
            }
            if let (Some(call), Some(total)) = (trace_call, trace_total) {
                eprintln!(
                    "[cuda-attn-dense-batch-chain] call={call} stage=total ms={:.3}",
                    total.elapsed().as_secs_f64() * 1000.0
                );
            }
            let output_id = self.insert_device_tensor_slot(output_dev, hidden_bytes, desc)?;
            return Ok(Some(output_id));
        }

        unsafe {
            self.api.memcpy_dtoh_async(
                hidden.as_mut_ptr().cast::<libc::c_void>(),
                hidden_dev,
                hidden_bytes,
                self.stream,
            )?;
        }
        if trace_call.is_some() {
            self.trace_dense_stage(
                trace_call,
                "cuda-attn-dense-batch-chain",
                "dtoh_hidden",
                &mut trace_stage,
            )?;
            if let (Some(call), Some(total)) = (trace_call, trace_total) {
                eprintln!(
                    "[cuda-attn-dense-batch-chain] call={call} stage=total ms={:.3}",
                    total.elapsed().as_secs_f64() * 1000.0
                );
            }
        } else {
            self.stream_synchronize()?;
        }
        Ok(None)
    }

    pub(super) fn gemma4_ple_q4k_batch_norm_residual(
        &mut self,
        gate_weights: &[u8],
        proj_weights: &[u8],
        post_norm_weight: &[f32],
        out_scale: Option<&[f32]>,
        ple_input: &[f32],
        ple_dim: usize,
        n_embd: usize,
        seq_len: usize,
        hidden: &mut [f32],
        norm_eps: f32,
    ) -> Result<(), String> {
        if seq_len <= 1 {
            return Err(format!(
                "Gemma4 PLE batch chain requires seq_len > 1, got {seq_len}"
            ));
        }
        if ple_dim == 0 || n_embd == 0 {
            return Err(format!(
                "Gemma4 PLE batch dims must be non-zero, got ple_dim={ple_dim} n_embd={n_embd}"
            ));
        }
        let expected_hidden = seq_len.checked_mul(n_embd).ok_or_else(|| {
            format!("Gemma4 PLE hidden length overflow: seq_len={seq_len} n_embd={n_embd}")
        })?;
        let expected_ple = seq_len.checked_mul(ple_dim).ok_or_else(|| {
            format!("Gemma4 PLE input length overflow: seq_len={seq_len} ple_dim={ple_dim}")
        })?;
        if hidden.len() != expected_hidden {
            return Err(format!(
                "Gemma4 PLE hidden length mismatch: got {}, expected {expected_hidden}",
                hidden.len()
            ));
        }
        if ple_input.len() != expected_ple {
            return Err(format!(
                "Gemma4 PLE input length mismatch: got {}, expected {expected_ple}",
                ple_input.len()
            ));
        }
        if post_norm_weight.len() != n_embd {
            return Err(format!(
                "Gemma4 PLE post norm length mismatch: got {}, expected {n_embd}",
                post_norm_weight.len()
            ));
        }
        if let Some(scale) = out_scale {
            if scale.is_empty() {
                return Err("Gemma4 PLE out_scale must be non-empty".to_string());
            }
        }
        let q4k_supported = ple_dim.is_multiple_of(256) && n_embd.is_multiple_of(256);
        let hidden_blocks = n_embd / 256;
        let ple_blocks = ple_dim / 256;
        let q4k_gate_expected = q4k_supported.then(|| {
            ple_dim
                .checked_mul(hidden_blocks)
                .and_then(|n| n.checked_mul(144))
        });
        let q4k_proj_expected = q4k_supported.then(|| {
            n_embd
                .checked_mul(ple_blocks)
                .and_then(|n| n.checked_mul(144))
        });
        let f32_gate_expected = ple_dim
            .checked_mul(n_embd)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Gemma4 PLE F32 gate byte overflow".to_string())?;
        let f32_proj_expected = n_embd
            .checked_mul(ple_dim)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Gemma4 PLE F32 proj byte overflow".to_string())?;
        let ple_weight_kind = if q4k_gate_expected.flatten() == Some(gate_weights.len())
            && q4k_proj_expected.flatten() == Some(proj_weights.len())
        {
            DensePleWeightKind::Q4K
        } else if gate_weights.len() == f32_gate_expected && proj_weights.len() == f32_proj_expected
        {
            DensePleWeightKind::F32
        } else {
            let q4k_gate_label = q4k_gate_expected
                .flatten()
                .map_or_else(|| "unsupported".to_string(), |bytes| bytes.to_string());
            let q4k_proj_label = q4k_proj_expected
                .flatten()
                .map_or_else(|| "unsupported".to_string(), |bytes| bytes.to_string());
            return Err(format!(
                "Gemma4 PLE weight byte mismatch: gate got {} expected q4k={} f32={}; proj got {} expected q4k={} f32={}",
                gate_weights.len(),
                q4k_gate_label,
                f32_gate_expected,
                proj_weights.len(),
                q4k_proj_label,
                f32_proj_expected
            ));
        };
        let q4k_gate_expected = ple_dim
            .checked_mul(hidden_blocks)
            .and_then(|n| n.checked_mul(144))
            .ok_or_else(|| "Gemma4 PLE gate byte overflow".to_string())?;
        let q4k_proj_expected = n_embd
            .checked_mul(ple_blocks)
            .and_then(|n| n.checked_mul(144))
            .ok_or_else(|| "Gemma4 PLE proj byte overflow".to_string())?;

        let hidden_bytes = std::mem::size_of_val(hidden);
        let ple_bytes = std::mem::size_of_val(ple_input);
        let hidden_dev = self.compute_full_gate_ptr(hidden_bytes)?;
        let ple_input_dev = self.compute_full_down_ptr(ple_bytes)?;
        let gate_dev = self.compute_mid_a_ptr(ple_bytes)?;
        let proj_dev = self.compute_output_ptr(hidden_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                hidden_dev,
                hidden.as_ptr().cast::<libc::c_void>(),
                hidden_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                ple_input_dev,
                ple_input.as_ptr().cast::<libc::c_void>(),
                ple_bytes,
                self.stream,
            )?;
        }

        match ple_weight_kind {
            DensePleWeightKind::Q4K => {
                if gate_weights.len() != q4k_gate_expected {
                    return Err(format!(
                        "Gemma4 PLE gate byte mismatch: got {}, expected {q4k_gate_expected}",
                        gate_weights.len()
                    ));
                }
                if proj_weights.len() != q4k_proj_expected {
                    return Err(format!(
                        "Gemma4 PLE proj byte mismatch: got {}, expected {q4k_proj_expected}",
                        proj_weights.len()
                    ));
                }
                self.q4k_batch_dev_input_to_dev(
                    gate_weights,
                    ple_dim,
                    hidden_blocks,
                    seq_len,
                    hidden_dev,
                    gate_dev,
                )?;
                self.launch_gelu_mul(gate_dev, ple_input_dev, expected_ple)?;
                self.q4k_batch_dev_input_to_dev(
                    proj_weights,
                    n_embd,
                    ple_blocks,
                    seq_len,
                    gate_dev,
                    proj_dev,
                )?;
            }
            DensePleWeightKind::F32 => {
                let gate_weights_dev =
                    self.resident_f32_weights_ptr_from_le_bytes(gate_weights, "Gemma4 PLE gate")?;
                self.sgemm_device(
                    gate_weights_dev,
                    ple_dim,
                    n_embd,
                    hidden_dev,
                    seq_len,
                    gate_dev,
                )?;
                self.launch_gelu_mul(gate_dev, ple_input_dev, expected_ple)?;
                let proj_weights_dev =
                    self.resident_f32_weights_ptr_from_le_bytes(proj_weights, "Gemma4 PLE proj")?;
                self.sgemm_device(
                    proj_weights_dev,
                    n_embd,
                    ple_dim,
                    gate_dev,
                    seq_len,
                    proj_dev,
                )?;
            }
        }
        let post_norm_dev = self.resident_f32_ptr(post_norm_weight)?;
        self.launch_rms_norm_add_rows_f32_inplace(
            proj_dev,
            post_norm_dev,
            hidden_dev,
            norm_eps,
            seq_len,
            n_embd,
            false,
        )?;
        if let Some(scale) = out_scale {
            let scale_dev = self.resident_f32_ptr(&scale[..1])?;
            self.launch_scale_rows_inplace(hidden_dev, scale_dev, seq_len * n_embd, 1)?;
        }

        unsafe {
            self.api.memcpy_dtoh_async(
                hidden.as_mut_ptr().cast::<libc::c_void>(),
                hidden_dev,
                hidden_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn gemma4_ple_f32_debug_stages(
        &mut self,
        gate_weights: &[u8],
        proj_weights: &[u8],
        post_norm_weight: &[f32],
        out_scale: Option<&[f32]>,
        ple_input: &[f32],
        ple_dim: usize,
        n_embd: usize,
        seq_len: usize,
        hidden: &[f32],
        norm_eps: f32,
    ) -> Result<Gemma4PleF32DebugStages, String> {
        if seq_len <= 1 {
            return Err(format!(
                "Gemma4 PLE F32 debug requires seq_len > 1, got {seq_len}"
            ));
        }
        if ple_dim == 0 || n_embd == 0 {
            return Err(format!(
                "Gemma4 PLE F32 debug dims must be non-zero, got ple_dim={ple_dim} n_embd={n_embd}"
            ));
        }
        let expected_hidden = seq_len.checked_mul(n_embd).ok_or_else(|| {
            format!(
                "Gemma4 PLE F32 debug hidden length overflow: seq_len={seq_len} n_embd={n_embd}"
            )
        })?;
        let expected_ple = seq_len.checked_mul(ple_dim).ok_or_else(|| {
            format!(
                "Gemma4 PLE F32 debug input length overflow: seq_len={seq_len} ple_dim={ple_dim}"
            )
        })?;
        if hidden.len() != expected_hidden {
            return Err(format!(
                "Gemma4 PLE F32 debug hidden length mismatch: got {}, expected {expected_hidden}",
                hidden.len()
            ));
        }
        if ple_input.len() != expected_ple {
            return Err(format!(
                "Gemma4 PLE F32 debug input length mismatch: got {}, expected {expected_ple}",
                ple_input.len()
            ));
        }
        if post_norm_weight.len() != n_embd {
            return Err(format!(
                "Gemma4 PLE F32 debug post norm length mismatch: got {}, expected {n_embd}",
                post_norm_weight.len()
            ));
        }
        if let Some(scale) = out_scale {
            if scale.is_empty() {
                return Err("Gemma4 PLE F32 debug out_scale must be non-empty".to_string());
            }
        }
        let f32_gate_expected = ple_dim
            .checked_mul(n_embd)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Gemma4 PLE F32 debug gate byte overflow".to_string())?;
        let f32_proj_expected = n_embd
            .checked_mul(ple_dim)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Gemma4 PLE F32 debug proj byte overflow".to_string())?;
        if gate_weights.len() != f32_gate_expected {
            return Err(format!(
                "Gemma4 PLE F32 debug gate byte mismatch: got {}, expected {f32_gate_expected}",
                gate_weights.len()
            ));
        }
        if proj_weights.len() != f32_proj_expected {
            return Err(format!(
                "Gemma4 PLE F32 debug proj byte mismatch: got {}, expected {f32_proj_expected}",
                proj_weights.len()
            ));
        }

        let hidden_bytes = std::mem::size_of_val(hidden);
        let ple_bytes = std::mem::size_of_val(ple_input);
        let hidden_dev = self.compute_full_gate_ptr(hidden_bytes)?;
        let ple_input_dev = self.compute_full_down_ptr(ple_bytes)?;
        let gate_dev = self.compute_mid_a_ptr(ple_bytes)?;
        let proj_dev = self.compute_output_ptr(hidden_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                hidden_dev,
                hidden.as_ptr().cast::<libc::c_void>(),
                hidden_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                ple_input_dev,
                ple_input.as_ptr().cast::<libc::c_void>(),
                ple_bytes,
                self.stream,
            )?;
        }

        let gate_weights_dev =
            self.resident_f32_weights_ptr_from_le_bytes(gate_weights, "Gemma4 PLE debug gate")?;
        self.sgemm_device(
            gate_weights_dev,
            ple_dim,
            n_embd,
            hidden_dev,
            seq_len,
            gate_dev,
        )?;
        let mut gate = vec![0.0f32; expected_ple];
        unsafe {
            self.api.memcpy_dtoh_async(
                gate.as_mut_ptr().cast::<libc::c_void>(),
                gate_dev,
                ple_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        self.launch_gelu_mul(gate_dev, ple_input_dev, expected_ple)?;
        let mut gated = vec![0.0f32; expected_ple];
        unsafe {
            self.api.memcpy_dtoh_async(
                gated.as_mut_ptr().cast::<libc::c_void>(),
                gate_dev,
                ple_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        let proj_weights_dev =
            self.resident_f32_weights_ptr_from_le_bytes(proj_weights, "Gemma4 PLE debug proj")?;
        self.sgemm_device(
            proj_weights_dev,
            n_embd,
            ple_dim,
            gate_dev,
            seq_len,
            proj_dev,
        )?;
        let mut projected = vec![0.0f32; expected_hidden];
        unsafe {
            self.api.memcpy_dtoh_async(
                projected.as_mut_ptr().cast::<libc::c_void>(),
                proj_dev,
                hidden_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        let post_norm_dev = self.resident_f32_ptr(post_norm_weight)?;
        self.launch_rms_norm_add_rows_f32_inplace(
            proj_dev,
            post_norm_dev,
            hidden_dev,
            norm_eps,
            seq_len,
            n_embd,
            false,
        )?;
        if let Some(scale) = out_scale {
            let scale_dev = self.resident_f32_ptr(&scale[..1])?;
            self.launch_scale_rows_inplace(hidden_dev, scale_dev, seq_len * n_embd, 1)?;
        }
        let mut final_hidden = vec![0.0f32; expected_hidden];
        unsafe {
            self.api.memcpy_dtoh_async(
                final_hidden.as_mut_ptr().cast::<libc::c_void>(),
                hidden_dev,
                hidden_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        Ok(Gemma4PleF32DebugStages {
            gate,
            gated,
            projected,
            final_hidden,
        })
    }

    fn trace_dense_stage(
        &mut self,
        call: Option<usize>,
        prefix: &str,
        stage: &str,
        start: &mut std::time::Instant,
    ) -> Result<(), String> {
        if let Some(call) = call {
            self.stream_synchronize()?;
            eprintln!(
                "[{prefix}] call={call} stage={stage} ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            *start = std::time::Instant::now();
        }
        Ok(())
    }

    pub(super) fn q4k_dev_input_to_dev(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        if dense_q4k_gemv_q8dot_enabled(rows >= 1024 && blocks_per_row >= 4) {
            let qs_dev = self.compute_input_ptr(blocks_per_row * 256)?;
            let ds_dev =
                self.compute_aux_output_ptr(blocks_per_row * 8 * std::mem::size_of::<f32>())?;
            self.launch_quantize_q8_1_by_32(input_dev, qs_dev, ds_dev, blocks_per_row * 256)?;
            self.launch_q4k_gemv_q8dot_to_dev(
                weights,
                rows,
                blocks_per_row,
                qs_dev,
                ds_dev,
                output_dev,
            )
        } else {
            self.launch_q4k_gemv_to_dev(weights, rows, blocks_per_row, input_dev, output_dev)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_dense_chain_graph_ops(
        &mut self,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        hidden_dev: u64,
        attn_out_dev: u64,
        normed_dev: u64,
        proj_dev: u64,
        ffn_uses_gelu: bool,
        ple_weight_kind: Option<DensePleWeightKind>,
        ple_gate_weights: Option<&[u8]>,
        ple_proj_weights: Option<&[u8]>,
        ple_post_norm_weight: Option<&[f32]>,
        ple_input_dev: Option<u64>,
        ple_gate_dev: Option<u64>,
        ple_gate_weight_dev: u64,
        ple_proj_weight_dev: u64,
        ple_dim: usize,
        unit_offset_ple_norm: bool,
        layer_output_scale: Option<f32>,
        trace_call: Option<usize>,
        trace_stage: &mut std::time::Instant,
    ) -> Result<(), String> {
        self.q4k_dev_input_to_dev(o_weights, n_embd, o_cols / 256, attn_out_dev, proj_dev)?;
        self.trace_dense_stage(trace_call, "cuda-dense-chain", "o_proj", trace_stage)?;
        let ffn_norm_dev = self.resident_f32_ptr(ffn_norm_weight)?;
        let mut prequantized_q8 = None;
        if let Some(weight) = post_attn_norm_weight {
            let norm_weight_dev = self.resident_f32_ptr(weight)?;
            if dense_combined_norms_enabled(true) {
                if dense_q8dot_gate_up_enabled(ffn_uses_gelu) {
                    let q8_qs_dev = self.compute_gate_ptrs_ptr(n_embd)?;
                    let q8_ds_dev =
                        self.compute_up_ptrs_ptr((n_embd / 32) * std::mem::size_of::<f32>())?;
                    self.launch_rms_norm_add_then_rms_norm_q8_1_f32(
                        proj_dev,
                        norm_weight_dev,
                        hidden_dev,
                        ffn_norm_dev,
                        normed_dev,
                        q8_qs_dev,
                        q8_ds_dev,
                        norm_eps,
                        n_embd,
                        unit_offset_post_attn_norm,
                        unit_offset_ffn_norm,
                    )?;
                    prequantized_q8 = Some((q8_qs_dev, q8_ds_dev));
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "post_attn_resid_norm+ffn_pre_norm+input_q8_quant",
                        trace_stage,
                    )?;
                } else {
                    self.launch_rms_norm_add_then_rms_norm_f32(
                        proj_dev,
                        norm_weight_dev,
                        hidden_dev,
                        ffn_norm_dev,
                        normed_dev,
                        norm_eps,
                        n_embd,
                        unit_offset_post_attn_norm,
                        unit_offset_ffn_norm,
                    )?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "post_attn_resid_norm+ffn_pre_norm",
                        trace_stage,
                    )?;
                }
            } else {
                self.launch_rms_norm_add_f32_inplace(
                    proj_dev,
                    norm_weight_dev,
                    hidden_dev,
                    norm_eps,
                    n_embd,
                    unit_offset_post_attn_norm,
                )?;
                self.trace_dense_stage(
                    trace_call,
                    "cuda-dense-chain",
                    "post_attn_resid_norm",
                    trace_stage,
                )?;
                self.launch_rms_norm_f32(
                    hidden_dev,
                    ffn_norm_dev,
                    normed_dev,
                    norm_eps,
                    n_embd,
                    unit_offset_ffn_norm,
                )?;
                self.trace_dense_stage(
                    trace_call,
                    "cuda-dense-chain",
                    "ffn_pre_norm",
                    trace_stage,
                )?;
            }
        } else {
            self.launch_add_f32_inplace(hidden_dev, proj_dev, n_embd)?;
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-chain",
                "post_attn_resid_norm",
                trace_stage,
            )?;
            self.launch_rms_norm_f32(
                hidden_dev,
                ffn_norm_dev,
                normed_dev,
                norm_eps,
                n_embd,
                unit_offset_ffn_norm,
            )?;
            self.trace_dense_stage(trace_call, "cuda-dense-chain", "ffn_pre_norm", trace_stage)?;
        }
        self.qwen35_expert_dev_input_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            normed_dev,
            proj_dev,
            prequantized_q8,
            ffn_uses_gelu,
        )?;
        self.trace_dense_stage(trace_call, "cuda-dense-chain", "ffn", trace_stage)?;
        if let Some(weight) = post_ffn_norm_weight {
            let norm_weight_dev = self.resident_f32_ptr(weight)?;
            self.launch_rms_norm_add_f32_inplace(
                proj_dev,
                norm_weight_dev,
                hidden_dev,
                norm_eps,
                n_embd,
                unit_offset_ffn_norm,
            )?;
        } else {
            self.launch_add_f32_inplace(hidden_dev, proj_dev, n_embd)?;
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-chain",
            "post_ffn_resid_norm",
            trace_stage,
        )?;
        if let (
            Some(ple_weight_kind),
            Some(ple_post_norm_weight),
            Some(ple_input_dev),
            Some(ple_gate_dev),
        ) = (
            ple_weight_kind,
            ple_post_norm_weight,
            ple_input_dev,
            ple_gate_dev,
        ) {
            match ple_weight_kind {
                DensePleWeightKind::Q4K => {
                    let ple_gate_weights = ple_gate_weights
                        .ok_or_else(|| "dense chain graph Q4K PLE gate missing".to_string())?;
                    let ple_proj_weights = ple_proj_weights
                        .ok_or_else(|| "dense chain graph Q4K PLE proj missing".to_string())?;
                    if dense_ple_gate_gelu_enabled(true) {
                        self.launch_q4k_gemv_gelu_mul_to_dev(
                            ple_gate_weights,
                            ple_dim,
                            n_embd / 256,
                            hidden_dev,
                            ple_input_dev,
                            ple_gate_dev,
                        )?;
                        self.trace_dense_stage(
                            trace_call,
                            "cuda-dense-chain",
                            "ple_gate+gelu",
                            trace_stage,
                        )?;
                    } else {
                        self.q4k_dev_input_to_dev(
                            ple_gate_weights,
                            ple_dim,
                            n_embd / 256,
                            hidden_dev,
                            ple_gate_dev,
                        )?;
                        self.trace_dense_stage(
                            trace_call,
                            "cuda-dense-chain",
                            "ple_gate",
                            trace_stage,
                        )?;
                        self.launch_gelu_mul(ple_gate_dev, ple_input_dev, ple_dim)?;
                        self.trace_dense_stage(
                            trace_call,
                            "cuda-dense-chain",
                            "ple_gelu",
                            trace_stage,
                        )?;
                    }
                    self.q4k_dev_input_to_dev(
                        ple_proj_weights,
                        n_embd,
                        ple_dim / 256,
                        ple_gate_dev,
                        proj_dev,
                    )?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "ple_proj",
                        trace_stage,
                    )?;
                }
                DensePleWeightKind::F32 => {
                    if ple_gate_weight_dev == 0 || ple_proj_weight_dev == 0 {
                        return Err(
                            "dense chain graph F32 PLE weight device ptr missing".to_string()
                        );
                    }
                    self.sgemm_device(
                        ple_gate_weight_dev,
                        ple_dim,
                        n_embd,
                        hidden_dev,
                        1,
                        ple_gate_dev,
                    )?;
                    self.launch_gelu_mul(ple_gate_dev, ple_input_dev, ple_dim)?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "ple_gate_f32+gelu",
                        trace_stage,
                    )?;
                    self.sgemm_device(
                        ple_proj_weight_dev,
                        n_embd,
                        ple_dim,
                        ple_gate_dev,
                        1,
                        proj_dev,
                    )?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "ple_proj_f32",
                        trace_stage,
                    )?;
                }
            }
            let norm_weight_dev = self.resident_f32_ptr(ple_post_norm_weight)?;
            self.launch_rms_norm_add_f32_inplace(
                proj_dev,
                norm_weight_dev,
                hidden_dev,
                norm_eps,
                n_embd,
                unit_offset_ple_norm,
            )?;
            self.trace_dense_stage(trace_call, "cuda-dense-chain", "ple_post_norm", trace_stage)?;
        }
        if let Some(scale) = layer_output_scale {
            self.launch_scale_f32_inplace(hidden_dev, scale, n_embd)?;
            self.trace_dense_stage(trace_call, "cuda-dense-chain", "out_scale", trace_stage)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_expert_dev_input_to_dev(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
        prequantized_q8: Option<(u64, u64)>,
        gelu: bool,
    ) -> Result<(), String> {
        let trace_call = dense_expert_trace_call();
        let trace_total = trace_call.map(|_| std::time::Instant::now());
        let mut trace_stage = std::time::Instant::now();
        let q8dot_gate_up = dense_q8dot_gate_up_enabled(gelu);
        let packed_q4_gate_up = if q8dot_gate_up && dense_q4_packed_q8dot_enabled(true) {
            let gate = self.resident_q4k_packed_ptrs(gate_weights, n_ff, n_embd / 256)?;
            let up = self.resident_q4k_packed_ptrs(up_weights, n_ff, n_embd / 256)?;
            match (gate, up) {
                (Some(gate), Some(up)) => Some((gate, up)),
                _ => None,
            }
        } else {
            None
        };
        let packed_q4_down = if gelu && down_quant == 12 && dense_q4_packed_q8dot_enabled(true) {
            self.resident_q4k_packed_ptrs(down_weights, n_embd, n_ff / 256)?
        } else {
            None
        };
        let packed_q6_down = if gelu && down_quant == 14 && dense_q6_packed_q8dot_enabled(true) {
            self.resident_q6k_packed_ptrs(down_weights, n_embd, n_ff / 256)?
        } else {
            None
        };
        let q8dot_down =
            dense_q8dot_down_enabled(gelu && (down_quant == 12 || packed_q6_down.is_some()));
        let gate_dev = self.compute_mid_a_ptr(n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(n_ff * std::mem::size_of::<f32>())?;
        let prequantized_q8 = if q8dot_gate_up { prequantized_q8 } else { None };
        let q8_input_prequantized = prequantized_q8.is_some();
        let mut q8_qs_dev = None;
        let mut q8_ds_dev = None;
        if q8dot_gate_up {
            let (qs_dev, ds_dev) = if let Some((qs_dev, ds_dev)) = prequantized_q8 {
                (qs_dev, ds_dev)
            } else {
                let qs_dev = self.compute_gate_ptrs_ptr(n_embd)?;
                let ds_dev =
                    self.compute_up_ptrs_ptr((n_embd / 32) * std::mem::size_of::<f32>())?;
                (qs_dev, ds_dev)
            };
            q8_qs_dev = Some(qs_dev);
            q8_ds_dev = Some(ds_dev);
        }
        let mut down_q8_qs_dev = None;
        let mut down_q8_ds_dev = None;
        if q8dot_down {
            let qs_dev = self.compute_input_ptr(n_ff)?;
            let ds_dev = self.compute_aux_output_ptr((n_ff / 32) * std::mem::size_of::<f32>())?;
            down_q8_qs_dev = Some(qs_dev);
            down_q8_ds_dev = Some(ds_dev);
        }
        let graph_enabled = tuning::dense_expert_graph_enabled() && trace_call.is_none();
        if graph_enabled {
            let (packed_q6_down_qs, packed_q6_down_d_super, packed_q6_down_sub_scale) =
                packed_q6_down.unwrap_or((0, 0, 0));
            let key = DenseExpertGraphKey {
                down_quant,
                n_ff,
                n_embd,
                gelu,
                q8dot_gate_up,
                q8dot_down,
                q8_input_prequantized,
                input_dev,
                output_dev,
                gate_dev,
                up_dev,
                gate_weight: gate_weights.as_ptr() as usize,
                up_weight: up_weights.as_ptr() as usize,
                down_weight: down_weights.as_ptr() as usize,
                packed_gate: packed_q4_gate_up.map(|(gate, _)| gate).unwrap_or(0),
                packed_up: packed_q4_gate_up.map(|(_, up)| up).unwrap_or(0),
                packed_q4_down: packed_q4_down.unwrap_or(0),
                packed_q6_down_qs,
                packed_q6_down_d_super,
                packed_q6_down_sub_scale,
                q8_qs_dev: q8_qs_dev.unwrap_or(0),
                q8_ds_dev: q8_ds_dev.unwrap_or(0),
                down_q8_qs_dev: down_q8_qs_dev.unwrap_or(0),
                down_q8_ds_dev: down_q8_ds_dev.unwrap_or(0),
            };
            if let Some(graph) = self.dense_expert_graphs.get(&key) {
                return unsafe {
                    self.api
                        .graph_launch(graph.exec as *mut libc::c_void, self.stream)
                };
            }
            if self.dense_expert_graph_warmed.contains(&key) {
                self.ensure_q4k_gemv_module()?;
                unsafe {
                    self.api.stream_begin_capture(self.stream)?;
                }
                let capture_result = self.launch_dense_expert_ops(
                    gate_weights,
                    up_weights,
                    down_weights,
                    down_quant,
                    n_ff,
                    n_embd,
                    input_dev,
                    output_dev,
                    gate_dev,
                    up_dev,
                    gelu,
                    q8dot_gate_up,
                    q8dot_down,
                    q8_input_prequantized,
                    packed_q4_gate_up,
                    packed_q4_down,
                    packed_q6_down,
                    q8_qs_dev,
                    q8_ds_dev,
                    down_q8_qs_dev,
                    down_q8_ds_dev,
                    None,
                    &mut trace_stage,
                );
                if let Err(err) = capture_result {
                    unsafe {
                        let _ = self.api.stream_end_capture(self.stream);
                    }
                    return Err(err);
                }
                let graph = unsafe { self.api.stream_end_capture(self.stream)? };
                let exec = unsafe { self.api.graph_instantiate(graph)? };
                self.dense_expert_graphs.insert(
                    key,
                    SparseMoeGraph {
                        graph: graph as usize,
                        exec: exec as usize,
                    },
                );
                let graph = self
                    .dense_expert_graphs
                    .get(&key)
                    .ok_or_else(|| "missing dense expert CUDA graph".to_string())?;
                return unsafe {
                    self.api
                        .graph_launch(graph.exec as *mut libc::c_void, self.stream)
                };
            }
            self.dense_expert_graph_warmed.insert(key);
        }
        self.launch_dense_expert_ops(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
            gate_dev,
            up_dev,
            gelu,
            q8dot_gate_up,
            q8dot_down,
            q8_input_prequantized,
            packed_q4_gate_up,
            packed_q4_down,
            packed_q6_down,
            q8_qs_dev,
            q8_ds_dev,
            down_q8_qs_dev,
            down_q8_ds_dev,
            trace_call,
            &mut trace_stage,
        )?;
        if let (Some(call), Some(total)) = (trace_call, trace_total) {
            eprintln!(
                "[cuda-dense-expert] call={call} stage=total ms={:.3}",
                total.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_dense_expert_ops(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
        gate_dev: u64,
        up_dev: u64,
        gelu: bool,
        q8dot_gate_up: bool,
        q8dot_down: bool,
        q8_input_prequantized: bool,
        packed_q4_gate_up: Option<(u64, u64)>,
        packed_q4_down: Option<u64>,
        packed_q6_down: Option<(u64, u64, u64)>,
        q8_qs_dev: Option<u64>,
        q8_ds_dev: Option<u64>,
        down_q8_qs_dev: Option<u64>,
        down_q8_ds_dev: Option<u64>,
        trace_call: Option<usize>,
        trace_stage: &mut std::time::Instant,
    ) -> Result<(), String> {
        if q8dot_gate_up && !q8_input_prequantized {
            let qs_dev = q8_qs_dev
                .ok_or_else(|| "missing dense expert gate/up Q8 qs device buffer".to_string())?;
            let ds_dev = q8_ds_dev
                .ok_or_else(|| "missing dense expert gate/up Q8 scale device buffer".to_string())?;
            self.launch_quantize_q8_1_by_32(input_dev, qs_dev, ds_dev, n_embd)?;
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-expert",
            "input_q8_quant",
            trace_stage,
        )?;

        if let (Some((gate_packed, up_packed)), Some(qs_dev), Some(ds_dev)) =
            (packed_q4_gate_up, q8_qs_dev, q8_ds_dev)
        {
            self.launch_q4k_packed_gate_up_gemv_q8dot_to_dev(
                gate_packed,
                up_packed,
                n_ff,
                n_embd / 256,
                qs_dev,
                ds_dev,
                gate_dev,
                up_dev,
            )?;
        } else if let (Some(qs_dev), Some(ds_dev)) = (q8_qs_dev, q8_ds_dev) {
            self.launch_q4k_gate_up_gemv_q8dot_to_dev(
                gate_weights,
                up_weights,
                n_ff,
                n_embd / 256,
                qs_dev,
                ds_dev,
                gate_dev,
                up_dev,
            )?;
        } else {
            self.launch_q4k_gate_up_gemv_to_dev(
                gate_weights,
                up_weights,
                n_ff,
                n_embd / 256,
                input_dev,
                gate_dev,
                up_dev,
            )?;
        }
        self.trace_dense_stage(trace_call, "cuda-dense-expert", "gate_up", trace_stage)?;

        if q8dot_down {
            let qs_dev = down_q8_qs_dev
                .ok_or_else(|| "missing dense expert down Q8 qs device buffer".to_string())?;
            let ds_dev = down_q8_ds_dev
                .ok_or_else(|| "missing dense expert down Q8 scale device buffer".to_string())?;
            if gelu {
                self.launch_gelu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, n_ff)?;
            } else {
                self.launch_silu_mul(gate_dev, up_dev, n_ff)?;
            }
        } else if gelu {
            self.launch_gelu_mul(gate_dev, up_dev, n_ff)?;
        } else {
            self.launch_silu_mul(gate_dev, up_dev, n_ff)?;
        }
        self.trace_dense_stage(trace_call, "cuda-dense-expert", "activation", trace_stage)?;

        if gelu {
            match down_quant {
                12 => {
                    if let (Some(qs_dev), Some(ds_dev)) = (down_q8_qs_dev, down_q8_ds_dev) {
                        if let Some(packed_dev) = packed_q4_down {
                            self.launch_q4k_packed_gemv_q8dot_to_dev(
                                packed_dev,
                                n_embd,
                                n_ff / 256,
                                qs_dev,
                                ds_dev,
                                output_dev,
                            )
                        } else {
                            self.launch_q4k_gemv_q8dot_to_dev(
                                down_weights,
                                n_embd,
                                n_ff / 256,
                                qs_dev,
                                ds_dev,
                                output_dev,
                            )
                        }
                    } else {
                        self.launch_q4k_gemv_to_dev(
                            down_weights,
                            n_embd,
                            n_ff / 256,
                            gate_dev,
                            output_dev,
                        )
                    }
                }
                13 => self.launch_q5k_gemv_to_dev(
                    down_weights,
                    n_embd,
                    n_ff / 256,
                    gate_dev,
                    output_dev,
                ),
                14 => {
                    if let (
                        Some((packed_qs_dev, packed_d_super_dev, packed_sub_scale_dev)),
                        Some(qs_dev),
                        Some(ds_dev),
                    ) = (packed_q6_down, down_q8_qs_dev, down_q8_ds_dev)
                    {
                        self.launch_q6k_packed_q8dot_to_dev(
                            packed_qs_dev,
                            packed_d_super_dev,
                            packed_sub_scale_dev,
                            n_embd,
                            n_ff / 256,
                            qs_dev,
                            ds_dev,
                            output_dev,
                        )
                    } else if let (Some(qs_dev), Some(ds_dev)) = (down_q8_qs_dev, down_q8_ds_dev) {
                        self.launch_q6k_gemv_q8dot_to_dev(
                            down_weights,
                            n_embd,
                            n_ff / 256,
                            qs_dev,
                            ds_dev,
                            output_dev,
                        )
                    } else {
                        self.launch_q6k_gemv_to_dev(
                            down_weights,
                            n_embd,
                            n_ff / 256,
                            gate_dev,
                            output_dev,
                        )
                    }
                }
                other => Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
            }
        } else {
            match down_quant {
                12 => self.launch_q4k_gemv_to_dev(
                    down_weights,
                    n_embd,
                    n_ff / 256,
                    gate_dev,
                    output_dev,
                ),
                13 => self.launch_q5k_gemv_to_dev(
                    down_weights,
                    n_embd,
                    n_ff / 256,
                    gate_dev,
                    output_dev,
                ),
                14 => self.launch_q6k_gemv_to_dev(
                    down_weights,
                    n_embd,
                    n_ff / 256,
                    gate_dev,
                    output_dev,
                ),
                other => Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
            }
        }?;
        self.trace_dense_stage(trace_call, "cuda-dense-expert", "down", trace_stage)
    }

    pub(super) fn bf16_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if weights.len() != rows * cols * std::mem::size_of::<u16>() {
            return Err(format!(
                "BF16 weight byte mismatch: got {}, expected {}",
                weights.len(),
                rows * cols * std::mem::size_of::<u16>()
            ));
        }
        if input.len() != cols {
            return Err(format!(
                "BF16 input length mismatch: got {}, expected {cols}",
                input.len()
            ));
        }
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.launch_bf16_gemv_to_dev(weights, rows, cols, input_dev, output_dev)?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn f16_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if weights.len() != rows * cols * std::mem::size_of::<u16>() {
            return Err(format!(
                "F16 weight byte mismatch: got {}, expected {}",
                weights.len(),
                rows * cols * std::mem::size_of::<u16>()
            ));
        }
        if input.len() != cols {
            return Err(format!(
                "F16 input length mismatch: got {}, expected {cols}",
                input.len()
            ));
        }
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.launch_f16_gemv_to_dev(weights, rows, cols, input_dev, output_dev)?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn f32_shared_expert(
        &mut self,
        gate_weights: &[f32],
        up_weights: &[f32],
        down_weights: &[f32],
        route: &[f32],
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if n_embd == 0 || n_ff == 0 {
            return Err("f32 shared expert dimensions must be non-zero".to_string());
        }
        let seq_len = input
            .len()
            .checked_div(n_embd)
            .ok_or_else(|| "f32 shared expert input cols must be non-zero".to_string())?;
        if input.len() != seq_len * n_embd {
            return Err(format!(
                "f32 shared expert input len mismatch: got {}, expected multiple of {n_embd}",
                input.len()
            ));
        }
        if route.len() != seq_len {
            return Err(format!(
                "f32 shared expert route len mismatch: got {}, expected {seq_len}",
                route.len()
            ));
        }
        if gate_weights.len() != n_ff * n_embd
            || up_weights.len() != n_ff * n_embd
            || down_weights.len() != n_embd * n_ff
        {
            return Err("f32 shared expert weight shape mismatch".to_string());
        }

        let gate_w_dev = self.compute_full_gate_ptr(std::mem::size_of_val(gate_weights))?;
        let up_w_dev = self.compute_full_up_ptr(std::mem::size_of_val(up_weights))?;
        let down_w_dev = self.compute_full_down_ptr(std::mem::size_of_val(down_weights))?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route))?;
        let gate_dev = self.compute_mid_a_ptr(seq_len * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(seq_len * n_ff * std::mem::size_of::<f32>())?;
        let output_len = seq_len * n_embd;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                gate_w_dev,
                gate_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(gate_weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                up_w_dev,
                up_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(up_weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                down_w_dev,
                down_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(down_weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                route_dev,
                route.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(route),
                self.stream,
            )?;
        }

        self.sgemm_device(gate_w_dev, n_ff, n_embd, input_dev, seq_len, gate_dev)?;
        self.sgemm_device(up_w_dev, n_ff, n_embd, input_dev, seq_len, up_dev)?;
        self.launch_silu_mul(gate_dev, up_dev, seq_len * n_ff)?;
        self.sgemm_device(down_w_dev, n_embd, n_ff, gate_dev, seq_len, output_dev)?;
        self.launch_scale_rows_inplace(output_dev, route_dev, n_embd, seq_len)?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_prefill_moe_f32_shared_sparse_by_token(
        &mut self,
        shared_gate: &[f32],
        shared_up: &[f32],
        shared_down: &[f32],
        shared_route: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.qwen35_prefill_moe_f32_shared_sparse_by_token_prepared(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_prefill_moe_f32_shared_sparse_by_token_prepared(
        &mut self,
        shared_gate: &[f32],
        shared_up: &[f32],
        shared_down: &[f32],
        shared_route: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        prepared_sparse_override: Option<PreparedQwen35SparseSlots>,
    ) -> Result<Vec<f32>, String> {
        if n_embd == 0 || n_ff == 0 {
            return Err("Qwen35 combined MoE dimensions must be non-zero".to_string());
        }
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let shared_gate_dev = self.compute_full_gate_ptr(std::mem::size_of_val(shared_gate))?;
        let shared_up_dev = self.compute_full_up_ptr(std::mem::size_of_val(shared_up))?;
        let shared_down_dev = self.compute_full_down_ptr(std::mem::size_of_val(shared_down))?;
        let shared_route_dev = self.compute_route_ptr(std::mem::size_of_val(shared_route))?;
        let shared_gate_out_dev =
            self.compute_mid_a_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let shared_up_out_dev =
            self.compute_mid_b_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let output_len = token_count * n_embd;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let sparse_output_dev = self.compute_aux_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                shared_gate_dev,
                shared_gate.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_gate),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_up_dev,
                shared_up.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_up),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_down_dev,
                shared_down.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_down),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_route_dev,
                shared_route.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_route),
                self.stream,
            )?;
        }
        let prepared_sparse = match prepared_sparse_override {
            Some(prepared) => Some(prepared),
            None if tuning::prefill_moe_weight_prefetch_enabled() => {
                Some(self.qwen35_prepare_sparse_slots_by_token(
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                    token_count,
                    true,
                )?)
            }
            None => None,
        };

        self.sgemm_device(
            shared_gate_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_gate_out_dev,
        )?;
        self.sgemm_device(
            shared_up_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_up_out_dev,
        )?;
        self.launch_silu_mul(shared_gate_out_dev, shared_up_out_dev, token_count * n_ff)?;
        self.sgemm_device(
            shared_down_dev,
            n_embd,
            n_ff,
            shared_gate_out_dev,
            token_count,
            output_dev,
        )?;
        self.launch_scale_rows_inplace(output_dev, shared_route_dev, n_embd, token_count)?;
        if tuning::prefill_moe_sync_before_sparse_enabled() {
            self.stream_synchronize()?;
        }
        self.qwen35_sparse_experts_by_token_to_dev_prepared(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            sparse_output_dev,
            true,
            false,
            Some(expert_ids),
            prepared_sparse,
        )?;
        self.launch_add_f32_inplace(sparse_output_dev, output_dev, output_len)?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                sparse_output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prefill_moe_f32_shared_sparse_by_token_device_input(
        &mut self,
        shared_gate: &[f32],
        shared_up: &[f32],
        shared_down: &[f32],
        shared_input_scale: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_id: rnb_backend_api::DeviceTensorId,
        input_desc: rnb_backend_api::DeviceTensorDesc,
        residual_id: rnb_backend_api::DeviceTensorId,
        residual_desc: rnb_backend_api::DeviceTensorDesc,
        prepared_sparse_override: Option<PreparedQwen35SparseSlots>,
        deferred_selected_base: Option<DeferredQwen35SelectedBaseSparse<'_>>,
    ) -> Result<rnb_backend_api::DeviceTensorId, String> {
        self.qwen35_prefill_moe_shared_sparse_by_token_device_input_impl(
            Qwen35DeviceSharedWeights::F32 {
                gate: shared_gate,
                up: shared_up,
                down: shared_down,
            },
            shared_input_scale,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            prepared_sparse_override,
            deferred_selected_base,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prefill_moe_f32_shared_sparse_by_token_device_input_reuse_residual(
        &mut self,
        shared_gate: &[f32],
        shared_up: &[f32],
        shared_down: &[f32],
        shared_input_scale: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_id: rnb_backend_api::DeviceTensorId,
        input_desc: rnb_backend_api::DeviceTensorDesc,
        residual_id: rnb_backend_api::DeviceTensorId,
        residual_desc: rnb_backend_api::DeviceTensorDesc,
        prepared_sparse_override: Option<PreparedQwen35SparseSlots>,
        deferred_selected_base: Option<DeferredQwen35SelectedBaseSparse<'_>>,
    ) -> Result<rnb_backend_api::DeviceTensorId, String> {
        self.qwen35_prefill_moe_shared_sparse_by_token_device_input_impl(
            Qwen35DeviceSharedWeights::F32 {
                gate: shared_gate,
                up: shared_up,
                down: shared_down,
            },
            shared_input_scale,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            prepared_sparse_override,
            deferred_selected_base,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prefill_moe_quant_shared_sparse_by_token_device_input(
        &mut self,
        shared_gate: &[u8],
        shared_gate_quant: u32,
        shared_up: &[u8],
        shared_up_quant: u32,
        shared_down: &[u8],
        shared_down_quant: u32,
        shared_input_scale: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_id: rnb_backend_api::DeviceTensorId,
        input_desc: rnb_backend_api::DeviceTensorDesc,
        residual_id: rnb_backend_api::DeviceTensorId,
        residual_desc: rnb_backend_api::DeviceTensorDesc,
        prepared_sparse_override: Option<PreparedQwen35SparseSlots>,
        deferred_selected_base: Option<DeferredQwen35SelectedBaseSparse<'_>>,
        reuse_residual_output: bool,
    ) -> Result<rnb_backend_api::DeviceTensorId, String> {
        self.qwen35_prefill_moe_shared_sparse_by_token_device_input_impl(
            Qwen35DeviceSharedWeights::Quant {
                gate: shared_gate,
                gate_quant: shared_gate_quant,
                up: shared_up,
                up_quant: shared_up_quant,
                down: shared_down,
                down_quant: shared_down_quant,
            },
            shared_input_scale,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_id,
            input_desc,
            residual_id,
            residual_desc,
            prepared_sparse_override,
            deferred_selected_base,
            reuse_residual_output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_prefill_moe_shared_sparse_by_token_device_input_impl(
        &mut self,
        shared: Qwen35DeviceSharedWeights<'_>,
        shared_input_scale: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_id: rnb_backend_api::DeviceTensorId,
        input_desc: rnb_backend_api::DeviceTensorDesc,
        residual_id: rnb_backend_api::DeviceTensorId,
        residual_desc: rnb_backend_api::DeviceTensorDesc,
        prepared_sparse_override: Option<PreparedQwen35SparseSlots>,
        deferred_selected_base: Option<DeferredQwen35SelectedBaseSparse<'_>>,
        reuse_residual_output: bool,
    ) -> Result<rnb_backend_api::DeviceTensorId, String> {
        if n_embd == 0 || n_ff == 0 || token_count == 0 {
            return Err(format!(
                "Qwen35 device-input MoE dimensions must be non-zero: tokens={token_count} n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        let expected_input_desc = rnb_backend_api::DeviceTensorDesc::new(
            token_count,
            n_embd,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::Normalized,
        );
        if input_desc != expected_input_desc {
            return Err(format!(
                "Qwen35 device-input MoE normalized desc mismatch: got {:?}, expected {:?}",
                input_desc, expected_input_desc
            ));
        }
        if residual_desc.rows() != token_count
            || residual_desc.cols() != n_embd
            || residual_desc.dtype() != rnb_backend_api::ScalarType::F32
            || residual_desc.role() != rnb_backend_api::DeviceTensorRole::Hidden
        {
            return Err(format!(
                "Qwen35 device-input MoE residual desc mismatch: got {}x{} {:?} {:?}, expected {}x{} F32 Hidden",
                residual_desc.rows(),
                residual_desc.cols(),
                residual_desc.dtype(),
                residual_desc.role(),
                token_count,
                n_embd
            ));
        }
        let shared_quant_blocks = match shared {
            Qwen35DeviceSharedWeights::F32 { gate, up, down } => {
                if gate.len() != n_ff * n_embd
                    || up.len() != n_ff * n_embd
                    || down.len() != n_embd * n_ff
                {
                    return Err(
                        "Qwen35 device-input MoE shared f32 weight shape mismatch".to_string()
                    );
                }
                None
            }
            Qwen35DeviceSharedWeights::Quant {
                gate,
                gate_quant,
                up,
                up_quant,
                down,
                down_quant,
            } => Some((
                super::mtp_verify::validate_mtp_verify_k_quant_matrix(
                    "Qwen35 device-input shared gate",
                    gate_quant,
                    gate,
                    n_ff,
                    n_embd,
                    n_embd,
                )?,
                super::mtp_verify::validate_mtp_verify_k_quant_matrix(
                    "Qwen35 device-input shared up",
                    up_quant,
                    up,
                    n_ff,
                    n_embd,
                    n_embd,
                )?,
                super::mtp_verify::validate_mtp_verify_k_quant_matrix(
                    "Qwen35 device-input shared down",
                    down_quant,
                    down,
                    n_embd,
                    n_ff,
                    n_ff,
                )?,
            )),
        };
        if shared_input_scale.len() != n_embd {
            return Err(format!(
                "Qwen35 device-input MoE shared scale length mismatch: got {}, expected {n_embd}",
                shared_input_scale.len()
            ));
        }
        if prepared_sparse_override.is_some() && deferred_selected_base.is_some() {
            return Err(
                "Qwen35 device-input MoE cannot take both prepared and deferred selected sparse"
                    .to_string(),
            );
        }
        let prepared_device_slot_count = prepared_sparse_override
            .as_ref()
            .and_then(|prepared| prepared.device_slot_ptrs.as_ref())
            .map(|slots| slots.expert_ids.len());
        let prepared_slot_count = prepared_sparse_override
            .as_ref()
            .and_then(|prepared| prepared.slot_count);
        let empty_selected_base_sparse = gate_weights.is_empty()
            && up_weights.is_empty()
            && down_weights.is_empty()
            && (prepared_device_slot_count.is_some()
                || prepared_slot_count.is_some()
                || deferred_selected_base.is_some());
        if empty_selected_base_sparse {
            let slots = qwen35_sparse_slot_count(
                gate_weights.len(),
                prepared_device_slot_count,
                prepared_slot_count
                    .or_else(|| deferred_selected_base.as_ref().map(|_| expert_ids.len())),
            );
            if expert_ids.len() != slots || route_weights.len() != slots || token_ids.len() != slots
            {
                return Err(format!(
                    "Qwen35 device-input MoE selected-base sparse length mismatch: slots={slots} experts={} route={} token={}",
                    expert_ids.len(),
                    route_weights.len(),
                    token_ids.len()
                ));
            }
        } else if expert_ids.len() != gate_weights.len()
            || expert_ids.len() != route_weights.len()
            || expert_ids.len() != token_ids.len()
            || gate_weights.len() != up_weights.len()
            || gate_weights.len() != down_weights.len()
        {
            return Err(format!(
                "Qwen35 device-input MoE sparse batch length mismatch: experts={} gate={} up={} down={} route={} token={}",
                expert_ids.len(),
                gate_weights.len(),
                up_weights.len(),
                down_weights.len(),
                route_weights.len(),
                token_ids.len()
            ));
        }

        let phase_profile = tuning::qwen35_device_moe_phase_profile_enabled();
        let total_start = phase_profile.then(std::time::Instant::now);
        let phase_start = phase_profile.then(std::time::Instant::now);
        let input_dev = self.device_tensor_ptr(input_id, input_desc)?;
        let residual_dev = self.device_tensor_ptr(residual_id, residual_desc)?;
        let output_desc = rnb_backend_api::DeviceTensorDesc::new(
            token_count,
            n_embd,
            rnb_backend_api::ScalarType::F32,
            rnb_backend_api::DeviceTensorRole::MoeOutput,
        );
        let output_len = output_desc.len();
        let output_bytes = output_desc
            .byte_len()
            .ok_or_else(|| "Qwen35 device-input MoE output byte overflow".to_string())?;

        self.set_current()?;
        let output_dev = if reuse_residual_output {
            residual_dev
        } else {
            unsafe { self.api.mem_alloc(output_bytes)? }
        };
        if let Some(start) = phase_start {
            log_qwen35_device_moe_compute_phase(
                "ptrs_alloc_output",
                token_count,
                n_ff,
                n_embd,
                start.elapsed(),
            );
        }
        let run = (|| -> Result<(), String> {
            macro_rules! phase_start {
                () => {
                    phase_profile.then(std::time::Instant::now)
                };
            }
            macro_rules! profile_phase {
                ($label:expr, $start:expr) => {
                    if let Some(start) = $start {
                        self.stream_synchronize()?;
                        log_qwen35_device_moe_compute_phase(
                            $label,
                            token_count,
                            n_ff,
                            n_embd,
                            start.elapsed(),
                        );
                    }
                };
            }

            let (shared_gate_dev, shared_up_dev, shared_down_dev) = match shared {
                Qwen35DeviceSharedWeights::F32 { gate, up, down } => (
                    Some(self.compute_full_gate_ptr(std::mem::size_of_val(gate))?),
                    Some(self.compute_full_up_ptr(std::mem::size_of_val(up))?),
                    Some(self.compute_full_down_ptr(std::mem::size_of_val(down))?),
                ),
                Qwen35DeviceSharedWeights::Quant { .. } => (None, None, None),
            };
            let shared_scale_dev = self.resident_f32_ptr(shared_input_scale)?;
            let shared_route_dev =
                self.compute_route_ptr(token_count * std::mem::size_of::<f32>())?;
            let shared_gate_out_dev =
                self.compute_mid_a_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
            let shared_up_out_dev =
                self.compute_mid_b_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
            let shared_output_dev = self.compute_output_ptr(output_bytes)?;
            let sparse_output_dev = self.compute_aux_output_ptr(output_bytes)?;

            let phase_start = phase_start!();
            unsafe {
                if !reuse_residual_output {
                    self.api.memcpy_dtod_async(
                        output_dev,
                        residual_dev,
                        output_bytes,
                        self.stream,
                    )?;
                }
                if let Qwen35DeviceSharedWeights::F32 { gate, up, down } = shared {
                    self.api.memcpy_htod_async(
                        shared_gate_dev.expect("F32 shared gate device buffer"),
                        gate.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(gate),
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        shared_up_dev.expect("F32 shared up device buffer"),
                        up.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(up),
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        shared_down_dev.expect("F32 shared down device buffer"),
                        down.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(down),
                        self.stream,
                    )?;
                }
            }
            profile_phase!("residual_copy_shared_h2d", phase_start);

            let stream_selected_base = tuning::qwen35_selected_base_stream_enabled(token_count)
                && deferred_selected_base.is_some();
            let delay_selected_sparse_prepare_until_sparse_phase = stream_selected_base
                || (qwen35_selected_sparse_fused_boundary_enabled()
                    && deferred_selected_base.is_some()
                    && !reuse_residual_output);
            let mut deferred_selected_base = deferred_selected_base;
            let phase_start = phase_start!();
            let mut prepared_sparse = if !delay_selected_sparse_prepare_until_sparse_phase {
                if let Some(deferred) = deferred_selected_base.take() {
                    Some(
                        self.qwen35_prepare_selected_base_temp_slab_device_slot_ptrs_by_token(
                            deferred.gate_all,
                            deferred.up_all,
                            deferred.down_all,
                            deferred.expert_ids,
                            deferred.down_quant,
                            deferred.n_ff,
                            deferred.n_embd,
                        )?,
                    )
                } else {
                    None
                }
            } else {
                None
            };
            profile_phase!("sparse_prepare_overlap", phase_start);

            let phase_start = phase_start!();
            self.launch_qwen35_shared_route_sigmoid_f32(
                shared_route_dev,
                input_dev,
                shared_scale_dev,
                token_count,
                n_embd,
            )?;
            profile_phase!("shared_route", phase_start);
            match (shared, shared_quant_blocks) {
                (Qwen35DeviceSharedWeights::F32 { .. }, None) => {
                    let phase_start = phase_start!();
                    self.sgemm_device(
                        shared_gate_dev.expect("F32 shared gate device buffer"),
                        n_ff,
                        n_embd,
                        input_dev,
                        token_count,
                        shared_gate_out_dev,
                    )?;
                    profile_phase!("shared_gate_gemm", phase_start);
                    let phase_start = phase_start!();
                    self.sgemm_device(
                        shared_up_dev.expect("F32 shared up device buffer"),
                        n_ff,
                        n_embd,
                        input_dev,
                        token_count,
                        shared_up_out_dev,
                    )?;
                    profile_phase!("shared_up_gemm", phase_start);
                    let phase_start = phase_start!();
                    self.launch_silu_mul(
                        shared_gate_out_dev,
                        shared_up_out_dev,
                        token_count * n_ff,
                    )?;
                    profile_phase!("shared_silu_mul", phase_start);
                    let phase_start = phase_start!();
                    self.sgemm_device(
                        shared_down_dev.expect("F32 shared down device buffer"),
                        n_embd,
                        n_ff,
                        shared_gate_out_dev,
                        token_count,
                        shared_output_dev,
                    )?;
                    profile_phase!("shared_down_gemm", phase_start);
                }
                (
                    Qwen35DeviceSharedWeights::Quant {
                        gate,
                        gate_quant,
                        up,
                        up_quant,
                        down,
                        down_quant,
                    },
                    Some((gate_blocks, up_blocks, down_blocks)),
                ) => {
                    let phase_start = phase_start!();
                    self.stage_mtp_verify_k_quant_projection_to_dev(
                        "Qwen35 device-input shared gate",
                        gate_quant,
                        gate,
                        n_ff,
                        gate_blocks,
                        token_count,
                        input_dev,
                        shared_gate_out_dev,
                    )?;
                    profile_phase!("shared_gate_quant", phase_start);
                    let phase_start = phase_start!();
                    self.stage_mtp_verify_k_quant_projection_to_dev(
                        "Qwen35 device-input shared up",
                        up_quant,
                        up,
                        n_ff,
                        up_blocks,
                        token_count,
                        input_dev,
                        shared_up_out_dev,
                    )?;
                    profile_phase!("shared_up_quant", phase_start);
                    let phase_start = phase_start!();
                    self.launch_silu_mul(
                        shared_gate_out_dev,
                        shared_up_out_dev,
                        token_count * n_ff,
                    )?;
                    profile_phase!("shared_silu_mul", phase_start);
                    let phase_start = phase_start!();
                    self.stage_mtp_verify_k_quant_projection_to_dev(
                        "Qwen35 device-input shared down",
                        down_quant,
                        down,
                        n_embd,
                        down_blocks,
                        token_count,
                        shared_gate_out_dev,
                        shared_output_dev,
                    )?;
                    profile_phase!("shared_down_quant", phase_start);
                }
                _ => unreachable!("shared weight representation and validation must agree"),
            }
            let phase_start = phase_start!();
            self.launch_scale_rows_inplace(
                shared_output_dev,
                shared_route_dev,
                n_embd,
                token_count,
            )?;
            self.launch_add_f32_inplace(output_dev, shared_output_dev, output_len)?;
            profile_phase!("shared_scale_add", phase_start);
            if tuning::prefill_moe_sync_before_sparse_enabled() {
                self.stream_synchronize()?;
            }

            let phase_start = phase_start!();
            let streamed_sparse = if stream_selected_base {
                if let Some(deferred) = deferred_selected_base.as_ref() {
                    self.qwen35_sparse_experts_selected_base_stream_to_dev(
                        deferred.gate_all,
                        deferred.up_all,
                        deferred.down_all,
                        deferred.expert_ids,
                        route_weights,
                        token_ids,
                        token_count,
                        deferred.down_quant,
                        deferred.n_ff,
                        deferred.n_embd,
                        input_dev,
                        sparse_output_dev,
                    )?
                } else {
                    false
                }
            } else {
                false
            };
            if !streamed_sparse {
                if prepared_sparse.is_none() {
                    prepared_sparse = if let Some(deferred) = deferred_selected_base.take() {
                        Some(
                            self.qwen35_prepare_selected_base_temp_slab_device_slot_ptrs_by_token(
                                deferred.gate_all,
                                deferred.up_all,
                                deferred.down_all,
                                deferred.expert_ids,
                                deferred.down_quant,
                                deferred.n_ff,
                                deferred.n_embd,
                            )?,
                        )
                    } else if let Some(prepared) = prepared_sparse_override {
                        Some(prepared)
                    } else if tuning::prefill_moe_weight_prefetch_enabled() {
                        Some(self.qwen35_prepare_sparse_slots_by_token(
                            gate_weights,
                            up_weights,
                            down_weights,
                            route_weights,
                            token_count,
                            true,
                        )?)
                    } else {
                        None
                    };
                }
                if qwen35_device_sparse_route_enabled() {
                    if let Some(prepared) = prepared_sparse.as_mut() {
                        prepared.device_route =
                            Some(self.qwen35_prepare_device_sparse_route_by_token(
                                route_weights,
                                token_ids,
                            )?);
                    }
                }
                self.qwen35_sparse_experts_by_token_to_dev_prepared(
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                    token_ids,
                    token_count,
                    down_quant,
                    n_ff,
                    n_embd,
                    input_dev,
                    sparse_output_dev,
                    true,
                    false,
                    Some(expert_ids),
                    prepared_sparse,
                )?;
            }
            profile_phase!("sparse_compute", phase_start);
            let phase_start = phase_start!();
            self.launch_add_f32_inplace(output_dev, sparse_output_dev, output_len)?;
            profile_phase!("sparse_add", phase_start);
            Ok(())
        })();
        if let Err(err) = run {
            if !reuse_residual_output {
                if let Err(cleanup_err) = unsafe { self.api.mem_free(output_dev) } {
                    return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
                }
            }
            return Err(err);
        }
        self.stream_synchronize()?;

        if reuse_residual_output {
            self.retag_device_tensor_slot(residual_id, residual_desc, output_desc)?;
            if let Some(start) = total_start {
                log_qwen35_device_moe_compute_phase(
                    "total",
                    token_count,
                    n_ff,
                    n_embd,
                    start.elapsed(),
                );
            }
            return Ok(residual_id);
        }

        let phase_start = phase_profile.then(std::time::Instant::now);
        let output_id = match self.insert_device_tensor_slot(output_dev, output_bytes, output_desc)
        {
            Ok(output_id) => output_id,
            Err(err) => {
                if let Err(cleanup_err) = unsafe { self.api.mem_free(output_dev) } {
                    return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
                }
                return Err(err);
            }
        };
        if let Some(start) = phase_start {
            log_qwen35_device_moe_compute_phase(
                "slot_insert",
                token_count,
                n_ff,
                n_embd,
                start.elapsed(),
            );
        }
        if let Some(start) = total_start {
            log_qwen35_device_moe_compute_phase(
                "total",
                token_count,
                n_ff,
                n_embd,
                start.elapsed(),
            );
        }
        Ok(output_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_prefill_moe_q4_shared_sparse_by_token_cached(
        &mut self,
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        shared_route: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        shared_down_quant: u32,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        self.qwen35_prefill_moe_q4_shared_sparse_by_token_cached_prepared(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            shared_down_quant,
            down_quant,
            n_ff,
            n_embd,
            input,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_prefill_moe_q4_shared_sparse_selected_base_by_token_cached(
        &mut self,
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        shared_route: &[f32],
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        shared_down_quant: u32,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        let direct_sparse = qwen35_selected_base_device_slot_ptrs_enabled()
            && qwen35_selected_base_direct_sparse_enabled();
        let selected = if direct_sparse {
            None
        } else {
            Some(qwen35_selected_base_sparse_inputs_from_full_layer(
                gate_all,
                up_all,
                down_all,
                expert_ids,
                route_weights,
                token_ids,
                down_quant,
                n_ff,
                n_embd,
            )?)
        };
        let prepared_sparse = if direct_sparse {
            Some(
                self.qwen35_prepare_selected_base_residency_aware_device_slot_ptrs_by_token(
                    gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
                )?,
            )
        } else if qwen35_selected_base_temp_slab_ptrs_enabled() {
            Some(self.qwen35_prepare_selected_base_temp_slab_slots_by_token(
                gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
            )?)
        } else {
            None
        };
        let empty_sparse_slots: [&[u8]; 0] = [];
        let (gate_weights, up_weights, down_weights, sparse_route_weights, sparse_token_ids) =
            match selected.as_ref() {
                Some(selected) => (
                    selected.gate_weights.as_slice(),
                    selected.up_weights.as_slice(),
                    selected.down_weights.as_slice(),
                    selected.route_weights,
                    selected.token_ids,
                ),
                None => (
                    &empty_sparse_slots[..],
                    &empty_sparse_slots[..],
                    &empty_sparse_slots[..],
                    route_weights,
                    token_ids,
                ),
            };
        self.qwen35_prefill_moe_q4_shared_sparse_by_token_cached_prepared(
            shared_gate,
            shared_up,
            shared_down,
            shared_route,
            gate_weights,
            up_weights,
            down_weights,
            expert_ids,
            sparse_route_weights,
            sparse_token_ids,
            token_count,
            shared_down_quant,
            down_quant,
            n_ff,
            n_embd,
            input,
            prepared_sparse,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_prefill_moe_q4_shared_sparse_by_token_cached_prepared(
        &mut self,
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        shared_route: &[f32],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        shared_down_quant: u32,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        prepared_sparse_override: Option<PreparedQwen35SparseSlots>,
    ) -> Result<Option<Vec<f32>>, String> {
        if !tuning::qwen35_shared_q4_f32_cache_enabled_for_seq(token_count) {
            return Ok(None);
        }
        if n_embd == 0 || n_ff == 0 {
            return Err("Qwen35 cached shared MoE dimensions must be non-zero".to_string());
        }
        if n_embd % 256 != 0 || n_ff % 256 != 0 {
            return Ok(None);
        }
        let shared_row_bytes_embd = (n_embd / 256) * 144;
        let shared_down_row_bytes = match shared_down_quant {
            12 => (n_ff / 256) * 144,
            14 => (n_ff / 256) * 210,
            _ => return Ok(None),
        };
        if shared_gate.len() != n_ff * shared_row_bytes_embd
            || shared_up.len() != n_ff * shared_row_bytes_embd
            || shared_down.len() != n_embd * shared_down_row_bytes
        {
            return Err("Qwen35 cached shared weight shape mismatch".to_string());
        }
        if shared_route.len() != token_count {
            return Err(format!(
                "Qwen35 cached shared route length mismatch: got {}, expected {token_count}",
                shared_route.len()
            ));
        }
        if input.len() != token_count * n_embd {
            return Err(format!(
                "Qwen35 cached shared input length mismatch: got {}, expected {}",
                input.len(),
                token_count * n_embd
            ));
        }

        let Some(shared_gate_dev) = self.resident_q4k_f32_ptr(shared_gate, n_ff, n_embd / 256)?
        else {
            return Ok(None);
        };
        let Some(shared_up_dev) = self.resident_q4k_f32_ptr(shared_up, n_ff, n_embd / 256)? else {
            return Ok(None);
        };
        let shared_down_dev = match shared_down_quant {
            12 => self.resident_q4k_f32_ptr(shared_down, n_embd, n_ff / 256)?,
            14 => self.resident_q6k_f32_ptr(shared_down, n_embd, n_ff / 256)?,
            _ => None,
        };
        let Some(shared_down_dev) = shared_down_dev else {
            return Ok(None);
        };

        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let shared_route_dev = self.compute_route_ptr(std::mem::size_of_val(shared_route))?;
        let shared_gate_out_dev =
            self.compute_mid_a_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let shared_up_out_dev =
            self.compute_mid_b_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let output_len = token_count * n_embd;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let sparse_output_dev = self.compute_aux_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_route_dev,
                shared_route.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_route),
                self.stream,
            )?;
        }
        let prepared_sparse = match prepared_sparse_override {
            Some(prepared_sparse) => Some(prepared_sparse),
            None if tuning::prefill_moe_weight_prefetch_enabled() => {
                Some(self.qwen35_prepare_sparse_slots_by_token(
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                    token_count,
                    true,
                )?)
            }
            None => None,
        };

        self.sgemm_device(
            shared_gate_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_gate_out_dev,
        )?;
        self.sgemm_device(
            shared_up_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_up_out_dev,
        )?;
        self.launch_silu_mul(shared_gate_out_dev, shared_up_out_dev, token_count * n_ff)?;
        self.sgemm_device(
            shared_down_dev,
            n_embd,
            n_ff,
            shared_gate_out_dev,
            token_count,
            output_dev,
        )?;
        self.launch_scale_rows_inplace(output_dev, shared_route_dev, n_embd, token_count)?;
        if tuning::prefill_moe_sync_before_sparse_enabled() {
            self.stream_synchronize()?;
        }
        self.qwen35_sparse_experts_by_token_to_dev_prepared(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            sparse_output_dev,
            true,
            false,
            Some(expert_ids),
            prepared_sparse,
        )?;
        self.launch_add_f32_inplace(sparse_output_dev, output_dev, output_len)?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                sparse_output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some(output))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_prefill_moe_q4_shared_sparse_full_layer_by_token_cached(
        &mut self,
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        shared_route: &[f32],
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        shared_down_quant: u32,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        if !tuning::qwen35_full_layer_shared_q4_f32_cache_enabled() {
            return Ok(None);
        }
        if n_embd == 0 || n_ff == 0 {
            return Err(
                "Qwen35 cached full-layer shared MoE dimensions must be non-zero".to_string(),
            );
        }
        if n_embd % 256 != 0 || n_ff % 256 != 0 {
            return Ok(None);
        }
        let shared_row_bytes_embd = (n_embd / 256) * 144;
        let shared_down_row_bytes = match shared_down_quant {
            12 => (n_ff / 256) * 144,
            14 => (n_ff / 256) * 210,
            _ => return Ok(None),
        };
        if shared_gate.len() != n_ff * shared_row_bytes_embd
            || shared_up.len() != n_ff * shared_row_bytes_embd
            || shared_down.len() != n_embd * shared_down_row_bytes
        {
            return Err("Qwen35 cached full-layer shared weight shape mismatch".to_string());
        }
        if shared_route.len() != token_count {
            return Err(format!(
                "Qwen35 cached full-layer shared route length mismatch: got {}, expected {token_count}",
                shared_route.len()
            ));
        }
        if input.len() != token_count * n_embd {
            return Err(format!(
                "Qwen35 cached full-layer shared input length mismatch: got {}, expected {}",
                input.len(),
                token_count * n_embd
            ));
        }

        let Some(shared_gate_dev) = self.resident_q4k_f32_ptr(shared_gate, n_ff, n_embd / 256)?
        else {
            return Ok(None);
        };
        let Some(shared_up_dev) = self.resident_q4k_f32_ptr(shared_up, n_ff, n_embd / 256)? else {
            return Ok(None);
        };
        let shared_down_dev = match shared_down_quant {
            12 => self.resident_q4k_f32_ptr(shared_down, n_embd, n_ff / 256)?,
            14 => self.resident_q6k_f32_ptr(shared_down, n_embd, n_ff / 256)?,
            _ => None,
        };
        let Some(shared_down_dev) = shared_down_dev else {
            return Ok(None);
        };

        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let shared_route_dev = self.compute_route_ptr(std::mem::size_of_val(shared_route))?;
        let shared_gate_out_dev =
            self.compute_mid_a_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let shared_up_out_dev =
            self.compute_mid_b_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let output_len = token_count * n_embd;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let sparse_output_dev = self.compute_aux_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_route_dev,
                shared_route.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_route),
                self.stream,
            )?;
        }

        self.sgemm_device(
            shared_gate_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_gate_out_dev,
        )?;
        self.sgemm_device(
            shared_up_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_up_out_dev,
        )?;
        self.launch_silu_mul(shared_gate_out_dev, shared_up_out_dev, token_count * n_ff)?;
        self.sgemm_device(
            shared_down_dev,
            n_embd,
            n_ff,
            shared_gate_out_dev,
            token_count,
            output_dev,
        )?;
        self.launch_scale_rows_inplace(output_dev, shared_route_dev, n_embd, token_count)?;
        if tuning::prefill_moe_sync_before_sparse_enabled() {
            self.stream_synchronize()?;
        }
        self.qwen35_sparse_experts_full_layer_by_token_to_dev(
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            sparse_output_dev,
        )?;
        self.launch_add_f32_inplace(sparse_output_dev, output_dev, output_len)?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                sparse_output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some(output))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen35_prefill_moe_f32_shared_sparse_full_layer_by_token(
        &mut self,
        shared_gate: &[f32],
        shared_up: &[f32],
        shared_down: &[f32],
        shared_route: &[f32],
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if n_embd == 0 || n_ff == 0 {
            return Err("Qwen35 full-layer combined MoE dimensions must be non-zero".to_string());
        }
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let shared_gate_dev = self.compute_full_gate_ptr(std::mem::size_of_val(shared_gate))?;
        let shared_up_dev = self.compute_full_up_ptr(std::mem::size_of_val(shared_up))?;
        let shared_down_dev = self.compute_full_down_ptr(std::mem::size_of_val(shared_down))?;
        let shared_route_dev = self.compute_route_ptr(std::mem::size_of_val(shared_route))?;
        let shared_gate_out_dev =
            self.compute_mid_a_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let shared_up_out_dev =
            self.compute_mid_b_ptr(token_count * n_ff * std::mem::size_of::<f32>())?;
        let output_len = token_count * n_embd;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let sparse_output_dev = self.compute_aux_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                shared_gate_dev,
                shared_gate.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_gate),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_up_dev,
                shared_up.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_up),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_down_dev,
                shared_down.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_down),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                shared_route_dev,
                shared_route.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(shared_route),
                self.stream,
            )?;
        }

        self.sgemm_device(
            shared_gate_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_gate_out_dev,
        )?;
        self.sgemm_device(
            shared_up_dev,
            n_ff,
            n_embd,
            input_dev,
            token_count,
            shared_up_out_dev,
        )?;
        self.launch_silu_mul(shared_gate_out_dev, shared_up_out_dev, token_count * n_ff)?;
        self.sgemm_device(
            shared_down_dev,
            n_embd,
            n_ff,
            shared_gate_out_dev,
            token_count,
            output_dev,
        )?;
        self.launch_scale_rows_inplace(output_dev, shared_route_dev, n_embd, token_count)?;
        if tuning::prefill_moe_sync_before_sparse_enabled() {
            self.stream_synchronize()?;
        }
        self.qwen35_sparse_experts_full_layer_by_token_to_dev(
            gate_all,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            sparse_output_dev,
        )?;
        self.launch_add_f32_inplace(sparse_output_dev, output_dev, output_len)?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                sparse_output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn qwen35_expert(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        gelu: bool,
    ) -> Result<Vec<f32>, String> {
        let trace = std::env::var("RNB_CUDA_DENSE_FFN_TRACE").ok().as_deref() == Some("1");
        let t_total = trace.then(std::time::Instant::now);
        let output_bytes = n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let t_h2d = trace.then(std::time::Instant::now);
        let input_dev = self.compute_full_gate_ptr(std::mem::size_of_val(input))?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        if trace {
            self.stream_synchronize()?;
        }
        let h2d_ms = t_h2d
            .map(|t| t.elapsed().as_micros() as f64 / 1000.0)
            .unwrap_or(0.0);
        let t_chain = trace.then(std::time::Instant::now);
        self.qwen35_expert_dev_input_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
            None,
            gelu,
        )?;
        if trace {
            self.stream_synchronize()?;
        }
        let chain_ms = t_chain
            .map(|t| t.elapsed().as_micros() as f64 / 1000.0)
            .unwrap_or(0.0);
        let mut output = vec![0.0f32; n_embd];
        let t_dtoh = trace.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        if let Some(t) = t_total {
            let total_ms = t.elapsed().as_micros() as f64 / 1000.0;
            eprintln!(
                "[cuda-dense-ffn] gelu={} down_quant={} n_ff={} n_embd={} h2d={:.3} chain={:.3} dtoh={:.3} total={:.3}",
                gelu,
                down_quant,
                n_ff,
                n_embd,
                h2d_ms,
                chain_ms,
                t_dtoh
                    .map(|t| t.elapsed().as_micros() as f64 / 1000.0)
                    .unwrap_or(0.0),
                total_ms
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_gelu_ffn_norm_residual(
        &mut self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        norm_weight: &[f32],
        post_norm_weight: Option<&[f32]>,
        n_ff: usize,
        n_embd: usize,
        hidden: &[f32],
        norm_eps: f32,
        unit_offset_norm: bool,
    ) -> Result<Vec<f32>, String> {
        if hidden.len() != n_embd {
            return Err(format!(
                "dense GELU FFN residual hidden length mismatch: got {}, expected {n_embd}",
                hidden.len()
            ));
        }
        if norm_weight.len() != n_embd {
            return Err(format!(
                "dense GELU FFN residual norm length mismatch: got {}, expected {n_embd}",
                norm_weight.len()
            ));
        }
        if let Some(post_norm_weight) = post_norm_weight {
            if post_norm_weight.len() != n_embd {
                return Err(format!(
                    "dense GELU FFN residual post norm length mismatch: got {}, expected {n_embd}",
                    post_norm_weight.len()
                ));
            }
        }
        let hidden_bytes = std::mem::size_of_val(hidden);
        let hidden_dev = self.compute_full_gate_ptr(hidden_bytes)?;
        let normed_dev = self.compute_full_up_ptr(hidden_bytes)?;
        let norm_weight_dev = self.resident_f32_ptr(norm_weight)?;
        let ffn_out_dev = self.compute_output_ptr(hidden_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                hidden_dev,
                hidden.as_ptr().cast::<libc::c_void>(),
                hidden_bytes,
                self.stream,
            )?;
        }
        self.launch_rms_norm_f32(
            hidden_dev,
            norm_weight_dev,
            normed_dev,
            norm_eps,
            n_embd,
            unit_offset_norm,
        )?;
        self.qwen35_expert_dev_input_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            normed_dev,
            ffn_out_dev,
            None,
            true,
        )?;
        if let Some(post_norm_weight) = post_norm_weight {
            let norm_weight_dev = self.resident_f32_ptr(post_norm_weight)?;
            self.launch_rms_norm_add_f32_inplace(
                ffn_out_dev,
                norm_weight_dev,
                hidden_dev,
                norm_eps,
                n_embd,
                unit_offset_norm,
            )?;
        } else {
            self.launch_add_f32_inplace(hidden_dev, ffn_out_dev, n_embd)?;
        }
        let mut output = vec![0.0f32; n_embd];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                hidden_dev,
                hidden_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_attention_output_gelu_ffn_norm_residual(
        &mut self,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        ple_gate_weights: Option<&[u8]>,
        ple_proj_weights: Option<&[u8]>,
        ple_post_norm_weight: Option<&[f32]>,
        ple_input: Option<&[f32]>,
        ple_input_device_offset: Option<usize>,
        ple_dim: usize,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_eps: f32,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_ple_norm: bool,
        // cu41 Phase 1: device-resident hidden state pipeline.
        // hidden_carrier_dev=Some(ptr): caller-provided dedicated device buffer.
        // attention.rs / gdn.rs / nemotron 의 compute_full_gate_ptr 충돌 회피.
        // skip_h2d_hidden=true: caller 가 이전 chain call 의 carrier 가 그대로 살아있음을 보장 (intermediate layer).
        // skip_d2h_hidden=true: caller 가 D2H + sync 안 받음 (다음 chain call 이 같은 device buffer 사용).
        // 셋 다 None/false = 기존 host pattern (외부 호환).
        hidden_carrier_dev: Option<u64>,
        skip_h2d_hidden: bool,
        skip_d2h_hidden: bool,
        // cu44 step 20: Gemma4 layer_output_scale 의 device 화. Some(scale) 시
        // chain end (D2H 직전) hidden_dev *= scale. host 의 apply_layer_output_scale_inplace
        // 와 동일 transform — chain path 시 host apply 가 skip 되어야 double 방지.
        layer_output_scale: Option<f32>,
        // cu47 step 33: attention forward 의 device output carrier. Some(ptr) 시
        // attn_out H2D skip + carrier 의 device 값 직접 사용. attention forward 가
        // device-resident path 사용한 경우만 caller 가 Some.
        attn_out_dev_carrier: Option<u64>,
        // cu58 step 3 (Task 6 fix): FFN activation 분기 — true=gelu (Gemma),
        // false=silu (Llama 등). 이전엔 함수 안에서 true hardcoded 였어서 silu arch
        // 진입 시 garbage. caller 가 arch 별로 정확히 전달.
        ffn_uses_gelu: bool,
        dense_chain_graph_allowed: bool,
        layer_segment_graph_context: Option<Cu71LayerSegmentGraphRuntimeContext>,
    ) -> Result<(), String> {
        // cu60 axis A step 1 — PLE branch 측정용 env gate.
        // RNB_CU60_NO_PLE=1 ⇒ PLE args 5개 강제 None → 기존 if let 분기
        // (line 4256 weight kind 판별, line 4513 PLE compute) 자동 skip.
        // Gemma 정확성 transient 깨짐 OK (측정 용도). Llama 등 PLE 미사용 arch
        // 는 build_*_args 에서 이미 None 이라 무영향 (대조군 검증).
        let (
            ple_gate_weights,
            ple_proj_weights,
            ple_post_norm_weight,
            ple_input,
            ple_input_device_offset,
        ) = if cu60_no_ple() {
            (None, None, None, None, None)
        } else {
            (
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
                ple_input_device_offset,
            )
        };
        // cu59 axis A step 2 fix — kernel phase timing start.
        // 기존엔 PLE weight kind 판별 후 시작 (앞쪽 ~110 line 의 length
        // 검증 / PLE weight kind 판별 / dense_chain_trace_call 누락).
        // chain function 본문 전체 cost 가 root cause 추적의 단위라
        // function body 첫 줄로 이동 — validation / weight kind 판별 / chain_trace
        // 모두 포함.
        // kernel = function body 시작 ~ memcpy_dtoh 직전 (모든 validation +
        // launch + h2d_async 포함).
        // d2h = memcpy_dtoh + stream_synchronize.
        let diag_active = super::chain_diag_bridge::is_active();
        let t_kernel_start = if diag_active {
            Some(std::time::Instant::now())
        } else {
            None
        };
        if hidden.len() != n_embd {
            return Err(format!(
                "dense attention+FFN hidden length mismatch: got {}, expected {n_embd}",
                hidden.len()
            ));
        }
        if attn_out.len() != o_cols {
            return Err(format!(
                "dense attention+FFN attn_out length mismatch: got {}, expected {o_cols}",
                attn_out.len()
            ));
        }
        if ffn_norm_weight.len() != n_embd {
            return Err(format!(
                "dense attention+FFN norm length mismatch: got {}, expected {n_embd}",
                ffn_norm_weight.len()
            ));
        }
        if let Some(weight) = post_attn_norm_weight {
            if weight.len() != n_embd {
                return Err(format!(
                    "dense attention+FFN post attention norm length mismatch: got {}, expected {n_embd}",
                    weight.len()
                ));
            }
        }
        if let Some(weight) = post_ffn_norm_weight {
            if weight.len() != n_embd {
                return Err(format!(
                    "dense attention+FFN post FFN norm length mismatch: got {}, expected {n_embd}",
                    weight.len()
                ));
            }
        }
        let ple_weight_kind = if ple_gate_weights.is_some()
            || ple_proj_weights.is_some()
            || ple_post_norm_weight.is_some()
            || ple_input.is_some()
            || ple_input_device_offset.is_some()
        {
            let (
                Some(ple_gate_weights),
                Some(ple_proj_weights),
                Some(ple_post_norm_weight),
                Some(ple_input),
            ) = (
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
            )
            else {
                return Err(
                    "dense attention+FFN PLE parameters must be all present or all absent"
                        .to_string(),
                );
            };
            if ple_dim == 0 || ple_dim % 256 != 0 {
                return Err(format!(
                    "dense attention+FFN PLE dim must be non-zero and divisible by 256, got {ple_dim}"
                ));
            }
            if ple_input.len() != ple_dim {
                return Err(format!(
                    "dense attention+FFN PLE input length mismatch: got {}, expected {ple_dim}",
                    ple_input.len()
                ));
            }
            if ple_post_norm_weight.len() != n_embd {
                return Err(format!(
                    "dense attention+FFN PLE post norm length mismatch: got {}, expected {n_embd}",
                    ple_post_norm_weight.len()
                ));
            }
            let ple_gate_row_bytes = (n_embd / 256) * 144;
            let ple_proj_row_bytes = (ple_dim / 256) * 144;
            let q4k_gate_bytes = ple_dim * ple_gate_row_bytes;
            let q4k_proj_bytes = n_embd * ple_proj_row_bytes;
            let f32_gate_bytes = ple_dim
                .checked_mul(n_embd)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "dense attention+FFN PLE F32 gate byte overflow".to_string())?;
            let f32_proj_bytes = n_embd
                .checked_mul(ple_dim)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "dense attention+FFN PLE F32 proj byte overflow".to_string())?;
            if ple_gate_weights.len() == q4k_gate_bytes && ple_proj_weights.len() == q4k_proj_bytes
            {
                Some(DensePleWeightKind::Q4K)
            } else if ple_gate_weights.len() == f32_gate_bytes
                && ple_proj_weights.len() == f32_proj_bytes
            {
                Some(DensePleWeightKind::F32)
            } else {
                return Err(format!(
                    "dense attention+FFN PLE byte mismatch: gate got {}, expected q4k {} or f32 {}; proj got {}, expected q4k {} or f32 {}",
                    ple_gate_weights.len(),
                    q4k_gate_bytes,
                    f32_gate_bytes,
                    ple_proj_weights.len(),
                    q4k_proj_bytes,
                    f32_proj_bytes
                ));
            }
        } else {
            None
        };
        let trace_call = dense_chain_trace_call();
        let trace_total = trace_call.map(|_| std::time::Instant::now());
        let mut trace_stage = std::time::Instant::now();
        // cu59 axis A step 2 fix: t_kernel_start / diag_active 는 function body
        // 첫 줄로 이동 (validation + weight kind 판별 + chain_trace 모두 포함).
        let hidden_bytes = std::mem::size_of_val(hidden);
        let attn_out_bytes = std::mem::size_of_val(attn_out);
        // cu41 Phase 1: caller-provided dedicated carrier 가 있으면 그걸 hidden_dev 로.
        // 없으면 기존 compute_full_gate_ptr cache (attention/gdn 와 공유, chain call
        // 사이에 살아남는다는 보장 없음 — caller 가 그래서 carrier 옵션 사용).
        let hidden_dev = match hidden_carrier_dev {
            Some(ptr) => ptr,
            None => self.compute_full_gate_ptr(hidden_bytes)?,
        };
        // cu47 step 33: caller-provided attn_out carrier 시 그걸로. H2D skip.
        let attn_out_dev = match attn_out_dev_carrier {
            Some(ptr) => ptr,
            None => self.compute_full_down_ptr(attn_out_bytes)?,
        };
        let normed_dev = self.compute_full_up_ptr(hidden_bytes)?;
        let proj_dev = self.compute_output_ptr(hidden_bytes)?;
        unsafe {
            if !skip_h2d_hidden {
                self.api.memcpy_htod_async(
                    hidden_dev,
                    hidden.as_ptr().cast::<libc::c_void>(),
                    hidden_bytes,
                    self.stream,
                )?;
            }
            // cu47 step 33: attn_out_dev_carrier 시 attention forward 가 이미
            // carrier 에 결과 write. H2D skip.
            if attn_out_dev_carrier.is_none() {
                self.api.memcpy_htod_async(
                    attn_out_dev,
                    attn_out.as_ptr().cast::<libc::c_void>(),
                    attn_out_bytes,
                    self.stream,
                )?;
            }
        }
        self.trace_dense_stage(
            trace_call,
            "cuda-dense-chain",
            "h2d_hidden_attn",
            &mut trace_stage,
        )?;

        let mut cu69_dense_chain_graph_launched = false;
        let mut cu71_layer_segment_graph_launched = false;
        let q4k_ple_graph_allowed = matches!(ple_weight_kind, Some(DensePleWeightKind::Q4K))
            && ple_input_device_offset.is_some()
            && !cu62_ple_megakernel();
        let f32_ple_graph_allowed = matches!(ple_weight_kind, Some(DensePleWeightKind::F32))
            && ple_input_device_offset.is_some();
        let graph_ple_weight_kind = if q4k_ple_graph_allowed {
            Some(DensePleWeightKind::Q4K)
        } else if f32_ple_graph_allowed {
            Some(DensePleWeightKind::F32)
        } else {
            None
        };
        let ple_graph_supported = ple_weight_kind.is_none() || graph_ple_weight_kind.is_some();
        let cu71_layer_segment_graph_requested =
            layer_segment_graph_context.is_some() && tuning::cu71_layer_segment_graph_enabled();
        let cu69_dense_chain_graph_requested = tuning::cu69_dense_chain_graph_enabled();
        if (dense_chain_graph_allowed || cu71_layer_segment_graph_requested)
            && (cu69_dense_chain_graph_requested || cu71_layer_segment_graph_requested)
            && !env_flag("RNB_CUDA_DENSE_CHAIN_TRACE")
            && !dense_expert_trace_enabled()
            && !tuning::dense_expert_graph_enabled()
            && !diag_active
            && skip_h2d_hidden
            && skip_d2h_hidden
            && ple_graph_supported
        {
            let combined_norms = dense_combined_norms_enabled(true);
            let o_blocks = o_cols / 256;
            let o_q8dot = dense_q4k_gemv_q8dot_enabled(n_embd >= 1024 && o_blocks >= 4);
            let o_weight_dev = self.resident_q4k_weights_ptr_pinned(o_weights)?;
            let gate_weight_dev = self.resident_q4k_weights_ptr_pinned(gate_weights)?;
            let up_weight_dev = self.resident_q4k_weights_ptr_pinned(up_weights)?;
            let down_weight_dev = self.resident_q4k_weights_ptr_pinned(down_weights)?;
            let (ple_gate_weight_dev, ple_proj_weight_dev) = match graph_ple_weight_kind {
                Some(DensePleWeightKind::Q4K) => (
                    self.resident_q4k_weights_ptr_pinned(
                        ple_gate_weights
                            .ok_or_else(|| "dense chain graph PLE gate missing".to_string())?,
                    )?,
                    self.resident_q4k_weights_ptr_pinned(
                        ple_proj_weights
                            .ok_or_else(|| "dense chain graph PLE proj missing".to_string())?,
                    )?,
                ),
                Some(DensePleWeightKind::F32) => (
                    self.resident_f32_weights_ptr_from_le_bytes(
                        ple_gate_weights
                            .ok_or_else(|| "dense chain graph PLE gate missing".to_string())?,
                        "dense chain graph F32 PLE gate",
                    )?,
                    self.resident_f32_weights_ptr_from_le_bytes(
                        ple_proj_weights
                            .ok_or_else(|| "dense chain graph PLE proj missing".to_string())?,
                        "dense chain graph F32 PLE proj",
                    )?,
                ),
                None => (0, 0),
            };
            let base_raw_weights_resident = [
                (o_weights, o_weight_dev),
                (gate_weights, gate_weight_dev),
                (up_weights, up_weight_dev),
                (down_weights, down_weight_dev),
            ]
            .into_iter()
            .all(|(weights, ptr)| {
                self.resident_q4k
                    .get(&q4k_resident_key(weights))
                    .is_some_and(|entry| entry.ptr == ptr && entry.pinned)
            });
            let ple_raw_weights_resident = if graph_ple_weight_kind == Some(DensePleWeightKind::Q4K)
            {
                [
                    (
                        ple_gate_weights
                            .ok_or_else(|| "dense chain graph PLE gate missing".to_string())?,
                        ple_gate_weight_dev,
                    ),
                    (
                        ple_proj_weights
                            .ok_or_else(|| "dense chain graph PLE proj missing".to_string())?,
                        ple_proj_weight_dev,
                    ),
                ]
                .into_iter()
                .all(|(weights, ptr)| {
                    self.resident_q4k
                        .get(&q4k_resident_key(weights))
                        .is_some_and(|entry| entry.ptr == ptr && entry.pinned)
                })
            } else {
                true
            };
            let raw_weights_resident = base_raw_weights_resident && ple_raw_weights_resident;
            if raw_weights_resident {
                let q8dot_gate_up = dense_q8dot_gate_up_enabled(ffn_uses_gelu);
                let packed_q4_gate_up = if q8dot_gate_up && dense_q4_packed_q8dot_enabled(true) {
                    let gate = self.resident_q4k_packed_ptrs(gate_weights, n_ff, n_embd / 256)?;
                    let up = self.resident_q4k_packed_ptrs(up_weights, n_ff, n_embd / 256)?;
                    match (gate, up) {
                        (Some(gate), Some(up)) => Some((gate, up)),
                        _ => None,
                    }
                } else {
                    None
                };
                let packed_q4_down =
                    if ffn_uses_gelu && down_quant == 12 && dense_q4_packed_q8dot_enabled(true) {
                        self.resident_q4k_packed_ptrs(down_weights, n_embd, n_ff / 256)?
                    } else {
                        None
                    };
                let packed_q6_down =
                    if ffn_uses_gelu && down_quant == 14 && dense_q6_packed_q8dot_enabled(true) {
                        self.resident_q6k_packed_ptrs(down_weights, n_embd, n_ff / 256)?
                    } else {
                        None
                    };
                let q8dot_down = dense_q8dot_down_enabled(
                    ffn_uses_gelu && (down_quant == 12 || packed_q6_down.is_some()),
                );
                let q8_input_prequantized =
                    post_attn_norm_weight.is_some() && combined_norms && q8dot_gate_up;
                let graph_ple_dim = if graph_ple_weight_kind.is_some() {
                    ple_dim
                } else {
                    0
                };
                let gate_dev =
                    self.compute_mid_a_ptr(n_ff.max(graph_ple_dim) * std::mem::size_of::<f32>())?;
                let up_dev = self.compute_mid_b_ptr(n_ff * std::mem::size_of::<f32>())?;
                let shared_q8_qs_bytes = [o_q8dot.then_some(o_cols), q8dot_down.then_some(n_ff)]
                    .into_iter()
                    .flatten()
                    .max()
                    .unwrap_or(0);
                let shared_q8_ds_bytes = [
                    o_q8dot.then_some(o_blocks * 8 * std::mem::size_of::<f32>()),
                    q8dot_down.then_some((n_ff / 32) * std::mem::size_of::<f32>()),
                ]
                .into_iter()
                .flatten()
                .max()
                .unwrap_or(0);
                let shared_q8_qs_dev = if shared_q8_qs_bytes > 0 {
                    self.compute_input_ptr(shared_q8_qs_bytes)?
                } else {
                    0
                };
                let shared_q8_ds_dev = if shared_q8_ds_bytes > 0 {
                    self.compute_aux_output_ptr(shared_q8_ds_bytes)?
                } else {
                    0
                };
                let (o_q8_qs_dev, o_q8_ds_dev) = if o_q8dot {
                    (shared_q8_qs_dev, shared_q8_ds_dev)
                } else {
                    (0, 0)
                };
                let (q8_qs_dev, q8_ds_dev) = if q8dot_gate_up {
                    (
                        self.compute_gate_ptrs_ptr(n_embd)?,
                        self.compute_up_ptrs_ptr((n_embd / 32) * std::mem::size_of::<f32>())?,
                    )
                } else {
                    (0, 0)
                };
                let (down_q8_qs_dev, down_q8_ds_dev) = if q8dot_down {
                    (shared_q8_qs_dev, shared_q8_ds_dev)
                } else {
                    (0, 0)
                };
                let ffn_norm_dev = self.resident_f32_ptr(ffn_norm_weight)?;
                let post_attn_norm_dev = post_attn_norm_weight
                    .map(|weight| self.resident_f32_ptr(weight))
                    .transpose()?
                    .unwrap_or(0);
                let post_ffn_norm_dev = post_ffn_norm_weight
                    .map(|weight| self.resident_f32_ptr(weight))
                    .transpose()?
                    .unwrap_or(0);
                let ple_post_norm_dev = if graph_ple_weight_kind.is_some() {
                    self.resident_f32_ptr(
                        ple_post_norm_weight
                            .ok_or_else(|| "dense chain graph PLE norm missing".to_string())?,
                    )?
                } else {
                    0
                };
                let ple_input_dev = if graph_ple_weight_kind.is_some() {
                    self.gemma_ple_base_slice_ptr(
                        ple_input_device_offset.ok_or_else(|| {
                            "dense chain graph PLE input offset missing".to_string()
                        })?,
                        ple_dim,
                    )?
                } else {
                    0
                };
                let (packed_q6_down_qs, packed_q6_down_d_super, packed_q6_down_sub_scale) =
                    packed_q6_down.unwrap_or((0, 0, 0));
                let key = DenseChainGraphKey {
                    down_quant,
                    o_cols,
                    n_ff,
                    n_embd,
                    norm_eps_bits: norm_eps.to_bits(),
                    ffn_uses_gelu,
                    combined_norms,
                    o_q8dot,
                    q8dot_gate_up,
                    q8dot_down,
                    q8_input_prequantized,
                    has_post_attn_norm: post_attn_norm_weight.is_some(),
                    has_post_ffn_norm: post_ffn_norm_weight.is_some(),
                    has_ple: graph_ple_weight_kind.is_some(),
                    has_layer_output_scale: layer_output_scale.is_some(),
                    ple_weight_kind: match graph_ple_weight_kind {
                        None => 0,
                        Some(DensePleWeightKind::Q4K) => 1,
                        Some(DensePleWeightKind::F32) => 2,
                    },
                    ple_dim: graph_ple_dim,
                    ple_gate_gelu: graph_ple_weight_kind == Some(DensePleWeightKind::Q4K)
                        && dense_ple_gate_gelu_enabled(true),
                    layer_output_scale_bits: layer_output_scale.map(f32::to_bits).unwrap_or(0),
                    unit_offset_post_attn_norm,
                    unit_offset_ffn_norm,
                    unit_offset_ple_norm,
                    hidden_dev,
                    attn_out_dev,
                    ple_input_dev,
                    normed_dev,
                    proj_dev,
                    gate_dev,
                    up_dev,
                    packed_gate: packed_q4_gate_up.map(|(gate, _)| gate).unwrap_or(0),
                    packed_up: packed_q4_gate_up.map(|(_, up)| up).unwrap_or(0),
                    packed_q4_down: packed_q4_down.unwrap_or(0),
                    packed_q6_down_qs,
                    packed_q6_down_d_super,
                    packed_q6_down_sub_scale,
                    o_q8_qs_dev,
                    o_q8_ds_dev,
                    q8_qs_dev,
                    q8_ds_dev,
                    down_q8_qs_dev,
                    down_q8_ds_dev,
                    o_weight_dev,
                    gate_weight_dev,
                    up_weight_dev,
                    down_weight_dev,
                    ple_gate_weight_dev,
                    ple_proj_weight_dev,
                    o_weight: o_weights.as_ptr() as usize,
                    gate_weight: gate_weights.as_ptr() as usize,
                    up_weight: up_weights.as_ptr() as usize,
                    down_weight: down_weights.as_ptr() as usize,
                    post_attn_norm_weight: post_attn_norm_dev as usize,
                    ffn_norm_weight: ffn_norm_dev as usize,
                    post_ffn_norm_weight: post_ffn_norm_dev as usize,
                    ple_gate_weight: ple_gate_weights
                        .map(|weights| weights.as_ptr() as usize)
                        .unwrap_or(0),
                    ple_proj_weight: ple_proj_weights
                        .map(|weights| weights.as_ptr() as usize)
                        .unwrap_or(0),
                    ple_post_norm_weight: ple_post_norm_dev as usize,
                };
                if let Some(layer_context) =
                    layer_segment_graph_context.filter(|_| cu71_layer_segment_graph_requested)
                {
                    let kv_len_dev = self.cu68_graph_kv_len_ptr()?;
                    let kv_cache_identity = layer_context
                        .kv_bucket
                        .k_identity()
                        .wrapping_mul(0x9e3779b185ebca87)
                        ^ layer_context.kv_bucket.v_identity().rotate_left(17);
                    let dense_graph_identity = o_weight_dev
                        ^ gate_weight_dev.rotate_left(7)
                        ^ up_weight_dev.rotate_left(13)
                        ^ down_weight_dev.rotate_left(29);
                    let layer_key = LayerSegmentGraphKey {
                        layer_idx: layer_context.layer_idx,
                        n_embd,
                        q_rows: layer_context.q_rows,
                        kv_dim: layer_context.kv_dim,
                        num_heads: layer_context.num_heads,
                        num_kv_heads: layer_context.num_kv_heads,
                        head_dim: layer_context.head_dim,
                        rope_theta_bits: layer_context.rope_theta.to_bits(),
                        norm_eps_bits: norm_eps.to_bits(),
                        attention_scale_bits: layer_context.attention_scale.to_bits(),
                        q_quant: layer_context.q_quant,
                        k_quant: layer_context.k_quant,
                        v_quant: layer_context.v_quant,
                        o_quant: crate::runtime::mtp_verify::GGML_Q4_K,
                        gate_quant: crate::runtime::mtp_verify::GGML_Q4_K,
                        up_quant: crate::runtime::mtp_verify::GGML_Q4_K,
                        down_quant,
                        q_carrier_dev: layer_context.q_carrier_dev,
                        k_carrier_dev: layer_context.k_f16_dev,
                        v_carrier_dev: layer_context.v_f16_dev,
                        k_f16_dev: layer_context.k_f16_dev,
                        v_f16_dev: layer_context.v_f16_dev,
                        kv_cache_identity,
                        kv_bucket: LayerSegmentKvBucketKey::from_bucket_view(
                            layer_context.kv_bucket,
                        ),
                        kv_len_dev,
                        attn_out_dev,
                        hidden_dev,
                        normed_dev,
                        proj_dev,
                        gate_dev,
                        up_dev,
                        q_norm_hash: layer_context.q_norm_hash,
                        k_norm_hash: layer_context.k_norm_hash,
                        post_attn_norm_hash: post_attn_norm_weight
                            .map(f32_key)
                            .map(|key| key.bit_hash)
                            .unwrap_or(0),
                        ffn_norm_hash: f32_key(ffn_norm_weight).bit_hash,
                        post_ffn_norm_hash: post_ffn_norm_weight
                            .map(f32_key)
                            .map(|key| key.bit_hash)
                            .unwrap_or(0),
                        q_weight_identity: layer_context.q_weight_identity,
                        k_weight_identity: layer_context.k_weight_identity,
                        v_weight_identity: layer_context.v_weight_identity,
                        o_weight_identity: o_weight_dev,
                        gate_weight_identity: gate_weight_dev,
                        up_weight_identity: up_weight_dev,
                        down_weight_identity: down_weight_dev,
                        packed_gate_identity: packed_q4_gate_up.map(|(gate, _)| gate).unwrap_or(0),
                        packed_up_identity: packed_q4_gate_up.map(|(_, up)| up).unwrap_or(0),
                        packed_down_identity: packed_q4_down
                            .or_else(|| packed_q6_down.map(|(qs, _, _)| qs))
                            .unwrap_or(0),
                        global_attention: true,
                        has_ple: graph_ple_weight_kind.is_some(),
                        has_layer_output_scale: layer_output_scale.is_some(),
                        has_post_attn_norm: post_attn_norm_weight.is_some(),
                        has_post_ffn_norm: post_ffn_norm_weight.is_some(),
                        ffn_uses_gelu,
                        q8dot_qkv: true,
                        q8dot_o: o_q8dot,
                        q8dot_gate_up,
                        q8dot_down,
                    };
                    let capture_inputs = Cu71LayerSegmentCaptureInputs {
                        qkv_ready: layer_context.q_carrier_dev != 0
                            && layer_context.k_f16_dev != 0
                            && layer_context.v_f16_dev != 0,
                        attention_ready: attn_out_dev != 0,
                        dense_ready: raw_weights_resident,
                        long_kv_split_preferred: layer_context.long_kv_split_preferred,
                        would_allocate_during_capture: false,
                        q_carrier_dev: layer_context.q_carrier_dev,
                        k_carrier_dev: layer_context.k_f16_dev,
                        v_carrier_dev: layer_context.v_f16_dev,
                        kv_cache_identity,
                        attn_out_dev,
                        hidden_dev,
                        dense_graph_identity,
                    };
                    match cu71_layer_segment_graph_step(
                        true,
                        capture_inputs,
                        layer_key,
                        &mut self.cu71_layer_segment_graph_warmed,
                        &self.cu71_layer_segment_graphs,
                    ) {
                        Cu71LayerSegmentGraphStep::Disabled => {}
                        Cu71LayerSegmentGraphStep::Rejected(reason) => {
                            if tuning::cu71_layer_segment_graph_trace_enabled() {
                                eprintln!(
                                    "[cu71 layer-segment-graph] state=rejected reason={reason:?} layer={}",
                                    layer_context.layer_idx
                                );
                            }
                        }
                        Cu71LayerSegmentGraphStep::Warm => {
                            if tuning::cu71_layer_segment_graph_trace_enabled() {
                                eprintln!(
                                    "[cu71 layer-segment-graph] state=warm layer={}",
                                    layer_context.layer_idx
                                );
                            }
                        }
                        Cu71LayerSegmentGraphStep::Capture => {
                            if tuning::cu71_layer_segment_graph_trace_enabled() {
                                eprintln!(
                                    "[cu71 layer-segment-graph] state=capture layer={}",
                                    layer_context.layer_idx
                                );
                            }
                            self.ensure_q4k_gemv_module()?;
                            unsafe {
                                self.api.stream_begin_capture(self.stream)?;
                            }
                            let capture_result = self.launch_dense_chain_graph_ops(
                                o_weights,
                                gate_weights,
                                up_weights,
                                down_weights,
                                down_quant,
                                post_attn_norm_weight,
                                ffn_norm_weight,
                                post_ffn_norm_weight,
                                o_cols,
                                n_ff,
                                n_embd,
                                norm_eps,
                                unit_offset_post_attn_norm,
                                unit_offset_ffn_norm,
                                hidden_dev,
                                attn_out_dev,
                                normed_dev,
                                proj_dev,
                                ffn_uses_gelu,
                                graph_ple_weight_kind,
                                if graph_ple_weight_kind == Some(DensePleWeightKind::Q4K) {
                                    ple_gate_weights
                                } else {
                                    None
                                },
                                if graph_ple_weight_kind == Some(DensePleWeightKind::Q4K) {
                                    ple_proj_weights
                                } else {
                                    None
                                },
                                if graph_ple_weight_kind.is_some() {
                                    ple_post_norm_weight
                                } else {
                                    None
                                },
                                graph_ple_weight_kind.map(|_| ple_input_dev),
                                graph_ple_weight_kind.map(|_| gate_dev),
                                ple_gate_weight_dev,
                                ple_proj_weight_dev,
                                graph_ple_dim,
                                unit_offset_ple_norm,
                                layer_output_scale,
                                None,
                                &mut trace_stage,
                            );
                            if let Err(err) = capture_result {
                                unsafe {
                                    let _ = self.api.stream_end_capture(self.stream);
                                }
                                return Err(err);
                            }
                            let graph = unsafe { self.api.stream_end_capture(self.stream)? };
                            let exec = unsafe { self.api.graph_instantiate(graph)? };
                            self.cu71_layer_segment_graphs.insert(
                                layer_key,
                                SparseMoeGraph {
                                    graph: graph as usize,
                                    exec: exec as usize,
                                },
                            );
                            let graph = self.cu71_layer_segment_graphs.get(&layer_key).ok_or_else(
                                || "missing cu71 layer segment CUDA graph".to_string(),
                            )?;
                            unsafe {
                                self.api
                                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
                            }
                            cu71_layer_segment_graph_launched = true;
                        }
                        Cu71LayerSegmentGraphStep::Replay => {
                            if tuning::cu71_layer_segment_graph_trace_enabled() {
                                eprintln!(
                                    "[cu71 layer-segment-graph] state=replay layer={}",
                                    layer_context.layer_idx
                                );
                            }
                            let graph = self.cu71_layer_segment_graphs.get(&layer_key).ok_or_else(
                                || "missing cu71 layer segment CUDA graph".to_string(),
                            )?;
                            unsafe {
                                self.api
                                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
                            }
                            cu71_layer_segment_graph_launched = true;
                        }
                    }
                }
                if !cu71_layer_segment_graph_launched && !cu71_layer_segment_graph_requested {
                    if let Some(graph) = self.cu69_dense_chain_graphs.get(&key) {
                        if tuning::cu69_dense_chain_graph_trace_enabled() {
                            eprintln!("[cu69 dense-chain-graph] state=replay");
                        }
                        unsafe {
                            self.api
                                .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
                        }
                        cu69_dense_chain_graph_launched = true;
                    } else if self.cu69_dense_chain_graph_warmed.contains(&key) {
                        if tuning::cu69_dense_chain_graph_trace_enabled() {
                            eprintln!("[cu69 dense-chain-graph] state=capture");
                        }
                        self.ensure_q4k_gemv_module()?;
                        unsafe {
                            self.api.stream_begin_capture(self.stream)?;
                        }
                        let capture_result = self.launch_dense_chain_graph_ops(
                            o_weights,
                            gate_weights,
                            up_weights,
                            down_weights,
                            down_quant,
                            post_attn_norm_weight,
                            ffn_norm_weight,
                            post_ffn_norm_weight,
                            o_cols,
                            n_ff,
                            n_embd,
                            norm_eps,
                            unit_offset_post_attn_norm,
                            unit_offset_ffn_norm,
                            hidden_dev,
                            attn_out_dev,
                            normed_dev,
                            proj_dev,
                            ffn_uses_gelu,
                            graph_ple_weight_kind,
                            if graph_ple_weight_kind == Some(DensePleWeightKind::Q4K) {
                                ple_gate_weights
                            } else {
                                None
                            },
                            if graph_ple_weight_kind == Some(DensePleWeightKind::Q4K) {
                                ple_proj_weights
                            } else {
                                None
                            },
                            if graph_ple_weight_kind.is_some() {
                                ple_post_norm_weight
                            } else {
                                None
                            },
                            graph_ple_weight_kind.map(|_| ple_input_dev),
                            graph_ple_weight_kind.map(|_| gate_dev),
                            ple_gate_weight_dev,
                            ple_proj_weight_dev,
                            graph_ple_dim,
                            unit_offset_ple_norm,
                            layer_output_scale,
                            None,
                            &mut trace_stage,
                        );
                        if let Err(err) = capture_result {
                            unsafe {
                                let _ = self.api.stream_end_capture(self.stream);
                            }
                            return Err(err);
                        }
                        let graph = unsafe { self.api.stream_end_capture(self.stream)? };
                        let exec = unsafe { self.api.graph_instantiate(graph)? };
                        self.cu69_dense_chain_graphs.insert(
                            key,
                            SparseMoeGraph {
                                graph: graph as usize,
                                exec: exec as usize,
                            },
                        );
                        let graph = self
                            .cu69_dense_chain_graphs
                            .get(&key)
                            .ok_or_else(|| "missing cu69 dense chain CUDA graph".to_string())?;
                        unsafe {
                            self.api
                                .graph_launch(graph.exec as *mut libc::c_void, self.stream)?;
                        }
                        cu69_dense_chain_graph_launched = true;
                    } else {
                        if tuning::cu69_dense_chain_graph_trace_enabled() {
                            eprintln!("[cu69 dense-chain-graph] state=warm");
                        }
                        self.cu69_dense_chain_graph_warmed.insert(key);
                    }
                }
            } else if tuning::cu69_dense_chain_graph_trace_enabled() {
                eprintln!("[cu69 dense-chain-graph] state=skip_raw_weight_temp");
            }
        } else if tuning::cu69_dense_chain_graph_trace_enabled()
            && dense_chain_graph_allowed
            && tuning::cu69_dense_chain_graph_enabled()
            && !ple_graph_supported
        {
            eprintln!("[cu69 dense-chain-graph] state=skip_ple_tail");
        }

        if !cu69_dense_chain_graph_launched && !cu71_layer_segment_graph_launched {
            self.q4k_dev_input_to_dev(o_weights, n_embd, o_cols / 256, attn_out_dev, proj_dev)?;
            self.trace_dense_stage(trace_call, "cuda-dense-chain", "o_proj", &mut trace_stage)?;
            let ffn_norm_dev = self.resident_f32_ptr(ffn_norm_weight)?;
            let mut prequantized_q8 = None;
            if let Some(weight) = post_attn_norm_weight {
                let norm_weight_dev = self.resident_f32_ptr(weight)?;
                if dense_combined_norms_enabled(true) {
                    if dense_q8dot_gate_up_enabled(ffn_uses_gelu) {
                        let q8_qs_dev = self.compute_gate_ptrs_ptr(n_embd)?;
                        let q8_ds_dev =
                            self.compute_up_ptrs_ptr((n_embd / 32) * std::mem::size_of::<f32>())?;
                        self.launch_rms_norm_add_then_rms_norm_q8_1_f32(
                            proj_dev,
                            norm_weight_dev,
                            hidden_dev,
                            ffn_norm_dev,
                            normed_dev,
                            q8_qs_dev,
                            q8_ds_dev,
                            norm_eps,
                            n_embd,
                            unit_offset_post_attn_norm,
                            unit_offset_ffn_norm,
                        )?;
                        prequantized_q8 = Some((q8_qs_dev, q8_ds_dev));
                        self.trace_dense_stage(
                            trace_call,
                            "cuda-dense-chain",
                            "post_attn_resid_norm+ffn_pre_norm+input_q8_quant",
                            &mut trace_stage,
                        )?;
                    } else {
                        self.launch_rms_norm_add_then_rms_norm_f32(
                            proj_dev,
                            norm_weight_dev,
                            hidden_dev,
                            ffn_norm_dev,
                            normed_dev,
                            norm_eps,
                            n_embd,
                            unit_offset_post_attn_norm,
                            unit_offset_ffn_norm,
                        )?;
                        self.trace_dense_stage(
                            trace_call,
                            "cuda-dense-chain",
                            "post_attn_resid_norm+ffn_pre_norm",
                            &mut trace_stage,
                        )?;
                    }
                } else {
                    self.launch_rms_norm_add_f32_inplace(
                        proj_dev,
                        norm_weight_dev,
                        hidden_dev,
                        norm_eps,
                        n_embd,
                        unit_offset_post_attn_norm,
                    )?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "post_attn_resid_norm",
                        &mut trace_stage,
                    )?;
                    self.launch_rms_norm_f32(
                        hidden_dev,
                        ffn_norm_dev,
                        normed_dev,
                        norm_eps,
                        n_embd,
                        unit_offset_ffn_norm,
                    )?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "ffn_pre_norm",
                        &mut trace_stage,
                    )?;
                }
            } else {
                self.launch_add_f32_inplace(hidden_dev, proj_dev, n_embd)?;
                self.trace_dense_stage(
                    trace_call,
                    "cuda-dense-chain",
                    "post_attn_resid_norm",
                    &mut trace_stage,
                )?;
                self.launch_rms_norm_f32(
                    hidden_dev,
                    ffn_norm_dev,
                    normed_dev,
                    norm_eps,
                    n_embd,
                    unit_offset_ffn_norm,
                )?;
                self.trace_dense_stage(
                    trace_call,
                    "cuda-dense-chain",
                    "ffn_pre_norm",
                    &mut trace_stage,
                )?;
            }
            self.qwen35_expert_dev_input_to_dev(
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                n_ff,
                n_embd,
                normed_dev,
                proj_dev,
                prequantized_q8,
                ffn_uses_gelu,
            )?;
            self.trace_dense_stage(trace_call, "cuda-dense-chain", "ffn", &mut trace_stage)?;
            if let Some(weight) = post_ffn_norm_weight {
                let norm_weight_dev = self.resident_f32_ptr(weight)?;
                self.launch_rms_norm_add_f32_inplace(
                    proj_dev,
                    norm_weight_dev,
                    hidden_dev,
                    norm_eps,
                    n_embd,
                    unit_offset_ffn_norm,
                )?;
            } else {
                self.launch_add_f32_inplace(hidden_dev, proj_dev, n_embd)?;
            }
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-chain",
                "post_ffn_resid_norm",
                &mut trace_stage,
            )?;

            if let (
                Some(ple_gate_weights),
                Some(ple_proj_weights),
                Some(ple_post_norm_weight),
                Some(ple_input),
            ) = (
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
            ) {
                let ple_bytes = std::mem::size_of_val(ple_input);
                let ple_input_dev = if let Some(offset) = ple_input_device_offset {
                    self.gemma_ple_base_slice_ptr(offset, ple_dim)?
                } else {
                    attn_out_dev
                };
                let ple_gate_dev = self.compute_mid_a_ptr(ple_bytes)?;
                if ple_input_device_offset.is_some() {
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "ple_base_slice",
                        &mut trace_stage,
                    )?;
                } else {
                    unsafe {
                        self.api.memcpy_htod_async(
                            ple_input_dev,
                            ple_input.as_ptr().cast::<libc::c_void>(),
                            ple_bytes,
                            self.stream,
                        )?;
                    }
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "ple_h2d",
                        &mut trace_stage,
                    )?;
                }
                let mut ple_megakernel_done = false;
                match ple_weight_kind
                    .ok_or_else(|| "dense attention+FFN PLE weight kind missing".to_string())?
                {
                    DensePleWeightKind::Q4K => {
                        if dense_ple_gate_gelu_enabled(true) {
                            self.launch_q4k_gemv_gelu_mul_to_dev(
                                ple_gate_weights,
                                ple_dim,
                                n_embd / 256,
                                hidden_dev,
                                ple_input_dev,
                                ple_gate_dev,
                            )?;
                            self.trace_dense_stage(
                                trace_call,
                                "cuda-dense-chain",
                                "ple_gate+gelu",
                                &mut trace_stage,
                            )?;
                        } else {
                            self.q4k_dev_input_to_dev(
                                ple_gate_weights,
                                ple_dim,
                                n_embd / 256,
                                hidden_dev,
                                ple_gate_dev,
                            )?;
                            self.trace_dense_stage(
                                trace_call,
                                "cuda-dense-chain",
                                "ple_gate",
                                &mut trace_stage,
                            )?;
                            self.launch_gelu_mul(ple_gate_dev, ple_input_dev, ple_dim)?;
                            self.trace_dense_stage(
                                trace_call,
                                "cuda-dense-chain",
                                "ple_gelu",
                                &mut trace_stage,
                            )?;
                        }

                        if cu62_ple_megakernel()
                            && dense_q4k_gemv_q8dot_enabled(n_embd >= 1024 && (ple_dim / 256) >= 4)
                        {
                            let proj_blocks_per_row = ple_dim / 256;
                            let qs_dev = self.compute_input_ptr(proj_blocks_per_row * 256)?;
                            let ds_dev = self.compute_aux_output_ptr(
                                proj_blocks_per_row * 8 * std::mem::size_of::<f32>(),
                            )?;
                            self.launch_quantize_q8_1_by_32(
                                ple_gate_dev,
                                qs_dev,
                                ds_dev,
                                proj_blocks_per_row * 256,
                            )?;
                            let weights_dev = self.resident_q4k_weights_ptr(ple_proj_weights)?;
                            let norm_weight_dev = self.resident_f32_ptr(ple_post_norm_weight)?;
                            self.launch_q4k_ple_megakernel_m1(
                                hidden_dev,
                                proj_dev,
                                weights_dev,
                                qs_dev,
                                ds_dev,
                                norm_weight_dev,
                                norm_eps,
                                n_embd as u32,
                                proj_blocks_per_row as u32,
                                unit_offset_ple_norm,
                            )?;
                            ple_megakernel_done = true;
                            self.trace_dense_stage(
                                trace_call,
                                "cuda-dense-chain",
                                "ple_megakernel_m1",
                                &mut trace_stage,
                            )?;
                        }
                        if !ple_megakernel_done {
                            self.q4k_dev_input_to_dev(
                                ple_proj_weights,
                                n_embd,
                                ple_dim / 256,
                                ple_gate_dev,
                                proj_dev,
                            )?;
                            self.trace_dense_stage(
                                trace_call,
                                "cuda-dense-chain",
                                "ple_proj",
                                &mut trace_stage,
                            )?;
                        }
                    }
                    DensePleWeightKind::F32 => {
                        let gate_dev = self.resident_f32_weights_ptr_from_le_bytes(
                            ple_gate_weights,
                            "dense attention+FFN PLE gate",
                        )?;
                        self.sgemm_device(gate_dev, ple_dim, n_embd, hidden_dev, 1, ple_gate_dev)?;
                        self.launch_gelu_mul(ple_gate_dev, ple_input_dev, ple_dim)?;
                        self.trace_dense_stage(
                            trace_call,
                            "cuda-dense-chain",
                            "ple_gate_f32+gelu",
                            &mut trace_stage,
                        )?;
                        let proj_weights_dev = self.resident_f32_weights_ptr_from_le_bytes(
                            ple_proj_weights,
                            "dense attention+FFN PLE proj",
                        )?;
                        self.sgemm_device(
                            proj_weights_dev,
                            n_embd,
                            ple_dim,
                            ple_gate_dev,
                            1,
                            proj_dev,
                        )?;
                        self.trace_dense_stage(
                            trace_call,
                            "cuda-dense-chain",
                            "ple_proj_f32",
                            &mut trace_stage,
                        )?;
                    }
                }
                if !ple_megakernel_done {
                    let norm_weight_dev = self.resident_f32_ptr(ple_post_norm_weight)?;
                    self.launch_rms_norm_add_f32_inplace(
                        proj_dev,
                        norm_weight_dev,
                        hidden_dev,
                        norm_eps,
                        n_embd,
                        unit_offset_ple_norm,
                    )?;
                    self.trace_dense_stage(
                        trace_call,
                        "cuda-dense-chain",
                        "ple_post_norm",
                        &mut trace_stage,
                    )?;
                }
            }

            // cu44 step 20: Gemma4 layer_output_scale 의 device-side apply.
            // host 의 apply_layer_output_scale_inplace (decode_inference.rs:653) 를
            // device 에서 동일 transform — chain end 의 hidden_dev *= scale.
            // chain path 시 caller (decode_inference.rs) 는 host apply 를 skip 해야.
            if let Some(scale) = layer_output_scale {
                self.launch_scale_f32_inplace(hidden_dev, scale, n_embd)?;
            }
        }
        // cu59 axis A — kernel 구간 끝. D2H 구간 시작.
        if let Some(start) = t_kernel_start {
            let kernel_us = start.elapsed().as_micros() as u64;
            super::chain_diag_bridge::stash_kernel_us(kernel_us);
        }
        let t_d2h_start = if diag_active {
            Some(std::time::Instant::now())
        } else {
            None
        };
        if !skip_d2h_hidden {
            unsafe {
                self.api.memcpy_dtoh_async(
                    hidden.as_mut_ptr().cast::<libc::c_void>(),
                    hidden_dev,
                    hidden_bytes,
                    self.stream,
                )?;
            }
        }
        if trace_call.is_some() {
            self.trace_dense_stage(
                trace_call,
                "cuda-dense-chain",
                "dtoh_hidden",
                &mut trace_stage,
            )?;
            if let (Some(call), Some(total)) = (trace_call, trace_total) {
                eprintln!(
                    "[cuda-dense-chain] call={call} stage=total ms={:.3}",
                    total.elapsed().as_secs_f64() * 1000.0
                );
            }
        } else if !skip_d2h_hidden {
            // intermediate layer (skip_d2h_hidden=true): sync 도 건너뜀.
            // 다음 chain call 의 same stream 안의 H2D/kernel 이 자동 ordering.
            // 마지막 layer 또는 host 결과 필요 시 caller 가 sync.
            self.stream_synchronize()?;
        }
        // cu59 axis A — D2H 구간 끝.
        if let Some(start) = t_d2h_start {
            let d2h_us = start.elapsed().as_micros() as u64;
            super::chain_diag_bridge::stash_d2h_us(d2h_us);
        }
        // cu44 diag: chain end + sync 후 host scratch.hidden print.
        // env ON/OFF 양쪽 path 모두 비교 가능 (carrier vs compute_full_gate).
        if !skip_d2h_hidden && std::env::var("RNB_CU44_DIAG_CHAIN_END").is_ok() {
            let n = hidden.len().min(4);
            eprintln!(
                "[cu44 diag chain_end carrier={}] host[..{n}]={:?}",
                hidden_carrier_dev.is_some(),
                &hidden[..n]
            );
        }
        if hidden_carrier_dev.is_some()
            && !skip_d2h_hidden
            && std::env::var("RNB_CU44_DIAG_CHAIN_END_CARRIER_RE").is_ok()
        {
            let mut dbg = vec![0.0f32; hidden_bytes / 4];
            unsafe {
                self.api.memcpy_dtoh_async(
                    dbg.as_mut_ptr().cast::<libc::c_void>(),
                    hidden_dev,
                    hidden_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            let n = hidden.len().min(4);
            let max_diff = hidden
                .iter()
                .zip(dbg.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[cu44 diag chain_end] host[..{n}]={:?} carrier_re[..{n}]={:?} max_diff={}",
                &hidden[..n],
                &dbg[..n],
                max_diff
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_attention_qkv(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        n_embd: usize,
        input: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<(), String> {
        let blocks_per_row = n_embd / 256;
        let q_bytes = q_rows * std::mem::size_of::<f32>();
        let kv_bytes = kv_rows * std::mem::size_of::<f32>();
        let q_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_dev = self.compute_mid_b_ptr(kv_bytes)?;
        let v_dev = self.compute_output_ptr(kv_bytes)?;
        if dense_q8dot_qkv_enabled(true) {
            let (qs, ds) = quantize_q8_1_by_32(input, blocks_per_row);
            let qs_dev = self.compute_input_ptr(qs.len())?;
            let ds_dev = self.compute_aux_output_ptr(std::mem::size_of_val(ds.as_slice()))?;
            unsafe {
                self.api.memcpy_htod_async(
                    qs_dev,
                    qs.as_ptr().cast::<libc::c_void>(),
                    qs.len(),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    ds_dev,
                    ds.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(ds.as_slice()),
                    self.stream,
                )?;
            }
            self.launch_q4k_qkv_gemv_q8dot_to_dev(
                q_weights,
                k_weights,
                v_weights,
                q_rows,
                kv_rows,
                blocks_per_row,
                qs_dev,
                ds_dev,
                q_dev,
                k_dev,
                v_dev,
            )?;
        } else {
            let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input),
                    self.stream,
                )?;
            }
            self.launch_q4k_qkv_gemv_to_dev(
                q_weights,
                k_weights,
                v_weights,
                q_rows,
                kv_rows,
                blocks_per_row,
                input_dev,
                q_dev,
                k_dev,
                v_dev,
            )?;
        }

        unsafe {
            self.api.memcpy_dtoh_async(
                q.as_mut_ptr().cast::<libc::c_void>(),
                q_dev,
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                k.as_mut_ptr().cast::<libc::c_void>(),
                k_dev,
                kv_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v.as_mut_ptr().cast::<libc::c_void>(),
                v_dev,
                kv_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(())
    }

    // cu41 Phase 1 step 4: QKV gemv 의 device input variant.
    // input 이 device (RMS norm carrier output) 에 이미 있어서 H2D 제거.
    // q8dot path 는 host CPU quantize 필요 → fallback `launch_q4k_qkv_gemv_to_dev`
    // 만 사용. output q/k/v 는 host (즉시 D2H + sync, 다음 step 에서 device out).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_attention_qkv_with_device_input(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        n_embd: usize,
        input_dev: u64,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<(), String> {
        let blocks_per_row = n_embd / 256;
        let q_bytes = q_rows * std::mem::size_of::<f32>();
        let kv_bytes = kv_rows * std::mem::size_of::<f32>();
        let q_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_dev = self.compute_mid_b_ptr(kv_bytes)?;
        let v_dev = self.compute_output_ptr(kv_bytes)?;
        self.launch_q4k_qkv_gemv_to_dev(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            input_dev,
            q_dev,
            k_dev,
            v_dev,
        )?;
        unsafe {
            self.api.memcpy_dtoh_async(
                q.as_mut_ptr().cast::<libc::c_void>(),
                q_dev,
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                k.as_mut_ptr().cast::<libc::c_void>(),
                k_dev,
                kv_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v.as_mut_ptr().cast::<libc::c_void>(),
                v_dev,
                kv_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(())
    }

    // cu29 Phase 2: Llama / Mistral hd=128 path — Q4K QKV GEMV + GPU RoPE +
    // K/V f16 pack 한 번에. host RoPE round-trip 제거 (Q는 RoPE 적용된 f32 반환,
    // K/V 는 f16 bits 반환 → KvCache append_bits_range 에 바로 쓸 수 있음).
    //
    // decode (seq_len=1) only. prefill batch 는 별도 wrapper (Phase 2-B 후속).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_attention_qkv_rope_hd128_decode(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        n_embd: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
        pos_start: usize,
        input: &[f32],
        q_rope: &mut [f32],
        k_bits: &mut [u16],
        v_bits: &mut [u16],
    ) -> Result<(), String> {
        // shape invariants — Llama / Mistral hd=128.
        if q_rows != num_heads * 128 || kv_rows != num_kv_heads * 128 {
            return Err(format!(
                "dense_q4k_attention_qkv_rope_hd128_decode: shape mismatch \
                 q_rows={q_rows} kv_rows={kv_rows} num_heads={num_heads} \
                 num_kv_heads={num_kv_heads}"
            ));
        }

        let blocks_per_row = n_embd / 256;
        let q_bytes = q_rows * std::mem::size_of::<f32>();
        let kv_bytes_f32 = kv_rows * std::mem::size_of::<f32>();
        let q_bits_bytes = q_rows * std::mem::size_of::<u16>();
        let kv_bits_bytes = kv_rows * std::mem::size_of::<u16>();
        let _ = q_bits_bytes;

        // Q4K QKV GEMV → device q_dev/k_dev/v_dev (f32). cu31: prefill wrapper 와
        // 동일 path (q4k_batch_dev_input_to_dev 3번) 로 통일. cu29 PoC 에서 사용한
        // launch_q4k_qkv_gemv_q8dot_to_dev (Q+K+V fused single-token kernel) 가
        // prefill 의 batch GEMV kernel 과 결과 미세 차이 가능성 — decode wrapper
        // 정확도 깨짐 진단 가설 E.
        let q_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_dev = self.compute_mid_b_ptr(kv_bytes_f32)?;
        let v_dev = self.compute_output_ptr(kv_bytes_f32)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.q4k_batch_dev_input_to_dev(q_weights, q_rows, blocks_per_row, 1, input_dev, q_dev)?;
        self.q4k_batch_dev_input_to_dev(k_weights, kv_rows, blocks_per_row, 1, input_dev, k_dev)?;
        self.q4k_batch_dev_input_to_dev(v_weights, kv_rows, blocks_per_row, 1, input_dev, v_dev)?;

        // RoPE sin/cos device table (cached).
        let rope = self.rope_table_ptrs(128, 1, pos_start, rope_theta)?;

        // RoPE output buffers (f32 q, f16 bits k/v).
        let q_rope_dev = self.compute_q_rope_out_ptr(q_bytes)?;
        let k_bits_dev = self.compute_k_bits_out_ptr(kv_bits_bytes)?;
        let v_bits_dev = self.compute_v_bits_out_ptr(kv_bits_bytes)?;

        self.launch_qk_rope_neox_hd128_f16kv(
            q_dev,
            k_dev,
            v_dev,
            rope.sin_ptr,
            rope.cos_ptr,
            q_rope_dev,
            k_bits_dev,
            v_bits_dev,
            1,
            num_heads,
            num_kv_heads,
        )?;

        // D2H — q_rope f32, k_bits/v_bits f16 (KV cache append 에 바로 사용).
        unsafe {
            self.api.memcpy_dtoh_async(
                q_rope.as_mut_ptr().cast::<libc::c_void>(),
                q_rope_dev,
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                k_bits_dev,
                kv_bits_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                v_bits_dev,
                kv_bits_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(())
    }

    // cu30 Phase 2c: Llama / Mistral hd=128 multi-token (prefill) 변형. 매
    // token Q4K QKV batch GEMV → GPU RoPE → f16 K/V pack → D2H. host RoPE
    // round-trip + per-token f32→f16 변환 제거. prefill 까지 device pipeline
    // 적용하면 KvCache 가 GPU RoPE 결과로 일관 (cu29 정확도 깨짐 fix).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_q4k_attention_qkv_rope_hd128_prefill(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        n_embd: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
        pos_start: usize,
        seq_len: usize,
        input: &[f32],
        q_rope: &mut [f32],
        k_bits: &mut [u16],
        v_bits: &mut [u16],
    ) -> Result<(), String> {
        if q_rows != num_heads * 128 || kv_rows != num_kv_heads * 128 {
            return Err(format!(
                "dense_q4k_attention_qkv_rope_hd128_prefill: shape mismatch \
                 q_rows={q_rows} kv_rows={kv_rows} num_heads={num_heads} \
                 num_kv_heads={num_kv_heads}"
            ));
        }
        if seq_len == 0 {
            return Ok(());
        }
        let blocks_per_row = n_embd / 256;
        let input_bytes = seq_len * n_embd * std::mem::size_of::<f32>();
        let q_bytes = seq_len * q_rows * std::mem::size_of::<f32>();
        let kv_bytes_f32 = seq_len * kv_rows * std::mem::size_of::<f32>();
        let kv_bits_bytes = seq_len * kv_rows * std::mem::size_of::<u16>();

        // H2D input — multi-token f32 활성화.
        let input_dev = self.compute_input_ptr(input_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
        }

        // Q4K QKV batch GEMV — 3 separate calls (q4k_batch_dev_input_to_dev
        // 가 q8dot 또는 raw kernel 자동 선택). cu27 pin/unpin race fix 가
        // multi-register OOM 보호 (직접 buffer slot 별 register 라 충돌 없음).
        let q_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_dev = self.compute_mid_b_ptr(kv_bytes_f32)?;
        let v_dev = self.compute_output_ptr(kv_bytes_f32)?;
        self.q4k_batch_dev_input_to_dev(
            q_weights,
            q_rows,
            blocks_per_row,
            seq_len,
            input_dev,
            q_dev,
        )?;
        self.q4k_batch_dev_input_to_dev(
            k_weights,
            kv_rows,
            blocks_per_row,
            seq_len,
            input_dev,
            k_dev,
        )?;
        self.q4k_batch_dev_input_to_dev(
            v_weights,
            kv_rows,
            blocks_per_row,
            seq_len,
            input_dev,
            v_dev,
        )?;

        // RoPE sin/cos table — seq_len 만큼 한 번에 (cached).
        let rope = self.rope_table_ptrs(128, seq_len, pos_start, rope_theta)?;

        // RoPE output buffers (multi-token).
        let q_rope_dev = self.compute_q_rope_out_ptr(q_bytes)?;
        let k_bits_dev = self.compute_k_bits_out_ptr(kv_bits_bytes)?;
        let v_bits_dev = self.compute_v_bits_out_ptr(kv_bits_bytes)?;

        // RoPE kernel — grid.x = seq_len. kernel 은 token 마다 다른 rope idx
        // 사용 (token * half + tid). cu29 에서 non-neox pair 수정 적용됨.
        self.launch_qk_rope_neox_hd128_f16kv(
            q_dev,
            k_dev,
            v_dev,
            rope.sin_ptr,
            rope.cos_ptr,
            q_rope_dev,
            k_bits_dev,
            v_bits_dev,
            seq_len,
            num_heads,
            num_kv_heads,
        )?;

        // D2H — q_rope f32, k/v_bits f16. host RoPE 적용 안 함 (이미 GPU RoPE
        // 적용된 결과), host f32→f16 변환 안 함 (이미 packed).
        unsafe {
            self.api.memcpy_dtoh_async(
                q_rope.as_mut_ptr().cast::<libc::c_void>(),
                q_rope_dev,
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                k_bits_dev,
                kv_bits_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                v_bits_dev,
                kv_bits_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(())
    }
}
