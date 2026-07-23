//! cu59 axis A — chain function 호출별 sub-phase timing diagnostic.
//!
//! Env: RNB_CU58_DIAG_CHAIN=1 (aggregate), =2 (verbose).
//! 비활성 시 zero-overhead (atomic check fast-path).

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum DiagMode {
    Off,
    Aggregate,
    Verbose,
}

fn diag_mode() -> DiagMode {
    static MODE: OnceLock<DiagMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match crate::engine::policy::env_string("RNB_CU58_DIAG_CHAIN").as_deref() {
            Some("1") => DiagMode::Aggregate,
            Some("2") => DiagMode::Verbose,
            _ => DiagMode::Off,
        }
    })
}

#[inline]
pub(in crate::engine) fn is_active() -> bool {
    diag_mode() != DiagMode::Off
}

/// Sub-phase identifiers (aggregate counter index).
#[derive(Debug, Clone, Copy)]
pub(in crate::engine) enum Phase {
    HelperArgs = 0,
    Acquire = 1,
    WeightResolve = 2,
    NormSlice = 3,
    Kernel = 4,
    D2H = 5,
    /// cu59 axis A Task 7: signal_ctx 생성 + chain_function_active 매 layer 호출.
    /// chain function 진입 *전* 의 host-side cost — 기존 6 phase 가 측정 밖이던 구간.
    SignalEval = 6,
}

const PHASE_COUNT: usize = 7;
const PHASE_NAMES: [&str; PHASE_COUNT] = [
    "helper_args",
    "acquire",
    "weight_resolve",
    "norm_slice",
    "kernel",
    "d2h",
    "signal_eval",
];

static SUMS_US: [AtomicU64; PHASE_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static PHASE_US: Cell<[u64; PHASE_COUNT]> = const { Cell::new([0; PHASE_COUNT]) };
    static CALL_LAYER: Cell<usize> = const { Cell::new(0) };
    static CALL_TOKEN: Cell<usize> = const { Cell::new(0) };
}

pub(in crate::engine) fn stash_phase_us(phase: Phase, us: u64) {
    if !is_active() {
        return;
    }
    PHASE_US.with(|p| {
        let mut arr = p.get();
        arr[phase as usize] = us;
        p.set(arr);
    });
}

pub(in crate::engine) fn stash_call_context(layer_idx: usize, token_idx: usize) {
    if !is_active() {
        return;
    }
    CALL_LAYER.with(|l| l.set(layer_idx));
    CALL_TOKEN.with(|t| t.set(token_idx));
}

pub(in crate::engine) fn flush_call() {
    if !is_active() {
        return;
    }
    // rnb-backend/cuda 의 chain_diag_bridge 에서 kernel / d2h 시간 drain.
    // rnb-llm 은 rnb-backend-cuda 직접 의존 없음 → rnb-runtime 의 cuda_inference
    // 경유로 re-export 된 bridge 사용 (cuda_runtime::chain_diag_bridge).
    #[cfg(feature = "cuda")]
    {
        use super::super::cuda_runtime::chain_diag_bridge;
        let kernel_us = chain_diag_bridge::drain_kernel_us();
        let d2h_us = chain_diag_bridge::drain_d2h_us();
        PHASE_US.with(|p| {
            let mut arr = p.get();
            arr[Phase::Kernel as usize] = kernel_us;
            arr[Phase::D2H as usize] = d2h_us;
            p.set(arr);
        });
    }

    let phase_us = PHASE_US.with(|p| p.get());
    let layer = CALL_LAYER.with(|l| l.get());
    let token = CALL_TOKEN.with(|t| t.get());
    record_call(layer, token, &phase_us);
    PHASE_US.with(|p| p.set([0; PHASE_COUNT]));
}

fn record_call(layer_idx: usize, token_idx: usize, phase_us: &[u64; PHASE_COUNT]) {
    match diag_mode() {
        DiagMode::Off => {}
        DiagMode::Aggregate => {
            for (i, &us) in phase_us.iter().enumerate() {
                SUMS_US[i].fetch_add(us, Ordering::Relaxed);
            }
            CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        DiagMode::Verbose => {
            eprintln!(
                "RNB_CU58_DIAG_CHAIN VERBOSE layer={} token={} \
                 helper_args={} acquire={} weight_resolve={} \
                 norm_slice={} kernel={} d2h={} signal_eval={}",
                layer_idx,
                token_idx,
                phase_us[0],
                phase_us[1],
                phase_us[2],
                phase_us[3],
                phase_us[4],
                phase_us[5],
                phase_us[6],
            );
            for (i, &us) in phase_us.iter().enumerate() {
                SUMS_US[i].fetch_add(us, Ordering::Relaxed);
            }
            CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// 세션 종료 시 aggregate 출력. crate-public — bench binary 의 main 끝에서
/// 명시적으로 호출 (Engine 에 Drop impl 박으면 기존 functional-update Engine
/// 생성 test 가 partial-move 로 E0509 깨짐 → Drop 대신 explicit 호출).
pub fn dump_aggregate() {
    if diag_mode() == DiagMode::Off {
        return;
    }
    let calls = CALL_COUNT.load(Ordering::Relaxed);
    if calls == 0 {
        return;
    }
    eprintln!("RNB_CU58_DIAG_CHAIN AGGREGATE:");
    eprintln!("  total_calls={}", calls);
    let mut total_us = 0u64;
    for (i, name) in PHASE_NAMES.iter().enumerate() {
        let sum = SUMS_US[i].load(Ordering::Relaxed);
        let avg = if calls > 0 { sum / calls as u64 } else { 0 };
        total_us += sum;
        eprintln!("  {}_us_avg={}", name, avg);
    }
    eprintln!("  total_us_sum={}", total_us);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_call_no_op_when_diag_off() {
        // OnceLock 으로 cached 라 env 변경 무시. 단 default 환경 (env unset) 에서
        // record_call 이 counter 증가 안 시키는지만 검증.
        let before = CALL_COUNT.load(Ordering::Relaxed);
        record_call(0, 0, &[10, 20, 30, 40, 50, 60, 70]);
        let after = CALL_COUNT.load(Ordering::Relaxed);
        assert_eq!(before, after, "Off mode 에서 counter 증가 안 함");
    }
}
