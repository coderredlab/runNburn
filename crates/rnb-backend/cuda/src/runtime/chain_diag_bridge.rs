//! cu59 axis A — chain function timing bridge.
//!
//! rnb-llm 의 chain_diag 가 직접 import 할 수 있도록 rnb-backend/cuda
//! 안에서 kernel / d2h timing 을 process-wide AtomicU64 로 stash.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static KERNEL_US: AtomicU64 = AtomicU64::new(0);
static D2H_US: AtomicU64 = AtomicU64::new(0);

/// RNB_CU58_DIAG_CHAIN env 가 active 인지 (1 또는 2).
///
/// cu59 step 2 fix: OnceLock cache — 매 호출마다 std::env::var 실행 시
/// chain function 진입 (Gemma E2B 의 경우 layer × token = 315회) 마다
/// syscall. chain_diag.rs 쪽과 동일한 cache 패턴 적용해서 비활성 시
/// zero-overhead 보장.
pub fn is_active() -> bool {
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| {
        matches!(
            std::env::var("RNB_CU58_DIAG_CHAIN").as_deref(),
            Ok("1") | Ok("2")
        )
    })
}

pub fn stash_kernel_us(us: u64) {
    KERNEL_US.store(us, Ordering::Relaxed);
}

pub fn stash_d2h_us(us: u64) {
    D2H_US.store(us, Ordering::Relaxed);
}

pub fn drain_kernel_us() -> u64 {
    KERNEL_US.swap(0, Ordering::Relaxed)
}

pub fn drain_d2h_us() -> u64 {
    D2H_US.swap(0, Ordering::Relaxed)
}
