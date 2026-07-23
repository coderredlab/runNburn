//! 예전 `rnb_core::ir::Graph`용 백엔드 실행 인터페이스.
//!
//! 이 모듈의 `Scheduler`와 `Backend`는 `rnb-scheduler`/`rnb-backend-api`의
//! 모델 레이어 배치·런타임 백엔드 계약이 아니라, core IR 그래프를 순서대로
//! 실행하는 옛 경로야. `rnb-cpu` 통합 테스트와 예전 그래프 실행 확인에서
//! 아직 공개 API로 쓰이므로 이름을 유지한다.

pub mod scheduler;
pub mod traits;

pub use scheduler::Scheduler;
pub use traits::Backend;
