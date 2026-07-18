# runNburn 개발 로드맵

Rust LLM 추론 엔진. Android ARM NEON 타겟. Qwen3.5-0.8B Q4_K_M 기준.

---

## Phase 1: 기반 구조 (3/24)

핵심 crate 4개를 설계 → 구현.

| Crate | 역할 | 상태 |
|-------|------|------|
| rnb-core | IR 그래프, Tensor, Backend trait, mmap | ✅ 완료 |
| rnb-cpu | CPU 커널 (matmul, norm, activation, rope, attention) | ✅ 완료 |
| rnb-loader | GGUF 파서, mmap zero-copy weight 로딩 | ✅ 완료 |
| rnb-llm | 추론 엔진 (engine, generate, kv_cache, sampler, tokenizer) | ✅ 완료 |

**결과:** F32 순수 Rust 추론 동작. LLaMA 아키텍처 지원.

## Phase 2: 정확도 (3/24-25)

토크나이저 + 추론 정확도 확보.

- ✅ SentencePiece BPE (LLaMA 계열)
- ✅ GPT-2 byte-level BPE (Qwen 계열)
- ✅ KV cache 통합
- ✅ K-quant dequant 수정 (Q4_K, Q5_K, Q6_K)
- ✅ RoPE position 버그 수정
- ✅ GGUF tensor offset 계산 수정
- ✅ 출력 품질: 가비지 → 올바른 영어 문장

**결과:** Qwen2 기반 모델에서 정확한 추론.

## Phase 3: Qwen3.5 아키텍처 (3/25)

Qwen3.5의 하이브리드 Attention + GatedDeltaNet(GDN) 구조 구현.

- ✅ GDN 커널 (softplus, sigmoid, l2_norm, conv1d, delta_net_scan)
- ✅ SSM 메타데이터 파싱
- ✅ 하이브리드 forward pass (Attention + GDN 교대)
- ✅ Partial RoPE (MRoPE)
- ✅ Attention bias (Qwen2)

**결과:** Qwen3.5-0.8B 정상 추론. ~2-3 tok/s (F32, x86).

## Phase 4: ARM NEON 최적화 (3/25-27)

Android ARM64에서 decode 속도를 끌어올림.

### 4a. 기본 NEON SIMD (3/25)
- ✅ aarch64 크로스 컴파일 + QEMU 테스트 셋업
- ✅ Q4_0, Q8_0 vdotq_s32 int8 dot product
- ✅ Q4_K, Q5_K int8 dot product (256-element K-quant)
- ✅ 자동 int8 dispatch (gemv_vec)
- **결과:** 10.6 tok/s (Flip 4). llama.cpp 대비 +38%.

### 4b. Q4_K Repacked GEMV 시도 (3/25)
- ✅ 8-row interleaved weight repacking
- ✅ 1×8 NEON 커널 구현
- ⚠️ 정확도 문제 발생 → **SUPERSEDED** by Q4_0 변환
- 📄 `specs/2026-03-25-q4k-repacked-gemv-design.md`

### 4c. Q4_K Assembly GEMV 시도 (3/27)
- ✅ cc crate + .S 파일 빌드 인프라
- ✅ Indexed sdot assembly 커널 구현
- ⚠️ 성능 향상 없음 (Q4_K format 복잡도가 asm 이점 상쇄)
- **ABANDONED** → Q4_0 변환이 더 효과적
- 📄 `specs/2026-03-27-q4k-asm-gemv-design.md`

### 4d. Q4_0 변환 + 커널 최적화 (3/27) ← 성공 경로
- ✅ Q4_K → Q4_0 format 변환 (load time)
- ✅ Q5_K, Q6_K → Q4_0 변환 (모든 K-quant 통합)
- ✅ rayon 스레드 수 ARM big.LITTLE 자동 감지
- ✅ gemv_vec_q8k → Q4_0 fast path 라우팅
- ✅ rayon chunk 크기 최적화
- ✅ Q8 input 사전 양자화 (레이어당 1회, GEMV에 재사용)
- ✅ SSM alpha/beta에 Q8 재사용 + fallback
- ✅ Fused gate+up GEMV (seq_len>1 correctness fix 포함)
- ✅ sched_setaffinity 자동 big core pinning
- **결과:** 26.5 tok/s. llama.cpp 대비 **1.9×**.

### 4e. GemvPool (custom thread pool) 시도 (3/27)
- ✅ Spin-wait thread pool 구현 (gemv_pool.rs)
- ✅ L2 cache 감지 + tiled dispatch
- ✅ Engine 통합 (26개 call site)
- ⚠️ rayon 대비 개선 없음 → **ABANDONED**
- 💡 교훈: rayon overhead는 전체의 ~5%. 병목은 DRAM bandwidth contention.
- 💡 교훈: 1T에서 이미 compute-bound (83% 효율). 커널 최적화보다 근본적 변화 필요.
- 📄 `specs/2026-03-27-gemv-pool-tiling-design.md`

### 4f. 기타 실패한 시도 (3/27)
- ❌ i8mm (SMMLA): GEMV에서 50% 효율 → 이점 없음 (GEMM용)
- ❌ 4-row multi-row kernel: cache pressure 증가로 regression
- ❌ All-core output pool: LITTLE core가 느려서 regression
- ❌ Prefetch 추가: 효과 없음 (hardware prefetcher가 이미 커버)

---

## 현재 상태 (2026-03-27)

```
Flip 4 (Snapdragon 8+ Gen 1), Qwen3.5-0.8B Q4_K_M:

runNburn:    26.5 tok/s (순수 decode, 4T big core)
llama.cpp:   14.8 tok/s (1T), 14.0 tok/s (기본)
MNN:         ~15-32 tok/s (추정, startup 분리 어려움)

runNburn vs llama.cpp: 1.9× 빠름
runNburn vs MNN:       동급 또는 소폭 우위
```

### 최적화 이력 (tok/s, Flip 4)
```
F32 baseline          →  ~3 tok/s
NEON int8 Q4_K        →  10.6
Q4_0 변환             →  14.5
big.LITTLE 4T tuning  →  19.8
rayon chunk 최적화    →  24.8
Q8 input 재사용       →  25.2
fused gate+up + affinity → 26.5  ← 현재
```

## 미해결 / 향후 방향

- **1T 성능 6% gap vs llama.cpp** — 커널 내부 개선 여지 (f16→f32 변환, bounds check 등)
- **4T scaling 1.8×** — DRAM bandwidth contention, 하드웨어 한계에 가까움
- **output_logits (23%)** — 248K vocab GEMV. approximate top-k 등으로 줄일 수 있으나 정확도 trade-off
- **더 큰 모델 (3B)** — 메모리 최적화 필요 (Q4_0 변환 시 원본 + 변환 둘 다 메모리 사용)
- **Prefill 최적화** — 현재 decode에만 집중, prefill은 미최적화
- **WASM 타겟** — 브라우저 실행 지원 (secondary target)
