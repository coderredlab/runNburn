# RNB 패킹 노트

## 2026-04-26 Qwen3.5 0.8B 후속 작업

Flip4 혼합 순서 벤치에서는 runNburn standalone `.rnb`가 llama.cpp GGUF보다
decode는 빠르지만, 아직 MNN보다는 느리다.

| 런타임 | 양자화 | Decode 중앙값 |
|---|---|---:|
| MNN | Q4/INT4 계열 | `34.80 tok/s` |
| runNburn `.rnb` | Q4_K_M | `18.00 tok/s` |
| llama.cpp GGUF | Q4_K_M | `13.23 tok/s` |
| LiteRT-LM | INT8/q8 | `8.47 tok/s` |

다음 패킹 작업은 전역 숫자 버전보다 컨테이너 종류와 내부 layout identity를
분리해서 진화시키는 쪽으로 정한다. 아직 배포 전이므로 기존 임시 파일 호환보다
새 기준의 단순함을 우선한다.

- 사용자에게 보이는 산출물은 `.rnb` 하나로 유지한다.
- Dense/standalone 컨테이너 magic은 `RNBD`로 둔다.
- Dense tensor layout 변화는 `QuantType`이 곧 layout identity가 되게 한다.
- GGUF sidecar는 비교/진단용이지 배포 경로가 아니다.
- MoE decode section 컨테이너 magic은 `RNBM`으로 둔다.
- MoE 내부 layout 변화는 `SectionId`와 section schema가 identity가 되게 한다.
- Standalone metadata manifest magic은 dense 계열임을 드러내는 `RNBDMT01`로 둔다.
- `packing_version` 같은 전역 숫자 축은 두지 않는다.
- 새 패킹은 기존 의미를 덮어쓰지 말고 새 `QuantType` 또는 새 section으로 추가한다.

우선 최적화 대상:

- 큰 엔진 재작성 전에 decode-side packed weight layout과 cache locality를 먼저 개선한다.
- 단독 smoke 숫자보다 MNN sustained decode를 실전 목표로 둔다.
- 패킹 변경은 같은 mixed-order 벤치 방식으로 검증한다.
