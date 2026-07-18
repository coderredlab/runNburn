# Runtime Crate Boundaries Design

## Goal

Make the final runtime architecture small, explicit, and platform-buildable.
`rnb-llm` must stop being the owner of OS policy, GPU policy, memory tiering,
and backend-specific runtime details. It should orchestrate LLM semantics only.

This design supersedes the earlier internal `rnb-llm::platform` direction as the
long-term target. Internal modules may be used only as temporary migration
staging points, not as the final ownership boundary.

## Final Crate Map

Backend crates are physically grouped under `crates/rnb-backend/` while keeping
their package names stable:

```text
crates/rnb-backend/api     -> rnb-backend-api
crates/rnb-backend/cpu     -> rnb-backend-cpu
crates/rnb-backend/cuda    -> rnb-backend-cuda
crates/rnb-backend/vulkan  -> rnb-backend-vulkan
crates/rnb-backend/opencl  -> rnb-backend-opencl
crates/rnb-backend/metal   -> rnb-backend-metal
crates/rnb-backend/qnn     -> rnb-backend-qnn
crates/rnb-backend/mediatek -> rnb-backend-mediatek
```

Do not collapse these into one implementation crate. The directory grouping is
for ownership readability; package-level dependency and feature boundaries stay
separate.

Experiment and analysis tools are physically grouped under `crates/rnb-tools/`.
They are not runtime ownership boundaries. For example,
`crates/rnb-tools/moe-cache-sim` owns the `rnb-moe-cache-sim` package and backs
the compatibility CLI binary `rnb-moe-tier`; only policies promoted into live
runtime execution should move from that tool crate into `rnb-memory` or
`rnb-scheduler`.

`crates/rnb-tools/dev-cli` owns the `rnb-dev-tools` package and backs debug,
probe, and benchmark binaries such as `rnb-llm-bench`, `rnb-mem-api-probe`, and
`rnb-storage-iobench`. These binaries may depend broadly for diagnosis, but
their code must not become the production runtime boundary. Production
entrypoints stay in `rnb-cli`. Tool binaries must use purpose-based names;
numbered `rnb-debugN` binaries are not an accepted long-term boundary. If a
diagnostic no longer has a clear owner or current use, delete it instead of
renaming it into the new tool crate.

`crates/rnb-tools/convert` owns the `rnb-convert` package and binary. It may
depend on `rnb-cpu`, `rnb-loader`, and `.rnb` packing/container code because it
is an offline artifact builder, not a production inference entrypoint.

### `rnb-core`

Pure shared primitives:

- tensor shape and dtype primitives
- quant format identity
- `.rnb` packed tensor storage type identity
- layout identity
- shared error/result types
- model-independent math/data contracts

Forbidden:

- OS detection
- GPU backend names or types
- environment variable policy
- tokenizer, sampler, or runtime session logic

### `rnb-model-ir`

Model structure and execution-neutral identifiers:

- model graph
- layer id
- tensor id
- weight id
- expert id
- attention/FFN/GDN/MoE structural descriptions
- weight storage descriptors

Forbidden:

- GGUF parser internals
- CUDA/Vulkan/OpenCL runtime types
- memory residency decisions
- scheduler decisions

### `rnb-loader`

File format loading:

- GGUF to `rnb-model-ir`
- RNB to `rnb-model-ir`
- tensor metadata and storage descriptor extraction
- standalone `.rnb` container header/table read/write
- mmap-backed `.rnb` packed model reader

Forbidden:

- execution backend selection
- CPU/GPU dispatch
- RAM/VRAM cache policy
- tokenizer-driven generation logic

### `rnb-cpu`

CPU execution and quantized kernels:

- implements CPU kernels for graph execution
- owns CPU quantization/dequantization helpers
- Q4_K/Q5_K/Q6_K/Q8_0 packed weight layout transforms
- aarch64 NEON/i8mm GEMV/GEMM kernels
- activation quantization helpers consumed by CPU LLM execution
- packed-kernel tuning policy

Forbidden:

- `.rnb` container header/table read/write
- GGUF/RNB model loading
- tokenizer/sampler/KV cache
- runtime backend registration

### `rnb-platform`

Platform facts and platform-only policy:

- OS: Linux, Android, Windows, macOS, iOS
- CPU architecture: x86_64, aarch64
- form factor metadata: desktop, mobile, server
- runtime target construction
- CPU feature detection
- Android affinity/cpuset policy
- Linux procfs/sysfs probing

Forbidden:

- CUDA/Vulkan kernel calls
- model-specific semantics
- cache eviction policy
- tokenizer/sampler/KV cache

### `rnb-memory`

Memory residency and tiering:

- RAM/VRAM/UFS/file/mmap descriptors
- memory budget model
- cache hit/miss accounting
- prefetch plans
- eviction policy
- PC MoE RAM to VRAM cache
- Android UFS to RAM strategy

Forbidden:

- CUDA kernel implementation
- Vulkan kernel implementation
- LLM model semantics
- tokenizer/sampler/KV cache

### `rnb-backend-api`

Backend-facing contracts only:

- `Backend` trait
- backend id/name
- backend capabilities
- supported op declarations
- execution request/result types
- backend error type
- memory requirement reporting

Forbidden:

- CUDA/Vulkan/OpenCL imports
- OS-specific probing
- model file parsing
- generation loop

### `rnb-backend-cpu`

CPU backend adapter:

- implements `rnb-backend-api`
- calls `rnb-cpu` kernels
- selects arch-specific kernel variants using `rnb-platform` facts

Forbidden:

- CUDA/Vulkan/OpenCL fallback
- tokenizer/sampler/KV cache
- file format parsing

### `rnb-backend-cuda`

CUDA backend:

- CUDA runtime entrypoints
- CUDA graph policy
- CUDA cache policy that is CUDA-specific
- CUDA env/tuning policy
- CUDA prefill/decode/MoE/GDN kernels
- CUDA tests

Forbidden:

- tokenizer/sampler/KV cache
- GGUF/RNB parser internals
- Vulkan/OpenCL fallback
- Android affinity policy

### `rnb-backend-vulkan`

Vulkan backend adapter:

- implements `rnb-backend-api`
- owns Vulkan runtime type exposure to upper layers
- owns Vulkan weight id construction
- owns Vulkan quant mapping
- owns Vulkan env/tuning policy
- owns the Vulkan kernel/runtime modules directly.

Forbidden:

- CUDA/OpenCL fallback
- tokenizer/sampler/KV cache
- model file parsing

### `rnb-backend-opencl`

OpenCL backend:

- implements `rnb-backend-api`
- reports capability and unsupported operations explicitly
- owns OpenCL runtime/kernels when an operation is implemented

Forbidden:

- fake execution fallback
- CUDA/Vulkan fallback hidden behind OpenCL selection
- tokenizer/sampler/KV cache

### `rnb-backend-qnn`

Qualcomm QNN/QAIRT backend adapter:

- implements `rnb-backend-api`
- owns QNN/QAIRT SDK loading and Android HTP runtime integration
- owns QNN graph/context binary cache interaction
- reports HTP/NPU capability explicitly
- returns explicit unsupported errors for CPU/GPU QNN modes that runNburn does not support

Forbidden:

- CUDA/Vulkan/Metal fallback
- silent CPU fallback behind a QNN success result
- tokenizer/sampler/KV cache
- model file parsing

### `rnb-backend-mediatek`

MediaTek NPU backend adapter:

- implements `rnb-backend-api`
- owns LiteRT NeuroPilot or MediaTek Neuron SDK loading and Android NPU runtime integration
- owns compiled graph/cache interaction for MediaTek NPU artifacts
- reports NPU capability explicitly
- returns explicit unsupported errors when no compiled graph/runtime is available

Forbidden:

- NNAPI diagnostic payloads becoming the product model format
- CUDA/Vulkan/Metal fallback
- silent CPU fallback behind a MediaTek NPU success result
- tokenizer/sampler/KV cache
- model file parsing

### `rnb-scheduler`

Execution placement:

- chooses backend per op/layer/tensor/expert
- combines backend capabilities, memory budget, and model IR
- creates prefetch plans
- creates cache residency plans
- owns PC/mobile/server policy composition

Forbidden:

- CUDA kernel implementation
- Vulkan kernel implementation
- tokenizer/sampler/KV cache internals
- file format parsing

### `rnb-runtime`

Runtime assembly and session lifecycle:

- builds runtime sessions from platform, memory, scheduler, model IR, and
  backend implementations
- owns backend registration
- validates feature/platform compatibility
- owns runtime-level errors for unsupported target/backend combinations

Forbidden:

- CUDA kernel implementation
- Vulkan kernel implementation
- tokenizer internals
- GGUF parser internals

### `rnb-llm`

LLM semantics:

- tokenizer integration
- sampler
- KV cache
- generation loop
- model-specific semantics such as Gemma/Qwen/GDN behavior
- high-level inference API

Allowed dependency:

- `rnb-runtime` through stable session/request APIs

Forbidden:

- `RNB_CUDA_*` parsing
- `target_os` / `target_arch` platform branching
- direct RAM/VRAM cache eviction policy
- backend-specific runtime type names

### `rnb-cli` and `rnb-ffi`

Entrypoints only:

- parse user options
- construct runtime configuration
- call `rnb-llm` or `rnb-runtime`
- expose Android/iOS/desktop API surfaces

`rnb-cli` owns only the product-facing inference binary: `runNburn`. Offline
conversion belongs to `rnb-convert` under `crates/rnb-tools/convert`. Debug,
probe, smoke, and benchmark binaries belong in `rnb-dev-tools` under
`crates/rnb-tools/dev-cli`.

Forbidden:

- backend kernel calls
- scheduler internals
- memory eviction logic
- model semantic decisions

## Dependency Direction

Allowed high-level direction:

```text
rnb-cli / rnb-ffi
        -> rnb-llm
        -> rnb-runtime
        -> rnb-scheduler
        -> rnb-backend-api

rnb-runtime
        -> rnb-platform
        -> rnb-memory
        -> rnb-model-ir

rnb-loader
        -> rnb-model-ir
        -> rnb-core

rnb-backend-cpu
        -> rnb-backend-api
        -> rnb-cpu
        -> rnb-platform

rnb-backend-cuda
        -> rnb-backend-api
        -> rnb-platform
        -> rnb-memory

rnb-backend-vulkan
        -> rnb-backend-api
        -> rnb-platform
        -> rnb-memory

rnb-backend-opencl
        -> rnb-backend-api
        -> rnb-platform
        -> rnb-memory

rnb-backend-metal
        -> rnb-backend-api
        -> rnb-platform
        -> rnb-memory

rnb-backend-qnn
        -> rnb-backend-api
        -> rnb-platform
        -> rnb-memory

rnb-backend-mediatek
        -> rnb-backend-api
        -> rnb-platform
        -> rnb-memory
```

Forbidden reverse dependencies:

- `rnb-core` must not depend on any runtime crate.
- `rnb-model-ir` must not depend on loader or backend crates.
- backend crates must not depend on `rnb-llm`.
- `rnb-llm` must not depend on concrete backend crates.
- `rnb-loader` must not depend on scheduler/runtime/backend crates.

## Platform Build Targets

The final build matrix must make valid combinations explicit:

```text
Linux x86_64 CPU:
  rnb-cli + rnb-llm + rnb-runtime + rnb-backend-cpu

Linux x86_64 CUDA:
  rnb-cli + rnb-llm + rnb-runtime + rnb-backend-cuda + rnb-backend-cpu

Linux x86_64 Vulkan:
  rnb-cli + rnb-llm + rnb-runtime + rnb-backend-vulkan + rnb-backend-cpu

Windows x86_64 CUDA:
  rnb-cli + rnb-llm + rnb-runtime + rnb-backend-cuda + rnb-backend-cpu

Android aarch64 CPU:
  rnb-ffi + rnb-llm + rnb-runtime + rnb-backend-cpu

Android aarch64 Vulkan:
  rnb-ffi + rnb-llm + rnb-runtime + rnb-backend-vulkan + rnb-backend-cpu

Android aarch64 QNN:
  rnb-ffi + rnb-llm + rnb-runtime + rnb-backend-qnn + rnb-backend-cpu

Android aarch64 MediaTek NPU:
  rnb-ffi + rnb-llm + rnb-runtime + rnb-backend-mediatek + rnb-backend-cpu

iOS aarch64 CPU:
  rnb-ffi + rnb-llm + rnb-runtime + rnb-backend-cpu

macOS aarch64 CPU:
  rnb-cli + rnb-llm + rnb-runtime + rnb-backend-cpu

macOS aarch64 Metal:
  rnb-cli + rnb-llm + rnb-runtime + rnb-backend-metal + rnb-backend-cpu

OpenCL:
  opt-in backend crate; unsupported operations must return explicit errors
```

Invalid combinations must fail clearly during build configuration or runtime
initialization:

- Android + CUDA
- iOS + Vulkan in the current design
- selected OpenCL execution without implemented OpenCL op support
- QNN HTP selected on non-Android target without an explicit host-tooling mode
- QNN CPU/GPU backend reported as HTP/NPU success
- CUDA env policy compiled into a non-CUDA backend
- `rnb-llm` direct dependency on concrete CUDA/Vulkan crates

## Feature Ownership

Features belong to crates that own the implementation:

- `rnb-backend-cuda/cuda`: CUDA runtime implementation at
  `crates/rnb-backend/cuda`.
- `rnb-backend-vulkan/vulkan`: Vulkan backend adapter at
  `crates/rnb-backend/vulkan`.
- `rnb-backend-opencl/opencl`: OpenCL backend adapter at
  `crates/rnb-backend/opencl`.
- `rnb-backend-metal/metal`: Metal backend adapter at `crates/rnb-backend/metal`.
- `rnb-backend-qnn/qnn`: Qualcomm QNN/QAIRT HTP backend adapter at
  `crates/rnb-backend/qnn`.
- `rnb-backend-mediatek/mediatek`: LiteRT NeuroPilot or MediaTek Neuron SDK
  backend adapter at `crates/rnb-backend/mediatek`.
- `rnb-runtime/cuda`: pulls in and registers `rnb-backend-cuda`.
- `rnb-runtime/vulkan`: pulls in and registers `rnb-backend-vulkan`.
- `rnb-runtime/opencl`: pulls in and registers `rnb-backend-opencl`.
- `rnb-runtime/metal`: pulls in and registers `rnb-backend-metal`.
- `rnb-runtime/qnn`: pulls in and registers `rnb-backend-qnn` for Android
  aarch64 HTP/NPU.
- `rnb-runtime/mediatek`: pulls in and registers `rnb-backend-mediatek` for
  Android aarch64 MediaTek NPU.
- `rnb-cli/cpu`, `rnb-cli/cuda`, `rnb-cli/vulkan`, `rnb-cli/opencl`,
  `rnb-cli/mediatek`:
  user-facing feature aliases. During migration they forward through
  `rnb-llm` so product entrypoints depend only on the LLM API crate.
- `rnb-ffi/cpu`, `rnb-ffi/vulkan`, `rnb-ffi/opencl`, `rnb-ffi/mediatek`:
  Android/iOS-facing feature aliases. Unsupported target/backend combinations
  must still fail through runtime target validation.
- `rnb-llm/cpu`, `rnb-llm/cuda`, `rnb-llm/vulkan`, `rnb-llm/opencl`,
  `rnb-llm/mediatek`: temporary bridge features that forward to `rnb-runtime`.
  The final target is to replace these with stable runtime session/config APIs
  so `rnb-llm` stops exposing backend implementation switches.

QNN and MediaTek NPU are not part of `GpuBackend`; future implementation should
add an NPU or generic accelerator selection path, or select their backend kinds
directly through runtime session configuration.

## Migration Order

### Phase 1: Contract Crates

Create small crates without moving large runtime bodies:

1. `rnb-platform`
2. `rnb-model-ir`
3. `rnb-memory`
4. `rnb-backend-api`

Move only pure types and policy parsers that do not require engine-private
types. Keep behavior unchanged.

### Phase 2: Runtime Assembly

Create `rnb-runtime` and make it own backend registration and platform/backend
compatibility validation. `rnb-llm` still calls old paths through adapters, but
new calls go through `rnb-runtime`.

### Phase 3: Backend Adapters

Create:

1. `rnb-backend-cpu`
2. `rnb-backend-vulkan`
3. `rnb-backend-cuda`
4. `rnb-backend-opencl`

Move adapter boundaries first. Move large backend implementation bodies only
after adapter tests compile and pass.

### Phase 4: Engine Slimming

Remove direct platform/backend policy from `rnb-llm`:

- no direct CUDA env parsing
- no direct Vulkan runtime type references
- no direct OS/arch cfg policy except unavoidable compile guards around FFI
- no memory tiering decisions

### Phase 5: Build Matrix Enforcement

Add CI/local commands for each supported platform/feature shape. Invalid
combinations should fail with clear messages.

Local verification starts with `scripts/check_build_matrix.sh`, which checks the
current migration feature aliases explicitly:

```text
host cli: cpu, cpu+cuda, cpu+vulkan, cpu+opencl
host dev tools: cpu
host runtime: cpu, cpu+cuda, cpu+vulkan, cpu+opencl
optional android ffi: cpu, cpu+vulkan
```

The script must not hide compiler output or continue after a failed case. Use
`scripts/check_build_matrix.sh --include-android` only when nightly Rust and the
Android target are installed locally.

## Acceptance Criteria

- `rnb-llm` has no direct `RNB_CUDA_*` parsing.
- `rnb-llm` has no direct Vulkan backend crate internals.
- `rnb-llm` does not depend on `rnb-backend-cuda`,
  `rnb-backend-vulkan`, or `rnb-backend-opencl`.
- CUDA/Vulkan/OpenCL backend implementations depend on `rnb-backend-api`, not on
  `rnb-llm`.
- Platform-specific builds can be expressed through top-level CLI/FFI/runtime
  features.
- Unsupported platform/backend combinations fail explicitly.
- Existing CPU-only build remains available with no GPU backend compiled.

## Non-Goals

- Do not rewrite kernels during boundary migration.
- Do not change model output quality as part of crate splitting.
- Do not introduce fake backend fallbacks.
- Do not move Gemma/Qwen/GDN semantics into platform or backend crates.
- Do not make desktop/mobile source roots.
