# runNburn

runNburn is a Rust inference runtime for running quantized GGUF language models
under tight RAM and VRAM budgets. Its focus is practical offloading: keep large
models usable on constrained devices without expanding quantized weights into
oversized resident copies.

> **Status:** pre-1.0 and under active development. The CPU path is the default.
> CUDA and Metal acceleration are model-aware and evolving; Vulkan, OpenCL, and
> MediaTek paths are experimental. Backend and model coverage is not uniform yet.

## What runNburn provides

- **Direct GGUF inference** — product entry points load GGUF files through mmap; no conversion step or generated sidecar is required.
- **Memory-aware execution** — host residency, sparse-expert page caches, staging buffers, and accelerated caches are bounded from detected or application-supplied memory budgets.
- **Quantized kernels** — common GGML formats include Q2_K through Q6_K, Q4_0, and Q8_0, with x86 and ARM NEON implementations.
- **Hybrid and sparse models** — architecture-specific paths cover attention, GatedDeltaNet, Mamba-style recurrence, and sparse MoE execution.
- **OpenAI-compatible serving** — Chat Completions and Responses APIs, SSE streaming, function tools, JSON Schema structured output, stored responses, and conversations.
- **Application embedding** — a Rust API and an Android-oriented C ABI ship alongside the `runNburn` CLI.

The project optimizes for capability first: models and context sizes that do not
fit a conventional fully resident runtime should still have a useful execution
path. Performance claims use same-device reference engines under matched
conditions.

## Runtime status

| Runtime path | Status | Notes |
|---|---|---|
| CPU on Linux/macOS | Default | x86 native kernels with portable fallbacks |
| CPU on Android ARM64 | Supported | ARM NEON; Android benchmarks must run through ADB rather than Termux SSH |
| NVIDIA CUDA | Active | Model-specific device residency, quantized kernels, and CPU fallback |
| Apple Metal | Active | Apple Silicon model-specific acceleration and CPU fallback |
| Vulkan | Experimental | Mobile defaults to CPU; Vulkan requires explicit diagnostic opt-in |
| OpenCL / MediaTek | Experimental | Buildable integration and diagnostic paths, not product defaults |

## Quick start

### Requirements

- Rust 2021 toolchain and Cargo
- A GGUF model supported by the selected runtime path
- Optional: CUDA toolkit for CUDA builds, Xcode toolchain for Metal builds, or Android NDK for Android builds

### Build the CPU CLI

```bash
cargo build --release -p rnb-cli --no-default-features --features cpu
```

Run a single prompt:

```bash
./target/release/runNburn /path/to/model.gguf "Explain why memory mapping helps large-model inference."
```

Omit the prompt for interactive input:

```bash
./target/release/runNburn /path/to/model.gguf
```

Set an explicit host working-set budget when the automatic policy is not appropriate:

```bash
./target/release/runNburn --ram-budget 16GiB /path/to/model.gguf "Hello"
```

Options for the direct CLI must appear before the GGUF path. Binary suffixes (`KiB`, `MiB`, `GiB`, `TiB`) and decimal suffixes (`KB`, `MB`, `GB`, `TB`) are accepted.

## OpenAI-compatible server

Each server process loads one GGUF model and serializes inference through a bounded worker queue. Start it with:

```bash
RNB_API_KEY=local-secret ./target/release/runNburn serve \
  --host 127.0.0.1 \
  --port 8000 \
  --model-name local-model \
  --ram-budget 16GiB \
  --response-cache-budget 2GiB \
  /path/to/model.gguf
```

Point an OpenAI client at `http://127.0.0.1:8000/v1`.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-secret",
)

response = client.responses.create(
    model="local-model",
    input="Summarize why bounded caches matter in one sentence.",
)
print(response.output_text)
```

A minimal request without an SDK:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","input":"Hello from runNburn"}'
```

### Implemented API surface

| Method | Endpoint | Notes |
|---|---|---|
| `GET` | `/v1/models` | Returns the model served by this process |
| `POST` | `/v1/chat/completions` | Non-streaming and SSE streaming; tools and structured output |
| `POST` | `/v1/responses` | Non-streaming and SSE streaming; tools, structured output, and stateful continuation |
| `GET`, `DELETE` | `/v1/responses/{id}` | In-memory stored response lookup and deletion |
| `GET` | `/v1/responses/{id}/input_items` | Cursor pagination with `order`, `after`, and `limit` |
| `POST` | `/v1/conversations` | Creates an in-memory conversation |
| `GET`, `DELETE` | `/v1/conversations/{id}` | Conversation lookup and deletion |

Unknown fields, unsupported parameters, invalid methods, and missing resources
return OpenAI-shaped JSON errors instead of being silently ignored.
Compatibility covers the surface above; this is not a complete OpenAI API
implementation.

### Continuation and cache behavior

`previous_response_id` and conversation references preserve canonical input and
output history. Stored responses and conversations are process-local, have a
30-day TTL, and can be evicted earlier by `--response-cache-budget`; they are not
a durable database.

CPU-only builds also retain compact Q8 KV data and the associated SSM sequence
state. When a snapshot is present, a continuation reuses the cached prefix.
Snapshot entries are evicted before canonical history, so a cache miss falls
back to a full replay without losing conversation content. Accelerated builds
replay canonical history because durable device sequence snapshots are not
enabled yet.

`RNB_API_KEY` is optional. Keep the default loopback bind for local use. If the
server is exposed beyond localhost, set an API key and terminate TLS in a reverse
proxy; the built-in server provides HTTP, not TLS.

Show the complete server options with:

```bash
./target/release/runNburn serve --help
```

## Accelerated builds

CPU fallback remains enabled in the accelerated binaries.

```bash
# Linux with NVIDIA CUDA
cargo build --release -p rnb-cli \
  --no-default-features --features cpu,cuda

# macOS with Apple Metal
cargo build --release -p rnb-cli \
  --no-default-features --features cpu,metal
```

The repository build matrix checks the supported feature combinations:

```bash
scripts/check_build_matrix.sh
```

Backend availability does not imply that every model operator runs on that
backend. Unsupported operations either fall back through the runtime boundary or
fail explicitly, depending on the execution contract.

## Models and file formats

The `runNburn` product path accepts **GGUF** model files. Standalone `.rnb` model
input is a legacy diagnostic format and is rejected by the CLI and HTTP server.
RNBC packed sidecars and conversion tools remain available for explicit
development experiments, but they are not discovered or regenerated
automatically by product loading.

Architecture-aware paths exist for:

- Llama-family and Phi models
- Gemma and Gemma 4
- Qwen2 and Qwen3.5 dense, hybrid, and MoE models
- Nemotron-H MoE
- HY3 sparse MoE
- GLM-DSA

Exact tensor layouts, quantization formats, context features, and accelerated
coverage vary by architecture. A recognized GGUF architecture is not a promise
that every community variant is supported.

## Memory policy

Without `--ram-budget`, runNburn detects physical RAM, reserves one quarter for
the operating system, KV cache, runtime buffers, and other processes, and uses
the remaining three quarters as the engine working-set budget. An explicit value
replaces that automatic budget.

The budget controls engine-owned host residency and file-backed sparse-expert
page caches. It is not an operating-system RSS limit: mapped weights, KV cache,
temporary buffers, runtime libraries, and unrelated process memory still
contribute to RSS.

On supported CUDA sparse-MoE paths, resident weight and hot-cache sizes are
derived from current free and total VRAM rather than fixed device-name presets.
Original quantized weights remain the source of truth; expanded resident F16/F32
projection copies are not a product default.

## Android and C ABI

Android builds require Android NDK 28 or newer, `cargo-ndk`, and the nightly
`aarch64-linux-android` Rust target. `cargo-ndk` discovers Android Studio's
latest installed NDK automatically; set `ANDROID_NDK_HOME` to select another
installation.

```bash
cargo install cargo-ndk
rustup target add --toolchain nightly aarch64-linux-android
rustup run nightly cargo ndk -t arm64-v8a -p 34 build --release \
  -p rnb-ffi --no-default-features --features cpu
```

The shared library is written to
`target/aarch64-linux-android/release/librnb_ffi.so`. The versioned public
header is [`crates/rnb-ffi/include/rnb.h`](crates/rnb-ffi/include/rnb.h).

A minimal pull-generation loop:

```c
#include "rnb.h"
#include <stdio.h>

int main(void) {
    RnbContext* ctx = rnb_load_with_ram_budget("model.gguf", 4ULL << 30);
    if (ctx == NULL || rnb_submit(ctx, "What is memory-mapped I/O?") != 0) {
        return 1;
    }

    const char* token;
    while ((token = rnb_next_token(ctx)) != NULL) {
        fputs(token, stdout);
    }

    rnb_free(ctx);
    return 0;
}
```

`rnb_submit()` applies the Qwen-style chat wrapper used by the mobile
integration. Use `rnb_submit_raw()` when the application renders the GGUF chat
template itself or targets another model family. The header's
`RNB_API_VERSION_*` macros track the `rnb-ffi` package version. The C ABI is
pre-1.0, so minor releases may change it.

For device benchmarks, use ADB shell. Termux SSH places workloads in Android's
`/moderate` cpuset on tested devices and can make CPU measurements 3–4 times
slower; use it only for correctness checks and file transfer.

## Performance

runNburn does not publish cross-device headline numbers: model files, prompts,
context lengths, memory budgets, backends, and thermals can change results by
orders of magnitude. Use `rnb-llm-bench` with the sample prompts in `prompts/`
and compare engines on the same device under matched conditions.

A useful comparison repeats warm runs, reports the median, and checks generated
output for semantic quality before accepting a speedup. Keep backend defaults
unchanged unless the experiment is explicitly testing a documented override.

## Workspace layout

```text
crates/
  rnb-core/          Shared tensor, dtype, quantization, and IR contracts
  rnb-cpu/           CPU kernels, quantization, packed GEMM/GEMV
  rnb-loader/        GGUF loading and explicit diagnostic sidecar formats
  rnb-llm/           Model semantics, generation, tokenizer, sampler, KV cache
  rnb-runtime/       Runtime facade and backend assembly
  rnb-platform/      Platform facts and policy inputs
  rnb-memory/        Residency, tiering, and byte-budgeted caches
  rnb-scheduler/     Placement, admission, and request scheduling
  rnb-backend/       CPU, CUDA, Metal, Vulkan, OpenCL, and MediaTek backends
  rnb-models/        Architecture-specific Gemma, Nemotron, and Qwen modules
  rnb-mtp/           Multi-token prediction support
  rnb-cli/           Product CLI and OpenAI-compatible server
  rnb-ffi/           C ABI for application embedding
  rnb-tools/         Conversion, diagnostics, probes, and benchmarks
```

The intended ownership boundaries are documented in
[`docs/superpowers/specs/2026-04-28-runtime-crate-boundaries-design.md`](docs/superpowers/specs/2026-04-28-runtime-crate-boundaries-design.md).

## Development checks

```bash
cargo fmt --all -- --check
cargo test --workspace
scripts/check_build_matrix.sh
```

Android runtime measurements require a physical device and ADB. CUDA and Metal
performance claims likewise require the target hardware; CPU-only compilation is
a correctness check, not a substitute benchmark.

## License

Licensed under the [Apache License 2.0](LICENSE).
