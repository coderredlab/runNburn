<p align="center">
  <img src="assets/runnburn-mark.png" alt="runNburn logo" width="148">
</p>

<h1 align="center">runNburn</h1>

<p align="center">
  <strong>Memory-aware GGUF inference for hardware with hard limits.</strong>
</p>

<p align="center">
  Run quantized language models across CPU, NVIDIA CUDA, Apple Metal, and Android<br>
  without expanding weights into oversized resident copies.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-EA3323?style=flat-square"></a>
  <img alt="Rust 2021" src="https://img.shields.io/badge/Rust-2021-111111?style=flat-square&logo=rust">
  <a href="#models-and-file-formats"><img alt="Model format: GGUF" src="https://img.shields.io/badge/model-GGUF-374151?style=flat-square"></a>
  <img alt="Status: pre-1.0" src="https://img.shields.io/badge/status-pre--1.0-D97706?style=flat-square">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#interactive-chat">Chat</a> ·
  <a href="#openai-compatible-server">Server</a> ·
  <a href="#backend-status">Backends</a> ·
  <a href="#models-and-file-formats">Models</a> ·
  <a href="#android-and-c-abi">Android</a> ·
  <a href="#license">License</a>
</p>

---

runNburn is a general Rust offloading runtime for quantized GGUF models too
large for fast memory. GGUF weights remain file-backed, host residency stays
bounded, and accelerator caches stay coordinated without requiring a converted
product model or expanded resident copy.

Model architecture determines semantics; detected RAM, VRAM, and backend
capability determine placement. A smaller machine may run more slowly, but the
product path does not silently requantize weights, alter router choices, or
attach generated sidecars to make a model fit.

> [!IMPORTANT]
> runNburn is pre-1.0 and under active development. CPU is the default path.
> CUDA and Metal acceleration are model-aware and evolving. Vulkan, OpenCL,
> and MediaTek paths are experimental; mobile Vulkan remains explicit opt-in.

## Why runNburn

| Capability | Design |
|---|---|
| **Direct GGUF execution** | Product entry points mmap GGUF weights as the source of truth. Conversion and generated sidecars are not part of product loading. |
| **Bounded offloading** | Host residency, sparse-expert pages, staging buffers, and accelerator caches scale from detected or supplied memory budgets. |
| **Model-aware execution** | Architecture paths cover dense attention, GatedDeltaNet, Mamba-style recurrence, and sparse MoE without changing pretrained routing semantics. |
| **Native quantized kernels** | Common GGML formats span Q2_K through Q6_K, Q4_0, and Q8_0 across x86, ARM NEON, CUDA, and Metal where implemented. |
| **One product contract** | The CLI, Rust API, Android C ABI, and OpenAI-compatible server share model-loading and memory-policy behavior. |

> [!NOTE]
> Correctness gates performance work. Comparisons use the same model, prompt,
> decode length, and device as the reference engine. A faster result that
> damages the response is not adopted.

### Product target

runNburn targets a personal, single-owner inference server on consumer
hardware. The primary optimization unit is one active generation: model
capacity under bounded RAM and VRAM, correctness, time to first token, prefill
and decode latency, continuation reuse, and predictable memory behavior.
Continuous batching, tenant isolation, requests-per-second throughput, and
distributed serving are non-goals.

## Backend status

| Runtime path | Status | Notes |
|---|---|---|
| CPU on Linux and macOS | **Default** | Native x86 kernels with portable fallbacks |
| CPU on Android ARM64 | **Supported** | ARM NEON; benchmark through ADB rather than Termux SSH |
| NVIDIA CUDA | **Active** | Model-specific device residency, quantized kernels, and CPU fallback |
| Apple Metal | **Active** | Apple Silicon model-specific acceleration and CPU fallback |
| Vulkan | Experimental | Desktop builds are available; mobile remains CPU-default and requires explicit opt-in |
| OpenCL and MediaTek | Experimental | Buildable diagnostic paths, not product defaults |

## Quick start

### Requirements

- Rust with Cargo
- A GGUF model supported by the selected runtime path
- Optional: CUDA toolkit, Xcode toolchain, or Android NDK for accelerated builds

### 1. Build the CPU CLI

```bash
cargo build --release -p rnb-cli --no-default-features --features cpu
```

### 2. Run a model

```bash
./target/release/runNburn /path/to/model.gguf \
  "Explain why memory mapping helps large-model inference."
```

Omit the prompt to start an interactive session:

```bash
./target/release/runNburn /path/to/model.gguf
```

### 3. Set a memory budget

```bash
./target/release/runNburn --ram-budget 16GiB \
  /path/to/model.gguf "Hello"
```

> [!TIP]
> Direct CLI options must appear before the GGUF path. Binary suffixes (`KiB`,
> `MiB`, `GiB`, `TiB`) and decimal suffixes (`KB`, `MB`, `GB`, `TB`) are
> accepted.

### Interactive chat

Load the model once and keep multi-turn conversation history in the CLI:

```bash
./target/release/runNburn chat \
  --system "Answer concisely." \
  --max-tokens 256 \
  /path/to/model.gguf
```

Responses stream as they are generated. Use `/clear` to reset conversation history,
`/set system <prompt>` to replace the system prompt, `/show system` to inspect it,
and `/bye` to exit. Run `runNburn chat --help` for sampling and memory options.

## OpenAI-compatible server

Each server process loads one GGUF model and serializes inference through a bounded worker queue. Start it with:

```bash
./target/release/runNburn serve \
  --host 127.0.0.1 \
  --port 8000 \
  --model-name local-model \
  --ram-budget 16GiB \
  --response-cache-budget 2GiB \
  --api-key-file /path/to/api-key \
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

Supported CPU, CUDA, and Metal paths retain KV and SSM sequence state. Exact
in-model and external MTP continuations also retain target and drafter state.
When a snapshot is present, a continuation reuses the cached prefix. Snapshot
entries are evicted before canonical history, so a cache miss or an unsupported
runtime falls back to a full replay without losing conversation content.

Authentication is optional only for loopback binds. A non-loopback bind is
rejected unless `--api-key-file` or `RNB_API_KEY` supplies a bearer key; the key
file takes precedence and must contain exactly one non-empty line. Terminate TLS
in a reverse proxy because the built-in server provides HTTP, not TLS.

`SIGINT`, `SIGTERM`, and `SIGHUP` stop new accepts, cancel active or queued
generation, release the loaded model, and exit after connection workers stop.

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

# Linux with experimental Vulkan
cargo build --release -p rnb-cli \
  --no-default-features --features cpu,vulkan
```

The repository build matrix checks the supported feature combinations:

```bash
scripts/check_build_matrix.sh
```

Backend availability does not imply that every model operator runs on that
backend. Unsupported operations either fall back through the runtime boundary or
fail explicitly, depending on the execution contract.

## Models and file formats

The `runNburn` product path accepts **GGUF** model files. Standalone `.rnb`
input is a retired legacy diagnostic format and is rejected by the CLI, Rust
loader, and HTTP server. Product loading does not discover, generate, or attach
converted model sidecars.

Architecture-aware paths exist for:

- Llama-family and Phi models
- Gemma and Gemma 4
- Qwen2 and Qwen3.5 dense, hybrid, and MoE models
- Nemotron-H MoE
- HY3 sparse MoE
- GLM-DSA

Exact tensor layouts, quantization formats, context features, and accelerated
coverage vary by architecture. Recognition does not imply full support. A GGUF
architecture may be recognized even when a particular community variant is unsupported.

## Memory policy

Without `--ram-budget`, runNburn detects physical RAM, reserves one quarter for
the operating system, KV cache, runtime buffers, and other processes, and uses
the remaining three quarters as the engine working-set budget. An explicit value
replaces that automatic budget.

The budget controls engine-owned host residency and file-backed sparse-expert
page caches. It is not an operating-system RSS limit: mapped weights, KV cache,
temporary buffers, runtime libraries, and unrelated process memory still
contribute to RSS.

Supported CUDA sparse-MoE paths size resident weights and hot caches from current
free and total VRAM. They do not use fixed device-name presets. Original quantized
weights remain authoritative, while expanded resident F16/F32 projection copies
stay outside the product default.

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
orders of magnitude. Use `rnb-llm-bench` with a fixed prompt and compare
engines on the same device under matched conditions.

A useful comparison repeats warm runs, reports the median, and checks generated
output for semantic quality before accepting a speedup. Keep backend defaults
unchanged unless the experiment is explicitly testing a documented override.

## Workspace layout

```text
crates/
  rnb-core/          Shared tensor, dtype, quantization, and IR contracts
  rnb-cpu/           CPU kernels, quantization, packed GEMM/GEMV
  rnb-loader/        GGUF loading, metadata, and tensor views
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
  rnb-tools/         Benchmarks, probes, and development utilities
```

## Development checks

```bash
cargo fmt --all --check
cargo test -p rnb-cli --no-default-features --features cpu
scripts/check_build_matrix.sh
```

Android runtime measurements require a physical device and ADB. CUDA and Metal
performance claims likewise require the target hardware; CPU-only compilation is
a correctness check, not a substitute benchmark.

## License

Licensed under the [Apache License 2.0](LICENSE). Portions derived from
third-party projects remain under their original terms; see [NOTICE](NOTICE).
