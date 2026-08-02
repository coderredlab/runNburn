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

For Qwen3.6 vision models, provide the projector and an initial image:

```bash
./target/release/runNburn chat \
  --mmproj /path/to/mmproj.gguf \
  --image /path/to/image.png \
  /path/to/model.gguf
```

Later turns reuse the retained multimodal sequence state instead of reprocessing
the image prefix. `/clear` discards both the conversation and retained image state.

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

Add `--mmproj /path/to/mmproj.gguf` when serving a Qwen3.6 vision model. The
Responses and Chat Completions endpoints accept base64 image data URLs in their
standard multimodal content arrays.

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
| `POST` | `/v1/chat/completions` | Non-streaming and SSE streaming; multimodal input, tools, and structured output |
| `POST` | `/v1/responses` | Non-streaming and SSE streaming; multimodal input, tools, structured output, and stateful continuation |
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
- DeepSeek 4 Flash

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

### Worked example: a 222 GiB model under a 32 GiB host budget

This transcript demonstrates the bounded-memory claim on named hardware. It is
a reproduction recipe, not a cross-device headline: absolute numbers depend on
your disk, page cache, and GPU.

Hardware: Ryzen 9 5950X, 64 GB RAM (62.7 GiB usable), RTX 3090 24 GB (CUDA
device 0), NVMe SSD, Linux. Model: `GLM-5.2` `UD-IQ2_M`, a 6-way split GGUF
with 222.18 GiB of mapped weights — roughly 2.5 times the machine's RAM and
VRAM combined.

```bash
cargo build --release -p rnb-dev-tools --features cuda --bin rnb-llm-bench

RNB_MODEL=/path/to/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf \
RNB_FORCE_GGUF=1 \
RNB_PROMPT="대한민국의 수도는" \
RNB_DECODE_TOKENS=10 \
RNB_HOST_RAM_BUDGET_BYTES=$((32 * 1024 * 1024 * 1024)) \
RNB_BENCH_WALL=1 \
/usr/bin/time -v ./target/release/rnb-llm-bench
```

Measured on 2026-07-30 (one warmup, then three consecutive runs; the working
set exceeds RAM, so every run repages from disk and there is no warm/cold
split):

| Run | Prefill (8 tokens) | Decode (10 tokens) | Peak RSS |
|---|---:|---:|---:|
| 1 | 34.92 s | 24.40 s | 28.7 GiB |
| 2 | 43.41 s | 29.48 s | 28.6 GiB |
| 3 | 42.44 s | 26.05 s | 28.6 GiB |
| **median** | **42.44 s** | **26.05 s** (~2.6 s/token) | **< 32 GiB budget** |

All runs produced token-identical output. The same budget applies to the
product CLI via `--ram-budget 32GiB`. Slower than a model that fits in
memory? Yes — the point is that it runs at all, with predictable memory,
instead of being rejected or OOM-killed.

#### Core advantage: a 222 GiB model on a 64 GiB Mac

This is the product capability runNburn is built for: executing a model far
larger than system memory while keeping residency bounded. The same 222.18 GiB
`GLM-5.2-UD-IQ2_M` split GGUF was measured on an Apple M5 Pro with 64 GiB
unified memory. Rather than trying to make the whole model resident, runNburn
streams the required expert weights through a bounded cache. The comparison
below measures that larger-than-memory capability from process start to
completion.

- runNburn `67e92b4`: release Metal build, automatic 48 GiB host budget,
  30.12 GiB sparse-expert page cache, and product-default MTP auto policy.
- llama.cpp `3018a11`: release Metal build, native six CPU threads,
  `-c 2048 -ngl 1 -fit off`. This was the stable partial-Metal configuration
  used for the repeated comparison.
- Input: `prompts/prompt_capital_qa_ko.txt` (20 prompt tokens), greedy
  generation, 8 decoded tokens, one warmup per engine, then alternating
  `runNburn / llama.cpp` runs without cooldown.

| Engine and configuration | External wall samples | Median |
|---|---:|---:|
| runNburn Metal, bounded expert streaming | 28.468 / 28.563 / 28.658 s | **28.563 s** |
| llama.cpp Metal build, one GPU layer | 156.402 / 155.867 / 156.618 s | **156.402 s** |

The measured external-wall ratio was `156.402 / 28.563 = 5.48x`. All three
runNburn measurements produced the same 8-token SHA-256
`8136b9f248ff5da30946ca049f8a32435feb4c184a2f8b6665dc5b29f46ca84d`.
The short generation limit checks repeatability and execution stability; it is
not a semantic answer-quality comparison.

llama.cpp's default auto-fit and explicit full-Metal attempts both reached a
Metal command-buffer out-of-memory error. Conservative 16 GiB and 32 GiB fit
targets failed the same way. `-ngl 8` failed a 65,536 MiB Metal allocation;
`-ngl 4` completed a one-off screen in 160.87 s, while `-ngl 1` was retained
for the repeated set above. This is runNburn's intended product advantage:
stable, bounded execution when the model exceeds available memory and
full-residency accelerator configurations cannot run. The measured `5.48x`
is the end-to-end product-wall result for this constrained-memory workload,
not a claim that runNburn's individual kernels are `5.48x` faster when a model
fits fully in memory.

### Availability comparison: 35.2 GiB full-Q8 MTP on a 14 GiB host

This is an availability result, not a throughput ranking. It uses one machine,
one model file, and matched generation inputs:

- Hardware: AMD BC-250, 14 GiB RAM, 7.7 GiB swap, 16,896 MiB Vulkan device
  memory, Mesa RADV.
- Model: `Qwen3.6-35B-A3B-Q8_0.gguf`, 37,801,097,504 bytes of mapped GGUF
  weights.
- Input: raw `대한민국의 수도는` prompt (5 tokens), greedy generation,
  8 decoded tokens, EOS ignored, 12 CPU threads, and native MTP depth 1 when enabled.
- Engines: runNburn Vulkan fullpath from change set `d9967dc`; llama.cpp
  `b1-fb30ba9` with Vulkan auto-fit and native `draft-mtp`.

| Engine and configuration | Observed result |
|---|---|
| runNburn Vulkan fullpath | Completed the warmup and six alternating target-only/MTP measurements. The three MTP runs had a 9.157 s median generation time, and all seven runs had SHA-256 `df80205ec3dac2214188a84c0ba30c2d327e9b9d752554e99d09eaafa139eb29`. A separate environment-unset smoke automatically enabled MTP and produced the same hash. |
| llama.cpp native MTP, 256 MiB fit target | One interactive smoke reached 8-token generation, but a clean single-turn repeat failed to allocate an 897,028,096-byte RADV buffer and its child process was killed. |
| llama.cpp native MTP, 2 GiB fit target | Did not complete; the host rebooted during the run under memory pressure. |

There is deliberately no speed ratio for this experiment: llama.cpp did not
produce a repeatable measured set on this host. This also does not claim that
every llama.cpp configuration fails. Target-only execution or substantially
less GPU offload may run, but those are different execution and memory
conditions. The demonstrated claim is narrower: under matched full-Q8 native
MTP conditions, runNburn completed repeatedly while the reference configuration
was not memory-stable.

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
