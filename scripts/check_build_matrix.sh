#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/check_build_matrix.sh [--host-only] [--include-android]

Runs the local Cargo build matrix for the current runtime crate boundary design.

Default checks:
  - host rnb-cli CPU-only
  - host rnb-cli CPU+CUDA
  - host rnb-cli CPU+Vulkan
  - host rnb-cli CPU+OpenCL
  - host rnb-dev-tools CPU-only
  - host rnb-runtime CPU-only, CPU+CUDA, CPU+Vulkan, CPU+OpenCL
  - host rnb-runtime CPU+MediaTek
  - host rnb-cli CPU-only (metal off)  [macOS only]
  - host rnb-cli CPU+metal             [macOS only]

Optional checks:
  --include-android  also checks Android aarch64 FFI CPU and CPU+Vulkan.
                     Nightly Rust and the aarch64-linux-android target must
                     already be installed.
USAGE
}

include_android=0

for arg in "$@"; do
    case "$arg" in
        --host-only)
            include_android=0
            ;;
        --include-android)
            include_android=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_case() {
    local name="$1"
    shift

    echo
    echo "==> $name"
    printf '+'
    printf ' %q' "$@"
    echo
    "$@"
}

check_target_installed() {
    local target="$1"
    local toolchain="${2:-}"
    local rustup_args=(target list --installed)
    if [[ -n "$toolchain" ]]; then
        rustup_args=(target list --toolchain "$toolchain" --installed)
    fi

    if ! rustup "${rustup_args[@]}" | grep -Fx "$target" >/dev/null; then
        echo "error: Rust target is not installed: $target" >&2
        if [[ -n "$toolchain" ]]; then
            echo "hint: rustup target add --toolchain $toolchain $target" >&2
        else
            echo "hint: rustup target add $target" >&2
        fi
        exit 2
    fi
}

run_case "host cli cpu-only" \
    cargo check -p rnb-cli --no-default-features --features cpu
run_case "host cli cpu+cuda" \
    cargo check -p rnb-cli --no-default-features --features cpu,cuda
run_case "host cli cpu+vulkan" \
    cargo check -p rnb-cli --no-default-features --features cpu,vulkan
run_case "host cli cpu+opencl" \
    cargo check -p rnb-cli --no-default-features --features cpu,opencl

run_case "host dev tools cpu-only" \
    cargo check -p rnb-dev-tools --no-default-features --features cpu

run_case "host runtime cpu-only" \
    cargo check -p rnb-runtime --no-default-features --features cpu
run_case "host runtime cpu+cuda" \
    cargo check -p rnb-runtime --no-default-features --features cpu,cuda
run_case "host runtime cpu+vulkan" \
    cargo check -p rnb-runtime --no-default-features --features cpu,vulkan
run_case "host runtime cpu+opencl" \
    cargo check -p rnb-runtime --no-default-features --features cpu,opencl
run_case "host runtime cpu+mediatek" \
    cargo check -p rnb-runtime --no-default-features --features cpu,mediatek

# macOS metal 케이스: host 가 macOS 일 때만 실행
if [[ "$(uname)" == "Darwin" ]]; then
    run_case "host cli cpu-only (metal off)" \
        cargo build -p rnb-cli --no-default-features --features cpu
    run_case "host cli cpu+metal" \
        cargo build -p rnb-cli --no-default-features --features cpu,metal
fi

if [[ "$include_android" -eq 1 ]]; then
    if ! rustup run nightly rustc --version >/dev/null; then
        echo "error: nightly Rust toolchain is required for Android checks" >&2
        echo "hint: rustup toolchain install nightly" >&2
        exit 2
    fi
    check_target_installed "aarch64-linux-android" "nightly"

    run_case "android ffi cpu-only" \
        rustup run nightly cargo check -p rnb-ffi --target aarch64-linux-android --no-default-features --features cpu
    run_case "android ffi cpu+vulkan" \
        rustup run nightly cargo check -p rnb-ffi --target aarch64-linux-android --no-default-features --features cpu,vulkan
    run_case "android ffi cpu+mediatek" \
        rustup run nightly cargo check -p rnb-ffi --target aarch64-linux-android --no-default-features --features cpu,mediatek
fi
