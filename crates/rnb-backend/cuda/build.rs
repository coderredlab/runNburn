fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/q4k_gemv.cu");
    println!("cargo:rerun-if-changed=cuda/nemotron_selected.cu");
    println!("cargo:rerun-if-changed=cuda/persistent_decode.cu");
    println!("cargo:rerun-if-changed=cuda/kernels");
    println!("cargo:rerun-if-env-changed=RNB_CUDA_ARCH");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if target_arch != "x86_64" || !matches!(target_os.as_str(), "linux" | "windows") {
        return;
    }

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let arch = std::env::var("RNB_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    println!("cargo:rustc-env=RNB_CUDA_COMPILED_ARCH={arch}");
    compile_ptx(&arch, "cuda/q4k_gemv.cu", &out_dir.join("q4k_gemv.ptx"));
    compile_cubin(&arch, "cuda/q4k_gemv.cu", &out_dir.join("q4k_gemv.cubin"));
    compile_ptx(
        &arch,
        "cuda/nemotron_selected.cu",
        &out_dir.join("nemotron_selected.ptx"),
    );
    compile_cubin(
        &arch,
        "cuda/nemotron_selected.cu",
        &out_dir.join("nemotron_selected.cubin"),
    );
    compile_ptx(
        &arch,
        "cuda/persistent_decode.cu",
        &out_dir.join("persistent_decode.ptx"),
    );
    compile_cubin(
        &arch,
        "cuda/persistent_decode.cu",
        &out_dir.join("persistent_decode.cubin"),
    );
}

fn compile_ptx(arch: &str, source: &str, ptx: &std::path::Path) {
    let status = std::process::Command::new("nvcc")
        .args(["-ptx", "-O3", "-std=c++17", "-arch", &arch, source, "-o"])
        .arg(ptx)
        .status()
        .unwrap_or_else(|err| panic!("failed to run nvcc for {source}: {err}"));
    if !status.success() {
        panic!("nvcc failed while compiling {source}");
    }
}

fn compile_cubin(arch: &str, source: &str, cubin: &std::path::Path) {
    let status = std::process::Command::new("nvcc")
        .args(["-cubin", "-O3", "-std=c++17", "-arch", arch, source, "-o"])
        .arg(cubin)
        .status()
        .unwrap_or_else(|err| panic!("failed to run nvcc for {source}: {err}"));
    if !status.success() {
        panic!("nvcc failed while compiling {source}");
    }
}
