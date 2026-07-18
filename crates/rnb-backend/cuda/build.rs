fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/q4k_gemv.cu");
    println!("cargo:rerun-if-changed=cuda/nemotron_selected.cu");
    println!("cargo:rerun-if-changed=cuda/persistent_decode.cu");
    println!("cargo:rerun-if-changed=cuda/kernels");

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let arch = std::env::var("RNB_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    compile_ptx(&arch, "cuda/q4k_gemv.cu", &out_dir.join("q4k_gemv.ptx"));
    compile_ptx(
        &arch,
        "cuda/nemotron_selected.cu",
        &out_dir.join("nemotron_selected.ptx"),
    );
    compile_ptx(
        &arch,
        "cuda/persistent_decode.cu",
        &out_dir.join("persistent_decode.ptx"),
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
