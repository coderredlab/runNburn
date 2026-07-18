//! O_DIRECT 가능 여부 + 실측 throughput 확인. axis G 사전 sanity.
//!
//! 사용: rnb-direct-check <path>
//! 출력: O_DIRECT open 성공 여부, sequential read 1GB 의 elapsed/throughput.

#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    use std::env;
    use std::os::unix::fs::OpenOptionsExt;

    pub(crate) fn main() {
        let args: Vec<String> = env::args().collect();
        if args.len() != 2 {
            eprintln!("usage: rnb-direct-check <path>");
            std::process::exit(2);
        }
        let path = &args[1];

        // O_DIRECT 시도. libc::O_DIRECT 가 architecture-specific.
        eprintln!("[INFO] libc::O_DIRECT = 0x{:x}", libc::O_DIRECT);
        let res = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path);

        let f = match res {
            Ok(f) => {
                eprintln!("[OK] open(O_DIRECT) succeeded for {path}");
                f
            }
            Err(e) => {
                eprintln!("[FAIL] open(O_DIRECT) failed: {e}");
                std::process::exit(1);
            }
        };

        // 1 GiB sequential read with 1 MiB buffer (4 KB align 보장).
        use std::io::Read;
        let buf_size: usize = 1024 * 1024;
        // Allocate 4 KB-aligned buffer (Vec is not aligned by default; use a Box<[u8]>
        // and check alignment empirically — most allocators give >= 16 B; we ensure
        // 4 KB by overallocating + offsetting).
        let raw = vec![0u8; buf_size + 4096];
        let raw_ptr = raw.as_ptr() as usize;
        let aligned_off = (4096 - (raw_ptr & 4095)) & 4095;
        let target_total: usize = 1024 * 1024 * 1024;

        let mut f = f;
        let t0 = std::time::Instant::now();
        let mut read_total: usize = 0;
        while read_total < target_total {
            // SAFETY: aligned slice within the original Vec
            let buf = unsafe {
                std::slice::from_raw_parts_mut((raw.as_ptr().add(aligned_off)) as *mut u8, buf_size)
            };
            match f.read(buf) {
                Ok(0) => break,
                Ok(n) => read_total += n,
                Err(e) => {
                    eprintln!("[ERR] read failed at {read_total} bytes: {e}");
                    std::process::exit(1);
                }
            }
        }
        let elapsed = t0.elapsed();
        let throughput_mbps = (read_total as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
        eprintln!(
            "[OK] O_DIRECT read: {} MB in {:.2}s = {:.1} MB/s",
            read_total / (1024 * 1024),
            elapsed.as_secs_f64(),
            throughput_mbps
        );

        // Micro-bench: per-expert random pread latency.
        // Gemma4 26B-A4B: per-expert gate_up = 2.23 MB, down = 1.49 MB; combined ~3.72 MB.
        // We measure 100 random reads of 4 MB (rounded up to 4 KB).
        use std::os::unix::fs::FileExt;
        let read_size: usize = 4 * 1024 * 1024; // 4 MiB, divisible by 4 KB
        let buf_2 = vec![0u8; read_size + 4096];
        let buf_2_off = (4096 - (buf_2.as_ptr() as usize & 4095)) & 4095;
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if file_size < (read_size as u64) * 2 {
            eprintln!("[WARN] file too small for micro-bench, skipping");
            return;
        }
        let max_offset = (file_size - read_size as u64) & !4095;

        let n_iters = 100;
        let mut state: u64 = 0xdead_beef_1234_5678;
        let mut total_latency = std::time::Duration::ZERO;
        let mut total_bytes: u64 = 0;
        for _ in 0..n_iters {
            // simple LCG for pseudo-random offset
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let offset = (state >> 16) % max_offset;
            let offset = offset & !4095;

            let buf = unsafe {
                std::slice::from_raw_parts_mut(
                    (buf_2.as_ptr().add(buf_2_off)) as *mut u8,
                    read_size,
                )
            };
            let t = std::time::Instant::now();
            let n = f.read_at(buf, offset).expect("pread failed");
            total_latency += t.elapsed();
            total_bytes += n as u64;
        }
        let avg_latency_ms = total_latency.as_secs_f64() * 1000.0 / n_iters as f64;
        let avg_throughput = (total_bytes as f64 / 1024.0 / 1024.0) / total_latency.as_secs_f64();
        eprintln!(
            "[BENCH] {} random pread of {} MB: avg latency = {:.2} ms, avg throughput = {:.1} MB/s",
            n_iters,
            read_size / (1024 * 1024),
            avg_latency_ms,
            avg_throughput,
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn main() {
    imp::main();
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn main() {
    eprintln!("rnb-direct-check requires Linux/Android (O_DIRECT).");
    std::process::exit(2);
}
