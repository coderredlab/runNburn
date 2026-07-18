//! mincore(2) syscall latency micro-bench.
//!
//! Maps a 32 MiB anonymous region, touches every page to populate it,
//! then calls `mincore` N times on successive 3.55 MiB sub-ranges
//! (matches Gemma4 26B-A4B per-expert size). Reports p50/p90/p99 latency
//! and projected overhead per decode step (240 calls = 30 layer × 8 expert).

use std::time::Instant;

#[cfg(any(target_os = "linux", target_os = "android"))]
fn main() {
    const REGION_BYTES: usize = 32 * 1024 * 1024;
    const EXPERT_BYTES: usize = 3_716_096; // ~3.55 MiB (Gemma4 per-expert avg)
    const N_CALLS: usize = 10_000;

    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let expert_pages = (EXPERT_BYTES + page_size - 1) / page_size;

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            REGION_BYTES,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED, "mmap failed");

    // Touch every page so mincore returns "resident" (tests realistic hit case).
    unsafe {
        std::ptr::write_bytes(ptr as *mut u8, 0x42, REGION_BYTES);
    }

    let mut vec_buf = vec![0u8; expert_pages];
    let mut lats = Vec::with_capacity(N_CALLS);

    // Linux mincore(2) requires `addr` to be a page-size multiple.
    // EXPERT_BYTES (3_716_096) is not a page multiple, so round the stride
    // down to the nearest page to keep every probe address page-aligned.
    let stride = (EXPERT_BYTES / page_size) * page_size;
    let max_off = ((REGION_BYTES - EXPERT_BYTES) / page_size) * page_size;
    assert!(
        stride > 0 && max_off > 0,
        "region too small for expert stride"
    );

    for i in 0..N_CALLS {
        let off = (i * stride) % max_off;
        let sub = unsafe { (ptr as *mut u8).add(off) };
        let t = Instant::now();
        let rc = unsafe { libc::mincore(sub as *mut _, EXPERT_BYTES, vec_buf.as_mut_ptr()) };
        lats.push(t.elapsed().as_nanos() as u64);
        assert_eq!(rc, 0, "mincore returned {}", rc);
    }

    lats.sort();
    let p50 = lats[N_CALLS / 2];
    let p90 = lats[N_CALLS * 9 / 10];
    let p99 = lats[N_CALLS * 99 / 100];
    let mean: u64 = lats.iter().sum::<u64>() / N_CALLS as u64;

    // 240 calls/decode step (30 layer × 8 expert)
    let projected_ns = mean * 240;

    println!(
        "mincore latency over {} calls on {} B range ({} pages):",
        N_CALLS, EXPERT_BYTES, expert_pages
    );
    println!("  p50  = {} ns", p50);
    println!("  p90  = {} ns", p90);
    println!("  p99  = {} ns", p99);
    println!("  mean = {} ns", mean);
    println!(
        "Projected 240 calls/step overhead: {} ns = {:.3} ms",
        projected_ns,
        projected_ns as f64 / 1e6
    );
    println!(
        "Baseline decode ~1000 ms/step → overhead {:.2}%",
        projected_ns as f64 / 1e7
    );

    unsafe { libc::munmap(ptr, REGION_BYTES) };
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn main() {
    eprintln!("rnb-mincore-bench is linux/android only");
    std::process::exit(1);
}
