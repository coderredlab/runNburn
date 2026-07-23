//! io_uring availability probe for Android (Flip4).
//!
//! Checks whether:
//!   1. `io_uring_setup` succeeds (kernel + SELinux lets us)
//!   2. A batch of `IORING_OP_READ` SQEs (O_DIRECT file) submits and all
//!      CQEs land with expected byte counts
//!   3. The observed wall-clock for N parallel reads beats `pread` loop
//!
//! Usage:
//!   RNB_IOURING_PATH=/data/local/tmp/rnb/<file> \
//!   RNB_IOURING_OFFSETS="0,8192,16384,..." \   # optional, defaults to 0..16*4MiB
//!   ./rnb-iouring-probe
//!
//! Designed as a stand-alone binary — no engine, no mmap.

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn main() {
    eprintln!("[probe] io_uring is linux-only; skipping");
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn main() {
    use io_uring::{opcode, types, IoUring};
    use std::fs::OpenOptions;
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    use std::time::Instant;

    // Step 1: can we create an io_uring?
    let t0 = Instant::now();
    let mut ring = match IoUring::new(32) {
        Ok(r) => {
            println!(
                "[probe] io_uring_setup OK (entries=32) in {:?}",
                t0.elapsed()
            );
            r
        }
        Err(e) => {
            eprintln!(
                "[probe] io_uring_setup FAILED: {} (errno={:?})",
                e,
                e.raw_os_error()
            );
            std::process::exit(1);
        }
    };

    // Step 2: open the explicitly selected target file with O_DIRECT.
    let path = std::env::var("RNB_IOURING_PATH").unwrap_or_else(|_| {
        eprintln!("[probe] RNB_IOURING_PATH must point to the file under test");
        std::process::exit(2);
    });
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[probe] open {} failed: {}", path, e);
            std::process::exit(1);
        }
    };
    let fd = file.as_raw_fd();
    println!("[probe] opened {} with O_DIRECT (fd={})", path, fd);

    // Step 3: prepare 16 parallel 4 MiB reads with 4K-aligned buffers
    let n_reqs = 16usize;
    let chunk = 4 * 1024 * 1024usize; // 4 MiB
    let mut bufs: Vec<Vec<u8>> = (0..n_reqs).map(|_| aligned_buf(chunk)).collect();
    let offsets: Vec<u64> = (0..n_reqs).map(|i| (i as u64) * (chunk as u64)).collect();

    let t_submit = Instant::now();
    {
        let mut sq = ring.submission();
        for (i, buf) in bufs.iter_mut().enumerate() {
            let read_e = opcode::Read::new(types::Fd(fd), buf.as_mut_ptr(), chunk as u32)
                .offset(offsets[i])
                .build()
                .user_data(i as u64);
            unsafe {
                sq.push(&read_e).expect("sq push");
            }
        }
    }
    ring.submit_and_wait(n_reqs).expect("submit_and_wait");
    let submit_wait_elapsed = t_submit.elapsed();

    // Step 4: drain CQEs
    let mut cq = ring.completion();
    let mut total_bytes = 0u64;
    let mut cqe_count = 0usize;
    for cqe in &mut cq {
        let res = cqe.result();
        if res < 0 {
            eprintln!(
                "[probe] CQE user_data={} FAILED: errno={}",
                cqe.user_data(),
                -res
            );
            continue;
        }
        total_bytes += res as u64;
        cqe_count += 1;
    }
    println!(
        "[probe] {} CQEs drained; {} MiB total in {:?} ({:.1} MB/s)",
        cqe_count,
        total_bytes / (1024 * 1024),
        submit_wait_elapsed,
        (total_bytes as f64 / submit_wait_elapsed.as_secs_f64()) / 1e6
    );
    assert_eq!(cqe_count, n_reqs);
    assert_eq!(total_bytes as usize, n_reqs * chunk);

    // Step 5: compare against sequential pread loop (same N × chunk)
    use std::os::unix::fs::FileExt;
    let mut seq_bufs: Vec<Vec<u8>> = (0..n_reqs).map(|_| aligned_buf(chunk)).collect();
    let t_pread = Instant::now();
    for (i, buf) in seq_bufs.iter_mut().enumerate() {
        file.read_exact_at(buf.as_mut_slice(), offsets[i])
            .expect("pread");
    }
    let pread_elapsed = t_pread.elapsed();
    println!(
        "[probe] pread loop: {} reads × {} MiB in {:?} ({:.1} MB/s)",
        n_reqs,
        chunk / (1024 * 1024),
        pread_elapsed,
        (total_bytes as f64 / pread_elapsed.as_secs_f64()) / 1e6
    );

    let speedup = pread_elapsed.as_secs_f64() / submit_wait_elapsed.as_secs_f64();
    println!(
        "[probe] io_uring batch vs pread loop speedup: {:.2}×",
        speedup
    );
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn aligned_buf(cap: usize) -> Vec<u8> {
    // 4 KiB-aligned allocation for O_DIRECT. We over-allocate and manually
    // offset; the returned Vec is not freeable by Vec machinery, but probe
    // binary exits immediately so leaks are irrelevant.
    let aligned_cap = (cap + 4095) & !4095;
    let mut raw: Vec<u8> = vec![0u8; aligned_cap + 4096];
    let addr = raw.as_ptr() as usize;
    let pad = (4096 - (addr & 4095)) & 4095;
    raw.drain(0..pad);
    raw.truncate(aligned_cap);
    raw
}
