//! Storage IO micro-bench: O_DIRECT + pread 로 page cache 우회 cold latency 측정.
//!
//! 사용법:
//!   rnb-storage-iobench <file> <pattern=random|seq> <chunk_bytes> <iters> [--no-direct]
//!
//! O_DIRECT 가 거부되면 자동으로 buffered 로 fallback.
#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    use std::env;
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::process::ExitCode;
    use std::time::Instant;

    use rand::{rngs::SmallRng, Rng, SeedableRng};

    const PAGE: usize = 4096;

    #[derive(Clone, Copy, Debug)]
    enum Pattern {
        Random,
        Sequential,
    }

    fn parse_pattern(s: &str) -> Option<Pattern> {
        match s {
            "random" | "rand" => Some(Pattern::Random),
            "seq" | "sequential" => Some(Pattern::Sequential),
            _ => None,
        }
    }

    /// `O_DIRECT | O_RDONLY` 로 오픈. 실패 시 buffered fallback. 두 번째 값이 `O_DIRECT` 활성 여부.
    fn open_file(path: &Path, force_buffered: bool) -> std::io::Result<(i32, bool)> {
        let cpath = CString::new(path.as_os_str().as_bytes())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        if !force_buffered {
            // Linux: O_DIRECT = 0o40000. Android Bionic 도 동일.
            let direct_flag = libc::O_DIRECT;
            let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY | direct_flag) };
            if fd >= 0 {
                return Ok((fd, true));
            }
            let err = std::io::Error::last_os_error();
            eprintln!(
                "WARN: O_DIRECT open failed (errno={:?}): {}. Falling back to buffered.",
                err.raw_os_error(),
                err
            );
        }

        let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok((fd, false))
    }

    /// page-aligned heap buffer (`posix_memalign`).
    fn aligned_buf(size: usize) -> *mut u8 {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let r = unsafe { libc::posix_memalign(&mut ptr as *mut _, PAGE, size) };
        assert_eq!(r, 0, "posix_memalign failed (size={})", size);
        ptr as *mut u8
    }

    fn file_size(fd: i32) -> u64 {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let r = unsafe { libc::fstat(fd, &mut st as *mut _) };
        assert_eq!(r, 0, "fstat failed");
        st.st_size as u64
    }

    fn bench(fd: i32, fsize: u64, chunk: usize, iters: usize, pattern: Pattern) -> Vec<f64> {
        let buf = aligned_buf(chunk);
        let mut rng = SmallRng::seed_from_u64(0xCAFEBABE);
        let max_offset = fsize.saturating_sub(chunk as u64);
        let aligned_max = (max_offset / PAGE as u64) * PAGE as u64;

        let mut latencies_ms = Vec::with_capacity(iters);
        for i in 0..iters {
            let off: u64 = match pattern {
                Pattern::Random => {
                    let r = rng.gen_range(0..=aligned_max.max(PAGE as u64));
                    (r / PAGE as u64) * PAGE as u64
                }
                Pattern::Sequential => {
                    let stride = chunk as u64;
                    let span = aligned_max.max(stride);
                    ((i as u64 * stride) % span / PAGE as u64) * PAGE as u64
                }
            };
            let t0 = Instant::now();
            let n = unsafe { libc::pread(fd, buf as *mut _, chunk, off as i64) };
            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if n < 0 {
                let err = std::io::Error::last_os_error();
                panic!(
                    "pread failed at off={} chunk={}: errno={:?} ({})",
                    off,
                    chunk,
                    err.raw_os_error(),
                    err
                );
            }
            if n as usize != chunk {
                panic!("short read at off={} chunk={}: got {}", off, chunk, n);
            }
            latencies_ms.push(dt_ms);
        }
        unsafe { libc::free(buf as *mut _) };
        latencies_ms
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f64 * p) as usize;
        sorted[idx]
    }

    fn report(label: &str, lat: &mut Vec<f64>, chunk: usize) {
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lat.len() as f64;
        let mean: f64 = lat.iter().sum::<f64>() / n;
        let p50 = percentile(lat, 0.50);
        let p90 = percentile(lat, 0.90);
        let p99 = percentile(lat, 0.99);
        let mb = chunk as f64 / (1024.0 * 1024.0);
        let throughput_mean = mb / (mean / 1000.0); // MB/s
        let raw: Vec<String> = lat.iter().map(|x| format!("{:.2}", x)).collect();
        println!(
        "{}: n={} mean={:.2}ms p50={:.2}ms p90={:.2}ms p99={:.2}ms throughput_mean={:.1}MB/s chunk={}MiB",
        label, lat.len(), mean, p50, p90, p99, throughput_mean, mb
    );
        println!("  raw_ms = [{}]", raw.join(", "));
    }

    pub(crate) fn main() -> ExitCode {
        let args: Vec<String> = env::args().collect();
        if args.len() < 5 {
            eprintln!("usage: rnb-storage-iobench <file> <pattern=random|seq> <chunk_bytes> <iters> [--no-direct]");
            return ExitCode::from(2);
        }
        let path = Path::new(&args[1]);
        let pattern = match parse_pattern(&args[2]) {
            Some(p) => p,
            None => {
                eprintln!("pattern must be random|seq, got {:?}", args[2]);
                return ExitCode::from(2);
            }
        };
        let chunk: usize = match args[3].parse() {
            Ok(n) => n,
            Err(e) => {
                eprintln!("chunk must be integer bytes: {}", e);
                return ExitCode::from(2);
            }
        };
        let iters: usize = match args[4].parse() {
            Ok(n) => n,
            Err(e) => {
                eprintln!("iters must be integer: {}", e);
                return ExitCode::from(2);
            }
        };
        let force_buffered = args.iter().any(|a| a == "--no-direct");
        if chunk % PAGE != 0 {
            eprintln!(
                "chunk must be a multiple of page size {} (got {})",
                PAGE, chunk
            );
            return ExitCode::from(2);
        }

        let (fd, direct) = match open_file(path, force_buffered) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("open failed: {}", e);
                return ExitCode::from(1);
            }
        };
        let fsize = file_size(fd);
        println!(
            "file={} size={} bytes ({:.2} GiB) O_DIRECT={}",
            path.display(),
            fsize,
            fsize as f64 / 1024.0 / 1024.0 / 1024.0,
            direct
        );
        println!(
            "pattern={:?} chunk={} ({}KiB) iters={}",
            pattern,
            chunk,
            chunk / 1024,
            iters
        );

        let mut lat = bench(fd, fsize, chunk, iters, pattern);
        let label = format!("{:?}/chunk={}KiB/direct={}", pattern, chunk / 1024, direct);
        report(&label, &mut lat, chunk);

        unsafe {
            libc::close(fd);
        }
        ExitCode::SUCCESS
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn main() {
    imp::main();
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn main() {
    eprintln!("rnb-storage-iobench requires Linux/Android (O_DIRECT).");
    std::process::exit(2);
}
