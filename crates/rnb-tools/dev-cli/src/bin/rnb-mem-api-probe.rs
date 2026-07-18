//! Session 68 memory-API probe. Allocates a chosen-size region through one
//! of four Android-relevant mechanisms and watches the kernel's reaction
//! (VmRSS/VmSwap/VmLck, /proc/meminfo Cached/MemAvailable/SwapFree,
//! pgmajfault / pswpout) across touch + hold phases.
//!
//! Usage:
//!   rnb-mem-api-probe <mode> <size_bytes> <hold_seconds>
//!
//! mode: one of
//!   memfd    — `memfd_create` + `ftruncate` + `mmap(MAP_SHARED)`
//!   ashmem   — `/dev/ashmem` + `ASHMEM_SET_SIZE` ioctl + `mmap(MAP_SHARED)`
//!              (+ optional `ASHMEM_PIN` once memory is touched)
//!   anon     — `mmap(MAP_ANONYMOUS|MAP_PRIVATE)` — zRAM-eligible baseline
//!   mlock    — `mmap(MAP_ANONYMOUS|MAP_PRIVATE) + mlock` — hit RLIMIT=64MB
//!
//! We run this as an adb-shell binary, not an app, so LMK won't actually
//! kill us (oom_score_adj=-1000). The interesting signals are:
//!   * Does the allocation succeed at N GB? (sysmem pressure / ENOMEM)
//!   * Does touching cause Cached to collapse? (pagecache eviction of the
//!     GGUF mmap we do NOT hold here — probe is stand-alone)
//!   * Does pswpout fire? (anon -> zRAM compression)
//!   * Does ashmem avoid the regressions that memfd/anon trigger?

#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    use std::env;
    use std::ffi::CString;
    use std::io::{self, Read, Write};
    use std::os::fd::RawFd;
    use std::time::{Duration, Instant};

    const PAGE_SIZE: usize = 4096;

    pub(crate) fn main() {
        let args: Vec<String> = env::args().collect();
        if args.len() < 4 {
            eprintln!(
                "usage: {} <mode:memfd|ashmem|anon|mlock> <size_bytes> <hold_secs>",
                args[0]
            );
            std::process::exit(2);
        }
        let mode = args[1].clone();
        let size: usize = args[2].parse().expect("size_bytes must be integer");
        let hold_secs: u64 = args[3].parse().expect("hold_secs must be integer");

        println!("=== rnb-mem-api-probe ===");
        println!(
            "mode={mode}  size={} MB  hold={hold_secs}s",
            size / 1024 / 1024
        );
        print_labeled_meminfo("before_alloc");
        print_labeled_status("before_alloc");

        let alloc = match mode.as_str() {
            "memfd" => alloc_memfd(size),
            "ashmem" => alloc_ashmem(size, true),
            "anon" => alloc_anon(size),
            "mlock" => alloc_mlock(size),
            other => {
                eprintln!("unknown mode: {other}");
                std::process::exit(2);
            }
        };
        let Some(alloc) = alloc else {
            eprintln!("[probe] allocation failed — terminating");
            print_labeled_meminfo("alloc_failed");
            std::process::exit(1);
        };

        print_labeled_status("after_alloc");
        print_labeled_meminfo("after_alloc");

        // Touch every page to commit.
        println!("[probe] touching {} pages", size / PAGE_SIZE);
        let touch_start = Instant::now();
        let ptr_u8 = alloc.ptr as *mut u8;
        let mut stride: usize = 0;
        for i in 0..(size / PAGE_SIZE) {
            unsafe {
                std::ptr::write_volatile(ptr_u8.add(i * PAGE_SIZE), (i as u8).wrapping_mul(37));
            }
            stride = stride.wrapping_add(1);
        }
        let touch_elapsed = touch_start.elapsed();
        println!(
            "[probe] touch done in {:.3}s ({:.1} MB/s)",
            touch_elapsed.as_secs_f64(),
            (size as f64) / touch_elapsed.as_secs_f64() / 1_048_576.0
        );

        print_labeled_status("after_touch");
        print_labeled_meminfo("after_touch");

        // Hold + periodic sample.
        let deadline = Instant::now() + Duration::from_secs(hold_secs);
        let mut tick = 0;
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_secs(5));
            tick += 1;
            print_labeled_status(&format!("hold_{tick}"));
            print_labeled_meminfo(&format!("hold_{tick}"));
        }

        // Random-access latency probe — how quickly can we read back 1000 pages?
        println!("[probe] random-access 1000 pages...");
        let pages = size / PAGE_SIZE;
        let mut acc: u64 = 0;
        let rng_start = Instant::now();
        for i in 0..1000 {
            let idx = (i * 2654435761usize) % pages; // Fibonacci hash — uneven
            unsafe {
                acc = acc.wrapping_add(std::ptr::read_volatile(ptr_u8.add(idx * PAGE_SIZE)) as u64);
            }
        }
        let rng_elapsed = rng_start.elapsed();
        println!(
            "[probe] 1000 random reads in {:.6}s (avg {:.2} ns/read), acc={acc}",
            rng_elapsed.as_secs_f64(),
            rng_elapsed.as_nanos() as f64 / 1000.0
        );

        drop(alloc);
        print_labeled_meminfo("after_drop");
        println!("[probe] done");
    }

    // ---- allocators ----

    struct Alloc {
        ptr: *mut libc::c_void,
        size: usize,
        fd: Option<RawFd>,
        mlocked: bool,
    }

    impl Drop for Alloc {
        fn drop(&mut self) {
            unsafe {
                if self.mlocked {
                    libc::munlock(self.ptr, self.size);
                }
                libc::munmap(self.ptr, self.size);
                if let Some(fd) = self.fd {
                    libc::close(fd);
                }
            }
        }
    }

    fn alloc_memfd(size: usize) -> Option<Alloc> {
        let name = CString::new("rnb-probe-memfd").unwrap();
        let fd = unsafe { libc::memfd_create(name.as_ptr(), 0) };
        if fd < 0 {
            eprintln!("memfd_create failed: {}", io::Error::last_os_error());
            return None;
        }
        if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
            eprintln!("ftruncate failed: {}", io::Error::last_os_error());
            unsafe { libc::close(fd) };
            return None;
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            eprintln!("mmap(memfd) failed: {}", io::Error::last_os_error());
            unsafe { libc::close(fd) };
            return None;
        }
        Some(Alloc {
            ptr,
            size,
            fd: Some(fd),
            mlocked: false,
        })
    }

    #[cfg(target_os = "android")]
    #[allow(non_snake_case)]
    #[link(name = "android")]
    extern "C" {
        fn ASharedMemory_create(name: *const libc::c_char, size: libc::size_t) -> libc::c_int;
    }

    #[cfg(not(target_os = "android"))]
    #[allow(non_snake_case)]
    unsafe extern "C" fn ASharedMemory_create(
        _name: *const libc::c_char,
        _size: libc::size_t,
    ) -> libc::c_int {
        -1
    }

    fn alloc_ashmem(size: usize, _do_pin: bool) -> Option<Alloc> {
        // /dev/ashmem direct open is blocked by SELinux for non-privileged
        // contexts on Android 11+. Go through the NDK wrapper instead; the
        // kernel still backs this with ashmem (or memfd w/ seals on newer
        // releases) but the open path is allow-listed for untrusted_app /
        // shell domains.
        let name = CString::new("rnb-probe-ashmem").unwrap();
        let fd = unsafe { ASharedMemory_create(name.as_ptr(), size) };
        if fd < 0 {
            eprintln!(
                "ASharedMemory_create failed: {}",
                io::Error::last_os_error()
            );
            return None;
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            eprintln!("mmap(ashmem) failed: {}", io::Error::last_os_error());
            unsafe { libc::close(fd) };
            return None;
        }
        // NOTE: NDK does not expose ASHMEM_PIN — pin/unpin is available only
        // from the Java SharedMemory#setProtect path (and only as coarse
        // "read-only seal"). For the probe we compare unpinned behavior.
        Some(Alloc {
            ptr,
            size,
            fd: Some(fd),
            mlocked: false,
        })
    }

    fn alloc_anon(size: usize) -> Option<Alloc> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            eprintln!("mmap(anon) failed: {}", io::Error::last_os_error());
            return None;
        }
        Some(Alloc {
            ptr,
            size,
            fd: None,
            mlocked: false,
        })
    }

    fn alloc_mlock(size: usize) -> Option<Alloc> {
        let a = alloc_anon(size)?;
        // Touch first (mlock locks resident pages).
        let ptr_u8 = a.ptr as *mut u8;
        for i in 0..(size / PAGE_SIZE) {
            unsafe { std::ptr::write_volatile(ptr_u8.add(i * PAGE_SIZE), 0u8) };
        }
        let rc = unsafe { libc::mlock(a.ptr, a.size) };
        if rc != 0 {
            eprintln!("mlock failed: {}", io::Error::last_os_error());
            return Some(a); // keep anon region, just not locked
        }
        let mut a = a;
        a.mlocked = true;
        Some(a)
    }

    // ---- observability ----

    fn print_labeled_status(label: &str) {
        let body = read_file("/proc/self/status").unwrap_or_default();
        let keys = [
            "VmSize",
            "VmRSS",
            "VmHWM",
            "VmSwap",
            "VmLck",
            "VmPin",
            "RssAnon",
            "RssFile",
            "RssShmem",
            "HugetlbPages",
        ];
        let mut line = format!("[status:{label}]");
        for key in keys {
            for l in body.lines() {
                if l.starts_with(&format!("{key}:")) {
                    let v = l.splitn(2, ':').nth(1).unwrap_or("").trim();
                    line.push_str(&format!("  {key}={v}"));
                }
            }
        }
        println!("{line}");
    }

    fn print_labeled_meminfo(label: &str) {
        let body = read_file("/proc/meminfo").unwrap_or_default();
        let keys = [
            "MemAvailable",
            "MemFree",
            "Cached",
            "SwapFree",
            "SwapTotal",
            "AnonPages",
            "Mapped",
            "Shmem",
            "SUnreclaim",
            "Mlocked",
        ];
        let mut line = format!("[mem:{label}]");
        for key in keys {
            for l in body.lines() {
                if l.starts_with(&format!("{key}:")) {
                    let v = l.splitn(2, ':').nth(1).unwrap_or("").trim();
                    line.push_str(&format!("  {key}={v}"));
                }
            }
        }
        // vmstat delta-ish values via /proc/vmstat keys we care about
        if let Some(vm) = read_file("/proc/vmstat").ok() {
            let mut extras = String::new();
            for key in &["pswpout", "pgmajfault", "pgsteal_file", "zswpout"] {
                for l in vm.lines() {
                    if let Some(rest) = l.strip_prefix(&format!("{key} ")) {
                        extras.push_str(&format!("  {key}={}", rest.trim()));
                    }
                }
            }
            line.push_str(&extras);
        }
        // zram orig_data_size (bytes) — raw decompressed footprint of compressed swap
        if let Some(zram) = read_file("/sys/block/zram0/mm_stat").ok() {
            let cols: Vec<&str> = zram.split_whitespace().collect();
            if let Some(orig) = cols.first() {
                line.push_str(&format!("  zram_orig={orig}"));
            }
        }
        println!("{line}");
    }

    fn read_file(path: &str) -> io::Result<String> {
        let mut f = std::fs::File::open(path)?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        Ok(s)
    }

    // Force stdout flushing so adb shell doesn't buffer the whole run.
    #[allow(dead_code)]
    fn flush() {
        let _ = io::stdout().flush();
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn main() {
    imp::main();
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn main() {
    eprintln!("rnb-mem-api-probe requires Linux/Android (memfd_create/ashmem).");
    std::process::exit(2);
}
