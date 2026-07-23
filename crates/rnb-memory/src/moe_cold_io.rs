//! Aligned direct-file reads for file-backed expert staging.
//!
//! This module owns reusable O_DIRECT and io_uring primitives used by backend
//! staging paths. It does not parse model containers or own expert residency
//! policy.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(any(target_os = "linux", target_os = "android"))]
use std::os::unix::fs::OpenOptionsExt;

/// 4 KB-aligned heap buffer.
pub struct AlignedBuf {
    raw: Vec<u8>,
    aligned_offset: usize,
    capacity: usize,
}

impl AlignedBuf {
    pub fn with_capacity(min_capacity: usize) -> Self {
        let aligned_cap = (min_capacity + 4095) & !4095;
        let raw = vec![0u8; aligned_cap + 4096];
        let raw_addr = raw.as_ptr() as usize;
        let aligned_offset = (4096 - (raw_addr & 4095)) & 4095;
        Self {
            raw,
            aligned_offset,
            capacity: aligned_cap,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn as_aligned_mut(&mut self, len: usize) -> &mut [u8] {
        assert!(
            len <= self.capacity,
            "len {} > buf capacity {}",
            len,
            self.capacity
        );
        let ptr = unsafe { self.raw.as_mut_ptr().add(self.aligned_offset) };
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

fn open_direct_read_file(path: &Path) -> io::Result<std::fs::File> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        opts.custom_flags(libc::O_DIRECT);
    }
    opts.open(path)
}

#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

pub struct DirectFileReadRequest<'a> {
    pub path: &'a Path,
    pub file_offset: u64,
    pub destination_offset: usize,
    pub read_len: usize,
    pub required_len: usize,
}

pub struct DirectFileReaderCache {
    files: HashMap<PathBuf, std::fs::File>,
    #[cfg(any(target_os = "linux", target_os = "android"))]
    uring: Option<io_uring::IoUring>,
    #[cfg(any(target_os = "linux", target_os = "android"))]
    uring_unavailable: bool,
}

impl Default for DirectFileReaderCache {
    fn default() -> Self {
        Self {
            files: HashMap::new(),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            uring: None,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            uring_unavailable: false,
        }
    }
}

impl DirectFileReaderCache {
    pub fn read_aligned(
        &mut self,
        path: &Path,
        file_offset: u64,
        destination: &mut [u8],
        required_len: usize,
    ) -> io::Result<usize> {
        const ALIGNMENT: usize = 4096;
        if !file_offset.is_multiple_of(ALIGNMENT as u64)
            || !(destination.as_ptr() as usize).is_multiple_of(ALIGNMENT)
            || !destination.len().is_multiple_of(ALIGNMENT)
            || required_len > destination.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "direct file read requires aligned offset, buffer, and length",
            ));
        }
        let file = if let Some(file) = self.files.get(path) {
            file
        } else {
            let file = open_direct_read_file(path)?;
            self.files.insert(path.to_path_buf(), file);
            self.files
                .get(path)
                .expect("direct file handle inserted before lookup")
        };
        loop {
            match read_at(file, destination, file_offset) {
                Ok(read) if read >= required_len => return Ok(read),
                Ok(read) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "short direct read: got {read} bytes, need {required_len} from {} at {file_offset}",
                            path.display()
                        ),
                    ))
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn ensure_io_uring(&mut self, queue_depth: usize) -> io::Result<()> {
        let queue_depth = queue_depth.max(1) as u32;
        if self
            .uring
            .as_ref()
            .is_some_and(|ring| ring.params().sq_entries() >= queue_depth)
        {
            return Ok(());
        }
        if self.uring_unavailable {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "io_uring is unavailable for this direct file reader",
            ));
        }
        let previous = self.uring.take();
        match io_uring::IoUring::new(queue_depth) {
            Ok(ring) => {
                self.uring = Some(ring);
                Ok(())
            }
            Err(_) if previous.is_some() => {
                self.uring = previous;
                Ok(())
            }
            Err(error) => {
                self.uring_unavailable = true;
                Err(error)
            }
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn read_aligned_batch(
        &mut self,
        requests: &[DirectFileReadRequest<'_>],
        destination: &mut [u8],
        queue_depth: usize,
    ) -> io::Result<usize> {
        use io_uring::{opcode, types};
        use std::os::fd::AsRawFd;

        const ALIGNMENT: usize = 4096;
        if requests.is_empty() {
            return Ok(0);
        }
        if !(destination.as_ptr() as usize).is_multiple_of(ALIGNMENT) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "direct file batch destination must be aligned",
            ));
        }
        let mut previous_end = 0usize;
        for request in requests {
            let end = request
                .destination_offset
                .checked_add(request.read_len)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "direct file batch destination range overflow",
                    )
                })?;
            if request.destination_offset < previous_end
                || end > destination.len()
                || !request.file_offset.is_multiple_of(ALIGNMENT as u64)
                || !request.destination_offset.is_multiple_of(ALIGNMENT)
                || !request.read_len.is_multiple_of(ALIGNMENT)
                || request.required_len > request.read_len
                || request.read_len > u32::MAX as usize
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "direct file batch requires ordered non-overlapping aligned ranges",
                ));
            }
            previous_end = end;
        }
        for request in requests {
            if !self.files.contains_key(request.path) {
                self.files.insert(
                    request.path.to_path_buf(),
                    open_direct_read_file(request.path)?,
                );
            }
        }

        let file_descriptors = requests
            .iter()
            .map(|request| {
                self.files
                    .get(request.path)
                    .expect("direct file handle inserted before batch lookup")
                    .as_raw_fd()
            })
            .collect::<Vec<_>>();
        let queue_depth = queue_depth.max(1).min(requests.len());
        self.ensure_io_uring(queue_depth)?;
        let ring = self
            .uring
            .as_mut()
            .expect("io_uring initialized before direct file batch");

        let queue_depth = queue_depth.min(ring.params().sq_entries() as usize);
        let mut total_bytes = 0usize;
        let destination_ptr = destination.as_mut_ptr();
        for batch_start in (0..requests.len()).step_by(queue_depth) {
            let batch_end = (batch_start + queue_depth).min(requests.len());
            {
                let mut submission = ring.submission();
                for index in batch_start..batch_end {
                    let request = &requests[index];
                    let entry = opcode::Read::new(
                        types::Fd(file_descriptors[index]),
                        unsafe { destination_ptr.add(request.destination_offset) },
                        request.read_len as u32,
                    )
                    .offset(request.file_offset)
                    .build()
                    .user_data(index as u64);
                    unsafe {
                        submission.push(&entry).map_err(|_| {
                            io::Error::other("io_uring submission queue filled unexpectedly")
                        })?;
                    }
                }
            }

            let batch_len = batch_end - batch_start;
            loop {
                match ring.submit_and_wait(batch_len) {
                    Ok(_) => break,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                }
            }
            let mut first_error = None;
            let mut completed = 0usize;
            {
                let mut completion = ring.completion();
                for entry in &mut completion {
                    completed += 1;
                    let index = entry.user_data() as usize;
                    if index < batch_start || index >= batch_end {
                        first_error.get_or_insert_with(|| {
                            io::Error::other("io_uring returned an unexpected request identifier")
                        });
                        continue;
                    }
                    let request = &requests[index];
                    let result = entry.result();
                    if result < 0 {
                        let source = io::Error::from_raw_os_error(-result);
                        first_error.get_or_insert_with(|| {
                            io::Error::new(
                                source.kind(),
                                format!(
                                    "io_uring direct read failed for {} at {}: {source}",
                                    request.path.display(),
                                    request.file_offset
                                ),
                            )
                        });
                    } else if (result as usize) < request.required_len {
                        first_error.get_or_insert_with(|| {
                            io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                format!(
                                    "short io_uring direct read: got {result} bytes, need {} from {} at {}",
                                    request.required_len,
                                    request.path.display(),
                                    request.file_offset
                                ),
                            )
                        });
                    } else {
                        total_bytes = total_bytes.saturating_add(result as usize);
                    }
                }
            }
            if completed != batch_len {
                return Err(io::Error::other(format!(
                    "io_uring completion count mismatch: got {completed}, expected {batch_len}"
                )));
            }
            if let Some(error) = first_error {
                return Err(error);
            }
        }
        Ok(total_bytes)
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub fn read_aligned_batch(
        &mut self,
        requests: &[DirectFileReadRequest<'_>],
        destination: &mut [u8],
        _queue_depth: usize,
    ) -> io::Result<usize> {
        let mut total_bytes = 0usize;
        for request in requests {
            let end = request
                .destination_offset
                .checked_add(request.read_len)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "direct file batch destination range overflow",
                    )
                })?;
            total_bytes += self.read_aligned(
                request.path,
                request.file_offset,
                &mut destination[request.destination_offset..end],
                request.required_len,
            )?;
        }
        Ok(total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn direct_file_reader_reads_aligned_batch_into_requested_offsets() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.as_file_mut().write_all(&vec![0x11; 4096]).unwrap();
        file.as_file_mut().write_all(&vec![0x22; 4096]).unwrap();
        file.as_file_mut().write_all(&vec![0x33; 4096]).unwrap();
        file.as_file_mut().sync_all().unwrap();

        let mut reader = DirectFileReaderCache::default();
        if reader.ensure_io_uring(2).is_err() {
            return;
        }
        let requests = [
            DirectFileReadRequest {
                path: file.path(),
                file_offset: 0,
                destination_offset: 0,
                read_len: 4096,
                required_len: 4096,
            },
            DirectFileReadRequest {
                path: file.path(),
                file_offset: 8192,
                destination_offset: 4096,
                read_len: 4096,
                required_len: 4096,
            },
        ];
        let mut buffer = AlignedBuf::with_capacity(8192);
        let destination = buffer.as_aligned_mut(8192);
        let read = reader
            .read_aligned_batch(&requests, destination, 2)
            .unwrap();

        assert_eq!(read, 8192);
        assert!(destination[..4096].iter().all(|&byte| byte == 0x11));
        assert!(destination[4096..].iter().all(|&byte| byte == 0x33));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn direct_file_reader_grows_io_uring_for_larger_batches() {
        let mut reader = DirectFileReaderCache::default();
        if reader.ensure_io_uring(1).is_err() {
            return;
        }
        let initial = reader.uring.as_ref().unwrap().params().sq_entries();
        let requested = initial.saturating_mul(2);

        reader.ensure_io_uring(requested as usize).unwrap();

        assert!(reader.uring.as_ref().unwrap().params().sq_entries() >= requested);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn direct_file_reader_rejects_overlapping_batch_destinations_before_open() {
        let mut reader = DirectFileReaderCache::default();
        let requests = [
            DirectFileReadRequest {
                path: Path::new("/path/need/not/exist"),
                file_offset: 0,
                destination_offset: 0,
                read_len: 4096,
                required_len: 4096,
            },
            DirectFileReadRequest {
                path: Path::new("/path/need/not/exist"),
                file_offset: 4096,
                destination_offset: 0,
                read_len: 4096,
                required_len: 4096,
            },
        ];
        let mut buffer = AlignedBuf::with_capacity(8192);
        let error = reader
            .read_aligned_batch(&requests, buffer.as_aligned_mut(8192), 2)
            .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
    #[test]
    fn aligned_buf_returns_4kb_aligned_pointer() {
        let mut buf = AlignedBuf::with_capacity(8 * 1024);
        let s = buf.as_aligned_mut(4096);
        assert_eq!(s.as_ptr() as usize & 4095, 0);
        assert_eq!(s.len(), 4096);
    }

    #[test]
    fn aligned_buf_capacity_rounds_up() {
        let buf = AlignedBuf::with_capacity(100);
        assert!(buf.capacity >= 4096);
    }

    #[test]
    fn direct_file_reader_rejects_unaligned_offsets_before_open() {
        let mut reader = DirectFileReaderCache::default();
        let mut buffer = AlignedBuf::with_capacity(4096);
        let error = reader
            .read_aligned(
                Path::new("/path/need/not/exist"),
                1,
                buffer.as_aligned_mut(4096),
                4096,
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}
