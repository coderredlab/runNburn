use super::super::*;

const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 8;

fn align_down(value: usize, alignment: usize) -> usize {
    value & !(alignment - 1)
}

fn align_up_checked(value: usize, alignment: usize) -> Result<usize, String> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "host register range overflow".to_string())
}

fn host_register_missing_ranges(
    registered: &[RegisteredHostRange],
    base: usize,
    end: usize,
) -> Vec<(usize, usize)> {
    let mut gaps = vec![(base, end)];
    for existing in registered {
        let existing_end = existing.base.saturating_add(existing.bytes);
        let mut next = Vec::with_capacity(gaps.len() + 1);
        for (gap_base, gap_end) in gaps {
            if existing_end <= gap_base || existing.base >= gap_end {
                next.push((gap_base, gap_end));
                continue;
            }
            if existing.base > gap_base {
                next.push((gap_base, existing.base.min(gap_end)));
            }
            if existing_end < gap_end {
                next.push((existing_end.max(gap_base), gap_end));
            }
        }
        gaps = next;
        if gaps.is_empty() {
            break;
        }
    }
    gaps
}

impl CudaState {
    pub(in crate::runtime) fn clear_host_registered_ranges(&mut self) -> Result<(), String> {
        if self.registered_host_ranges.is_empty() {
            return Ok(());
        }
        self.set_current()?;
        self.stream_synchronize()?;
        unsafe {
            self.api.stream_synchronize(self.copy_stream)?;
        }

        let ranges = std::mem::take(&mut self.registered_host_ranges);
        let range_count = ranges.len();
        let registered_bytes = ranges
            .iter()
            .fold(0usize, |sum, range| sum.saturating_add(range.bytes));
        let mut remaining = Vec::new();
        let mut first_error = None;
        for range in ranges {
            let result = unsafe {
                self.api
                    .mem_host_unregister(range.base as *mut libc::c_void)
            };
            if let Err(err) = result {
                if first_error.is_none() {
                    first_error = Some(format!(
                        "cuMemHostUnregister failed for base=0x{:x} bytes={}: {err}",
                        range.base, range.bytes
                    ));
                }
                remaining.push(range);
            }
        }
        self.registered_host_ranges = remaining;
        if let Some(err) = first_error {
            return Err(err);
        }
        if std::env::var("RNB_CUDA_HOST_REGISTER_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cuda-host-register] cleared_ranges={} cleared_mb={:.2}",
                range_count,
                registered_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Ok(())
    }

    pub(in crate::runtime) fn ensure_host_registered(
        &mut self,
        ptr: *const u8,
        bytes: usize,
    ) -> Result<(), String> {
        if bytes == 0 {
            return Ok(());
        }
        self.set_current()?;
        let start = ptr as usize;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| "host register source range overflow".to_string())?;
        let alignment = tuning::prefill_temp_host_register_granularity_bytes();
        let aligned_base = align_down(start, alignment);
        let aligned_end = align_up_checked(end, alignment)?;
        if aligned_end <= aligned_base {
            return Ok(());
        }

        let gaps =
            host_register_missing_ranges(&self.registered_host_ranges, aligned_base, aligned_end);
        if gaps.is_empty() {
            return Ok(());
        }
        for (gap_base, gap_end) in gaps {
            let gap_bytes = gap_end.saturating_sub(gap_base);
            if gap_bytes == 0 {
                continue;
            }
            unsafe {
                self.api.mem_host_register(
                    gap_base as *mut libc::c_void,
                    gap_bytes,
                    CU_MEMHOSTREGISTER_READ_ONLY,
                )?;
            }
            self.registered_host_ranges.push(RegisteredHostRange {
                base: gap_base,
                bytes: gap_bytes,
            });
        }
        self.registered_host_ranges
            .sort_unstable_by_key(|range| range.base);
        if std::env::var("RNB_CUDA_HOST_REGISTER_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            let registered_bytes = self
                .registered_host_ranges
                .iter()
                .fold(0usize, |sum, range| sum.saturating_add(range.bytes));
            eprintln!(
                "[cuda-host-register] ranges={} registered_mb={:.2}",
                self.registered_host_ranges.len(),
                registered_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_register_missing_ranges_split_existing_overlap() {
        let registered = [
            RegisteredHostRange {
                base: 0x2000,
                bytes: 0x2000,
            },
            RegisteredHostRange {
                base: 0x6000,
                bytes: 0x1000,
            },
        ];
        let gaps = host_register_missing_ranges(&registered, 0x1000, 0x8000);
        assert_eq!(
            gaps,
            vec![(0x1000, 0x2000), (0x4000, 0x6000), (0x7000, 0x8000)]
        );
    }

    #[test]
    fn host_register_missing_ranges_returns_empty_when_covered() {
        let registered = [RegisteredHostRange {
            base: 0x1000,
            bytes: 0x7000,
        }];
        assert!(host_register_missing_ranges(&registered, 0x2000, 0x7000).is_empty());
    }
}
