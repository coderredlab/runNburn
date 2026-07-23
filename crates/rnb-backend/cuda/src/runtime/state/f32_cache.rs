use super::super::*;

impl CudaState {
    pub(in crate::runtime) fn resident_f32_ptr(&mut self, data: &[f32]) -> Result<u64, String> {
        let key = f32_key(data);
        self.resident_f32_ptr_for_key(data, key)
    }

    /// Uses allocation identity instead of hashing the full slice on every lookup.
    /// The caller must keep this source allocation alive and immutable for this
    /// `CudaState`'s lifetime.
    pub(in crate::runtime) fn resident_f32_ptr_stable_source(
        &mut self,
        data: &[f32],
    ) -> Result<u64, String> {
        let key = F32Key {
            ptr: data.as_ptr() as usize,
            len: data.len(),
            bit_hash: 0,
        };
        self.resident_f32_ptr_for_key(data, key)
    }

    fn resident_f32_ptr_for_key(&mut self, data: &[f32], key: F32Key) -> Result<u64, String> {
        if data.is_empty() {
            return Err("empty F32 resident upload".to_string());
        }
        if let Some(entry) = self.resident_f32.get(&key) {
            return Ok(entry.ptr);
        }
        let bytes = std::mem::size_of_val(data);
        self.reclaim_residency_for_transient(bytes)?;
        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                data.as_ptr().cast::<libc::c_void>(),
                bytes,
                self.stream,
            )?;
        }
        self.resident_f32.insert(key, ResidentF32 { ptr });
        self.record_native_f32_residency(bytes);
        Ok(ptr)
    }

    pub(in crate::runtime) fn upload_gemma_ple_base(&mut self, data: &[f32]) -> Result<(), String> {
        if data.is_empty() {
            return Err("empty Gemma PLE base upload".to_string());
        }
        let bytes = std::mem::size_of_val(data);
        let ptr = self.gemma_ple_base_ptr(bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                ptr,
                data.as_ptr().cast::<libc::c_void>(),
                bytes,
                self.stream,
            )?;
        }
        self.gemma_ple_base_len = data.len();
        Ok(())
    }

    pub(in crate::runtime) fn gemma_ple_base_slice_ptr(
        &mut self,
        offset: usize,
        len: usize,
    ) -> Result<u64, String> {
        if len == 0 {
            return Err("empty Gemma PLE base slice".to_string());
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| format!("Gemma PLE base slice overflow: offset={offset} len={len}"))?;
        if end > self.gemma_ple_base_len {
            return Err(format!(
                "Gemma PLE base slice out of range: offset={offset} len={len} base_len={}",
                self.gemma_ple_base_len
            ));
        }
        let Some(base) = self.gemma_ple_base else {
            return Err("Gemma PLE base is not uploaded".to_string());
        };
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("Gemma PLE base byte offset overflow: offset={offset}"))?;
        Ok(base + byte_offset as u64)
    }
}
