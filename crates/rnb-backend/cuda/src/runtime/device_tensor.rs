use super::*;
use rnb_backend_api::{BackendKind, DeviceTensorDesc, DeviceTensorId, ScalarType};

impl CudaState {
    pub(in crate::runtime) fn insert_device_tensor_slot(
        &mut self,
        ptr: u64,
        capacity: usize,
        desc: DeviceTensorDesc,
    ) -> Result<DeviceTensorId, String> {
        self.insert_device_tensor_slot_with_storage(ptr, capacity, desc, DeviceTensorStorage::Owned)
    }

    pub(in crate::runtime) fn insert_device_tensor_slot_with_storage(
        &mut self,
        ptr: u64,
        capacity: usize,
        desc: DeviceTensorDesc,
        storage: DeviceTensorStorage,
    ) -> Result<DeviceTensorId, String> {
        let raw = self.next_device_tensor_id;
        let next_raw = match raw.checked_add(1) {
            Some(next_raw) => next_raw,
            None => {
                if storage.is_owned() {
                    let _ = unsafe { self.api.mem_free(ptr) };
                }
                return Err("CUDA device tensor id overflow".to_string());
            }
        };
        self.next_device_tensor_id = next_raw;
        self.device_tensors.insert(
            raw,
            DeviceTensorSlot {
                ptr,
                capacity,
                desc,
                storage,
            },
        );
        Ok(DeviceTensorId::new(BackendKind::Cuda, raw))
    }

    pub(in crate::runtime) fn upload_device_tensor_f32(
        &mut self,
        desc: DeviceTensorDesc,
        input: &[f32],
    ) -> Result<DeviceTensorId, String> {
        if desc.dtype() != ScalarType::F32 {
            return Err(format!(
                "CUDA device tensor upload expects F32, got {:?}",
                desc.dtype()
            ));
        }
        let bytes = desc
            .byte_len()
            .ok_or_else(|| "CUDA device tensor upload byte overflow".to_string())?;
        if input.len() != desc.len() {
            return Err(format!(
                "CUDA device tensor upload len mismatch: got {}, expected {}",
                input.len(),
                desc.len()
            ));
        }
        self.set_current()?;
        let ptr = unsafe { self.api.mem_alloc(bytes)? };
        let upload = unsafe {
            self.api.memcpy_htod_async(
                ptr,
                input.as_ptr().cast::<libc::c_void>(),
                bytes,
                self.stream,
            )
        }
        .and_then(|_| self.stream_synchronize());
        if let Err(err) = upload {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }
        self.insert_device_tensor_slot(ptr, bytes, desc)
    }

    pub(in crate::runtime) fn download_device_tensor_f32(
        &mut self,
        id: DeviceTensorId,
    ) -> Result<Vec<f32>, String> {
        if id.backend() != BackendKind::Cuda {
            return Err(format!(
                "CUDA device tensor backend mismatch: {:?}",
                id.backend()
            ));
        }
        let slot = self
            .device_tensors
            .get(&id.raw())
            .ok_or_else(|| format!("missing CUDA device tensor id {}", id.raw()))?;
        if slot.desc.dtype() != ScalarType::F32 {
            return Err(format!(
                "CUDA device tensor download expects F32, got {:?}",
                slot.desc.dtype()
            ));
        }
        let bytes = slot
            .desc
            .byte_len()
            .ok_or_else(|| "CUDA device tensor download byte overflow".to_string())?;
        if bytes > slot.capacity {
            return Err(format!(
                "CUDA device tensor download bytes exceed allocation: bytes={}, capacity={}",
                bytes, slot.capacity
            ));
        }
        let mut output = vec![0.0f32; slot.desc.len()];
        self.set_current()?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                slot.ptr,
                bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(in crate::runtime) fn download_device_tensor_f32_row(
        &mut self,
        id: DeviceTensorId,
        row: usize,
    ) -> Result<Vec<f32>, String> {
        if id.backend() != BackendKind::Cuda {
            return Err(format!(
                "CUDA device tensor backend mismatch: {:?}",
                id.backend()
            ));
        }
        let slot = self
            .device_tensors
            .get(&id.raw())
            .ok_or_else(|| format!("missing CUDA device tensor id {}", id.raw()))?;
        if slot.desc.dtype() != ScalarType::F32 {
            return Err(format!(
                "CUDA device tensor row download expects F32, got {:?}",
                slot.desc.dtype()
            ));
        }
        if row >= slot.desc.rows() {
            return Err(format!(
                "CUDA device tensor row out of range: row={}, rows={}",
                row,
                slot.desc.rows()
            ));
        }
        let row_bytes = slot
            .desc
            .cols()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "CUDA device tensor row byte overflow".to_string())?;
        let offset = row
            .checked_mul(row_bytes)
            .ok_or_else(|| "CUDA device tensor row offset overflow".to_string())?;
        let end = offset
            .checked_add(row_bytes)
            .ok_or_else(|| "CUDA device tensor row end overflow".to_string())?;
        if end > slot.capacity {
            return Err(format!(
                "CUDA device tensor row download bytes exceed allocation: end={}, capacity={}",
                end, slot.capacity
            ));
        }
        let src = slot
            .ptr
            .checked_add(offset as u64)
            .ok_or_else(|| "CUDA device tensor row pointer overflow".to_string())?;
        let mut output = vec![0.0f32; slot.desc.cols()];
        self.set_current()?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                src,
                row_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(in crate::runtime) fn release_device_tensor(
        &mut self,
        id: rnb_backend_api::DeviceTensorId,
    ) -> Result<bool, String> {
        if id.backend() != BackendKind::Cuda {
            return Err(format!(
                "CUDA device tensor backend mismatch: {:?}",
                id.backend()
            ));
        }
        let Some(slot) = self.device_tensors.remove(&id.raw()) else {
            return Ok(false);
        };
        self.set_current()?;
        match slot.storage {
            DeviceTensorStorage::Owned => unsafe { self.api.mem_free(slot.ptr)? },
            DeviceTensorStorage::NemotronWorkspace {
                arena_id,
                offset,
                bytes,
            } => self.release_nemotron_prefill_workspace_lease(arena_id, offset, bytes)?,
        }
        Ok(true)
    }

    pub(in crate::runtime) fn retag_device_tensor_slot(
        &mut self,
        id: DeviceTensorId,
        expected_desc: DeviceTensorDesc,
        new_desc: DeviceTensorDesc,
    ) -> Result<(), String> {
        if id.backend() != BackendKind::Cuda {
            return Err(format!(
                "CUDA device tensor backend mismatch: {:?}",
                id.backend()
            ));
        }
        let slot = self
            .device_tensors
            .get_mut(&id.raw())
            .ok_or_else(|| format!("missing CUDA device tensor id {}", id.raw()))?;
        if slot.desc != expected_desc {
            return Err(format!(
                "CUDA device tensor retag desc mismatch: got {:?}, expected {:?}",
                slot.desc, expected_desc
            ));
        }
        let bytes = new_desc
            .byte_len()
            .ok_or_else(|| "CUDA device tensor retag byte overflow".to_string())?;
        if bytes > slot.capacity {
            return Err(format!(
                "CUDA device tensor retag bytes exceed allocation: bytes={}, capacity={}",
                bytes, slot.capacity
            ));
        }
        slot.desc = new_desc;
        Ok(())
    }

    #[allow(dead_code)]
    pub(in crate::runtime) fn device_tensor_ptr(
        &self,
        id: DeviceTensorId,
        desc: DeviceTensorDesc,
    ) -> Result<u64, String> {
        if id.backend() != BackendKind::Cuda {
            return Err(format!(
                "CUDA device tensor backend mismatch: {:?}",
                id.backend()
            ));
        }
        let slot = self
            .device_tensors
            .get(&id.raw())
            .ok_or_else(|| format!("missing CUDA device tensor id {}", id.raw()))?;
        if slot.desc != desc {
            return Err(format!(
                "CUDA device tensor desc mismatch: got {:?}, expected {:?}",
                slot.desc, desc
            ));
        }
        Ok(slot.ptr)
    }
}
