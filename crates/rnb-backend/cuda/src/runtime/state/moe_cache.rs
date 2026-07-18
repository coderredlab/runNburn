use super::super::*;
use std::sync::atomic::Ordering;

impl CudaState {
    pub(in crate::runtime) fn register_qwen35_moe_layer(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<bool, String> {
        self.register_qwen35_moe_layer_with_policy(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd, true,
        )
    }

    pub(in crate::runtime) fn register_qwen35_moe_layer_without_eviction(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<bool, String> {
        self.register_qwen35_moe_layer_with_policy(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd, false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_qwen35_moe_layer_with_policy(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        allow_eviction: bool,
    ) -> Result<bool, String> {
        self.set_current()?;
        if self.resident_moe_layer_limit == 0 {
            return Ok(false);
        }
        let key = qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        if self.resident_moe_layers.contains_key(&key) {
            self.touch_resident_moe_layer(key);
            return Ok(false);
        }
        let bytes = gate_all
            .len()
            .saturating_add(up_all.len())
            .saturating_add(down_all.len());
        if bytes == 0 || self.resident_moe_layer_limit < bytes {
            return Ok(false);
        }
        let effective_limit = qwen35_moe_layer_effective_limit(
            self.resident_moe_layer_limit,
            bytes,
            tuning::moe_layer_cache_enabled(),
        );
        if effective_limit < bytes {
            return Ok(false);
        }
        if allow_eviction {
            self.evict_resident_moe_layers_until(bytes, effective_limit)?;
        }
        if self.resident_moe_layer_bytes.saturating_add(bytes) > effective_limit {
            return Ok(false);
        }

        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        let gate_base = ptr;
        let up_base = gate_base + gate_all.len() as u64;
        let down_base = up_base + up_all.len() as u64;
        let upload = unsafe {
            self.api
                .memcpy_htod_async(
                    gate_base,
                    gate_all.as_ptr().cast::<libc::c_void>(),
                    gate_all.len(),
                    self.stream,
                )
                .and_then(|_| {
                    self.api.memcpy_htod_async(
                        up_base,
                        up_all.as_ptr().cast::<libc::c_void>(),
                        up_all.len(),
                        self.stream,
                    )
                })
                .and_then(|_| {
                    self.api.memcpy_htod_async(
                        down_base,
                        down_all.as_ptr().cast::<libc::c_void>(),
                        down_all.len(),
                        self.stream,
                    )
                })
        };
        if let Err(err) = upload {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }
        self.stream_synchronize()?;
        let epoch = self.next_resident_moe_layer_epoch();
        self.resident_moe_layers.insert(
            key,
            ResidentMoeLayer {
                gate_base,
                up_base,
                down_base,
                ptr,
                bytes,
                epoch,
            },
        );
        self.resident_moe_layer_lru.push_back((key, epoch));
        self.resident_moe_layer_bytes = self.resident_moe_layer_bytes.saturating_add(bytes);
        if std::env::var("RNB_CUDA_MOE_LAYER_CACHE_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cuda] resident MoE layer cached bytes={} used={} limit={}",
                bytes, self.resident_moe_layer_bytes, effective_limit
            );
        }
        Ok(true)
    }

    pub(in crate::runtime) fn register_nemotron_q5_layer(
        &mut self,
        up_all: &[u8],
        down_all: &[u8],
        _n_expert: usize,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<bool, String> {
        self.set_current()?;
        if self.resident_moe_layer_limit == 0 {
            return Ok(false);
        }
        let key = nemotron_q5_layer_key(up_all, down_all, n_ff, n_embd);
        if self.resident_moe_layers.contains_key(&key) {
            self.touch_resident_moe_layer(key);
            return Ok(false);
        }
        let bytes = up_all.len().saturating_add(down_all.len());
        if bytes == 0 || self.resident_moe_layer_limit < bytes {
            return Ok(false);
        }
        self.evict_resident_moe_layers_until(bytes, self.resident_moe_layer_limit)?;
        if self.resident_moe_layer_bytes.saturating_add(bytes) > self.resident_moe_layer_limit {
            return Ok(false);
        }

        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        let up_base = ptr;
        let down_base = up_base + up_all.len() as u64;
        let upload = unsafe {
            self.api
                .memcpy_htod_async(
                    up_base,
                    up_all.as_ptr().cast::<libc::c_void>(),
                    up_all.len(),
                    self.stream,
                )
                .and_then(|_| {
                    self.api.memcpy_htod_async(
                        down_base,
                        down_all.as_ptr().cast::<libc::c_void>(),
                        down_all.len(),
                        self.stream,
                    )
                })
        };
        if let Err(err) = upload {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }
        self.stream_synchronize()?;
        let epoch = self.next_resident_moe_layer_epoch();
        self.resident_moe_layers.insert(
            key,
            ResidentMoeLayer {
                gate_base: 0,
                up_base,
                down_base,
                ptr,
                bytes,
                epoch,
            },
        );
        self.resident_moe_layer_lru.push_back((key, epoch));
        self.resident_moe_layer_bytes = self.resident_moe_layer_bytes.saturating_add(bytes);
        cache_stats()
            .resident_upload_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        if std::env::var("RNB_CUDA_MOE_LAYER_CACHE_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cuda] resident Nemotron Q5 layer cached bytes={} used={} limit={}",
                bytes, self.resident_moe_layer_bytes, self.resident_moe_layer_limit
            );
        }
        Ok(true)
    }

    pub(in crate::runtime) fn register_nemotron_q5_q8_layer(
        &mut self,
        up_all: &[u8],
        down_all: &[u8],
        _n_expert: usize,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<bool, String> {
        self.register_nemotron_layer_with_down_quant(up_all, down_all, n_ff, n_embd, 80)
    }

    pub(in crate::runtime) fn register_nemotron_layer_with_down_quant(
        &mut self,
        up_all: &[u8],
        down_all: &[u8],
        n_ff: usize,
        n_embd: usize,
        down_quant: u32,
    ) -> Result<bool, String> {
        self.set_current()?;
        if self.resident_moe_layer_limit == 0 {
            return Ok(false);
        }
        let key = nemotron_layer_key(up_all, down_all, n_ff, n_embd, down_quant);
        if self.resident_moe_layers.contains_key(&key) {
            self.touch_resident_moe_layer(key);
            return Ok(false);
        }
        let bytes = up_all.len().saturating_add(down_all.len());
        if bytes == 0 || self.resident_moe_layer_limit < bytes {
            return Ok(false);
        }
        self.evict_resident_moe_layers_until(bytes, self.resident_moe_layer_limit)?;
        if self.resident_moe_layer_bytes.saturating_add(bytes) > self.resident_moe_layer_limit {
            return Ok(false);
        }

        let ptr = unsafe { self.api.mem_alloc(bytes) }?;
        let up_base = ptr;
        let down_base = up_base + up_all.len() as u64;
        let upload = unsafe {
            self.api
                .memcpy_htod_async(
                    up_base,
                    up_all.as_ptr().cast::<libc::c_void>(),
                    up_all.len(),
                    self.stream,
                )
                .and_then(|_| {
                    self.api.memcpy_htod_async(
                        down_base,
                        down_all.as_ptr().cast::<libc::c_void>(),
                        down_all.len(),
                        self.stream,
                    )
                })
        };
        if let Err(err) = upload {
            let _ = unsafe { self.api.mem_free(ptr) };
            return Err(err);
        }
        self.stream_synchronize()?;
        let epoch = self.next_resident_moe_layer_epoch();
        self.resident_moe_layers.insert(
            key,
            ResidentMoeLayer {
                gate_base: 0,
                up_base,
                down_base,
                ptr,
                bytes,
                epoch,
            },
        );
        self.resident_moe_layer_lru.push_back((key, epoch));
        self.resident_moe_layer_bytes = self.resident_moe_layer_bytes.saturating_add(bytes);
        Ok(true)
    }

    pub(in crate::runtime) fn touch_resident_moe_layer(&mut self, key: Qwen35MoeLayerKey) {
        let epoch = self.next_resident_moe_layer_epoch();
        if let Some(entry) = self.resident_moe_layers.get_mut(&key) {
            entry.epoch = epoch;
            self.resident_moe_layer_lru.push_back((key, epoch));
        }
    }

    pub(in crate::runtime) fn next_resident_moe_layer_epoch(&mut self) -> u64 {
        self.resident_moe_layer_epoch = self.resident_moe_layer_epoch.wrapping_add(1);
        if self.resident_moe_layer_epoch == 0 {
            self.resident_moe_layer_epoch = 1;
        }
        self.resident_moe_layer_epoch
    }

    pub(in crate::runtime) fn evict_resident_moe_layers_until(
        &mut self,
        incoming: usize,
        limit: usize,
    ) -> Result<(), String> {
        let mut synced = false;
        while self.resident_moe_layer_bytes.saturating_add(incoming) > limit {
            let Some((key, epoch)) = self.resident_moe_layer_lru.pop_front() else {
                break;
            };
            if self
                .resident_moe_layers
                .get(&key)
                .is_some_and(|entry| entry.epoch != epoch)
            {
                continue;
            }
            if let Some(entry) = self.resident_moe_layers.remove(&key) {
                // cu27: 다른 layer/path의 in-flight kernel이 이 ptr를 launch
                // param으로 잡고 있을 수 있어서 첫 free 직전에 한 번 sync.
                // Llama 8B 100-decode 도중 LRU eviction이 자주 일어나면서
                // cuMemcpyDtoHAsync CUDA 700 (illegal address)가 발생함.
                if !synced {
                    self.stream_synchronize()?;
                    synced = true;
                }
                unsafe { self.api.mem_free(entry.ptr)? };
                self.resident_moe_layer_bytes =
                    self.resident_moe_layer_bytes.saturating_sub(entry.bytes);
                if std::env::var("RNB_CUDA_MOE_LAYER_CACHE_TRACE")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    eprintln!(
                        "[cuda] resident MoE layer evicted bytes={} used={} limit={}",
                        entry.bytes, self.resident_moe_layer_bytes, limit
                    );
                }
            }
        }
        Ok(())
    }

    pub(in crate::runtime) fn clear_resident_moe_layer_cache(&mut self) -> Result<(), String> {
        self.set_current()?;
        // cu27: in-flight launch가 resident MoE ptr를 kernel param으로 들고 있을 수
        // 있어서 mem_free 전에 stream sync 보장. 안 그러면 후속 kernel이 freed
        // ptr 접근해서 CUDA 700 illegal address (Llama 8B OOM retry path).
        if !self.resident_moe_layers.is_empty() {
            self.stream_synchronize()?;
        }
        let released = self.resident_moe_layer_bytes;
        for (_, entry) in self.resident_moe_layers.drain() {
            unsafe { self.api.mem_free(entry.ptr)? };
        }
        self.resident_moe_layer_lru.clear();
        self.resident_moe_layer_epoch = 0;
        self.resident_moe_layer_bytes = 0;
        if released > 0 {
            self.qwen35_target_decode_q4k_limit_checked = false;
            self.nemotron_decode_q4k_limit_checked = false;
        }
        if released > 0
            && std::env::var("RNB_CUDA_MOE_LAYER_CACHE_TRACE")
                .ok()
                .as_deref()
                == Some("1")
        {
            eprintln!("[cuda] resident MoE layer cache released bytes={released}");
        }
        Ok(())
    }
}
