use super::super::super::*;
use rnb_backend_api::{DeviceTensorDesc, DeviceTensorId, DeviceTensorRole, ScalarType};

#[derive(Debug, Clone, Copy)]
pub struct NemotronDeviceRouterLogitsOutput {
    pub normalized_id: DeviceTensorId,
    pub normalized_desc: DeviceTensorDesc,
    pub router_logits_id: DeviceTensorId,
    pub router_logits_desc: DeviceTensorDesc,
}

#[derive(Debug, Clone, Copy)]
pub struct NemotronDeviceRoutePackOutput {
    pub expert_ids_dev: u64,
    pub route_weights_dev: u64,
    pub token_ids_dev: u64,
    pub slots: usize,
    pub seq_len: usize,
    pub expert_used: usize,
    pub storage: NemotronRoutePackStorage,
}

impl CudaState {
    fn cleanup_nemotron_route_pack_allocations(
        &mut self,
        expert_ids_dev: u64,
        route_weights_dev: u64,
        token_ids_dev: u64,
        storage: NemotronRoutePackStorage,
    ) -> Result<(), String> {
        match storage {
            NemotronRoutePackStorage::Owned => unsafe {
                self.api.mem_free(expert_ids_dev)?;
                self.api.mem_free(route_weights_dev)?;
                self.api.mem_free(token_ids_dev)?;
            },
            // Route pack storage uses fixed original/reordered slices inside the active
            // prefill arena and does not take a live lease.
            NemotronRoutePackStorage::Workspace { .. } => {}
        }
        Ok(())
    }

    fn cleanup_nemotron_router_tensor_allocation(
        &mut self,
        ptr: u64,
        storage: DeviceTensorStorage,
    ) -> Result<(), String> {
        match storage {
            DeviceTensorStorage::Owned => unsafe { self.api.mem_free(ptr)? },
            DeviceTensorStorage::NemotronWorkspace {
                arena_id,
                offset,
                bytes,
            } => self.release_nemotron_prefill_workspace_lease(arena_id, offset, bytes)?,
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_device_route_pack_from_logits(
        &mut self,
        router_logits_id: DeviceTensorId,
        router_logits_desc: DeviceTensorDesc,
        bias: Option<&[f32]>,
        seq_len: usize,
        n_expert: usize,
        expert_used: usize,
        expert_weight_scale: f32,
    ) -> Result<NemotronDeviceRoutePackOutput, String> {
        if router_logits_desc.rows() != seq_len || router_logits_desc.cols() != n_expert {
            return Err(format!(
                "Nemotron route pack logits shape mismatch: input={}x{} expected {}x{}",
                router_logits_desc.rows(),
                router_logits_desc.cols(),
                seq_len,
                n_expert
            ));
        }
        if router_logits_desc.dtype() != ScalarType::F32 {
            return Err(format!(
                "Nemotron route pack expects F32 logits, got {:?}",
                router_logits_desc.dtype()
            ));
        }
        if let Some(bias) = bias {
            if bias.len() != n_expert {
                return Err(format!(
                    "Nemotron route pack bias len mismatch: got {}, expected {n_expert}",
                    bias.len()
                ));
            }
        }
        if expert_used == 0 || expert_used > 32 {
            return Err(format!(
                "Nemotron route pack selected expert count must be 1..=32, got {expert_used}"
            ));
        }
        let slots = seq_len.checked_mul(expert_used).ok_or_else(|| {
            format!(
                "Nemotron route pack slot overflow: seq_len={seq_len} expert_used={expert_used}"
            )
        })?;
        let ids_bytes = slots
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| format!("Nemotron route pack id byte overflow: slots={slots}"))?;
        let weights_bytes = slots
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("Nemotron route pack weight byte overflow: slots={slots}"))?;
        let logits_ptr = self.device_tensor_ptr(router_logits_id, router_logits_desc)?;
        let bias_dev = match bias {
            Some(bias) => Some(self.resident_f32_ptr(bias)?),
            None => None,
        };
        self.set_current()?;
        let (expert_ids_dev, route_weights_dev, token_ids_dev, storage) =
            match self.nemotron_workspace_route_pack_ptrs(false, ids_bytes, weights_bytes) {
                Some((expert_ids_dev, route_weights_dev, token_ids_dev, storage)) => {
                    (expert_ids_dev, route_weights_dev, token_ids_dev, storage)
                }
                None => {
                    let expert_ids_dev = unsafe { self.api.mem_alloc(ids_bytes)? };
                    let route_weights_dev = match unsafe { self.api.mem_alloc(weights_bytes) } {
                        Ok(ptr) => ptr,
                        Err(err) => {
                            let _ = unsafe { self.api.mem_free(expert_ids_dev) };
                            return Err(err);
                        }
                    };
                    let token_ids_dev = match unsafe { self.api.mem_alloc(ids_bytes) } {
                        Ok(ptr) => ptr,
                        Err(err) => {
                            let _ = unsafe { self.api.mem_free(expert_ids_dev) };
                            let _ = unsafe { self.api.mem_free(route_weights_dev) };
                            return Err(err);
                        }
                    };
                    (
                        expert_ids_dev,
                        route_weights_dev,
                        token_ids_dev,
                        NemotronRoutePackStorage::Owned,
                    )
                }
            };
        let run_result = self.launch_nemotron_sigmoid_topk_route_f32(
            logits_ptr,
            bias_dev,
            expert_ids_dev,
            route_weights_dev,
            token_ids_dev,
            seq_len,
            n_expert,
            expert_used,
            expert_weight_scale,
        );
        if let Err(err) = run_result {
            let _ = self.cleanup_nemotron_route_pack_allocations(
                expert_ids_dev,
                route_weights_dev,
                token_ids_dev,
                storage,
            );
            return Err(err);
        }
        Ok(NemotronDeviceRoutePackOutput {
            expert_ids_dev,
            route_weights_dev,
            token_ids_dev,
            slots,
            seq_len,
            expert_used,
            storage,
        })
    }

    pub(in crate::runtime) fn release_nemotron_device_route_pack(
        &mut self,
        route: NemotronDeviceRoutePackOutput,
    ) -> Result<(), String> {
        self.cleanup_nemotron_route_pack_allocations(
            route.expert_ids_dev,
            route.route_weights_dev,
            route.token_ids_dev,
            route.storage,
        )
    }

    pub(in crate::runtime) fn nemotron_reorder_device_route_pack(
        &mut self,
        route: NemotronDeviceRoutePackOutput,
        order_indices: &[u32],
    ) -> Result<NemotronDeviceRoutePackOutput, String> {
        if order_indices.len() != route.slots {
            return Err(format!(
                "Nemotron route reorder order length mismatch: got {}, expected {}",
                order_indices.len(),
                route.slots
            ));
        }
        if order_indices.iter().any(|&idx| idx as usize >= route.slots) {
            return Err(format!(
                "Nemotron route reorder order index out of range: slots={}",
                route.slots
            ));
        }
        let ids_bytes = route
            .slots
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                format!(
                    "Nemotron route reorder id byte overflow: slots={}",
                    route.slots
                )
            })?;
        let weights_bytes = route
            .slots
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Nemotron route reorder weight byte overflow: slots={}",
                    route.slots
                )
            })?;
        self.set_current()?;
        let order_dev = self.compute_group_meta_ptr(std::mem::size_of_val(order_indices))?;
        unsafe {
            self.api.memcpy_htod_async(
                order_dev,
                order_indices.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(order_indices),
                self.stream,
            )?;
        }
        let (expert_ids_dev, route_weights_dev, token_ids_dev, storage) =
            match self.nemotron_workspace_route_pack_ptrs(true, ids_bytes, weights_bytes) {
                Some((expert_ids_dev, route_weights_dev, token_ids_dev, storage)) => {
                    (expert_ids_dev, route_weights_dev, token_ids_dev, storage)
                }
                None => {
                    let expert_ids_dev = unsafe { self.api.mem_alloc(ids_bytes)? };
                    let route_weights_dev = match unsafe { self.api.mem_alloc(weights_bytes) } {
                        Ok(ptr) => ptr,
                        Err(err) => {
                            let _ = unsafe { self.api.mem_free(expert_ids_dev) };
                            return Err(err);
                        }
                    };
                    let token_ids_dev = match unsafe { self.api.mem_alloc(ids_bytes) } {
                        Ok(ptr) => ptr,
                        Err(err) => {
                            let _ = unsafe { self.api.mem_free(expert_ids_dev) };
                            let _ = unsafe { self.api.mem_free(route_weights_dev) };
                            return Err(err);
                        }
                    };
                    (
                        expert_ids_dev,
                        route_weights_dev,
                        token_ids_dev,
                        NemotronRoutePackStorage::Owned,
                    )
                }
            };
        let run_result = self.launch_nemotron_reorder_route_slots(
            route.expert_ids_dev,
            route.route_weights_dev,
            route.token_ids_dev,
            order_dev,
            expert_ids_dev,
            route_weights_dev,
            token_ids_dev,
            route.slots,
        );
        if let Err(err) = run_result {
            let _ = self.cleanup_nemotron_route_pack_allocations(
                expert_ids_dev,
                route_weights_dev,
                token_ids_dev,
                storage,
            );
            return Err(err);
        }
        Ok(NemotronDeviceRoutePackOutput {
            expert_ids_dev,
            route_weights_dev,
            token_ids_dev,
            slots: route.slots,
            seq_len: route.seq_len,
            expert_used: route.expert_used,
            storage,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_router_logits_from_device_f32(
        &mut self,
        input_id: DeviceTensorId,
        input_desc: DeviceTensorDesc,
        norm_weight: &[f32],
        router_weight_f32: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        n_expert: usize,
        norm_eps: f32,
    ) -> Result<NemotronDeviceRouterLogitsOutput, String> {
        if input_desc.rows() != seq_len || input_desc.cols() != hidden_dim {
            return Err(format!(
                "Nemotron router logits shape mismatch: input={}x{} expected {}x{}",
                input_desc.rows(),
                input_desc.cols(),
                seq_len,
                hidden_dim
            ));
        }
        if input_desc.dtype() != ScalarType::F32 {
            return Err(format!(
                "Nemotron router logits expects F32 input, got {:?}",
                input_desc.dtype()
            ));
        }
        if norm_weight.len() != hidden_dim {
            return Err(format!(
                "Nemotron router norm length mismatch: got {}, expected {}",
                norm_weight.len(),
                hidden_dim
            ));
        }
        let router_weight_len = n_expert
            .checked_mul(hidden_dim)
            .ok_or_else(|| "Nemotron router F32 weight length overflow".to_string())?;
        if router_weight_f32.len() != router_weight_len {
            return Err(format!(
                "Nemotron router F32 weight length mismatch: got {}, expected {}",
                router_weight_f32.len(),
                router_weight_len
            ));
        }

        let normalized_desc = DeviceTensorDesc::new(
            seq_len,
            hidden_dim,
            ScalarType::F32,
            DeviceTensorRole::Normalized,
        );
        let router_logits_desc = DeviceTensorDesc::new(
            seq_len,
            n_expert,
            ScalarType::F32,
            DeviceTensorRole::RouterLogits,
        );

        let input_ptr = self.device_tensor_ptr(input_id, input_desc)?;
        let normalized_bytes = normalized_desc
            .byte_len()
            .ok_or_else(|| "Nemotron normalized byte overflow".to_string())?;
        let logits_bytes = router_logits_desc
            .byte_len()
            .ok_or_else(|| "Nemotron router logits byte overflow".to_string())?;
        self.set_current()?;
        let (normalized_ptr, normalized_storage, logits_ptr, logits_storage) =
            match self.nemotron_workspace_router_logits_ptrs(normalized_bytes, logits_bytes) {
                Some(((normalized_ptr, normalized_storage), (logits_ptr, logits_storage))) => (
                    normalized_ptr,
                    normalized_storage,
                    logits_ptr,
                    logits_storage,
                ),
                None => {
                    let normalized_ptr = unsafe { self.api.mem_alloc(normalized_bytes)? };
                    let logits_ptr = match unsafe { self.api.mem_alloc(logits_bytes) } {
                        Ok(ptr) => ptr,
                        Err(err) => {
                            let cleanup = unsafe { self.api.mem_free(normalized_ptr) };
                            return match cleanup {
                                Ok(()) => Err(err),
                                Err(cleanup_err) => {
                                    Err(format!("{err}; cleanup failed: {cleanup_err}"))
                                }
                            };
                        }
                    };
                    (
                        normalized_ptr,
                        DeviceTensorStorage::Owned,
                        logits_ptr,
                        DeviceTensorStorage::Owned,
                    )
                }
            };

        let run_result = self.nemotron_router_logits_from_device_f32_ptrs(
            input_ptr,
            normalized_ptr,
            logits_ptr,
            norm_weight,
            router_weight_f32,
            seq_len,
            hidden_dim,
            n_expert,
            norm_eps,
        );
        if let Err(err) = run_result {
            let _ =
                self.cleanup_nemotron_router_tensor_allocation(normalized_ptr, normalized_storage);
            let _ = self.cleanup_nemotron_router_tensor_allocation(logits_ptr, logits_storage);
            return Err(err);
        }

        let normalized_id = match self.insert_device_tensor_slot_with_storage(
            normalized_ptr,
            normalized_bytes,
            normalized_desc,
            normalized_storage,
        ) {
            Ok(id) => id,
            Err(err) => {
                if !normalized_storage.is_owned() {
                    let _ = self.cleanup_nemotron_router_tensor_allocation(
                        normalized_ptr,
                        normalized_storage,
                    );
                }
                let _ = self.cleanup_nemotron_router_tensor_allocation(logits_ptr, logits_storage);
                return Err(err);
            }
        };
        let router_logits_id = match self.insert_device_tensor_slot_with_storage(
            logits_ptr,
            logits_bytes,
            router_logits_desc,
            logits_storage,
        ) {
            Ok(id) => id,
            Err(err) => {
                let _ = self.release_device_tensor(normalized_id);
                if !logits_storage.is_owned() {
                    let _ =
                        self.cleanup_nemotron_router_tensor_allocation(logits_ptr, logits_storage);
                }
                return Err(err);
            }
        };
        Ok(NemotronDeviceRouterLogitsOutput {
            normalized_id,
            normalized_desc,
            router_logits_id,
            router_logits_desc,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn nemotron_router_logits_from_device_f32_ptrs(
        &mut self,
        input_ptr: u64,
        normalized_ptr: u64,
        logits_ptr: u64,
        norm_weight: &[f32],
        router_weight_f32: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        n_expert: usize,
        norm_eps: f32,
    ) -> Result<(), String> {
        let norm_dev = self.resident_f32_ptr(norm_weight)?;
        let router_dev = self.resident_f32_ptr(router_weight_f32)?;
        self.launch_rms_norm_rows_f32_serial(
            input_ptr,
            norm_dev,
            normalized_ptr,
            norm_eps,
            seq_len,
            hidden_dim,
            false,
        )?;
        self.sgemm_device(
            router_dev,
            n_expert,
            hidden_dim,
            normalized_ptr,
            seq_len,
            logits_ptr,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_prefill_moe_device(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input_id: DeviceTensorId,
        residual_id: DeviceTensorId,
    ) -> Result<DeviceTensorId, String> {
        let residual_desc = DeviceTensorDesc::new(
            token_count,
            n_embd,
            ScalarType::F32,
            DeviceTensorRole::Residual,
        );
        self.nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_id,
            residual_id,
            residual_desc,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input_id: DeviceTensorId,
        residual_id: DeviceTensorId,
        residual_desc: DeviceTensorDesc,
    ) -> Result<DeviceTensorId, String> {
        let input_desc = DeviceTensorDesc::new(
            token_count,
            n_embd,
            ScalarType::F32,
            DeviceTensorRole::Normalized,
        );
        if residual_desc.rows() != token_count || residual_desc.cols() != n_embd {
            return Err(format!(
                "Nemotron device residual shape mismatch: residual={}x{} {:?} expected {}x{} F32",
                residual_desc.rows(),
                residual_desc.cols(),
                residual_desc.dtype(),
                token_count,
                n_embd
            ));
        }
        if residual_desc.dtype() != ScalarType::F32 {
            return Err(format!(
                "Nemotron device residual dtype mismatch: residual={:?} expected F32",
                residual_desc.dtype()
            ));
        }
        let output_desc = DeviceTensorDesc::new(
            token_count,
            n_embd,
            ScalarType::F32,
            DeviceTensorRole::MoeOutput,
        );
        let input_dev = self.device_tensor_ptr(input_id, input_desc)?;
        let residual_dev = self.device_tensor_ptr(residual_id, residual_desc)?;
        let output_bytes = output_desc
            .byte_len()
            .ok_or_else(|| "Nemotron device output byte overflow".to_string())?;
        self.set_current()?;
        let (output_dev, output_storage) =
            if let Some((ptr, storage)) = self.nemotron_workspace_hidden_output_ptr(output_bytes) {
                (ptr, storage)
            } else {
                (
                    unsafe { self.api.mem_alloc(output_bytes)? },
                    DeviceTensorStorage::Owned,
                )
            };

        if let Err(err) = self.nemotron_q8_shared_q5_sparse_prefill_moe_device_ptrs(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_dev,
            residual_dev,
            output_dev,
        ) {
            if let Err(cleanup_err) =
                self.cleanup_device_tensor_storage_allocation(output_dev, output_storage)
            {
                return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
            }
            return Err(err);
        }

        match self.insert_device_tensor_slot_with_storage(
            output_dev,
            output_bytes,
            output_desc,
            output_storage,
        ) {
            Ok(output_id) => Ok(output_id),
            Err(err) => {
                if let Err(cleanup_err) =
                    self.release_workspace_storage_after_insert_failure(output_storage)
                {
                    return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
                }
                Err(err)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack(
        &mut self,
        shared_up: &[u8],
        shared_down: &[u8],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_pack: NemotronDeviceRoutePackOutput,
        shared_ff: usize,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
        input_id: DeviceTensorId,
        residual_id: DeviceTensorId,
        residual_desc: DeviceTensorDesc,
    ) -> Result<DeviceTensorId, String> {
        let input_desc = DeviceTensorDesc::new(
            token_count,
            n_embd,
            ScalarType::F32,
            DeviceTensorRole::Normalized,
        );
        if residual_desc.rows() != token_count || residual_desc.cols() != n_embd {
            return Err(format!(
                "Nemotron device residual shape mismatch: residual={}x{} {:?} expected {}x{} F32",
                residual_desc.rows(),
                residual_desc.cols(),
                residual_desc.dtype(),
                token_count,
                n_embd
            ));
        }
        if residual_desc.dtype() != ScalarType::F32 {
            return Err(format!(
                "Nemotron device residual dtype mismatch: residual={:?} expected F32",
                residual_desc.dtype()
            ));
        }
        let output_desc = DeviceTensorDesc::new(
            token_count,
            n_embd,
            ScalarType::F32,
            DeviceTensorRole::MoeOutput,
        );
        let input_dev = self.device_tensor_ptr(input_id, input_desc)?;
        let residual_dev = self.device_tensor_ptr(residual_id, residual_desc)?;
        let output_bytes = output_desc
            .byte_len()
            .ok_or_else(|| "Nemotron device output byte overflow".to_string())?;
        self.set_current()?;
        let (output_dev, output_storage) =
            if let Some((ptr, storage)) = self.nemotron_workspace_hidden_output_ptr(output_bytes) {
                (ptr, storage)
            } else {
                (
                    unsafe { self.api.mem_alloc(output_bytes)? },
                    DeviceTensorStorage::Owned,
                )
            };

        if let Err(err) = self.nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack_ptrs(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_pack,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_dev,
            residual_dev,
            output_dev,
        ) {
            if let Err(cleanup_err) =
                self.cleanup_device_tensor_storage_allocation(output_dev, output_storage)
            {
                return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
            }
            return Err(err);
        }

        match self.insert_device_tensor_slot_with_storage(
            output_dev,
            output_bytes,
            output_desc,
            output_storage,
        ) {
            Ok(output_id) => Ok(output_id),
            Err(err) => {
                if let Err(cleanup_err) =
                    self.release_workspace_storage_after_insert_failure(output_storage)
                {
                    return Err(format!("{err}; output cleanup failed: {cleanup_err}"));
                }
                Err(err)
            }
        }
    }
}
