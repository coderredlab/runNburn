use super::super::*;

#[allow(dead_code)]
const NEMOTRON_WORKSPACE_ALIGNMENT: usize = 256;

#[allow(dead_code)]
fn align_up_checked(value: usize, alignment: usize, label: &str) -> Result<usize, String> {
    if !alignment.is_power_of_two() {
        return Err(format!(
            "Nemotron workspace {label} align invalid: {alignment}"
        ));
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|aligned| aligned & !mask)
        .ok_or_else(|| format!("Nemotron workspace {label} align byte overflow"))
}

#[allow(dead_code)]
fn push_slice(
    cursor: &mut usize,
    bytes: usize,
    label: &str,
) -> Result<NemotronWorkspaceSlice, String> {
    let offset = align_up_checked(*cursor, NEMOTRON_WORKSPACE_ALIGNMENT, label)?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| format!("Nemotron workspace {label} byte overflow"))?;
    *cursor = end;
    Ok(NemotronWorkspaceSlice { offset, bytes })
}

#[allow(dead_code)]
pub(in crate::runtime) fn release_nemotron_prefill_workspace_live_lease(
    stats: &mut NemotronPrefillWorkspaceStats,
) -> Result<(), String> {
    stats.live_leases = stats
        .live_leases
        .checked_sub(1)
        .ok_or_else(|| "Nemotron prefill workspace live lease underflow".to_string())?;
    Ok(())
}

pub(in crate::runtime) fn reject_nemotron_prefill_workspace_begin_with_live_leases(
    workspace: &Option<NemotronPrefillWorkspaceArena>,
) -> Result<(), String> {
    let Some(workspace) = workspace.as_ref() else {
        return Ok(());
    };
    if workspace.stats.live_leases == 0 {
        return Ok(());
    }

    Err(format!(
        "Nemotron prefill workspace begin rejected with live leases: live_leases={} arena_id={} arena_capacity={}",
        workspace.stats.live_leases, workspace.id, workspace.capacity
    ))
}

fn nemotron_prefill_workspace_end_live_lease_error(
    summary: NemotronPrefillWorkspaceSummary,
) -> String {
    format!(
        "Nemotron prefill workspace ended with live leases: live_leases={} arena_bytes={} hit_bytes={} miss_bytes={} owned_alloc_count={}",
        summary.live_leases,
        summary.arena_bytes,
        summary.hit_bytes,
        summary.miss_bytes,
        summary.owned_alloc_count
    )
}

impl NemotronPrefillWorkspaceLayout {
    #[allow(dead_code)]
    pub(in crate::runtime) fn new(
        hidden_bytes: usize,
        normalized_bytes: usize,
        router_logits_bytes: usize,
        route_bytes: usize,
        moe_shared_mid_bytes: usize,
        moe_sparse_mid_bytes: usize,
    ) -> Result<Self, String> {
        let mut cursor = 0usize;
        let hidden_a = push_slice(&mut cursor, hidden_bytes, "hidden_a")?;
        let hidden_b = push_slice(&mut cursor, hidden_bytes, "hidden_b")?;
        let normalized = push_slice(&mut cursor, normalized_bytes, "normalized")?;
        let router_logits = push_slice(&mut cursor, router_logits_bytes, "router_logits")?;
        let route_pack = push_slice(
            &mut cursor,
            route_bytes
                .checked_mul(2)
                .ok_or_else(|| "Nemotron workspace route pack byte overflow".to_string())?,
            "route_pack",
        )?;
        let moe_shared_mid = push_slice(&mut cursor, moe_shared_mid_bytes, "moe_shared_mid")?;
        let moe_sparse_mid = push_slice(&mut cursor, moe_sparse_mid_bytes, "moe_sparse_mid")?;
        Ok(Self {
            hidden_a,
            hidden_b,
            normalized,
            router_logits,
            route_pack,
            moe_shared_mid,
            moe_sparse_mid,
            total_bytes: align_up_checked(cursor, NEMOTRON_WORKSPACE_ALIGNMENT, "total_bytes")?,
        })
    }
}

impl CudaState {
    fn record_nemotron_workspace_hidden_output_miss(&mut self, bytes: usize) {
        if let Some(workspace) = self.nemotron_prefill_workspace.as_mut() {
            workspace.stats.miss_bytes = workspace.stats.miss_bytes.saturating_add(bytes);
            workspace.stats.owned_alloc_count = workspace.stats.owned_alloc_count.saturating_add(1);
        }
    }

    fn record_nemotron_workspace_moe_mid_miss(&mut self, bytes: usize) {
        if let Some(workspace) = self.nemotron_prefill_workspace.as_mut() {
            workspace.stats.miss_bytes = workspace.stats.miss_bytes.saturating_add(bytes);
            workspace.stats.owned_alloc_count = workspace.stats.owned_alloc_count.saturating_add(2);
        }
    }

    fn nemotron_workspace_slice_is_live(
        &self,
        arena_id: u64,
        slice: NemotronWorkspaceSlice,
    ) -> bool {
        self.device_tensors.values().any(|slot| {
            matches!(
                slot.storage,
                DeviceTensorStorage::NemotronWorkspace {
                    arena_id: live_arena_id,
                    offset,
                    ..
                } if live_arena_id == arena_id && offset == slice.offset
            )
        })
    }

    pub(in crate::runtime) fn nemotron_workspace_slice_ptr(
        &mut self,
        slice: NemotronWorkspaceSlice,
    ) -> Option<(u64, DeviceTensorStorage)> {
        let workspace = self.nemotron_prefill_workspace.as_mut()?;
        if !workspace.active || slice.bytes == 0 {
            return None;
        }
        let ptr = workspace.ptr.checked_add(slice.offset as u64)?;
        workspace.stats.hit_bytes = workspace.stats.hit_bytes.saturating_add(slice.bytes);
        workspace.stats.live_leases = workspace.stats.live_leases.saturating_add(1);
        Some((
            ptr,
            DeviceTensorStorage::NemotronWorkspace {
                arena_id: workspace.id,
                offset: slice.offset,
                bytes: slice.bytes,
            },
        ))
    }

    pub(in crate::runtime) fn nemotron_workspace_hidden_output_ptr(
        &mut self,
        bytes: usize,
    ) -> Option<(u64, DeviceTensorStorage)> {
        let (arena_id, arena_ptr, active, hidden_a, hidden_b) = {
            let workspace = self.nemotron_prefill_workspace.as_ref()?;
            (
                workspace.id,
                workspace.ptr,
                workspace.active,
                workspace.layout.hidden_a,
                workspace.layout.hidden_b,
            )
        };
        if !active || bytes == 0 {
            self.record_nemotron_workspace_hidden_output_miss(bytes);
            return None;
        }

        let hidden_a_live = self.nemotron_workspace_slice_is_live(arena_id, hidden_a);
        let hidden_b_live = self.nemotron_workspace_slice_is_live(arena_id, hidden_b);
        let slice = match (hidden_a_live, hidden_b_live) {
            (false, _) => hidden_a,
            (true, false) => hidden_b,
            (true, true) => {
                self.record_nemotron_workspace_hidden_output_miss(bytes);
                return None;
            }
        };
        if bytes > slice.bytes {
            self.record_nemotron_workspace_hidden_output_miss(bytes);
            return None;
        }

        let ptr = match arena_ptr.checked_add(slice.offset as u64) {
            Some(ptr) => ptr,
            None => {
                self.record_nemotron_workspace_hidden_output_miss(bytes);
                return None;
            }
        };
        let workspace = self.nemotron_prefill_workspace.as_mut()?;
        workspace.stats.hit_bytes = workspace.stats.hit_bytes.saturating_add(bytes);
        workspace.stats.live_leases = workspace.stats.live_leases.saturating_add(1);
        Some((
            ptr,
            DeviceTensorStorage::NemotronWorkspace {
                arena_id,
                offset: slice.offset,
                bytes,
            },
        ))
    }

    pub(in crate::runtime) fn cleanup_device_tensor_storage_allocation(
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

    pub(in crate::runtime) fn release_workspace_storage_after_insert_failure(
        &mut self,
        storage: DeviceTensorStorage,
    ) -> Result<(), String> {
        if let DeviceTensorStorage::NemotronWorkspace {
            arena_id,
            offset,
            bytes,
        } = storage
        {
            self.release_nemotron_prefill_workspace_lease(arena_id, offset, bytes)?;
        }
        Ok(())
    }

    pub(in crate::runtime) fn nemotron_workspace_router_logits_ptrs(
        &mut self,
        normalized_bytes: usize,
        logits_bytes: usize,
    ) -> Option<((u64, DeviceTensorStorage), (u64, DeviceTensorStorage))> {
        let workspace = self.nemotron_prefill_workspace.as_mut()?;
        let requested = normalized_bytes.saturating_add(logits_bytes);
        if !workspace.active
            || normalized_bytes == 0
            || logits_bytes == 0
            || normalized_bytes > workspace.layout.normalized.bytes
            || logits_bytes > workspace.layout.router_logits.bytes
        {
            workspace.stats.miss_bytes = workspace.stats.miss_bytes.saturating_add(requested);
            workspace.stats.owned_alloc_count = workspace.stats.owned_alloc_count.saturating_add(2);
            return None;
        }
        let layout = workspace.layout;
        let normalized = self.nemotron_workspace_slice_ptr(layout.normalized)?;
        let logits = match self.nemotron_workspace_slice_ptr(layout.router_logits) {
            Some(logits) => logits,
            None => {
                if let DeviceTensorStorage::NemotronWorkspace {
                    arena_id,
                    offset,
                    bytes,
                } = normalized.1
                {
                    let _ = self.release_nemotron_prefill_workspace_lease(arena_id, offset, bytes);
                }
                return None;
            }
        };
        Some((normalized, logits))
    }

    pub(in crate::runtime) fn nemotron_workspace_route_pack_ptrs(
        &mut self,
        reordered: bool,
        ids_bytes: usize,
        weights_bytes: usize,
    ) -> Option<(u64, u64, u64, NemotronRoutePackStorage)> {
        let workspace = self.nemotron_prefill_workspace.as_mut()?;
        let required = ids_bytes
            .checked_add(weights_bytes)?
            .checked_add(ids_bytes)?;
        if !workspace.active {
            workspace.stats.miss_bytes = workspace.stats.miss_bytes.saturating_add(required);
            workspace.stats.owned_alloc_count = workspace.stats.owned_alloc_count.saturating_add(3);
            return None;
        }
        let route_slice = workspace.layout.route_pack;
        let half = route_slice.bytes / 2;
        let base_offset = route_slice
            .offset
            .checked_add(if reordered { half } else { 0 })?;
        if required > half {
            workspace.stats.miss_bytes = workspace.stats.miss_bytes.saturating_add(required);
            workspace.stats.owned_alloc_count = workspace.stats.owned_alloc_count.saturating_add(3);
            return None;
        }
        let expert_ids_dev = workspace.ptr.checked_add(base_offset as u64)?;
        let route_weights_dev = expert_ids_dev.checked_add(ids_bytes as u64)?;
        let token_ids_dev = route_weights_dev.checked_add(weights_bytes as u64)?;
        workspace.stats.hit_bytes = workspace.stats.hit_bytes.saturating_add(required);
        Some((
            expert_ids_dev,
            route_weights_dev,
            token_ids_dev,
            NemotronRoutePackStorage::Workspace {
                arena_id: workspace.id,
            },
        ))
    }

    pub(in crate::runtime) fn nemotron_workspace_moe_mid_ptrs(
        &mut self,
        shared_mid_bytes: usize,
        sparse_mid_bytes: usize,
    ) -> Option<(u64, u64)> {
        let requested = shared_mid_bytes.saturating_add(sparse_mid_bytes);
        let workspace = self.nemotron_prefill_workspace.as_ref()?;
        if !workspace.active
            || shared_mid_bytes > workspace.layout.moe_shared_mid.bytes
            || sparse_mid_bytes > workspace.layout.moe_sparse_mid.bytes
        {
            self.record_nemotron_workspace_moe_mid_miss(requested);
            return None;
        }

        let shared_mid_dev = match workspace
            .ptr
            .checked_add(workspace.layout.moe_shared_mid.offset as u64)
        {
            Some(ptr) => ptr,
            None => {
                self.record_nemotron_workspace_moe_mid_miss(requested);
                return None;
            }
        };
        let sparse_mid_dev = match workspace
            .ptr
            .checked_add(workspace.layout.moe_sparse_mid.offset as u64)
        {
            Some(ptr) => ptr,
            None => {
                self.record_nemotron_workspace_moe_mid_miss(requested);
                return None;
            }
        };

        let workspace = self.nemotron_prefill_workspace.as_mut()?;
        workspace.stats.hit_bytes = workspace.stats.hit_bytes.saturating_add(requested);
        Some((shared_mid_dev, sparse_mid_dev))
    }

    pub(in crate::runtime) fn begin_nemotron_prefill_workspace(
        &mut self,
        config: NemotronPrefillWorkspaceConfig,
    ) -> Result<NemotronPrefillWorkspaceSummary, String> {
        if !config.enabled {
            return Ok(NemotronPrefillWorkspaceSummary {
                active: false,
                arena_bytes: 0,
                live_leases: 0,
                hit_bytes: 0,
                miss_bytes: 0,
                owned_alloc_count: 0,
            });
        }

        reject_nemotron_prefill_workspace_begin_with_live_leases(&self.nemotron_prefill_workspace)?;

        let layout = NemotronPrefillWorkspaceLayout::new(
            config.hidden_bytes,
            config.normalized_bytes,
            config.router_logits_bytes,
            config.route_bytes,
            config.moe_shared_mid_bytes,
            config.moe_sparse_mid_bytes,
        )?;
        let arena_bytes = layout.total_bytes.max(config.required_workspace_bytes);
        self.set_current()?;

        let needs_alloc = self
            .nemotron_prefill_workspace
            .as_ref()
            .map(|workspace| workspace.capacity < arena_bytes)
            .unwrap_or(true);
        if needs_alloc {
            let id = self.next_nemotron_prefill_workspace_id;
            let next_id = id
                .checked_add(1)
                .ok_or_else(|| "Nemotron prefill workspace id overflow".to_string())?;

            self.stream_synchronize()?;
            if let Some(old) = self.nemotron_prefill_workspace.take() {
                unsafe { self.api.mem_free(old.ptr)? };
            }
            self.reclaim_residency_for_transient(arena_bytes)?;
            let ptr = unsafe { self.api.mem_alloc(arena_bytes)? };
            self.next_nemotron_prefill_workspace_id = next_id;
            self.nemotron_prefill_workspace = Some(NemotronPrefillWorkspaceArena {
                id,
                ptr,
                capacity: arena_bytes,
                active: true,
                layout,
                stats: NemotronPrefillWorkspaceStats::default(),
            });
        } else if let Some(workspace) = self.nemotron_prefill_workspace.as_mut() {
            workspace.active = true;
            workspace.layout = layout;
            workspace.stats = NemotronPrefillWorkspaceStats::default();
        }

        self.nemotron_prefill_workspace_summary()
    }

    pub(in crate::runtime) fn end_nemotron_prefill_workspace(
        &mut self,
    ) -> Result<NemotronPrefillWorkspaceSummary, String> {
        let summary = self.nemotron_prefill_workspace_summary()?;
        if let Some(workspace) = self.nemotron_prefill_workspace.as_mut() {
            workspace.active = false;
        }
        if summary.live_leases != 0 {
            return Err(nemotron_prefill_workspace_end_live_lease_error(summary));
        }
        Ok(summary)
    }

    pub(in crate::runtime) fn nemotron_prefill_workspace_summary(
        &self,
    ) -> Result<NemotronPrefillWorkspaceSummary, String> {
        let Some(workspace) = self.nemotron_prefill_workspace.as_ref() else {
            return Ok(NemotronPrefillWorkspaceSummary {
                active: false,
                arena_bytes: 0,
                live_leases: 0,
                hit_bytes: 0,
                miss_bytes: 0,
                owned_alloc_count: 0,
            });
        };

        Ok(NemotronPrefillWorkspaceSummary {
            active: workspace.active,
            arena_bytes: workspace.capacity,
            live_leases: workspace.stats.live_leases,
            hit_bytes: workspace.stats.hit_bytes,
            miss_bytes: workspace.stats.miss_bytes,
            owned_alloc_count: workspace.stats.owned_alloc_count,
        })
    }

    pub(in crate::runtime) fn release_nemotron_prefill_workspace_lease(
        &mut self,
        arena_id: u64,
        _offset: usize,
        _bytes: usize,
    ) -> Result<(), String> {
        let workspace = self
            .nemotron_prefill_workspace
            .as_mut()
            .ok_or_else(|| "missing Nemotron prefill workspace arena".to_string())?;
        if workspace.id != arena_id {
            return Err(format!(
                "Nemotron prefill workspace arena id mismatch: got {arena_id}, active {}",
                workspace.id
            ));
        }
        release_nemotron_prefill_workspace_live_lease(&mut workspace.stats)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nemotron_workspace_release_live_lease_reports_underflow() {
        let mut stats = NemotronPrefillWorkspaceStats::default();
        let err = release_nemotron_prefill_workspace_live_lease(&mut stats)
            .expect_err("zero live leases must be an error");

        assert!(err.contains("live lease underflow"));
    }

    #[test]
    fn nemotron_workspace_public_config_and_summary_are_constructible() {
        let config = NemotronPrefillWorkspaceConfig {
            hidden_bytes: 1024,
            normalized_bytes: 1024,
            router_logits_bytes: 256,
            route_bytes: 128,
            moe_shared_mid_bytes: 4096,
            moe_sparse_mid_bytes: 8192,
            required_workspace_bytes: 16 * 1024,
            enabled: true,
        };
        let summary = NemotronPrefillWorkspaceSummary {
            active: true,
            arena_bytes: config.required_workspace_bytes,
            live_leases: 0,
            hit_bytes: 0,
            miss_bytes: 0,
            owned_alloc_count: 0,
        };

        assert!(config.enabled);
        assert_eq!(summary.arena_bytes, 16 * 1024);
    }

    #[test]
    fn nemotron_workspace_live_lease_guard_rejects_begin_without_resetting_stats() {
        let layout =
            NemotronPrefillWorkspaceLayout::new(1024, 256, 128, 64, 4096, 8192).expect("layout");
        let workspace = NemotronPrefillWorkspaceArena {
            id: 42,
            ptr: 7,
            capacity: 65_536,
            active: true,
            layout,
            stats: NemotronPrefillWorkspaceStats {
                hit_bytes: 1024,
                miss_bytes: 2048,
                owned_alloc_count: 3,
                live_leases: 2,
            },
        };
        let previous_stats = workspace.stats;
        let existing = Some(workspace);

        let err = reject_nemotron_prefill_workspace_begin_with_live_leases(&existing)
            .expect_err("live leases must reject begin");

        assert!(err.contains("live_leases=2"));
        assert!(err.contains("arena_id=42"));
        assert!(err.contains("arena_capacity=65536"));
        assert_eq!(existing.as_ref().expect("workspace").stats, previous_stats);
    }

    #[test]
    fn nemotron_workspace_end_live_lease_error_reports_summary_fields() {
        let summary = NemotronPrefillWorkspaceSummary {
            active: false,
            arena_bytes: 65_536,
            live_leases: 2,
            hit_bytes: 4096,
            miss_bytes: 2048,
            owned_alloc_count: 3,
        };

        let err = nemotron_prefill_workspace_end_live_lease_error(summary);

        assert!(err.contains("live_leases=2"));
        assert!(err.contains("arena_bytes=65536"));
        assert!(err.contains("hit_bytes=4096"));
        assert!(err.contains("miss_bytes=2048"));
        assert!(err.contains("owned_alloc_count=3"));
    }
}
