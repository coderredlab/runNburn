use super::super::super::*;
use rnb_memory::{
    evaluate_expert_bundle_admission, CurrentLookupTransfer, ExpertBundleAdmissionCandidate,
    ExpertBundleCacheStats, ExpertBundleFootprint, ExpertBundleObservationReceipt,
    ExpertBundleResidency, SparseExpertCacheKey,
};

fn qwen35_decode_hot_resident_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_DECODE_HOT_RESIDENT")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

fn qwen35_decode_hot_resident_budget_bytes(resident_q4k_limit: usize) -> usize {
    std::env::var("RNB_CUDA_QWEN35_DECODE_HOT_RESIDENT_MB")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or_else(|| qwen35_decode_hot_resident_default_budget_bytes(resident_q4k_limit))
}

fn qwen35_decode_moe_graph_can_fallback(err: &str) -> bool {
    err.contains("missing CUDA driver symbol cuStreamBeginCapture")
        || err.contains("missing CUDA driver symbol cuStreamEndCapture")
        || err.contains("missing CUDA driver symbol cuGraphInstantiateWithFlags")
        || err.contains("missing CUDA driver symbol cuGraphLaunch")
}

fn qwen35_decode_hot_resident_batch_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_DECODE_HOT_RESIDENT_BATCH")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

fn accumulate_expert_bundle_stats(
    total: &mut ExpertBundleCacheStats,
    delta: ExpertBundleCacheStats,
) {
    total.bundle_lookups = total.bundle_lookups.saturating_add(delta.bundle_lookups);
    total.bundle_hits = total.bundle_hits.saturating_add(delta.bundle_hits);
    total.bundle_partial_hits = total
        .bundle_partial_hits
        .saturating_add(delta.bundle_partial_hits);
    total.bundle_misses = total.bundle_misses.saturating_add(delta.bundle_misses);
    total.bundle_admissions = total
        .bundle_admissions
        .saturating_add(delta.bundle_admissions);
    total.bundle_evictions = total
        .bundle_evictions
        .saturating_add(delta.bundle_evictions);
    total.admitted_bytes = total.admitted_bytes.saturating_add(delta.admitted_bytes);
    total.evicted_bytes = total.evicted_bytes.saturating_add(delta.evicted_bytes);
    total.h2d_bytes = total.h2d_bytes.saturating_add(delta.h2d_bytes);
    total.temp_h2d_bytes = total.temp_h2d_bytes.saturating_add(delta.temp_h2d_bytes);
}

#[derive(Clone, Copy)]
struct Qwen35DecodeExpertBundle<'a> {
    expert_id: usize,
    slot: usize,
    aggregate_route_weight: f64,
    gate: &'a [u8],
    up: &'a [u8],
    down: &'a [u8],
}

#[cfg(test)]
thread_local! {
    static QWEN35_DECODE_REUSE_SCORE_OVERRIDE: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
}

fn validate_qwen35_decode_expert_bundle_inputs(
    selected_expert_ids: &[usize],
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
) -> Result<(), String> {
    if gate_weights.len() != up_weights.len()
        || gate_weights.len() != down_weights.len()
        || gate_weights.len() != route_weights.len()
    {
        return Err("Qwen35 selected expert bundle input length mismatch".to_string());
    }
    if selected_expert_ids.len() > gate_weights.len() {
        return Err(format!(
            "Qwen35 selected expert IDs exceed CUDA slots: ids={} slots={}",
            selected_expert_ids.len(),
            gate_weights.len()
        ));
    }
    for (slot, &route_weight) in route_weights
        .iter()
        .take(selected_expert_ids.len())
        .enumerate()
    {
        if !route_weight.is_finite() || route_weight < 0.0 {
            return Err(format!(
                "Qwen35 selected expert route weight must be finite and non-negative: slot={slot} weight={route_weight}"
            ));
        }
    }
    Ok(())
}

fn qwen35_decode_expert_bundles<'a>(
    selected_expert_ids: &[usize],
    gate_weights: &'a [&'a [u8]],
    up_weights: &'a [&'a [u8]],
    down_weights: &'a [&'a [u8]],
    route_weights: &[f32],
) -> Result<Vec<Qwen35DecodeExpertBundle<'a>>, String> {
    validate_qwen35_decode_expert_bundle_inputs(
        selected_expert_ids,
        gate_weights,
        up_weights,
        down_weights,
        route_weights,
    )?;

    let mut bundles = std::collections::BTreeMap::new();
    for (slot, &expert_id) in selected_expert_ids.iter().enumerate() {
        let route_weight = f64::from(route_weights[slot]);
        match bundles.entry(expert_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(Qwen35DecodeExpertBundle {
                    expert_id,
                    slot,
                    aggregate_route_weight: route_weight,
                    gate: gate_weights[slot],
                    up: up_weights[slot],
                    down: down_weights[slot],
                });
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let bundle = entry.get_mut();
                if q4k_resident_key(bundle.gate) != q4k_resident_key(gate_weights[slot])
                    || q4k_resident_key(bundle.up) != q4k_resident_key(up_weights[slot])
                    || q4k_resident_key(bundle.down) != q4k_resident_key(down_weights[slot])
                {
                    return Err(format!(
                        "Qwen35 duplicate expert {expert_id} referenced different weight slices"
                    ));
                }
                bundle.aggregate_route_weight += route_weight;
            }
        }
    }
    Ok(bundles.into_values().collect())
}

fn qwen35_decode_expert_bundle_owned_role_bytes(bundle: Qwen35DecodeExpertBundle<'_>) -> [u64; 3] {
    let gate_key = q4k_resident_key(bundle.gate);
    let up_key = q4k_resident_key(bundle.up);
    let down_key = q4k_resident_key(bundle.down);
    [
        bundle.gate.len() as u64,
        if up_key == gate_key {
            0
        } else {
            bundle.up.len() as u64
        },
        if down_key == gate_key || down_key == up_key {
            0
        } else {
            bundle.down.len() as u64
        },
    ]
}

fn qwen35_decode_expert_bundle_candidate(
    bundle: Qwen35DecodeExpertBundle<'_>,
    key: SparseExpertCacheKey,
    mut is_resident: impl FnMut(&[u8]) -> bool,
) -> ExpertBundleAdmissionCandidate {
    let [gate_bytes, up_bytes, down_bytes] = qwen35_decode_expert_bundle_owned_role_bytes(bundle);
    ExpertBundleAdmissionCandidate::new(
        ExpertBundleFootprint::new(key, gate_bytes, up_bytes, down_bytes),
        is_resident(bundle.gate),
        is_resident(bundle.up),
        is_resident(bundle.down),
    )
}

fn qwen35_selected_q4k_keys(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    selected_count: usize,
) -> HashSet<(usize, usize)> {
    gate_weights[..selected_count]
        .iter()
        .chain(up_weights[..selected_count].iter())
        .chain(down_weights[..selected_count].iter())
        .map(|weights| q4k_resident_key(weights))
        .collect()
}

fn qwen35_decode_selected_expert_bundle_stats(
    bundles: &[Qwen35DecodeExpertBundle<'_>],
    mut is_resident: impl FnMut(&[u8]) -> bool,
) -> ExpertBundleCacheStats {
    let mut stats = ExpertBundleCacheStats::default();
    for bundle in bundles {
        stats.record_lookup(ExpertBundleResidency::from_roles(
            is_resident(bundle.gate),
            is_resident(bundle.up),
            is_resident(bundle.down),
        ));
    }
    stats
}

impl CudaState {
    #[cfg(test)]
    pub(in crate::runtime) fn inject_qwen35_decode_failure_for_test(
        &mut self,
        point: Qwen35DecodeFailurePoint,
    ) {
        self.qwen35_decode_failure_point = Some(point);
    }

    #[cfg(test)]
    fn fail_qwen35_decode_for_test(
        &mut self,
        point: Qwen35DecodeFailurePoint,
    ) -> Result<(), String> {
        if self.qwen35_decode_failure_point == Some(point) {
            self.qwen35_decode_failure_point = None;
            return Err(format!("injected Qwen35 decode failure at {point:?}"));
        }
        Ok(())
    }

    fn resident_moe_layer_decode_ptrs_if_all_available(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        selected_expert_ids: &[usize],
    ) -> Option<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)> {
        if gate_weights.len() != up_weights.len()
            || gate_weights.len() != down_weights.len()
            || selected_expert_ids.len() > gate_weights.len()
        {
            return None;
        }

        let mut gate_ptrs = Vec::with_capacity(gate_weights.len());
        let mut up_ptrs = Vec::with_capacity(up_weights.len());
        let mut down_ptrs = Vec::with_capacity(down_weights.len());
        let mut touched_layers = HashSet::new();
        let mut touched_q4k = HashSet::new();

        for slot in 0..gate_weights.len() {
            if let Some(&expert_id) = selected_expert_ids.get(slot) {
                let gate = gate_weights[slot];
                let up = up_weights[slot];
                let down = down_weights[slot];
                let resident = self.resident_moe_layers.iter().find_map(|(&key, entry)| {
                    let gate_offset = expert_id.checked_mul(gate.len())?;
                    let up_offset = expert_id.checked_mul(up.len())?;
                    let down_offset = expert_id.checked_mul(down.len())?;
                    let gate_host = key.gate_ptr.checked_add(gate_offset)?;
                    let up_host = key.up_ptr.checked_add(up_offset)?;
                    let down_host = key.down_ptr.checked_add(down_offset)?;
                    let gate_end = gate_offset.checked_add(gate.len())?;
                    let up_end = up_offset.checked_add(up.len())?;
                    let down_end = down_offset.checked_add(down.len())?;
                    (gate_host == gate.as_ptr() as usize
                        && up_host == up.as_ptr() as usize
                        && down_host == down.as_ptr() as usize
                        && gate_end <= key.gate_len
                        && up_end <= key.up_len
                        && down_end <= key.down_len)
                        .then_some((
                            key,
                            entry.gate_base + gate_offset as u64,
                            entry.up_base + up_offset as u64,
                            entry.down_base + down_offset as u64,
                        ))
                })?;
                touched_layers.insert(resident.0);
                gate_ptrs.push(resident.1);
                up_ptrs.push(resident.2);
                down_ptrs.push(resident.3);
            } else {
                for (weights, ptrs) in [
                    (gate_weights[slot], &mut gate_ptrs),
                    (up_weights[slot], &mut up_ptrs),
                    (down_weights[slot], &mut down_ptrs),
                ] {
                    let key = q4k_resident_key(weights);
                    let ptr = self.resident_q4k.get(&key)?.ptr;
                    touched_q4k.insert(key);
                    ptrs.push(ptr);
                }
            }
        }

        for key in touched_layers {
            self.touch_resident_moe_layer(key);
        }
        for key in touched_q4k {
            self.touch_resident_q4k(key);
        }
        Some((gate_ptrs, up_ptrs, down_ptrs, Vec::new()))
    }

    pub(in crate::runtime) fn decode_selected_weight_ptrs(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        mut bundle_observation_receipt: Option<&mut ExpertBundleObservationReceipt>,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        self.raise_resident_q4k_limit_for_qwen35_target_decode()?;
        if selected_expert_ids.len() > gate_weights.len() {
            return Err(format!(
                "Qwen35 selected expert IDs exceed CUDA slots: ids={} slots={}",
                selected_expert_ids.len(),
                gate_weights.len()
            ));
        }
        if let Some(ptrs) = self.resident_moe_layer_decode_ptrs_if_all_available(
            gate_weights,
            up_weights,
            down_weights,
            selected_expert_ids,
        ) {
            return Ok(ptrs);
        }
        let slot_groups = [gate_weights, up_weights, down_weights];
        let selected_keys = qwen35_selected_q4k_keys(
            gate_weights,
            up_weights,
            down_weights,
            selected_expert_ids.len(),
        );
        let observe_and_apply = bundle_observation_receipt
            .as_deref()
            .is_some_and(|receipt| !receipt.consumed());
        let reuse_current_token = bundle_observation_receipt
            .as_deref()
            .is_some_and(|receipt| receipt.consumed());
        let tracked_keys = bundle_observation_receipt
            .is_some()
            .then_some(&selected_keys);
        if observe_and_apply {
            if let Some(layer_idx) = layer_idx {
                self.admit_qwen35_decode_expert_bundles(
                    layer_idx,
                    selected_expert_ids,
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                    bundle_observation_receipt
                        .as_deref_mut()
                        .expect("Q2/Q3 bundle observation receipt"),
                    CurrentLookupTransfer::ReplacesTempUpload,
                )?;
                if selected_expert_ids.len() < gate_weights.len() {
                    let mut protected_keys = HashSet::new();
                    for &weights in gate_weights
                        .iter()
                        .chain(up_weights.iter())
                        .chain(down_weights.iter())
                    {
                        protected_keys.insert(q4k_resident_key(weights));
                    }
                    let shared_start = selected_expert_ids.len();
                    self.promote_qwen35_decode_hot_resident_slots(
                        &gate_weights[shared_start..],
                        &up_weights[shared_start..],
                        &down_weights[shared_start..],
                        &route_weights[shared_start..],
                        None,
                        Some(&protected_keys),
                    )?;
                }
                if let Some(ptrs) = self.resident_q4k_slot_ptrs_3_if_all_resident(
                    gate_weights,
                    up_weights,
                    down_weights,
                ) {
                    return Ok(ptrs);
                }
                return if self.q4k_slot_groups_have_resident(&slot_groups) {
                    self.mixed_resident_temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
                        gate_weights,
                        up_weights,
                        down_weights,
                        &selected_keys,
                    )
                } else {
                    self.temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
                        gate_weights,
                        up_weights,
                        down_weights,
                        &selected_keys,
                    )
                };
            }

            let bundles = qwen35_decode_expert_bundles(
                selected_expert_ids,
                gate_weights,
                up_weights,
                down_weights,
                route_weights,
            )?;
            let bundle_stats = qwen35_decode_selected_expert_bundle_stats(&bundles, |weights| {
                self.resident_q4k.contains_key(&q4k_resident_key(weights))
            });
            cache_stats().record_expert_bundles(bundle_stats);
            let receipt = bundle_observation_receipt
                .as_deref_mut()
                .expect("Q2/Q3 bundle observation receipt");
            receipt.record_stats(bundle_stats);
            receipt.mark_consumed();
        }
        if let Some(ptrs) =
            self.resident_q4k_slot_ptrs_3_if_all_resident(gate_weights, up_weights, down_weights)
        {
            return Ok(ptrs);
        }
        if reuse_current_token {
            return if self.q4k_slot_groups_have_resident(&slot_groups) {
                self.mixed_resident_temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
                    gate_weights,
                    up_weights,
                    down_weights,
                    &selected_keys,
                )
            } else {
                self.temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
                    gate_weights,
                    up_weights,
                    down_weights,
                    &selected_keys,
                )
            };
        }
        self.promote_qwen35_decode_hot_resident_slots(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            tracked_keys,
            None,
        )?;
        if let Some(ptrs) =
            self.resident_q4k_slot_ptrs_3_if_all_resident(gate_weights, up_weights, down_weights)
        {
            return Ok(ptrs);
        }
        let plan = qwen35_decode_selected_slot_ptr_plan(
            gate_weights,
            up_weights,
            down_weights,
            self.resident_q4k_limit,
            tuning::qwen35_decode_resident_batch_enabled(),
            self.q4k_slot_groups_have_resident(&slot_groups),
        );
        match plan {
            Qwen35DecodeSelectedSlotPtrPlan::MixedResidentTemp => {
                return if let Some(tracked_keys) = tracked_keys {
                    self.mixed_resident_temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
                        gate_weights,
                        up_weights,
                        down_weights,
                        tracked_keys,
                    )
                } else {
                    self.mixed_resident_temp_q4k_slot_ptrs_3(gate_weights, up_weights, down_weights)
                };
            }
            Qwen35DecodeSelectedSlotPtrPlan::ResidentBatch => {
                let local_ptrs = HashMap::new();
                if let Some(tracked_keys) = tracked_keys {
                    self.batch_resident_q4k_slot_misses_many_recording_expert_bundle_h2d(
                        &slot_groups,
                        &local_ptrs,
                        tracked_keys,
                    )?;
                } else {
                    self.batch_resident_q4k_slot_misses_many(&slot_groups, &local_ptrs)?;
                }
                if let Some(ptrs) = self.resident_q4k_slot_ptrs_3_if_all_resident(
                    gate_weights,
                    up_weights,
                    down_weights,
                ) {
                    return Ok(ptrs);
                }
                if self.q4k_slot_groups_have_resident(&slot_groups) {
                    return if let Some(tracked_keys) = tracked_keys {
                        self.mixed_resident_temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
                            gate_weights,
                            up_weights,
                            down_weights,
                            tracked_keys,
                        )
                    } else {
                        self.mixed_resident_temp_q4k_slot_ptrs_3(
                            gate_weights,
                            up_weights,
                            down_weights,
                        )
                    };
                }
            }
            Qwen35DecodeSelectedSlotPtrPlan::TempSlab => {}
        }

        if let Some(tracked_keys) = tracked_keys {
            self.temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
                gate_weights,
                up_weights,
                down_weights,
                tracked_keys,
            )
        } else {
            self.temp_q4k_slot_ptrs_3(gate_weights, up_weights, down_weights)
        }
    }

    fn admit_qwen35_decode_expert_bundles(
        &mut self,
        layer_idx: usize,
        selected_expert_ids: &[usize],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
        current_lookup_transfer: CurrentLookupTransfer,
    ) -> Result<ExpertBundleCacheStats, String> {
        let bundles = qwen35_decode_expert_bundles(
            selected_expert_ids,
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
        )?;
        let bundle_stats = qwen35_decode_selected_expert_bundle_stats(&bundles, |weights| {
            self.resident_q4k.contains_key(&q4k_resident_key(weights))
        });
        cache_stats().record_expert_bundles(bundle_stats);
        bundle_observation_receipt.record_stats(bundle_stats);
        if bundles.is_empty() {
            bundle_observation_receipt.mark_consumed();
            return Ok(bundle_stats);
        }

        let bundle_payload_bytes = bundles
            .iter()
            .map(|bundle| {
                qwen35_decode_expert_bundle_owned_role_bytes(*bundle)
                    .into_iter()
                    .fold(0u64, u64::saturating_add)
                    .min(usize::MAX as u64) as usize
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let history_capacity = self.resident_q4k_limit / bundle_payload_bytes;
        self.qwen35_expert_bundle_reuse_history
            .set_capacity(history_capacity);

        let mut protected_keys = HashSet::new();
        for &weights in gate_weights
            .iter()
            .chain(up_weights.iter())
            .chain(down_weights.iter())
        {
            protected_keys.insert(q4k_resident_key(weights));
        }

        let scored = bundles
            .into_iter()
            .map(|bundle| {
                let key = SparseExpertCacheKey::new(layer_idx, bundle.expert_id);
                let observed_reuses = self.qwen35_expert_bundle_reuse_history.observe(key);
                #[cfg(test)]
                let predicted_future_reuses = QWEN35_DECODE_REUSE_SCORE_OVERRIDE
                    .with(|score| score.get().unwrap_or(observed_reuses));
                #[cfg(not(test))]
                let predicted_future_reuses = observed_reuses;
                (bundle, key, predicted_future_reuses)
            })
            .collect::<Vec<_>>();
        bundle_observation_receipt.mark_consumed();
        #[cfg(test)]
        self.fail_qwen35_decode_for_test(Qwen35DecodeFailurePoint::AdmissionAfterObservation)?;
        let admission_stats = self.admit_scored_qwen35_decode_expert_bundles(
            layer_idx,
            scored,
            &protected_keys,
            current_lookup_transfer,
        )?;
        bundle_observation_receipt.record_stats(admission_stats);
        let mut operation_stats = bundle_stats;
        accumulate_expert_bundle_stats(&mut operation_stats, admission_stats);
        Ok(operation_stats)
    }

    fn admit_scored_qwen35_decode_expert_bundles(
        &mut self,
        layer_idx: usize,
        mut scored: Vec<(Qwen35DecodeExpertBundle<'_>, SparseExpertCacheKey, u64)>,
        protected_keys: &HashSet<(usize, usize)>,
        current_lookup_transfer: CurrentLookupTransfer,
    ) -> Result<ExpertBundleCacheStats, String> {
        scored.sort_by(|(left, _, left_reuses), (right, _, right_reuses)| {
            right_reuses
                .cmp(left_reuses)
                .then_with(|| {
                    right
                        .aggregate_route_weight
                        .partial_cmp(&left.aggregate_route_weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.expert_id.cmp(&right.expert_id))
                .then_with(|| left.slot.cmp(&right.slot))
        });

        let trace_cache = std::env::var("RNB_CUDA_CACHE_TRACE").ok().as_deref() == Some("1");
        let mut operation_stats = ExpertBundleCacheStats::default();
        for (bundle, bundle_key, predicted_future_reuses) in scored {
            let candidate = qwen35_decode_expert_bundle_candidate(bundle, bundle_key, |weights| {
                self.resident_q4k.contains_key(&q4k_resident_key(weights))
            });
            if candidate.is_fully_resident() {
                let mut touched = HashSet::new();
                for weights in [bundle.gate, bundle.up, bundle.down] {
                    let key = q4k_resident_key(weights);
                    if touched.insert(key) {
                        self.touch_resident_q4k(key);
                    }
                }
            }
            let incoming = self.qwen35_decode_expert_bundle_missing_allocation_bytes(bundle);
            let Some(eviction_plan) =
                self.resident_q4k_eviction_plan_for_incoming(incoming, protected_keys)
            else {
                if trace_cache {
                    eprintln!(
                        "[cuda-cache] qwen35_bundle_admission layer={} expert={} predicted_future_reuses={} missing_bytes={} eviction_reload_bytes=unavailable admit=false",
                        layer_idx,
                        bundle.expert_id,
                        predicted_future_reuses,
                        candidate.missing_admission_bytes(),
                    );
                }
                continue;
            };
            let decision = evaluate_expert_bundle_admission(
                candidate,
                predicted_future_reuses,
                eviction_plan.reload_payload_bytes,
                current_lookup_transfer,
            );
            if trace_cache {
                eprintln!(
                    "[cuda-cache] qwen35_bundle_admission layer={} expert={} predicted_future_reuses={} missing_bytes={} eviction_reload_bytes={} predicted_saved_bytes={} net_saved_bytes={} admit={}",
                    layer_idx,
                    bundle.expert_id,
                    predicted_future_reuses,
                    decision.cost.missing_admission_bytes,
                    decision.cost.eviction_reload_bytes,
                    decision.cost.predicted_saved_bytes,
                    decision.cost.net_saved_bytes,
                    decision.admit,
                );
            }
            if !decision.admit {
                continue;
            }
            let additional_oom_reload_budget =
                u64::try_from(decision.cost.net_saved_bytes.saturating_sub(1)).unwrap_or(u64::MAX);
            let delta = self.admit_qwen35_decode_expert_bundle_roles(
                bundle,
                bundle_key,
                protected_keys,
                eviction_plan,
                additional_oom_reload_budget,
            )?;
            accumulate_expert_bundle_stats(&mut operation_stats, delta);
        }
        Ok(operation_stats)
    }

    fn qwen35_decode_expert_bundle_missing_allocation_bytes(
        &self,
        bundle: Qwen35DecodeExpertBundle<'_>,
    ) -> usize {
        let mut missing = std::collections::BTreeMap::new();
        for weights in [bundle.gate, bundle.up, bundle.down] {
            let key = q4k_resident_key(weights);
            if !self.resident_q4k.contains_key(&key) {
                missing.entry(key).or_insert(weights);
            }
        }
        if missing.len() < 2 {
            return missing.values().map(|weights| weights.len()).sum();
        }
        missing.values().fold(0usize, |slab_bytes, weights| {
            align_up(slab_bytes, 256).saturating_add(weights.len())
        })
    }

    fn admit_qwen35_decode_expert_bundle_roles(
        &mut self,
        bundle: Qwen35DecodeExpertBundle<'_>,
        bundle_key: SparseExpertCacheKey,
        protected_keys: &HashSet<(usize, usize)>,
        eviction_plan: ResidentQ4kEvictionPlan,
        additional_oom_reload_budget: u64,
    ) -> Result<ExpertBundleCacheStats, String> {
        let mut missing = std::collections::BTreeMap::new();
        for weights in [bundle.gate, bundle.up, bundle.down] {
            let key = q4k_resident_key(weights);
            if !self.resident_q4k.contains_key(&key) {
                missing.entry(key).or_insert(weights);
            }
        }
        if missing.is_empty() {
            return Ok(ExpertBundleCacheStats::default());
        }

        let admitted_bytes = missing.values().fold(0u64, |bytes, weights| {
            bytes.saturating_add(weights.len() as u64)
        });
        let missing_weights = missing.values().copied().collect::<Vec<_>>();
        let mut operation_stats = self.execute_resident_q4k_eviction_plan(eviction_plan)?;
        let admission_result = if missing_weights.len() == 1 {
            self.preload_resident_q4k_weight_slice_for_profitable_bundle(
                missing_weights[0],
                protected_keys,
                additional_oom_reload_budget,
            )?
        } else {
            self.batch_resident_q4k_slot_misses_many_for_profitable_bundle(
                &[missing_weights.as_slice()],
                &HashMap::new(),
                protected_keys,
                additional_oom_reload_budget,
            )?
        };
        accumulate_expert_bundle_stats(&mut operation_stats, admission_result.evictions);
        if !admission_result.uploaded
            || [bundle.gate, bundle.up, bundle.down]
                .into_iter()
                .any(|weights| !self.resident_q4k.contains_key(&q4k_resident_key(weights)))
        {
            return Ok(operation_stats);
        }

        let roles = [bundle.gate, bundle.up, bundle.down]
            .into_iter()
            .map(q4k_resident_key)
            .collect::<HashSet<_>>();
        if self
            .register_qwen35_q2q3_bundle_ownership(bundle_key, &roles)
            .is_none()
        {
            return Ok(operation_stats);
        }
        cache_stats().record_expert_bundle_h2d(admitted_bytes, false);
        let mut admission_delta = ExpertBundleCacheStats::default();
        admission_delta.bundle_admissions = 1;
        admission_delta.admitted_bytes = admitted_bytes;
        cache_stats().record_expert_bundles(admission_delta);
        admission_delta.h2d_bytes = admitted_bytes;
        accumulate_expert_bundle_stats(&mut operation_stats, admission_delta);
        Ok(operation_stats)
    }
    #[cfg(test)]
    pub(in crate::runtime) fn qwen35_decode_selected_expert_bundle_stats_for_test(
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        resident_keys: &HashSet<(usize, usize)>,
    ) -> ExpertBundleCacheStats {
        let expert_ids = (0..gate_weights.len()).collect::<Vec<_>>();
        Self::qwen35_decode_selected_expert_bundle_stats_with_ids_for_test(
            &expert_ids,
            gate_weights,
            up_weights,
            down_weights,
            resident_keys,
        )
    }

    #[cfg(test)]
    pub(in crate::runtime) fn qwen35_decode_selected_expert_bundle_stats_with_ids_for_test(
        expert_ids: &[usize],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        resident_keys: &HashSet<(usize, usize)>,
    ) -> ExpertBundleCacheStats {
        let route_weights = vec![0.0; gate_weights.len()];
        let bundles = qwen35_decode_expert_bundles(
            expert_ids,
            gate_weights,
            up_weights,
            down_weights,
            &route_weights,
        )
        .expect("valid selected expert bundles");
        qwen35_decode_selected_expert_bundle_stats(&bundles, |weights| {
            resident_keys.contains(&q4k_resident_key(weights))
        })
    }

    #[cfg(test)]
    pub(in crate::runtime) fn qwen35_decode_expert_bundle_routes_for_test(
        expert_ids: &[usize],
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
    ) -> Result<Vec<(usize, usize, f64)>, String> {
        qwen35_decode_expert_bundles(
            expert_ids,
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
        )
        .map(|bundles| {
            bundles
                .into_iter()
                .map(|bundle| (bundle.expert_id, bundle.slot, bundle.aggregate_route_weight))
                .collect()
        })
    }

    #[cfg(test)]
    pub(in crate::runtime) fn qwen35_with_decode_reuse_score_override_for_test<T>(
        &mut self,
        score: u64,
        run: impl FnOnce(&mut Self) -> T,
    ) -> T {
        struct ResetReuseScore(Option<u64>);

        impl Drop for ResetReuseScore {
            fn drop(&mut self) {
                QWEN35_DECODE_REUSE_SCORE_OVERRIDE.with(|score| score.set(self.0));
            }
        }

        let previous =
            QWEN35_DECODE_REUSE_SCORE_OVERRIDE.with(|current| current.replace(Some(score)));
        let _reset = ResetReuseScore(previous);
        run(self)
    }

    #[cfg(test)]
    pub(in crate::runtime) fn qwen35_decode_expert_bundle_candidate_for_test(
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        resident_keys: &HashSet<(usize, usize)>,
    ) -> ExpertBundleAdmissionCandidate {
        qwen35_decode_expert_bundle_candidate(
            Qwen35DecodeExpertBundle {
                expert_id: 0,
                slot: 0,
                aggregate_route_weight: 0.0,
                gate,
                up,
                down,
            },
            SparseExpertCacheKey::new(0, 0),
            |weights| resident_keys.contains(&q4k_resident_key(weights)),
        )
    }

    fn validate_qwen35_q2q3_selected_weights(
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        n_ff: usize,
        n_embd: usize,
        input: Option<&[f32]>,
    ) -> Result<(), String> {
        let selected = gate_weights.len();
        if selected != up_weights.len() || selected != down_weights.len() {
            return Err(format!(
                "Qwen35 Q2_K/Q3_K selected length mismatch: gate={} up={} down={}",
                selected,
                up_weights.len(),
                down_weights.len()
            ));
        }
        if selected > 32 {
            return Err(format!(
                "Qwen35 Q2_K/Q3_K selected path supports up to 32 slots, got {selected}"
            ));
        }
        if n_ff == 0 || n_embd == 0 || n_ff % 256 != 0 || n_embd % 256 != 0 {
            return Err(format!(
                "Qwen35 Q2_K/Q3_K dims must be nonzero and divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        if let Some(input) = input {
            if input.len() != n_embd {
                return Err(format!(
                    "Qwen35 Q2_K/Q3_K input length mismatch: got {}, expected {n_embd}",
                    input.len()
                ));
            }
        }

        let gate_row_bytes = (n_embd / 256)
            .checked_mul(84)
            .ok_or_else(|| "Qwen35 Q2_K gate row byte count overflow".to_string())?;
        let gate_bytes = n_ff
            .checked_mul(gate_row_bytes)
            .ok_or_else(|| "Qwen35 Q2_K gate byte count overflow".to_string())?;
        let down_row_bytes = (n_ff / 256)
            .checked_mul(110)
            .ok_or_else(|| "Qwen35 Q3_K down row byte count overflow".to_string())?;
        let down_bytes = n_embd
            .checked_mul(down_row_bytes)
            .ok_or_else(|| "Qwen35 Q3_K down byte count overflow".to_string())?;
        for (slot, weights) in gate_weights.iter().enumerate() {
            if weights.len() != gate_bytes {
                return Err(format!(
                    "Qwen35 Q2_K gate[{slot}] byte mismatch: got {}, expected {gate_bytes}",
                    weights.len()
                ));
            }
        }
        for (slot, weights) in up_weights.iter().enumerate() {
            if weights.len() != gate_bytes {
                return Err(format!(
                    "Qwen35 Q2_K up[{slot}] byte mismatch: got {}, expected {gate_bytes}",
                    weights.len()
                ));
            }
        }
        for (slot, weights) in down_weights.iter().enumerate() {
            if weights.len() != down_bytes {
                return Err(format!(
                    "Qwen35 Q3_K down[{slot}] byte mismatch: got {}, expected {down_bytes}",
                    weights.len()
                ));
            }
        }
        Ok(())
    }

    fn qwen35_selected_bundle_resident_mask(
        &self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Vec<bool> {
        gate_weights
            .iter()
            .zip(up_weights)
            .zip(down_weights)
            .map(|((&gate, &up), &down)| {
                [gate, up, down]
                    .into_iter()
                    .all(|weights| self.resident_q4k.contains_key(&q4k_resident_key(weights)))
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prepare_selected_bundle_residency(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<Vec<bool>, String> {
        Self::validate_qwen35_q2q3_selected_weights(
            gate_weights,
            up_weights,
            down_weights,
            n_ff,
            n_embd,
            None,
        )?;
        if selected_expert_ids.len() != gate_weights.len() {
            return Err(format!(
                "Qwen35 Q2_K/Q3_K selected expert ID mismatch: ids={} slots={}",
                selected_expert_ids.len(),
                gate_weights.len()
            ));
        }
        validate_qwen35_decode_expert_bundle_inputs(
            selected_expert_ids,
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
        )?;

        if !bundle_observation_receipt.consumed() {
            if let Some(layer_idx) = layer_idx {
                self.raise_resident_q4k_limit_for_qwen35_target_decode()?;
                self.admit_qwen35_decode_expert_bundles(
                    layer_idx,
                    selected_expert_ids,
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                    bundle_observation_receipt,
                    CurrentLookupTransfer::NoUpload,
                )?;
            } else {
                qwen35_decode_expert_bundles(
                    selected_expert_ids,
                    gate_weights,
                    up_weights,
                    down_weights,
                    route_weights,
                )?;
                bundle_observation_receipt.mark_consumed();
            }
        }

        Ok(self.qwen35_selected_bundle_resident_mask(gate_weights, up_weights, down_weights))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_per_slot_resident(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if down_quant != 11 {
            return Err(format!(
                "Qwen35 resident per-slot path requires Q3_K down quant code 11, got {down_quant}"
            ));
        }
        Self::validate_qwen35_q2q3_selected_weights(
            gate_weights,
            up_weights,
            down_weights,
            n_ff,
            n_embd,
            Some(input),
        )?;
        let selected = gate_weights.len();
        if selected == 0 {
            return Ok(Vec::new());
        }
        let resident_mask =
            self.qwen35_selected_bundle_resident_mask(gate_weights, up_weights, down_weights);
        if let Some(slot) = resident_mask.iter().position(|resident| !resident) {
            return Err(format!(
                "Qwen35 resident per-slot path requires every gate/up/down role resident; slot {slot} is missing"
            ));
        }

        let mid_len = selected
            .checked_mul(n_ff)
            .ok_or_else(|| "Qwen35 resident per-slot intermediate length overflow".to_string())?;
        let mid_bytes = mid_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                "Qwen35 resident per-slot intermediate byte count overflow".to_string()
            })?;
        let output_len = selected
            .checked_mul(n_embd)
            .ok_or_else(|| "Qwen35 resident per-slot output length overflow".to_string())?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Qwen35 resident per-slot output byte count overflow".to_string())?;
        let ptr_bytes = selected
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "Qwen35 resident per-slot pointer byte count overflow".to_string())?;
        let meta_bytes = ptr_bytes
            .checked_mul(3)
            .ok_or_else(|| "Qwen35 resident per-slot metadata byte count overflow".to_string())?;

        self.set_current()?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let gate_dev = self.compute_mid_a_ptr(mid_bytes)?;
        let up_dev = self.compute_mid_b_ptr(mid_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(meta_bytes)?;
        let up_ptrs_dev = gate_ptrs_dev + ptr_bytes as u64;
        let down_ptrs_dev = gate_ptrs_dev + (ptr_bytes * 2) as u64;

        let collect_ptrs = |role: &str, slot_weights: &[&[u8]]| -> Result<Vec<u64>, String> {
            slot_weights
                .iter()
                .enumerate()
                .map(|(slot, &weights)| {
                    self.resident_q4k
                        .get(&q4k_resident_key(weights))
                        .map(|resident| resident.ptr)
                        .ok_or_else(|| {
                            format!(
                                "Qwen35 resident per-slot {role}[{slot}] was evicted before launch"
                            )
                        })
                })
                .collect()
        };
        let gate_ptrs = collect_ptrs("gate", gate_weights)?;
        let up_ptrs = collect_ptrs("up", up_weights)?;
        let down_ptrs = collect_ptrs("down", down_weights)?;
        let mut meta_stack = [0u8; 32 * std::mem::size_of::<u64>() * 3];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gate_ptrs.as_ptr().cast::<u8>(),
                meta_stack.as_mut_ptr(),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                up_ptrs.as_ptr().cast::<u8>(),
                meta_stack.as_mut_ptr().add(ptr_bytes),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                down_ptrs.as_ptr().cast::<u8>(),
                meta_stack.as_mut_ptr().add(ptr_bytes * 2),
                ptr_bytes,
            );
            self.api.memcpy_htod_async(
                gate_ptrs_dev,
                meta_stack.as_ptr().cast::<libc::c_void>(),
                meta_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.launch_selected_q2k_gate_up_gemv_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            n_ff,
            selected,
            n_embd / 256,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        self.launch_q3k_selected_down_silu_per_slot(
            down_ptrs_dev,
            n_embd,
            selected,
            n_ff / 256,
            gate_dev,
            up_dev,
            output_dev,
        )?;

        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[cfg(test)]
    pub(in crate::runtime) fn qwen35_decode_selected_q4k_ptrs_for_test(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        let mut bundle_observation_receipt = ExpertBundleObservationReceipt::default();
        self.decode_selected_weight_ptrs(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            Some(&mut bundle_observation_receipt),
        )
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_decode_selected_q4k_ptrs_with_receipt_for_test(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        self.decode_selected_weight_ptrs(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            Some(bundle_observation_receipt),
        )
    }

    #[cfg(test)]
    pub(in crate::runtime) fn qwen35_expert_bundle_history_state_for_test(
        &self,
    ) -> (usize, u64, usize) {
        (
            self.qwen35_expert_bundle_reuse_history.capacity(),
            self.qwen35_expert_bundle_reuse_history.epoch(),
            self.qwen35_expert_bundle_reuse_history.len(),
        )
    }

    fn promote_qwen35_decode_hot_resident_slots(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        tracked_expert_bundle_keys: Option<&HashSet<(usize, usize)>>,
        protected_keys: Option<&HashSet<(usize, usize)>>,
    ) -> Result<(), String> {
        if !qwen35_decode_hot_resident_enabled() {
            return Ok(());
        }
        let available = self
            .resident_q4k_limit
            .saturating_sub(self.resident_q4k_bytes);
        let mut budget_left =
            qwen35_decode_hot_resident_budget_bytes(self.resident_q4k_limit).min(available);
        if budget_left == 0 {
            return Ok(());
        }

        let mut slots = (0..gate_weights.len()).collect::<Vec<_>>();
        slots.sort_by(|&left, &right| {
            let right_weight = route_weights
                .get(right)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(0.0);
            let left_weight = route_weights
                .get(left)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(0.0);
            right_weight
                .partial_cmp(&left_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(&right))
        });

        let mut promoted_keys = HashSet::new();
        if qwen35_decode_hot_resident_batch_enabled() {
            let mut promote_gate = Vec::new();
            let mut promote_up = Vec::new();
            let mut promote_down = Vec::new();
            for slot in slots {
                let mut missing_bytes = 0usize;
                let mut missing_gate = None;
                let mut missing_up = None;
                let mut missing_down = None;
                for (idx, weights) in [gate_weights[slot], up_weights[slot], down_weights[slot]]
                    .into_iter()
                    .enumerate()
                {
                    let key = q4k_resident_key(weights);
                    if self.resident_q4k.contains_key(&key) || !promoted_keys.insert(key) {
                        continue;
                    }
                    missing_bytes = missing_bytes.saturating_add(weights.len());
                    match idx {
                        0 => missing_gate = Some(weights),
                        1 => missing_up = Some(weights),
                        2 => missing_down = Some(weights),
                        _ => unreachable!("three selected weight groups"),
                    }
                }
                if missing_bytes == 0 || missing_bytes > budget_left {
                    continue;
                }
                if let Some(weights) = missing_gate {
                    promote_gate.push(weights);
                }
                if let Some(weights) = missing_up {
                    promote_up.push(weights);
                }
                if let Some(weights) = missing_down {
                    promote_down.push(weights);
                }
                budget_left = budget_left.saturating_sub(missing_bytes);
                if budget_left == 0 {
                    break;
                }
            }
            if !promote_gate.is_empty() || !promote_up.is_empty() || !promote_down.is_empty() {
                let local_ptrs = HashMap::new();
                match (tracked_expert_bundle_keys, protected_keys) {
                    (Some(tracked_keys), Some(protected_keys)) => {
                        self.batch_resident_q4k_slot_misses_many_recording_expert_bundle_h2d_protecting(
                            &[&promote_gate, &promote_up, &promote_down],
                            &local_ptrs,
                            protected_keys,
                            tracked_keys,
                        )?;
                    }
                    (None, Some(protected_keys)) => {
                        self.batch_resident_q4k_slot_misses_many_protecting(
                            &[&promote_gate, &promote_up, &promote_down],
                            &local_ptrs,
                            protected_keys,
                        )?;
                    }
                    (Some(tracked_keys), None) => {
                        self.batch_resident_q4k_slot_misses_many_recording_expert_bundle_h2d(
                            &[&promote_gate, &promote_up, &promote_down],
                            &local_ptrs,
                            tracked_keys,
                        )?;
                    }
                    (None, None) => {
                        self.batch_resident_q4k_slot_misses_many(
                            &[&promote_gate, &promote_up, &promote_down],
                            &local_ptrs,
                        )?;
                    }
                }
            }
            return Ok(());
        }

        for slot in slots {
            let mut missing = Vec::new();
            let mut missing_bytes = 0usize;
            for weights in [gate_weights[slot], up_weights[slot], down_weights[slot]] {
                let key = q4k_resident_key(weights);
                if self.resident_q4k.contains_key(&key) || !promoted_keys.insert(key) {
                    continue;
                }
                missing_bytes = missing_bytes.saturating_add(weights.len());
                missing.push(weights);
            }
            if missing.is_empty() || missing_bytes > budget_left {
                continue;
            }
            let mut promoted_bytes = 0usize;
            for weights in missing {
                let admitted = if let Some(protected_keys) = protected_keys {
                    self.preload_resident_q4k_weight_slice_protecting(weights, protected_keys)?
                } else {
                    self.preload_resident_q4k_weight_slice(weights)?
                };
                if admitted {
                    promoted_bytes = promoted_bytes.saturating_add(weights.len());
                    if tracked_expert_bundle_keys.is_some_and(|tracked_keys| {
                        tracked_keys.contains(&q4k_resident_key(weights))
                    }) {
                        cache_stats().record_expert_bundle_h2d(weights.len() as u64, false);
                    }
                }
            }
            budget_left = budget_left.saturating_sub(promoted_bytes);
            if budget_left == 0 {
                break;
            }
        }
        Ok(())
    }

    fn trace_qwen35_sparse_buffer(
        &mut self,
        label: &str,
        dev: u64,
        len: usize,
    ) -> Result<(), String> {
        if std::env::var("RNB_CUDA_QWEN35_SPARSE_TRACE")
            .ok()
            .as_deref()
            != Some("1")
        {
            return Ok(());
        }
        let mut host = vec![0.0f32; len];
        unsafe {
            self.api.memcpy_dtoh_async(
                host.as_mut_ptr().cast::<libc::c_void>(),
                dev,
                len * std::mem::size_of::<f32>(),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        let nonfinite = host.iter().filter(|value| !value.is_finite()).count();
        let max_abs = host
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        let head = host.iter().take(4).copied().collect::<Vec<_>>();
        eprintln!(
            "[cuda-qwen35-sparse-trace] label={} len={} nonfinite={} max_abs={:.6e} head={:?}",
            label, len, nonfinite, max_abs, head
        );
        Ok(())
    }

    pub(in crate::runtime) fn qwen35_sparse_experts(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let trace_cache = std::env::var("RNB_CUDA_CACHE_TRACE").ok().as_deref() == Some("1");
        let trace_phase = std::env::var("RNB_CUDA_PHASE_TRACE").ok().as_deref() == Some("1");
        let trace_t0 = (trace_cache || trace_phase).then(std::time::Instant::now);
        let trace_before = trace_cache.then(cache_snapshot);
        let mut weight_ptr_ms = 0.0f64;
        let mut setup_h2d_ms = 0.0f64;
        let mut kernels_ms = 0.0f64;
        let mut dtoh_ms = 0.0f64;
        let selected = gate_weights.len();
        if selected <= 32 {
            let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
            let output_bytes = n_embd * std::mem::size_of::<f32>();
            let output_dev = self.compute_output_ptr(output_bytes)?;
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input),
                    self.stream,
                )?;
            }
            self.qwen35_sparse_experts_to_dev(
                gate_weights,
                up_weights,
                down_weights,
                route_weights,
                layer_idx,
                selected_expert_ids,
                bundle_observation_receipt,
                down_quant,
                n_ff,
                n_embd,
                input_dev,
                output_dev,
            )?;
            let mut output = vec![0.0f32; n_embd];
            unsafe {
                self.api.memcpy_dtoh_async(
                    output.as_mut_ptr().cast::<libc::c_void>(),
                    output_dev,
                    output_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            return Ok(output);
        }
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let gate_dev = self.compute_mid_a_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        let output_bytes = n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(selected * std::mem::size_of::<u64>())?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(selected * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(selected * std::mem::size_of::<u64>())?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        self.set_current()?;
        let (gate_ptrs, up_ptrs, down_ptrs, temp_slab_ptrs) = self.decode_selected_weight_ptrs(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            (down_quant == 11).then_some(bundle_observation_receipt),
        )?;
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            weight_ptr_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                gate_ptrs_dev,
                gate_ptrs.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(gate_ptrs.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                up_ptrs_dev,
                up_ptrs.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(up_ptrs.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                down_ptrs_dev,
                down_ptrs.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(down_ptrs.as_slice()),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                route_dev,
                route_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(route_weights),
                self.stream,
            )?;
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            setup_h2d_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        self.launch_selected_q4k_gate_up_gemv_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            n_ff,
            selected,
            n_embd / 256,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        if !temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled() {
            unsafe { self.api.stream_synchronize(self.copy_stream)? };
        }
        match down_quant {
            12 => self.launch_selected_down_silu_rowreduce(
                "rnb_q4k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            )?,
            13 => self.launch_selected_down_silu_rowreduce(
                "rnb_q5k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            )?,
            14 => self.launch_selected_down_silu_rowreduce(
                "rnb_q6k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            )?,
            other => return Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            kernels_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let mut output = vec![0.0f32; n_embd];
        let phase_t0 = trace_phase.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        if let Some(t0) = phase_t0 {
            dtoh_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        if let (Some(t0), Some(before)) = (trace_t0, trace_before) {
            let after = cache_snapshot();
            let delta = after.delta(before);
            let hit_rate = if delta.lookups == 0 {
                0.0
            } else {
                (delta.hits as f64 * 100.0) / delta.lookups as f64
            };
            eprintln!(
                "[cuda-cache] qwen35_decode tokens=1 selected={} lookups={} hits={} misses={} hit_rate={:.1}% evictions={} resident_upload_mb={:.2} temp_upload_mb={:.2} resident_q4k_mb={:.2} q2q3_bundle_lookups={} q2q3_bundle_hits={} q2q3_bundle_partial_hits={} q2q3_bundle_misses={} q2q3_bundle_admissions={} q2q3_bundle_admitted_bytes={} q2q3_bundle_evictions={} q2q3_bundle_evicted_bytes={} q2q3_resident_payload_bytes={} q2q3_bundle_h2d_mb={:.2} q2q3_bundle_temp_h2d_mb={:.2} q2q3_bundle_h2d_bytes_per_token={:.1} elapsed_ms={:.1}",
                selected,
                delta.lookups,
                delta.hits,
                delta.misses,
                hit_rate,
                delta.evictions,
                delta.resident_upload_bytes as f64 / (1024.0 * 1024.0),
                delta.temp_upload_bytes as f64 / (1024.0 * 1024.0),
                self.resident_q4k_bytes as f64 / (1024.0 * 1024.0),
                delta.expert_bundles.bundle_lookups,
                delta.expert_bundles.bundle_hits,
                delta.expert_bundles.bundle_partial_hits,
                delta.expert_bundles.bundle_misses,
                delta.expert_bundles.bundle_admissions,
                delta.expert_bundles.admitted_bytes,
                delta.expert_bundles.bundle_evictions,
                delta.expert_bundles.evicted_bytes,
                after.resident_payload_bytes,
                delta.expert_bundles.h2d_bytes as f64 / (1024.0 * 1024.0),
                delta.expert_bundles.temp_h2d_bytes as f64 / (1024.0 * 1024.0),
                delta.expert_bundles.h2d_bytes_per_token(1),
                t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        if let Some(t0) = trace_t0.filter(|_| trace_phase) {
            eprintln!(
                "[cuda-phase] qwen35_decode selected={} weight_ptr_ms={:.1} setup_h2d_ms={:.1} kernels_ms={:.1} dtoh_ms={:.1} total_ms={:.1}",
                selected,
                weight_ptr_ms,
                setup_h2d_ms,
                kernels_ms,
                dtoh_ms,
                t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_into(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.qwen35_sparse_experts_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
        )?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_add_residual_into(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        residual: &mut [f32],
    ) -> Result<(), String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(residual);
        let moe_output_dev = self.compute_output_ptr(output_bytes)?;
        let residual_dev = self.compute_aux_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                residual_dev,
                residual.as_ptr().cast::<libc::c_void>(),
                output_bytes,
                self.stream,
            )?;
        }
        #[cfg(test)]
        self.fail_qwen35_decode_for_test(
            Qwen35DecodeFailurePoint::ResidualBeforeSelectedObservation,
        )?;
        self.qwen35_sparse_experts_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            moe_output_dev,
        )?;
        self.launch_add_f32_inplace(residual_dev, moe_output_dev, n_embd)?;
        unsafe {
            self.api.memcpy_dtoh_async(
                residual.as_mut_ptr().cast::<libc::c_void>(),
                residual_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_iq4xs(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.qwen35_sparse_experts_iq4xs_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
        )?;
        let mut output = vec![0.0f32; n_embd];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_iq4xs_into(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.qwen35_sparse_experts_iq4xs_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
        )?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_iq4xs_add_residual_into(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        residual: &mut [f32],
    ) -> Result<(), String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(residual);
        let moe_output_dev = self.compute_output_ptr(output_bytes)?;
        let residual_dev = self.compute_aux_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                residual_dev,
                residual.as_ptr().cast::<libc::c_void>(),
                output_bytes,
                self.stream,
            )?;
        }
        self.qwen35_sparse_experts_iq4xs_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            moe_output_dev,
        )?;
        self.launch_add_f32_inplace(residual_dev, moe_output_dev, n_embd)?;
        unsafe {
            self.api.memcpy_dtoh_async(
                residual.as_mut_ptr().cast::<libc::c_void>(),
                residual_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_decode_moe_shared_sparse_into(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
        down_quant: u32,
        shared_gate: &[u8],
        shared_up: &[u8],
        shared_down: &[u8],
        shared_route: f32,
        shared_down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let shared_output_dev = self.compute_aux_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.qwen35_sparse_experts_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
        )?;
        #[cfg(test)]
        self.fail_qwen35_decode_for_test(Qwen35DecodeFailurePoint::CombinedShared)?;
        self.qwen35_sparse_experts_to_dev_with_optional_receipt(
            &[shared_gate],
            &[shared_up],
            &[shared_down],
            &[shared_route],
            None,
            &[],
            shared_down_quant,
            n_ff,
            n_embd,
            input_dev,
            shared_output_dev,
            None,
        )?;
        #[cfg(test)]
        self.fail_qwen35_decode_for_test(Qwen35DecodeFailurePoint::CombinedAdd)?;
        self.launch_add_f32_inplace(output_dev, shared_output_dev, n_embd)?;
        #[cfg(test)]
        self.fail_qwen35_decode_for_test(Qwen35DecodeFailurePoint::CombinedDtoh)?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        #[cfg(test)]
        self.fail_qwen35_decode_for_test(Qwen35DecodeFailurePoint::CombinedSync)?;
        self.stream_synchronize()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_iq4xs_to_dev(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let selected = gate_weights.len();
        if selected == 0 {
            return Err(
                "Qwen35 device sparse IQ4_XS MoE requires at least one selected expert".to_string(),
            );
        }
        if selected != up_weights.len()
            || selected != down_weights.len()
            || selected != route_weights.len()
        {
            return Err(format!(
                "Qwen35 device sparse IQ4_XS MoE selection mismatch: gate={} up={} down={} route={}",
                selected,
                up_weights.len(),
                down_weights.len(),
                route_weights.len()
            ));
        }
        if selected > 32 {
            return Err(format!(
                "Qwen35 device sparse IQ4_XS MoE supports up to 32 slots, got {selected}"
            ));
        }

        let gate_dev = self.compute_mid_a_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        self.set_current()?;

        let (gate_ptrs, up_ptrs, down_ptrs, temp_slab_ptrs) = self.decode_selected_weight_ptrs(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            None,
            &[],
            None,
        )?;

        let ptr_bytes = selected * std::mem::size_of::<u64>();
        let route_bytes = std::mem::size_of_val(route_weights);
        let meta_bytes = ptr_bytes * 3 + route_bytes;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(meta_bytes)?;
        let up_ptrs_dev = gate_ptrs_dev + ptr_bytes as u64;
        let down_ptrs_dev = gate_ptrs_dev + (ptr_bytes * 2) as u64;
        let route_dev = gate_ptrs_dev + (ptr_bytes * 3) as u64;
        let mut meta = vec![0u8; meta_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gate_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr(),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                up_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                down_ptrs.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes * 2),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr().cast::<u8>(),
                meta.as_mut_ptr().add(ptr_bytes * 3),
                route_bytes,
            );
        }
        unsafe {
            self.api.memcpy_htod_async(
                gate_ptrs_dev,
                meta.as_ptr().cast::<libc::c_void>(),
                meta_bytes,
                self.stream,
            )?;
        }

        self.launch_selected_iq4_xs_gate_up_gemv_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            n_ff,
            selected,
            n_embd / 256,
            input_dev,
            gate_dev,
            up_dev,
        )?;
        if !temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled() {
            unsafe { self.api.stream_synchronize(self.copy_stream)? };
        }
        match down_quant {
            12 => self.launch_selected_down_silu_rowreduce(
                "rnb_q4k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            13 => self.launch_selected_down_silu_rowreduce(
                "rnb_q5k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            14 => self.launch_selected_down_silu_rowreduce(
                "rnb_q6k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            23 => self.launch_selected_down_silu_rowreduce(
                "rnb_iq4_xs_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            other => Err(format!(
                "unsupported Qwen35 CUDA IQ4_XS down quant code {other}"
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_to_dev(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        bundle_observation_receipt: &mut ExpertBundleObservationReceipt,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        self.qwen35_sparse_experts_to_dev_with_optional_receipt(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
            (down_quant == 11).then_some(bundle_observation_receipt),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_sparse_experts_to_dev_with_optional_receipt(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        layer_idx: Option<usize>,
        selected_expert_ids: &[usize],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
        mut bundle_observation_receipt: Option<&mut ExpertBundleObservationReceipt>,
    ) -> Result<(), String> {
        let trace_phase = std::env::var("RNB_CUDA_PHASE_TRACE").ok().as_deref() == Some("1");
        let trace_kernel = std::env::var("RNB_CUDA_KERNEL_TRACE").ok().as_deref() == Some("1");
        let trace_cache = std::env::var("RNB_CUDA_CACHE_TRACE").ok().as_deref() == Some("1");
        let pending_bundle_stats_before = bundle_observation_receipt
            .as_deref()
            .map(|receipt| receipt.pending_stats())
            .unwrap_or_default();
        let trace_t0 = trace_phase.then(std::time::Instant::now);
        let cache_before = trace_cache.then(cache_snapshot);
        let cache_t0 = trace_cache.then(std::time::Instant::now);
        let mut weight_ptr_ms = 0.0f64;
        let mut meta_h2d_ms = 0.0f64;
        let mut gate_up_ms = 0.0f64;
        let mut down_ms = 0.0f64;
        let selected = gate_weights.len();
        let q2q3 = down_quant == 11;
        if selected == 0 {
            return Err(
                "Qwen35 device sparse MoE requires at least one selected expert".to_string(),
            );
        }
        if selected != up_weights.len()
            || selected != down_weights.len()
            || selected != route_weights.len()
        {
            return Err(format!(
                "Qwen35 device sparse MoE selection mismatch: gate={} up={} down={} route={}",
                selected,
                up_weights.len(),
                down_weights.len(),
                route_weights.len()
            ));
        }
        if selected > 32 {
            return Err(format!(
                "Qwen35 device sparse MoE supports up to 32 slots, got {selected}"
            ));
        }

        let gate_dev = self.compute_mid_a_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(selected * n_ff * std::mem::size_of::<f32>())?;
        let phase_t0 = trace_phase.then(std::time::Instant::now);
        self.set_current()?;
        let (gate_ptrs, up_ptrs, down_ptrs, temp_slab_ptrs) = self.decode_selected_weight_ptrs(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            layer_idx,
            selected_expert_ids,
            bundle_observation_receipt.as_deref_mut(),
        )?;
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            weight_ptr_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        let ptr_bytes = selected * std::mem::size_of::<u64>();
        let route_bytes = std::mem::size_of_val(route_weights);
        let meta_bytes = ptr_bytes * 3 + route_bytes;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(meta_bytes)?;
        let up_ptrs_dev = gate_ptrs_dev + ptr_bytes as u64;
        let down_ptrs_dev = gate_ptrs_dev + (ptr_bytes * 2) as u64;
        let route_dev = gate_ptrs_dev + (ptr_bytes * 3) as u64;
        let mut meta_stack =
            [0u8; 32 * std::mem::size_of::<u64>() * 3 + 32 * std::mem::size_of::<f32>()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gate_ptrs.as_ptr().cast::<u8>(),
                meta_stack.as_mut_ptr(),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                up_ptrs.as_ptr().cast::<u8>(),
                meta_stack.as_mut_ptr().add(ptr_bytes),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                down_ptrs.as_ptr().cast::<u8>(),
                meta_stack.as_mut_ptr().add(ptr_bytes * 2),
                ptr_bytes,
            );
            std::ptr::copy_nonoverlapping(
                route_weights.as_ptr().cast::<u8>(),
                meta_stack.as_mut_ptr().add(ptr_bytes * 3),
                route_bytes,
            );
        }
        unsafe {
            self.api.memcpy_htod_async(
                gate_ptrs_dev,
                meta_stack.as_ptr().cast::<libc::c_void>(),
                meta_bytes,
                self.stream,
            )?;
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            meta_h2d_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        if !q2q3 && tuning::qwen35_decode_moe_graph_enabled() {
            if !temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled() {
                unsafe { self.api.stream_synchronize(self.copy_stream)? };
            }
            let phase_t0 = trace_phase.then(std::time::Instant::now);
            let result = self.launch_qwen35_sparse_graph(
                down_quant,
                n_ff,
                n_embd,
                selected,
                input_dev,
                gate_dev,
                up_dev,
                output_dev,
                gate_ptrs_dev,
                up_ptrs_dev,
                down_ptrs_dev,
                route_dev,
            );
            if let Some(t0) = phase_t0 {
                self.stream_synchronize()?;
                let kernels_ms = t0.elapsed().as_micros() as f64 / 1000.0;
                if let Some(total_t0) = trace_t0 {
                    eprintln!(
                        "[cuda-phase] qwen35_decode_to_dev selected={} quant={} weight_ptr_ms={:.3} meta_h2d_ms={:.3} graph_ms={:.3} total_ms={:.3}",
                        selected,
                        down_quant,
                        weight_ptr_ms,
                        meta_h2d_ms,
                        kernels_ms,
                        total_t0.elapsed().as_micros() as f64 / 1000.0
                    );
                }
            }
            match result {
                Ok(()) => return Ok(()),
                Err(err) if qwen35_decode_moe_graph_can_fallback(&err) => {
                    self.qwen35_sparse_graphs.clear();
                    if trace_phase {
                        eprintln!("[cuda-phase] qwen35_decode_to_dev graph fallback: {}", err);
                    }
                }
                Err(err) => return Err(err),
            }
        }

        let phase_t0 = trace_kernel.then(std::time::Instant::now);
        if q2q3 {
            self.launch_selected_q2k_gate_up_gemv_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                n_ff,
                selected,
                n_embd / 256,
                input_dev,
                gate_dev,
                up_dev,
            )?;
        } else {
            self.launch_selected_q4k_gate_up_gemv_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                n_ff,
                selected,
                n_embd / 256,
                input_dev,
                gate_dev,
                up_dev,
            )?;
        }
        self.trace_qwen35_sparse_buffer("gate", gate_dev, selected * n_ff)?;
        self.trace_qwen35_sparse_buffer("up", up_dev, selected * n_ff)?;
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            gate_up_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        if !temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled() {
            unsafe { self.api.stream_synchronize(self.copy_stream)? };
        }
        let phase_t0 = trace_kernel.then(std::time::Instant::now);
        let result = match down_quant {
            11 => self.launch_selected_down_silu_rowreduce(
                "rnb_q3k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            12 => self.launch_selected_down_silu_rowreduce(
                "rnb_q4k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            13 => self.launch_selected_down_silu_rowreduce(
                "rnb_q5k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            14 => self.launch_selected_down_silu_rowreduce(
                "rnb_q6k_selected_down_silu_rowreduce",
                down_ptrs_dev,
                n_embd,
                selected,
                n_ff / 256,
                gate_dev,
                up_dev,
                route_dev,
                output_dev,
            ),
            other => Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
        };
        if result.is_ok() {
            self.trace_qwen35_sparse_buffer("down", output_dev, n_embd)?;
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            down_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        if let Some(total_t0) = trace_t0 {
            eprintln!(
                "[cuda-phase] qwen35_decode_to_dev selected={} quant={} weight_ptr_ms={:.3} meta_h2d_ms={:.3} kernels_ms={:.3} total_ms={:.3}",
                selected,
                down_quant,
                weight_ptr_ms,
                meta_h2d_ms,
                gate_up_ms + down_ms,
                total_t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        if trace_kernel {
            eprintln!(
                "[cuda-kernel] qwen35_decode_to_dev selected={} quant={} gate_up_ms={:.3} down_ms={:.3}",
                selected, down_quant, gate_up_ms, down_ms
            );
        }
        if let (Some(before), Some(t0)) = (cache_before, cache_t0) {
            let after = cache_snapshot();
            let delta = after.delta(before);
            let hit_rate = if delta.lookups == 0 {
                0.0
            } else {
                delta.hits as f64 * 100.0 / delta.lookups as f64
            };
            let mut bundle_delta = delta.expert_bundles;
            bundle_delta.saturating_add_assign(pending_bundle_stats_before);
            eprintln!(
                "[cuda-cache] qwen35_decode_to_dev tokens=1 selected={} lookups={} hits={} misses={} hit_rate={:.1}% evictions={} resident_upload_mb={:.2} temp_upload_mb={:.2} resident_q4k_mb={:.2} q2q3_bundle_lookups={} q2q3_bundle_hits={} q2q3_bundle_partial_hits={} q2q3_bundle_misses={} q2q3_bundle_admissions={} q2q3_bundle_admitted_bytes={} q2q3_bundle_evictions={} q2q3_bundle_evicted_bytes={} q2q3_resident_payload_bytes={} q2q3_bundle_h2d_mb={:.2} q2q3_bundle_temp_h2d_mb={:.2} q2q3_bundle_h2d_bytes_per_token={:.1} elapsed_ms={:.3}",
                selected,
                delta.lookups,
                delta.hits,
                delta.misses,
                hit_rate,
                delta.evictions,
                delta.resident_upload_bytes as f64 / (1024.0 * 1024.0),
                delta.temp_upload_bytes as f64 / (1024.0 * 1024.0),
                self.resident_q4k_bytes as f64 / (1024.0 * 1024.0),
                bundle_delta.bundle_lookups,
                bundle_delta.bundle_hits,
                bundle_delta.bundle_partial_hits,
                bundle_delta.bundle_misses,
                bundle_delta.bundle_admissions,
                bundle_delta.admitted_bytes,
                bundle_delta.bundle_evictions,
                bundle_delta.evicted_bytes,
                after.resident_payload_bytes,
                bundle_delta.h2d_bytes as f64 / (1024.0 * 1024.0),
                bundle_delta.temp_h2d_bytes as f64 / (1024.0 * 1024.0),
                bundle_delta.h2d_bytes_per_token(1),
                t0.elapsed().as_micros() as f64 / 1000.0
            );
            if let Some(receipt) = bundle_observation_receipt.as_deref_mut() {
                receipt.clear_stats();
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn launch_qwen35_sparse_graph(
        &mut self,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        selected: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
        output_dev: u64,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        down_ptrs_dev: u64,
        route_dev: u64,
    ) -> Result<(), String> {
        let key = SparseMoeGraphKey {
            down_quant,
            n_ff,
            n_embd,
            selected,
            input_dev,
            gate_dev,
            up_dev,
            output_dev,
            gate_ptrs_dev,
            up_ptrs_dev,
            down_ptrs_dev,
            route_dev,
        };
        if let Some(graph) = self.qwen35_sparse_graphs.get(&key) {
            return unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)
            };
        }

        self.ensure_q4k_gemv_module()?;
        unsafe {
            self.api.stream_begin_capture(self.stream)?;
        }
        let capture_result = (|| {
            self.launch_selected_q4k_gate_up_gemv_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                n_ff,
                selected,
                n_embd / 256,
                input_dev,
                gate_dev,
                up_dev,
            )?;
            match down_quant {
                12 => self.launch_selected_down_silu_rowreduce(
                    "rnb_q4k_selected_down_silu_rowreduce",
                    down_ptrs_dev,
                    n_embd,
                    selected,
                    n_ff / 256,
                    gate_dev,
                    up_dev,
                    route_dev,
                    output_dev,
                ),
                13 => self.launch_selected_down_silu_rowreduce(
                    "rnb_q5k_selected_down_silu_rowreduce",
                    down_ptrs_dev,
                    n_embd,
                    selected,
                    n_ff / 256,
                    gate_dev,
                    up_dev,
                    route_dev,
                    output_dev,
                ),
                14 => self.launch_selected_down_silu_rowreduce(
                    "rnb_q6k_selected_down_silu_rowreduce",
                    down_ptrs_dev,
                    n_embd,
                    selected,
                    n_ff / 256,
                    gate_dev,
                    up_dev,
                    route_dev,
                    output_dev,
                ),
                other => Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
            }
        })();
        if let Err(err) = capture_result {
            unsafe {
                let _ = self.api.stream_end_capture(self.stream);
            }
            return Err(err);
        }
        let graph = unsafe { self.api.stream_end_capture(self.stream)? };
        let exec = unsafe { self.api.graph_instantiate(graph)? };
        self.qwen35_sparse_graphs.insert(
            key,
            SparseMoeGraph {
                graph: graph as usize,
                exec: exec as usize,
            },
        );
        let graph = self
            .qwen35_sparse_graphs
            .get(&key)
            .ok_or_else(|| "missing Qwen35 sparse CUDA graph".to_string())?;
        unsafe {
            self.api
                .graph_launch(graph.exec as *mut libc::c_void, self.stream)
        }
    }
}
