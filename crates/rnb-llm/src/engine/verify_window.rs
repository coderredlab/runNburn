#[derive(Debug)]
pub(crate) struct VerifyWindowResult {
    pub(crate) target_tokens: Vec<u32>,
    pub(crate) mtp_hidden_rows: Vec<f32>,
    pub(crate) hidden_dim: usize,
    pub(crate) prefix_state: Option<VerifyWindowPrefixState>,
    pub(crate) prefix_states: Vec<VerifyWindowPrefixState>,
    #[cfg(any(feature = "cuda", test))]
    pub(crate) ssm_final_states: Vec<VerifyWindowSsmLayerFinalState>,
    #[cfg(any(feature = "cuda", test))]
    pub(crate) attention_kv_states: Vec<VerifyWindowAttentionKvState>,
}

impl VerifyWindowResult {
    #[cfg(test)]
    pub(crate) fn from_device_result(
        target_tokens: Vec<u32>,
        mtp_hidden_rows: Vec<f32>,
        hidden_dim: usize,
    ) -> crate::error::Result<Self> {
        Self::from_device_parts(
            target_tokens,
            mtp_hidden_rows,
            hidden_dim,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(crate) fn from_device_result_with_prefix_states(
        target_tokens: Vec<u32>,
        mtp_hidden_rows: Vec<f32>,
        hidden_dim: usize,
        prefix_states: Vec<crate::engine::cuda_runtime::MtpDeviceVerifyPrefixState>,
    ) -> crate::error::Result<Self> {
        Self::from_device_result_with_state_payload(
            target_tokens,
            mtp_hidden_rows,
            hidden_dim,
            prefix_states,
            Vec::new(),
            Vec::new(),
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn from_device_result_with_state_payload(
        target_tokens: Vec<u32>,
        mtp_hidden_rows: Vec<f32>,
        hidden_dim: usize,
        prefix_states: Vec<crate::engine::cuda_runtime::MtpDeviceVerifyPrefixState>,
        ssm_final_states: Vec<crate::engine::cuda_runtime::MtpDeviceVerifySsmLayerFinalState>,
        attention_kv_states: Vec<crate::engine::cuda_runtime::MtpDeviceVerifyAttentionKvState>,
    ) -> crate::error::Result<Self> {
        let prefix_states = prefix_states
            .into_iter()
            .map(|prefix| VerifyWindowPrefixState {
                prefix_tokens: prefix.prefix_tokens,
                layers: prefix
                    .layers
                    .into_iter()
                    .map(|layer| VerifyWindowSsmLayerPrefixState {
                        layer_idx: layer.layer_idx,
                        conv_state: layer.conv_state,
                        resident_conv_snapshot: layer.resident_conv_snapshot,
                        delta_input: None,
                        delta_state: None,
                        resident_delta_snapshot: layer.resident_delta_snapshot,
                    })
                    .collect(),
            })
            .collect();
        let ssm_final_states = ssm_final_states
            .into_iter()
            .map(|state| VerifyWindowSsmLayerFinalState {
                layer_idx: state.layer_idx,
                conv_state: state.conv_state,
                device_resident: state.device_resident,
            })
            .collect();
        let attention_kv_states = attention_kv_states
            .into_iter()
            .map(|state| VerifyWindowAttentionKvState {
                layer_idx: state.layer_idx,
                window_tokens: state.window_tokens,
                kv_rows: state.kv_rows,
                k_bits: state.k_bits,
                v_bits: state.v_bits,
                device_resident: state.device_resident,
            })
            .collect();
        Self::from_device_parts(
            target_tokens,
            mtp_hidden_rows,
            hidden_dim,
            prefix_states,
            ssm_final_states,
            attention_kv_states,
        )
    }

    #[cfg(any(feature = "cuda", test))]
    fn from_device_parts(
        target_tokens: Vec<u32>,
        mtp_hidden_rows: Vec<f32>,
        hidden_dim: usize,
        prefix_states: Vec<VerifyWindowPrefixState>,
        ssm_final_states: Vec<VerifyWindowSsmLayerFinalState>,
        attention_kv_states: Vec<VerifyWindowAttentionKvState>,
    ) -> crate::error::Result<Self> {
        if hidden_dim == 0 {
            return Err(crate::error::LlmError::Forward(
                "MTP device verify hidden_dim must be non-zero".to_string(),
            ));
        }
        let expected = target_tokens.len().checked_mul(hidden_dim).ok_or_else(|| {
            crate::error::LlmError::Forward(
                "MTP device verify hidden rows length overflow".to_string(),
            )
        })?;
        if mtp_hidden_rows.len() != expected {
            return Err(crate::error::LlmError::Forward(format!(
                "MTP device verify hidden rows mismatch: got {}, expected {}",
                mtp_hidden_rows.len(),
                expected
            )));
        }
        Ok(Self {
            target_tokens,
            mtp_hidden_rows,
            hidden_dim,
            prefix_state: None,
            prefix_states,
            ssm_final_states,
            attention_kv_states,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.target_tokens.len()
    }

    pub(crate) fn hidden_rows(&self) -> usize {
        if self.hidden_dim == 0 {
            0
        } else {
            self.mtp_hidden_rows.len() / self.hidden_dim
        }
    }

    pub(crate) fn mtp_hidden_prefix_rows(&self, rows: usize) -> crate::error::Result<&[f32]> {
        let len = rows.checked_mul(self.hidden_dim).ok_or_else(|| {
            crate::error::LlmError::Forward("MTP hidden prefix row length overflow".to_string())
        })?;
        self.mtp_hidden_rows.get(..len).ok_or_else(|| {
            crate::error::LlmError::Forward(format!(
                "MTP hidden prefix rows mismatch: got {}, need {}",
                self.mtp_hidden_rows.len(),
                len
            ))
        })
    }

    pub(crate) fn prefix_state_for(
        &self,
        prefix_tokens: usize,
    ) -> Option<&VerifyWindowPrefixState> {
        self.prefix_state
            .as_ref()
            .filter(|state| state.prefix_tokens == prefix_tokens)
            .or_else(|| {
                self.prefix_states
                    .iter()
                    .find(|state| state.prefix_tokens == prefix_tokens)
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MtpVerifyBonus {
    Include,
    Omit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MtpVerifyWindowRequest {
    pub(crate) current_token: u32,
    pub(crate) draft_tokens: Vec<u32>,
    pub(crate) bonus: MtpVerifyBonus,
}

impl MtpVerifyWindowRequest {
    pub(crate) fn new(current_token: u32, draft_tokens: &[u32], bonus: MtpVerifyBonus) -> Self {
        Self {
            current_token,
            draft_tokens: draft_tokens.to_vec(),
            bonus,
        }
    }

    pub(crate) fn verify_tokens(&self) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(1 + self.draft_tokens.len());
        tokens.push(self.current_token);
        match self.bonus {
            MtpVerifyBonus::Include => tokens.extend_from_slice(&self.draft_tokens),
            MtpVerifyBonus::Omit => {
                if self.draft_tokens.len() > 1 {
                    tokens.extend_from_slice(&self.draft_tokens[..self.draft_tokens.len() - 1]);
                }
            }
        }
        tokens
    }

    pub(crate) fn prefix_tokens(&self) -> Vec<usize> {
        let max_prefix = match self.bonus {
            MtpVerifyBonus::Include => self.draft_tokens.len(),
            MtpVerifyBonus::Omit => self.draft_tokens.len().saturating_sub(1),
        };
        (1..=max_prefix).collect()
    }

    #[allow(dead_code)]
    pub(crate) fn shadow_commit_prefix_tokens(&self) -> Vec<usize> {
        let mut prefix_tokens = self.prefix_tokens();
        let full_window = self.verify_tokens().len();
        if full_window > 0 && prefix_tokens.last().copied() != Some(full_window) {
            prefix_tokens.push(full_window);
        }
        prefix_tokens
    }

    pub(crate) fn full_accept_committed_tokens(&self) -> usize {
        self.draft_tokens.len() + 1
    }
}

#[derive(Debug)]
pub(crate) struct VerifyWindowPrefixState {
    pub(crate) prefix_tokens: usize,
    pub(crate) layers: Vec<VerifyWindowSsmLayerPrefixState>,
}

#[derive(Debug)]
pub(crate) struct VerifyWindowSsmLayerPrefixState {
    pub(crate) layer_idx: usize,
    pub(crate) conv_state: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub(crate) resident_conv_snapshot: Option<crate::engine::cuda_runtime::DeltaStateSnapshot>,
    pub(crate) delta_input: Option<VerifyWindowSsmDeltaInput>,
    pub(crate) delta_state: Option<Vec<f32>>,
    #[cfg(feature = "cuda")]
    pub(crate) resident_delta_snapshot: Option<crate::engine::cuda_runtime::DeltaStateSnapshot>,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug)]
pub(crate) struct VerifyWindowSsmLayerFinalState {
    pub(crate) layer_idx: usize,
    pub(crate) conv_state: Vec<f32>,
    pub(crate) device_resident: bool,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug)]
pub(crate) struct VerifyWindowAttentionKvState {
    pub(crate) layer_idx: usize,
    pub(crate) window_tokens: usize,
    pub(crate) kv_rows: usize,
    pub(crate) k_bits: Vec<u16>,
    pub(crate) v_bits: Vec<u16>,
    pub(crate) device_resident: bool,
}

#[cfg(feature = "cuda")]
impl Drop for VerifyWindowSsmLayerPrefixState {
    fn drop(&mut self) {
        if let Some(snapshot) = self.resident_conv_snapshot.take() {
            let _ = crate::engine::cuda_runtime::free_delta_state_snapshot(snapshot);
        }
        if let Some(snapshot) = self.resident_delta_snapshot.take() {
            let _ = crate::engine::cuda_runtime::free_delta_state_snapshot(snapshot);
        }
    }
}

#[derive(Debug)]
pub(crate) struct VerifyWindowSsmDeltaInput {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) num_heads: usize,
    pub(crate) head_k_dim: usize,
    pub(crate) head_v_dim: usize,
}

pub(in crate::engine) struct GdnPrefixStateCollector {
    prefix_tokens: Vec<usize>,
    layers: Vec<Vec<VerifyWindowSsmLayerPrefixState>>,
    incomplete_layer: Vec<Option<usize>>,
}

impl GdnPrefixStateCollector {
    pub(in crate::engine) fn new_many(prefix_tokens: impl IntoIterator<Item = usize>) -> Self {
        let mut prefix_tokens = prefix_tokens.into_iter().collect::<Vec<_>>();
        prefix_tokens.sort_unstable();
        prefix_tokens.dedup();
        let layers = (0..prefix_tokens.len()).map(|_| Vec::new()).collect();
        let incomplete_layer = vec![None; prefix_tokens.len()];
        Self {
            prefix_tokens,
            layers,
            incomplete_layer,
        }
    }

    pub(in crate::engine) fn snapshot_prefix_tokens(&self, seq_len: usize) -> Vec<usize> {
        self.prefix_tokens
            .iter()
            .copied()
            .filter(|&prefix_tokens| prefix_tokens > 0 && prefix_tokens < seq_len)
            .collect()
    }

    pub(in crate::engine) fn wants_snapshot_for(&self, seq_len: usize) -> bool {
        self.prefix_tokens
            .iter()
            .any(|&prefix_tokens| prefix_tokens > 0 && prefix_tokens < seq_len)
    }

    pub(in crate::engine) fn mark_incomplete(&mut self, layer_idx: usize) {
        for incomplete_layer in &mut self.incomplete_layer {
            incomplete_layer.get_or_insert(layer_idx);
        }
    }

    pub(in crate::engine) fn mark_incomplete_for_prefix(
        &mut self,
        prefix_tokens: usize,
        layer_idx: usize,
    ) {
        if let Some(idx) = self.index_for_prefix(prefix_tokens) {
            self.incomplete_layer[idx].get_or_insert(layer_idx);
        }
    }

    pub(in crate::engine) fn record_layer_for_prefix(
        &mut self,
        prefix_tokens: usize,
        layer_idx: usize,
        conv_state: Vec<f32>,
        delta_input: VerifyWindowSsmDeltaInput,
    ) {
        if let Some(idx) = self.index_for_prefix(prefix_tokens) {
            self.layers[idx].push(VerifyWindowSsmLayerPrefixState {
                layer_idx,
                conv_state,
                #[cfg(feature = "cuda")]
                resident_conv_snapshot: None,
                delta_input: Some(delta_input),
                delta_state: None,
                #[cfg(feature = "cuda")]
                resident_delta_snapshot: None,
            });
        }
    }

    pub(in crate::engine) fn record_layer_with_delta_state_for_prefix(
        &mut self,
        prefix_tokens: usize,
        layer_idx: usize,
        conv_state: Vec<f32>,
        delta_state: Vec<f32>,
    ) {
        if let Some(idx) = self.index_for_prefix(prefix_tokens) {
            self.layers[idx].push(VerifyWindowSsmLayerPrefixState {
                layer_idx,
                conv_state,
                #[cfg(feature = "cuda")]
                resident_conv_snapshot: None,
                delta_input: None,
                delta_state: Some(delta_state),
                #[cfg(feature = "cuda")]
                resident_delta_snapshot: None,
            });
        }
    }

    #[cfg(feature = "cuda")]
    pub(in crate::engine) fn record_layer_with_resident_delta_snapshot_for_prefix(
        &mut self,
        prefix_tokens: usize,
        layer_idx: usize,
        conv_state: Vec<f32>,
        resident_delta_snapshot: crate::engine::cuda_runtime::DeltaStateSnapshot,
    ) {
        if let Some(idx) = self.index_for_prefix(prefix_tokens) {
            self.layers[idx].push(VerifyWindowSsmLayerPrefixState {
                layer_idx,
                conv_state,
                resident_conv_snapshot: None,
                delta_input: None,
                delta_state: None,
                resident_delta_snapshot: Some(resident_delta_snapshot),
            });
        }
    }

    /// pm116: attention-only 모델용 — recurrent layer 가 없어 layers 가 비는 것이
    /// 정상. prefix_tokens 별 빈 state 를 돌려줘 KV truncate 전용 restore 에 쓴다.
    pub(in crate::engine) fn finish_many_allow_empty(self) -> Vec<VerifyWindowPrefixState> {
        self.prefix_tokens
            .into_iter()
            .zip(self.layers.into_iter())
            .map(|(prefix_tokens, layers)| VerifyWindowPrefixState {
                prefix_tokens,
                layers,
            })
            .collect()
    }

    pub(in crate::engine) fn finish_many_required(
        self,
    ) -> crate::error::Result<Vec<VerifyWindowPrefixState>> {
        let mut states = Vec::with_capacity(self.prefix_tokens.len());
        for ((prefix_tokens, layers), incomplete_layer) in self
            .prefix_tokens
            .into_iter()
            .zip(self.layers.into_iter())
            .zip(self.incomplete_layer.into_iter())
        {
            if let Some(layer_idx) = incomplete_layer {
                return Err(crate::error::LlmError::Forward(format!(
                    "GDN prefix state snapshot was unavailable for layer {layer_idx}"
                )));
            }
            if layers.is_empty() {
                return Err(crate::error::LlmError::Forward(
                    "GDN prefix state snapshot captured no layers".to_string(),
                ));
            }
            states.push(VerifyWindowPrefixState {
                prefix_tokens,
                layers,
            });
        }
        Ok(states)
    }

    fn index_for_prefix(&self, prefix_tokens: usize) -> Option<usize> {
        self.prefix_tokens
            .iter()
            .position(|&known| known == prefix_tokens)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum PrefixDeltaRestoreKind {
    OneStepDeltaInput,
    ResidentSnapshot,
    Unsupported,
}

pub(in crate::engine) fn prefix_delta_restore_kind(
    prefix_tokens: usize,
    resident_snapshot_available: bool,
) -> PrefixDeltaRestoreKind {
    if resident_snapshot_available {
        PrefixDeltaRestoreKind::ResidentSnapshot
    } else if prefix_tokens == 1 {
        PrefixDeltaRestoreKind::OneStepDeltaInput
    } else {
        PrefixDeltaRestoreKind::Unsupported
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_window_result_records_one_token_per_input() {
        let result = VerifyWindowResult {
            target_tokens: vec![10, 20],
            mtp_hidden_rows: vec![0.0; 4],
            hidden_dim: 2,
            prefix_state: None,
            prefix_states: Vec::new(),
            ssm_final_states: Vec::new(),
            attention_kv_states: Vec::new(),
        };

        assert_eq!(result.len(), 2);
        assert_eq!(result.hidden_rows(), 2);
    }

    #[test]
    fn shadow_prefix_tokens_include_full_verify_window() {
        let include = MtpVerifyWindowRequest::new(10, &[11, 12, 13], MtpVerifyBonus::Include);
        assert_eq!(include.prefix_tokens(), vec![1, 2, 3]);
        assert_eq!(include.shadow_commit_prefix_tokens(), vec![1, 2, 3, 4]);

        let omit = MtpVerifyWindowRequest::new(10, &[11, 12, 13], MtpVerifyBonus::Omit);
        assert_eq!(omit.prefix_tokens(), vec![1, 2]);
        assert_eq!(omit.shadow_commit_prefix_tokens(), vec![1, 2, 3]);
    }

    #[test]
    fn multi_token_prefix_restore_requires_resident_delta_snapshot() {
        assert_eq!(
            prefix_delta_restore_kind(1, false),
            PrefixDeltaRestoreKind::OneStepDeltaInput
        );
        assert_eq!(
            prefix_delta_restore_kind(2, false),
            PrefixDeltaRestoreKind::Unsupported
        );
        assert_eq!(
            prefix_delta_restore_kind(2, true),
            PrefixDeltaRestoreKind::ResidentSnapshot
        );
    }

    #[test]
    fn verify_window_result_slices_committed_mtp_hidden_rows() {
        let result = VerifyWindowResult {
            target_tokens: vec![10, 20, 30],
            mtp_hidden_rows: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            hidden_dim: 2,
            prefix_state: None,
            prefix_states: Vec::new(),
            ssm_final_states: Vec::new(),
            attention_kv_states: Vec::new(),
        };

        assert_eq!(
            result.mtp_hidden_prefix_rows(2).unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn verify_window_result_finds_multi_prefix_state() {
        let result = VerifyWindowResult {
            target_tokens: vec![10, 20, 30],
            mtp_hidden_rows: Vec::new(),
            hidden_dim: 2,
            prefix_state: None,
            prefix_states: vec![
                VerifyWindowPrefixState {
                    prefix_tokens: 1,
                    layers: Vec::new(),
                },
                VerifyWindowPrefixState {
                    prefix_tokens: 2,
                    layers: Vec::new(),
                },
            ],
            ssm_final_states: Vec::new(),
            attention_kv_states: Vec::new(),
        };

        assert_eq!(result.prefix_state_for(2).unwrap().prefix_tokens, 2);
        assert!(result.prefix_state_for(3).is_none());
    }

    #[test]
    fn mtp_verify_window_request_builds_full_bonus_input() {
        let request = MtpVerifyWindowRequest::new(10, &[11, 12], MtpVerifyBonus::Include);

        assert_eq!(request.verify_tokens(), vec![10, 11, 12]);
        assert_eq!(request.prefix_tokens(), vec![1, 2]);
        assert_eq!(request.full_accept_committed_tokens(), 3);
    }

    #[test]
    fn mtp_verify_window_request_omits_terminal_bonus_input() {
        let request = MtpVerifyWindowRequest::new(10, &[11, 12], MtpVerifyBonus::Omit);

        assert_eq!(request.verify_tokens(), vec![10, 11]);
        assert_eq!(request.prefix_tokens(), vec![1]);
        assert_eq!(request.full_accept_committed_tokens(), 3);
    }

    #[test]
    fn device_verify_result_requires_one_hidden_row_per_target_token() {
        let err = VerifyWindowResult::from_device_result(vec![10, 20], vec![1.0, 2.0], 2)
            .expect_err("hidden rows should be incomplete");

        assert!(err
            .to_string()
            .contains("MTP device verify hidden rows mismatch"));
    }

    #[test]
    fn device_verify_result_accepts_exact_hidden_rows() {
        let result =
            VerifyWindowResult::from_device_result(vec![10, 20], vec![1.0, 2.0, 3.0, 4.0], 2)
                .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result.hidden_rows(), 2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_verify_result_carries_prefix_states_for_reject_restore() {
        let result = VerifyWindowResult::from_device_result_with_state_payload(
            vec![10, 20],
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            vec![crate::engine::cuda_runtime::MtpDeviceVerifyPrefixState {
                prefix_tokens: 1,
                layers: vec![
                    crate::engine::cuda_runtime::MtpDeviceVerifySsmLayerPrefixState {
                        layer_idx: 3,
                        conv_state: vec![0.25, 0.5],
                        resident_conv_snapshot: None,
                        resident_delta_snapshot: None,
                    },
                ],
            }],
            vec![
                crate::engine::cuda_runtime::MtpDeviceVerifySsmLayerFinalState {
                    layer_idx: 3,
                    conv_state: vec![0.75, 1.0],
                    device_resident: false,
                },
            ],
            vec![
                crate::engine::cuda_runtime::MtpDeviceVerifyAttentionKvState {
                    layer_idx: 4,
                    window_tokens: 2,
                    kv_rows: 3,
                    k_bits: vec![1, 2, 3, 4, 5, 6],
                    v_bits: vec![7, 8, 9, 10, 11, 12],
                    device_resident: false,
                },
            ],
        )
        .unwrap();

        let prefix_state = result.prefix_state_for(1).expect("prefix state");
        assert_eq!(prefix_state.layers.len(), 1);
        assert_eq!(prefix_state.layers[0].layer_idx, 3);
        assert_eq!(prefix_state.layers[0].conv_state, vec![0.25, 0.5]);
        assert!(prefix_state.layers[0].resident_delta_snapshot.is_none());
        assert_eq!(result.ssm_final_states.len(), 1);
        assert_eq!(result.ssm_final_states[0].layer_idx, 3);
        assert_eq!(result.ssm_final_states[0].conv_state, vec![0.75, 1.0]);
        assert_eq!(result.attention_kv_states.len(), 1);
        assert_eq!(result.attention_kv_states[0].layer_idx, 4);
        assert_eq!(result.attention_kv_states[0].window_tokens, 2);
        assert_eq!(result.attention_kv_states[0].kv_rows, 3);
        assert_eq!(result.attention_kv_states[0].k_bits, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(
            result.attention_kv_states[0].v_bits,
            vec![7, 8, 9, 10, 11, 12]
        );
    }
}
