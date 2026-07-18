//! MoE memory-tier policy switches.
//!
//! This module centralizes user/runtime policy knobs shared by the MoE
//! memory helpers. It owns env parsing and pure policy decisions such as
//! cold-file selection, preheat enablement, unified cold activation, and the
//! effective hot-expert count. Data movement stays in `moe_hot_pool`,
//! `moe_preheat`, and `moe_cold_io`.

use std::path::{Path, PathBuf};

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

pub fn cold_rnb_path_for(hot_rnb_path: &Path) -> PathBuf {
    std::env::var("RNB_MOE_COLD_RNB")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_cold_rnb_path(hot_rnb_path))
}

pub fn default_cold_rnb_path(hot_rnb_path: &Path) -> PathBuf {
    let stem = hot_rnb_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    hot_rnb_path.with_file_name(format!("{stem}.cold.rnb"))
}

pub fn unified_cold_enabled() -> bool {
    env_flag("RNB_MOE_UNIFIED_COLD")
}

pub fn unified_cold_active(cold_reader_attached: bool) -> bool {
    unified_cold_enabled() && cold_reader_attached
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertResidencySource {
    UnifiedColdPread,
    RuntimeHotPool,
    HotMmap,
    ColdPread,
    ColdMmap,
}

pub fn expert_residency_source(
    expert_rank: usize,
    hot_count: usize,
    runtime_hot_count: usize,
    cold_reader_attached: bool,
    unified_cold_active: bool,
) -> ExpertResidencySource {
    if unified_cold_active && cold_reader_attached {
        ExpertResidencySource::UnifiedColdPread
    } else if hot_count == 0 {
        ExpertResidencySource::HotMmap
    } else if expert_rank < runtime_hot_count {
        ExpertResidencySource::RuntimeHotPool
    } else if expert_rank < hot_count {
        ExpertResidencySource::HotMmap
    } else if cold_reader_attached {
        ExpertResidencySource::ColdPread
    } else {
        ExpertResidencySource::ColdMmap
    }
}

pub fn preheat_enabled() -> bool {
    env_flag("RNB_MOE_PREHEAT")
}

pub fn hot_count_override() -> Option<usize> {
    env_usize("RNB_MOE_HOT_COUNT")
}

/// Explains which input selected the effective hot-expert count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveHotCountSource {
    /// `.rnb` metadata supplied the count and there was no env override.
    Metadata,
    /// `RNB_MOE_HOT_COUNT` supplied the count without needing a metadata cap.
    Override,
    /// `RNB_MOE_HOT_COUNT` exceeded metadata and was capped to metadata.
    OverrideCappedToMetadata,
}

/// Effective hot-expert count plus the policy source that selected it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveHotCount {
    pub count: usize,
    pub source: EffectiveHotCountSource,
}

pub fn effective_hot_count(metadata_hot: Option<usize>) -> Option<usize> {
    effective_hot_count_with_override(metadata_hot, hot_count_override())
}

pub fn effective_hot_count_with_override(
    metadata_hot: Option<usize>,
    override_hot: Option<usize>,
) -> Option<usize> {
    effective_hot_count_decision_with_override(metadata_hot, override_hot)
        .map(|decision| decision.count)
}

pub fn effective_hot_count_decision(metadata_hot: Option<usize>) -> Option<EffectiveHotCount> {
    effective_hot_count_decision_with_override(metadata_hot, hot_count_override())
}

pub fn effective_hot_count_decision_with_override(
    metadata_hot: Option<usize>,
    override_hot: Option<usize>,
) -> Option<EffectiveHotCount> {
    match (metadata_hot, override_hot) {
        (Some(meta), Some(override_hot)) if override_hot > meta => Some(EffectiveHotCount {
            count: meta,
            source: EffectiveHotCountSource::OverrideCappedToMetadata,
        }),
        (Some(_), Some(override_hot)) => Some(EffectiveHotCount {
            count: override_hot,
            source: EffectiveHotCountSource::Override,
        }),
        (Some(meta), None) => Some(EffectiveHotCount {
            count: meta,
            source: EffectiveHotCountSource::Metadata,
        }),
        (None, Some(override_hot)) => Some(EffectiveHotCount {
            count: override_hot,
            source: EffectiveHotCountSource::Override,
        }),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_hot_count_caps_override_to_metadata() {
        assert_eq!(
            effective_hot_count_with_override(Some(64), Some(96)),
            Some(64)
        );
        assert_eq!(
            effective_hot_count_with_override(Some(64), Some(32)),
            Some(32)
        );
        assert_eq!(effective_hot_count_with_override(Some(64), None), Some(64));
        assert_eq!(effective_hot_count_with_override(None, Some(32)), Some(32));
        assert_eq!(effective_hot_count_with_override(None, None), None);
    }

    #[test]
    fn effective_hot_count_decision_reports_source() {
        assert_eq!(
            effective_hot_count_decision_with_override(Some(64), Some(96)),
            Some(EffectiveHotCount {
                count: 64,
                source: EffectiveHotCountSource::OverrideCappedToMetadata,
            })
        );
        assert_eq!(
            effective_hot_count_decision_with_override(Some(64), Some(32)),
            Some(EffectiveHotCount {
                count: 32,
                source: EffectiveHotCountSource::Override,
            })
        );
        assert_eq!(
            effective_hot_count_decision_with_override(Some(64), None),
            Some(EffectiveHotCount {
                count: 64,
                source: EffectiveHotCountSource::Metadata,
            })
        );
        assert_eq!(effective_hot_count_decision_with_override(None, None), None);
    }

    #[test]
    fn unified_cold_active_requires_attached_reader() {
        let enabled = unified_cold_enabled();
        assert_eq!(unified_cold_active(false), false);
        assert_eq!(unified_cold_active(true), enabled);
    }

    #[test]
    fn expert_residency_source_prefers_unified_cold_when_active() {
        assert_eq!(
            expert_residency_source(0, 8, 4, true, true),
            ExpertResidencySource::UnifiedColdPread
        );
        assert_eq!(
            expert_residency_source(7, 8, 4, true, true),
            ExpertResidencySource::UnifiedColdPread
        );
    }

    #[test]
    fn expert_residency_source_uses_flat_hot_when_model_has_no_split() {
        assert_eq!(
            expert_residency_source(17, 0, 0, true, false),
            ExpertResidencySource::HotMmap
        );
    }

    #[test]
    fn expert_residency_source_separates_runtime_hot_hot_and_cold_tiers() {
        assert_eq!(
            expert_residency_source(1, 8, 3, true, false),
            ExpertResidencySource::RuntimeHotPool
        );
        assert_eq!(
            expert_residency_source(5, 8, 3, true, false),
            ExpertResidencySource::HotMmap
        );
        assert_eq!(
            expert_residency_source(9, 8, 3, true, false),
            ExpertResidencySource::ColdPread
        );
        assert_eq!(
            expert_residency_source(9, 8, 3, false, false),
            ExpertResidencySource::ColdMmap
        );
    }
}
