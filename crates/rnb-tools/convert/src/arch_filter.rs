//! Arch-based decision: should this GGUF be packed into a v3 sidecar at all?
//!
//! Architectures whose dense FFN packing yields a measured win go to
//! [`ConvertDecision::Convert`]. MoE-only architectures (no dense FFN to
//! repack) go to [`ConvertDecision::Skip`] and the engine reads the GGUF
//! directly. The split is per-spec §3.7 of
//! `docs/superpowers/specs/2026-05-02-rnb-sidecar-cache-design.md`.
//!
//! Phase 1 keeps the table small and explicit. Future arches need a single
//! arm added here; metadata-based fallback for genuinely unknown arches lives
//! in Task 9 (`metadata_fallback`).

use rnb_loader::arch::Architecture;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ConvertDecision {
    /// Pack into a v3 sidecar. Engine will mmap the cache for inference.
    Convert,
    /// Do not pack. Engine reads the GGUF directly.
    Skip,
}

/// Classify an architecture into a convert/skip decision. The
/// `_tensor_names` slice is reserved for the Task 9 metadata fallback path
/// and is currently unused for known variants.
pub fn classify(arch: Architecture, _tensor_names: &[&str]) -> ConvertDecision {
    if skip_reason(arch).is_some() {
        ConvertDecision::Skip
    } else {
        ConvertDecision::Convert
    }
}

/// Heuristic safety net for future arch additions: returns `true` if any of
/// the GGUF tensor names look like a dense FFN weight (`ffn_gate.weight`,
/// `ffn_up.weight`, or `ffn_down.weight`). MoE expert tensors
/// (`ffn_gate_exps.weight`, etc.) explicitly do not count — they are the MoE
/// path, not dense FFN.
///
/// `classify` does not consult this function for the variants enumerated in
/// `Architecture` — those are dispatched by enum match. The helper exists so
/// that, when a new arch variant is added, the convert path can fall back on
/// the GGUF tensor table instead of forcing a code change before the model
/// can be packed.
pub fn dense_ffn_present(tensor_names: &[&str]) -> bool {
    tensor_names.iter().any(|n| {
        // Match on stem (stripped of `blk.<n>.` prefix and trailing `.weight`).
        // Endings are stable per GGUF convention.
        n.ends_with(".ffn_gate.weight")
            || n.ends_with(".ffn_up.weight")
            || n.ends_with(".ffn_down.weight")
    })
}

/// Return a human-readable skip reason for architectures that a diagnostic
/// RNBC conversion caller should leave on GGUF direct.
pub fn skip_reason(arch: Architecture) -> Option<&'static str> {
    match arch {
        Architecture::Qwen35MoE => Some(
            "Qwen3-Next style MoE-only arch — packed sidecar showed +12% regression \
             (see PROJECT_JOURNAL session s73). Use GGUF direct.",
        ),
        Architecture::NemotronHMoE => Some(
            "Nemotron-H Mamba2+GDN+MoE hybrid. Sidecar packing not measured yet; \
             skip until validated.",
        ),
        Architecture::Hy3 => Some(
            "Hy3 dense+MoE hybrid. Sidecar packing and expert tiering are not validated yet; \
             use GGUF direct.",
        ),
        Architecture::GlmDsa => Some(
            "GLM DSA+MLA+MoE architecture. Sidecar packing is not validated yet; \
             use GGUF direct.",
        ),
        Architecture::Gemma4Assistant => Some(
            "Gemma4 assistant (drafter) — separate VQ/MTP pipeline that does not \
             share the target's dense GEMM hot path. Sidecar packing benefit \
             unmeasured; skip until mt82 evaluates it.",
        ),
        Architecture::LLaMA
        | Architecture::Gemma
        | Architecture::Gemma4
        | Architecture::Phi
        | Architecture::Qwen2
        | Architecture::Qwen35 => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_loader::arch::Architecture;

    #[test]
    fn dense_archs_convert() {
        assert_eq!(classify(Architecture::LLaMA, &[]), ConvertDecision::Convert);
        assert_eq!(classify(Architecture::Gemma, &[]), ConvertDecision::Convert);
        assert_eq!(
            classify(Architecture::Gemma4, &[]),
            ConvertDecision::Convert
        );
        assert_eq!(classify(Architecture::Phi, &[]), ConvertDecision::Convert);
        assert_eq!(classify(Architecture::Qwen2, &[]), ConvertDecision::Convert);
        assert_eq!(
            classify(Architecture::Qwen35, &[]),
            ConvertDecision::Convert
        );
    }

    #[test]
    fn moe_only_archs_skip() {
        assert_eq!(
            classify(Architecture::Qwen35MoE, &[]),
            ConvertDecision::Skip
        );
        assert_eq!(
            classify(Architecture::NemotronHMoE, &[]),
            ConvertDecision::Skip
        );
    }

    #[test]
    fn gemma4_assistant_drafter_skips() {
        // Drafter GGUF 는 별도 VQ/MTP 파이프라인이라 dense GEMM 핫패스를 공유하지 않음.
        // skip_reason 도 같이 노출돼야 convert subcommand 가 stderr 로 이유를 찍을 수 있음.
        assert_eq!(
            classify(Architecture::Gemma4Assistant, &[]),
            ConvertDecision::Skip
        );
        assert!(skip_reason(Architecture::Gemma4Assistant).is_some());
    }

    #[test]
    fn skip_decision_carries_reason() {
        if let ConvertDecision::Skip = classify(Architecture::Qwen35MoE, &[]) {
            // Skip variant exists; the reason is exposed via the impl below.
        } else {
            panic!("expected Skip");
        }
        assert!(skip_reason(Architecture::Qwen35MoE).is_some());
        assert!(skip_reason(Architecture::Gemma4).is_none());
    }
}

#[cfg(test)]
mod ffn_heuristic_tests {
    use super::*;

    #[test]
    fn dense_ffn_present_detects_gate_up_down() {
        let names = [
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        assert!(dense_ffn_present(&names));
    }

    #[test]
    fn dense_ffn_present_detects_partial_dense_ffn() {
        // gate alone is enough — gemma3 / phi style
        let names = ["blk.0.ffn_gate.weight", "blk.0.attn_q.weight"];
        assert!(dense_ffn_present(&names));
    }

    #[test]
    fn dense_ffn_absent_for_moe_only_tensors() {
        let names = [
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.attn_q.weight",
        ];
        assert!(!dense_ffn_present(&names));
    }

    #[test]
    fn dense_ffn_absent_for_attention_only() {
        let names = ["blk.0.attn_q.weight", "blk.0.attn_k.weight"];
        assert!(!dense_ffn_present(&names));
    }
}
