use crate::engine::layer_weights::{LayerType, ModelWeights};
use crate::engine::memory_runtime::moe_preheat;
use crate::engine::packed_runtime::PackedModel;
use crate::engine::types::ModelMetadata;
use std::path::Path;
use std::sync::Arc;

pub(in crate::engine) fn wire_shadow_model(weights: &mut ModelWeights, sm: &Arc<PackedModel>) {
    let mut n_shadow_wired = 0usize;
    for i in 0..weights.layers.len() {
        if let LayerType::Attention(ref mut w) = &mut weights.layers[i] {
            if let Some(ref mut mw) = w.moe {
                let gu = format!("blk.{i}.ffn_gate_up_exps.weight");
                if sm.get_weight(&gu).is_some() {
                    mw.shadow_model = Some(sm.clone());
                    mw.shadow_gate_up_rnb_name = Some(gu);
                    n_shadow_wired += 1;
                }
            }
        }
    }
    eprintln!(
        "[INFO] MoE mixed-precision shadow wiring: {} MoE layers linked to Q2_K gate_up",
        n_shadow_wired
    );
}

pub(in crate::engine) fn wire_packed_model_weights(
    _path: &Path,
    weights: &mut ModelWeights,
    metadata: &ModelMetadata,
    pm: &Arc<PackedModel>,
) {
    let permutation_all = moe_preheat::parse_permutation(&pm.metadata);
    let mut n_moe_layers_wired = 0usize;

    for i in 0..metadata.num_layers {
        if let LayerType::Attention(ref mut w) = &mut weights.layers[i] {
            if let Some(ref mut mw) = w.moe {
                let gu = format!("blk.{i}.ffn_gate_up_exps.weight");
                let dn = format!("blk.{i}.ffn_down_exps.weight");
                let rt = format!("blk.{i}.ffn_gate_inp.weight");
                let has_gu = pm.get_weight(&gu).is_some();
                let has_dn = pm.get_weight(&dn).is_some();
                let has_rt = pm.get_weight(&rt).is_some();
                // Partial sidecar coverage is OK: any tensor missing from the
                // sidecar falls back to the GGUF mmap through the
                // `gate_up_bytes()` / `down_bytes()` / `router_f32()` helpers
                // (those check `packed_model + *_rnb_name` first, then drop
                // back to `self.gate_up_exps.as_bytes()` etc). Gemma4 sidecars
                // typically only carry the fused `gate_up_exps`; `down_exps`
                // (Q5_1) and the F32 router stay GGUF-side.
                if has_gu {
                    mw.packed_model = Some(pm.clone());
                    mw.gate_up_rnb_name = Some(gu.clone());
                    if has_dn {
                        mw.down_rnb_name = Some(dn.clone());
                    }
                    if has_rt {
                        mw.router_rnb_name = Some(rt);
                    }
                    if let Some(ref perms) = permutation_all {
                        if let Some(layer_perm) = perms.get(i) {
                            if layer_perm.len() == mw.n_expert {
                                mw.rank_to_original = Some(layer_perm.clone());
                            } else {
                                eprintln!(
                                    "[WARN] layer {i}: moe_expert_permutation length {} != n_expert {}; leaving rank_to_original=None",
                                    layer_perm.len(),
                                    mw.n_expert
                                );
                            }
                        }
                    }
                    // in4: residency view is wired whenever the sidecar
                    // contains MoE expert tiers, regardless of whether
                    // `metadata_v3` carries a popularity blob. Without the
                    // blob `OwnedMoeResidency` falls back to identity order
                    // + all-hot, which is still correct.
                    mw.gate_up_residency = pm.moe_residency_view(&gu);
                    if has_dn {
                        mw.down_residency = pm.moe_residency_view(&dn);
                    }
                    n_moe_layers_wired += 1;
                }
            }
        }
    }
    eprintln!(
        "[INFO] gemma4 MoE wired: {} layers from .rnb",
        n_moe_layers_wired
    );

    let _ = (metadata, pm);
}
