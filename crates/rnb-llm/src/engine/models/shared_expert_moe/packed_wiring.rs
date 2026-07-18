use crate::engine::layer_weights::{LayerType, ModelWeights, SharedExpertMoELayerWeights};
use crate::engine::memory_runtime::moe_preheat;
use crate::engine::packed_runtime::PackedModel;
use crate::engine::types::ModelMetadata;
use std::sync::Arc;

pub(in crate::engine) fn wire_shadow_model(weights: &mut ModelWeights, sm: &Arc<PackedModel>) {
    let mut n_shadow_wired = 0usize;
    for i in 0..weights.layers.len() {
        let moe_opt: Option<&mut SharedExpertMoELayerWeights> = match &mut weights.layers[i] {
            LayerType::Attention(w) => w.shared_expert_moe.as_mut(),
            LayerType::GatedDeltaNet(w) => w.shared_expert_moe.as_mut(),
            LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => None,
        };
        let Some(mw) = moe_opt else {
            continue;
        };
        let gu_tile = format!("blk.{i}.ffn_gate_up_tile.weight");
        let g = format!("blk.{i}.ffn_gate_exps.weight");
        let u = format!("blk.{i}.ffn_up_exps.weight");
        let d = format!("blk.{i}.ffn_down_exps.weight");
        let has_gu_tile = sm.get_weight(&gu_tile).is_some();
        let has_g = sm.get_weight(&g).is_some();
        let has_u = sm.get_weight(&u).is_some();
        if has_gu_tile || (has_g && has_u) {
            mw.shadow_model = Some(sm.clone());
            if has_gu_tile {
                mw.shadow_gate_up_tile_rnb_name = Some(gu_tile);
            }
            if has_g && has_u {
                mw.shadow_gate_rnb_name = Some(g);
                mw.shadow_up_rnb_name = Some(u);
            }
            // down shadow 는 optional (n_ff 256 배수일 때만 shadow 에 존재)
            if sm.get_weight(&d).is_some() {
                mw.shadow_down_rnb_name = Some(d);
            }
            n_shadow_wired += 1;
        }
    }
    eprintln!(
        "[INFO] MoE mixed-precision shadow wiring (qwen35moe): {} MoE layers linked (gate+up, down optional)",
        n_shadow_wired
    );
}

pub(in crate::engine) fn wire_packed_model_weights(
    weights: &mut ModelWeights,
    metadata: &ModelMetadata,
    pm: &Arc<PackedModel>,
) {
    let permutation_all = moe_preheat::parse_permutation(&pm.metadata);
    let mut n_wired = 0usize;
    for i in 0..metadata.num_layers {
        let moe_opt: Option<&mut SharedExpertMoELayerWeights> = match &mut weights.layers[i] {
            LayerType::Attention(w) => w.shared_expert_moe.as_mut(),
            LayerType::GatedDeltaNet(w) => w.shared_expert_moe.as_mut(),
            LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => None,
        };
        let Some(mw) = moe_opt else {
            continue;
        };

        let g = format!("blk.{i}.ffn_gate_exps.weight");
        let u = format!("blk.{i}.ffn_up_exps.weight");
        let d = format!("blk.{i}.ffn_down_exps.weight");
        let r = format!("blk.{i}.ffn_gate_inp.weight");
        let has_core = pm.get_weight(&g).is_some()
            && pm.get_weight(&u).is_some()
            && pm.get_weight(&d).is_some()
            && pm.get_weight(&r).is_some();
        if has_core {
            mw.packed_model = Some(pm.clone());
            mw.gate_exps_rnb_name = Some(g.clone());
            mw.up_exps_rnb_name = Some(u.clone());
            mw.down_exps_rnb_name = Some(d.clone());
            mw.router_rnb_name = Some(r);
            // in4: when the .rnb is a v3 sidecar, also wire residency views
            // so sparse fanout can dispatch hot/cold ranks through the trait.
            if pm.metadata_v3.is_some() {
                mw.gate_residency = pm.moe_residency_view(&g);
                mw.up_residency = pm.moe_residency_view(&u);
                mw.down_residency = pm.moe_residency_view(&d);
            }

            for (field, suffix) in [
                (&mut mw.shared_gate_rnb_name, "ffn_gate_shexp.weight"),
                (&mut mw.shared_up_rnb_name, "ffn_up_shexp.weight"),
                (&mut mw.shared_down_rnb_name, "ffn_down_shexp.weight"),
                (&mut mw.shared_scale_rnb_name, "ffn_gate_inp_shexp.weight"),
            ] {
                let n = format!("blk.{i}.{suffix}");
                if pm.get_weight(&n).is_some() {
                    *field = Some(n);
                }
            }

            if let Some(ref perms) = permutation_all {
                if let Some(lp) = perms.get(i) {
                    if lp.len() == mw.n_expert {
                        mw.rank_to_original = Some(lp.iter().map(|&x| x as usize).collect());
                    } else {
                        eprintln!(
                            "[WARN] qwen35moe layer {i}: perm length {} != n_expert {}",
                            lp.len(),
                            mw.n_expert
                        );
                    }
                }
            }
            n_wired += 1;
        }
    }
    eprintln!("[INFO] qwen35moe MoE wired: {} layers from .rnb", n_wired);
}
