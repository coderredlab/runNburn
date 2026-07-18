use super::layer_weights::{LayerType, ModelWeights};
use super::models::{gemma, shared_expert_moe};
use super::packed_runtime::{self, PackedModel};
use super::policy;
use super::quantized_weight::{gdn_gate_weight_name, gdn_qkv_weight_name};
use super::quantized_weight_types::QuantizedWeight;
use super::types::ModelMetadata;
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};
use std::path::Path;
use std::sync::Arc;

pub(super) fn open_diagnostic_packed_model(sidecar: Option<&Path>) -> Option<Arc<PackedModel>> {
    sidecar.and_then(packed_runtime::open_sidecar_v3_packed_model)
}

pub(super) fn open_shadow_model(path: &Path) -> Option<Arc<PackedModel>> {
    packed_runtime::open_shadow_model(path)
}

fn packed_name_allowed(name: &str) -> bool {
    let only_terms = policy::packed_only_match_terms();
    if !only_terms.is_empty() && !only_terms.iter().any(|term| name.contains(term)) {
        return false;
    }
    let skip_terms = policy::packed_skip_match_terms();
    if skip_terms.iter().any(|term| name.contains(term)) {
        return false;
    }
    true
}

fn packed_row_pair_payload_len(
    quant_type: packed_runtime::QuantType,
    rows: usize,
    cols: usize,
) -> Result<Option<usize>, String> {
    let bytes_per_row_group = match quant_type {
        packed_runtime::QuantType::Q4K => 2240usize,
        packed_runtime::QuantType::Q6K => 2208usize,
        _ => return Ok(None),
    };
    if cols % 256 != 0 {
        return Err(format!("cols {cols} is not divisible by 256"));
    }
    rows.div_ceil(8)
        .checked_mul(cols / 256)
        .and_then(|blocks| blocks.checked_mul(bytes_per_row_group))
        .map(Some)
        .ok_or_else(|| format!("payload length overflow for shape {rows}x{cols}"))
}

fn packed_weight_compatible(
    w: &QuantizedWeight,
    name: &str,
    pw: &packed_runtime::PackedWeight,
) -> bool {
    if pw.rows != w.rows || pw.cols != w.cols {
        eprintln!(
            "[WARN] Packed GEMM skipped `{name}`: sidecar shape {}x{} does not match GGUF shape {}x{}",
            pw.rows, pw.cols, w.rows, w.cols
        );
        return false;
    }
    let expected = match packed_row_pair_payload_len(pw.quant_type, pw.rows, pw.cols) {
        Ok(expected) => expected,
        Err(err) => {
            eprintln!("[WARN] Packed GEMM skipped `{name}`: sidecar {err}");
            return false;
        }
    };
    if let Some(expected) = expected {
        let actual = pw.data().len();
        if actual != expected {
            eprintln!(
                "[WARN] Packed GEMM skipped `{name}`: sidecar payload is {actual} bytes, expected {expected}"
            );
            return false;
        }
    }
    true
}

fn wire_packed(w: &mut QuantizedWeight, name: &str, pm: &PackedModel) {
    if !packed_name_allowed(name) {
        return;
    }
    let skip = match w.ggml_type {
        GGMLType::Q4_K => policy::packed_skip_q4k_enabled(),
        GGMLType::Q5_K => policy::packed_skip_q5k_enabled(),
        GGMLType::Q6_K => policy::packed_skip_q6k_enabled(),
        _ => false,
    };
    if skip {
        return;
    }
    if let Some(pw) = pm.get_weight(name) {
        if !packed_weight_compatible(w, name, pw) {
            return;
        }
        let data = pw.data();
        w.attach_packed_gemm(pw.quant_type, data);
    }
}

pub(super) fn wire_packed_dense_weights(
    weights: &mut ModelWeights,
    metadata: &ModelMetadata,
    pm: &PackedModel,
) {
    wire_packed(&mut weights.output, "output.weight", pm);

    for i in 0..metadata.num_layers {
        match &mut weights.layers[i] {
            LayerType::Attention(w) => {
                wire_packed(&mut w.q_weight, &format!("blk.{i}.attn_q.weight"), pm);
                wire_packed(&mut w.k_weight, &format!("blk.{i}.attn_k.weight"), pm);
                wire_packed(&mut w.v_weight, &format!("blk.{i}.attn_v.weight"), pm);
                wire_packed(&mut w.o_weight, &format!("blk.{i}.attn_output.weight"), pm);
                wire_packed(
                    &mut w.ffn_gate_weight,
                    &format!("blk.{i}.ffn_gate.weight"),
                    pm,
                );
                wire_packed(&mut w.ffn_up_weight, &format!("blk.{i}.ffn_up.weight"), pm);
                wire_packed(
                    &mut w.ffn_down_weight,
                    &format!("blk.{i}.ffn_down.weight"),
                    pm,
                );
            }
            LayerType::GatedDeltaNet(w) => {
                wire_packed(&mut w.qkv_weight, &gdn_qkv_weight_name(i), pm);
                wire_packed(&mut w.gate_weight, &gdn_gate_weight_name(i), pm);
                wire_packed(&mut w.ssm_alpha, &format!("blk.{i}.ssm_a.weight"), pm);
                wire_packed(&mut w.ssm_beta, &format!("blk.{i}.ssm_b.weight"), pm);
                wire_packed(&mut w.ssm_out, &format!("blk.{i}.ssm_out.weight"), pm);
                wire_packed(
                    &mut w.ffn_gate_weight,
                    &format!("blk.{i}.ffn_gate.weight"),
                    pm,
                );
                wire_packed(&mut w.ffn_up_weight, &format!("blk.{i}.ffn_up.weight"), pm);
                wire_packed(
                    &mut w.ffn_down_weight,
                    &format!("blk.{i}.ffn_down.weight"),
                    pm,
                );
            }
            LayerType::NemotronMamba2(w) => {
                wire_packed(&mut w.ssm_in, &format!("blk.{i}.ssm_in.weight"), pm);
                wire_packed(&mut w.ssm_out, &format!("blk.{i}.ssm_out.weight"), pm);
            }
            LayerType::NemotronMoE(w) => {
                wire_packed(&mut w.router, &format!("blk.{i}.ffn_gate_inp.weight"), pm);
                wire_packed(
                    &mut w.expert_down,
                    &format!("blk.{i}.ffn_down_exps.weight"),
                    pm,
                );
                wire_packed(&mut w.expert_up, &format!("blk.{i}.ffn_up_exps.weight"), pm);
                wire_packed(
                    &mut w.shared_expert_down,
                    &format!("blk.{i}.ffn_down_shexp.weight"),
                    pm,
                );
                wire_packed(
                    &mut w.shared_expert_up,
                    &format!("blk.{i}.ffn_up_shexp.weight"),
                    pm,
                );
                if let Some(latent_down) = &mut w.latent_down {
                    wire_packed(latent_down, &format!("blk.{i}.ffn_down.weight"), pm);
                }
                if let Some(latent_up) = &mut w.latent_up {
                    wire_packed(latent_up, &format!("blk.{i}.ffn_up.weight"), pm);
                }
            }
        }
    }
}

pub(super) fn wire_shadow_model(
    weights: &mut ModelWeights,
    architecture: ModelArchitecture,
    sm: &Arc<PackedModel>,
) {
    match architecture {
        ModelArchitecture::Qwen35MoE => {
            shared_expert_moe::packed_wiring::wire_shadow_model(weights, sm)
        }
        _ => gemma::packed_wiring::wire_shadow_model(weights, sm),
    }
}

pub(super) fn count_packed_gemm_wired(weights: &ModelWeights) -> usize {
    weights
        .layers
        .iter()
        .flat_map(|l| match l {
            LayerType::Attention(w) => vec![
                &w.q_weight,
                &w.k_weight,
                &w.v_weight,
                &w.o_weight,
                &w.ffn_gate_weight,
                &w.ffn_up_weight,
                &w.ffn_down_weight,
            ],
            LayerType::GatedDeltaNet(w) => vec![
                &w.qkv_weight,
                &w.gate_weight,
                &w.ssm_alpha,
                &w.ssm_beta,
                &w.ssm_out,
                &w.ffn_gate_weight,
                &w.ffn_up_weight,
                &w.ffn_down_weight,
            ],
            LayerType::NemotronMamba2(w) => vec![&w.ssm_in, &w.ssm_out],
            LayerType::NemotronMoE(w) => {
                let mut weights = vec![
                    &w.router,
                    &w.expert_down,
                    &w.expert_up,
                    &w.shared_expert_down,
                    &w.shared_expert_up,
                ];
                if let Some(latent_down) = &w.latent_down {
                    weights.push(latent_down);
                }
                if let Some(latent_up) = &w.latent_up {
                    weights.push(latent_up);
                }
                weights
            }
        })
        .filter(|w| w.packed_gemm_data.is_some())
        .count()
        + usize::from(weights.output.packed_gemm_data.is_some())
}

pub(super) fn wire_packed_model_weights(
    path: &Path,
    weights: &mut ModelWeights,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    pm: &Arc<PackedModel>,
) {
    wire_packed_dense_weights(weights, metadata, pm);

    match architecture {
        ModelArchitecture::Qwen35MoE => {
            shared_expert_moe::packed_wiring::wire_packed_model_weights(weights, metadata, pm)
        }
        _ => gemma::packed_wiring::wire_packed_model_weights(path, weights, metadata, pm),
    }

    let n_wired = count_packed_gemm_wired(weights);
    eprintln!(
        "[INFO] Packed GEMM wired: {}/{} weights",
        n_wired,
        pm.weights.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_core::tensor::{QuantType, Tensor};
    use rnb_loader::rnb_file::write_rnb;

    fn temp_rnb(
        name: &str,
        rows: usize,
        cols: usize,
        quant: QuantType,
        data_len: usize,
    ) -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().unwrap();
        let tensors = vec![(name.to_string(), quant, rows, cols, vec![0x5au8; data_len])];
        write_rnb(file.path(), &tensors, b"").unwrap();
        file
    }

    fn q4_weight(rows: usize, cols: usize) -> QuantizedWeight {
        let blocks_per_row = cols / 256;
        let raw = vec![0u8; rows * blocks_per_row * 144];
        QuantizedWeight::new(
            Tensor::from_vec(raw, &[rows * blocks_per_row * 144]),
            GGMLType::Q4_K,
            rows,
            cols,
        )
    }

    fn q6_weight(rows: usize, cols: usize) -> QuantizedWeight {
        let blocks_per_row = cols / 256;
        let raw = vec![0u8; rows * blocks_per_row * 210];
        QuantizedWeight::new(
            Tensor::from_vec(raw, &[rows * blocks_per_row * 210]),
            GGMLType::Q6_K,
            rows,
            cols,
        )
    }

    #[test]
    fn wire_packed_rejects_sidecar_shape_mismatch_even_when_payload_group_len_matches() {
        let file = temp_rnb("w", 1, 256, QuantType::Q4K, 2240);
        let packed = PackedModel::open(file.path()).unwrap();
        let mut weight = q4_weight(8, 256);

        wire_packed(&mut weight, "w", &packed);

        assert!(weight.packed_gemm_data.is_none());
        assert!(weight.packed_gemm_quant_type.is_none());
    }

    #[test]
    fn wire_packed_rejects_sidecar_payload_length_mismatch() {
        let file = temp_rnb("w", 8, 256, QuantType::Q4K, 144);
        let packed = PackedModel::open(file.path()).unwrap();
        let mut weight = q4_weight(8, 256);

        wire_packed(&mut weight, "w", &packed);

        assert!(weight.packed_gemm_data.is_none());
        assert!(weight.packed_gemm_quant_type.is_none());
    }

    #[test]
    fn wire_packed_accepts_matching_sidecar_shape_and_payload_length() {
        let file = temp_rnb("w", 8, 256, QuantType::Q4K, 2240);
        let packed = PackedModel::open(file.path()).unwrap();
        let mut weight = q4_weight(8, 256);

        wire_packed(&mut weight, "w", &packed);

        assert!(weight.packed_gemm_data.is_some());
        assert_eq!(weight.packed_gemm_quant_type, Some(QuantType::Q4K));
    }

    #[test]
    fn wire_packed_rejects_q6_sidecar_payload_length_mismatch() {
        let file = temp_rnb("w", 8, 256, QuantType::Q6K, 210);
        let packed = PackedModel::open(file.path()).unwrap();
        let mut weight = q6_weight(8, 256);

        wire_packed(&mut weight, "w", &packed);

        assert!(weight.packed_gemm_data.is_none());
        assert!(weight.packed_gemm_quant_type.is_none());
    }
}
