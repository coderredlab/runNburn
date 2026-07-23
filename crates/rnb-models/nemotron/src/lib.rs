#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronLayerKind {
    Mamba2,
    MoE,
    Attention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridPatternError {
    pub index: usize,
    pub byte: u8,
}

pub fn decode_hybrid_pattern(pattern: &str) -> Result<Vec<NemotronLayerKind>, HybridPatternError> {
    pattern
        .bytes()
        .enumerate()
        .map(|(index, byte)| match byte {
            b'M' => Ok(NemotronLayerKind::Mamba2),
            b'E' => Ok(NemotronLayerKind::MoE),
            b'*' | b'A' => Ok(NemotronLayerKind::Attention),
            byte => Err(HybridPatternError { index, byte }),
        })
        .collect()
}

pub fn classify_layer_from_tensor_names<'a>(
    layer_idx: usize,
    tensor_names: impl IntoIterator<Item = &'a str>,
) -> Result<Option<NemotronLayerKind>, String> {
    let prefix = format!("blk.{layer_idx}.");
    let mut kind = None;

    for name in tensor_names {
        let Some(suffix) = name.strip_prefix(&prefix) else {
            continue;
        };
        let candidate = if suffix.starts_with("ssm_") {
            Some(NemotronLayerKind::Mamba2)
        } else if suffix.starts_with("ffn_") || suffix == "exp_probs_b.bias" {
            Some(NemotronLayerKind::MoE)
        } else if suffix.starts_with("attn_") && suffix != "attn_norm.weight" {
            Some(NemotronLayerKind::Attention)
        } else {
            None
        };

        let Some(candidate) = candidate else {
            continue;
        };
        if let Some(existing) = kind {
            if existing != candidate {
                return Err(format!(
                    "conflicting Nemotron layer {layer_idx} tensor kinds: {existing:?} vs {candidate:?}"
                ));
            }
        } else {
            kind = Some(candidate);
        }
    }

    Ok(kind)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

pub fn sigmoid_topk_route(
    logits: &[f32],
    bias: Option<&[f32]>,
    expert_used_count: usize,
    weight_scale: f32,
) -> Route {
    if let Some(bias) = bias {
        assert_eq!(bias.len(), logits.len());
    }
    let selected_len = expert_used_count.min(logits.len());
    let mut scores: Vec<(usize, f32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| {
            let weight = sigmoid(logit);
            let selection_score = weight + bias.map(|b| b[idx]).unwrap_or(0.0);
            (idx, selection_score, weight)
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(selected_len);

    let sum: f32 = scores.iter().map(|(_, _, weight)| *weight).sum();
    let experts = scores.iter().map(|(idx, _, _)| *idx).collect();
    let weights = if sum > 0.0 {
        scores
            .iter()
            .map(|(_, _, weight)| (weight / sum) * weight_scale)
            .collect()
    } else {
        vec![0.0; selected_len]
    };
    Route { experts, weights }
}

#[inline]
pub fn relu_sqr(x: f32) -> f32 {
    let y = x.max(0.0);
    y * y
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_pattern_decodes_text_layer_kinds() {
        let kinds = decode_hybrid_pattern("ME*A").unwrap();

        assert_eq!(
            kinds,
            vec![
                NemotronLayerKind::Mamba2,
                NemotronLayerKind::MoE,
                NemotronLayerKind::Attention,
                NemotronLayerKind::Attention,
            ]
        );
    }

    #[test]
    fn hybrid_pattern_rejects_unknown_marker() {
        let err = decode_hybrid_pattern("MX").unwrap_err();

        assert_eq!(err.byte, b'X');
        assert_eq!(err.index, 1);
    }

    #[test]
    fn tensor_names_classify_mamba_moe_and_attention_layers() {
        assert_eq!(
            classify_layer_from_tensor_names(
                0,
                [
                    "blk.0.attn_norm.weight",
                    "blk.0.ssm_in.weight",
                    "blk.0.ssm_out.weight",
                ],
            )
            .unwrap(),
            Some(NemotronLayerKind::Mamba2)
        );
        assert_eq!(
            classify_layer_from_tensor_names(
                1,
                [
                    "blk.1.attn_norm.weight",
                    "blk.1.ffn_gate_inp.weight",
                    "blk.1.ffn_up_exps.weight",
                ],
            )
            .unwrap(),
            Some(NemotronLayerKind::MoE)
        );
        assert_eq!(
            classify_layer_from_tensor_names(
                5,
                [
                    "blk.5.attn_norm.weight",
                    "blk.5.attn_q.weight",
                    "blk.5.attn_output.weight",
                ],
            )
            .unwrap(),
            Some(NemotronLayerKind::Attention)
        );
    }

    #[test]
    fn sigmoid_topk_route_uses_bias_for_selection_only() {
        let logits = [0.0, 2.0, 1.0];
        let bias = [3.0, 0.0, 0.0];
        let route = sigmoid_topk_route(&logits, Some(&bias), 2, 1.0);

        assert_eq!(route.experts, vec![0, 1]);
        let sum: f32 = route.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(route.weights[0] < route.weights[1]);
    }

    #[test]
    fn relu_sqr_matches_nemotron_ffn_activation() {
        assert_eq!(relu_sqr(-2.0), 0.0);
        assert_eq!(relu_sqr(3.0), 9.0);
    }
}
