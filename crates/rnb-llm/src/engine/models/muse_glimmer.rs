use crate::engine::cpu_runtime::kernels;
use crate::engine::types::ModelMetadata;
use rnb_core::tensor::Tensor;
use rnb_loader::Architecture as ModelArchitecture;

pub(in crate::engine) fn uses_muse_glimmer_semantics(architecture: ModelArchitecture) -> bool {
    matches!(architecture, ModelArchitecture::MuseGlimmer)
}

pub(in crate::engine) fn normalize_token_embeddings(
    hidden: Tensor,
    metadata: &ModelMetadata,
) -> Tensor {
    let mut data = kernels::tensor_as_f32_slice(&hidden).to_vec();
    normalize_token_embeddings_inplace(&mut data, metadata);
    Tensor::from_vec(data, hidden.shape())
}

pub(in crate::engine) fn normalize_token_embeddings_inplace(
    hidden: &mut [f32],
    metadata: &ModelMetadata,
) {
    normalize_rows_inplace(hidden, metadata.hidden_dim, metadata.norm_eps);
}

fn normalize_rows_inplace(hidden: &mut [f32], hidden_dim: usize, eps: f32) {
    for row in hidden.chunks_exact_mut(hidden_dim) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / hidden_dim as f32;
        let scale = (mean_square + eps).sqrt().recip();
        for value in row {
            *value *= scale;
        }
    }
}

pub(in crate::engine) fn scale_logits_inplace(logits: &mut [f32], scale: f32) {
    if scale == 1.0 {
        return;
    }
    for value in logits {
        *value *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_normalization_is_independent_per_token_row() {
        let mut rows = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0];

        normalize_rows_inplace(&mut rows, 4, 0.0);

        assert_eq!(&rows[..4], &[1.2, 1.6, 0.0, 0.0]);
        assert_eq!(&rows[4..], &[0.0, 0.0, 1.2, 1.6]);
    }

    #[test]
    fn output_scale_precedes_tanh_softcap_contract() {
        let mut logits = vec![-100.0, 0.0, 100.0];

        scale_logits_inplace(&mut logits, 0.2);
        crate::engine::models::gemma::apply_logit_softcapping(&mut logits, 20.0);

        assert!(logits[0] > -15.24 && logits[0] < -15.23);
        assert_eq!(logits[1], 0.0);
        assert!(logits[2] > 15.23 && logits[2] < 15.24);
    }
}
