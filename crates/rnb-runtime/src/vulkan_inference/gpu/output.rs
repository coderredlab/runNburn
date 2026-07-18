use super::*;

pub fn output_logits_quant_for_test(ggml_type: GGMLType) -> Option<Quant> {
    ggml_to_output_quant(ggml_type)
}

#[allow(clippy::too_many_arguments)]
pub fn try_output_logits(
    runtime: Option<&mut Runtime>,
    ggml_type: GGMLType,
    rows: usize,
    cols: usize,
    raw: &[u8],
    normed_data: &[f32],
    output: &mut [f32],
    profiling: bool,
) -> bool {
    if !gpu_output_logits_enabled() {
        return false;
    }
    if profiling {
        eprintln!(
            "  [FWD] output_quant     {:?} (ggml={:?})",
            ggml_to_output_quant(ggml_type),
            ggml_type
        );
    }
    let Some(vk) = runtime else {
        return false;
    };
    let Some(quant) = ggml_to_output_quant(ggml_type) else {
        return false;
    };
    match vk.gemv(
        output_logits_id(),
        raw,
        rows,
        cols,
        quant,
        normed_data,
        &mut output[..rows],
    ) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[gpu] output GEMV failed, CPU fallback: {}", e);
            false
        }
    }
}
