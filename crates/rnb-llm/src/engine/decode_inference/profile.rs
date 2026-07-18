//! Decode-layer profiling report formatting.

use super::*;

pub(super) fn report_decode_layer_profile(
    weights: &ModelWeights,
    layer_times: &[f64],
    layers_ms: f64,
    profiling: bool,
    verbose: bool,
) {
    if verbose {
        let mut gdn_times = Vec::new();
        let mut atn_times = Vec::new();
        for (i, &t) in layer_times.iter().enumerate() {
            match &weights.layers[i] {
                LayerType::GatedDeltaNet(_) => gdn_times.push(t),
                LayerType::Attention(_) => atn_times.push(t),
                LayerType::NemotronMamba2(_) => gdn_times.push(t),
                LayerType::NemotronMoE(_) => gdn_times.push(t),
            }
        }
        let gdn_total: f64 = gdn_times.iter().sum();
        let atn_total: f64 = atn_times.iter().sum();

        eprintln!("  [DEC] layers_total     {:.1}ms", layers_ms);
        if gdn_times.is_empty() {
            eprintln!("  [DEC]   GDN ×0: 0.0ms total");
        } else {
            let gdn_min = gdn_times.iter().cloned().fold(f64::MAX, f64::min);
            let gdn_max = gdn_times.iter().cloned().fold(0.0f64, f64::max);
            eprintln!(
                "  [DEC]   GDN ×{}: {:.1}ms total (avg {:.2}ms, min {:.2}ms, max {:.2}ms)",
                gdn_times.len(),
                gdn_total,
                gdn_total / gdn_times.len() as f64,
                gdn_min,
                gdn_max
            );
        }
        if atn_times.is_empty() {
            eprintln!("  [DEC]   ATN ×0: 0.0ms total");
        } else {
            let atn_min = atn_times.iter().cloned().fold(f64::MAX, f64::min);
            let atn_max = atn_times.iter().cloned().fold(0.0f64, f64::max);
            eprintln!(
                "  [DEC]   ATN ×{}: {:.1}ms total (avg {:.2}ms, min {:.2}ms, max {:.2}ms)",
                atn_times.len(),
                atn_total,
                atn_total / atn_times.len() as f64,
                atn_min,
                atn_max
            );
        }
        for (i, &t) in layer_times.iter().enumerate() {
            let kind = match &weights.layers[i] {
                LayerType::GatedDeltaNet(_) => "GDN",
                LayerType::Attention(_) => "ATN",
                LayerType::NemotronMamba2(_) => "NMT-M",
                LayerType::NemotronMoE(_) => "NMT-E",
            };
            eprintln!("  [DEC]   L{:2} ({}) {:.2}ms", i, kind, t);
        }
        let sum_layers: f64 = layer_times.iter().sum();
        eprintln!(
            "  [DEC]   sum={:.1}ms, measured={:.1}ms, overhead={:.2}ms",
            sum_layers,
            layers_ms,
            layers_ms - sum_layers
        );
    } else if profiling {
        eprintln!("  [DEC] layers_total     {:.1}ms", layers_ms);
    }
}
