use super::layer_weights::{LayerType, ModelWeights, SharedExpertMoELayerWeights};
use super::policy;
use std::path::Path;
use std::sync::Arc;

// Engine-owned `.rnb` `MOE_DECODE_SECTION` layout.
// Parser output (`rnb_loader::rnb_moe_reader::MoeDecodeLayer<'a>`) borrows
// `&'a [u8]` slices into the mmap-backed file. `SharedExpertMoELayerWeights` is
// `'static`-lifetime-dominated (stored in `ModelWeights`), so we cannot carry
// those borrow lifetimes directly. Instead we record `(offset, len)` into a
// shared `Arc<memmap2::Mmap>` buffer owned by the engine; at decode time the forward
// path does `&file_bytes[offset..offset + len]` to reconstruct the slice.
//
// `SharedExpertMoEView::forward` consumes these fields on aarch64 only when
// `RNB_MOE_DECODE=1` explicitly enables this diagnostic sidecar path.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MoeSectionRowGU {
    pub gate_mul: f32,
    pub up_mul: f32,
    pub blocks_offset: usize,
    pub blocks_len: usize,
    pub scale_offset: Option<usize>,
    pub scale_len: usize,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MoeSectionRowDown {
    pub down_mul: f32,
    pub blocks_offset: usize,
    pub blocks_len: usize,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MoeSectionExpert {
    /// One entry per row; `len() == d_ff`.
    pub gate_up_rows: Vec<MoeSectionRowGU>,
    /// One entry per row; `len() == n_embd`.
    pub down_rows: Vec<MoeSectionRowDown>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MoeSectionShared {
    pub d_ff_s: u32,
    pub shared_gate_up_rows: Vec<MoeSectionRowGU>,
    pub shared_down_rows: Vec<MoeSectionRowDown>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MoeSectionDecodeLayer {
    /// Shared ownership of the `.rnb` MoE section file bytes. Kept here so the
    /// `(offset, len)` pairs below remain valid for the lifetime of the
    /// engine. Cloned once per layer from the top-level `Arc` held in
    /// `Engine::moe_section_decode_bytes`.
    pub file_bytes: Arc<memmap2::Mmap>,
    pub n_experts: u32,
    pub d_ff: u32,
    pub n_embd: u32,
    /// Quant tag for the per-expert gate+up rows (see `rnb_moe_reader` docs).
    pub gate_up_quant: u8,
    /// Quant tag for the per-expert down rows.
    pub down_quant: u8,
    /// `0xFF` (`rnb_moe_reader::SHARED_QUANT_NONE`) when no shared expert.
    pub shared_quant: u8,
    pub experts: Vec<MoeSectionExpert>,
    pub shared_expert: Option<MoeSectionShared>,
}

pub(super) fn moe_section_decode_disabled() -> bool {
    policy::moe_section_decode_disabled()
}

/// Subslice-within-parent byte offset. Both pointers must be into the same
/// allocation (`rnb-loader::rnb_moe_reader` returns slices carved out of the
/// `bytes` buffer passed to `RnbMoeView::from_bytes`, so this always holds).
#[inline]
pub(super) fn offset_of_subslice(sub: &[u8], parent: &[u8]) -> usize {
    let sub_p = sub.as_ptr() as usize;
    let parent_p = parent.as_ptr() as usize;
    sub_p
        .checked_sub(parent_p)
        .expect("subslice pointer must be inside parent buffer")
}

/// Convert parser output (`rnb_loader::rnb_moe_reader::MoeDecodeLayer<'_>`, which
/// borrows `&[u8]` slices from the file bytes) into the engine-owned
/// `MoeSectionDecodeLayer` that stores `(offset, len)` pairs against the shared
/// `bytes_arc`. Split out of `attach_moe_section_decode` so it can be exercised in
/// unit tests without driving a full engine load.
pub(super) fn convert_moe_section_decode_layer<'a>(
    parsed: rnb_loader::rnb_moe_reader::MoeDecodeLayer<'a>,
    bytes_arc: Arc<memmap2::Mmap>,
) -> MoeSectionDecodeLayer {
    let parent: &[u8] = &bytes_arc[..];

    let experts: Vec<MoeSectionExpert> = parsed
        .experts
        .into_iter()
        .map(|e| MoeSectionExpert {
            gate_up_rows: e
                .gate_up_rows
                .into_iter()
                .map(|r| MoeSectionRowGU {
                    gate_mul: r.gate_mul,
                    up_mul: r.up_mul,
                    blocks_offset: offset_of_subslice(r.blocks_bytes, parent),
                    blocks_len: r.blocks_bytes.len(),
                    scale_offset: r.scale_bytes.map(|bytes| offset_of_subslice(bytes, parent)),
                    scale_len: r.scale_bytes.map_or(0, |bytes| bytes.len()),
                })
                .collect(),
            down_rows: e
                .down_rows
                .into_iter()
                .map(|r| MoeSectionRowDown {
                    down_mul: r.down_mul,
                    blocks_offset: offset_of_subslice(r.blocks_bytes, parent),
                    blocks_len: r.blocks_bytes.len(),
                })
                .collect(),
        })
        .collect();

    let shared_expert: Option<MoeSectionShared> = parsed.shared_expert.map(|s| MoeSectionShared {
        d_ff_s: s.d_ff_s,
        shared_gate_up_rows: s
            .shared_gate_up_rows
            .into_iter()
            .map(|r| MoeSectionRowGU {
                gate_mul: r.gate_mul,
                up_mul: r.up_mul,
                blocks_offset: offset_of_subslice(r.blocks_bytes, parent),
                blocks_len: r.blocks_bytes.len(),
                scale_offset: r.scale_bytes.map(|bytes| offset_of_subslice(bytes, parent)),
                scale_len: r.scale_bytes.map_or(0, |bytes| bytes.len()),
            })
            .collect(),
        shared_down_rows: s
            .shared_down_rows
            .into_iter()
            .map(|r| MoeSectionRowDown {
                down_mul: r.down_mul,
                blocks_offset: offset_of_subslice(r.blocks_bytes, parent),
                blocks_len: r.blocks_bytes.len(),
            })
            .collect(),
    });

    MoeSectionDecodeLayer {
        file_bytes: bytes_arc,
        n_experts: parsed.n_experts,
        d_ff: parsed.d_ff,
        n_embd: parsed.n_embd,
        gate_up_quant: parsed.gate_up_quant,
        down_quant: parsed.down_quant,
        shared_quant: parsed.shared_quant,
        experts,
        shared_expert,
    }
}

/// Detect and attach an `.rnb` `MOE_DECODE_SECTION` to every
/// `SharedExpertMoELayerWeights` in `weights`. Returns the mmap buffer
/// that must outlive those attachments (stored on the `Engine`), or `None`
/// when no MoE section file/section applies.
///
/// Lookup order:
///   1. `RNB_MOE_DECODE=1` must explicitly opt in.
///   2. `<model_stem>.rnb` must exist with `RNBM` magic.
///
/// Errors (missing layers, truncated sections, etc.) are logged and the
/// function returns `None` — the engine stays usable via its existing
/// GGUF/v1 paths, which matches the spec's "kill switch" behavior.
pub(super) fn moe_section_decode_sidecar_requested(path: &Path) -> bool {
    if moe_section_decode_disabled() {
        return false;
    }

    let rnb_path = path.with_extension("rnb");
    let mut probe = [0u8; 4];
    let Ok(mut f) = std::fs::File::open(&rnb_path) else {
        return false;
    };

    use std::io::Read;
    f.read_exact(&mut probe).is_ok() && probe == *b"RNBM"
}

pub(super) fn attach_moe_section_decode(
    path: &Path,
    weights: &mut ModelWeights,
) -> Option<Arc<memmap2::Mmap>> {
    if moe_section_decode_disabled() {
        return None;
    }

    let rnb_path = path.with_extension("rnb");
    if !moe_section_decode_sidecar_requested(path) {
        return None;
    }

    let file = match std::fs::File::open(&rnb_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "[WARN] MoE section MOE_DECODE open failed ({:?}): {}",
                rnb_path, e
            );
            return None;
        }
    };
    let file_bytes = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(mmap) => mmap,
        Err(e) => {
            eprintln!(
                "[WARN] MoE section MOE_DECODE mmap failed ({:?}): {}",
                rnb_path, e
            );
            return None;
        }
    };
    let bytes_arc = Arc::new(file_bytes);

    let view = match rnb_loader::rnb_moe_reader::RnbMoeView::from_bytes(&bytes_arc) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[WARN] MoE section MOE_DECODE header parse failed: {}", e);
            return None;
        }
    };

    let moe_result = match view.parse_moe_decode() {
        Some(r) => r,
        None => {
            eprintln!(
                "[INFO] MoE section `.rnb` {:?} has no MOE_DECODE_SECTION, skipping",
                rnb_path
            );
            return None;
        }
    };

    let moe = match moe_result {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[WARN] MoE section MOE_DECODE parse failed: {}", e);
            return None;
        }
    };

    let mut moe_layer_indices: Vec<usize> = Vec::new();
    for (i, layer) in weights.layers.iter().enumerate() {
        let has_shared_expert_moe = match layer {
            LayerType::Attention(w) => w.shared_expert_moe.is_some(),
            LayerType::GatedDeltaNet(w) => w.shared_expert_moe.is_some(),
            LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => false,
        };
        if has_shared_expert_moe {
            moe_layer_indices.push(i);
        }
    }

    if moe.layers.len() != moe_layer_indices.len() {
        eprintln!(
            "[WARN] MoE section MOE_DECODE layer count {} != engine MoE-layer count {}, skipping",
            moe.layers.len(),
            moe_layer_indices.len()
        );
        return None;
    }

    for (parsed_layer, layer_idx) in moe
        .layers
        .into_iter()
        .zip(moe_layer_indices.iter().copied())
    {
        let moe_section_layer = convert_moe_section_decode_layer(parsed_layer, bytes_arc.clone());

        let moe_slot: &mut Option<SharedExpertMoELayerWeights> =
            match &mut weights.layers[layer_idx] {
                LayerType::Attention(w) => &mut w.shared_expert_moe,
                LayerType::GatedDeltaNet(w) => &mut w.shared_expert_moe,
                LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => continue,
            };
        if let Some(mw) = moe_slot.as_mut() {
            mw.moe_section_decode = Some(moe_section_layer);
        }
    }

    eprintln!(
        "[INFO] MoE section MOE_DECODE attached: {} layers from {:?}",
        moe_layer_indices.len(),
        rnb_path.file_name().unwrap_or_default()
    );
    Some(bytes_arc)
}
