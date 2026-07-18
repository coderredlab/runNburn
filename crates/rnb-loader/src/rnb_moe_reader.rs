//! `.rnb` MoE section file detection + section parsers.
//!
//! Reader side of the section-based `.rnb` MoE section format emitted
//! by `rnb-convert`. This module provides a thin zero-copy view over the raw
//! bytes (typically an mmap) and a structural parser for the
//! `MOE_DECODE_SECTION` body.
//!
//! The parser walks the byte stream exactly mirroring the encoder in
//! `crates/rnb-tools/convert/src/bin/rnb-convert.rs::encode_moe_decode_section_qwen35moe`
//! so a byte-level round trip (encode → parse) stays valid. Blocks are
//! returned as `&[u8]` slices; callers (Task 12 engine wiring) reinterpret
//! them as `&[GUPairQ4K]` / `&[Q5KIntScale]` / `&[SharedGUQ8KUnit]` /
//! `&[Q80IntScale]` via `slice::from_raw_parts` in the hot loop — the
//! parser itself is dependency-free on `rnb-cpu`.
//!
//! See `docs/superpowers/specs/2026-04-23-moe-decode-packing-redesign-design.md`
//! §5.1 for the on-disk layout.

use rnb_core::rnb_moe::{MoeHeader, SectionId};

// -----------------------------------------------------------------------------
// Block byte sizes (must stay in sync with rnb-cpu::quantize::moe_blocks)
// -----------------------------------------------------------------------------

/// Bytes per `Q4KIntScale` block (rnb-cpu::quantize::moe_blocks::Q4KIntScale).
pub const Q4K_INTSCALE_BYTES: usize = 144;
/// Bytes per `Q5KIntScale` block.
pub const Q5K_INTSCALE_BYTES: usize = 176;
/// Bytes per `Q80IntScale` block (flat Q8_0 shared-down unit).
pub const Q80_INTSCALE_BYTES: usize = 48;
/// Bytes per `GUPairQ4K` (2 × Q4KIntScale).
pub const GU_PAIR_Q4K_BYTES: usize = 2 * Q4K_INTSCALE_BYTES; // 288
/// Bytes per `GUPairQ4KScaleMin`.
pub const GU_PAIR_Q4K_SCALE_MIN_BYTES: usize = 32;
/// Bytes per `GUPairQ4KUnpackedScales`.
pub const GU_PAIR_Q4K_UNPACKED_SCALES_BYTES: usize = GU_PAIR_Q4K_BYTES + 32; // 320
/// Bytes per `SharedGUQ8KUnit` (16 × Q80IntScale — 8 gate + 8 up).
pub const SHARED_GU_Q8K_UNIT_BYTES: usize = 16 * Q80_INTSCALE_BYTES; // 768

/// Gate/up row quant tag for baseline Q4_K `GUPairQ4K` units.
pub const GATE_UP_QUANT_Q4K_PAIR: u8 = 0x12;
/// Gate/up row quant tag for Q4_K units with unpacked scale/min side data.
pub const GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES: u8 = 0x32;
/// Gate/up row quant tag for Q4_K rows with a separate scale/min plane.
pub const GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE: u8 = 0x33;

/// Sentinel byte in the per-layer `shared_quant` field meaning "no shared
/// expert" (used by Gemma4 26B MoE).
pub const SHARED_QUANT_NONE: u8 = 0xFF;

// -----------------------------------------------------------------------------
// RnbMoeView
// -----------------------------------------------------------------------------

/// A view of a `.rnb` MoE section file backed by a slice of bytes (typically an mmap).
///
/// Construction parses only the `MoeHeader` + section table — section bodies
/// are returned as zero-copy slices on demand via [`section`].
///
/// [`section`]: RnbMoeView::section
pub struct RnbMoeView<'a> {
    pub header: MoeHeader,
    pub bytes: &'a [u8],
}

impl<'a> RnbMoeView<'a> {
    /// Parse the fixed-size MoE section header + section table. Returns `Err` if the
    /// magic or version don't match, or if the section table is truncated.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, String> {
        let header = MoeHeader::from_bytes(bytes)?;
        Ok(Self { header, bytes })
    }

    /// Return the body slice of the first section with matching `id`, or
    /// `None` if not present. The slice is exactly `size` bytes long
    /// (section-table `size`, not the 16B-padded on-disk extent).
    pub fn section(&self, id: SectionId) -> Option<&'a [u8]> {
        let entry = self.header.sections.iter().find(|s| s.id == id)?;
        let off = entry.offset as usize;
        let sz = entry.size as usize;
        if off
            .checked_add(sz)
            .map_or(true, |end| end > self.bytes.len())
        {
            return None;
        }
        Some(&self.bytes[off..off + sz])
    }

    /// Convenience: parse the `MOE_DECODE_SECTION` body if present.
    /// Outer `Option` = section missing; inner `Result` = parse error.
    pub fn parse_moe_decode(&self) -> Option<Result<MoeDecodeParsed<'a>, String>> {
        self.section(SectionId::MoeDecode)
            .map(parse_moe_decode_section)
    }
}

// -----------------------------------------------------------------------------
// MOE_DECODE parsed structures
// -----------------------------------------------------------------------------

/// Parsed `MOE_DECODE_SECTION` — one entry per layer in arch-declared order.
pub struct MoeDecodeParsed<'a> {
    pub layers: Vec<MoeDecodeLayer<'a>>,
}

/// One layer's MoE metadata + per-expert byte ranges + optional shared expert.
pub struct MoeDecodeLayer<'a> {
    pub n_experts: u32,
    pub d_ff: u32,
    pub n_embd: u32,
    pub gate_up_quant: u8,
    pub down_quant: u8,
    /// `0xFF` ([`SHARED_QUANT_NONE`]) sentinel = no shared expert.
    pub shared_quant: u8,
    pub experts: Vec<MoeDecodeExpert<'a>>,
    pub shared_expert: Option<MoeDecodeShared<'a>>,
}

/// One expert's per-row byte ranges.
///
/// Callers reinterpret `gate_up_rows[r].blocks_bytes` as `&[GUPairQ4K]` of
/// length `n_embd / 256`, and `down_rows[r].blocks_bytes` as
/// `&[Q5KIntScale]` of length `d_ff / 256`.
pub struct MoeDecodeExpert<'a> {
    /// One entry per gate/up row, length == layer.d_ff.
    pub gate_up_rows: Vec<RowGU<'a>>,
    /// One entry per down row, length == layer.n_embd.
    pub down_rows: Vec<RowDown<'a>>,
}

/// A single gate+up row (either per-expert Q4_K pair or shared-expert
/// Q8_0 unit; reader does not distinguish — caller decides how to cast
/// `blocks_bytes` based on the layer's `gate_up_quant` / `shared_quant`).
pub struct RowGU<'a> {
    pub gate_mul: f32,
    pub up_mul: f32,
    pub blocks_bytes: &'a [u8],
    pub scale_bytes: Option<&'a [u8]>,
}

/// A single down row.
pub struct RowDown<'a> {
    pub down_mul: f32,
    pub blocks_bytes: &'a [u8],
}

/// Shared-expert byte ranges. For Qwen3.6 the parser assumes
/// `d_ff_s == layer.d_ff` (the encoder hardcodes this); `shared_gate_up_rows`
/// has exactly `d_ff_s` entries and `shared_down_rows` has `n_embd` entries.
pub struct MoeDecodeShared<'a> {
    pub d_ff_s: u32,
    pub shared_gate_up_rows: Vec<RowGU<'a>>,
    pub shared_down_rows: Vec<RowDown<'a>>,
}

// -----------------------------------------------------------------------------
// Parser internals
// -----------------------------------------------------------------------------

fn read_u32_le(bytes: &[u8], cur: &mut usize) -> Result<u32, String> {
    let end = cur
        .checked_add(4)
        .ok_or_else(|| "cursor overflow reading u32".to_string())?;
    if end > bytes.len() {
        return Err(format!(
            "truncated u32: need {}..{} of {}",
            cur,
            end,
            bytes.len()
        ));
    }
    let v = u32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
    *cur = end;
    Ok(v)
}

fn read_u8(bytes: &[u8], cur: &mut usize) -> Result<u8, String> {
    if *cur >= bytes.len() {
        return Err(format!("truncated u8 at {} of {}", cur, bytes.len()));
    }
    let v = bytes[*cur];
    *cur += 1;
    Ok(v)
}

fn read_f32_le(bytes: &[u8], cur: &mut usize) -> Result<f32, String> {
    let end = cur
        .checked_add(4)
        .ok_or_else(|| "cursor overflow reading f32".to_string())?;
    if end > bytes.len() {
        return Err(format!(
            "truncated f32: need {}..{} of {}",
            cur,
            end,
            bytes.len()
        ));
    }
    let v = f32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
    *cur = end;
    Ok(v)
}

fn align_to(cur: &mut usize, boundary: usize) -> Result<(), String> {
    if !boundary.is_power_of_two() {
        return Err(format!("align_to: non-power-of-two boundary {}", boundary));
    }
    let mask = boundary - 1;
    let aligned = (*cur + mask) & !mask;
    *cur = aligned;
    Ok(())
}

fn slice_bytes<'a>(bytes: &'a [u8], cur: &mut usize, len: usize) -> Result<&'a [u8], String> {
    let end = cur
        .checked_add(len)
        .ok_or_else(|| "cursor overflow slicing bytes".to_string())?;
    if end > bytes.len() {
        return Err(format!(
            "truncated slice: need {}..{} ({} bytes) of {}",
            cur,
            end,
            len,
            bytes.len()
        ));
    }
    let out = &bytes[*cur..end];
    *cur = end;
    Ok(out)
}

fn gate_up_block_bytes(gate_up_quant: u8) -> Result<usize, String> {
    match gate_up_quant {
        GATE_UP_QUANT_Q4K_PAIR | GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE => Ok(GU_PAIR_Q4K_BYTES),
        GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES => Ok(GU_PAIR_Q4K_UNPACKED_SCALES_BYTES),
        other => Err(format!("unsupported gate_up_quant {other:#x}")),
    }
}

fn gate_up_scale_plane_bytes(gate_up_quant: u8, n_embd_blocks: usize) -> Option<usize> {
    (gate_up_quant == GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE)
        .then_some(n_embd_blocks * GU_PAIR_Q4K_SCALE_MIN_BYTES)
}

// -----------------------------------------------------------------------------
// MOE_DECODE body parser
// -----------------------------------------------------------------------------

/// Parse a `MOE_DECODE_SECTION` body. `body` must be the exact logical body
/// (the slice returned by [`RnbMoeView::section`] for
/// `SectionId::MoeDecode`) — NOT including any 16B inter-section padding.
pub fn parse_moe_decode_section<'a>(body: &'a [u8]) -> Result<MoeDecodeParsed<'a>, String> {
    let mut cur = 0usize;
    let per_layer_count = read_u32_le(body, &mut cur)?;
    let mut layers = Vec::with_capacity(per_layer_count as usize);

    for layer_idx in 0..per_layer_count {
        let n_experts = read_u32_le(body, &mut cur)?;
        let d_ff = read_u32_le(body, &mut cur)?;
        let n_embd = read_u32_le(body, &mut cur)?;
        let gate_up_quant = read_u8(body, &mut cur)?;
        let down_quant = read_u8(body, &mut cur)?;
        let shared_quant = read_u8(body, &mut cur)?;
        align_to(&mut cur, 16)?;

        if n_embd as usize % 256 != 0 {
            return Err(format!(
                "layer {}: n_embd={} not a multiple of 256",
                layer_idx, n_embd
            ));
        }
        if d_ff as usize % 256 != 0 {
            return Err(format!(
                "layer {}: d_ff={} not a multiple of 256",
                layer_idx, d_ff
            ));
        }

        let n_embd_blocks = (n_embd as usize) / 256;
        let d_ff_blocks = (d_ff as usize) / 256;
        let gate_up_unit_bytes =
            gate_up_block_bytes(gate_up_quant).map_err(|e| format!("layer {layer_idx}: {e}"))?;

        let mut experts = Vec::with_capacity(n_experts as usize);
        for _e in 0..n_experts {
            // gate_up_rows[d_ff] — each row is (f32 gate_mul, f32 up_mul, n_embd/256 GUPairQ4K)
            let mut gate_up_rows = Vec::with_capacity(d_ff as usize);
            for _r in 0..d_ff {
                let gate_mul = read_f32_le(body, &mut cur)?;
                let up_mul = read_f32_le(body, &mut cur)?;
                let blocks_len = n_embd_blocks * gate_up_unit_bytes;
                let blocks_bytes = slice_bytes(body, &mut cur, blocks_len)?;
                gate_up_rows.push(RowGU {
                    gate_mul,
                    up_mul,
                    blocks_bytes,
                    scale_bytes: None,
                });
                align_to(&mut cur, 64)?;
            }
            if let Some(scale_row_len) = gate_up_scale_plane_bytes(gate_up_quant, n_embd_blocks) {
                for row in gate_up_rows.iter_mut() {
                    let scale_bytes = slice_bytes(body, &mut cur, scale_row_len)?;
                    row.scale_bytes = Some(scale_bytes);
                    align_to(&mut cur, 64)?;
                }
            }

            // down_rows[n_embd] — each row is (f32 down_mul, d_ff/256 Q5KIntScale)
            let mut down_rows = Vec::with_capacity(n_embd as usize);
            for _r in 0..n_embd {
                let down_mul = read_f32_le(body, &mut cur)?;
                let blocks_len = d_ff_blocks * Q5K_INTSCALE_BYTES;
                let blocks_bytes = slice_bytes(body, &mut cur, blocks_len)?;
                down_rows.push(RowDown {
                    down_mul,
                    blocks_bytes,
                });
                align_to(&mut cur, 64)?;
            }

            experts.push(MoeDecodeExpert {
                gate_up_rows,
                down_rows,
            });
        }

        let shared_expert = if shared_quant != SHARED_QUANT_NONE {
            // Qwen3.6 encoder hardcodes d_ff_s = d_ff. We mirror that here;
            // future archs with a distinct shared d_ff will need a header
            // field addition (and a version bump).
            let d_ff_s = d_ff;
            let n_embd_blocks_shared = n_embd_blocks; // n_embd / 256
            let mut shared_gate_up_rows = Vec::with_capacity(d_ff_s as usize);
            for _r in 0..d_ff_s {
                let gate_mul = read_f32_le(body, &mut cur)?;
                let up_mul = read_f32_le(body, &mut cur)?;
                let blocks_len = n_embd_blocks_shared * SHARED_GU_Q8K_UNIT_BYTES;
                let blocks_bytes = slice_bytes(body, &mut cur, blocks_len)?;
                shared_gate_up_rows.push(RowGU {
                    gate_mul,
                    up_mul,
                    blocks_bytes,
                    scale_bytes: None,
                });
                align_to(&mut cur, 64)?;
            }

            // shared down: flat Q8_0 per n_embd row. Block count = d_ff_s / 32.
            if (d_ff_s as usize) % 32 != 0 {
                return Err(format!(
                    "layer {}: shared d_ff_s={} not a multiple of 32",
                    layer_idx, d_ff_s
                ));
            }
            let d_ff_s_blocks_q80 = (d_ff_s as usize) / 32;
            let mut shared_down_rows = Vec::with_capacity(n_embd as usize);
            for _r in 0..n_embd {
                let down_mul = read_f32_le(body, &mut cur)?;
                let blocks_len = d_ff_s_blocks_q80 * Q80_INTSCALE_BYTES;
                let blocks_bytes = slice_bytes(body, &mut cur, blocks_len)?;
                shared_down_rows.push(RowDown {
                    down_mul,
                    blocks_bytes,
                });
                align_to(&mut cur, 64)?;
            }

            Some(MoeDecodeShared {
                d_ff_s,
                shared_gate_up_rows,
                shared_down_rows,
            })
        } else {
            None
        };

        layers.push(MoeDecodeLayer {
            n_experts,
            d_ff,
            n_embd,
            gate_up_quant,
            down_quant,
            shared_quant,
            experts,
            shared_expert,
        });
    }

    // Cursor may land short of body.len() if the last row's 64B pad runs
    // past the declared section size (the encoder's final pad bytes live in
    // the 16B *inter-section* pad, not the section body itself) — that's
    // fine. We only reject overshoots, which would indicate a truncated
    // section size in the header.
    if cur > body.len() {
        return Err(format!(
            "parser overran body: cursor {} > len {}",
            cur,
            body.len()
        ));
    }

    Ok(MoeDecodeParsed { layers })
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_core::rnb_moe::{
        SectionTableEntry, MOE_HEADER_FIXED_LEN, MOE_MAGIC, MOE_SECTION_ENTRY_LEN,
    };

    // ----- RnbMoeView basics ---------------------------------------------------

    #[test]
    fn moe_view_from_bytes_rejects_non_rnbm() {
        // Short buffer
        assert!(RnbMoeView::from_bytes(&[0u8; 4]).is_err());
        // Wrong magic (but enough bytes)
        let mut buf = Vec::new();
        buf.extend_from_slice(b"WRNG");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        assert!(RnbMoeView::from_bytes(&buf).is_err());
        // Wrong version
        let mut buf = Vec::new();
        buf.extend_from_slice(&MOE_MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        assert!(RnbMoeView::from_bytes(&buf).is_err());
    }

    #[test]
    fn moe_view_section_returns_slice() {
        // Build a minimal MoE section file in-memory with one section:
        // - header (12B) + 1 entry (17B) = 29B
        // - padded to 32B (16B alignment)
        // - body = 5 bytes of arbitrary contents
        let body: &[u8] = &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let header_size = MOE_HEADER_FIXED_LEN + MOE_SECTION_ENTRY_LEN; // 29
        let body_start = (header_size + 15) & !15; // 32

        let mut file = Vec::new();
        file.extend_from_slice(&MOE_MAGIC);
        file.extend_from_slice(&2u32.to_le_bytes());
        file.extend_from_slice(&1u32.to_le_bytes()); // count=1
                                                     // entry: id=MoeDecode(0x02), offset=32, size=5
        file.push(SectionId::MoeDecode as u8);
        file.extend_from_slice(&(body_start as u64).to_le_bytes());
        file.extend_from_slice(&(body.len() as u64).to_le_bytes());
        // pad to 16B body start
        while file.len() < body_start {
            file.push(0);
        }
        file.extend_from_slice(body);

        let view = RnbMoeView::from_bytes(&file).expect("parse header");
        assert_eq!(view.header.sections.len(), 1);
        assert_eq!(view.header.sections[0].id, SectionId::MoeDecode);

        let got = view.section(SectionId::MoeDecode).expect("body present");
        assert_eq!(got, body);
        // Unknown section kinds return None.
        assert!(view.section(SectionId::AttnDecode).is_none());
    }

    #[test]
    fn moe_view_section_rejects_out_of_bounds() {
        // Hand-craft a view whose section-table entry points past EOF.
        let header = MoeHeader {
            magic: MOE_MAGIC,
            version: 2,
            sections: vec![SectionTableEntry {
                id: SectionId::MoeDecode,
                offset: 1024,
                size: 16,
            }],
        };
        let bytes = header.to_bytes();
        let view = RnbMoeView::from_bytes(&bytes).expect("parse header");
        assert!(view.section(SectionId::MoeDecode).is_none());
    }

    // ----- parse_moe_decode_section: synthetic fixture ------------------------

    /// Build a `MOE_DECODE_SECTION` body by hand (without touching the
    /// encoder in `rnb-convert`) for the smallest interesting shape:
    /// 1 layer, 1 expert, n_embd=256, d_ff=256, shared_quant=0xFF.
    fn build_synthetic_body_qwen_min() -> Vec<u8> {
        let mut out = Vec::<u8>::new();
        // per_layer_count = 1
        out.extend_from_slice(&1u32.to_le_bytes());
        // layer header: n_experts=1, d_ff=256, n_embd=256
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&256u32.to_le_bytes());
        out.extend_from_slice(&256u32.to_le_bytes());
        // gate_up_quant=0x12, down_quant=0x14, shared_quant=0xFF (no shared)
        out.push(0x12);
        out.push(0x14);
        out.push(0xFF);
        // pad to 16B
        while out.len() % 16 != 0 {
            out.push(0);
        }

        // Expert 0.
        // gate_up_rows: d_ff=256 rows; each row = 8B muls + 1 GUPairQ4K (288B).
        // Pad each row to 64B boundary.
        for r in 0..256u32 {
            let gate_mul = (r as f32) + 0.125;
            let up_mul = (r as f32) + 0.250;
            out.extend_from_slice(&gate_mul.to_le_bytes());
            out.extend_from_slice(&up_mul.to_le_bytes());
            // 1 GUPairQ4K = 288 bytes of payload; use distinctive pattern.
            let pat = (r & 0xFF) as u8;
            for _ in 0..GU_PAIR_Q4K_BYTES {
                out.push(pat);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }
        // down_rows: n_embd=256 rows; each row = 4B down_mul + 1 Q5KIntScale (176B).
        for r in 0..256u32 {
            let down_mul = (r as f32) * 2.0 + 1.0;
            out.extend_from_slice(&down_mul.to_le_bytes());
            let pat = ((r ^ 0xA5) & 0xFF) as u8;
            for _ in 0..Q5K_INTSCALE_BYTES {
                out.push(pat);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }

        out
    }

    #[test]
    fn parse_moe_decode_section_synthetic() {
        let body = build_synthetic_body_qwen_min();
        let parsed = parse_moe_decode_section(&body).expect("parse ok");
        assert_eq!(parsed.layers.len(), 1);
        let layer = &parsed.layers[0];
        assert_eq!(layer.n_experts, 1);
        assert_eq!(layer.d_ff, 256);
        assert_eq!(layer.n_embd, 256);
        assert_eq!(layer.gate_up_quant, 0x12);
        assert_eq!(layer.down_quant, 0x14);
        assert_eq!(layer.shared_quant, 0xFF);
        assert!(layer.shared_expert.is_none());
        assert_eq!(layer.experts.len(), 1);

        let expert = &layer.experts[0];
        assert_eq!(expert.gate_up_rows.len(), 256);
        assert_eq!(expert.down_rows.len(), 256);

        // Spot-check row 0 and row 42.
        for &r in &[0usize, 42, 255] {
            let row = &expert.gate_up_rows[r];
            assert!((row.gate_mul - ((r as f32) + 0.125)).abs() < 1e-6);
            assert!((row.up_mul - ((r as f32) + 0.250)).abs() < 1e-6);
            assert_eq!(row.blocks_bytes.len(), GU_PAIR_Q4K_BYTES);
            let pat = (r & 0xFF) as u8;
            assert!(row.blocks_bytes.iter().all(|b| *b == pat));

            let drow = &expert.down_rows[r];
            assert!((drow.down_mul - ((r as f32) * 2.0 + 1.0)).abs() < 1e-6);
            assert_eq!(drow.blocks_bytes.len(), Q5K_INTSCALE_BYTES);
            let pat = ((r ^ 0xA5) & 0xFF) as u8;
            assert!(drow.blocks_bytes.iter().all(|b| *b == pat));
        }
    }

    #[test]
    fn parse_moe_decode_section_unpacked_scale_gate_up_rows() {
        let mut out = Vec::<u8>::new();
        out.extend_from_slice(&1u32.to_le_bytes()); // per_layer_count
        out.extend_from_slice(&1u32.to_le_bytes()); // n_experts
        out.extend_from_slice(&256u32.to_le_bytes()); // d_ff
        out.extend_from_slice(&256u32.to_le_bytes()); // n_embd
        out.push(GATE_UP_QUANT_Q4K_PAIR_UNPACKED_SCALES);
        out.push(0x14);
        out.push(0xFF);
        while out.len() % 16 != 0 {
            out.push(0);
        }

        for r in 0..256u32 {
            out.extend_from_slice(&(r as f32).to_le_bytes());
            out.extend_from_slice(&((r as f32) + 0.5).to_le_bytes());
            let pat = (r & 0xFF) as u8;
            for _ in 0..GU_PAIR_Q4K_UNPACKED_SCALES_BYTES {
                out.push(pat);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }
        for r in 0..256u32 {
            out.extend_from_slice(&(r as f32).to_le_bytes());
            for _ in 0..Q5K_INTSCALE_BYTES {
                out.push(0xA5);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }

        let parsed = parse_moe_decode_section(&out).expect("parse ok");
        let expert = &parsed.layers[0].experts[0];
        let row = &expert.gate_up_rows[42];
        assert_eq!(row.blocks_bytes.len(), GU_PAIR_Q4K_UNPACKED_SCALES_BYTES);
        assert!(row.blocks_bytes.iter().all(|b| *b == 42));
        let drow = &expert.down_rows[42];
        assert_eq!(drow.blocks_bytes.len(), Q5K_INTSCALE_BYTES);
        assert!(drow.blocks_bytes.iter().all(|b| *b == 0xA5));
    }

    #[test]
    fn parse_moe_decode_section_scale_plane_gate_up_rows() {
        let mut out = Vec::<u8>::new();
        out.extend_from_slice(&1u32.to_le_bytes()); // per_layer_count
        out.extend_from_slice(&1u32.to_le_bytes()); // n_experts
        out.extend_from_slice(&256u32.to_le_bytes()); // d_ff
        out.extend_from_slice(&256u32.to_le_bytes()); // n_embd
        out.push(GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE);
        out.push(0x14);
        out.push(0xFF);
        while out.len() % 16 != 0 {
            out.push(0);
        }

        for r in 0..256u32 {
            out.extend_from_slice(&(r as f32).to_le_bytes());
            out.extend_from_slice(&((r as f32) + 0.5).to_le_bytes());
            let pat = (r & 0xFF) as u8;
            for _ in 0..GU_PAIR_Q4K_BYTES {
                out.push(pat);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }
        for r in 0..256u32 {
            let pat = 0x80 | (r & 0x7F) as u8;
            for _ in 0..GU_PAIR_Q4K_SCALE_MIN_BYTES {
                out.push(pat);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }
        for r in 0..256u32 {
            out.extend_from_slice(&(r as f32).to_le_bytes());
            for _ in 0..Q5K_INTSCALE_BYTES {
                out.push(0xA5);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }

        let parsed = parse_moe_decode_section(&out).expect("parse ok");
        let layer = &parsed.layers[0];
        assert_eq!(layer.gate_up_quant, GATE_UP_QUANT_Q4K_PAIR_SCALE_PLANE);
        let row = &layer.experts[0].gate_up_rows[42];
        assert_eq!(row.blocks_bytes.len(), GU_PAIR_Q4K_BYTES);
        assert!(row.blocks_bytes.iter().all(|b| *b == 42));
        let scale_bytes = row.scale_bytes.expect("scale plane bytes");
        assert_eq!(scale_bytes.len(), GU_PAIR_Q4K_SCALE_MIN_BYTES);
        assert!(scale_bytes.iter().all(|b| *b == 0xAA));
    }

    #[test]
    fn parse_moe_decode_section_rejects_bad_n_embd() {
        let mut body = Vec::<u8>::new();
        body.extend_from_slice(&1u32.to_le_bytes()); // per_layer_count
        body.extend_from_slice(&1u32.to_le_bytes()); // n_experts
        body.extend_from_slice(&256u32.to_le_bytes()); // d_ff
        body.extend_from_slice(&200u32.to_le_bytes()); // n_embd NOT multiple of 256
        body.push(0x12);
        body.push(0x14);
        body.push(0xFF);
        while body.len() % 16 != 0 {
            body.push(0);
        }
        let err = match parse_moe_decode_section(&body) {
            Ok(_) => panic!("expected error for non-256-multiple n_embd"),
            Err(e) => e,
        };
        assert!(err.contains("n_embd"), "unexpected error: {err}");
    }

    #[test]
    fn parse_moe_decode_section_truncated_header() {
        // per_layer_count claims 1 layer but only 2 bytes follow.
        let mut body = Vec::<u8>::new();
        body.extend_from_slice(&1u32.to_le_bytes());
        body.push(0);
        body.push(0);
        assert!(parse_moe_decode_section(&body).is_err());
    }

    #[test]
    fn parse_moe_decode_section_with_shared_expert() {
        // 1 layer, 1 expert, n_embd=256, d_ff=256, shared_quant=0x08 (Q8_0).
        // We construct the full body including the shared_expert suffix and
        // verify the parser exposes shared_gate_up_rows[d_ff] and
        // shared_down_rows[n_embd].
        let mut out = Vec::<u8>::new();
        out.extend_from_slice(&1u32.to_le_bytes()); // per_layer_count
        out.extend_from_slice(&1u32.to_le_bytes()); // n_experts
        out.extend_from_slice(&256u32.to_le_bytes()); // d_ff
        out.extend_from_slice(&256u32.to_le_bytes()); // n_embd
        out.push(0x12); // gate_up_quant Q4_K
        out.push(0x14); // down_quant   Q5_K
        out.push(0x08); // shared_quant Q8_0
        while out.len() % 16 != 0 {
            out.push(0);
        }
        // Expert 0: d_ff=256 gate_up rows.
        for r in 0..256u32 {
            out.extend_from_slice(&(r as f32).to_le_bytes());
            out.extend_from_slice(&((r as f32) + 1.0).to_le_bytes());
            for _ in 0..GU_PAIR_Q4K_BYTES {
                out.push(0);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }
        // Expert 0: n_embd=256 down rows.
        for r in 0..256u32 {
            out.extend_from_slice(&(r as f32).to_le_bytes());
            for _ in 0..Q5K_INTSCALE_BYTES {
                out.push(0);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }
        // Shared expert: d_ff_s == d_ff == 256 rows of (gate_mul, up_mul,
        // n_embd/256 * SharedGUQ8KUnit bytes).
        for r in 0..256u32 {
            out.extend_from_slice(&((r as f32) * 10.0).to_le_bytes());
            out.extend_from_slice(&((r as f32) * 10.0 + 1.0).to_le_bytes());
            for _ in 0..SHARED_GU_Q8K_UNIT_BYTES {
                out.push(0xCD);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }
        // Shared down: n_embd=256 rows of (down_mul, d_ff_s/32 * Q80IntScale).
        let d_ff_s_blocks = 256usize / 32;
        for r in 0..256u32 {
            out.extend_from_slice(&((r as f32) * -1.0).to_le_bytes());
            for _ in 0..(d_ff_s_blocks * Q80_INTSCALE_BYTES) {
                out.push(0xEF);
            }
            while out.len() % 64 != 0 {
                out.push(0);
            }
        }

        let parsed = parse_moe_decode_section(&out).expect("parse ok");
        let layer = &parsed.layers[0];
        assert_eq!(layer.shared_quant, 0x08);
        let shared = layer.shared_expert.as_ref().expect("shared present");
        assert_eq!(shared.d_ff_s, 256);
        assert_eq!(shared.shared_gate_up_rows.len(), 256);
        assert_eq!(shared.shared_down_rows.len(), 256);

        // Spot-check the byte patterns and f32 muls survive the round trip.
        let row = &shared.shared_gate_up_rows[10];
        assert!((row.gate_mul - 100.0).abs() < 1e-6);
        assert!((row.up_mul - 101.0).abs() < 1e-6);
        assert_eq!(row.blocks_bytes.len(), SHARED_GU_Q8K_UNIT_BYTES);
        assert!(row.blocks_bytes.iter().all(|b| *b == 0xCD));

        let drow = &shared.shared_down_rows[7];
        assert!((drow.down_mul - (-7.0)).abs() < 1e-6);
        assert_eq!(drow.blocks_bytes.len(), d_ff_s_blocks * Q80_INTSCALE_BYTES);
        assert!(drow.blocks_bytes.iter().all(|b| *b == 0xEF));
    }

    #[test]
    fn align_to_is_idempotent_power_of_two() {
        let mut c = 12usize;
        align_to(&mut c, 16).unwrap();
        assert_eq!(c, 16);
        align_to(&mut c, 16).unwrap();
        assert_eq!(c, 16);
        align_to(&mut c, 64).unwrap();
        assert_eq!(c, 64);
    }
}
