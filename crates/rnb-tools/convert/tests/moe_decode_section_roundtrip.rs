//! Integration test: synthetic .rnb MoE section file with MOE_DECODE_SECTION;
//! rnb-loader's rnb_moe_reader parses it; structural sanity is verified.
//!
//! Phase 1 Task 10 originally also ran the legacy `rnb-convert` binary
//! end-to-end on a real Qwen3.6 35B GGUF. That binary was deleted in
//! Phase 1 Task 17 (sidecar v3 spec replaces the v2 RNBM layout exercised
//! here), so only the synthetic byte-layout test remains. The synthetic
//! test still pins the exact byte layout the reader expects, which is
//! valuable until the v2 reader itself is retired.

#[cfg(test)]
mod tests {
    /// Fast unit-style test: build a synthetic MOE_DECODE body in memory,
    /// wrap as a MoE section file, parse, verify structure. Runs on every
    /// `cargo test` (no GGUF needed).
    #[test]
    fn moe_decode_synthetic_bytes_parse_correctly() {
        use rnb_core::rnb_moe::{MoeHeader, SectionId, SectionTableEntry, MOE_MAGIC, MOE_VERSION};
        use rnb_loader::rnb_moe_reader::{
            RnbMoeView, GU_PAIR_Q4K_BYTES, Q5K_INTSCALE_BYTES, SHARED_QUANT_NONE,
        };

        // Minimal synthetic: 1 layer, 1 expert, n_experts=1, d_ff=256,
        // n_embd=256, shared=0xFF.
        let n_layers: u32 = 1;
        let n_experts: u32 = 1;
        let d_ff: u32 = 256;
        let n_embd: u32 = 256;

        // Build body.
        let mut body: Vec<u8> = Vec::new();
        body.extend_from_slice(&n_layers.to_le_bytes());
        body.extend_from_slice(&n_experts.to_le_bytes());
        body.extend_from_slice(&d_ff.to_le_bytes());
        body.extend_from_slice(&n_embd.to_le_bytes());
        body.push(0x12); // Q4_K
        body.push(0x14); // Q5_K
        body.push(SHARED_QUANT_NONE); // no shared
        while body.len() % 16 != 0 {
            body.push(0);
        }

        // Expert 0: gate_up_rows[256] — 256 rows with distinct gate_mul /
        // up_mul and zeroed GUPairQ4K bytes.
        let n_embd_blocks = (n_embd as usize) / 256; // = 1
        let gu_bytes_per_row = n_embd_blocks * GU_PAIR_Q4K_BYTES; // = 288
        for r in 0..d_ff {
            let gate_mul: f32 = 1.0 + r as f32 * 0.01;
            let up_mul: f32 = 2.0 + r as f32 * 0.01;
            body.extend_from_slice(&gate_mul.to_le_bytes());
            body.extend_from_slice(&up_mul.to_le_bytes());
            body.extend(std::iter::repeat(0u8).take(gu_bytes_per_row));
            while body.len() % 64 != 0 {
                body.push(0);
            }
        }

        // Expert 0: down_rows[256] with distinct down_mul + zeroed
        // Q5KIntScale.
        let d_ff_blocks = (d_ff as usize) / 256; // = 1
        let down_bytes_per_row = d_ff_blocks * Q5K_INTSCALE_BYTES; // = 176
        for r in 0..n_embd {
            let down_mul: f32 = 3.0 + r as f32 * 0.01;
            body.extend_from_slice(&down_mul.to_le_bytes());
            body.extend(std::iter::repeat(0u8).take(down_bytes_per_row));
            while body.len() % 64 != 0 {
                body.push(0);
            }
        }

        // No shared expert (sentinel 0xFF above).

        // Wrap in MoE section file.
        let mut header = MoeHeader {
            magic: MOE_MAGIC,
            version: MOE_VERSION,
            sections: vec![SectionTableEntry {
                id: SectionId::MoeDecode,
                offset: 0, // filled below
                size: body.len() as u64,
            }],
        };
        let header_bytes_initial = header.to_bytes();
        let mut file: Vec<u8> = Vec::new();
        file.extend_from_slice(&header_bytes_initial);
        while file.len() % 16 != 0 {
            file.push(0);
        }
        // Now fix offset in section table.
        let body_offset = file.len() as u64;
        header.sections[0].offset = body_offset;
        let header_bytes_final = header.to_bytes();
        file[..header_bytes_final.len()].copy_from_slice(&header_bytes_final);
        // Append body.
        file.extend_from_slice(&body);

        // Parse + verify.
        let view = RnbMoeView::from_bytes(&file).expect("MoE section parse");
        let parsed = view
            .parse_moe_decode()
            .expect("MoeDecode section should be present")
            .expect("parser ok");

        assert_eq!(parsed.layers.len(), 1);
        let layer = &parsed.layers[0];
        assert_eq!(layer.n_experts, n_experts);
        assert_eq!(layer.d_ff, d_ff);
        assert_eq!(layer.n_embd, n_embd);
        assert_eq!(layer.shared_quant, SHARED_QUANT_NONE);
        assert!(layer.shared_expert.is_none());

        let expert = &layer.experts[0];
        assert_eq!(expert.gate_up_rows.len(), d_ff as usize);
        assert_eq!(expert.down_rows.len(), n_embd as usize);

        // Spot-check multiplier values.
        assert!((expert.gate_up_rows[0].gate_mul - 1.0).abs() < 1e-6);
        assert!((expert.gate_up_rows[0].up_mul - 2.0).abs() < 1e-6);
        assert!((expert.gate_up_rows[255].gate_mul - (1.0 + 255.0 * 0.01)).abs() < 1e-5);

        assert!((expert.down_rows[0].down_mul - 3.0).abs() < 1e-6);
    }
}
