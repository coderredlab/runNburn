/// Q6_K int8 dot accuracy test: compare with f32 dequant dot
fn main() {
    let path = std::path::PathBuf::from("models/Qwen3.5-0.8B-Q4_K_M.gguf");
    let model = rnb_loader::load_model(&path).unwrap();

    // Find a Q6_K weight (ffn_down)
    let name = "blk.0.ffn_down.weight";
    let tensor = model.weights.get(name).unwrap();
    let ggml_type = model.tensor_ggml_types.get(name).copied().unwrap();
    let float_shape = model.float_shapes.get(name).unwrap();
    eprintln!(
        "Weight: {} type={:?} shape={:?}",
        name, ggml_type, float_shape
    );

    let bytes = tensor.as_bytes().unwrap();
    let rows = float_shape[0]; // 1024
    let cols = float_shape[1]; // 3584
    let bytes_per_row = bytes.len() / rows;
    eprintln!(
        "rows={} cols={} bytes_per_row={}",
        rows, cols, bytes_per_row
    );

    // Create dummy input
    let input: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();

    // 1. F32 dequant dot (reference)
    let row0 = &bytes[0..bytes_per_row];
    let n_blocks = cols / 256;

    let mut ref_acc = 0.0f32;
    let mut tmp = [0.0f32; 256];
    for bi in 0..n_blocks {
        let bstart = bi * 210;
        let chunk = &row0[bstart..bstart + 210];
        let block = unsafe { &*(chunk.as_ptr() as *const rnb_cpu::quantize::BlockQ6_K) };
        rnb_cpu::quantize::dequantize_q6_k(block, &mut tmp);
        for i in 0..256 {
            ref_acc += tmp[i] * input[bi * 256 + i];
        }
    }
    eprintln!("Reference f32 dot: {:.6}", ref_acc);

    // 2. Q8K quantized input
    let n_blocks_q8 = cols / 256;
    let mut q8k_blocks = Vec::new();
    for bi in 0..n_blocks_q8 {
        let chunk = &input[bi * 256..(bi + 1) * 256];
        let mut amax = 0.0f32;
        for &x in chunk {
            amax = amax.max(x.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 8];
        for i in 0..256 {
            let q = (chunk[i] * id).round().clamp(-128.0, 127.0) as i8;
            qs[i] = q;
            bsums[i / 32] += q as i16;
        }
        q8k_blocks.push((d, qs, bsums));
    }

    // 3. Manual Q6_K × Q8K int8 dot (reproducing the NEON function logic)
    let mut int8_acc = 0.0f32;
    for bi in 0..n_blocks {
        let boff = bi * 210;
        let ql = &row0[boff..boff + 128];
        let qh = &row0[boff + 128..boff + 192];
        let scales = &row0[boff + 192..boff + 208];
        let d =
            half::f16::from_bits(u16::from_le_bytes([row0[boff + 208], row0[boff + 209]])).to_f32();
        let (q8d, ref q8qs, _) = &q8k_blocks[bi];

        let mut sumi = 0i32;
        for n in 0..2 {
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;
            let x_base = n * 128;

            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    ((ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4))
                    as i32
                    - 32;
                let q3 = ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i32 - 32;

                let sc1 = scales[sc_base + is] as i8 as i32;
                let sc2 = scales[sc_base + is + 2] as i8 as i32;
                let sc3 = scales[sc_base + is + 4] as i8 as i32;
                let sc4 = scales[sc_base + is + 6] as i8 as i32;

                sumi += sc1 * q1 * q8qs[x_base + l] as i32;
                sumi += sc2 * q2 * q8qs[x_base + l + 32] as i32;
                sumi += sc3 * q3 * q8qs[x_base + l + 64] as i32;
                sumi += sc4 * q4 * q8qs[x_base + l + 96] as i32;
            }
        }
        int8_acc += d * q8d * sumi as f32;
    }
    eprintln!("Int8 Q6K dot:     {:.6}", int8_acc);
    eprintln!(
        "Diff:             {:.6} ({:.2}%)",
        (ref_acc - int8_acc).abs(),
        (ref_acc - int8_acc).abs() / ref_acc.abs() * 100.0
    );

    // Test Q5_K
    let q5k_name = "blk.0.attn_qkv.weight";
    if let Some(q5t) = model.weights.get(q5k_name) {
        let q5_type = model.tensor_ggml_types.get(q5k_name).copied().unwrap();
        let q5_shape = model.float_shapes.get(q5k_name).unwrap();
        eprintln!(
            "\n=== Q5_K test: {} type={:?} shape={:?} ===",
            q5k_name, q5_type, q5_shape
        );
        let q5_bytes = q5t.as_bytes().unwrap();
        let q5_rows = q5_shape[0];
        let q5_cols = q5_shape[1];
        let q5_bpr = q5_bytes.len() / q5_rows;
        let q5_input: Vec<f32> = (0..q5_cols)
            .map(|i| ((i as f32 * 0.01).sin()) * 0.1)
            .collect();

        // Quantize input
        let q5_n_blocks = q5_cols / 256;
        let mut q5_q8k = Vec::new();
        for bi in 0..q5_n_blocks {
            let chunk = &q5_input[bi * 256..(bi + 1) * 256];
            let mut amax = 0.0f32;
            for &x in chunk {
                amax = amax.max(x.abs());
            }
            let dd = amax / 127.0;
            let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            let mut qs_arr = [0i8; 256];
            let mut bsums_arr = [0i16; 8];
            for i in 0..256 {
                let q = (chunk[i] * id).round().clamp(-128.0, 127.0) as i8;
                qs_arr[i] = q;
                bsums_arr[i / 32] += q as i16;
            }
            q5_q8k.push((dd, qs_arr, bsums_arr));
        }

        // Reference: dequant + dot
        let rb = &q5_bytes[0..q5_bpr];
        let mut ref_v = 0.0f32;
        let mut tmp5 = [0.0f32; 256];
        for bi in 0..q5_n_blocks {
            let bstart = bi * 176;
            let chunk = &rb[bstart..bstart + 176];
            let block = unsafe { &*(chunk.as_ptr() as *const rnb_cpu::quantize::BlockQ5_K) };
            rnb_cpu::quantize::dequantize_q5_k(block, &mut tmp5);
            for i in 0..256 {
                ref_v += tmp5[i] * q5_input[bi * 256 + i];
            }
        }

        // Int8: scalar version of Q5_K dot
        let mut int8_v = 0.0f32;
        for bi in 0..q5_n_blocks {
            let boff = bi * 176;
            let d5 = half::f16::from_bits(u16::from_le_bytes([rb[boff], rb[boff + 1]])).to_f32();
            let dmin5 =
                half::f16::from_bits(u16::from_le_bytes([rb[boff + 2], rb[boff + 3]])).to_f32();
            let sb = &rb[boff + 4..boff + 16];
            let qh5 = &rb[boff + 16..boff + 48];
            let qs5 = &rb[boff + 48..boff + 176];
            let mut sc5 = [0u8; 8];
            let mut mn5 = [0u8; 8];
            for j in 0..4 {
                sc5[j] = sb[j] & 63;
                mn5[j] = sb[j + 4] & 63;
            }
            for j in 4..8 {
                sc5[j] = (sb[j + 4] & 0xF) | ((sb[j - 4] >> 6) << 4);
                mn5[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
            }
            let (q8d, ref q8qs, ref q8bs) = q5_q8k[bi];
            let mut sumi5 = 0i32;
            let mut summ5 = 0i32;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for group in 0..4 {
                let q_off = group * 32;
                let x_off = group * 64;
                let is5 = group * 2;
                let mut isum_lo = 0i32;
                let mut isum_hi = 0i32;
                for l in 0..32 {
                    let h1 = if qh5[l] & u1 != 0 { 16u8 } else { 0 };
                    let h2 = if qh5[l] & u2 != 0 { 16u8 } else { 0 };
                    let w_lo = ((qs5[q_off + l] & 0xF) + h1) as i8 as i32;
                    let w_hi = ((qs5[q_off + l] >> 4) + h2) as i8 as i32;
                    isum_lo += w_lo * q8qs[x_off + l] as i32;
                    isum_hi += w_hi * q8qs[x_off + 32 + l] as i32;
                }
                sumi5 += sc5[is5] as i32 * isum_lo + sc5[is5 + 1] as i32 * isum_hi;
                summ5 += mn5[is5] as i32 * q8bs[group * 2] as i32
                    + mn5[is5 + 1] as i32 * q8bs[group * 2 + 1] as i32;
                u1 <<= 2;
                u2 <<= 2;
            }
            int8_v += d5 * q8d * sumi5 as f32 - dmin5 * q8d * summ5 as f32;
        }

        eprintln!(
            "  Q5_K ref={:.6} int8={:.6} diff={:.2}%",
            ref_v,
            int8_v,
            if ref_v.abs() > 1e-8 {
                (ref_v - int8_v).abs() / ref_v.abs() * 100.0
            } else {
                0.0
            }
        );
    }

    // Test multiple rows
    eprintln!("\n=== Multi-row test (first 10 rows) ===");
    for row in 0..10.min(rows) {
        let rb = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];

        // Reference
        let mut ref_v = 0.0f32;
        for bi in 0..n_blocks {
            let bstart = bi * 210;
            let chunk = &rb[bstart..bstart + 210];
            let block = unsafe { &*(chunk.as_ptr() as *const rnb_cpu::quantize::BlockQ6_K) };
            rnb_cpu::quantize::dequantize_q6_k(block, &mut tmp);
            for i in 0..256 {
                ref_v += tmp[i] * input[bi * 256 + i];
            }
        }

        // Int8
        let mut int8_v = 0.0f32;
        for bi in 0..n_blocks {
            let boff = bi * 210;
            let ql = &rb[boff..boff + 128];
            let qh = &rb[boff + 128..boff + 192];
            let scales = &rb[boff + 192..boff + 208];
            let d =
                half::f16::from_bits(u16::from_le_bytes([rb[boff + 208], rb[boff + 209]])).to_f32();
            let (q8d, ref q8qs, _) = &q8k_blocks[bi];
            let mut sumi = 0i32;
            for n in 0..2 {
                let ql_base = n * 64;
                let qh_base = n * 32;
                let sc_base = n * 8;
                let x_base = n * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4))
                        as i32
                        - 32;
                    let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4))
                        as i32
                        - 32;
                    let q3 =
                        ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4))
                        as i32
                        - 32;
                    let sc1 = scales[sc_base + is] as i8 as i32;
                    let sc2 = scales[sc_base + is + 2] as i8 as i32;
                    let sc3 = scales[sc_base + is + 4] as i8 as i32;
                    let sc4 = scales[sc_base + is + 6] as i8 as i32;
                    sumi += sc1 * q1 * q8qs[x_base + l] as i32;
                    sumi += sc2 * q2 * q8qs[x_base + l + 32] as i32;
                    sumi += sc3 * q3 * q8qs[x_base + l + 64] as i32;
                    sumi += sc4 * q4 * q8qs[x_base + l + 96] as i32;
                }
            }
            int8_v += d * q8d * sumi as f32;
        }

        let diff_pct = if ref_v.abs() > 1e-8 {
            (ref_v - int8_v).abs() / ref_v.abs() * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  row {:4}: ref={:10.6} int8={:10.6} diff={:.2}%",
            row, ref_v, int8_v, diff_pct
        );
    }
}
