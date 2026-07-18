use half::f16;
use rnb_cpu::gemm::{
    pack_q4k::pack_q4k,
    pack_q5k::pack_q5k,
    pack_q6k::{pack_q6k, unpack_q6k},
    tile_q4k::gemm_q4k_packed,
    tile_q5k::gemm_q5k_packed,
    tile_q6k::gemm_q6k_packed,
};
use rnb_loader::packed::PackedModel;
use rnb_loader::GGMLType;

fn quantize_q8k(x: &[f32]) -> (Vec<i8>, f32, [i16; 8]) {
    let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inv_d = 1.0 / d;

    let mut qs = vec![0i8; x.len()];
    let mut bsums = [0i16; 8];
    for (i, &v) in x.iter().enumerate() {
        let q = (v * inv_d).round().clamp(-128.0, 127.0) as i8;
        qs[i] = q;
        bsums[i / 32] += q as i16;
    }

    (qs, d, bsums)
}

fn unpack_q4k_unsigned(qs: &[u8; 128], out: &mut [u8; 256]) {
    let mut q_off = 0usize;
    let mut y_off = 0usize;
    for _ in 0..4 {
        for l in 0..32 {
            out[y_off + l] = qs[q_off + l] & 0x0F;
        }
        for l in 0..32 {
            out[y_off + 32 + l] = qs[q_off + l] >> 4;
        }
        q_off += 32;
        y_off += 64;
    }
}

fn unpack_q5k_unsigned(qs: &[u8; 128], qh: &[u8; 32], out: &mut [u8; 256]) {
    for g in 0..4usize {
        let group_out_base = g * 64;
        let group_qs_base = g * 32;

        for l in 0..32usize {
            let low = qs[group_qs_base + l] & 0x0F;
            let high_bit = (qh[l] >> (2 * g)) & 1;
            out[group_out_base + l] = low | (high_bit << 4);
        }

        for l in 32..64usize {
            let low = qs[group_qs_base + (l - 32)] >> 4;
            let high_bit = (qh[l - 32] >> (2 * g + 1)) & 1;
            out[group_out_base + l] = low | (high_bit << 4);
        }
    }
}

fn exact_q4k_q8_ref(
    src: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    rows: usize,
    cols: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * rows];
    for s in 0..seq_len {
        for row in 0..rows {
            let mut acc = 0.0f32;
            for bi in 0..cols {
                let src_off = (row * cols + bi) * 144;
                let block = &src[src_off..src_off + 144];
                let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                let scales_12: &[u8; 12] = block[4..16].try_into().unwrap();
                let qs_raw: &[u8; 128] = block[16..144].try_into().unwrap();
                let (sc_raw, mn_raw) = rnb_cpu::gemm::pack_q4k::decode_q4k_scales_raw(scales_12);

                let mut w_unsigned = [0u8; 256];
                unpack_q4k_unsigned(qs_raw, &mut w_unsigned);

                let x_off = (s * cols + bi) * 256;
                let x_qs = &input_qs[x_off..x_off + 256];
                let x_d = input_d[s * cols + bi];
                let bs_off = (s * cols + bi) * 8;
                let x_bsums = &input_bsums[bs_off..bs_off + 8];

                let mut sumi = 0i32;
                let mut summ = 0i32;
                for sb in 0..8usize {
                    let mut dot = 0i32;
                    for k in 0..32usize {
                        let idx = sb * 32 + k;
                        dot += w_unsigned[idx] as i32 * x_qs[idx] as i32;
                    }
                    sumi += sc_raw[sb] as i32 * dot;
                    summ += mn_raw[sb] as i32 * x_bsums[sb] as i32;
                }

                acc += x_d * (d * sumi as f32 - dmin * summ as f32);
            }
            out[s * rows + row] = acc;
        }
    }
    out
}

fn exact_q5k_q8_ref(
    src: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    rows: usize,
    cols: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * rows];
    for s in 0..seq_len {
        for row in 0..rows {
            let mut acc = 0.0f32;
            for bi in 0..cols {
                let src_off = (row * cols + bi) * 176;
                let block = &src[src_off..src_off + 176];
                let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                let scales_12: &[u8; 12] = block[4..16].try_into().unwrap();
                let qh_raw: &[u8; 32] = block[16..48].try_into().unwrap();
                let qs_raw: &[u8; 128] = block[48..176].try_into().unwrap();
                let (sc_raw, mn_raw) = rnb_cpu::gemm::pack_q4k::decode_q4k_scales_raw(scales_12);

                let mut w_unsigned = [0u8; 256];
                unpack_q5k_unsigned(qs_raw, qh_raw, &mut w_unsigned);

                let x_off = (s * cols + bi) * 256;
                let x_qs = &input_qs[x_off..x_off + 256];
                let x_d = input_d[s * cols + bi];
                let bs_off = (s * cols + bi) * 8;
                let x_bsums = &input_bsums[bs_off..bs_off + 8];

                let mut sumi = 0i32;
                let mut summ = 0i32;
                for sb in 0..8usize {
                    let mut dot = 0i32;
                    for k in 0..32usize {
                        let idx = sb * 32 + k;
                        dot += w_unsigned[idx] as i32 * x_qs[idx] as i32;
                    }
                    sumi += sc_raw[sb] as i32 * dot;
                    summ += mn_raw[sb] as i32 * x_bsums[sb] as i32;
                }

                acc += x_d * (d * sumi as f32 - dmin * summ as f32);
            }
            out[s * rows + row] = acc;
        }
    }
    out
}

fn exact_q6k_q8_ref(
    src: &[u8],
    input_qs: &[i8],
    input_d: &[f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * rows];
    for s in 0..seq_len {
        for row in 0..rows {
            let mut acc = 0.0f32;
            for bi in 0..cols {
                let src_off = (row * cols + bi) * 210;
                let block = &src[src_off..src_off + 210];
                let ql: &[u8; 128] = block[0..128].try_into().unwrap();
                let qh: &[u8; 64] = block[128..192].try_into().unwrap();
                let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

                let mut w_signed = [0i8; 256];
                unpack_q6k(ql, qh, &mut w_signed);

                let x_off = (s * cols + bi) * 256;
                let x_qs = &input_qs[x_off..x_off + 256];
                let x_d = input_d[s * cols + bi];

                let mut sumi = 0i32;
                for sb in 0..16usize {
                    let sc_raw = block[192 + sb] as i8 as i32;
                    let mut dot = 0i32;
                    for k in 0..16usize {
                        let idx = sb * 16 + k;
                        dot += w_signed[idx] as i32 * x_qs[idx] as i32;
                    }
                    sumi += sc_raw * dot;
                }

                acc += x_d * d * sumi as f32;
            }
            out[s * rows + row] = acc;
        }
    }
    out
}

fn main() {
    let model_path = std::env::var("RNB_MODEL")
        .unwrap_or_else(|_| "models/Qwen3.5-0.8B-Q4_K_M.gguf".to_string());
    let weight_name =
        std::env::var("RNB_WEIGHT").unwrap_or_else(|_| "blk.0.ffn_down.weight".to_string());
    let seq_len = std::env::var("RNB_SEQ_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(4);

    let model = rnb_loader::load_model(&std::path::PathBuf::from(&model_path)).unwrap();
    let tensor = model.weights.get(&weight_name).unwrap();
    let ggml_type = model.tensor_ggml_types.get(&weight_name).copied().unwrap();
    let shape = model.float_shapes.get(&weight_name).unwrap();

    let src = tensor.as_bytes().unwrap();
    let rows = shape[0];
    let input_dim = shape[1];
    let cols = input_dim / 256;

    let total_tokens = seq_len * cols;
    let mut input_qs = vec![0i8; total_tokens * 256];
    let mut input_d = vec![0.0f32; total_tokens];
    let mut input_bsums = vec![0i16; total_tokens * 8];
    for t in 0..total_tokens {
        let mut x = [0.0f32; 256];
        for (k, v) in x.iter_mut().enumerate() {
            let idx = t * 256 + k;
            *v = ((idx as f32) * 0.017).sin() * 0.75 + ((idx as f32) * 0.031).cos() * 0.25;
        }
        let (qs, d, bsums) = quantize_q8k(&x);
        input_qs[t * 256..(t + 1) * 256].copy_from_slice(&qs);
        input_d[t] = d;
        input_bsums[t * 8..(t + 1) * 8].copy_from_slice(&bsums);
    }

    let (packed, packed_out, exact_out) = match ggml_type {
        GGMLType::Q4_K => {
            let packed = pack_q4k(src, rows, cols);
            let mut out = vec![0.0f32; seq_len * rows];
            gemm_q4k_packed(
                &packed,
                &input_qs,
                &input_d,
                &input_bsums,
                &mut out,
                rows,
                cols,
                seq_len,
            );
            let exact =
                exact_q4k_q8_ref(src, &input_qs, &input_d, &input_bsums, rows, cols, seq_len);
            (packed, out, exact)
        }
        GGMLType::Q5_K => {
            let packed = pack_q5k(src, rows, cols);
            let mut out = vec![0.0f32; seq_len * rows];
            gemm_q5k_packed(
                &packed,
                &input_qs,
                &input_d,
                &input_bsums,
                &mut out,
                rows,
                cols,
                seq_len,
            );
            let exact =
                exact_q5k_q8_ref(src, &input_qs, &input_d, &input_bsums, rows, cols, seq_len);
            (packed, out, exact)
        }
        GGMLType::Q6_K => {
            let packed = pack_q6k(src, rows, cols);
            let mut out = vec![0.0f32; seq_len * rows];
            gemm_q6k_packed(&packed, &input_qs, &input_d, &mut out, rows, cols, seq_len);
            let exact = exact_q6k_q8_ref(src, &input_qs, &input_d, rows, cols, seq_len);
            (packed, out, exact)
        }
        other => panic!(
            "unsupported ggml type for rnb-q6k-packed-backend-check: {:?}",
            other
        ),
    };

    let backend = std::env::var("RNB_Q6K_PACKED_BACKEND").unwrap_or_else(|_| "auto".to_string());
    eprintln!(
        "model={model_path} weight={weight_name} type={ggml_type:?} backend={backend} rows={rows} input_dim={input_dim} cols={cols} seq_len={seq_len}"
    );

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut offenders: Vec<(f32, usize, usize, f32, f32)> = Vec::new();
    for s in 0..seq_len {
        for row in 0..rows {
            let got = packed_out[s * rows + row];
            let exp = exact_out[s * rows + row];
            let abs_err = (got - exp).abs();
            let rel_err = if exp.abs() > 1e-6 {
                abs_err / exp.abs()
            } else {
                abs_err
            };
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);
            offenders.push((abs_err, s, row, got, exp));
        }
    }
    offenders.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    eprintln!("max_abs_err={max_abs_err:.8} max_rel_err={max_rel_err:.8}");
    for (rank, (abs_err, s, row, got, exp)) in offenders.into_iter().take(8).enumerate() {
        let rel_err = if exp.abs() > 1e-6 {
            abs_err / exp.abs()
        } else {
            abs_err
        };
        eprintln!(
            "  #{rank}: s={s} row={row} got={got:.8} ref={exp:.8} abs_err={abs_err:.8} rel_err={rel_err:.8}"
        );
    }

    let packed_model_path = std::env::var("RNB_PACKED_MODEL").ok().or_else(|| {
        let candidate = std::path::PathBuf::from(&model_path).with_extension("rnb");
        candidate
            .exists()
            .then(|| candidate.to_string_lossy().into_owned())
    });

    if let Some(packed_model_path) = packed_model_path {
        let packed_model = PackedModel::open(std::path::Path::new(&packed_model_path)).unwrap();
        let disk_weight = packed_model.get_weight(&weight_name).unwrap_or_else(|| {
            panic!("missing packed weight {weight_name} in {packed_model_path}")
        });
        let disk_bytes = disk_weight.data();

        eprintln!(
            "packed_model={} disk_len={} fresh_len={}",
            packed_model_path,
            disk_bytes.len(),
            packed.len()
        );

        let equal = disk_bytes == packed.as_slice();
        eprintln!("packed_bytes_equal={equal}");
        if !equal {
            let first_diff = disk_bytes
                .iter()
                .zip(packed.iter())
                .enumerate()
                .find(|(_, (a, b))| a != b);
            if let Some((idx, (disk, fresh))) = first_diff {
                eprintln!("first_packed_diff idx={idx} disk={disk} fresh={fresh}");
            }
        }

        let mut disk_out = vec![0.0f32; seq_len * rows];
        match ggml_type {
            GGMLType::Q4_K => gemm_q4k_packed(
                disk_bytes,
                &input_qs,
                &input_d,
                &input_bsums,
                &mut disk_out,
                rows,
                cols,
                seq_len,
            ),
            GGMLType::Q5_K => gemm_q5k_packed(
                disk_bytes,
                &input_qs,
                &input_d,
                &input_bsums,
                &mut disk_out,
                rows,
                cols,
                seq_len,
            ),
            GGMLType::Q6_K => gemm_q6k_packed(
                disk_bytes,
                &input_qs,
                &input_d,
                &mut disk_out,
                rows,
                cols,
                seq_len,
            ),
            _ => unreachable!(),
        }

        let mut disk_max_abs_err = 0.0f32;
        let mut disk_max_rel_err = 0.0f32;
        for s in 0..seq_len {
            for row in 0..rows {
                let got = disk_out[s * rows + row];
                let exp = exact_out[s * rows + row];
                let abs_err = (got - exp).abs();
                let rel_err = if exp.abs() > 1e-6 {
                    abs_err / exp.abs()
                } else {
                    abs_err
                };
                disk_max_abs_err = disk_max_abs_err.max(abs_err);
                disk_max_rel_err = disk_max_rel_err.max(rel_err);
            }
        }
        eprintln!(
            "disk_packed max_abs_err={disk_max_abs_err:.8} max_rel_err={disk_max_rel_err:.8}"
        );
    }
}
