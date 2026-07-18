/// Q5_K int8 NEON debug: compare scalar vs NEON on actual device
fn main() {
    let path = std::path::PathBuf::from(
        std::env::var("RNB_MODEL")
            .unwrap_or_else(|_| "models/gemma-4-E2B-it-Q4_K_M.gguf".to_string()),
    );
    let model = rnb_loader::load_model(&path).unwrap();

    let name =
        std::env::var("RNB_TENSOR").unwrap_or_else(|_| "per_layer_token_embd.weight".to_string());
    let tensor = model.weights.get(&name).unwrap();
    let ggml_type = model.tensor_ggml_types.get(&name).copied().unwrap();
    let float_shape = model.float_shapes.get(&name).unwrap();
    eprintln!("{} type={:?} shape={:?}", name, ggml_type, float_shape);

    let bytes = tensor.as_bytes().unwrap();
    let rows = float_shape[0];
    let cols = float_shape[1];
    let bpr = bytes.len() / rows;

    // Create input
    let input: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();

    // Q8K quantize
    let n_blocks = cols / 256;
    struct Q8K {
        d: f32,
        qs: [i8; 256],
        bsums: [i16; 8],
    }
    let mut q8k: Vec<Q8K> = Vec::new();
    for bi in 0..n_blocks {
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
        q8k.push(Q8K { d, qs, bsums });
    }

    // Test first 5 rows
    for row in 0..5 {
        let rb = &bytes[row * bpr..(row + 1) * bpr];

        // 1. Reference: full f32 dequant + dot
        let mut ref_v = 0.0f32;
        let mut tmp = [0.0f32; 256];
        for bi in 0..n_blocks {
            let chunk = &rb[bi * 176..bi * 176 + 176];
            let block = unsafe { &*(chunk.as_ptr() as *const rnb_cpu::quantize::BlockQ5_K) };
            rnb_cpu::quantize::dequantize_q5_k(block, &mut tmp);
            for i in 0..256 {
                ref_v += tmp[i] * input[bi * 256 + i];
            }
        }

        // 2. Scalar int8 dot
        let mut scalar_v = 0.0f32;
        for bi in 0..n_blocks {
            let boff = bi * 176;
            let d = half::f16::from_bits(u16::from_le_bytes([rb[boff], rb[boff + 1]])).to_f32();
            let dmin =
                half::f16::from_bits(u16::from_le_bytes([rb[boff + 2], rb[boff + 3]])).to_f32();
            let sb = &rb[boff + 4..boff + 16];
            let qh = &rb[boff + 16..boff + 48];
            let qs = &rb[boff + 48..boff + 176];
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..4 {
                sc[j] = sb[j] & 63;
                mn[j] = sb[j + 4] & 63;
            }
            for j in 4..8 {
                sc[j] = (sb[j + 4] & 0xF) | ((sb[j - 4] >> 6) << 4);
                mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
            }
            let q8b = &q8k[bi];
            let mut sumi = 0i32;
            let mut summ = 0i32;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for group in 0..4 {
                let q_off = group * 32;
                let x_off = group * 64;
                let is = group * 2;
                let mut isum_lo = 0i32;
                let mut isum_hi = 0i32;
                for l in 0..32 {
                    let h1: u8 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let h2: u8 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    let w_lo = ((qs[q_off + l] & 0xF) + h1) as i8 as i32;
                    let w_hi = ((qs[q_off + l] >> 4) + h2) as i8 as i32;
                    isum_lo += w_lo * q8b.qs[x_off + l] as i32;
                    isum_hi += w_hi * q8b.qs[x_off + 32 + l] as i32;
                }
                sumi += sc[is] as i32 * isum_lo + sc[is + 1] as i32 * isum_hi;
                summ += mn[is] as i32 * q8b.bsums[group * 2] as i32
                    + mn[is + 1] as i32 * q8b.bsums[group * 2 + 1] as i32;
                u1 <<= 2;
                u2 <<= 2;
            }
            scalar_v += d * q8b.d * sumi as f32 - dmin * q8b.d * summ as f32;
        }

        eprintln!(
            "row {:3}: ref={:10.6} scalar_int8={:10.6} diff={:.2}%",
            row,
            ref_v,
            scalar_v,
            if ref_v.abs() > 1e-8 {
                (ref_v - scalar_v).abs() / ref_v.abs() * 100.0
            } else {
                0.0
            }
        );
    }
}
