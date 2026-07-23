//! 26B-A4B 모델 load + expert tensor hex dump + layer 0 MoE FFN 단일 forward.
//! Pattern B (expert outermost) 검증 이후 실제 MoE 계산 경로 스모크.
//!
//! 사용법: rnb-moe-smoke <gguf-path> [--forward]
use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use rnb_llm::MoeLayerView;
use rnb_loader::load_model;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("usage: rnb-moe-smoke <gguf-path> [--forward]");
        return ExitCode::from(2);
    }
    let path = Path::new(&args[1]);
    let do_forward = args.iter().any(|a| a == "--forward");
    println!("loading {}...", path.display());
    let model = match load_model(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {:?}", e);
            return ExitCode::from(1);
        }
    };
    println!("model loaded");
    println!("  architecture = {:?}", model.metadata.architecture);
    println!("  num_layers = {}", model.metadata.num_layers);
    println!("  hidden_size = {}", model.metadata.hidden_size);
    println!("  expert_count = {}", model.metadata.expert_count);
    println!("  expert_used_count = {}", model.metadata.expert_used_count);
    println!(
        "  expert_feed_forward_length = {}",
        model.metadata.expert_feed_forward_length
    );
    println!(
        "  head_count_kv_per_layer = {:?}",
        model
            .metadata
            .head_count_kv_per_layer
            .as_ref()
            .map(|v| v.len())
    );
    println!("  weights.len() = {}", model.weights.len());

    // 1. Router F32 tensor (blk.0.ffn_gate_inp.weight) 검증 — Pattern A vs B 가장 명확한 곳.
    let router_key = "blk.0.ffn_gate_inp.weight";
    match model.weights.get(router_key) {
        Some(t) => {
            let dtype = model.tensor_ggml_types.get(router_key);
            let offset = model.tensor_file_offsets.get(router_key);
            println!(
                "\n[router] {}: shape={:?} dtype={:?} file_offset={:?}",
                router_key,
                t.shape(),
                dtype,
                offset
            );
            match t.as_bytes() {
                Some(bytes) => {
                    println!("  bytes.len() = {}", bytes.len());
                    print!("  first 64 bytes hex:\n   ");
                    for (i, b) in bytes.iter().take(64).enumerate() {
                        if i > 0 && i % 16 == 0 {
                            print!("\n   ");
                        }
                        print!("{:02x} ", b);
                    }
                    println!();
                    // F32 해석: 첫 16 element (= expert 0 의 첫 16 dim of router weight, 만약 row-major 라면)
                    // shape = [128, 2816]. ggml convention: ne[0]=128 contiguous → 128 expert 가 first 128 elems
                    if bytes.len() >= 128 * 4 {
                        let f32s: &[f32] = unsafe {
                            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, 128)
                        };
                        println!("  hypothesis A (ne[0]=128 contiguous, first 128 = expert[0..128] of dim 0):");
                        print!("    first 8:");
                        for v in &f32s[..8] {
                            print!(" {:.4}", v);
                        }
                        println!();
                        print!("    last 8: ");
                        for v in &f32s[120..128] {
                            print!(" {:.4}", v);
                        }
                        println!();
                    }
                    if bytes.len() >= 2816 * 4 {
                        let f32s: &[f32] = unsafe {
                            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, 2816)
                        };
                        println!("  hypothesis B (ne[1]=2816 contiguous, first 2816 = expert[0] all dims):");
                        print!("    first 8:");
                        for v in &f32s[..8] {
                            print!(" {:.4}", v);
                        }
                        println!();
                        print!("    last 8: ");
                        for v in &f32s[2808..2816] {
                            print!(" {:.4}", v);
                        }
                        println!();
                        // statistics
                        let mean: f32 = f32s.iter().sum::<f32>() / f32s.len() as f32;
                        let var: f32 = f32s.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                            / f32s.len() as f32;
                        println!(
                            "    expert[0] all 2816 dims: mean={:.6} std={:.6}",
                            mean,
                            var.sqrt()
                        );
                    }
                }
                None => println!("  as_bytes() returned None"),
            }
        }
        None => {
            println!("\n[router] NOT FOUND: {}", router_key);
            for k in model.weights.keys() {
                if k.contains("ffn_gate_inp") || k.contains("router") {
                    println!("  candidate: {}", k);
                }
            }
        }
    }

    // 2. Expert weight tensor count 확인 (전 layer)
    let mut gate_up_count = 0;
    let mut down_count = 0;
    let mut router_count = 0;
    for k in model.weights.keys() {
        if k.contains("ffn_gate_up_exps") {
            gate_up_count += 1;
        }
        if k.contains("ffn_down_exps") {
            down_count += 1;
        }
        if k.contains("ffn_gate_inp") {
            router_count += 1;
        }
    }
    println!(
        "\n[expert tensors] gate_up={} down={} router={}",
        gate_up_count, down_count, router_count
    );

    // 3. Q4_K gate_up_exps layout probe (Pattern A innermost vs B outermost)
    // shape=[128,1408,2816]. Q4_K block = 256 elem / 144 bytes.
    // per-expert bytes assuming expert outermost = 1408*2816/256*144 = 2,230,272.
    // expert innermost (ne[0]=128) 이면 block 이 expert 경계를 넘음.
    let gate_up_key = "blk.0.ffn_gate_up_exps.weight";
    if let Some(t) = model.weights.get(gate_up_key) {
        let dtype = model.tensor_ggml_types.get(gate_up_key);
        let offset = model.tensor_file_offsets.get(gate_up_key);
        let shape = model
            .float_shapes
            .get(gate_up_key)
            .cloned()
            .unwrap_or_default();
        println!(
            "\n[gate_up] {}: float_shape={:?} dtype={:?} file_offset={:?}",
            gate_up_key, shape, dtype, offset
        );
        if let Some(bytes) = t.as_bytes() {
            println!(
                "  bytes.len() = {}  (expected 285,474,816 = 272.25 MiB)",
                bytes.len()
            );
            println!("  block 0 (first 144 bytes = one Q4_K block):");
            for i in 0..144 {
                if i % 16 == 0 {
                    print!("\n    {:04x} ", i);
                }
                print!("{:02x} ", bytes[i]);
            }
            println!();
            let per_expert = 1408_usize * 2816 / 256 * 144;
            assert_eq!(per_expert, 2_230_272);
            println!(
                "\n  Pattern B hypothesis: expert 1 starts at byte {}",
                per_expert
            );
            if bytes.len() >= per_expert + 16 {
                print!(
                    "  block at per-expert boundary (offset {}..+16):\n   ",
                    per_expert
                );
                for i in 0..16 {
                    print!("{:02x} ", bytes[per_expert + i]);
                }
                println!();
            }
            // Q4_K block layout: d(f16) | dmin(f16) | scales[12] | qs[128]
            let d0 = half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
            let dmin0 = half::f16::from_le_bytes([bytes[2], bytes[3]]).to_f32();
            println!("  block0: d={:.6e} dmin={:.6e}", d0, dmin0);
            if bytes.len() >= per_expert + 4 {
                let d1 =
                    half::f16::from_le_bytes([bytes[per_expert], bytes[per_expert + 1]]).to_f32();
                let dmin1 =
                    half::f16::from_le_bytes([bytes[per_expert + 2], bytes[per_expert + 3]])
                        .to_f32();
                println!("  blockE: d={:.6e} dmin={:.6e}", d1, dmin1);
            }
            // stride probe (Pattern A hypothesis): ne[0]=128 이면 byte 0 과 byte 4 (F32 스케일 아님 주의)
            // Q4_K 구조상 expert innermost 이면 scale/dmin 이 여러 expert 에 걸쳐 섞여 있음 → d가 거대하거나 0 에 가까운 이상치 기대
            // sample d values across first 16 blocks
            println!("  first 16 blocks' d (f16) values:");
            for b in 0..16 {
                let off = b * 144;
                let d = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
                let dmin = half::f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
                println!("    block {:2}: d={:+.4e} dmin={:+.4e}", b, d, dmin);
            }
        } else {
            println!("  as_bytes() returned None");
        }
    } else {
        println!("\n[gate_up] NOT FOUND");
    }

    // 4. Q5_1 down_exps layout probe
    // shape=[128,2816,704]. Q5_1 block = 32 elem / 24 bytes.
    // per-expert bytes (outermost) = 2816*704/32*24 = 1,486,848.
    let down_key = "blk.0.ffn_down_exps.weight";
    if let Some(t) = model.weights.get(down_key) {
        let dtype = model.tensor_ggml_types.get(down_key);
        let offset = model.tensor_file_offsets.get(down_key);
        let shape = model
            .float_shapes
            .get(down_key)
            .cloned()
            .unwrap_or_default();
        println!(
            "\n[down] {}: float_shape={:?} dtype={:?} file_offset={:?}",
            down_key, shape, dtype, offset
        );
        if let Some(bytes) = t.as_bytes() {
            println!(
                "  bytes.len() = {}  (expected 190,316,544 = 181.50 MiB)",
                bytes.len()
            );
            let per_expert = 2816_usize * 704 / 32 * 24;
            assert_eq!(per_expert, 1_486_848);
            print!("  block 0 (first 24 bytes):\n   ");
            for i in 0..24 {
                print!("{:02x} ", bytes[i]);
            }
            println!();
            if bytes.len() >= per_expert + 24 {
                print!("  block at per-expert boundary ({}..+24):\n   ", per_expert);
                for i in 0..24 {
                    print!("{:02x} ", bytes[per_expert + i]);
                }
                println!();
            }
            // Q5_1 block: d(f16) | m(f16) | qh(4 bytes) | qs(16 bytes) = 24 bytes
            let d0 = half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
            let m0 = half::f16::from_le_bytes([bytes[2], bytes[3]]).to_f32();
            println!("  block0: d={:.6e} m={:.6e}", d0, m0);
            if bytes.len() >= per_expert + 4 {
                let d1 =
                    half::f16::from_le_bytes([bytes[per_expert], bytes[per_expert + 1]]).to_f32();
                let m1 = half::f16::from_le_bytes([bytes[per_expert + 2], bytes[per_expert + 3]])
                    .to_f32();
                println!("  blockE: d={:.6e} m={:.6e}", d1, m1);
            }
            println!("  first 16 blocks' d/m:");
            for b in 0..16 {
                let off = b * 24;
                let d = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
                let m = half::f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
                println!("    block {:2}: d={:+.4e} m={:+.4e}", b, d, m);
            }
        }
    }

    // 5. Single MoE layer forward (layer 0) — opt-in because first call pages in ~455 MiB.
    if do_forward {
        let n_embd = model.metadata.hidden_size;
        let n_ff = model.metadata.expert_feed_forward_length;
        let n_expert = model.metadata.expert_count;
        let n_used = model.metadata.expert_used_count;
        println!(
            "\n[forward] layer 0: n_embd={} n_ff={} n_expert={} n_used={}",
            n_embd, n_ff, n_expert, n_used
        );

        let router_t = match model.weights.get("blk.0.ffn_gate_inp.weight") {
            Some(t) => t,
            None => {
                eprintln!("router missing");
                return ExitCode::from(1);
            }
        };
        let gate_up_t = match model.weights.get("blk.0.ffn_gate_up_exps.weight") {
            Some(t) => t,
            None => {
                eprintln!("gate_up missing");
                return ExitCode::from(1);
            }
        };
        let down_t = match model.weights.get("blk.0.ffn_down_exps.weight") {
            Some(t) => t,
            None => {
                eprintln!("down missing");
                return ExitCode::from(1);
            }
        };

        let router_bytes = router_t.as_bytes().expect("router as_bytes");
        let router_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(router_bytes.as_ptr() as *const f32, router_bytes.len() / 4)
        };
        assert_eq!(router_f32.len(), n_expert * n_embd);
        let gate_up_bytes = gate_up_t.as_bytes().expect("gate_up as_bytes");
        let down_bytes = down_t.as_bytes().expect("down as_bytes");
        let down_scale_name = "blk.0.ffn_down_exps.scale";
        let down_scale = model
            .weights
            .get(down_scale_name)
            .and_then(|t| t.as_bytes())
            .map(|bytes| {
                let ptr = bytes.as_ptr() as *const f32;
                let len = bytes.len() / std::mem::size_of::<f32>();
                unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
            })
            .unwrap_or_else(|| vec![1.0f32; n_expert]);
        let down_quant = *model
            .tensor_ggml_types
            .get("blk.0.ffn_down_exps.weight")
            .unwrap_or(&rnb_loader::GGMLType::Q5_1);

        let view = MoeLayerView {
            router_w: router_f32,
            gate_up_bytes,
            down_bytes,
            down_scale: &down_scale,
            down_quant,
            n_embd,
            n_ff,
            n_expert,
            n_expert_used: n_used,
            layer_idx: None,
        };
        println!(
            "  per_expert gate_up={} bytes, down={} bytes",
            view.per_expert_gate_up_bytes(),
            view.per_expert_down_bytes(),
        );

        // Build a normalized random-ish input h — unit gaussian works fine for smoke.
        let mut h = vec![0f32; n_embd];
        let mut state = 0x1234567u64;
        for v in h.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (state >> 32) as u32;
            // cheap approximate Gaussian via central-limit of 4 uniforms
            let u = (r as f32) / (u32::MAX as f32);
            *v = (u - 0.5) * 0.5;
        }
        let norm = (h.iter().map(|x| x * x).sum::<f32>() / n_embd as f32).sqrt();
        for v in h.iter_mut() {
            *v /= norm;
        }

        let mut out = vec![0f32; n_embd];
        let t0 = Instant::now();
        view.forward(&h, &mut out);
        let elapsed = t0.elapsed();

        let mean = out.iter().sum::<f32>() / n_embd as f32;
        let var = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n_embd as f32;
        let std = var.sqrt();
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let nan = out.iter().filter(|x| x.is_nan()).count();
        let inf = out.iter().filter(|x| x.is_infinite()).count();
        println!(
            "  elapsed = {:.3} ms  (single layer, {} top-{} experts)",
            elapsed.as_secs_f64() * 1000.0,
            n_expert,
            n_used
        );
        println!(
            "  out stats: mean={:+.6} std={:.6} min={:+.6} max={:+.6} nan={} inf={}",
            mean, std, min, max, nan, inf
        );
        let est_total_ms = elapsed.as_secs_f64() * 1000.0 * model.metadata.num_layers as f64;
        let est_tok_per_s = 1000.0 / est_total_ms;
        println!(
            "  extrapolated all-layer decode cost: {:.1} ms → {:.2} tok/s (ignoring attention, PLE, etc.)",
            est_total_ms, est_tok_per_s
        );
    }

    ExitCode::SUCCESS
}
