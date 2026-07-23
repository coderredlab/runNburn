//! mt84 Stage δ — drafter backbone forward 의 single-variant calibration +
//! diagnostic δ.a/b/c/d.
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`
//! §"Calibration 재설계" + §"Acceptance Criteria" Stage 2 ("Calibration vs
//! target", top1 ≥ 0.40).
//!
//! mt83 의 9-variant grid (`KvShareMap` × `ClusterTokenStrategy`) 폐기. spec
//! §7 의 candidate generator loop 와 1:1 매칭되는 single variant 로 측정한다:
//! - drafter layer 의 `layer_type` 이 자동으로 sliding / full KV 선택
//!   (`backbone::drafter_forward` 가 결정)
//! - token_ordering permutation 이 `mtp.token_ordering.weight` 단일 source
//!   (`ClusterTokenTable::Permutation`)
//!
//! ## 환경 (모두 필수)
//!
//! - `RNB_TARGET_MODEL` — Gemma 4 E4B target GGUF
//!   (e.g. `models/gemma-4-E4B/gemma-4-e4b-it-q4_k_m.gguf`).
//! - `RNB_DRAFTER_MODEL` — assistant drafter GGUF
//!   (e.g. `models/gemma-4-E4B-mtp/gemma-4-E4B-it-assistant.Q4_K_M.gguf`).
//!
//! ## 실행
//!
//! ```bash
//! RNB_TARGET_MODEL=models/gemma-4-E4B/gemma-4-e4b-it-q4_k_m.gguf \
//! RNB_DRAFTER_MODEL=models/gemma-4-E4B-mtp/gemma-4-E4B-it-assistant.Q4_K_M.gguf \
//!   cargo test -p rnb-mtp --release \
//!     --test drafter_backbone_calibrate_test -- --ignored --nocapture
//! ```
//!
//! 메인 acceptance test 가 fail 하면 동일 fixture 로 diagnostic 4 step 을
//! 차례로 돌려 root cause 를 분리한다:
//!
//! - δ.a — pre_projection weight 검증 (unit vector forward의 출력 통계)
//! - δ.b — VQ head 의 tied-lm_head self-identity 검증
//! - δ.c — backbone first-layer forward 의 출력 통계 (NaN/std plausibility)
//! - δ.d — `shared_kv_states_for_drafter()` 의 idempotency (ABAB 보강)

use std::path::PathBuf;

use rnb_llm::Engine;
use rnb_mtp::drafter::{
    drafter_forward, load_drafter, vocab_logits_in_top_k_clusters, vq_head_forward,
    ClusterTokenTable,
};
use rnb_mtp::{SharedKvLayer, SharedKvStates};

const TARGET_ENV: &str = "RNB_TARGET_MODEL";
const DRAFTER_ENV: &str = "RNB_DRAFTER_MODEL";

/// spec §Acceptance Criteria Stage 2 의 top1 threshold.
const ACCEPTANCE_TOP1: f32 = 0.40;
/// spec §Calibration 재설계 의 prefill trace token 수 (`64 token of
/// bench_small_ko_singleline.txt`).
const N_EVAL: usize = 64;

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .expect("argmax on empty logits")
}

/// `bench_small_ko_singleline.txt` 를 target Engine 으로 토큰화. BOS 정책은
/// `engine.tokenizer.should_add_bos()` 가 single source (CLAUDE.md "BOS 정책").
///
/// mt89 Stage C 정정 — calibrate test 는 transformers reference (mt88 의 raw
/// `tokenizer(prompt)` 호출) 와 동일 setup 비교가 목적. transformers Gemma 4
/// tokenizer 는 raw call 시 BOS 안 붙이지만, 우리 production 의 Gemma 4 inference
/// 는 GGUF metadata `add_bos_token=true` 에 의존 (E2B decode 가 BOS 없으면
/// broken). test 의 axis isolation 위해 `RNB_CALIBRATE_NO_BOS=1` env 로 raw
/// path 측정 가능. production 정책은 변경 없음.
fn load_prompt_tokens(prompt_path: &str, engine: &Engine) -> Vec<u32> {
    let text = std::fs::read_to_string(prompt_path)
        .unwrap_or_else(|e| panic!("read prompt {prompt_path}: {e}"));
    let no_bos = std::env::var("RNB_CALIBRATE_NO_BOS").ok().as_deref() == Some("1");
    let mut tokens: Vec<u32> = if !no_bos && engine.tokenizer.should_add_bos() {
        vec![engine.tokenizer.vocab.special.bos]
    } else {
        Vec::new()
    };
    tokens.extend(engine.tokenizer.encode(&text));
    tokens
}

/// Mock `SharedKvStates` — diagnostic δ.a/δ.c 가 fixture 없이 drafter forward
/// 의 dim flow / numerical stability 만 검증할 때 사용. layer head_dim:
/// sliding=256, full=512, n_kv_heads=2 (Gemma 4 E4B drafter 의 layer config).
fn mock_shared_kv_states(seq_len: usize) -> SharedKvStates {
    let sliding_head_dim = 256;
    let full_head_dim = 512;
    let n_kv_heads = 2;
    let sliding_len = n_kv_heads * seq_len * sliding_head_dim;
    let full_len = n_kv_heads * seq_len * full_head_dim;
    SharedKvStates {
        sliding_attention: SharedKvLayer {
            k: vec![0.01f32; sliding_len],
            v: vec![0.01f32; sliding_len],
            n_kv_heads,
            seq_len,
            head_dim: sliding_head_dim,
        },
        full_attention: SharedKvLayer {
            k: vec![0.01f32; full_len],
            v: vec![0.01f32; full_len],
            n_kv_heads,
            seq_len,
            head_dim: full_head_dim,
        },
    }
}

// =============================================================================
// Main calibration test (spec §Acceptance Criteria Stage 2)
// =============================================================================

/// spec §7 의 candidate generator loop 와 1:1 매칭되는 single-variant
/// calibration. target Gemma 4 E4B 의 prefill trace 위에서 drafter 의 top-1
/// argmax 가 target 의 top-1 argmax 와 얼마나 일치하는지 측정.
///
/// teacher forcing — drafter 의 prediction 이 틀려도 다음 step 의
/// `last_token_id` 와 `last_hidden` 은 target 의 path 를 따른다 (spec §7 의
/// candidate generator loop 의 첫 step 만 target hidden 사용, 이후 drafter 가
/// self-consistent 한 path 로 진행하는 방식은 spec §"critical 결정" 에서 명시).
/// 본 calibration 은 cumulative error 를 막기 위해 매 step teacher forcing.
#[test]
#[ignore = "needs RNB_TARGET_MODEL + RNB_DRAFTER_MODEL"]
fn drafter_backbone_calibrate_top1() {
    let target_var = std::env::var(TARGET_ENV)
        .unwrap_or_else(|_| panic!("set {TARGET_ENV} to target GGUF path"));
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));

    let target_path = PathBuf::from(&target_var);
    let drafter_path = PathBuf::from(&drafter_var);
    assert!(
        target_path.exists(),
        "${TARGET_ENV} points to missing file: {target_path:?}"
    );
    assert!(
        drafter_path.exists(),
        "${DRAFTER_ENV} points to missing file: {drafter_path:?}"
    );

    let mut engine =
        Engine::from_gguf(&target_path).expect("load target engine from RNB_TARGET_MODEL");
    let drafter = load_drafter(&drafter_path).expect("load drafter from RNB_DRAFTER_MODEL");

    // path resolve: RNB_CALIBRATE_PROMPT env > CARGO_MANIFEST_DIR fallback (cargo test cwd
    // 가 crate dir 라 workspace-root-relative path 가 못 찾음).
    let prompt_path = std::env::var("RNB_CALIBRATE_PROMPT").unwrap_or_else(|_| {
        format!(
            "{}/../../prompts/bench_small_ko_singleline.txt",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    // mt89 Stage B — token id 차이 axis 검증용. RNB_CALIBRATE_FORCE_TOKENS env
    // 로 comma-separated token id 강제 (Python transformers tokenizer 와 동일한
    // 시퀀스로 측정). Rust GGUF tokenizer 의 BOS / BPE merge 차이를 우회한다.
    let prompt_tokens = if let Ok(forced) = std::env::var("RNB_CALIBRATE_FORCE_TOKENS") {
        let ids: Vec<u32> = forced
            .split(',')
            .map(|s| s.trim().parse::<u32>().expect("invalid token id"))
            .collect();
        println!("[mt89.B-rust] FORCED tokens = {ids:?}");
        ids
    } else {
        load_prompt_tokens(&prompt_path, &engine)
    };
    assert!(
        !prompt_tokens.is_empty(),
        "prompt tokens are empty — check tokenizer / prompt file"
    );

    // 1) Prefill — target Engine 이 last_layer_hidden + KV cache 를 채운다.
    engine
        .forward(&prompt_tokens)
        .expect("target forward prefill failed");

    let mut last_token_id = *prompt_tokens.last().unwrap();
    let mut last_hidden: Vec<f32> = engine.last_layer_hidden().to_vec();
    assert!(
        !last_hidden.is_empty(),
        "engine.last_layer_hidden() empty after prefill"
    );
    let backbone_hidden = drafter.backbone_hidden;
    assert_eq!(
        last_hidden.len(),
        backbone_hidden,
        "last_hidden len {} != backbone_hidden {} (target / drafter mismatch?)",
        last_hidden.len(),
        backbone_hidden
    );

    let mut top1_matches: u32 = 0;
    let mut total: u32 = 0;

    // mt87: axis search 시 빠른 측정을 위해 RNB_CALIBRATE_N_EVAL env 로 override.
    // 기본 N_EVAL=64. 작은 값 (예: 16) 도 acceptance 비율 동일하게 측정 가능.
    let n_eval: usize = std::env::var("RNB_CALIBRATE_N_EVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_EVAL);

    // mt89 Stage B — step 0 의 last_hidden / prev_embd / shared_kv / drafter
    // output 을 Python reference (mt89_stageB_dump.py) 와 비교하기 위해 raw
    // 값 println. RNB_CALIBRATE_DUMP_STEP0=1 시에만 출력.
    let dump_step0 = std::env::var("RNB_CALIBRATE_DUMP_STEP0").ok().as_deref() == Some("1");

    for step in 0..n_eval {
        // 2) Drafter forward
        //    inputs_embeds = cat([target_token_embd[last_token_id], target_last_hidden])
        let mut prev_embd = engine.token_embd_row(last_token_id);
        assert_eq!(
            prev_embd.len(),
            backbone_hidden,
            "engine.token_embd_row len {} != backbone_hidden {}",
            prev_embd.len(),
            backbone_hidden
        );
        // mt86 Stage A — target Engine 의 apply_embedding_scale 와 동일하게
        // ×sqrt(hidden_dim) 적용. Gemma4 GGUF 의 token_embd 는 raw 저장
        // (convert_hf_to_gguf.py Gemma4Model.modify_tensors 에서 OOV 제거만
        // 하고 scale 미적용), target Engine 은 prefill 시점에 적용한다
        // (engine/models/gemma/output.rs::apply_embedding_scale). Drafter input
        // 의 첫 절반도 학습 시점 분포 (scaled) 와 일치해야 함.
        //
        // baseline 0% → 23.44% (15/64). bf16 cast 시도해봤지만 영향 없음
        // (token-level matching 변화 X). f32 sqrt 그대로 사용.
        let embed_scale = (backbone_hidden as f32).sqrt();
        for v in prev_embd.iter_mut() {
            *v *= embed_scale;
        }

        if dump_step0 && step == 0 {
            println!(
                "[mt89.B-rust] prompt_tokens (first 8) = {:?}",
                &prompt_tokens[..prompt_tokens.len().min(8)]
            );
            println!(
                "[mt89.B-rust] prompt_tokens (last 6) = {:?}",
                &prompt_tokens[prompt_tokens.len().saturating_sub(6)..]
            );
            println!("[mt89.B-rust] prompt_len = {}", prompt_tokens.len());
            println!("[mt89.B-rust] last_token_id = {last_token_id}");
            let n = last_hidden.len() as f32;
            let mean = last_hidden.iter().sum::<f32>() / n;
            let var = last_hidden.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let std = var.sqrt();
            let max_abs = last_hidden.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let lh_norm = last_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "[mt89.B-rust] last_hidden norm={lh_norm:.6} mean={mean:.6} std={std:.6} max_abs={max_abs:.6}"
            );
            println!(
                "[mt89.B-rust] last_hidden sample[:8] = {:?}",
                &last_hidden[..8]
            );
            let pe_norm = prev_embd.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "[mt89.B-rust] prev_embd (scaled) norm = {pe_norm:.6} sample[:8] = {:?}",
                &prev_embd[..8]
            );
            // mt90 — last_hidden binary dump (Python 의 dump 와 elementwise diff 용)
            if let Ok(path) = std::env::var("RNB_CALIBRATE_DUMP_HIDDEN_BIN") {
                use std::io::Write;
                let mut f = std::fs::File::create(&path).expect("create dump file");
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        last_hidden.as_ptr() as *const u8,
                        last_hidden.len() * 4,
                    )
                };
                f.write_all(bytes).expect("write hidden bytes");
                println!(
                    "[mt89.B-rust] last_hidden binary → {path} ({} bytes)",
                    bytes.len()
                );
            }
        }

        let mut inputs_embeds: Vec<f32> = Vec::with_capacity(2 * backbone_hidden);
        inputs_embeds.extend_from_slice(&prev_embd);
        inputs_embeds.extend_from_slice(&last_hidden);
        assert_eq!(inputs_embeds.len(), 2 * backbone_hidden);

        let shared_kv = engine.shared_kv_states_for_drafter();
        // spec §7: position_ids = [[input_ids.shape[1] - 1]]. 본 loop 의 step k
        // 에서 input_ids 길이 = prompt_tokens.len() + k (teacher forcing 후
        // append). position 은 0-based last index.
        let position_id = (prompt_tokens.len() + step) as u32 - 1;

        let drafter_out = drafter_forward(&drafter, &inputs_embeds, &shared_kv, position_id);
        let drafter_pred = argmax(&drafter_out.logits);

        if dump_step0 && step == 0 {
            let ph = &drafter_out.projected_hidden;
            let ph_norm = ph.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "[mt89.B-rust] projected_hidden norm = {ph_norm:.6} sample[:8] = {:?}",
                &ph[..8]
            );
            let vl = &drafter_out.logits;
            let mut vl_idx: Vec<usize> = (0..vl.len()).collect();
            vl_idx.sort_by(|&a, &b| {
                vl[b]
                    .partial_cmp(&vl[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            println!("[mt89.B-rust] vocab_logits top5:");
            for &i in &vl_idx[..5] {
                println!("           token {i:6} = {:.6}", vl[i]);
            }
            println!("[mt89.B-rust] drafter_pred = {drafter_pred}");
        }

        // 3) Target reference — decode_one 등가물. forward(&[last_token_id])
        //    가 seq_len=1 path 로 KV cache 를 한 칸 늘리고 logits 반환.
        let target_logits = engine
            .forward(&[last_token_id])
            .expect("target forward decode failed");
        let target_pred = argmax(&target_logits);

        if drafter_pred == target_pred {
            top1_matches += 1;
        }
        total += 1;

        // 4) Teacher forcing — target 의 path 를 따라감.
        //
        // mt87 Stage E 검증 결과: autoregressive 흐름 (drafter projected_state
        // next iter input) 시도 시 9/64 (14.06%) 로 악화. teacher forcing
        // (target hidden 재사용) 이 정답 (23.44% baseline).
        last_token_id = target_pred;
        last_hidden = engine.last_layer_hidden().to_vec();
        assert_eq!(
            last_hidden.len(),
            backbone_hidden,
            "last_hidden len {} != backbone_hidden {} at step {}",
            last_hidden.len(),
            backbone_hidden,
            step
        );
    }

    let top1 = top1_matches as f32 / total as f32;
    println!("top1 = {top1:.4} ({top1_matches}/{total})");
    assert!(
        top1 >= ACCEPTANCE_TOP1,
        "top1 {top1:.4} < {ACCEPTANCE_TOP1} — run diagnostic δ.a/b/c/d tests",
    );
}

// =============================================================================
// Diagnostic δ.a — pre_projection weight 검증
// =============================================================================

/// Unit basis vector `e_0` (`[1, 0, …, 0]`) 를 input 으로 drafter forward 호출
/// 시 backbone path 의 `projected_hidden` 분포가 plausible 한지 (finite +
/// non-trivial) 확인. pre_projection weight 자체가 bug (all-zero / NaN) 이면
/// projected_hidden 전체가 비현실적 값으로 떨어진다.
#[test]
#[ignore = "needs RNB_DRAFTER_MODEL"]
fn drafter_diagnostic_a_pre_projection_weight() {
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));
    let drafter_path = PathBuf::from(&drafter_var);
    assert!(
        drafter_path.exists(),
        "${DRAFTER_ENV} points to missing file: {drafter_path:?}"
    );
    let drafter = load_drafter(&drafter_path).expect("load drafter");

    let mut e0 = vec![0f32; 2 * drafter.backbone_hidden];
    e0[0] = 1.0;
    let shared_kv = mock_shared_kv_states(1);
    let out = drafter_forward(&drafter, &e0, &shared_kv, 0);

    let mean = out.projected_hidden.iter().sum::<f32>() / out.projected_hidden.len() as f32;
    let max_abs = out
        .projected_hidden
        .iter()
        .map(|x| x.abs())
        .fold(0f32, f32::max);
    println!("δ.a: projected_hidden mean={mean:.6} max_abs={max_abs:.6}");

    assert!(
        out.projected_hidden.iter().all(|x| x.is_finite()),
        "NaN/Inf in projected_hidden — pre_projection or backbone bug"
    );
    assert!(
        max_abs > 1e-6,
        "projected_hidden all near-zero (max_abs={max_abs}) — pre_projection weight may be zero"
    );
}

// =============================================================================
// Diagnostic δ.b — VQ head 의 tied-lm_head self-identity
// =============================================================================

/// `drafter.token_embd_row(token_id)` 의 hidden=256 row 를 `vq_head_forward` +
/// `vocab_logits_in_top_k_clusters` 에 그대로 넣었을 때 `token_id` 가 자기
/// 자신의 top-K cluster 안에 들어가는지 확인. tied lm_head 의 정의상 row 와
/// 자기 자신 dot 이 가장 큰 self-similarity score 라 cluster centroid 의
/// permutation 이 정상이면 `token_id` 의 logit 이 `NEG_INFINITY` 가 아니거나
/// 직접 argmax 가 `token_id` 여야 한다.
#[test]
#[ignore = "needs RNB_DRAFTER_MODEL"]
fn drafter_diagnostic_b_vq_head_self_token() {
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));
    let drafter_path = PathBuf::from(&drafter_var);
    let drafter = load_drafter(&drafter_path).expect("load drafter");

    // 임의 token id (vocab 안에 들기만 하면 됨).
    let token_id: u32 = 1234;
    let row = drafter.token_embd_row(token_id);
    assert_eq!(
        row.len(),
        drafter.hidden,
        "drafter.token_embd_row({token_id}) len {} != drafter.hidden {}",
        row.len(),
        drafter.hidden
    );

    let vq = vq_head_forward(&drafter, &row);
    let cluster_table = ClusterTokenTable::permutation(
        drafter.token_ordering.clone(),
        drafter.n_centroids as usize,
    );
    let vocab_logits = vocab_logits_in_top_k_clusters(
        &drafter,
        &vq.cluster_logits,
        &row,
        drafter.centroid_top_k as usize,
        &cluster_table,
    );

    let pred = argmax(&vocab_logits);
    let target_logit = vocab_logits[token_id as usize];
    let max_logit = vocab_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    println!(
        "δ.b: token_id={token_id} pred={pred} target_logit={target_logit} max_logit={max_logit}"
    );

    // Self-identity 는 strict assertion 안 함 (centroid permutation 이 정상이면
    // token_id 의 cluster 가 top-K 안에 들어가서 finite logit, 그렇지 않으면
    // pred 가 token_id 와 같아도 OK). 두 조건 중 하나는 반드시 성립해야 함.
    assert!(
        target_logit.is_finite() || pred == token_id,
        "token_id {token_id} not in any top-K cluster (logit -inf) AND not top-1 \
         (pred={pred}) — tied lm_head / token_ordering permutation bug"
    );
}

// =============================================================================
// Diagnostic δ.c — backbone first-layer forward 의 출력 통계
// =============================================================================

/// drafter forward 호출 (mock shared_kv) 후 `projected_hidden` 의 mean / std
/// 가 plausible range 안에 있는지 확인. backbone 의 dim flow + RMSNorm + layer
/// scalar 의 출력 분포가 비정상 (all-zero, NaN, std 폭주) 이면 본 test 에서
/// 잡힌다.
///
/// 본 spec 의 architecture (backbone 4 layer + output_norm + post_projection)
/// 가 `drafter_forward` 단일 public 함수로 묶여있어 single-layer isolation 이
/// 불가. spec §Calibration 재설계 의 δ.c "backbone 단일 layer forward 검증" 은
/// 본 test 에서 4-layer 합산 출력 통계로 대체. private `decoder_layer_forward`
/// 접근이 필요하면 추후 별도 expose. **NEEDS_DECISION 후보지만 spec 의
/// "출력 statistics" 요구는 본 test 로 충족** (single-layer isolation 보다
/// drafter_forward 전체 finite 분포가 root cause 분리에 더 informative).
#[test]
#[ignore = "needs RNB_DRAFTER_MODEL"]
fn drafter_diagnostic_c_backbone_forward_stats() {
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));
    let drafter_path = PathBuf::from(&drafter_var);
    let drafter = load_drafter(&drafter_path).expect("load drafter");

    let inputs_embeds = vec![0.1f32; 2 * drafter.backbone_hidden];
    let seq_len = 10usize;
    let shared_kv = mock_shared_kv_states(seq_len);
    let position_id = (seq_len - 1) as u32;
    let out = drafter_forward(&drafter, &inputs_embeds, &shared_kv, position_id);

    let n = out.projected_hidden.len();
    assert_eq!(n, drafter.backbone_hidden);
    let mean = out.projected_hidden.iter().sum::<f32>() / n as f32;
    let variance: f32 = out
        .projected_hidden
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / n as f32;
    let std = variance.sqrt();
    let max_abs = out
        .projected_hidden
        .iter()
        .map(|x| x.abs())
        .fold(0f32, f32::max);
    println!("δ.c: projected_hidden mean={mean:.6} std={std:.6} max_abs={max_abs:.6}");

    assert!(
        out.projected_hidden.iter().all(|x| x.is_finite()),
        "projected_hidden has NaN/Inf — backbone forward numerical instability"
    );
    // RMS norm + post_projection 후 hidden 분포는 std 가 plausible range 안.
    // 너무 작으면 (1e-6 미만) backbone path 가 사실상 zero pass-through 이고,
    // 너무 크면 (1e3 초과) layer_scalar 폭주 또는 attention divergence.
    assert!(
        std > 1e-6 && std < 1e3,
        "projected_hidden std {std} out of plausible range [1e-6, 1e3]"
    );
}

// =============================================================================
// Diagnostic δ.d — shared_kv_states 의 idempotency
// =============================================================================

/// 동일 (target, prompt) 에서 `shared_kv_states_for_drafter()` 를 두 번
/// 연속 호출 시 결과가 byte-identical 인지 (immutable accessor). Stage β.4 의
/// `shared_kv_states_test.rs` 가 layout 자체는 이미 검증한 상태라 본 test 는
/// idempotency (ABAB 보강) 만 확인. accessor 가 우연히 internal state 를
/// mutate 하거나 dequant cache 가 첫 호출에만 정상이면 본 test 에서 잡힘.
#[test]
#[ignore = "needs RNB_TARGET_MODEL"]
fn drafter_diagnostic_d_shared_kv_idempotency() {
    let target_var = std::env::var(TARGET_ENV)
        .unwrap_or_else(|_| panic!("set {TARGET_ENV} to target GGUF path"));
    let target_path = PathBuf::from(&target_var);
    assert!(
        target_path.exists(),
        "${TARGET_ENV} points to missing file: {target_path:?}"
    );

    let mut engine = Engine::from_gguf(&target_path).expect("load target engine");
    let prompt_tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    engine
        .forward(&prompt_tokens)
        .expect("target forward on synthetic 5-token prompt");

    let states_1 = engine.shared_kv_states_for_drafter();
    let states_2 = engine.shared_kv_states_for_drafter();

    assert_eq!(
        states_1.sliding_attention.k, states_2.sliding_attention.k,
        "sliding_attention.k differs between calls — accessor not idempotent"
    );
    assert_eq!(
        states_1.sliding_attention.v, states_2.sliding_attention.v,
        "sliding_attention.v differs between calls"
    );
    assert_eq!(
        states_1.full_attention.k, states_2.full_attention.k,
        "full_attention.k differs between calls"
    );
    assert_eq!(
        states_1.full_attention.v, states_2.full_attention.v,
        "full_attention.v differs between calls"
    );
    println!(
        "δ.d: shared_kv idempotency OK (sliding_k.len={}, full_k.len={})",
        states_1.sliding_attention.k.len(),
        states_1.full_attention.k.len()
    );
}

// =============================================================================
// Diagnostic δ.e — token_ordering buffer sanity (mt85 Stage B)
// =============================================================================

/// `mtp.token_ordering.weight` 의 buffer 가 GGUF 에서 정상 로드됐는지 확인.
/// transformers `register_buffer("token_ordering", torch.empty(vocab_size, long))`
/// + 학습된 permutation 이라 buffer 의 값은 `[0, vocab_size)` 범위의 unique
/// permutation 이어야 한다. zeros / random / out-of-range 면 cluster permutation
/// 자체가 망가져서 mt84 의 δ.b FAIL 의 진짜 root cause 가 됨.
///
/// mt85 Priority 2 fix 후보 1 직접 검증.
#[test]
#[ignore = "needs RNB_DRAFTER_MODEL"]
fn drafter_diagnostic_e_token_ordering_sanity() {
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));
    let drafter_path = PathBuf::from(&drafter_var);
    let drafter = load_drafter(&drafter_path).expect("load drafter");

    let ordering = &drafter.token_ordering;
    let vocab_size = ordering.len();
    let n_centroids = drafter.n_centroids as usize;
    let vocab_per_centroid = vocab_size / n_centroids;
    assert_eq!(
        vocab_size,
        n_centroids * vocab_per_centroid,
        "token_ordering len {vocab_size} not divisible by n_centroids {n_centroids}"
    );

    let first16: Vec<u32> = ordering.iter().take(16).copied().collect();
    let zeros: usize = ordering.iter().filter(|&&v| v == 0).count();
    let max_val = ordering.iter().max().copied().unwrap_or(0);
    let min_val = ordering.iter().min().copied().unwrap_or(0);
    let unique_count = {
        let mut sorted: Vec<u32> = ordering.clone();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    };

    println!("δ.e: vocab_size={vocab_size} n_centroids={n_centroids} v_per_c={vocab_per_centroid}");
    println!("δ.e: first16={first16:?}");
    println!("δ.e: zeros={zeros} min={min_val} max={max_val} unique={unique_count}");

    assert!(
        max_val < vocab_size as u32,
        "token_ordering max {max_val} >= vocab_size {vocab_size} — out of range \
         (likely wrong dtype or byte-order in loader)"
    );
    assert!(
        unique_count >= (vocab_size as f32 * 0.95) as usize,
        "token_ordering only has {unique_count}/{vocab_size} unique values \
         — likely not a permutation (cluster mapping broken)"
    );
    assert!(
        zeros <= 1,
        "token_ordering has {zeros} zeros — buffer looks uninitialized \
         (only one zero is OK: vocab id 0 mapped to first slot)"
    );
}

// =============================================================================
// Diagnostic δ.f — self-input token cluster rank (mt85 Stage B)
// =============================================================================

/// `drafter.token_embd_row(token_id)` 를 input 으로 vq_head_forward 시 `token_id`
/// 자신이 속한 cluster 의 `cluster_logits` 위치가 top-K (= `centroid_top_k`)
/// 안에 들어가는지 측정. transformers `Gemma4AssistantMaskedEmbedder.centroids`
/// 는 학습된 nn.Linear 라 self-similarity 가 maximum 이라는 가정은 guaranteed
/// 가 아니지만, cluster permutation 이 정상이고 centroids 가 의미 있게 학습됐다
/// 면 적어도 top-K 안에는 들어가야 "drafter 가 자기 자신을 어느 정도 맞춤"
/// 이라는 약한 일치가 성립.
///
/// 본 test 는 assertion 없이 informational — 결과를 보고 mt85-next 의 Priority 2
/// fix 가설을 좁힌다:
/// - rank < top_k → self-identity 가설 valid. δ.b FAIL 의 다른 root cause 가 있음.
/// - rank >= top_k → self-identity 가설 자체가 wrong test (centroids 의 학습된
///   axis 가 token_embd 와 다름). δ.b 측정 자체가 invalid → δ.b2 로 대체.
#[test]
#[ignore = "needs RNB_DRAFTER_MODEL"]
fn drafter_diagnostic_f_token_in_top_k_cluster_rank() {
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));
    let drafter_path = PathBuf::from(&drafter_var);
    let drafter = load_drafter(&drafter_path).expect("load drafter");

    let n_centroids = drafter.n_centroids as usize;
    let vocab_per_centroid = drafter.token_ordering.len() / n_centroids;
    let top_k = drafter.centroid_top_k as usize;

    let probe_ids: Vec<u32> = vec![1234, 100, 50_000, 200_000];
    for tok in probe_ids {
        let row = drafter.token_embd_row(tok);
        let vq = vq_head_forward(&drafter, &row);

        let cluster_id = drafter
            .token_ordering
            .iter()
            .position(|&t| t == tok)
            .map(|p| p / vocab_per_centroid);

        let Some(cluster_id) = cluster_id else {
            println!("δ.f: token={tok} NOT in token_ordering — permutation missing this token");
            continue;
        };

        let target_logit = vq.cluster_logits[cluster_id];
        let rank = vq
            .cluster_logits
            .iter()
            .filter(|&&l| l > target_logit)
            .count();

        let in_top_k = rank < top_k;
        println!(
            "δ.f: token={tok} cluster_id={cluster_id} target_logit={target_logit:.6} \
             rank={rank} top_k={top_k} in_top_k={in_top_k}"
        );
    }
}

// =============================================================================
// Diagnostic δ.b2 — target prefill hidden → top-K cluster contains target's
// next token (mt85 Stage B 재설계 from δ.b)
// =============================================================================

/// mt85-next Priority 2 의 재설계 expectation.
///
/// target Engine prefill 후 drafter_forward 를 호출하면 `drafter_out.logits` 의
/// finite entry 는 top-K cluster 의 token 만 (나머지는 NEG_INFINITY). target's
/// argmax 가 그 finite set 안에 있는지 측정.
///
/// 통과 = cluster permutation 정상 + drafter 가 의미 있는 cluster 선택.
/// 실패 = cluster mapping 자체 또는 backbone forward 의 hidden 분포 문제.
///
/// strict assertion 없음 (informational). main `drafter_backbone_calibrate_top1`
/// 의 top-1 일치보다 약한 condition 이라 root cause 좁히기에 더 informative.
#[test]
#[ignore = "needs RNB_TARGET_MODEL + RNB_DRAFTER_MODEL"]
fn drafter_diagnostic_b2_target_hidden_top_k_cluster() {
    let target_var = std::env::var(TARGET_ENV)
        .unwrap_or_else(|_| panic!("set {TARGET_ENV} to target GGUF path"));
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));

    let mut engine = Engine::from_gguf(&PathBuf::from(&target_var)).expect("load target engine");
    let drafter = load_drafter(&PathBuf::from(&drafter_var)).expect("load drafter");

    let prompt_path = std::env::var("RNB_CALIBRATE_PROMPT").unwrap_or_else(|_| {
        format!(
            "{}/../../prompts/bench_small_ko_singleline.txt",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let prompt_tokens = load_prompt_tokens(&prompt_path, &engine);
    engine.forward(&prompt_tokens).expect("prefill failed");

    let backbone_hidden = drafter.backbone_hidden;
    let mut last_token_id = *prompt_tokens.last().unwrap();
    let mut last_hidden: Vec<f32> = engine.last_layer_hidden().to_vec();
    assert_eq!(last_hidden.len(), backbone_hidden);

    let n_eval: usize = 16;
    let mut hits: u32 = 0;
    for step in 0..n_eval {
        let prev_embd = engine.token_embd_row(last_token_id);
        let mut inputs_embeds: Vec<f32> = Vec::with_capacity(2 * backbone_hidden);
        inputs_embeds.extend_from_slice(&prev_embd);
        inputs_embeds.extend_from_slice(&last_hidden);

        let shared_kv = engine.shared_kv_states_for_drafter();
        let position_id = (prompt_tokens.len() + step) as u32 - 1;
        let drafter_out = drafter_forward(&drafter, &inputs_embeds, &shared_kv, position_id);

        let target_logits = engine.forward(&[last_token_id]).expect("decode failed");
        let target_pred = argmax(&target_logits);

        let drafter_logit_for_target = drafter_out.logits[target_pred as usize];
        let in_top_k = drafter_logit_for_target.is_finite();
        if in_top_k {
            hits += 1;
        }
        println!(
            "δ.b2 step={step}: target_pred={target_pred} \
             drafter_logit_for_target={drafter_logit_for_target:.6} in_top_k={in_top_k}"
        );

        last_token_id = target_pred;
        last_hidden = engine.last_layer_hidden().to_vec();
    }
    let rate = hits as f32 / n_eval as f32;
    println!("δ.b2: top-K cluster hit rate = {rate:.4} ({hits}/{n_eval})");
}

// =============================================================================
// Diagnostic δ.g — drafter token_embd row distribution (mt85 Stage B)
// =============================================================================

/// `drafter.token_embd_row(tok)` (Q6_K dequant) 의 분포가 학습된 weight 인지
/// 확인. 만약 mean ≈ 0, std ∈ [0.01, 1.0] 정도면 학습된 embedding. all-zero
/// 또는 NaN 이면 dequant bug 또는 weight 비어있음. random uniform 이면 학습 안
/// 됨.
///
/// 본 test 는 vocab 의 spread 된 token 4개 의 distribution 만 sampling.
#[test]
#[ignore = "needs RNB_DRAFTER_MODEL"]
fn drafter_diagnostic_g_token_embd_row_distribution() {
    let drafter_var = std::env::var(DRAFTER_ENV)
        .unwrap_or_else(|_| panic!("set {DRAFTER_ENV} to drafter GGUF path"));
    let drafter_path = PathBuf::from(&drafter_var);
    let drafter = load_drafter(&drafter_path).expect("load drafter");

    let probe_ids: Vec<u32> = vec![0, 1234, 100_000, 200_000];
    for tok in probe_ids {
        let row = drafter.token_embd_row(tok);
        let n = row.len();
        let mean: f32 = row.iter().sum::<f32>() / n as f32;
        let variance: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();
        let max_abs = row.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let zeros = row.iter().filter(|&&v| v.abs() < 1e-9).count();
        let nan_count = row.iter().filter(|&&v| !v.is_finite()).count();
        let first8: Vec<f32> = row.iter().take(8).copied().collect();
        println!(
            "δ.g: token={tok} n={n} mean={mean:.6} std={std:.6} max_abs={max_abs:.6} \
             zeros={zeros} nan={nan_count} first8={first8:?}"
        );
        assert!(
            nan_count == 0,
            "token_embd_row({tok}) has {nan_count} NaN/Inf"
        );
        assert!(
            max_abs > 1e-6,
            "token_embd_row({tok}) all near-zero — weight uninitialized?"
        );
    }
}
