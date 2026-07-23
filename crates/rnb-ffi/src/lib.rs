use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rnb_llm::generate::GenerateParams;
use rnb_llm::sampler::SamplerChain;
use rnb_llm::tokenizer::TokenStreamDecoder;
use rnb_llm::Engine;

/// 생성 통계
#[repr(C)]
pub struct RnbStats {
    pub prefill_ms: f32,
    pub decode_ms: f32,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
}

/// 내부 상태
enum ContextState {
    /// 모델 로드됨, 프롬프트 대기
    Ready,
    /// prefill 완료, decode 진행 중
    Generating {
        next_token: u32,
        generated_tokens: Vec<u32>,
        decode_start: Instant,
    },
    /// EOS 도달, 생성 완료
    Done,
}

/// Opaque context — C에서는 포인터로만 사용
pub struct RnbContext {
    engine: Engine,
    sampler: SamplerChain,
    /// 현재 활성 샘플러 설정. rnb_set_sampler 로 갱신되고,
    /// rnb_reset 은 이 값으로 sampler 를 재생성해서 호출자가
    /// 설정한 값을 보존한다.
    sampler_params: GenerateParams,
    rng: SmallRng,
    state: ContextState,
    // chat template special token IDs
    im_start: u32,
    im_end: u32,
    nl: u32,
    // 통계
    prefill_ms: f32,
    prefill_tokens: u32,
    decode_tokens: u32,
    decode_ms: f32,
    // 반환용 버퍼 (다음 호출 전까지 유효)
    token_buf: CString,
    // GPT-2 byte-level BPE 토큰 경계를 넘는 UTF-8 시퀀스 보존
    text_decoder: TokenStreamDecoder,
}

/// 모델 로드. 실패 시 NULL.
#[no_mangle]
pub unsafe extern "C" fn rnb_load(model_path: *const c_char) -> *mut RnbContext {
    rnb_load_with_ram_budget(model_path, 0)
}

/// 모델 로드 + 호스트 RAM 예산 지정. `ram_budget_bytes == 0`이면 자동 정책.
#[no_mangle]
pub unsafe extern "C" fn rnb_load_with_ram_budget(
    model_path: *const c_char,
    ram_budget_bytes: u64,
) -> *mut RnbContext {
    if model_path.is_null() {
        return std::ptr::null_mut();
    }
    let path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let path = Path::new(path_str);
    let engine = match if ram_budget_bytes == 0 {
        Engine::from_gguf(path)
    } else {
        Engine::from_gguf_with_host_ram_budget(path, ram_budget_bytes)
    } {
        Ok(e) => e,
        Err(_) => return std::ptr::null_mut(),
    };

    let im_start = engine.tokenizer.vocab.token_id("<|im_start|>").unwrap_or(0);
    let im_end = engine.tokenizer.vocab.token_id("<|im_end|>").unwrap_or(0);
    let nl = engine
        .tokenizer
        .vocab
        .token_id("\n")
        .or_else(|| engine.tokenizer.vocab.token_id("Ċ"))
        .unwrap_or(198);

    let params = GenerateParams {
        repetition_penalty: 1.1,
        temperature: 0.0,
        ..GenerateParams::default()
    };
    let sampler = SamplerChain::from_params(&params);
    let rng = SmallRng::seed_from_u64(42);

    let ctx = Box::new(RnbContext {
        engine,
        sampler,
        sampler_params: params,
        rng,
        state: ContextState::Ready,
        im_start,
        im_end,
        nl,
        prefill_ms: 0.0,
        prefill_tokens: 0,
        decode_tokens: 0,
        decode_ms: 0.0,
        token_buf: CString::new("").unwrap(),
        text_decoder: TokenStreamDecoder::new(),
    });
    Box::into_raw(ctx)
}

/// 프롬프트 제출. chat template 적용 + prefill. 0=성공, -1=에러.
#[no_mangle]
pub unsafe extern "C" fn rnb_submit(ctx: *mut RnbContext, prompt: *const c_char) -> i32 {
    if ctx.is_null() || prompt.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // chat template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n
    let prompt_tokens = ctx.engine.tokenizer.encode(prompt_str);
    let mut tokens = Vec::new();
    tokens.push(ctx.im_start);
    tokens.extend(ctx.engine.tokenizer.encode("user"));
    tokens.push(ctx.nl);
    tokens.extend(&prompt_tokens);
    tokens.push(ctx.im_end);
    tokens.push(ctx.nl);
    tokens.push(ctx.im_start);
    tokens.extend(ctx.engine.tokenizer.encode("assistant"));
    tokens.push(ctx.nl);
    tokens.extend(ctx.engine.tokenizer.encode("<think>"));
    tokens.push(ctx.nl);

    prefill_tokens(ctx, &tokens, "rnb_submit")
}

/// Template 이 이미 적용된 raw 텍스트를 그대로 prefill 한다.
/// 호출 측에서 GGUF `tokenizer.chat_template` 또는 자체 규약으로
/// `<|im_start|>...<|im_end|>` 같은 special token 을 포함한 완전한
/// prompt 를 조립한 뒤 한 덩어리로 넘겨야 한다.
///
/// `rnb_submit` 과 달리 엔진이 user/assistant/`<think>` 마크업을
/// 추가하지 않는다. 시스템 프롬프트, thinking 토글, 다중 턴 누적,
/// 비-Qwen 모델 대응 등 chat-level 정책은 전부 호출자 책임.
///
/// 반환: 0=성공, -1=ctx/text NULL 또는 forward 실패.
#[no_mangle]
pub unsafe extern "C" fn rnb_submit_raw(ctx: *mut RnbContext, text: *const c_char) -> i32 {
    if ctx.is_null() || text.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;
    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let tokens = ctx.engine.tokenizer.encode(text_str);
    prefill_tokens(ctx, &tokens, "rnb_submit_raw")
}

/// 공통 prefill 루틴. `rnb_submit` / `rnb_submit_raw` 가 tokens 벡터만
/// 서로 다르게 조립하고 나머지(forward → 첫 토큰 샘플링 → state 전환)는 동일해서
/// 여기로 모은다.
fn prefill_tokens(ctx: &mut RnbContext, tokens: &[u32], tag: &str) -> i32 {
    eprintln!(
        "[{}] prefill {} tokens, kv_len={}",
        tag,
        tokens.len(),
        ctx.engine.kv_cache.current_len()
    );
    let prefill_start = Instant::now();
    let mut logits = match ctx.engine.forward(tokens) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[{}] forward error: {:?}", tag, e);
            return -1;
        }
    };
    ctx.prefill_ms = prefill_start.elapsed().as_secs_f64() as f32 * 1000.0;
    ctx.prefill_tokens = tokens.len() as u32;

    // 첫 토큰 샘플링
    let first_token = ctx.sampler.sample(&mut logits, &[], &mut ctx.rng);

    ctx.state = ContextState::Generating {
        next_token: first_token,
        generated_tokens: vec![first_token],
        decode_start: Instant::now(),
    };
    ctx.decode_tokens = 0;
    ctx.decode_ms = 0.0;
    ctx.text_decoder = TokenStreamDecoder::new();

    0
}

/// 다음 토큰 생성. UTF-8 문자열 반환, NULL이면 EOS/에러.
#[no_mangle]
pub unsafe extern "C" fn rnb_next_token(ctx: *mut RnbContext) -> *const c_char {
    if ctx.is_null() {
        return std::ptr::null();
    }
    let ctx = &mut *ctx;

    // state에서 next_token 꺼내기
    let (current_token, generated_tokens_snapshot) = match &ctx.state {
        ContextState::Generating {
            next_token,
            generated_tokens,
            ..
        } => (*next_token, generated_tokens.clone()),
        _ => return std::ptr::null(),
    };

    // EOS 체크
    if current_token == ctx.im_end {
        let decode_start = match &ctx.state {
            ContextState::Generating { decode_start, .. } => *decode_start,
            _ => Instant::now(),
        };
        ctx.decode_ms = decode_start.elapsed().as_secs_f64() as f32 * 1000.0;
        ctx.state = ContextState::Done;
        return std::ptr::null();
    }

    // GPT-2 byte-level BPE는 한 UTF-8 문자를 여러 토큰으로 나눌 수 있다.
    let tok_str = ctx.text_decoder.push(&ctx.engine.tokenizer, current_token);

    // forward로 다음 logits 계산
    let mut logits = match ctx.engine.forward(&[current_token]) {
        Ok(l) => l,
        Err(_) => {
            ctx.state = ContextState::Done;
            return std::ptr::null();
        }
    };

    // 다음 토큰 샘플링
    let next = ctx
        .sampler
        .sample(&mut logits, &generated_tokens_snapshot, &mut ctx.rng);
    ctx.decode_tokens += 1;

    // state 업데이트
    if let ContextState::Generating {
        next_token,
        generated_tokens,
        ..
    } = &mut ctx.state
    {
        *next_token = next;
        generated_tokens.push(next);
    }

    ctx.token_buf = CString::new(tok_str).unwrap_or_else(|_| CString::new("").unwrap());
    ctx.token_buf.as_ptr()
}

/// 생성 통계 조회
#[no_mangle]
pub unsafe extern "C" fn rnb_get_stats(ctx: *mut RnbContext, out: *mut RnbStats) {
    if ctx.is_null() || out.is_null() {
        return;
    }
    let ctx = &*ctx;
    (*out).prefill_ms = ctx.prefill_ms;
    (*out).prefill_tokens = ctx.prefill_tokens;
    (*out).decode_tokens = ctx.decode_tokens;
    (*out).decode_ms = ctx.decode_ms;
}

/// 컨텍스트 리셋 (새 대화)
#[no_mangle]
pub unsafe extern "C" fn rnb_reset(ctx: *mut RnbContext) {
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    // kv_cache.clear()는 전체 262144 시퀀스를 0으로 채우려 해서 OOM.
    // current_len만 리셋하면 이전 데이터는 덮어씌워지므로 안전.
    ctx.engine.kv_cache.set_len(0);
    // SSM state만 별도로 초기화 (작은 메모리)
    for ssm in ctx.engine.kv_cache.ssm_states.iter_mut() {
        if let Some(s) = ssm {
            s.clear();
        }
    }
    ctx.state = ContextState::Ready;
    ctx.prefill_ms = 0.0;
    ctx.prefill_tokens = 0;
    ctx.decode_tokens = 0;
    ctx.decode_ms = 0.0;
    ctx.text_decoder = TokenStreamDecoder::new();

    // sampler 재생성 — rnb_set_sampler 로 갱신된 현재 설정을 보존한다.
    ctx.sampler = SamplerChain::from_params(&ctx.sampler_params);
    ctx.rng = SmallRng::seed_from_u64(42);
}

/// 런타임 sampler 파라미터 변경. 모델 재로딩 없이 즉시 적용.
/// `top_k = 0` 은 top-k 비활성을 의미 (전체 vocab 사용).
/// 반환: 0=성공, -1=ctx NULL / NaN / 범위 외.
#[no_mangle]
pub unsafe extern "C" fn rnb_set_sampler(
    ctx: *mut RnbContext,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repetition_penalty: f32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    if !temperature.is_finite()
        || !top_p.is_finite()
        || !repetition_penalty.is_finite()
        || temperature < 0.0
        || !(0.0..=1.0).contains(&top_p)
        || repetition_penalty <= 0.0
    {
        return -1;
    }
    let ctx = &mut *ctx;
    ctx.sampler_params.temperature = temperature;
    ctx.sampler_params.top_p = top_p;
    ctx.sampler_params.top_k = top_k as usize;
    ctx.sampler_params.repetition_penalty = repetition_penalty;
    ctx.sampler = SamplerChain::from_params(&ctx.sampler_params);
    0
}

/// 메모리 해제
#[no_mangle]
pub unsafe extern "C" fn rnb_free(ctx: *mut RnbContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}
