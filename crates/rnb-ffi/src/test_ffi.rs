use rnb_ffi::{rnb_free, rnb_load, rnb_next_token, rnb_reset, rnb_submit};
/// FFI API 테스트 — reset 포함
use std::ffi::CString;

fn run_question(ctx: *mut rnb_ffi::RnbContext, prompt: &str, max_tokens: usize) {
    unsafe {
        let p = CString::new(prompt).unwrap();
        eprintln!("[test] Submit: {}", prompt);
        let ret = rnb_submit(ctx, p.as_ptr());
        if ret != 0 {
            eprintln!("[test] FAIL: rnb_submit returned {}", ret);
            return;
        }

        let mut count = 0;
        loop {
            let tok = rnb_next_token(ctx);
            if tok.is_null() {
                break;
            }
            let s = std::ffi::CStr::from_ptr(tok).to_str().unwrap_or("");
            eprint!("{}", s);
            count += 1;
            if count >= max_tokens {
                break;
            }
        }
        eprintln!("\n[test] {} tokens\n", count);
    }
}

fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[PANIC] {}", info);
    }));
    unsafe {
        let model_path = CString::new("models/Qwen3.5-0.8B-Q4_K_M.gguf").unwrap();
        eprintln!("[test] Loading model...");
        let ctx = rnb_load(model_path.as_ptr());
        if ctx.is_null() {
            eprintln!("[test] FAIL: rnb_load returned NULL");
            return;
        }
        eprintln!("[test] Model loaded OK\n");

        // 질문 1
        run_question(ctx, "인공지능이란 무엇인가요?", 200);

        // 리셋 후 질문 2
        eprintln!("[test] === RESET ===");
        rnb_reset(ctx);
        run_question(ctx, "사랑이란 무엇일까요?", 200);

        rnb_free(ctx);
        eprintln!("[test] All done!");
    }
}
