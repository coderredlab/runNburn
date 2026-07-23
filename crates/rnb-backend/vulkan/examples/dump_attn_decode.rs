//! mv38 B: SPIR-V binary dump for emit_attention_decode debugging.
//!
//! `cargo run --example dump_attn_decode -p rnb-backend-vulkan`
//!
//! Writes SPIR-V binary to `/tmp/attn_decode.spv`. Disassemble:
//!   spirv-dis /tmp/attn_decode.spv | less

use rnb_backend_vulkan::spirv::emit_attention_decode;

fn main() {
    let words: Vec<u32> = emit_attention_decode(64);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = "/tmp/attn_decode.spv";
    std::fs::write(out, bytes).expect("write failed");
    println!(
        "wrote {} ({} words = {} bytes)",
        out,
        words.len(),
        words.len() * 4
    );
}
