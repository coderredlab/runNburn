use memmap2::Mmap;
use rnb_loader::gguf::parser::GGUFFile;
use std::fs::File;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump-gguf <gguf>");
    let f = File::open(&path).expect("open");
    let mmap = unsafe { Mmap::map(&f) }.expect("mmap");
    let gguf = GGUFFile::parse(&mmap[..]).expect("parse");
    println!("=== metadata ({}) ===", gguf.metadata.len());
    for (k, v) in &gguf.metadata {
        println!("{} = {:?}", k, v);
    }
    println!("=== tensors ({}) ===", gguf.tensor_infos.len());
    for t in &gguf.tensor_infos {
        println!("{:?} {:?} {}", t.ggml_type, t.shape, t.name);
    }
}
