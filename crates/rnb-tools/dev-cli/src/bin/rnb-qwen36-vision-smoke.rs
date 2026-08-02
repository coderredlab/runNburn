use std::env;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use rnb_loader::load_vision_projector;
use rnb_model_qwen::{
    encode_qwen36_vision_intermediate, inspect_qwen36_vision_projector,
    prepare_qwen36_vision_intermediate, Qwen36TensorStats, RgbImage,
};
use sha2::{Digest, Sha256};

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| PathBuf::from(value).file_name().map(|name| name.to_owned()))
        .unwrap_or_else(|| "rnb-qwen36-vision-smoke".into());
    let usage = || {
        eprintln!(
            "usage: {} <path-to-mmproj.gguf> [--white-reference] [--model <path-to-model.gguf>] [--write-image <path.ppm>]",
            program.to_string_lossy()
        );
    };
    let Some(path) = args.next().map(PathBuf::from) else {
        usage();
        return ExitCode::from(2);
    };
    let mut white_reference = false;
    let mut model_path = None;
    let mut reference_image_path = None;
    while let Some(value) = args.next() {
        if value == OsStr::new("--white-reference") {
            white_reference = true;
        } else if value == OsStr::new("--model") {
            let Some(value) = args.next() else {
                usage();
                return ExitCode::from(2);
            };
            model_path = Some(PathBuf::from(value));
        } else if value == OsStr::new("--write-image") {
            let Some(value) = args.next() else {
                usage();
                return ExitCode::from(2);
            };
            reference_image_path = Some(PathBuf::from(value));
        } else {
            usage();
            return ExitCode::from(2);
        }
    }

    let projector = match load_vision_projector(&path) {
        Ok(projector) => projector,
        Err(error) => {
            eprintln!("projector load failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let capability = match inspect_qwen36_vision_projector(&projector.descriptor) {
        Ok(capability) => capability,
        Err(error) => {
            eprintln!("Qwen3.6 vision capability check failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let image = match fixed_image(white_reference) {
        Ok(image) => image,
        Err(error) => {
            eprintln!("fixed smoke image construction failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(reference_image_path) = reference_image_path.as_ref() {
        if let Err(error) = write_ppm(reference_image_path, &image) {
            eprintln!(
                "reference image write failed ({}): {error}",
                reference_image_path.display()
            );
            return ExitCode::FAILURE;
        }
    }
    let intermediate = match prepare_qwen36_vision_intermediate(&projector, &image) {
        Ok(intermediate) => intermediate,
        Err(error) => {
            eprintln!("Qwen3.6 vision intermediate failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    println!("status = supported");
    println!("path = {}", path.display());
    println!(
        "architecture = {}",
        projector.descriptor.envelope.architecture
    );
    println!("role = {}", projector.descriptor.envelope.kind);
    println!("projector_type = {}", capability.projector_type);
    println!("image_size = {}", capability.image_size);
    println!("patch_size = {}", capability.patch_size);
    println!("spatial_merge_size = {}", capability.spatial_merge_size);
    println!("embedding_length = {}", capability.embedding_length);
    println!("feed_forward_length = {}", capability.feed_forward_length);
    println!("block_count = {}", capability.block_count);
    println!("head_count = {}", capability.head_count);
    println!("projection_dim = {}", capability.projection_dim);
    println!("layer_norm_epsilon = {:.9}", capability.layer_norm_epsilon);
    println!("image_mean = {:?}", capability.image_mean);
    println!("image_std = {:?}", capability.image_std);
    println!("tensor_count = {}", capability.tensor_count);
    println!("tensor_bytes = {}", capability.tensor_bytes);
    println!("mapped_weight_count = {}", projector.weights.len());
    println!(
        "image_source = {}",
        if white_reference {
            "white-reference"
        } else {
            "deterministic-rgb-gradient"
        }
    );
    println!("source_image = {}x{}", image.width(), image.height());
    println!(
        "target_image = {}x{}",
        intermediate.target_width, intermediate.target_height
    );
    println!(
        "patch_grid = {}x{}",
        intermediate.patch_grid_width, intermediate.patch_grid_height
    );
    println!(
        "merged_grid = {}x{}",
        intermediate.merged_grid_width, intermediate.merged_grid_height
    );
    println!(
        "intermediate_shape = [{}, {}]",
        intermediate.patch_grid_width * intermediate.patch_grid_height,
        intermediate.embedding_length
    );
    print_stats("normalized", intermediate.normalized_stats);
    print_stats("temporal_patch", intermediate.temporal_patch_stats);
    print_stats("position", intermediate.position_stats);
    print_stats("intermediate", intermediate.intermediate_stats);
    println!(
        "intermediate_sha256 = {}",
        hash_f32(&intermediate.patch_embeddings)
    );
    println!(
        "intermediate_first8 = {:?}",
        &intermediate.patch_embeddings[..8]
    );

    let output = match encode_qwen36_vision_intermediate(&projector, intermediate) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("Qwen3.6 vision encoder failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    for index in [
        0,
        output.layer_summaries.len() / 2,
        output.layer_summaries.len() - 1,
    ] {
        let summary = &output.layer_summaries[index];
        print_stats(&format!("block_{:02}", summary.layer_index), summary.stats);
        println!(
            "block_{:02}_first8 = {:?}",
            summary.layer_index, summary.first_values
        );
    }
    print_stats("post_layer_norm", output.post_layer_norm_stats);
    println!(
        "embedding_shape = [{}, {}]",
        output.merged_grid_width * output.merged_grid_height,
        output.projection_dim
    );
    print_stats("embedding", output.embedding_stats);
    println!("embedding_sha256 = {}", hash_f32(&output.embeddings));
    println!("embedding_first8 = {:?}", &output.embeddings[..8]);

    if let Some(model_path) = model_path {
        const PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What are the dominant colors in this image?<|im_end|>\n<|im_start|>assistant\n<think>\n";
        let config = rnb_llm::EngineLoadConfig::default().with_vision_projector(path.as_path());
        let mut engine = match rnb_llm::Engine::from_gguf_with_config(&model_path, config) {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("base model load failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let logits = match engine.debug_qwen36_multimodal_prefill_logits(PROMPT, &image) {
            Ok(logits) => logits,
            Err(error) => {
                eprintln!("multimodal prefill failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        println!("base_model = {}", model_path.display());
        println!("mixed_prompt = {:?}", PROMPT);
        println!("first_logits_sha256 = {}", hash_f32(&logits));
        for (rank, (token_id, logit)) in ranked.iter().take(20).enumerate() {
            println!(
                "first_logit_top{:02} = id={} logit={:.9} piece={:?}",
                rank + 1,
                token_id,
                logit,
                engine.tokenizer.decode_token(*token_id as u32)
            );
        }
    }

    ExitCode::SUCCESS
}

fn fixed_image(white_reference: bool) -> Result<RgbImage, rnb_core::image::ImageError> {
    const WIDTH: usize = 768;
    const HEIGHT: usize = 768;
    if white_reference {
        return RgbImage::new(WIDTH, HEIGHT, vec![255; WIDTH * HEIGHT * 3]);
    }
    let mut pixels = Vec::with_capacity(WIDTH * HEIGHT * 3);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push((x * 255 / (WIDTH - 1)) as u8);
            pixels.push((y * 255 / (HEIGHT - 1)) as u8);
            pixels.push(((x + y) * 255 / (WIDTH + HEIGHT - 2)) as u8);
        }
    }
    RgbImage::new(WIDTH, HEIGHT, pixels)
}

fn write_ppm(path: &std::path::Path, image: &RgbImage) -> std::io::Result<()> {
    let header = format!("P6\n{} {}\n255\n", image.width(), image.height());
    let mut bytes = Vec::with_capacity(header.len() + image.pixels().len());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(image.pixels());
    std::fs::write(path, bytes)
}

fn print_stats(name: &str, stats: Qwen36TensorStats) {
    println!(
        "{name}_stats = count:{} mean:{:.9} stddev:{:.9} min:{:.9} max:{:.9}",
        stats.count, stats.mean, stats.stddev, stats.min, stats.max
    );
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}
