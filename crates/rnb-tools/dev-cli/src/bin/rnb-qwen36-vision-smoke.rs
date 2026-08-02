use std::env;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use rnb_loader::load_vision_projector;
use rnb_model_qwen::{
    encode_qwen36_vision_intermediate, inspect_qwen36_vision_projector,
    prepare_qwen36_vision_intermediate, Qwen36RgbImage, Qwen36TensorStats,
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
            "usage: {} <path-to-mmproj.gguf> [--white-reference]",
            program.to_string_lossy()
        );
    };
    let Some(path) = args.next().map(PathBuf::from) else {
        usage();
        return ExitCode::from(2);
    };
    let white_reference = match args.next() {
        None => false,
        Some(value) if value == OsStr::new("--white-reference") => true,
        Some(_) => {
            usage();
            return ExitCode::from(2);
        }
    };
    if args.next().is_some() {
        usage();
        return ExitCode::from(2);
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

    ExitCode::SUCCESS
}

fn fixed_image(white_reference: bool) -> Result<Qwen36RgbImage, rnb_model_qwen::Qwen36VisionError> {
    const WIDTH: usize = 768;
    const HEIGHT: usize = 768;
    if white_reference {
        return Qwen36RgbImage::new(WIDTH, HEIGHT, vec![255; WIDTH * HEIGHT * 3]);
    }
    let mut pixels = Vec::with_capacity(WIDTH * HEIGHT * 3);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push((x * 255 / (WIDTH - 1)) as u8);
            pixels.push((y * 255 / (HEIGHT - 1)) as u8);
            pixels.push(((x + y) * 255 / (WIDTH + HEIGHT - 2)) as u8);
        }
    }
    Qwen36RgbImage::new(WIDTH, HEIGHT, pixels)
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
