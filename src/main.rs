use std::path::Path;

mod lab;
mod clap;
mod mosaic;
#[cfg(feature = "cuda")]
mod gpu;

use mosaic::{mosaic, ColorSpace};
#[cfg(feature = "cuda")]
use mosaic::mosaic_gpu;
use clap::get_matches;

fn main() {
    let matches = get_matches();
    let target = matches.get_one::<String>("target").expect("required");
    let row_size = *matches.get_one::<u32>("row_size").expect("required");
    let col_size = *matches.get_one::<u32>("col_size").expect("required");
    let images = matches.get_one::<String>("images").expect("required");
    let output = matches.get_one::<String>("output").expect("required");
    let color_space = matches
        .get_one::<ColorSpace>("color_space")
        .expect("required");
    let avoid_duplicates = matches.get_flag("avoid_duplicates");
    let use_gpu = matches.get_flag("gpu");

    #[cfg(feature = "cuda")]
    if use_gpu {
        mosaic_gpu(
            Path::new(target),
            row_size,
            col_size,
            Path::new(images),
            Path::new(output),
            *color_space,
            avoid_duplicates,
        );
        return;
    }

    #[cfg(not(feature = "cuda"))]
    if use_gpu {
        eprintln!("Warning: GPU support not compiled. Rebuild with --features cuda");
        eprintln!("Falling back to CPU mode...");
    }

    mosaic(
        Path::new(target),
        row_size,
        col_size,
        Path::new(images),
        Path::new(output),
        *color_space,
        avoid_duplicates,
    );
}
