use std::{collections::BTreeSet, fs, path::Path};

use anyhow::Result;
use clap::{builder::PossibleValue, ValueEnum};
use image::{
    imageops::{crop_imm, replace, resize, FilterType::Lanczos3},
    DynamicImage, ImageReader, Luma, Pixel, Rgb, Rgb32FImage,
};
use indicatif::ProgressBar;
use itertools::{iproduct, Itertools};
use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;

use crate::lab::{Lab, PixelLabExt};
#[cfg(feature = "cuda")]
use crate::gpu::GpuProcessor;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ColorSpace {
    Rgb,
    Lab,
    Gray,
}

impl ValueEnum for ColorSpace {
    fn value_variants<'a>() -> &'a [Self] {
        &[ColorSpace::Rgb, ColorSpace::Lab, ColorSpace::Gray]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(match self {
            ColorSpace::Rgb => {
                PossibleValue::new("rgb").help("Use RGB color space for matching tiles.")
            }
            ColorSpace::Lab => PossibleValue::new("lab")
                .help("Use L*a*b* color space for more perceptually uniform matching."),
            ColorSpace::Gray => PossibleValue::new("gray")
                .help("Use grayscale for matching tiles based on intensity."),
        })
    }
}

impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

/// GPU-accelerated mosaic generation
#[cfg(feature = "cuda")]
pub(crate) fn mosaic_gpu(
    target: &Path,
    row_size: u32,
    col_size: u32,
    images: &Path,
    output: &Path,
    color_space: ColorSpace,
    avoid_duplicates: bool,
) {
    println!("[GPU Mode] Initializing CUDA...");
    let gpu = match GpuProcessor::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}. Falling back to CPU.", e);
            return mosaic(target, row_size, col_size, images, output, color_space, avoid_duplicates);
        }
    };
    println!("[GPU Mode] CUDA initialized successfully.");

    println!("[1/3] Preprocessing the target image.");
    let target = ImageReader::open(target)
        .expect("Failed to open the target image.")
        .decode()
        .expect("Failed to decode the target image.")
        .into_rgb32f();
    let width = target.width() / row_size;
    let height = target.height() / col_size;
    let mut target = resize(&target, width * row_size, height * col_size, Lanczos3);
    println!("[1/3] Finished preprocessing the target image.");

    println!("[2/3] Preprocessing the source images.");
    let images =
        read_images_from_directory(images).expect("Failed to read images from the directory.");

    let pb = ProgressBar::new(images.len() as u64);

    // Resize images on CPU (still parallel)
    let images_resized = images
        .par_iter()
        .map(|img| {
            pb.inc(1);
            resize(img, width, height, Lanczos3)
        })
        .collect::<Vec<_>>();
    pb.finish_and_clear();

    // Convert to flat f32 arrays for GPU processing
    println!("[2/3] Converting images to GPU format...");
    let pb = ProgressBar::new(images_resized.len() as u64);

    let images_flat: Vec<Vec<f32>> = images_resized
        .iter()
        .map(|img| {
            pb.inc(1);
            image_to_flat_rgb(img)
        })
        .collect();
    pb.finish_and_clear();

    // Process color space conversion on GPU
    println!("[2/3] Converting color space on GPU...");
    let pb = ProgressBar::new(images_flat.len() as u64);

    let images_converted: Vec<(Rgb32FImage, Vec<f32>)> = match color_space {
        ColorSpace::Lab => {
            images_resized
                .into_iter()
                .zip(images_flat.iter())
                .map(|(img, flat)| {
                    pb.inc(1);
                    let converted = gpu.rgb_to_lab(flat).expect("GPU Lab conversion failed");
                    (img, converted)
                })
                .collect()
        }
        ColorSpace::Gray => {
            images_resized
                .into_iter()
                .zip(images_flat.iter())
                .map(|(img, flat)| {
                    pb.inc(1);
                    let converted = gpu.rgb_to_gray(flat).expect("GPU Gray conversion failed");
                    (img, converted)
                })
                .collect()
        }
        ColorSpace::Rgb => {
            images_resized
                .into_iter()
                .zip(images_flat.into_iter())
                .map(|(img, flat)| {
                    pb.inc(1);
                    (img, flat)
                })
                .collect()
        }
    };
    pb.finish_and_clear();

    let mut used = BTreeSet::new();
    println!("[2/3] Finished preprocessing the source images.");

    println!("[3/3] Generating the mosaic image using GPU...");
    let mut rng = thread_rng();
    let mut block_index = iproduct!(0..col_size, 0..row_size).collect_vec();
    block_index.shuffle(&mut rng);
    let pb = ProgressBar::new(block_index.len() as u64);

    let channels = match color_space {
        ColorSpace::Gray => 1,
        _ => 3,
    };

    for (y, x) in block_index {
        if avoid_duplicates && used.len() == images_converted.len() {
            used.clear();
        }

        let block = crop_imm(&target, x * width, y * height, width, height);
        let block_image = block.to_image();
        let block_flat = image_to_flat_rgb(&block_image);

        // Convert block to same color space
        let block_converted = match color_space {
            ColorSpace::Lab => gpu.rgb_to_lab(&block_flat).expect("GPU Lab conversion failed"),
            ColorSpace::Gray => gpu.rgb_to_gray(&block_flat).expect("GPU Gray conversion failed"),
            ColorSpace::Rgb => block_flat,
        };

        // Prepare data for GPU similarity computation
        let available_images: Vec<(usize, &Vec<f32>)> = images_converted
            .iter()
            .enumerate()
            .filter(|(i, _)| !avoid_duplicates || !used.contains(i))
            .map(|(i, (_, col))| (i, col))
            .collect();

        if available_images.is_empty() {
            continue;
        }

        // Flatten all available images for GPU
        let images_data: Vec<f32> = available_images
            .iter()
            .flat_map(|(_, col)| col.iter().copied())
            .collect();

        // Find best match on GPU
        let (local_idx, _score) = gpu
            .find_best_match(
                &block_converted,
                &images_data,
                width,
                height,
                channels,
                available_images.len(),
            )
            .expect("GPU similarity computation failed");

        let (original_idx, _) = available_images[local_idx];

        if avoid_duplicates {
            used.insert(original_idx);
        }

        let best = &images_converted[original_idx].0;
        replace(&mut target, best, (x * width) as i64, (y * height) as i64);
        pb.inc(1);
    }

    DynamicImage::ImageRgb32F(target)
        .to_rgb8()
        .save(output)
        .expect("Failed to save the mosaic image.");
    pb.finish_and_clear();
    println!("[3/3] Finished generating the mosaic image.");
    println!("All done.");
}

pub(crate) fn mosaic(
    target: &Path,
    row_size: u32,
    col_size: u32,
    images: &Path,
    output: &Path,
    color_space: ColorSpace,
    avoid_duplicates: bool,
) {
    println!("[1/3] Preprocessing the target image.");
    let target = ImageReader::open(target)
        .expect("Failed to open the target image.")
        .decode()
        .expect("Failed to decode the target image.")
        .into_rgb32f();
    let width = target.width() / row_size;
    let height = target.height() / col_size;
    // いろ空間の変更
    let mut target = resize(&target, width * row_size, height * col_size, Lanczos3);
    println!("[1/3] Finished preprocessing the target image.");

    println!("[2/3] Preprocessing the source images.");
    let images =
        read_images_from_directory(images).expect("Failed to read images from the directory.");
    let color_space = match color_space {
        ColorSpace::Rgb => rgb_identity,
        ColorSpace::Lab => rgb2lab,
        ColorSpace::Gray => rgb2gray,
    };
    let pb = ProgressBar::new(images.len() as u64);
    let images = images
        .par_iter()
        .map(|img| {
            pb.inc(1);
            let img = resize(img, width, height, Lanczos3);
            let col = color_space(&img);
            (img, col)
        })
        .collect::<Vec<_>>();
    let mut used = BTreeSet::new();
    pb.finish_and_clear();
    println!("[2/3] Finished preprocessing the source images.");

    println!("[3/3] Generating the mosaic image.");
    let mut rng = thread_rng();
    let mut block_index = iproduct!(0..col_size, 0..row_size).collect_vec();
    block_index.shuffle(&mut rng);
    let pb = ProgressBar::new(block_index.len() as u64);
    for (y, x) in block_index {
        if avoid_duplicates && used.len() == images.len() {
            used.clear();
        }
        let block = crop_imm(&target, x * width, y * height, width, height);
        let block_image = block.to_image();
        let (_score, idx, best) = images
            .par_iter()
            .enumerate()
            .filter_map(|(i, (img, col))| {
                if avoid_duplicates && used.contains(&i) {
                    return None;
                }
                let block_col = color_space(&block_image);
                similarity(&block_col, col).map(|s| (s, i, img))
            })
            .min_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Failed to compare similarity scores.")
            })
            .expect("Failed to find the best matching image.");
        if avoid_duplicates {
            used.insert(idx);
        }
        replace(&mut target, best, (x * width) as i64, (y * height) as i64);
        pb.inc(1);
    }
    DynamicImage::ImageRgb32F(target)
        .to_rgb8()
        .save(output)
        .expect("Failed to save the mosaic image.");
    pb.finish_and_clear();
    println!("[3/3] Finished generating the mosaic image.");
    println!("All done.");
}

fn read_images_from_directory(directory: &Path) -> Result<Vec<Rgb32FImage>> {
    let mut images = vec![];
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let img = ImageReader::open(path)?.decode()?.into_rgb32f();
        images.push(img);
    }
    Ok(images)
}

fn rgb_identity(image: &Rgb32FImage) -> Vec<Vec<Vec<f32>>> {
    let mut tmp = vec![vec![vec![]; image.height() as usize]; image.width() as usize];
    for (x, y, p) in image.enumerate_pixels() {
        let Rgb(rgb) = p;
        tmp[x as usize][y as usize] = rgb.to_vec();
    }
    tmp
}

fn rgb2lab(image: &Rgb32FImage) -> Vec<Vec<Vec<f32>>> {
    let mut tmp = vec![vec![vec![]; image.height() as usize]; image.width() as usize];
    for (x, y, p) in image.enumerate_pixels() {
        let Lab(lab) = p.to_lab();
        tmp[x as usize][y as usize] = lab.to_vec();
    }
    tmp
}

fn rgb2gray(image: &Rgb32FImage) -> Vec<Vec<Vec<f32>>> {
    let mut tmp = vec![vec![vec![]; image.height() as usize]; image.width() as usize];
    for (x, y, p) in image.enumerate_pixels() {
        let Luma(luma) = p.to_luma();
        tmp[x as usize][y as usize] = luma.to_vec();
    }
    tmp
}

fn similarity(a: &[Vec<Vec<f32>>], b: &[Vec<Vec<f32>>]) -> Option<f32> {
    if !(a.len() == b.len() && a[0].len() == b[0].len()) {
        return None;
    }
    let s = iproduct!(0..a.len(), 0..a[0].len())
        .map(|(x, y)| {
            a[x][y]
                .iter()
                .zip(&b[x][y])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .sum();
    Some(s)
}

/// Convert RGB image to flat f32 array for GPU processing
#[cfg(feature = "cuda")]
fn image_to_flat_rgb(image: &Rgb32FImage) -> Vec<f32> {
    let mut flat = Vec::with_capacity((image.width() * image.height() * 3) as usize);
    for pixel in image.pixels() {
        let Rgb(rgb) = pixel;
        flat.extend_from_slice(rgb);
    }
    flat
}
