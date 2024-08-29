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
