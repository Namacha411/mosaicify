use std::{collections::BTreeSet, fs, path::Path};

use anyhow::Result;
use clap::{arg, builder::PossibleValue, value_parser, Arg, ArgAction, Command, ValueEnum};
use image::{
    imageops::{crop_imm, replace, resize, FilterType::Lanczos3},
    ImageReader, Rgb, RgbImage,
};
use indicatif::ProgressBar;
use itertools::{iproduct, Itertools};
use num::pow::Pow;
use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;

fn main() {
    let matches = Command::new("mosaicify")
        .version("0.2.0")
        .author("Namacha411 <thdyk.4.11@gmail.com>")
        .about("Generates a mosaic image from a target image and a set of source images.")
        .arg(
            Arg::new("target")
                .help("Path to the target image")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("row_size")
                .help("Number of rows in the mosaic")
                .required(true)
                .index(2)
                .value_parser(clap::value_parser!(u32)),
        )
        .arg(
            Arg::new("col_size")
                .help("Number of columns in the mosaic")
                .required(true)
                .index(3)
                .value_parser(clap::value_parser!(u32)),
        )
        .arg(
            Arg::new("images")
                .help("Path to the directory containing source images")
                .required(true)
                .index(4),
        )
        .arg(
            arg!(-c --color_space [COLOR_SPACE] "Color space to use for matching tiles. Options: 'rgb' for RGB space, 'lab' for Lab space, 'gray' for grayscale.")
                .value_parser(value_parser!(ColorSpace))
                .default_value("lab"),
        )
        .arg(
            arg!(-o --output [OUTPUT] "output image file path")
                .default_value("mosaic.jpg")
        )
        .arg(
            Arg::new("avoid_duplicates")
                .help("Avoid using duplicate images in the mosaic")
                .short('d')
                .long("avoid-duplicates")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let target = matches.get_one::<String>("target").expect("required");
    let row_size = *matches.get_one::<u32>("row_size").expect("required");
    let col_size = *matches.get_one::<u32>("col_size").expect("required");
    let images = matches.get_one::<String>("images").expect("required");
    let output = matches.get_one::<String>("output").expect("required");
    let color_space = matches
        .get_one::<ColorSpace>("color_space")
        .expect("required");
    let avoid_duplicates = matches.get_flag("avoid_duplicates");

    mosaic(
        Path::new(target),
        row_size,
        col_size,
        Path::new(images),
        Path::new(output),
        color_space.to_fn(),
        avoid_duplicates,
    );
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ColorSpace {
    Rgb,
    Lab,
    Gray,
}

impl ColorSpace {
    fn to_fn(&self) -> fn(&[u8; 3]) -> [f64; 3] {
        match self {
            ColorSpace::Rgb => identity_rgb,
            ColorSpace::Lab => rgb2lab,
            ColorSpace::Gray => rgb2gray,
        }
    }
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

fn mosaic(
    target: &Path,
    row_size: u32,
    col_size: u32,
    images: &Path,
    output: &Path,
    color_space: fn(&[u8; 3]) -> [f64; 3],
    avoid_duplicates: bool,
) {
    println!("[1/3] Preprocessing the target image.");
    let target = ImageReader::open(target)
        .expect("Failed to open the target image.")
        .decode()
        .expect("Failed to decode the target image.")
        .into_rgb8();
    let width = target.width() / row_size;
    let height = target.height() / col_size;
    let mut target = resize(&target, width * row_size, height * col_size, Lanczos3);
    println!("[1/3] Finished preprocessing the target image.");

    println!("[2/3] Preprocessing the source images.");
    let images =
        read_images_from_directory(images).expect("Failed to read images from the directory.");
    let pb = ProgressBar::new(images.len() as u64);
    let images = images
        .par_iter()
        .map(|img| {
            pb.inc(1);
            small_lab(&img, width, height)
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
            .filter_map(|(i, img)| {
                if avoid_duplicates && used.contains(&i) {
                    return None;
                }
                if let Some(s) = similarity(&block_image, img, color_space) {
                    Some((s, i, img))
                } else {
                    None
                }
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
    target
        .save(output)
        .expect("Failed to save the mosaic image.");
    pb.finish_and_clear();
    println!("[3/3] Finished generating the mosaic image.");
    println!("All done.");
}

fn read_images_from_directory(directory: &Path) -> Result<Vec<RgbImage>> {
    let mut images = vec![];
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let img = ImageReader::open(path)?.decode()?.into_rgb8();
        images.push(img);
    }
    Ok(images)
}

fn small_lab(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    resize(img, width, height, Lanczos3)
}

fn similarity<F>(a: &RgbImage, b: &RgbImage, color_transform: F) -> Option<f64>
where
    F: Fn(&[u8; 3]) -> [f64; 3],
{
    if !(a.height() == b.height() && a.width() == b.width()) {
        return None;
    }
    let s = a
        .enumerate_pixels()
        .map(|(x, y, pixel)| {
            let Rgb(a_rgb) = pixel;
            let Rgb(b_rgb) = b.get_pixel(x, y);
            let a_tr = color_transform(a_rgb);
            let b_tr = color_transform(b_rgb);
            (0..3)
                .map(|i| (a_tr[i] - b_tr[i]).pow(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum();
    Some(s)
}

fn rgb2gray(rgb: &[u8; 3]) -> [f64; 3] {
    let [r, g, b] = rgb.map(|c| c as f64);
    let gray = r * 0.3 + g * 0.59 + b * 0.11;
    [gray, 0.0, 0.0]
}

fn identity_rgb(rgb: &[u8; 3]) -> [f64; 3] {
    rgb.map(|c| c as f64)
}

/// https://en.wikipedia.org/wiki/CIELAB_color_space
/// lを２倍に
fn rgb2lab(rgb: &[u8; 3]) -> [f64; 3] {
    let [mut r, mut g, mut b] = rgb.map(|c| c as f64 / 255.0);
    r = if r > 0.04045 {
        f64::powf((r + 0.055) / 1.055, 2.4)
    } else {
        r / 12.92
    };
    g = if g > 0.04045 {
        f64::powf((g + 0.055) / 1.055, 2.4)
    } else {
        g / 12.92
    };
    b = if b > 0.04045 {
        f64::powf((b + 0.055) / 1.055, 2.4)
    } else {
        b / 12.92
    };
    let mut x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    let mut y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
    let mut z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
    x = if x > 0.008856 {
        f64::powf(x, 1.0 / 3.0)
    } else {
        (7.787 * x) + 16.0 / 116.0
    };
    y = if y > 0.008856 {
        f64::powf(y, 1.0 / 3.0)
    } else {
        (7.787 * y) + 16.0 / 116.0
    };
    z = if z > 0.008856 {
        f64::powf(z, 1.0 / 3.0)
    } else {
        (7.787 * z) + 16.0 / 116.0
    };
    let l = (116.0 * y) - 16.0;
    let a = 500.0 * (x - y);
    let b = 200.0 * (y - z);
    [2.0 * l, a, b]
}
