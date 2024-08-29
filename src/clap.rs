use clap::{arg, value_parser, Arg, ArgAction, ArgMatches, Command};

use crate::mosaic::ColorSpace;

pub fn get_matches() -> ArgMatches {
    Command::new("mosaicify")
        .version("0.3.0")
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
        .get_matches()
}
