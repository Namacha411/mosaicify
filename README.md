# Mosaicify

Mosaicify is a command-line tool that generates a mosaic image from a target image and a set of source images. The tool allows you to create visually appealing mosaic art by dividing the target image into a grid and filling each cell with a source image.

## Features

- Generate mosaic images with customizable grid sizes.
- Option to avoid using duplicate images in the mosaic.
- Supports a wide range of image formats.
- Multithreaded processing for faster performance.

## Installation

To install Mosaicify, you need to have [Rust](https://www.rust-lang.org/tools/install) installed. You can build the project from the source code using Cargo, Rust's package manager.

```sh
git clone https://github.com/yourusername/mosaicify.git
cd mosaicify
cargo install --path .
```

## Usage

The following command generates a mosaic image:

```sh
./mosaicify <target_image> <row_size> <col_size> <source_images_directory> [OPTIONS]
```

### Arguments

- `<target_image>`: Path to the target image file.
- `<row_size>`: Number of rows in the mosaic grid.
- `<col_size>`: Number of columns in the mosaic grid.
- `<source_images_directory>`: Path to the directory containing the source images.

### Options

- `-d`, `--avoid-duplicates`: Avoid using duplicate images in the mosaic.

## Example

```sh
./mosaicify example/target.jpg 10 10 example/source_images/ -d
```

This command will generate a 10x10 mosaic using the images in the example/source_images/ directory, avoiding duplicate images in the mosaic.

## License

This project is licensed under the terms of both the Apache License 2.0 and the MIT License. You may choose either license to use the project under.

- Apache License 2.0
- MIT License

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes.
