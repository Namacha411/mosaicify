# Mosaicify

Mosaicify is a command-line tool that generates a mosaic image from a target image and a set of source images. The tool allows you to create visually appealing mosaic art by dividing the target image into a grid and filling each cell with a source image.

## Features

- Generate mosaic images with customizable grid sizes.
- Option to avoid using duplicate images in the mosaic.
- Supports a wide range of image formats.
- Multithreaded processing for faster performance.
- **GPU acceleration with CUDA** for 10-50x speedup (optional).

## Installation

To install Mosaicify, you need to have [Rust](https://www.rust-lang.org/tools/install) installed. You can build the project from the source code using Cargo, Rust's package manager.

### CPU-Only Version (Default)

```sh
git clone https://github.com/yourusername/mosaicify.git
cd mosaicify
cargo install --path .
```

### GPU-Accelerated Version (with CUDA)

If you have an NVIDIA GPU and CUDA Toolkit installed:

```sh
git clone https://github.com/yourusername/mosaicify.git
cd mosaicify
cargo install --path . --features cuda
```

**Requirements**:
- NVIDIA GPU with CUDA Compute Capability 5.0 or higher
- CUDA Toolkit 12.0 or later
- Set `CUDA_PATH` environment variable (e.g., `/usr/local/cuda`)

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
- `-c`, `--color-space <COLOR_SPACE>`: Color space for matching tiles. Options: `rgb`, `lab` (default), `gray`.
- `-o`, `--output <OUTPUT>`: Output file path (default: `mosaic.jpg`).
- `-g`, `--gpu`: Enable GPU acceleration with CUDA (requires NVIDIA GPU and CUDA toolkit).

## Examples

### Basic Usage (CPU)

```sh
./mosaicify example/target.jpg 10 10 example/source_images/ -d
```

This command will generate a 10x10 mosaic using the images in the example/source_images/ directory, avoiding duplicate images in the mosaic.

### GPU Accelerated (10-50x faster)

```sh
./mosaicify example/target.jpg 50 50 example/source_images/ -g --color-space lab
```

This command uses GPU acceleration for significantly faster processing, especially with larger grids and many source images.

## GPU Acceleration

GPU acceleration provides significant performance improvements:

- **10-50x faster** than CPU-only mode
- Parallel color space conversion (RGB → Lab/Grayscale)
- Parallel similarity computation on thousands of CUDA cores
- Automatic fallback to CPU if CUDA is not available

### Requirements for GPU Mode

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 11.0 or later recommended)
- Set `CUDA_PATH` environment variable

**Note**: GPU mode is optional. The tool works perfectly fine in CPU mode without CUDA installed.

## License

This project is licensed under the terms of both the Apache License 2.0 and the MIT License. You may choose either license to use the project under.

- Apache License 2.0
- MIT License

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes.
