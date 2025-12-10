#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use cudarc::driver::*;
use std::sync::Arc;

/// CUDA kernel for RGB to Lab color space conversion
const RGB2LAB_KERNEL: &str = r#"
extern "C" __global__ void rgb2lab(
    const float* rgb,
    float* lab,
    int num_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int base = idx * 3;
    float r = rgb[base] / 255.0f;
    float g = rgb[base + 1] / 255.0f;
    float b = rgb[base + 2] / 255.0f;

    // sRGB to linear RGB
    r = (r > 0.04045f) ? powf((r + 0.055f) / 1.055f, 2.4f) : r / 12.92f;
    g = (g > 0.04045f) ? powf((g + 0.055f) / 1.055f, 2.4f) : g / 12.92f;
    b = (b > 0.04045f) ? powf((b + 0.055f) / 1.055f, 2.4f) : b / 12.92f;

    // RGB to XYZ
    float x = (r * 0.4124f + g * 0.3576f + b * 0.1805f) / 0.95047f;
    float y = (r * 0.2126f + g * 0.7152f + b * 0.0722f) / 1.00000f;
    float z = (r * 0.0193f + g * 0.1192f + b * 0.9505f) / 1.08883f;

    // XYZ to Lab
    x = (x > 0.008856f) ? powf(x, 1.0f / 3.0f) : (7.787f * x) + 16.0f / 116.0f;
    y = (y > 0.008856f) ? powf(y, 1.0f / 3.0f) : (7.787f * y) + 16.0f / 116.0f;
    z = (z > 0.008856f) ? powf(z, 1.0f / 3.0f) : (7.787f * z) + 16.0f / 116.0f;

    float l = (116.0f * y) - 16.0f;
    float a = 500.0f * (x - y);
    float b_val = 200.0f * (y - z);

    lab[base] = 2.0f * l;
    lab[base + 1] = a;
    lab[base + 2] = b_val;
}
"#;

/// CUDA kernel for RGB to Grayscale conversion
const RGB2GRAY_KERNEL: &str = r#"
extern "C" __global__ void rgb2gray(
    const float* rgb,
    float* gray,
    int num_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int rgb_base = idx * 3;
    float r = rgb[rgb_base];
    float g = rgb[rgb_base + 1];
    float b = rgb[rgb_base + 2];

    // Standard luminance formula
    gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
}
"#;

/// CUDA kernel for computing similarity (L2 distance) between images
const SIMILARITY_KERNEL: &str = r#"
extern "C" __global__ void compute_similarity(
    const float* block,
    const float* images,
    float* distances,
    int width,
    int height,
    int channels,
    int num_images
) {
    int img_idx = blockIdx.x;
    if (img_idx >= num_images) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int num_pixels = width * height;

    extern __shared__ float shared_sum[];

    float local_sum = 0.0f;

    // Each thread processes multiple pixels
    for (int i = tid; i < num_pixels; i += block_size) {
        float pixel_dist = 0.0f;
        for (int c = 0; c < channels; c++) {
            int idx = i * channels + c;
            int img_offset = img_idx * num_pixels * channels;
            float diff = block[idx] - images[img_offset + idx];
            pixel_dist += diff * diff;
        }
        local_sum += sqrtf(pixel_dist);
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Write result for this image
    if (tid == 0) {
        distances[img_idx] = shared_sum[0];
    }
}
"#;

/// GPU accelerated mosaic processor
pub struct GpuProcessor {
    device: Arc<CudaDevice>,
    rgb2lab_func: CudaFunction,
    rgb2gray_func: CudaFunction,
    similarity_func: CudaFunction,
}

impl GpuProcessor {
    /// Initialize GPU processor with CUDA device
    pub fn new() -> Result<Self> {
        // Initialize CUDA device
        let device = CudaDevice::new(0).context("Failed to initialize CUDA device")?;

        // Compile and load kernels
        let ptx = compile_ptx(RGB2LAB_KERNEL).context("Failed to compile rgb2lab kernel")?;
        device.load_ptx(ptx.clone(), "rgb2lab_module", &["rgb2lab"])?;
        let rgb2lab_func = device.get_func("rgb2lab_module", "rgb2lab").unwrap();

        let ptx = compile_ptx(RGB2GRAY_KERNEL).context("Failed to compile rgb2gray kernel")?;
        device.load_ptx(ptx.clone(), "rgb2gray_module", &["rgb2gray"])?;
        let rgb2gray_func = device.get_func("rgb2gray_module", "rgb2gray").unwrap();

        let ptx = compile_ptx(SIMILARITY_KERNEL).context("Failed to compile similarity kernel")?;
        device.load_ptx(ptx.clone(), "similarity_module", &["compute_similarity"])?;
        let similarity_func = device.get_func("similarity_module", "compute_similarity").unwrap();

        Ok(Self {
            device,
            rgb2lab_func,
            rgb2gray_func,
            similarity_func,
        })
    }

    /// Convert RGB image to Lab color space on GPU
    pub fn rgb_to_lab(&self, rgb_data: &[f32]) -> Result<Vec<f32>> {
        let num_pixels = rgb_data.len() / 3;

        // Allocate GPU memory
        let rgb_gpu = self.device.htod_copy(rgb_data.to_vec())?;
        let lab_gpu = self.device.alloc_zeros::<f32>(rgb_data.len())?;

        // Launch kernel
        let block_size = 256;
        let grid_size = (num_pixels + block_size - 1) / block_size;

        unsafe {
            self.rgb2lab_func.launch(
                LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&rgb_gpu, &lab_gpu, num_pixels as i32),
            )?;
        }

        // Copy result back
        let result = self.device.dtoh_sync_copy(&lab_gpu)?;
        Ok(result)
    }

    /// Convert RGB image to grayscale on GPU
    pub fn rgb_to_gray(&self, rgb_data: &[f32]) -> Result<Vec<f32>> {
        let num_pixels = rgb_data.len() / 3;

        // Allocate GPU memory
        let rgb_gpu = self.device.htod_copy(rgb_data.to_vec())?;
        let gray_gpu = self.device.alloc_zeros::<f32>(num_pixels)?;

        // Launch kernel
        let block_size = 256;
        let grid_size = (num_pixels + block_size - 1) / block_size;

        unsafe {
            self.rgb2gray_func.launch(
                LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&rgb_gpu, &gray_gpu, num_pixels as i32),
            )?;
        }

        // Copy result back
        let result = self.device.dtoh_sync_copy(&gray_gpu)?;
        Ok(result)
    }

    /// Find the best matching image from a set of images using GPU
    pub fn find_best_match(
        &self,
        block_data: &[f32],
        images_data: &[f32],
        width: u32,
        height: u32,
        channels: u32,
        num_images: usize,
    ) -> Result<(usize, f32)> {
        // Allocate GPU memory
        let block_gpu = self.device.htod_copy(block_data.to_vec())?;
        let images_gpu = self.device.htod_copy(images_data.to_vec())?;
        let distances_gpu = self.device.alloc_zeros::<f32>(num_images)?;

        // Launch kernel
        let block_size = 256;
        let shared_mem_size = block_size * std::mem::size_of::<f32>();

        unsafe {
            self.similarity_func.launch(
                LaunchConfig {
                    grid_dim: (num_images as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: shared_mem_size as u32,
                },
                (&block_gpu, &images_gpu, &distances_gpu, width as i32, height as i32, channels as i32, num_images as i32),
            )?;
        }

        // Copy results back
        let distances = self.device.dtoh_sync_copy(&distances_gpu)?;

        // Find minimum distance on CPU (small array)
        let (min_idx, min_dist) = distances
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((min_idx, *min_dist))
    }

    /// Batch process: find best matches for multiple blocks
    pub fn find_best_matches_batch(
        &self,
        blocks_data: &[f32],
        images_data: &[f32],
        width: u32,
        height: u32,
        channels: u32,
        num_blocks: usize,
        num_images: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let block_size = (width * height * channels) as usize;
        let mut results = Vec::with_capacity(num_blocks);

        // Upload images data once
        let images_gpu = self.device.htod_copy(images_data.to_vec())?;

        for block_idx in 0..num_blocks {
            let block_start = block_idx * block_size;
            let block_end = block_start + block_size;
            let block_data = &blocks_data[block_start..block_end];

            let block_gpu = self.device.htod_copy(block_data.to_vec())?;
            let distances_gpu = self.device.alloc_zeros::<f32>(num_images)?;

            let thread_block_size = 256;
            let shared_mem_size = thread_block_size * std::mem::size_of::<f32>();

            unsafe {
                self.similarity_func.launch(
                    LaunchConfig {
                        grid_dim: (num_images as u32, 1, 1),
                        block_dim: (thread_block_size as u32, 1, 1),
                        shared_mem_bytes: shared_mem_size as u32,
                    },
                    (&block_gpu, &images_gpu, &distances_gpu, width as i32, height as i32, channels as i32, num_images as i32),
                )?;
            }

            let distances = self.device.dtoh_sync_copy(&distances_gpu)?;
            let (min_idx, min_dist) = distances
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            results.push((min_idx, *min_dist));
        }

        Ok(results)
    }
}

/// Compile CUDA kernel source to PTX
fn compile_ptx(src: &str) -> Result<cudarc::nvrtc::Ptx> {
    use cudarc::nvrtc::compile_ptx;
    compile_ptx(src).context("Failed to compile CUDA kernel to PTX")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run if CUDA is available
    fn test_gpu_processor_init() {
        let processor = GpuProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    #[ignore]
    fn test_rgb_to_lab() {
        let processor = GpuProcessor::new().unwrap();
        let rgb = vec![255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0];
        let lab = processor.rgb_to_lab(&rgb).unwrap();
        assert_eq!(lab.len(), 9);
    }
}
