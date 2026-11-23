//! 16x16 block means of a frame, which the repeat and cut detectors compare (framegen/cadence.py
//! DuplicateDetector.update: `avg_pool2d(x, 16, ceil_mode=True)` on 0..255 BGR, edge blocks averaged over the pixels
//! they cover). Worked out on the GPU, read back as a few thousand floats.

use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice,
    MTLResourceOptions, MTLSize,
};

use crate::frame::{Frame, PixelKind};
use crate::gpu::{Gpu, Pipeline};

pub const BLOCK: usize = 16;

/// Block means, `rows` x `cols` blocks of 3 values each (B, G, R, 0..255), row-major: `data[(y * cols + x) * 3 + c]`.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockMeans {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One thread per block: the mean of the pixels it covers (fewer at the right and bottom edges), in 0..255, BGR.
// Summed as whole numbers, then divided once in float32, as torch's avg_pool2d does on 0..255 values, so the result
// is exact: a frame whose blocks change by exactly the repeat threshold is called the same way as in Python.
kernel void block_means(texture2d<float, access::read> src [[texture(0)]],
                        device float *out [[buffer(0)]],
                        constant uint &block [[buffer(1)]],
                        uint2 b [[thread_position_in_grid]]) {
    uint w = src.get_width(), h = src.get_height();
    uint cols = (w + block - 1) / block, rows = (h + block - 1) / block;
    if (b.x >= cols || b.y >= rows) return;
    uint x0 = b.x * block, y0 = b.y * block, x1 = min(x0 + block, w), y1 = min(y0 + block, h);
    uint3 sum = 0;
    for (uint y = y0; y < y1; y++)
        for (uint x = x0; x < x1; x++)
            sum += uint3(round(src.read(uint2(x, y)).bgr * 255.0));  // BGRA8 reads back as RGBA; .bgr is B, G, R
    float3 mean = float3(sum) / float((x1 - x0) * (y1 - y0));
    uint i = (b.y * cols + b.x) * 3;
    out[i] = mean.x; out[i + 1] = mean.y; out[i + 2] = mean.z;
}
"#;

fn pipeline() -> &'static Retained<Pipeline> {
    static P: OnceLock<Retained<Pipeline>> = OnceLock::new();
    P.get_or_init(|| {
        let gpu = Gpu::shared();
        let lib = gpu.library(SHADER).expect("block_means shader");
        gpu.pipeline(&lib, "block_means").expect("block_means pipeline")
    })
}

/// The frame's block means (an SDR frame; for HDR, pass its 8-bit stand-in, as the Python side does).
pub fn block_means(frame: &Frame) -> BlockMeans {
    assert_eq!(frame.kind, PixelKind::Bgra8, "block means are taken on 8-bit frames");
    let gpu = Gpu::shared();
    let cols = frame.width.div_ceil(BLOCK);
    let rows = frame.height.div_ceil(BLOCK);
    let n = rows * cols * 3;
    unsafe {
        let out = gpu
            .device
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .expect("buffer");
        let block = BLOCK as u32;
        let cb = gpu.queue.commandBuffer().expect("command buffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline());
        enc.setTexture_atIndex(Some(&frame.texture), 0);
        enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
        enc.setBytes_length_atIndex(NonNull::from(&block).cast(), 4, 1);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: cols, height: rows, depth: 1 },
            MTLSize { width: 8, height: 8, depth: 1 },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let data = std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, n).to_vec();
        BlockMeans { rows, cols, data }
    }
}

/// The same sums on the CPU, from packed BGR bytes: the reference the GPU version is tested against (equal to it
/// exactly).
pub fn block_means_cpu(bgr: &[u8], width: usize, height: usize) -> BlockMeans {
    let cols = width.div_ceil(BLOCK);
    let rows = height.div_ceil(BLOCK);
    let mut data = vec![0f32; rows * cols * 3];
    for by in 0..rows {
        for bx in 0..cols {
            let (x0, y0) = (bx * BLOCK, by * BLOCK);
            let (x1, y1) = ((x0 + BLOCK).min(width), (y0 + BLOCK).min(height));
            let mut s = [0u32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = (y * width + x) * 3;
                    for c in 0..3 {
                        s[c] += bgr[p + c] as u32;
                    }
                }
            }
            let count = ((x1 - x0) * (y1 - y0)) as f32;
            for c in 0..3 {
                data[(by * cols + bx) * 3 + c] = s[c] as f32 / count;
            }
        }
    }
    BlockMeans { rows, cols, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matches_cpu_with_partial_edge_blocks() {
        let (w, h) = (1920, 1080); // 1080 = 67.5 blocks: the last row of blocks is half height
        let bgr: Vec<u8> = (0..w * h * 3).map(|i| ((i * 2654435761usize) >> 13) as u8).collect();
        let gpu = block_means(&Frame::from_bgr(&bgr, w, h));
        let cpu = block_means_cpu(&bgr, w, h);
        assert_eq!((gpu.rows, gpu.cols), (68, 120));
        assert_eq!(gpu.data, cpu.data, "block means must be exact (whole-number sums, one float32 division)");
    }
}
