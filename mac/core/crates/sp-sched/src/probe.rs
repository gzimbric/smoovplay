//! The pipeline's view of pixels: the block means the repeat and cut detectors compare, the delay test's 18x32
//! brightness blocks, and the sums --score adds up. [`GpuProbe`] does all three with small Metal kernels that read
//! the frames where they are. [`CpuProbe`] reads them back with `Frame::to_bgr`. It is slow, and serves as the
//! reference and for tests.
//!
//! The block-mean sums are exact, with each 8-bit level added as an integer and divided once on the CPU. So they
//! equal torch's avg_pool2d on the same uint8 frame bit for bit, which the parity traces rely on.
//! sp_core::stats::block_means adds normalised floats and comes out up to about 1e-4 off. That is enough to flip a
//! frame that changes by exactly the threshold (2 levels) from repeat to new. An HDR frame (RGBA16F, PQ) is taken at
//! its 8-bit stand-in, hdr.py's to_u8 (level8, truncated).

use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use sp_core::stats::{BLOCK, BlockMeans};
use sp_core::{Frame, Gpu, PixelKind};

use crate::delaytest::{self, BLOCKS};
use crate::score::{self, ScoreSums};

pub trait Probe: Send {
    /// 16x16 block means, 0 to 255, BGR (the DuplicateDetector's input).
    fn block_means(&mut self, frame: &Frame) -> BlockMeans;
    /// The delay test's 18x32 brightness blocks (framegen/delaytest.py small()).
    fn small(&mut self, frame: &Frame) -> Vec<f32>;
    /// --score's sums. `a` is the previous real frame, `b` the next one, `out` the AI frame and `gt` the real
    /// in-between one.
    fn score(&mut self, a: &Frame, b: &Frame, out: &Frame, gt: &Frame) -> ScoreSums;
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params { uint kind; float l8; uint step; uint seg; };

// A pixel's R, G and B in 8-bit levels. BGRA8 gives its 0 to 255 value, and RGBA16F (PQ) goes through hdr.py's
// level8.
inline float3 lv8(float4 p, constant Params &q) {
    return q.kind == 0 ? rint(p.rgb * 255.0) : clamp(p.rgb * q.l8, 0.0, 255.0);
}

// Exact 16x16 block sums of the 8-bit levels, B, G, R per block. An HDR frame uses its truncated to_u8 stand-in.
// One 16x16 threadgroup per block and one pixel per thread, added up in integers through the simdgroups.
kernel void block_sums(texture2d<float, access::read> src [[texture(0)]],
                       device uint *out [[buffer(0)]],
                       constant Params &q [[buffer(1)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 lid [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup uint3 part[32];
    uint w = src.get_width(), h = src.get_height();
    uint x = tg.x * 16 + lid.x, y = tg.y * 16 + lid.y;
    uint3 v = 0;
    if (x < w && y < h) {
        float3 f = lv8(src.read(uint2(x, y)), q);
        v = uint3(q.kind == 0 ? f : floor(f));
    }
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.x == 0 && lid.y == 0) {
        uint3 sum = 0;
        for (uint i = 0; i < nsg; i++) sum += part[i];
        uint i = (tg.y * ((w + 15) / 16) + tg.x) * 3;
        out[i] = sum.z; out[i + 1] = sum.y; out[i + 2] = sum.x;
    }
}

// The delay test's 18x32 blocks of brightness from every step-th pixel, averaged as adaptive_avg_pool2d does.
kernel void small_blocks(texture2d<float, access::read> src [[texture(0)]],
                         device float *out [[buffer(0)]],
                         constant Params &q [[buffer(1)]],
                         uint2 b [[thread_position_in_grid]]) {
    if (b.x >= 32 || b.y >= 18) return;
    uint w = src.get_width(), h = src.get_height(), s = q.step;
    uint hs = (h + s - 1) / s, ws = (w + s - 1) / s;
    uint y0 = (b.y * hs) / 18, y1 = ((b.y + 1) * hs + 17) / 18;
    uint x0 = (b.x * ws) / 32, x1 = ((b.x + 1) * ws + 31) / 32;
    float sum = 0.0;
    for (uint y = y0; y < y1; y++)
        for (uint x = x0; x < x1; x++) {
            float3 v = lv8(src.read(uint2(x * s, y * s)), q);
            sum += v.z * 0.114 + v.y * 0.587 + v.x * 0.299;
        }
    out[b.y * 32 + b.x] = sum / float((y1 - y0) * (x1 - x0));
}

inline float lum(texture2d<float, access::read> g, int x, int y, int w, int h, constant Params &q) {
    float3 v = lv8(g.read(uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1))), q);
    return v.z * 0.114 + v.y * 0.587 + v.x * 0.299;
}

inline float maxabs(float3 d) { d = abs(d); return max(d.x, max(d.y, d.z)); }

// --score's per-pixel numbers over a run of `seg` pixels in one row, as 11 partial sums per thread.
kernel void score_parts(texture2d<float, access::read> ta [[texture(0)]],
                        texture2d<float, access::read> tb [[texture(1)]],
                        texture2d<float, access::read> to [[texture(2)]],
                        texture2d<float, access::read> tg [[texture(3)]],
                        device float *out [[buffer(0)]],
                        constant Params &q [[buffer(1)]],
                        uint2 id [[thread_position_in_grid]]) {
    int w = tg.get_width(), h = tg.get_height();
    uint segs = (uint(w) + q.seg - 1) / q.seg;
    if (id.x >= segs || int(id.y) >= h) return;
    float s[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int y = id.y;
    int xend = min(int((id.x + 1) * q.seg), w);
    for (int x = id.x * q.seg; x < xend; x++) {
        uint2 p = uint2(x, y);
        float3 va = lv8(ta.read(p), q), vb = lv8(tb.read(p), q), vo = lv8(to.read(p), q), vg = lv8(tg.read(p), q);
        float3 eo = vo - vg, er = va - vg, eb = rint((va + vb) * 0.5) - vg;
        float se_o = eo.x * eo.x + eo.y * eo.y + eo.z * eo.z;
        float se_r = er.x * er.x + er.y * er.y + er.z * er.z;
        float se_b = eb.x * eb.x + eb.y * eb.y + eb.z * eb.z;
        bool still = maxabs(vb - va) <= 3.0;
        bool off = maxabs(eo) > 12.0, off_rep = maxabs(er) > 12.0;
        bool hud = false;
        if (still && maxabs(vg - va) <= 3.0) {
            float l00 = lum(tg, x - 1, y - 1, w, h, q), l01 = lum(tg, x, y - 1, w, h, q), l02 = lum(tg, x + 1, y - 1, w, h, q);
            float l10 = lum(tg, x - 1, y, w, h, q), l12 = lum(tg, x + 1, y, w, h, q);
            float l20 = lum(tg, x - 1, y + 1, w, h, q), l21 = lum(tg, x, y + 1, w, h, q), l22 = lum(tg, x + 1, y + 1, w, h, q);
            float gx = -l00 + l02 - 2.0 * l10 + 2.0 * l12 - l20 + l22;
            float gy = -l00 - 2.0 * l01 - l02 + l20 + 2.0 * l21 + l22;
            hud = abs(gx) + abs(gy) > 128.0;
        }
        s[0] += se_o; s[1] += se_r; s[2] += se_b;
        if (still) { s[3] += 1.0; s[4] += se_o; s[5] += se_r; }
        s[6] += off ? 1.0 : 0.0; s[7] += off_rep ? 1.0 : 0.0;
        if (hud) { s[8] += 1.0; s[9] += se_o; s[10] += off ? 1.0 : 0.0; }
    }
    device float *o = out + (id.y * segs + id.x) * 11;
    for (int k = 0; k < 11; k++) o[k] = s[k];
}
"#;

/// Pixels of one row each score thread adds up. Per-thread sums of 8-bit squared errors stay under 2^24, so the
/// float32 partials are exact and so is the CPU's float64 total.
const SEG: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    kind: u32,
    l8: f32,
    step: u32,
    seg: u32,
}

type Pipe = ProtocolObject<dyn MTLComputePipelineState>;

struct Kernels {
    block_sums: Retained<Pipe>,
    small: Retained<Pipe>,
    score: Retained<Pipe>,
}

// Metal pipeline states are thread-safe.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(|| {
        let gpu = Gpu::shared();
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe); // IEEE arithmetic, because the sums must match the CPU's
        let lib = gpu
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(SHADER), Some(&options))
            .unwrap_or_else(|e| panic!("sp-sched shaders: {}", e.localizedDescription()));
        let pipe = |name: &str| {
            let f = lib.newFunctionWithName(&NSString::from_str(name)).expect("shader function");
            gpu.device.newComputePipelineStateWithFunction_error(&f).expect("compute pipeline")
        };
        Kernels { block_sums: pipe("block_sums"), small: pipe("small_blocks"), score: pipe("score_parts") }
    })
}

fn params(kind: PixelKind) -> Params {
    Params {
        kind: if kind == PixelKind::Bgra8 { 0 } else { 1 },
        l8: score::level8_scale() as f32,
        step: 1,
        seg: SEG as u32,
    }
}

/// How a kernel is laid out: `Threads(w, h)` threads in 8x8 groups, or `Groups(w, h)` groups of 16x16.
#[derive(Clone, Copy)]
enum Grid {
    Threads(usize, usize),
    Groups(usize, usize),
}

/// Runs one kernel with `frames` as textures 0.. and an output buffer of `n` 4-byte values, and waits for it.
fn dispatch<T: Copy>(pipe: &Pipe, frames: &[&Frame], p: Params, grid: Grid, n: usize) -> Vec<T> {
    let gpu = Gpu::shared();
    unsafe {
        let out = gpu
            .device
            .newBufferWithLength_options(n.max(1) * 4, MTLResourceOptions::StorageModeShared)
            .expect("buffer");
        let cb = gpu.queue.commandBuffer().expect("command buffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipe);
        for (i, f) in frames.iter().enumerate() {
            enc.setTexture_atIndex(Some(&f.texture), i);
        }
        enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
        enc.setBytes_length_atIndex(NonNull::from(&p).cast(), std::mem::size_of::<Params>(), 1);
        match grid {
            Grid::Threads(w, h) => enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: w, height: h, depth: 1 },
                MTLSize { width: 8, height: 8, depth: 1 },
            ),
            Grid::Groups(w, h) => enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: w, height: h, depth: 1 },
                MTLSize { width: 16, height: 16, depth: 1 },
            ),
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        std::slice::from_raw_parts(out.contents().as_ptr() as *const T, n).to_vec()
    }
}

/// All three on the GPU. A block-means pass is one small dispatch (about 0.1 ms at 1080p). Scoring is one pass over
/// the four frames plus 2000-odd floats added up on the CPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuProbe;

impl GpuProbe {
    pub fn new() -> GpuProbe {
        kernels();
        GpuProbe
    }
}

impl Probe for GpuProbe {
    fn block_means(&mut self, frame: &Frame) -> BlockMeans {
        let (w, h) = (frame.width, frame.height);
        let (cols, rows) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
        let sums: Vec<u32> = dispatch(&kernels().block_sums, &[frame], params(frame.kind), Grid::Groups(cols, rows), rows * cols * 3);
        let mut data = vec![0f32; rows * cols * 3];
        for by in 0..rows {
            let ny = (BLOCK.min(h - by * BLOCK)) as f32;
            for bx in 0..cols {
                let count = ny * (BLOCK.min(w - bx * BLOCK)) as f32;
                for c in 0..3 {
                    let i = (by * cols + bx) * 3 + c;
                    data[i] = sums[i] as f32 / count;
                }
            }
        }
        BlockMeans { rows, cols, data }
    }

    fn small(&mut self, frame: &Frame) -> Vec<f32> {
        let mut p = params(frame.kind);
        p.step = delaytest::step(frame.height, frame.width) as u32;
        dispatch(&kernels().small, &[frame], p, Grid::Threads(BLOCKS.1, BLOCKS.0), BLOCKS.0 * BLOCKS.1)
    }

    fn score(&mut self, a: &Frame, b: &Frame, out: &Frame, gt: &Frame) -> ScoreSums {
        let (w, h) = (gt.width, gt.height);
        let segs = w.div_ceil(SEG);
        let parts: Vec<f32> = dispatch(&kernels().score, &[a, b, out, gt], params(gt.kind), Grid::Threads(segs, h), segs * h * 11);
        let mut t = [0f64; 11];
        for chunk in parts.as_chunks::<11>().0 {
            for k in 0..11 {
                t[k] += chunk[k] as f64;
            }
        }
        ScoreSums {
            se_out: t[0],
            se_rep: t[1],
            se_blend: t[2],
            n_still: t[3],
            se_out_still: t[4],
            se_rep_still: t[5],
            n_off: t[6],
            n_off_rep: t[7],
            n_hud: t[8],
            se_out_hud: t[9],
            n_off_hud: t[10],
            pixels: (w * h) as f64,
        }
    }
}

/// The same from frames read back to the CPU, for SDR frames only. It is slow at 1080p, about 30 ms for four 8 MB
/// read-backs and a scalar pass, so it is only for tests and as the reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuProbe;

impl Probe for CpuProbe {
    fn block_means(&mut self, frame: &Frame) -> BlockMeans {
        sp_core::stats::block_means_cpu(&frame.to_bgr(), frame.width, frame.height)
    }
    fn small(&mut self, frame: &Frame) -> Vec<f32> {
        delaytest::small_bgr(&frame.to_bgr(), frame.width, frame.height)
    }
    fn score(&mut self, a: &Frame, b: &Frame, out: &Frame, gt: &Frame) -> ScoreSums {
        score::score_sums_bgr(&a.to_bgr(), &b.to_bgr(), &out.to_bgr(), &gt.to_bgr(), gt.width, gt.height)
    }
}
