//! The replay's GPU side: frames into the encoder's NV12 buffers (BT.709 limited range, replay.py's sums: Y per
//! pixel, Cb/Cr from the mean of each 2x2 block), and the before & after picture, composed straight from the decoded
//! NV12 into the encoder's. Pixel buffers are seen as Metal textures through a CVMetalTextureCache (no copies).

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBuffer, CVPixelBufferGetHeightOfPlane,
    CVPixelBufferGetWidthOfPlane, kCVMetalTextureUsage,
};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLPixelFormat, MTLSize,
    MTLTextureUsage,
};
use sp_core::Gpu;
use sp_core::gpu::Pipeline;
use sp_media::cf;

use crate::{Error, Result, yuv};

pub type Texture = ProtocolObject<dyn objc2_metal::MTLTexture>;
pub type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;

const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ToNv12 { float4 ky; float4 ku; float4 kv; uint w; uint h; uint pad0; uint pad1; };

// A frame (BGRA8, or anything read as RGBA 0..1) into NV12 planes, BT.709 limited range: replay.py bgr_to_i420.
kernel void to_nv12(texture2d<float, access::read> src [[texture(0)]],
                    texture2d<float, access::write> oy [[texture(1)]],
                    texture2d<float, access::write> ouv [[texture(2)]],
                    constant ToNv12& p [[buffer(0)]],
                    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w / 2 || g.y >= p.h / 2) return;
    uint2 o = g * 2;
    float3 sum = float3(0.0);
    for (uint dy = 0; dy < 2; dy++) {
        for (uint dx = 0; dx < 2; dx++) {
            uint2 q = o + uint2(dx, dy);
            float4 c = src.read(q);
            float3 bgr = round(float3(c.b, c.g, c.r) * 255.0);
            float y = dot(bgr, p.ky.xyz) + 16.0;
            oy.write(float4(clamp(rint(y), 0.0, 255.0) / 255.0), q);
            sum += bgr;
        }
    }
    float3 m = sum * 0.25;
    float u = dot(m, p.ku.xyz) + 128.0;
    float v = dot(m, p.kv.xyz) + 128.0;
    ouv.write(float4(clamp(rint(u), 0.0, 255.0) / 255.0, clamp(rint(v), 0.0, 255.0) / 255.0, 0.0, 0.0), g);
}

struct Compose {
    float4 amber;   // Y, U, V of the divider
    uint ow, oh;    // the output (even)
    uint layout;    // 0 split, 1 side by side
    uint w, h;      // the after picture's even size (what's shown of it)
    uint hw;        // side by side: each half's width
    uint aw, ah;    // after's luma size
    uint bw, bh;    // before's luma size
    uint mid;
    int lbx, lby, lbw, lbh;   // the before label: where, and its size
    int lax, lay, law, lah;   // the after label
    uint has_b;
};

static float4 px(texture2d<float, access::read> t, int2 q) {
    q = clamp(q, int2(0), int2(int(t.get_width()) - 1, int(t.get_height()) - 1));
    return round(t.read(uint2(q)) * 255.0);
}

// Pixel p of a dsize picture made from the (0, 0, src) region of plane t: copied when the sizes match, else the mean
// of the area it covers (shrinking, as OpenCV's INTER_AREA) or bilinear (growing). In 0..255.
static float4 fetch(texture2d<float, access::read> t, float2 src, float2 dsize, uint2 p) {
    if (abs(src.x - dsize.x) < 0.5 && abs(src.y - dsize.y) < 0.5) return px(t, int2(p));
    float2 s = src / dsize;
    if (s.x <= 1.0 && s.y <= 1.0) {
        float2 c = (float2(p) + 0.5) * s - 0.5;
        int2 i = int2(floor(c));
        float2 f = c - floor(c);
        float4 a = mix(px(t, i), px(t, i + int2(1, 0)), f.x);
        float4 b = mix(px(t, i + int2(0, 1)), px(t, i + int2(1, 1)), f.x);
        return mix(a, b, f.y);
    }
    float2 a0 = float2(p) * s, a1 = a0 + s;
    float4 sum = float4(0.0);
    float wsum = 0.0;
    for (int y = int(floor(a0.y)); y < int(ceil(a1.y)); y++) {
        float wy = min(float(y + 1), a1.y) - max(float(y), a0.y);
        for (int x = int(floor(a0.x)); x < int(ceil(a1.x)); x++) {
            float wx = min(float(x + 1), a1.x) - max(float(x), a0.x);
            sum += px(t, int2(x, y)) * (wx * wy);
            wsum += wx * wy;
        }
    }
    return sum / max(wsum, 1e-6);
}

static bool inside(int2 q, int x, int y, int w, int h) {
    return q.x >= x && q.y >= y && q.x < x + w && q.y < y + h;
}

// The before & after picture (replay.py compose): the game's frame on the left, SmoovPlay's on the right, split down
// the middle or side by side, the amber line between, the labels on translucent boxes.
kernel void compose(texture2d<float, access::read> ay [[texture(0)]],
                    texture2d<float, access::read> auv [[texture(1)]],
                    texture2d<float, access::read> by [[texture(2)]],
                    texture2d<float, access::read> buv [[texture(3)]],
                    texture2d<float, access::read> lb [[texture(4)]],
                    texture2d<float, access::read> la [[texture(5)]],
                    texture2d<float, access::write> oy [[texture(6)]],
                    texture2d<float, access::write> ouv [[texture(7)]],
                    constant Compose& p [[buffer(0)]],
                    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.ow / 2 || g.y >= p.oh / 2) return;
    float2 a_src = float2(p.w, p.h);
    float2 b_src = a_src * float2(p.bw, p.bh) / float2(p.aw, p.ah);
    float2 dsize = p.layout == 0 ? float2(p.w, p.h) : float2(p.hw, p.oh);
    int mid = int(p.mid);
    float alpha_b = 0.0, alpha_a = 0.0;
    for (uint dy = 0; dy < 2; dy++) {
        for (uint dx = 0; dx < 2; dx++) {
            uint2 q = g * 2 + uint2(dx, dy);
            bool before = p.layout == 0 ? q.x < p.w / 2 : q.x < p.hw;
            uint2 at = (p.layout == 1 && !before) ? uint2(q.x - p.hw, q.y) : q;
            float v = (before && p.has_b != 0) ? fetch(by, b_src, dsize, at).x : fetch(ay, a_src, dsize, at).x;
            v = rint(v);
            int2 qi = int2(q);
            if (qi.x >= mid - 2 && qi.x < mid + 2) v = p.amber.x;
            if (inside(qi, p.lbx, p.lby, p.lbw, p.lbh)) {
                float2 l = lb.read(uint2(qi - int2(p.lbx, p.lby))).xy;
                v = rint(l.y * l.x + (1.0 - l.y) * v);
                alpha_b += l.y;
            }
            if (inside(qi, p.lax, p.lay, p.law, p.lah)) {
                float2 l = la.read(uint2(qi - int2(p.lax, p.lay))).xy;
                v = rint(l.y * l.x + (1.0 - l.y) * v);
                alpha_a += l.y;
            }
            oy.write(float4(clamp(v, 0.0, 255.0) / 255.0), q);
        }
    }
    float2 c_dsize = dsize * 0.5;
    bool before = p.layout == 0 ? g.x < (p.w / 2) / 2 : g.x < p.hw / 2;
    uint2 at = (p.layout == 1 && !before) ? uint2(g.x - p.hw / 2, g.y) : g;
    float2 c = (before && p.has_b != 0) ? fetch(buv, b_src * 0.5, c_dsize, at).xy : fetch(auv, a_src * 0.5, c_dsize, at).xy;
    c = rint(c);
    int gx = int(g.x);
    if (gx >= mid / 2 - 1 && gx < mid / 2 + 1) c = p.amber.yz;
    // A label's Cb/Cr are 128 (white on black); its weight here is the mean of the 2x2 block's
    int2 gi = int2(g) * 2;
    if (inside(gi, p.lbx, p.lby, p.lbw, p.lbh)) {
        float a = alpha_b * 0.25;
        c = rint(a * 128.0 + (1.0 - a) * c);
    }
    if (inside(gi, p.lax, p.lay, p.law, p.lah)) {
        float a = alpha_a * 0.25;
        c = rint(a * 128.0 + (1.0 - a) * c);
    }
    ouv.write(float4(clamp(c, 0.0, 255.0) / 255.0, 0.0, 0.0), g);
}
"#;

pub struct Kernels {
    pub to_nv12: Retained<Pipeline>,
    pub compose: Retained<Pipeline>,
}

// Compute pipeline states are thread-safe.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

/// The kernels, compiled once a process.
pub fn kernels() -> Result<&'static Kernels> {
    static K: OnceLock<Kernels> = OnceLock::new();
    if let Some(k) = K.get() {
        return Ok(k);
    }
    let gpu = Gpu::shared();
    let lib = gpu.library(SHADERS)?;
    let k = Kernels { to_nv12: gpu.pipeline(&lib, "to_nv12")?, compose: gpu.pipeline(&lib, "compose")? };
    Ok(K.get_or_init(|| k))
}

/// A pixel buffer's two planes as textures (R8 luma, RG8 chroma); they keep the buffer's surface in use while held.
pub struct Planes {
    _cv: [CFRetained<CVMetalTexture>; 2],
    pub y: Retained<Texture>,
    pub uv: Retained<Texture>,
}

// CVMetalTextures and Metal textures are safe to hand between threads.
unsafe impl Send for Planes {}
unsafe impl Sync for Planes {}

pub struct TexCache {
    cache: CFRetained<CVMetalTextureCache>,
    // One caller at a time (the pipeline's thread and the replay's both make textures)
    lock: std::sync::Mutex<()>,
}

// The cache is a CF object, used under its lock; Metal textures are thread-safe.
unsafe impl Send for TexCache {}
unsafe impl Sync for TexCache {}

impl TexCache {
    pub fn new() -> Result<TexCache> {
        let usage = cf::num((MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite).0 as i64);
        let attrs = unsafe { cf::dict(&[(kCVMetalTextureUsage, &usage)]) };
        let mut cache: *mut CVMetalTextureCache = std::ptr::null_mut();
        let st = unsafe {
            CVMetalTextureCache::create(None, None, &Gpu::shared().device, Some(&attrs), NonNull::from(&mut cache))
        };
        if st != 0 || cache.is_null() {
            return Err(Error::Status { what: "CVMetalTextureCacheCreate", status: st });
        }
        Ok(TexCache { cache: unsafe { CFRetained::from_raw(NonNull::new(cache).unwrap()) }, lock: std::sync::Mutex::new(()) })
    }

    fn plane(&self, buf: &CVPixelBuffer, index: usize, format: MTLPixelFormat) -> Result<(CFRetained<CVMetalTexture>, Retained<Texture>)> {
        let w = CVPixelBufferGetWidthOfPlane(buf, index);
        let h = CVPixelBufferGetHeightOfPlane(buf, index);
        let mut tex: *mut CVMetalTexture = std::ptr::null_mut();
        let st = unsafe {
            CVMetalTextureCache::create_texture_from_image(None, &self.cache, buf, None, format, w, h, index, NonNull::from(&mut tex))
        };
        if st != 0 || tex.is_null() {
            return Err(Error::Status { what: "CVMetalTextureCacheCreateTextureFromImage", status: st });
        }
        let cvt = unsafe { CFRetained::from_raw(NonNull::new(tex).unwrap()) };
        let mt = CVMetalTextureGetTexture(&cvt).ok_or_else(|| Error::Other("no Metal texture for a plane".into()))?;
        Ok((cvt, mt))
    }

    /// An NV12 buffer's planes, for reading or writing.
    pub fn planes(&self, buf: &CVPixelBuffer) -> Result<Planes> {
        let _g = self.lock.lock().unwrap();
        let (cy, y) = self.plane(buf, 0, MTLPixelFormat::R8Unorm)?;
        let (cuv, uv) = self.plane(buf, 1, MTLPixelFormat::RG8Unorm)?;
        Ok(Planes { _cv: [cy, cuv], y, uv })
    }

    /// Lets go of textures nothing holds any more.
    pub fn flush(&self) {
        let _g = self.lock.lock().unwrap();
        self.cache.flush(0);
    }
}

#[repr(C)]
struct ToNv12 {
    ky: [f32; 4],
    ku: [f32; 4],
    kv: [f32; 4],
    w: u32,
    h: u32,
    pad: [u32; 2],
}

fn bytes<T>(v: &T) -> (NonNull<c_void>, usize) {
    (NonNull::from(v).cast(), std::mem::size_of::<T>())
}

/// Queues the conversion of `src` (its top-left w x h, both even) into `dst`'s planes.
pub fn encode_to_nv12(cb: &CommandBuffer, src: &Texture, w: usize, h: usize, dst: &Planes) -> Result<()> {
    let k = kernels()?;
    let m = yuv::m709();
    let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
    let p = ToNv12 { ky: row(m[0]), ku: row(m[1]), kv: row(m[2]), w: w as u32, h: h as u32, pad: [0; 2] };
    let enc = cb.computeCommandEncoder().ok_or_else(|| Error::Other("no compute encoder".into()))?;
    enc.setComputePipelineState(&k.to_nv12);
    unsafe {
        enc.setTexture_atIndex(Some(src), 0);
        enc.setTexture_atIndex(Some(&dst.y), 1);
        enc.setTexture_atIndex(Some(&dst.uv), 2);
        let (ptr, len) = bytes(&p);
        enc.setBytes_length_atIndex(ptr, len, 0);
    }
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: w / 2, height: h / 2, depth: 1 },
        MTLSize { width: 16, height: 16, depth: 1 },
    );
    enc.endEncoding();
    Ok(())
}

/// What the compose kernel is told (gpu.rs's `struct Compose`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ComposeParams {
    pub amber: [f32; 4],
    pub ow: u32,
    pub oh: u32,
    pub layout: u32,
    pub w: u32,
    pub h: u32,
    pub hw: u32,
    pub aw: u32,
    pub ah: u32,
    pub bw: u32,
    pub bh: u32,
    pub mid: u32,
    pub lb: [i32; 4],
    pub la: [i32; 4],
    pub has_b: u32,
}

/// Queues the before & after picture: `after` and `before` planes (before may be after's again), the labels as
/// RG32Float textures of (Y, weight), into `out`.
pub fn encode_compose(
    cb: &CommandBuffer,
    after: &Planes,
    before: &Planes,
    labels: [&Texture; 2],
    out: &Planes,
    p: &ComposeParams,
) -> Result<()> {
    let k = kernels()?;
    let enc = cb.computeCommandEncoder().ok_or_else(|| Error::Other("no compute encoder".into()))?;
    enc.setComputePipelineState(&k.compose);
    unsafe {
        for (i, t) in [&*after.y, &*after.uv, &*before.y, &*before.uv, labels[0], labels[1], &*out.y, &*out.uv].iter().enumerate() {
            enc.setTexture_atIndex(Some(t), i);
        }
        let (ptr, len) = bytes(p);
        enc.setBytes_length_atIndex(ptr, len, 0);
    }
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: (p.ow / 2) as usize, height: (p.oh / 2) as usize, depth: 1 },
        MTLSize { width: 16, height: 16, depth: 1 },
    );
    enc.endEncoding();
    Ok(())
}

/// Queues a copy of `src` into `dst` (same size and format).
pub fn encode_copy(cb: &CommandBuffer, src: &Texture, dst: &Texture) -> Result<()> {
    let blit = cb.blitCommandEncoder().ok_or_else(|| Error::Other("no blit encoder".into()))?;
    unsafe { blit.copyFromTexture_toTexture(src, dst) };
    blit.endEncoding();
    Ok(())
}
