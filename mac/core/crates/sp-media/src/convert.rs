//! Decoded pictures to sp_core frames on the GPU.
//!
//! Metal reads a decoder's CVPixelBuffer (IOSurface-backed NV12 or P010) through a CVMetalTextureCache, without a
//! copy. Plane 0 is seen as R8Uint or R16Uint and plane 1 as RG8Uint or RG16Uint. One compute pass makes the frame:
//!
//! - SDR ([`Converter::sdr`]): what framegen/chiaki_tap.py's VideoToolbox path hands on, which is PyAV's
//!   `to_ndarray("bgr24")` through swscale. Chroma is taken as is for each 2x2 block, with no interpolation. The
//!   matrix comes from the stream's tags (BT.709 or BT.2020 if tagged so, else BT.601), and the range is limited
//!   unless tagged full.
//! - HDR ([`Converter::hdr`]): framegen/hdr.py's maths per pixel, to "sdr8" (BGRA8) or "pq16" (RGBA16F, PQ BT.2020).
//!   Chroma is bilinear, as F.interpolate with align_corners=False. Then PQ or HLG to light, BT.2390's roll-off,
//!   BT.2020 to BT.709 and SDR gamma. Python runs the same maths through a 3D LUT. This is the exact version.
//! - Decoded frames from the tap ([`Converter::nv12_bytes`], [`Converter::bgra_bytes`]): framegen/capture.py
//!   raw_to_bgr, with bilinear chroma and the header's colours.
//!
//! Each call waits for its GPU work, so the frame is complete when it's returned. Frames come from a pool, and one
//! is reused once nothing else holds it. Keep the Frame, not only its texture, while using it.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeightOfPlane,
    CVPixelBufferGetWidthOfPlane,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice,
    MTLPixelFormat, MTLResourceOptions, MTLSize,
};
use sp_core::gpu::{Pipeline, Queue};
use sp_core::{Frame, Gpu, PixelKind};

use crate::colour::{self, FrameInfo, PRI_BT2020, TRC_HLG, TRC_PQ};
use crate::vt::Decoded;

const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Sdr {
    float y_lo, y_mul, c_mid, c_mul;   // lum = (Y - y_lo) * y_mul, c = (C - c_mid) * c_mul, both on 0..255
    float kr, kb, pad0, pad1;
    uint shift, bilinear, cw, ch;      // 6 for P010, chroma interpolated or not, chroma plane size
    uint width, height, pad2, pad3;    // for frames in a buffer
};

static float3 ycc_rgb(float y, float u, float v, constant Sdr &p) {
    float lum = (y - p.y_lo) * p.y_mul;
    float cb = (u - p.c_mid) * p.c_mul, cr = (v - p.c_mid) * p.c_mul;
    float kr = p.kr, kb = p.kb, kg = 1.0f - kr - kb;
    float b = lum + 2.0f * (1.0f - kb) * cb;
    float g = lum - (2.0f * kb * (1.0f - kb) / kg) * cb - (2.0f * kr * (1.0f - kr) / kg) * cr;
    float r = lum + 2.0f * (1.0f - kr) * cr;
    return clamp(rint(float3(r, g, b)), 0.0f, 255.0f) / 255.0f;
}

// Chroma for pixel p, bilinear as torch's F.interpolate(scale 2, align_corners=False). The source position is
// (p + 0.5) / 2 - 0.5 clamped at 0, and the next sample is clamped at the edge.
struct Bilinear { uint2 a, b; float2 f; };
static Bilinear bilinear(uint2 p, uint cw, uint ch) {
    float sx = max((float(p.x) + 0.5f) * 0.5f - 0.5f, 0.0f), sy = max((float(p.y) + 0.5f) * 0.5f - 0.5f, 0.0f);
    uint x0 = uint(sx), y0 = uint(sy);
    Bilinear o;
    o.a = uint2(x0, y0);
    o.b = uint2(min(x0 + 1, cw - 1), min(y0 + 1, ch - 1));
    o.f = float2(sx - float(x0), sy - float(y0));
    return o;
}

static float2 chroma_tex(texture2d<uint, access::read> uv, uint2 p, uint shift, uint bilin, uint cw, uint ch) {
    if (bilin == 0) return float2(uv.read(p / 2).rg >> shift);
    Bilinear s = bilinear(p, cw, ch);
    float2 c00 = float2(uv.read(s.a).rg >> shift), c10 = float2(uv.read(uint2(s.b.x, s.a.y)).rg >> shift);
    float2 c01 = float2(uv.read(uint2(s.a.x, s.b.y)).rg >> shift), c11 = float2(uv.read(s.b).rg >> shift);
    return (1.0f - s.f.y) * ((1.0f - s.f.x) * c00 + s.f.x * c10) + s.f.y * ((1.0f - s.f.x) * c01 + s.f.x * c11);
}

kernel void sdr_planes(texture2d<uint, access::read> yp [[texture(0)]],
                       texture2d<uint, access::read> uvp [[texture(1)]],
                       texture2d<float, access::write> out [[texture(2)]],
                       constant Sdr &p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float y = float(yp.read(gid).r >> p.shift);
    float2 c = chroma_tex(uvp, gid, p.shift, p.bilinear, p.cw, p.ch);
    out.write(float4(ycc_rgb(y, c.x, c.y, p), 1.0f), gid);
}

// NV12 packed tightly in a buffer (a decoded frame from the tap)
kernel void nv12_buffer(device const uchar *src [[buffer(1)]],
                        texture2d<float, access::write> out [[texture(0)]],
                        constant Sdr &p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint w = p.width, h = p.height;
    if (gid.x >= w || gid.y >= h) return;
    device const uchar *uv = src + w * h;
    float y = float(src[gid.y * w + gid.x]);
    float2 c;
    if (p.bilinear == 0) {
        uint i = ((gid.y / 2) * (w / 2) + gid.x / 2) * 2;
        c = float2(uv[i], uv[i + 1]);
    } else {
        Bilinear s = bilinear(gid, p.cw, p.ch);
        #define UV(x, y) float2(uv[((y) * p.cw + (x)) * 2], uv[((y) * p.cw + (x)) * 2 + 1])
        float2 c00 = UV(s.a.x, s.a.y), c10 = UV(s.b.x, s.a.y), c01 = UV(s.a.x, s.b.y), c11 = UV(s.b.x, s.b.y);
        c = (1.0f - s.f.y) * ((1.0f - s.f.x) * c00 + s.f.x * c10) + s.f.y * ((1.0f - s.f.x) * c01 + s.f.x * c11);
    }
    out.write(float4(ycc_rgb(y, c.x, c.y, p), 1.0f), gid);
}

// BGRA packed tightly in a buffer (alpha ignored)
kernel void bgra_buffer(device const uchar4 *src [[buffer(1)]],
                        texture2d<float, access::write> out [[texture(0)]],
                        constant Sdr &p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    uchar4 px = src[gid.y * p.width + gid.x];  // B, G, R, A
    out.write(float4(float(px.z), float(px.y), float(px.x), 255.0f) / 255.0f, gid);
}

// ------------------------------------------------------------------------------------------- HDR (hdr.py)
constant float M1 = 2610.0f / 16384.0f, M2 = 2523.0f / 4096.0f * 128.0f;
constant float C1 = 3424.0f / 4096.0f, C2 = 2413.0f / 4096.0f * 32.0f, C3 = 2392.0f / 4096.0f * 32.0f;
constant float HLG_A = 0.17883277f, HLG_B = 0.28466892f, HLG_C = 0.55991073f;

struct Hdr {
    float4 m0, m1, m2;                   // matrix rows in linear light, BT.2020 to 709 (sdr8) or 709 to 2020 (pq16)
    float4 luma;                         // BT.2020's luma weights, for HLG
    float lo, ys, cs, mid;               // code values: lum = (Y - lo) / ys, c = (C - mid) / cs
    float kr, kb, src_pq, max_lum;       // BT.2390 roll-off from src_pq (PQ of the source peak) to max_lum of it
    float ks, dst_nits, sdr_nits, hlg_nits;
    uint transfer, out, mode, shift;     // transfer 0 SDR gamma, 1 PQ, 2 HLG. out 0 sdr8, 1 pq16.
                                         // mode 0 the maths, 1 HDR10 kept as is (pq16), 2 SDR in 10 bits (sdr8)
    uint use_matrix, tonemap, cw, ch;    // tonemap 0 none, 1 clamp to dst_nits, 2 roll off to dst_nits
};

static float3 pq_eotf(float3 e) {
    float3 p = precise::pow(clamp(e, 0.0f, 1.0f), 1.0f / M2);
    return precise::pow(max(p - C1, 0.0f) / (C2 - C3 * p), 1.0f / M1);
}
static float pq_eotf1(float e) {
    float p = precise::pow(clamp(e, 0.0f, 1.0f), 1.0f / M2);
    return precise::pow(max(p - C1, 0.0f) / (C2 - C3 * p), 1.0f / M1);
}
static float3 pq_oetf(float3 y) {
    float3 p = precise::pow(clamp(y, 0.0f, 1.0f), M1);
    return precise::pow((C1 + C2 * p) / (1.0f + C3 * p), M2);
}
static float pq_oetf1(float y) {
    float p = precise::pow(clamp(y, 0.0f, 1.0f), M1);
    return precise::pow((C1 + C2 * p) / (1.0f + C3 * p), M2);
}

static float eetf(float e, constant Hdr &p) {
    float x = max(e / p.src_pq, 0.0f);
    float t = clamp((x - p.ks) / (1.0f - p.ks), 0.0f, 1.0f);
    float t2 = t * t, t3 = t2 * t;
    float q = (2 * t3 - 3 * t2 + 1) * p.ks + (t3 - 2 * t2 + t) * (1.0f - p.ks) + (-2 * t3 + 3 * t2) * p.max_lum;
    return (x < p.ks ? x : (x >= 1.0f ? p.max_lum : q)) * p.src_pq;
}

// As tone_map_nits, the brightest channel rolled off and all three scaled alike
static float3 tone_map(float3 rgb, constant Hdr &p) {
    if (p.tonemap == 1) return clamp(rgb, 0.0f, p.dst_nits);
    float m = max(max(max(rgb.r, rgb.g), rgb.b), 1e-6f);
    float mapped = pq_eotf1(eetf(pq_oetf1(m / 10000.0f), p)) * 10000.0f;
    return max(rgb, 0.0f) * (mapped / m);
}

static float3 matrix3(float3 v, constant Hdr &p) {
    return float3(dot(p.m0.xyz, v), dot(p.m1.xyz, v), dot(p.m2.xyz, v));
}

kernel void hdr_planes(texture2d<uint, access::read> yp [[texture(0)]],
                       texture2d<uint, access::read> uvp [[texture(1)]],
                       texture2d<float, access::write> out [[texture(2)]],
                       constant Hdr &p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float y = float(yp.read(gid).r >> p.shift);
    float2 c = chroma_tex(uvp, gid, p.shift, 1, p.cw, p.ch);
    // yuv_to_rgb
    float lum = (y - p.lo) / p.ys, cb = (c.x - p.mid) / p.cs, cr = (c.y - p.mid) / p.cs;
    float kr = p.kr, kb = p.kb, kg = 1.0f - kr - kb;
    float r = lum + 2.0f * (1.0f - kr) * cr, b = lum + 2.0f * (1.0f - kb) * cb;
    float g = (lum - kr * r - kb * b) / kg;
    float3 rgb = clamp(float3(r, g, b), 0.0f, 1.0f);
    if (p.mode == 1) { out.write(float4(rgb, 1.0f), gid); return; }
    if (p.mode == 2) { out.write(float4(rint(rgb * 255.0f) / 255.0f, 1.0f), gid); return; }
    // rgb_to_nits
    float3 nits;
    if (p.transfer == 1) {
        nits = pq_eotf(rgb) * 10000.0f;
    } else if (p.transfer == 2) {
        float3 e = rgb;
        float3 scene = select((precise::exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0f, e * e / 3.0f, e <= 0.5f);
        float ysc = dot(scene, p.luma.xyz);
        nits = p.hlg_nits * precise::pow(max(ysc, 1e-9f), 0.2f) * scene;
    } else {
        nits = precise::pow(rgb, 2.4f) * p.sdr_nits;
    }
    if (p.out == 0) {  // nits_to_sdr8
        float3 lin = (p.tonemap ? tone_map(nits, p) : nits) / p.sdr_nits;
        if (p.use_matrix) lin = matrix3(lin, p);
        float3 e = precise::pow(clamp(lin, 0.0f, 1.0f), 1.0f / 2.4f);
        out.write(float4(rint(e * 255.0f) / 255.0f, 1.0f), gid);
    } else {           // nits_to_pq16
        if (p.use_matrix) nits = matrix3(nits, p);
        if (p.tonemap) nits = tone_map(nits, p);
        out.write(float4(pq_oetf(max(nits, 0.0f) / 10000.0f), 1.0f), gid);
    }
}

// An HDR10 frame (pq16, PQ BT.2020 R'G'B') as 8 bits. Mode 0 is hdr.py's level8 and to_u8, for the checks tuned on
// SDR frames. It puts SDR white's PQ value at 255 with one multiply, truncated. Mode 1 is pq16_to_sdr8, the same
// picture "sdr8" makes.
kernel void pq16_to_8(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> out [[texture(1)]],
                      constant Hdr &p [[buffer(0)]],
                      constant uint &mode [[buffer(1)]],
                      constant float &scale [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float3 e = src.read(gid).rgb;
    if (mode == 0) { out.write(float4(floor(clamp(e * scale, 0.0f, 255.0f)) / 255.0f, 1.0f), gid); return; }
    float3 lin = tone_map(pq_eotf(e) * 10000.0f, p) / p.sdr_nits;
    lin = matrix3(lin, p);
    float3 g = precise::pow(clamp(lin, 0.0f, 1.0f), 1.0f / 2.4f);
    out.write(float4(rint(g * 255.0f) / 255.0f, 1.0f), gid);
}

// ------------------------------------------------------------------------------------------- Ps5Sim's noise
static uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// One of a few fixed noise planes (`plane`), normal with sd `sigma` in 0..255 levels, truncated towards 0 and
// added with saturation, as Ps5SimSource does with cv2.subtract(cv2.add(f, pos), neg).
kernel void add_noise(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> out [[texture(1)]],
                      constant float &sigma [[buffer(0)]],
                      constant uint &plane [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float4 v = src.read(gid);
    float3 n;
    for (uint c = 0; c < 3; c++) {
        uint k = hash(hash(hash(gid.x * 3 + c) ^ gid.y) ^ (plane * 0x9e3779b9U));
        float u1 = (float(hash(k) >> 8) + 0.5f) / 16777216.0f, u2 = float(hash(k ^ 0x5bd1e995U) >> 8) / 16777216.0f;
        n[c] = clamp(trunc(sqrt(-2.0f * log(u1)) * cos(6.2831853f * u2) * sigma), -255.0f, 255.0f);
    }
    out.write(float4(clamp(rint(v.rgb * 255.0f) + n, 0.0f, 255.0f) / 255.0f, 1.0f), gid);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SdrParams {
    y_lo: f32,
    y_mul: f32,
    c_mid: f32,
    c_mul: f32,
    kr: f32,
    kb: f32,
    pad0: f32,
    pad1: f32,
    shift: u32,
    bilinear: u32,
    cw: u32,
    ch: u32,
    width: u32,
    height: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HdrParams {
    m0: [f32; 4],
    m1: [f32; 4],
    m2: [f32; 4],
    luma: [f32; 4],
    lo: f32,
    ys: f32,
    cs: f32,
    mid: f32,
    kr: f32,
    kb: f32,
    src_pq: f32,
    max_lum: f32,
    ks: f32,
    dst_nits: f32,
    sdr_nits: f32,
    hlg_nits: f32,
    transfer: u32,
    out: u32,
    mode: u32,
    shift: u32,
    use_matrix: u32,
    tonemap: u32,
    cw: u32,
    ch: u32,
}

/// SmoovPlay's HDR setting for an HDR stream (framegen/hdr.py StreamConverter), which picks what frames to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrOut {
    /// Tone-mapped to 8-bit SDR (BGRA8).
    Sdr8,
    /// Kept HDR10, as PQ BT.2020 half floats (RGBA16F).
    Pq16,
}

impl HdrOut {
    pub fn parse(s: &str) -> Option<HdrOut> {
        match s {
            "sdr8" | "sdr" => Some(HdrOut::Sdr8),
            "pq16" | "hdr10" => Some(HdrOut::Pq16),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrSettings {
    pub out: HdrOut,
    /// The game's brightest highlights, nits (hdr.py SOURCE_PEAK_NITS).
    pub src_peak: f64,
    /// For pq16, roll highlights off to this for a dimmer HDR display. None leaves them as they are.
    pub display_peak: Option<f64>,
    /// SDR white, nits (BT.2408's 203).
    pub sdr_nits: f64,
}

impl HdrSettings {
    pub fn new(out: HdrOut, display_peak: Option<f64>) -> HdrSettings {
        HdrSettings { out, src_peak: colour::source_peak_nits(), display_peak, sdr_nits: colour::sdr_white_nits() }
    }
}

/// Frames reused once nothing else holds them. Their `owner` is the pool's mark, and a count of 1 means only the
/// pool holds one.
struct Pool {
    frames: Vec<Frame>,
    max: usize,
}

struct PoolMark;

impl Pool {
    fn get(&mut self, w: usize, h: usize, kind: PixelKind) -> Frame {
        for f in &self.frames {
            if (f.width, f.height, f.kind) == (w, h, kind) && f.owner.as_ref().is_some_and(|o| Arc::strong_count(o) == 1) {
                return f.clone();
            }
        }
        let mut f = Frame::new(w, h, kind);
        f.owner = Some(Arc::new(PoolMark));
        if self.frames.len() < self.max {
            self.frames.push(f.clone());
        }
        f
    }
}

type Buffer = ProtocolObject<dyn objc2_metal::MTLBuffer>;
type Texture = ProtocolObject<dyn objc2_metal::MTLTexture>;

/// The kernels, compiled once a process (a few hundred milliseconds the first time).
struct Pipelines {
    sdr: Retained<Pipeline>,
    hdr: Retained<Pipeline>,
    nv12: Retained<Pipeline>,
    bgra: Retained<Pipeline>,
    noise: Retained<Pipeline>,
    pq8: Retained<Pipeline>,
}

// Compute pipeline states are thread-safe.
unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

fn pipelines() -> sp_core::Result<&'static Pipelines> {
    static P: std::sync::OnceLock<Pipelines> = std::sync::OnceLock::new();
    if let Some(p) = P.get() {
        return Ok(p);
    }
    let gpu = Gpu::shared();
    let lib = gpu.library(SHADERS)?;
    let p = Pipelines {
        sdr: gpu.pipeline(&lib, "sdr_planes")?,
        hdr: gpu.pipeline(&lib, "hdr_planes")?,
        nv12: gpu.pipeline(&lib, "nv12_buffer")?,
        bgra: gpu.pipeline(&lib, "bgra_buffer")?,
        noise: gpu.pipeline(&lib, "add_noise")?,
        pq8: gpu.pipeline(&lib, "pq16_to_8")?,
    };
    Ok(P.get_or_init(|| p))
}

pub struct Converter {
    queue: Retained<Queue>,
    cache: CFRetained<CVMetalTextureCache>,
    k: &'static Pipelines,
    pool: Pool,
    staging: Option<Retained<Buffer>>,
}

// Metal objects and the texture cache are thread-safe, and a Converter is used from one thread at a time.
unsafe impl Send for Converter {}

fn err(what: &str) -> sp_core::Error {
    sp_core::Error::Metal(what.to_string())
}

impl Converter {
    pub fn new() -> sp_core::Result<Converter> {
        let gpu = Gpu::shared();
        // The shader source asks for precise maths where it matters. The rest may be fast.
        let k = pipelines()?;
        let queue = gpu.device.newCommandQueue().ok_or_else(|| err("no command queue"))?;
        queue.setLabel(Some(&NSString::from_str("sp-media convert")));
        let mut cache: *mut CVMetalTextureCache = std::ptr::null_mut();
        let st = unsafe { CVMetalTextureCache::create(None, None, &gpu.device, None, NonNull::from(&mut cache)) };
        if st != 0 || cache.is_null() {
            return Err(err(&format!("CVMetalTextureCacheCreate: {st}")));
        }
        Ok(Converter {
            queue,
            cache: unsafe { CFRetained::from_raw(NonNull::new(cache).unwrap()) },
            k,
            pool: Pool { frames: Vec::new(), max: 24 },
            staging: None,
        })
    }

    /// A plane of the decoder's picture as a texture, without a copy. The CVMetalTexture keeps it valid.
    fn plane(&self, d: &Decoded, index: usize, format: MTLPixelFormat) -> sp_core::Result<(CFRetained<CVMetalTexture>, Retained<Texture>)> {
        let w = CVPixelBufferGetWidthOfPlane(&d.buffer, index);
        let h = CVPixelBufferGetHeightOfPlane(&d.buffer, index);
        let mut tex: *mut CVMetalTexture = std::ptr::null_mut();
        let st = unsafe {
            CVMetalTextureCache::create_texture_from_image(
                None,
                &self.cache,
                &d.buffer,
                None,
                format,
                w,
                h,
                index,
                NonNull::from(&mut tex),
            )
        };
        if st != 0 || tex.is_null() {
            return Err(err(&format!("CVMetalTextureCacheCreateTextureFromImage (plane {index}): {st}")));
        }
        let cvt = unsafe { CFRetained::from_raw(NonNull::new(tex).unwrap()) };
        let mt = CVMetalTextureGetTexture(&cvt).ok_or_else(|| err("no Metal texture for a plane"))?;
        Ok((cvt, mt))
    }

    fn planes(&self, d: &Decoded) -> sp_core::Result<[(CFRetained<CVMetalTexture>, Retained<Texture>); 2]> {
        let (f0, f1) =
            if d.ten_bit { (MTLPixelFormat::R16Uint, MTLPixelFormat::RG16Uint) } else { (MTLPixelFormat::R8Uint, MTLPixelFormat::RG8Uint) };
        Ok([self.plane(d, 0, f0)?, self.plane(d, 1, f1)?])
    }

    fn run(
        &self,
        pipeline: &Pipeline,
        textures: &[&Texture],
        params: &[u8],
        buffer: Option<&Buffer>,
        size: (usize, usize),
    ) {
        let cb = self.queue.commandBuffer().expect("command buffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            for (i, t) in textures.iter().enumerate() {
                enc.setTexture_atIndex(Some(t), i);
            }
            enc.setBytes_length_atIndex(NonNull::new(params.as_ptr() as *mut c_void).unwrap(), params.len(), 0);
            if let Some(b) = buffer {
                enc.setBuffer_offset_atIndex(Some(b), 0, 1);
            }
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: size.0, height: size.1, depth: 1 },
            MTLSize { width: 16, height: 16, depth: 1 },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    fn bytes<T: Copy>(v: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
    }

    /// An SDR frame (BGRA8) as PyAV's to_ndarray("bgr24") makes it from the same picture. A 10-bit picture (an HDR
    /// stream with SmoovPlay's HDR off) is taken the same way, its colours washed out, as in Python.
    pub fn sdr(&mut self, d: &Decoded) -> sp_core::Result<Frame> {
        let (kr, kb) = d.tags.swscale_kr_kb();
        let s = if d.ten_bit { 4.0 } else { 1.0 };
        let (y_lo, y_mul, c_mul) =
            if d.tags.full_range { (0.0, 1.0 / s, 1.0 / s) } else { (16.0 * s, 255.0 / (219.0 * s), 255.0 / (224.0 * s)) };
        let p = SdrParams {
            y_lo,
            y_mul,
            c_mid: 128.0 * s,
            c_mul,
            kr: kr as f32,
            kb: kb as f32,
            shift: if d.ten_bit { 6 } else { 0 },
            bilinear: 0,
            cw: d.width.div_ceil(2) as u32,
            ch: d.height.div_ceil(2) as u32,
            ..Default::default()
        };
        let planes = self.planes(d)?;
        let out = self.pool.get(d.width, d.height, PixelKind::Bgra8);
        self.run(&self.k.sdr, &[&planes[0].1, &planes[1].1, &out.texture], Self::bytes(&p), None, (d.width, d.height));
        drop(planes);
        self.cache.flush(0);
        Ok(out)
    }

    /// An HDR stream's picture as the HDR setting asks (hdr.py StreamConverter). Returns the frame and what its
    /// samples meant.
    pub fn hdr(&mut self, d: &Decoded, s: &HdrSettings) -> sp_core::Result<(Frame, FrameInfo)> {
        let info = FrameInfo::of(&d.tags, d.bits());
        let p = hdr_params(&info, s, d.width, d.height);
        let kind = if s.out == HdrOut::Sdr8 { PixelKind::Bgra8 } else { PixelKind::Rgba16f };
        let planes = self.planes(d)?;
        let out = self.pool.get(d.width, d.height, kind);
        self.run(&self.k.hdr, &[&planes[0].1, &planes[1].1, &out.texture], Self::bytes(&p), None, (d.width, d.height));
        drop(planes);
        self.cache.flush(0);
        Ok((out, info))
    }

    /// A buffer the next nv12_staged or bgra_staged call reads from. Copy a frame into it from the tap's slot, then
    /// call.
    pub fn staging(&mut self, len: usize) -> &mut [u8] {
        let have = self.staging.as_ref().map(|b| b.length()).unwrap_or(0);
        if have < len {
            self.staging = Gpu::shared().device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared);
        }
        let b = self.staging.as_ref().expect("staging buffer");
        unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr() as *mut u8, len) }
    }

    /// The NV12 frame in the staging buffer (tightly packed) as BGRA8, by capture.py raw_to_bgr's maths with
    /// bilinear chroma. BT.709 or BT.601, limited or full range.
    pub fn nv12_staged(&mut self, w: usize, h: usize, bt709: bool, full_range: bool) -> Frame {
        let (kr, kb) = if bt709 { (0.2126, 0.0722) } else { (0.299, 0.114) };
        let (y_lo, y_mul, c_mul) = if full_range { (0.0, 1.0, 1.0) } else { (16.0, 255.0 / 219.0, 255.0 / 224.0) };
        let p = SdrParams {
            y_lo,
            y_mul,
            c_mid: 128.0,
            c_mul,
            kr,
            kb,
            bilinear: 1,
            cw: (w / 2) as u32,
            ch: (h / 2) as u32,
            width: w as u32,
            height: h as u32,
            ..Default::default()
        };
        let out = self.pool.get(w, h, PixelKind::Bgra8);
        let buf = self.staging.clone().expect("staging() first");
        self.run(&self.k.nv12, &[&out.texture], Self::bytes(&p), Some(&buf), (w, h));
        out
    }

    /// The BGRA frame in the staging buffer as a frame (alpha ignored).
    pub fn bgra_staged(&mut self, w: usize, h: usize) -> Frame {
        let p = SdrParams { width: w as u32, height: h as u32, ..Default::default() };
        let out = self.pool.get(w, h, PixelKind::Bgra8);
        let buf = self.staging.clone().expect("staging() first");
        self.run(&self.k.bgra, &[&out.texture], Self::bytes(&p), Some(&buf), (w, h));
        out
    }

    /// NV12 bytes (tightly packed) to a frame.
    pub fn nv12_bytes(&mut self, nv12: &[u8], w: usize, h: usize, bt709: bool, full_range: bool) -> Frame {
        self.staging(nv12.len()).copy_from_slice(nv12);
        self.nv12_staged(w, h, bt709, full_range)
    }

    /// BGRA bytes (tightly packed) to a frame.
    pub fn bgra_bytes(&mut self, bgra: &[u8], w: usize, h: usize) -> Frame {
        self.staging(bgra.len()).copy_from_slice(bgra);
        self.bgra_staged(w, h)
    }

    /// An HDR10 frame (Rgba16f, PQ BT.2020) as BGRA8. With `exact` false it's hdr.py's level8 and to_u8, the 8-bit
    /// stand-in that the repeat and cut checks, the delay test and test mode's scores take. That is one multiply that
    /// puts SDR white's PQ value at 255 and every highlight at 255. With `exact` true it's pq16_to_sdr8, the picture
    /// "sdr8" would have made, for instant replay's clips while the window shows HDR. An SDR frame comes back as it is.
    pub fn pq16_to_8(&mut self, src: &Frame, exact: bool, s: &HdrSettings) -> Frame {
        if src.kind == PixelKind::Bgra8 {
            return src.clone();
        }
        let info = FrameInfo { bits: 10, full_range: false, matrix: 9, transfer: TRC_PQ, primaries: PRI_BT2020 };
        let p = hdr_params(&info, &HdrSettings { out: HdrOut::Sdr8, ..*s }, src.width, src.height);
        let mode = exact as u32;
        let scale = (255.0 / colour::pq_of_nits(s.sdr_nits)) as f32;
        let out = self.pool.get(src.width, src.height, PixelKind::Bgra8);
        let cb = self.queue.commandBuffer().expect("command buffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(&self.k.pq8);
        unsafe {
            enc.setTexture_atIndex(Some(&src.texture), 0);
            enc.setTexture_atIndex(Some(&out.texture), 1);
            let b = Self::bytes(&p);
            enc.setBytes_length_atIndex(NonNull::new(b.as_ptr() as *mut c_void).unwrap(), b.len(), 0);
            enc.setBytes_length_atIndex(NonNull::from(&mode).cast(), 4, 1);
            enc.setBytes_length_atIndex(NonNull::from(&scale).cast(), 4, 2);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: src.width, height: src.height, depth: 1 },
            MTLSize { width: 16, height: 16, depth: 1 },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        out
    }

    /// `src` with one of a few fixed noise planes added (Ps5SimSource's noise option), sd `sigma` levels.
    pub fn add_noise(&mut self, src: &Frame, plane: u32, sigma: f32) -> Frame {
        let out = self.pool.get(src.width, src.height, PixelKind::Bgra8);
        let cb = self.queue.commandBuffer().expect("command buffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(&self.k.noise);
        unsafe {
            enc.setTexture_atIndex(Some(&src.texture), 0);
            enc.setTexture_atIndex(Some(&out.texture), 1);
            enc.setBytes_length_atIndex(NonNull::from(&sigma).cast(), 4, 0);
            enc.setBytes_length_atIndex(NonNull::from(&plane).cast(), 4, 1);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: src.width, height: src.height, depth: 1 },
            MTLSize { width: 16, height: 16, depth: 1 },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        out
    }
}

fn row(m: &[[f64; 3]; 3], i: usize) -> [f32; 4] {
    [m[i][0] as f32, m[i][1] as f32, m[i][2] as f32, 0.0]
}

/// The constants for hdr_planes, as hdr.py's convert_yuv works them out.
fn hdr_params(info: &FrameInfo, s: &HdrSettings, w: usize, h: usize) -> HdrParams {
    let scale = (1u32 << (info.bits - 8)) as f64;
    let (lo, ys, cs) = if info.full_range { (0.0, 255.0 * scale, 255.0 * scale) } else { (16.0 * scale, 219.0 * scale, 224.0 * scale) };
    let (kr, kb) = info.kr_kb();
    let mut p = HdrParams {
        lo: lo as f32,
        ys: ys as f32,
        cs: cs as f32,
        mid: (128.0 * scale) as f32,
        kr: kr as f32,
        kb: kb as f32,
        sdr_nits: s.sdr_nits as f32,
        hlg_nits: colour::HLG_DISPLAY_NITS as f32,
        transfer: match info.transfer {
            TRC_PQ => 1,
            TRC_HLG => 2,
            _ => 0,
        },
        out: (s.out == HdrOut::Pq16) as u32,
        shift: if info.bits > 8 { 6 } else { 0 },
        cw: w.div_ceil(2) as u32,
        ch: h.div_ceil(2) as u32,
        ..Default::default()
    };
    let l = colour::bt2020_luma();
    p.luma = [l[0] as f32, l[1] as f32, l[2] as f32, 0.0];
    let ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let (m, dst) = match s.out {
        HdrOut::Pq16 => {
            if info.transfer == TRC_PQ
                && info.primaries == PRI_BT2020
                && s.display_peak.is_none_or(|d| d >= s.src_peak)
            {
                p.mode = 1; // already HDR10, so only the layout changes
            }
            p.use_matrix = (info.primaries != PRI_BT2020) as u32;
            let tm = s.display_peak.filter(|d| *d < s.src_peak);
            p.tonemap = if tm.is_some() { 2 } else { 0 };
            (if p.use_matrix == 1 { colour::bt709_to_2020() } else { ident }, tm.unwrap_or(s.src_peak))
        }
        HdrOut::Sdr8 => {
            if !info.hdr() && info.primaries != PRI_BT2020 {
                p.mode = 2; // SDR in a 10-bit stream, taken to 8 bits
            }
            p.use_matrix = (info.primaries == PRI_BT2020) as u32;
            p.tonemap = if s.sdr_nits >= s.src_peak { 1 } else { 2 };
            (if p.use_matrix == 1 { colour::bt2020_to_709() } else { ident }, s.sdr_nits)
        }
    };
    (p.m0, p.m1, p.m2) = (row(&m, 0), row(&m, 1), row(&m, 2));
    p.dst_nits = dst as f32;
    // bt2390_eetf's constants for a roll-off from src_peak to dst
    let (src_pq, dst_pq) = (colour::pq_of_nits(s.src_peak), colour::pq_of_nits(dst));
    let max_lum = dst_pq / src_pq;
    p.src_pq = src_pq as f32;
    p.max_lum = max_lum as f32;
    p.ks = (1.5 * max_lum - 0.5).max(0.0) as f32;
    p
}
