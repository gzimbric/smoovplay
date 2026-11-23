//! The Metal kernels around the convolutions: framegen/metalrife.py's KERNELS, nearly verbatim (the same math, so
//! the same pixels), reading and writing plain buffers and the frames' textures instead of PyTorch tensors, plus
//! Steady HUD (framegen/steady.py) as two kernels and a blend folded into the final pass.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use sp_core::gpu::{Gpu, Pipeline};

use crate::Result;

pub type Buffer = ProtocolObject<dyn MTLBuffer>;
pub type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

pub const MAX_BLOCKS: usize = 5;

pub const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Where one block's output is and how big. A block ends in ConvTranspose2d(c, 4 * k, 4, 2, 1) and PixelShuffle(2)
// (k = 13 channels: flow 4, mask 1, features 8; the last block's 5). Here the transposed convolution is a 3x3
// convolution at a quarter of the block's resolution (graph.rs deconv_as_conv3), each of its pixels making the
// 4x4 patch of block-resolution pixels above it: sub-pixel q = (y & 3) * 4 + (x & 3)... in the order
// sub = (((y >> 1) & 1) * 2 + ((x >> 1) & 1)) * 4 + (y & 1) * 2 + (x & 1). Its channels are laid out for vector
// loads (net.rs): flow at sub * 4 (a half4), mask at 64 + sub, features at 80 + sub * 8 (two half4s).
// (h, w): the block's resolution = full resolution / s. sh, sw: row and pixel strides of the convolution's output.
struct BlockOut {
    int s, h, w, pad, sh, sw;
};

// What the flow kernels take (Rust: kernels::Params). n: blocks run so far; so: the next block's scale.
struct Params {
    int PH, PW, n, so, H, W, steady, pad0;
    float t, pad1, pad2, pad3;
    BlockOut b[5];
};

struct Tmp {
    device const half* d;
    int s, h, w, sh, sw;
};

inline Tmp tmp_at(device const half* d, constant BlockOut& o) {
    Tmp t;
    t.d = d;
    t.s = o.s; t.h = o.h; t.w = o.w;
    t.sh = o.sh; t.sw = o.sw;
    return t;
}

#define TMP_ARGS device const half* d0 [[buffer(3)]], device const half* d1 [[buffer(4)]], \
                 device const half* d2 [[buffer(5)]], device const half* d3 [[buffer(6)]], \
                 device const half* d4 [[buffer(7)]], constant Params& p [[buffer(8)]]

struct Tmps {
    Tmp t0, t1, t2, t3, t4;
};

inline Tmps tmps(device const half* d0, device const half* d1, device const half* d2, device const half* d3,
                 device const half* d4, constant Params& p) {
    Tmps T;
    T.t0 = tmp_at(d0, p.b[0]); T.t1 = tmp_at(d1, p.b[1]); T.t2 = tmp_at(d2, p.b[2]);
    T.t3 = tmp_at(d3, p.b[3]); T.t4 = tmp_at(d4, p.b[4]);
    return T;
}

inline Tmp pick(thread const Tmps& T, int j) {
    return j == 0 ? T.t0 : j == 1 ? T.t1 : j == 2 ? T.t2 : j == 3 ? T.t3 : T.t4;
}

// A block-resolution pixel's channels in its block's output.
inline device const half* tpix(Tmp t, int y, int x, thread int& sub) {
    sub = (((y >> 1) & 1) * 2 + ((x >> 1) & 1)) * 4 + (y & 1) * 2 + (x & 1);
    return t.d + (y >> 2) * t.sh + (x >> 2) * t.sw;
}

inline float4 tflow(Tmp t, int y, int x) {
    int sub;
    device const half* d = tpix(t, y, x, sub);
    return float4(*(device const half4*)(d + sub * 4));
}

inline float tmask(Tmp t, int y, int x) {
    int sub;
    device const half* d = tpix(t, y, x, sub);
    return float(d[64 + sub]);
}

inline void tfeat(Tmp t, int y, int x, thread float4& lo, thread float4& hi) {
    int sub;
    device const half* d = tpix(t, y, x, sub);
    device const half4* f = (device const half4*)(d + 80 + sub * 8);
    lo = float4(f[0]);
    hi = float4(f[1]);
}

// Bilinear upsampling by t.s (PyTorch, align_corners=False) at full-resolution pixel (Y, X).
struct Taps {
    int y0, y1, x0, x1;
    float ly, lx;
};

inline Taps taps(Tmp t, int Y, int X) {
    Taps k;
    if (t.s == 1) {
        k.y0 = k.y1 = Y; k.x0 = k.x1 = X; k.ly = k.lx = 0.0f;
        return k;
    }
    float inv = 1.0f / float(t.s);
    float fy = max((float(Y) + 0.5f) * inv - 0.5f, 0.0f);
    float fx = max((float(X) + 0.5f) * inv - 0.5f, 0.0f);
    k.y0 = int(fy); k.x0 = int(fx);
    k.ly = fy - float(k.y0); k.lx = fx - float(k.x0);
    k.y1 = min(k.y0 + 1, t.h - 1); k.x1 = min(k.x0 + 1, t.w - 1);
    return k;
}

template <typename T>
inline T lerp2(Taps k, T a, T b, T e, T f) {
    return (1.0f - k.ly) * ((1.0f - k.lx) * a + k.lx * b) + k.ly * ((1.0f - k.lx) * e + k.lx * f);
}

inline float4 up_flow(Tmp t, Taps k) {
    return lerp2(k, tflow(t, k.y0, k.x0), tflow(t, k.y0, k.x1), tflow(t, k.y1, k.x0), tflow(t, k.y1, k.x1));
}

inline float up_mask(Tmp t, Taps k) {
    return lerp2(k, tmask(t, k.y0, k.x0), tmask(t, k.y0, k.x1), tmask(t, k.y1, k.x0), tmask(t, k.y1, k.x1));
}

inline void up_feat(Tmp t, Taps k, thread float4& lo, thread float4& hi) {
    float4 al, ah, bl, bh, el, eh, fl, fh;
    tfeat(t, k.y0, k.x0, al, ah); tfeat(t, k.y0, k.x1, bl, bh);
    tfeat(t, k.y1, k.x0, el, eh); tfeat(t, k.y1, k.x1, fl, fh);
    lo = lerp2(k, al, bl, el, fl);
    hi = lerp2(k, ah, bh, eh, fh);
}

// Flow after blocks 0..n-1 at full-resolution pixel (Y, X): the sum of each block's upsampled flow times its scale.
inline float4 flow_at(thread const Tmps& T, int n, int Y, int X) {
    float4 fl = float4(0.0f);
    for (int j = 0; j < n; ++j) {
        Tmp t = pick(T, j);
        Taps k = taps(t, Y, X);
        fl += float(t.s) * up_flow(t, k);
    }
    return fl;
}

// Bilinear sample of an fp16 [8-channel] frame at (px, py) in pixels; border padding, align_corners=True.
inline void warp8(device const half4* S, int PH, int PW, float px, float py, thread float4& lo, thread float4& hi) {
    float ix = clamp(px, 0.0f, float(PW - 1)), iy = clamp(py, 0.0f, float(PH - 1));
    int x0 = int(ix), y0 = int(iy);
    int x1 = min(x0 + 1, PW - 1), y1 = min(y0 + 1, PH - 1);
    float wx = ix - float(x0), wy = iy - float(y0);
    long a = (long(y0) * PW + x0) * 2, b = (long(y0) * PW + x1) * 2;
    long c = (long(y1) * PW + x0) * 2, d = (long(y1) * PW + x1) * 2;
    float wa = (1.0f - wx) * (1.0f - wy), wb = wx * (1.0f - wy), wc = (1.0f - wx) * wy, wd = wx * wy;
    lo = wa * float4(S[a]) + wb * float4(S[b]) + wc * float4(S[c]) + wd * float4(S[d]);
    hi = wa * float4(S[a + 1]) + wb * float4(S[b + 1]) + wc * float4(S[c + 1]) + wd * float4(S[d + 1]);
}

inline float3 warp3(device const half4* S, int PH, int PW, float px, float py) {
    float ix = clamp(px, 0.0f, float(PW - 1)), iy = clamp(py, 0.0f, float(PH - 1));
    int x0 = int(ix), y0 = int(iy);
    int x1 = min(x0 + 1, PW - 1), y1 = min(y0 + 1, PH - 1);
    float wx = ix - float(x0), wy = iy - float(y0);
    long a = (long(y0) * PW + x0) * 2, b = (long(y0) * PW + x1) * 2;
    long c = (long(y1) * PW + x0) * 2, d = (long(y1) * PW + x1) * 2;
    float wa = (1.0f - wx) * (1.0f - wy), wb = wx * (1.0f - wy), wc = (1.0f - wx) * wy, wd = wx * wy;
    float4 v = wa * float4(S[a]) + wb * float4(S[b]) + wc * float4(S[c]) + wd * float4(S[d]);
    return v.xyz;
}

// Input of block n (NHWC, 16 channels for block 0, 32 after; the last ones are zero):
//   [a_img 3, b_img 3, a_feat 4, b_feat 4, t] for block 0, then [mask, feat 8, flow / so 4]
// where a, b are the frames warped by the flow so far and everything is downsampled by `so` (block n's scale).
// Grid: (PW / so, PH / so).
kernel void block_input(device half4* xo [[buffer(0)]],
                        device const half4* S0 [[buffer(1)]],
                        device const half4* S1 [[buffer(2)]],
                        TMP_ARGS,
                        uint2 g [[thread_position_in_grid]]) {
    int PH = p.PH, PW = p.PW, n = p.n, so = p.so;
    float t = p.t;
    int Wo = PW / so, Ho = PH / so;
    if (int(g.x) >= Wo || int(g.y) >= Ho) return;
    Tmps T = tmps(d0, d1, d2, d3, d4, p);
    Tmp tl = pick(T, max(n - 1, 0));  // the latest block: its mask and features
    int k = so == 1 ? 1 : 2;
    int by = so == 1 ? int(g.y) : int(g.y) * so + so / 2 - 1;
    int bx = so == 1 ? int(g.x) : int(g.x) * so + so / 2 - 1;
    float4 a_lo = 0.0f, a_hi = 0.0f, b_lo = 0.0f, b_hi = 0.0f, fl_sum = 0.0f;
    float m_sum = 0.0f;
    float4 f_lo = 0.0f, f_hi = 0.0f;
    for (int dy = 0; dy < k; ++dy) {
        for (int dx = 0; dx < k; ++dx) {
            int Y = by + dy, X = bx + dx;
            float4 lo, hi;
            if (n == 0) {
                long i = (long(Y) * PW + X) * 2;
                a_lo += float4(S0[i]); a_hi += float4(S0[i + 1]);
                b_lo += float4(S1[i]); b_hi += float4(S1[i + 1]);
                continue;
            }
            float4 fl = flow_at(T, n, Y, X);
            fl_sum += fl;
            warp8(S0, PH, PW, float(X) + fl.x, float(Y) + fl.y, lo, hi);
            a_lo += lo; a_hi += hi;
            warp8(S1, PH, PW, float(X) + fl.z, float(Y) + fl.w, lo, hi);
            b_lo += lo; b_hi += hi;
            Taps kk = taps(tl, Y, X);
            m_sum += up_mask(tl, kk);
            float4 fl4, fh4;
            up_feat(tl, kk, fl4, fh4);
            f_lo += fl4;
            f_hi += fh4;
        }
    }
    float w = 1.0f / float(k * k);
    a_lo *= w; a_hi *= w; b_lo *= w; b_hi *= w;
    long o = (long(g.y) * Wo + g.x) * (n == 0 ? 4 : 8);
    xo[o + 0] = half4(a_lo.x, a_lo.y, a_lo.z, b_lo.x);
    xo[o + 1] = half4(b_lo.y, b_lo.z, a_lo.w, a_hi.x);
    xo[o + 2] = half4(a_hi.y, a_hi.z, b_lo.w, b_hi.x);
    if (n == 0) {
        xo[o + 3] = half4(b_hi.y, b_hi.z, t, 0.0f);
        return;
    }
    float m = m_sum * w;
    f_lo *= w; f_hi *= w;
    float4 fl = fl_sum * (w / float(so));
    xo[o + 3] = half4(b_hi.y, b_hi.z, t, m);
    xo[o + 4] = half4(f_lo);
    xo[o + 5] = half4(f_hi);
    xo[o + 6] = half4(fl);
    xo[o + 7] = half4(0.0f);  // (zero, but written: a pixel is one whole 64-byte line, and partial lines cost more)
}

// The in-between frame at (Y, X): both images warped by the final flow, blended by sigmoid(mask).
inline float3 final_mid(device const half4* S0, device const half4* S1, thread const Tmps& T, int n,
                        int PH, int PW, int Y, int X) {
    float4 fl = flow_at(T, n, Y, X);
    Tmp tl = pick(T, n - 1);
    float m = up_mask(tl, taps(tl, Y, X));
    m = 1.0f / (1.0f + exp(-m));
    float3 w0 = warp3(S0, PH, PW, float(X) + fl.x, float(Y) + fl.y);
    float3 w1 = warp3(S1, PH, PW, float(X) + fl.z, float(Y) + fl.w);
    return w0 * m + w1 * (1.0f - m);
}

// The output frame (BGRA8, or RGBA16F for HDR: hdr = 1), H x W. With p.steady, Steady HUD: the real pixels where
// its mask says, blended at t (framegen/steady.py SteadyHud.blend), before the conversion. Grid: (W, H).
kernel void final_out(texture2d<float, access::write> out [[texture(0)]],
                      device const half4* S0 [[buffer(1)]],
                      device const half4* S1 [[buffer(2)]],
                      TMP_ARGS,
                      device const half* hud [[buffer(9)]],
                      constant int& hdr [[buffer(10)]],
                      uint2 g [[thread_position_in_grid]]) {
    int PH = p.PH, PW = p.PW, n = p.n, H = p.H, W = p.W;
    if (int(g.x) >= W || int(g.y) >= H) return;
    Tmps T = tmps(d0, d1, d2, d3, d4, p);
    float3 mid = final_mid(S0, S1, T, n, PH, PW, int(g.y), int(g.x));
    if (p.steady) {
        long i = long(g.y) * PW + g.x;
        float w = float(hud[i]);
        float3 ra = float3(S0[i * 2].xyz), rb = float3(S1[i * 2].xyz);
        float3 real = ra + p.t * (rb - ra);
        mid = mid + w * (real - mid);
    }
    float3 v = clamp(mid, 0.0f, 1.0f);
    if (hdr == 0) {
        v = rint(v * 255.0f) / 255.0f;  // as framegen's mul(255).round() to uint8; the texture stores it exactly
    } else {
        v = float3(half3(v));
    }
    out.write(float4(v, 1.0f), g);
}

// A frame -> the encoder's input: (PH, PW, 4) fp16 [r, g, b, 0] 0..1, zero padded. Reads BGRA8 (as r, g, b) or
// RGBA16F (clamped to 0..1, as framegen's pq16 frames are). Grid: (PW, PH).
kernel void frame_in(device half4* x [[buffer(0)]], texture2d<float, access::read> f [[texture(0)]],
                     constant int4& p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    int H = p.x, W = p.y, PH = p.z, PW = p.w;
    if (int(g.x) >= PW || int(g.y) >= PH) return;
    half3 v = 0.0h;
    if (int(g.y) < H && int(g.x) < W) {
        v = clamp(half3(f.read(g).rgb), 0.0h, 1.0h);
    }
    x[long(g.y) * PW + g.x] = half4(v, 0.0h);
}

// The encoder's last layer, LeakyReLU then ConvTranspose2d(16, 4, 4, stride 2, padding 1), packed with the image
// into a frame slot: (PH, PW, 8) fp16 [r, g, b, f0, f1, f2, f3, 0].
// a: the layer before, (PH/2, PW/2, 16) fp16 before its LeakyReLU. wt: the weights as float4 (4 outputs) per
// (ky, kx, input channel). b: the bias. p: PH, PW. Grid: (PW, PH).
kernel void encode_out(device half4* S [[buffer(0)]], device const half4* x [[buffer(1)]],
                       device const half4* a [[buffer(2)]], device const float4* wt [[buffer(3)]],
                       constant float4& b [[buffer(4)]], constant int2& p [[buffer(5)]],
                       uint2 g [[thread_position_in_grid]]) {
    int PH = p.x, PW = p.y;
    int Y = int(g.y), X = int(g.x);
    if (X >= PW || Y >= PH) return;
    int h2 = PH / 2, w2 = PW / 2;
    float4 acc = b;
    // Output row Y takes input row y = (Y + 1 - ky) / 2 for the two ky of the right parity (likewise columns).
    for (int ty = 0; ty < 2; ++ty) {
        int ky = ((Y + 1) & 1) + 2 * ty;
        int y = (Y + 1 - ky) / 2;
        if (y < 0 || y >= h2) continue;
        for (int tx = 0; tx < 2; ++tx) {
            int kx = ((X + 1) & 1) + 2 * tx;
            int xx = (X + 1 - kx) / 2;
            if (xx < 0 || xx >= w2) continue;
            long base = (long(y) * w2 + xx) * 4;
            device const float4* wk = wt + (ky * 4 + kx) * 16;
            for (int q = 0; q < 4; ++q) {
                float4 v = float4(a[base + q]);
                v = select(v * 0.2f, v, v >= 0.0f);
                acc += v.x * wk[q * 4] + v.y * wk[q * 4 + 1] + v.z * wk[q * 4 + 2] + v.w * wk[q * 4 + 3];
            }
        }
    }
    long o = long(Y) * PW + X;
    half4 img = x[o];
    S[o * 2] = half4(img.x, img.y, img.z, half(acc.x));
    S[o * 2 + 1] = half4(half(acc.y), half(acc.z), half(acc.w), 0.0h);
}

// Steady HUD (framegen/steady.py), in half precision as there.
// edges: 1 where the slot's brightness has a step (Sobel L1 above STEADY_HUD_EDGE). Grid: (PW, PH).
inline half luma_at(device const half4* S, int PH, int PW, int y, int x) {
    if (y < 0 || x < 0 || y >= PH || x >= PW) return 0.0h;  // conv2d's zero padding
    float3 c = float3(S[(long(y) * PW + x) * 2].xyz);
    return half(c.x * float(half(0.299f)) + c.y * float(half(0.587f)) + c.z * float(half(0.114f)));
}

kernel void steady_edges(device uchar* e [[buffer(0)]], device const half4* S [[buffer(1)]],
                         constant int2& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    int PH = p.x, PW = p.y, Y = int(g.y), X = int(g.x);
    if (X >= PW || Y >= PH) return;
    float l[3][3];
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            l[dy + 1][dx + 1] = float(luma_at(S, PH, PW, Y + dy, X + dx));
    half gx = half((l[0][2] - l[0][0]) + 2.0f * (l[1][2] - l[1][0]) + (l[2][2] - l[2][0]));
    half gy = half((l[2][0] - l[0][0]) + 2.0f * (l[2][1] - l[0][1]) + (l[2][2] - l[0][2]));
    half s = half(float(abs(gx)) + float(abs(gy)));
    e[long(Y) * PW + X] = float(s) > 128.0f / 255.0f ? 1 : 0;
}

// mask: how much of the real pixel to take between real frames a -> b: 1 below half the threshold, 0 above it,
// and only within one pixel of an edge in either frame. Grid: (PW, PH).
kernel void steady_mask(device half* w [[buffer(0)]], device const half4* Sa [[buffer(1)]],
                        device const half4* Sb [[buffer(2)]], device const uchar* ea [[buffer(3)]],
                        device const uchar* eb [[buffer(4)]], constant int2& p [[buffer(5)]],
                        constant float& still [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
    int PH = p.x, PW = p.y, Y = int(g.y), X = int(g.x);
    if (X >= PW || Y >= PH) return;
    long i = long(Y) * PW + X;
    half3 d3 = abs(Sb[i * 2].xyz - Sa[i * 2].xyz);
    half d = max(max(d3.x, d3.y), d3.z);
    half thr = half(still);
    half k = half(2.0f / float(thr));
    half v = clamp(half((thr - d) * k), 0.0h, 1.0h);
    uchar edge = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        int y = Y + dy;
        if (y < 0 || y >= PH) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int x = X + dx;
            if (x < 0 || x >= PW) continue;
            long j = long(y) * PW + x;
            edge |= ea[j] | eb[j];
        }
    }
    w[i] = edge ? v : 0.0h;
}
"#;

/// The flow kernels' parameters (the MSL `Params`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub ph: i32,
    pub pw: i32,
    pub n: i32,
    pub so: i32,
    pub h: i32,
    pub w: i32,
    pub steady: i32,
    pub pad0: i32,
    pub t: f32,
    pub pad1: [f32; 3],
    pub b: [BlockOut; MAX_BLOCKS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockOut {
    pub s: i32,
    pub h: i32,
    pub w: i32,
    pub pad: i32,
    pub sh: i32,
    pub sw: i32,
}

pub struct Kernels {
    pub block_input: Retained<Pipeline>,
    pub final_out: Retained<Pipeline>,
    pub frame_in: Retained<Pipeline>,
    pub encode_out: Retained<Pipeline>,
    pub steady_edges: Retained<Pipeline>,
    pub steady_mask: Retained<Pipeline>,
}

unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    pub fn new() -> Result<Kernels> {
        let gpu = Gpu::shared();
        let lib = gpu.library(SOURCE)?;
        Ok(Kernels {
            block_input: gpu.pipeline(&lib, "block_input")?,
            final_out: gpu.pipeline(&lib, "final_out")?,
            frame_in: gpu.pipeline(&lib, "frame_in")?,
            encode_out: gpu.pipeline(&lib, "encode_out")?,
            steady_edges: gpu.pipeline(&lib, "steady_edges")?,
            steady_mask: gpu.pipeline(&lib, "steady_mask")?,
        })
    }
}

/// A shared-storage buffer of `bytes` (zeroed).
pub fn buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Retained<Buffer> {
    let b = device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("couldn't make a Metal buffer");
    unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0, bytes.max(16)) };
    b
}

/// A buffer holding `data`.
pub fn buffer_with<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> Retained<Buffer> {
    let bytes = std::mem::size_of_val(data);
    let b = buffer(device, bytes);
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, b.contents().as_ptr() as *mut u8, bytes) };
    b
}

pub fn set_bytes<T: Copy>(enc: &Encoder, value: &T, index: usize) {
    unsafe { enc.setBytes_length_atIndex(NonNull::from(value).cast(), std::mem::size_of::<T>(), index) };
}

pub fn set_buffer(enc: &Encoder, buf: &Buffer, index: usize) {
    unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, index) };
}

/// Dispatches `pipeline` over a (w, h) grid in 32 x 8 groups (as metalrife.py's group_size), or smaller when the
/// pipeline can't take that many threads.
pub fn dispatch(enc: &Encoder, pipeline: &Pipeline, w: usize, h: usize) {
    enc.setComputePipelineState(pipeline);
    let max = pipeline.maxTotalThreadsPerThreadgroup();
    let (tw, th) = if max >= 256 { (32, 8) } else { (32, (max / 32).max(1)) };
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: w, height: h, depth: 1 },
        MTLSize { width: tw, height: th, depth: 1 },
    );
}
