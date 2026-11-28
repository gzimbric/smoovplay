"""RIFE's flow network on Apple's GPU, with PyTorch's MPS convolutions and hand-written Metal kernels for the rest.

FastIFNet's math (framegen/fastrife.py), laid out for the Mac. Eager FastIFNet on MPS spends under half of a 1080p
forward in convolutions. The rest is full-resolution glue, each step a separate pass over tens of megabytes. It warps
[image | features] after every block, upsamples each block's output to full resolution, downsamples the warped
tensors again for the next block, concatenates the next block's input and converts the result to 8 bits. Here they
are one kernel per block plus one at the end, and they touch only the pixels that are used.

* A block's input is built in one pass at the block's own resolution. Bilinear downsampling by a whole factor s
  (PyTorch's, align_corners=False) averages the 2x2 pixels at the centre of each s x s cell, so only those four
  full-resolution pixels are warped, not all s*s of them. The flow, mask and features there are bilinearly upsampled
  from the earlier blocks' outputs on the fly, so no full-resolution flow is ever stored.
* Each block's output is read straight from its transposed convolution, before the PixelShuffle (an index change).
  The last block's 8 feature channels are never used, so its transposed convolution makes only flow and mask.
* The last pass warps the two images, blends them by the mask and writes the uint8 BGR frame.
* Frames are kept as fp16 [r, g, b, f0, f1, f2, f3, 0] pixels (16 bytes, two half4 loads per bilinear tap). That's
  exact, because the image and the encoder's features are fp16 to begin with. The warps interpolate in fp32.
* In the residual blocks, the per-channel `beta` is folded into the convolution's weights, and the add and the
  LeakyReLU are one kernel. The encoder's last layer, a transposed convolution from 16 channels to 4 that is slow in
  MPS, is a kernel that also packs the frame slot.

On an M5 Max at 1080p with RIFE 4.22 lite, an in-between frame takes 6.8 ms at full motion detail and 4.4 ms at
half, against 18.1 and 12.8 ms for eager FastIFNet. Preparing a frame takes 0.7 ms, against 2.3. Results match eager
FastIFNet to fp16 rounding (scripts/check_fast.py, tests/test_metalrife.py). It takes whole-number block scales only,
so motion detail 1.0, 0.5 or 0.25. `supported()` says whether it applies, and otherwise RifeInterpolator runs eager
FastIFNet.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

KERNELS = r"""
#include <metal_stdlib>
using namespace metal;

// One block's output, read from its transposed convolution before the PixelShuffle(2), is
// tmp[c, y, x] = d[c*4 + (y&1)*2 + (x&1), y/2, x/2]. s is its scale, so full resolution is s * (h, w).
struct Tmp {
    device const half* d;
    int s, h, w;
    long sc, sh, sw;
};

inline Tmp tmp_at(device const half* d, constant long* p) {
    Tmp t;
    t.d = d;
    t.s = int(p[0]); t.h = int(p[1]); t.w = int(p[2]);
    t.sc = p[3]; t.sh = p[4]; t.sw = p[5];
    return t;
}

// Up to five blocks' outputs (RIFE 4.22 has four blocks, 4.25 and 4.26 five). The kernels that read them take the
// buffers d0..d4 and a parameter list p. It holds a header of 5 numbers, then 6 per block, which are the scale,
// height and width, and the strides of channels, rows and columns.
#define TMP_ARGS device const half* d0 [[buffer(3)]], device const half* d1 [[buffer(4)]], \
                 device const half* d2 [[buffer(5)]], device const half* d3 [[buffer(6)]], \
                 device const half* d4 [[buffer(7)]], constant long* p [[buffer(8)]]
#define NHEAD 5

struct Tmps {
    Tmp t0, t1, t2, t3, t4;
};

inline Tmps tmps(device const half* d0, device const half* d1, device const half* d2, device const half* d3,
                 device const half* d4, constant long* p) {
    Tmps T;
    T.t0 = tmp_at(d0, p + NHEAD); T.t1 = tmp_at(d1, p + NHEAD + 6); T.t2 = tmp_at(d2, p + NHEAD + 12);
    T.t3 = tmp_at(d3, p + NHEAD + 18); T.t4 = tmp_at(d4, p + NHEAD + 24);
    return T;
}

inline Tmp pick(thread const Tmps& T, int j) {
    return j == 0 ? T.t0 : j == 1 ? T.t1 : j == 2 ? T.t2 : j == 3 ? T.t3 : T.t4;
}

inline float tval(Tmp t, int c, int y, int x) {
    return float(t.d[(c * 4 + (y & 1) * 2 + (x & 1)) * t.sc + (y >> 1) * t.sh + (x >> 1) * t.sw]);
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

inline float up(Tmp t, Taps k, int c) {
    float a = tval(t, c, k.y0, k.x0), b = tval(t, c, k.y0, k.x1);
    float e = tval(t, c, k.y1, k.x0), f = tval(t, c, k.y1, k.x1);
    return (1.0f - k.ly) * ((1.0f - k.lx) * a + k.lx * b) + k.ly * ((1.0f - k.lx) * e + k.lx * f);
}

// Flow after blocks 0..n-1 at full-resolution pixel (Y, X), the sum of each block's upsampled flow times its scale.
inline float4 flow_at(thread const Tmps& T, int n, int Y, int X) {
    float4 fl = float4(0.0f);
    for (int j = 0; j < n; ++j) {
        Tmp t = pick(T, j);
        Taps k = taps(t, Y, X);
        fl += float(t.s) * float4(up(t, k, 0), up(t, k, 1), up(t, k, 2), up(t, k, 3));
    }
    return fl;
}

// Bilinear sample of an 8-channel fp16 frame at (px, py) in pixels, with border padding and align_corners=True.
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

// Input of block n, channels_last, with 16 channels for block 0 and 32 after, the last ones zero. Block 0 gets
//   [a_img 3, b_img 3, a_feat 4, b_feat 4, t] and later blocks also [mask, feat 8, flow / so 4],
// where a, b are the frames warped by the flow so far and everything is downsampled by `so` (block n's scale).
// p header: PH, PW, n, so, 0. Grid: (PW / so, PH / so).
kernel void block_input(device half4* xo [[buffer(0)]],
                        device const half4* S0 [[buffer(1)]],
                        device const half4* S1 [[buffer(2)]],
                        TMP_ARGS,
                        constant float& t [[buffer(9)]],
                        uint2 g [[thread_position_in_grid]]) {
    int PH = int(p[0]), PW = int(p[1]), n = int(p[2]), so = int(p[3]);
    int Wo = PW / so, Ho = PH / so;
    if (int(g.x) >= Wo || int(g.y) >= Ho) return;
    Tmps T = tmps(d0, d1, d2, d3, d4, p);
    Tmp tl = pick(T, max(n - 1, 0));  // the latest block, for its mask and features
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
            m_sum += up(tl, kk, 4);
            f_lo += float4(up(tl, kk, 5), up(tl, kk, 6), up(tl, kk, 7), up(tl, kk, 8));
            f_hi += float4(up(tl, kk, 9), up(tl, kk, 10), up(tl, kk, 11), up(tl, kk, 12));
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
    xo[o + 7] = half4(0.0f);
}

// The in-between frame at (Y, X), both images warped by the final flow and blended by sigmoid(mask).
inline float3 final_mid(device const half4* S0, device const half4* S1, thread const Tmps& T, int n,
                        int PH, int PW, int Y, int X) {
    float4 fl = flow_at(T, n, Y, X);
    Tmp tl = pick(T, n - 1);
    float m = up(tl, taps(tl, Y, X), 4);
    m = 1.0f / (1.0f + exp(-m));
    float3 w0 = warp3(S0, PH, PW, float(X) + fl.x, float(Y) + fl.y);
    float3 w1 = warp3(S1, PH, PW, float(X) + fl.z, float(Y) + fl.w);
    return w0 * m + w1 * (1.0f - m);
}

// p header: PH, PW, the number of blocks n, and the frame's unpadded size H, W. Grid: (W, H).
kernel void final_u8(device uchar* out [[buffer(0)]],
                     device const half4* S0 [[buffer(1)]],
                     device const half4* S1 [[buffer(2)]],
                     TMP_ARGS,
                     uint2 g [[thread_position_in_grid]]) {
    int PH = int(p[0]), PW = int(p[1]), n = int(p[2]), H = int(p[3]), W = int(p[4]);
    if (int(g.x) >= W || int(g.y) >= H) return;
    Tmps T = tmps(d0, d1, d2, d3, d4, p);
    float3 mid = final_mid(S0, S1, T, n, PH, PW, int(g.y), int(g.x));
    float3 v = rint(clamp(mid, 0.0f, 1.0f) * 255.0f);
    long o = (long(g.y) * W + g.x) * 3;
    out[o] = uchar(v.z); out[o + 1] = uchar(v.y); out[o + 2] = uchar(v.x);  // BGR
}

kernel void final_f16(device half* out [[buffer(0)]],
                      device const half4* S0 [[buffer(1)]],
                      device const half4* S1 [[buffer(2)]],
                      TMP_ARGS,
                      uint2 g [[thread_position_in_grid]]) {
    int PH = int(p[0]), PW = int(p[1]), n = int(p[2]), H = int(p[3]), W = int(p[4]);
    if (int(g.x) >= W || int(g.y) >= H) return;
    Tmps T = tmps(d0, d1, d2, d3, d4, p);
    float3 mid = clamp(final_mid(S0, S1, T, n, PH, PW, int(g.y), int(g.x)), 0.0f, 1.0f);
    long o = (long(g.y) * W + g.x) * 3;
    out[o] = half(mid.z); out[o + 1] = half(mid.y); out[o + 2] = half(mid.x);  // BGR
}

// Unclamped fp32 (1, 3, PH, PW) RGB for Steady HUD, which blends real pixels in before the conversion.
// Grid: (PW, PH).
kernel void final_mid_f32(device float* out [[buffer(0)]],
                          device const half4* S0 [[buffer(1)]],
                          device const half4* S1 [[buffer(2)]],
                          TMP_ARGS,
                          uint2 g [[thread_position_in_grid]]) {
    int PH = int(p[0]), PW = int(p[1]), n = int(p[2]);
    if (int(g.x) >= PW || int(g.y) >= PH) return;
    Tmps T = tmps(d0, d1, d2, d3, d4, p);
    float3 mid = final_mid(S0, S1, T, n, PH, PW, int(g.y), int(g.x));
    long plane = long(PH) * PW, o = long(g.y) * PW + g.x;
    out[o] = mid.x; out[o + plane] = mid.y; out[o + 2 * plane] = mid.z;
}

// Turns an HxWx3 BGR frame into the encoder's input, (PH, PW, 4) fp16 [r, g, b, 0] 0..1, zero padded.
// p: H, W, PH, PW.
kernel void frame_in_u8(device half4* x [[buffer(0)]], device const uchar* f [[buffer(1)]],
                        constant long* p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    int H = int(p[0]), W = int(p[1]), PH = int(p[2]), PW = int(p[3]);
    if (int(g.x) >= PW || int(g.y) >= PH) return;
    float3 v = 0.0f;
    if (int(g.y) < H && int(g.x) < W) {
        long i = (long(g.y) * W + g.x) * 3;
        v = float3(f[i + 2], f[i + 1], f[i]) / 255.0f;
    }
    x[long(g.y) * PW + g.x] = half4(half3(v), 0.0h);
}

kernel void frame_in_f16(device half4* x [[buffer(0)]], device const half* f [[buffer(1)]],
                         constant long* p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    int H = int(p[0]), W = int(p[1]), PH = int(p[2]), PW = int(p[3]);
    if (int(g.x) >= PW || int(g.y) >= PH) return;
    half3 v = 0.0h;
    if (int(g.y) < H && int(g.x) < W) {
        long i = (long(g.y) * W + g.x) * 3;
        v = clamp(half3(f[i + 2], f[i + 1], f[i]), 0.0h, 1.0h);
    }
    x[long(g.y) * PW + g.x] = half4(v, 0.0h);
}

// The encoder's last layer, LeakyReLU then ConvTranspose2d(16, 4, 4, stride 2, padding 1), packed with the image
// into a (PH, PW, 8) fp16 frame slot [r, g, b, f0, f1, f2, f3, 0].
// a: the layer before, (PH/2, PW/2, 16) fp16 before its LeakyReLU. wt: the weights, a float4 of the 4 outputs per
// ky, kx and input channel. b: the bias. p: PH, PW. Grid: (PW, PH).
kernel void encode_out(device half4* S [[buffer(0)]], device const half4* x [[buffer(1)]],
                       device const half4* a [[buffer(2)]], device const float4* wt [[buffer(3)]],
                       constant float* b [[buffer(4)]], constant long* p [[buffer(5)]],
                       uint2 g [[thread_position_in_grid]]) {
    int PH = int(p[0]), PW = int(p[1]);
    int Y = int(g.y), X = int(g.x);
    if (X >= PW || Y >= PH) return;
    int h2 = PH / 2, w2 = PW / 2;
    float4 acc = float4(b[0], b[1], b[2], b[3]);
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

// Residual block tail, in place on the convolution's output, y = leaky_relu(y + x, 0.2). Its weights hold beta.
kernel void add_lrelu(device half4* y [[buffer(0)]], device const half4* x [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float4 v = float4(y[i]) + float4(x[i]);
    y[i] = half4(select(v * 0.2f, v, v >= 0.0f));
}
"""


MAX_BLOCKS = 5


def supported(scale_list) -> bool:
    """Whether every block's scale is a whole number (motion detail 1.0, 0.5 or 0.25, not 2.0)."""
    return len(scale_list) <= MAX_BLOCKS and all(float(s).is_integer() and s >= 1 for s in scale_list)


_lib = None


def library():
    """The compiled kernels. They compile once per process, in a fraction of a second."""
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(KERNELS)
    return _lib


def _conv(src: torch.nn.Conv2d, cin: int | None = None, beta: torch.Tensor | None = None) -> torch.nn.Conv2d:
    """A copy of src (channels_last), with its input zero-padded to `cin` channels, `beta` folded in, or both."""
    w, b = src.weight.detach(), src.bias.detach()
    if beta is not None:
        beta = beta.detach().reshape(-1).to(w.dtype)
        w, b = w * beta.view(-1, 1, 1, 1), b * beta
    if cin is not None and cin > w.shape[1]:
        w = F.pad(w, (0, 0, 0, 0, 0, cin - w.shape[1]))
    c = torch.nn.Conv2d(w.shape[1], w.shape[0], src.kernel_size, src.stride, src.padding, src.dilation,
                        bias=True).to(w.device, w.dtype)
    with torch.no_grad():
        c.weight.copy_(w)
        c.bias.copy_(b)
    return c.eval().requires_grad_(False).to(memory_format=torch.channels_last)


def _deconv(src: torch.nn.ConvTranspose2d, cout: int) -> torch.nn.ConvTranspose2d:
    """A copy of src (channels_last) with only its first `cout` outputs."""
    d = torch.nn.ConvTranspose2d(src.in_channels, cout, src.kernel_size, src.stride, src.padding).to(
        src.weight.device, src.weight.dtype)
    with torch.no_grad():
        d.weight.copy_(src.weight[:, :cout])
        d.bias.copy_(src.bias[:cout])
    return d.eval().requires_grad_(False).to(memory_format=torch.channels_last)


class _Block:
    """One IFBlock's convolutions, from its input (channels_last, padded to 16 or 32 channels) to its transposed conv's
    output."""

    def __init__(self, blk: torch.nn.Module, cin: int, last: bool) -> None:
        self.c0a = _conv(blk.conv0[0][0], cin)
        self.c0b = _conv(blk.conv0[1][0])
        self.res = [_conv(rc.conv, beta=rc.beta) for rc in blk.convblock]
        # The last block's features (channels 5..12 once pixel-shuffled, 20..51 before) are never used, so only its
        # flow and mask are computed.
        self.deconv = _deconv(blk.lastconv[0], 20 if last else blk.lastconv[0].out_channels)

    def __call__(self, x: torch.Tensor, lib) -> torch.Tensor:
        x = F.leaky_relu(self.c0a(x), 0.2, inplace=True)
        x = F.leaky_relu(self.c0b(x), 0.2, inplace=True)
        for c in self.res:
            y = c(x)
            lib.add_lrelu(y, x, threads=y.numel() // 4)
            x = y
        return self.deconv(x)  # (1, 52, h/2, w/2), 13 channels at (h, w) once pixel-shuffled. The last block's has 20.


class MetalIFNet:
    """FastIFNet's forward for MPS. Frames sit in new_slot()'s (PH, PW, 8) fp16 slots, and the output is written in
    place."""

    def __init__(self, net: torch.nn.Module, encode: torch.nn.Module, h: int, w: int, ph: int, pw: int,
                 device: torch.device) -> None:
        if not supported(net.scale_list):
            raise ValueError(f"MetalIFNet needs whole-number block scales, not {net.scale_list}")
        self.lib = library()
        self.h, self.w, self.ph, self.pw = h, w, ph, pw
        self.device = device
        self.scales = [int(s) for s in net.scale_list]
        n = len(self.scales)
        if not 1 <= n <= MAX_BLOCKS:
            raise ValueError(f"MetalIFNet takes up to {MAX_BLOCKS} blocks, not {n}")
        for i in range(n):  # block_input writes 15 channels for the first block, then 28
            blk = getattr(net, f"block{i}")
            if blk.conv0[0][0].in_channels != (15 if i == 0 else 28) or blk.lastconv[0].out_channels != 52:
                raise ValueError("MetalIFNet expects RIFE 4.2x's blocks")
        with torch.no_grad():
            self.blocks = [_Block(getattr(net, f"block{i}"), 16 if i == 0 else 32, i == n - 1) for i in range(n)]
            # the encoder (framegen/rife Head), channels_last, its last layer a kernel
            self.enc = [_conv(encode.cnn0, 4), _conv(encode.cnn1), _conv(encode.cnn2)]
            w3 = encode.cnn3.weight.detach().float()  # (16 in, 4 out, 4, 4), permuted below to (ky, kx, in, out)
            if tuple(w3.shape) != (16, 4, 4, 4) or encode.cnn3.stride != (2, 2) or encode.cnn3.padding != (1, 1):
                raise ValueError("MetalIFNet expects RIFE 4.2x's encoder")
            self.enc_w = w3.permute(2, 3, 0, 1).contiguous()
            self.enc_b = encode.cnn3.bias.detach().float().cpu().tolist()
        self.xin = torch.zeros((1, 4, ph, pw), dtype=torch.half, device=device).contiguous(
            memory_format=torch.channels_last)  # the encoder's input
        self.x = [torch.zeros((1, 16 if i == 0 else 32, ph // s, pw // s), dtype=torch.half, device=device)
                  .contiguous(memory_format=torch.channels_last) for i, s in enumerate(self.scales)]
        self.mid = torch.zeros((1, 3, ph, pw), dtype=torch.float, device=device)  # Steady HUD's
        self.grid_full = (pw, ph)

    def new_slot(self) -> torch.Tensor:
        """A frame slot for prepare(), (PH, PW, 8) fp16 [r, g, b, f0, f1, f2, f3, 0]."""
        return torch.zeros((self.ph, self.pw, 8), dtype=torch.half, device=self.device)

    @staticmethod
    def rgb(slot: torch.Tensor) -> torch.Tensor:
        """A slot's frame as (1, 3, PH, PW) fp32 RGB 0..1."""
        return slot[:, :, :3].permute(2, 0, 1).unsqueeze(0).float()

    @torch.no_grad()
    def prepare(self, frame: torch.Tensor, slot: torch.Tensor) -> None:
        """Encodes an HxWx3 BGR frame on the GPU (uint8, or fp16 0..1) into `slot`."""
        k = self.lib.frame_in_u8 if frame.dtype == torch.uint8 else self.lib.frame_in_f16
        k(self.xin, frame, [self.h, self.w, self.ph, self.pw], threads=self.grid_full, group_size=(32, 8))
        a = F.leaky_relu(self.enc[0](self.xin), 0.2, inplace=True)
        a = F.leaky_relu(self.enc[1](a), 0.2, inplace=True)
        a = self.enc[2](a).contiguous(memory_format=torch.channels_last)
        self.lib.encode_out(slot, self.xin, a, self.enc_w, self.enc_b, [self.ph, self.pw],
                            threads=self.grid_full, group_size=(32, 8))

    def _tmp_params(self, i: int, d: torch.Tensor) -> list[int]:
        s = self.scales[i]
        st = d.stride()
        return [s, self.ph // s, self.pw // s, st[1], st[2], st[3]]

    @staticmethod
    def _args(ds: list[torch.Tensor], tp: list[int], head: list[int], spare: torch.Tensor) -> list:
        """A kernel's TMP_ARGS, the five block outputs (any tensor stands in for an unused one) and the parameter
        list."""
        return ds + [spare] * (MAX_BLOCKS - len(ds)) + [head + tp + [0] * (6 * MAX_BLOCKS - len(tp))]

    @torch.no_grad()
    def flow(self, src0: torch.Tensor, src1: torch.Tensor, t: float) -> tuple[list[torch.Tensor], list[int]]:
        """Runs the blocks. Returns their outputs and the parameters the final kernels read them by."""
        lib = self.lib
        ds: list[torch.Tensor] = []
        tp: list[int] = []
        for i, (blk, s) in enumerate(zip(self.blocks, self.scales)):
            x = self.x[i]
            lib.block_input(x, src0, src1, *self._args(ds, tp, [self.ph, self.pw, i, s, 0], src0), float(t),
                            threads=(self.pw // s, self.ph // s), group_size=(32, 8))
            d = blk(x, lib)
            ds.append(d)
            tp += self._tmp_params(i, d)
        return ds, tp

    @torch.no_grad()
    def interpolate(self, src0: torch.Tensor, src1: torch.Tensor, t: float, out: torch.Tensor) -> None:
        """Writes the frame at t between two slots to `out`, HxWx3 BGR (uint8, or fp16 0..1)."""
        ds, tp = self.flow(src0, src1, t)
        k = self.lib.final_u8 if out.dtype == torch.uint8 else self.lib.final_f16
        k(out, src0, src1, *self._args(ds, tp, [self.ph, self.pw, len(ds), self.h, self.w], src0),
          threads=(self.w, self.h), group_size=(32, 8))

    @torch.no_grad()
    def interpolate_mid(self, src0: torch.Tensor, src1: torch.Tensor, t: float) -> torch.Tensor:
        """The frame at t as (1, 3, PH, PW) fp32 RGB, unclamped (FastIFNet's output, for Steady HUD)."""
        ds, tp = self.flow(src0, src1, t)
        self.lib.final_mid_f32(self.mid, src0, src1, *self._args(ds, tp, [self.ph, self.pw, len(ds), self.h, self.w],
                                                                 src0), threads=self.grid_full, group_size=(32, 8))
        return self.mid
