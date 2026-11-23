//! The MPSGraph layers against a plain CPU reference (PyTorch's Conv2d / ConvTranspose2d semantics), on small random
//! tensors: checks layouts (NHWC data, OIHW weights, the transposed convolution's weight order) and padding.

use half::f16;
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use objc2_metal_performance_shaders::MPSCommandBuffer;
use sp_core::Gpu;
use sp_rife::graph::{Conv, Layer, Stack, TensorBuf};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn halves(&mut self, n: usize, scale: f32) -> Vec<f16> {
        (0..n).map(|_| f16::from_f32(self.next() * scale)).collect()
    }
}

/// x: (h, w, c) NHWC. Returns (oh, ow, cout).
fn conv_ref(x: &[f32], h: usize, w: usize, c: &Conv) -> (Vec<f32>, usize, usize) {
    let oh = (h + 2 * c.pad - c.k) / c.stride + 1;
    let ow = (w + 2 * c.pad - c.k) / c.stride + 1;
    let mut out = vec![0f32; oh * ow * c.cout];
    for oy in 0..oh {
        for ox in 0..ow {
            for o in 0..c.cout {
                let mut acc = c.b[o].to_f32();
                for ky in 0..c.k {
                    for kx in 0..c.k {
                        let y = (oy * c.stride + ky) as isize - c.pad as isize;
                        let xx = (ox * c.stride + kx) as isize - c.pad as isize;
                        if y < 0 || xx < 0 || y >= h as isize || xx >= w as isize {
                            continue;
                        }
                        for i in 0..c.cin {
                            acc += x[(y as usize * w + xx as usize) * c.cin + i]
                                * c.w[((o * c.cin + i) * c.k + ky) * c.k + kx].to_f32();
                        }
                    }
                }
                out[(oy * ow + ox) * c.cout + o] = acc;
            }
        }
    }
    (out, oh, ow)
}

/// PyTorch ConvTranspose2d: weights (cin, cout, k, k).
fn deconv_ref(x: &[f32], h: usize, w: usize, c: &Conv) -> (Vec<f32>, usize, usize) {
    let oh = (h - 1) * c.stride + c.k - 2 * c.pad;
    let ow = (w - 1) * c.stride + c.k - 2 * c.pad;
    let mut out = vec![0f32; oh * ow * c.cout];
    for p in 0..oh * ow {
        for o in 0..c.cout {
            out[p * c.cout + o] = c.b[o].to_f32();
        }
    }
    for y in 0..h {
        for xx in 0..w {
            for ky in 0..c.k {
                for kx in 0..c.k {
                    let oy = (y * c.stride + ky) as isize - c.pad as isize;
                    let ox = (xx * c.stride + kx) as isize - c.pad as isize;
                    if oy < 0 || ox < 0 || oy >= oh as isize || ox >= ow as isize {
                        continue;
                    }
                    for i in 0..c.cin {
                        let v = x[(y * w + xx) * c.cin + i];
                        for o in 0..c.cout {
                            out[(oy as usize * ow + ox as usize) * c.cout + o] +=
                                v * c.w[((i * c.cout + o) * c.k + ky) * c.k + kx].to_f32();
                        }
                    }
                }
            }
        }
    }
    (out, oh, ow)
}

fn lrelu(v: f32) -> f32 {
    if v >= 0.0 { v } else { 0.2 * v }
}

#[test]
fn layers_match_pytorch_semantics() {
    let mut r = Rng(7);
    let (h, w, cin) = (12usize, 20usize, 8usize);
    let mk = |r: &mut Rng, cout, cin, k, stride| Conv {
        cout, cin, k, stride, pad: if k == 4 { 1 } else { k / 2 },
        w: r.halves(cout * cin * k * k, 0.3), b: r.halves(cout, 0.1),
    };
    let c0 = mk(&mut r, 16, cin, 3, 2);
    let res = mk(&mut r, 16, 16, 3, 1);
    let dc = Conv { cout: 6, cin: 16, k: 4, stride: 2, pad: 1, w: r.halves(16 * 6 * 16, 0.3), b: r.halves(6, 0.1) };
    let layers = vec![Layer::ConvLrelu(c0.clone()), Layer::Residual(res.clone()), Layer::Deconv(dc.clone())];
    let x: Vec<f16> = r.halves(h * w * cin, 1.0);

    // CPU
    let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let (a, ah, aw) = conv_ref(&xf, h, w, &c0);
    let a: Vec<f32> = a.iter().map(|&v| lrelu(v)).collect();
    let (b, _, _) = conv_ref(&a, ah, aw, &res);
    let b: Vec<f32> = b.iter().zip(&a).map(|(&y, &x)| lrelu(y + x)).collect();
    let (want, oh, ow) = deconv_ref(&b, ah, aw, &dc);
    assert_eq!((oh, ow), (h, w));

    // GPU
    let gpu = Gpu::shared();
    let st = Stack::new(&gpu.device, &layers, h, w, cin);
    assert_eq!(st.out_shape, [1, h, w, 6]);
    let tin = TensorBuf::new(&gpu.device, st.in_shape);
    let tout = TensorBuf::new(&gpu.device, st.out_shape);
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), tin.buf.contents().as_ptr() as *mut f16, x.len());
    }
    let cb = unsafe { MPSCommandBuffer::commandBufferFromCommandQueue(&gpu.queue) };
    st.encode(&cb, &tin, &tout);
    cb.commit();
    unsafe { cb.commandBuffer() }.waitUntilCompleted();
    let got = unsafe { std::slice::from_raw_parts(tout.buf.contents().as_ptr() as *const f16, want.len()) };
    let worst = got.iter().zip(&want).map(|(g, w)| (g.to_f32() - w).abs()).fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(worst < 0.01 * scale.max(1.0), "worst difference {worst} (largest value {scale})");
}
