//! The flow network and encoder for one stream size and motion detail. Weights are laid out as framegen/metalrife.py
//! lays them out: fp16, beta folded into the residual convolutions, inputs padded to 16 or 32 channels, and the last
//! block's unused feature outputs dropped. The convolutions run as MPSGraph executables (graph.rs) and everything
//! between them as Metal kernels (kernels.rs).
//!
//! There is one change from metalrife.py. Each block's last layer, a 4x4 stride-2 transposed convolution, runs as a
//! 3x3 convolution at a quarter of the block's resolution that makes all four output parities at once. That's the
//! same sums plus zeros, and MPSGraph's transposed convolution runs at a third of the speed. Its output channels are
//! ordered so the kernels load flow, mask and features as half4s (subpixel_layout). At 1080p that saves about 0.6 ms.

use objc2::rc::Retained;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use objc2_metal_performance_shaders::MPSCommandBuffer;
use sp_core::{Frame, Gpu, PixelKind};

use half::f16;

use crate::graph::{Conv, Layer, Stack, TensorBuf};
use crate::kernels::{self, BlockOut, Buffer, Kernels, MAX_BLOCKS, Params};
use crate::weights::{Tensor, Weights};
use crate::{Error, Result};

/// A RIFE version this engine runs, with its padding modulo (vs-rife's) and its blocks' scales at motion detail 1.0.
#[derive(Clone, Copy, Debug)]
pub struct Model {
    pub version: &'static str,
    pub modulo: usize,
    pub scales: &'static [usize],
}

pub const MODELS: &[Model] = &[
    Model { version: "4.22.lite", modulo: 32, scales: &[8, 4, 2, 1] },
    Model { version: "4.25.lite", modulo: 128, scales: &[32, 16, 8, 4, 1] },
    Model { version: "4.25", modulo: 64, scales: &[16, 8, 4, 2, 1] },
    Model { version: "4.26", modulo: 64, scales: &[16, 8, 4, 2, 1] },
];

pub fn model(version: &str) -> Result<Model> {
    MODELS.iter().copied().find(|m| m.version == version).ok_or_else(|| {
        let known: Vec<&str> = MODELS.iter().map(|m| m.version).collect();
        Error::Other(format!("unknown RIFE model {version:?}; choose from {known:?}"))
    })
}

/// The blocks' scales at a motion detail (framegen's `scale_list = [8 / scale, ...]`), if all are whole numbers.
/// The kernels take only whole numbers, so motion detail 1.0, 0.5 or 0.25. Detail 2.0 would need half-pixel scales.
pub fn block_scales(m: &Model, detail: f32) -> Result<Vec<usize>> {
    if ![0.25f32, 0.5, 1.0, 2.0].contains(&detail) {
        return Err(Error::Other(format!("motion detail must be 0.25, 0.5, 1.0 or 2.0, not {detail}")));
    }
    let mut out = Vec::new();
    for &s in m.scales {
        let v = s as f32 / detail;
        if v < 1.0 || v.fract() != 0.0 {
            return Err(Error::Other(format!(
                "RIFE {} at motion detail {detail} needs a block scale of {v}; the Metal engine takes whole-number \
                 scales only (motion detail 1.0, 0.5 or 0.25)",
                m.version
            )));
        }
        out.push(v as usize);
    }
    Ok(out)
}

/// framegen/interpolator.py's padding: a multiple of `max(modulo, modulo / scale)`.
pub fn padded(m: &Model, detail: f32, w: usize, h: usize) -> (usize, usize) {
    let step = (m.modulo).max((m.modulo as f32 / detail) as usize);
    (w.div_ceil(step) * step, h.div_ceil(step) * step)
}

fn get<'a>(wts: &'a Weights, name: &str) -> Result<&'a Tensor> {
    wts.get(&format!("module.{name}"))
        .or_else(|| wts.get(name))
        .ok_or_else(|| Error::Other(format!("RIFE checkpoint is missing {name}")))
}

fn h16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

/// A Conv2d's weights in fp16, with input channels zero-padded to `cin`, `beta` folded in, or both. As in
/// metalrife.py _conv, beta is folded in fp16, so w * beta is rounded once.
fn conv(wts: &Weights, name: &str, stride: usize, cin: Option<usize>, beta: Option<&str>) -> Result<Conv> {
    let w = get(wts, &format!("{name}.weight"))?;
    let b = get(wts, &format!("{name}.bias"))?;
    let [cout, cin0, k, k2] = w.shape[..] else { return Err(Error::Other(format!("{name}: not a 2D convolution"))) };
    if k != k2 || b.numel() != cout {
        return Err(Error::Other(format!("{name}: unexpected shape {:?}", w.shape)));
    }
    let mut wh = h16(&w.data);
    let mut bh = h16(&b.data);
    if let Some(beta) = beta {
        let beta = h16(&get(wts, beta)?.data);
        if beta.len() != cout {
            return Err(Error::Other(format!("{name}: beta doesn't match")));
        }
        for o in 0..cout {
            let s = beta[o].to_f32();
            for v in &mut wh[o * cin0 * k * k..(o + 1) * cin0 * k * k] {
                *v = f16::from_f32(v.to_f32() * s);
            }
            bh[o] = f16::from_f32(bh[o].to_f32() * s);
        }
    }
    let cin = cin.unwrap_or(cin0);
    if cin < cin0 {
        return Err(Error::Other(format!("{name}: {cin0} input channels, more than the {cin} expected")));
    }
    if cin > cin0 {
        let mut p = vec![f16::ZERO; cout * cin * k * k];
        for o in 0..cout {
            for i in 0..cin0 {
                let src = (o * cin0 + i) * k * k;
                let dst = (o * cin + i) * k * k;
                p[dst..dst + k * k].copy_from_slice(&wh[src..src + k * k]);
            }
        }
        wh = p;
    }
    Ok(Conv { cout, cin, k, stride, pad: k / 2, w: wh, b: bh })
}

/// A ConvTranspose2d (4x4, stride 2, padding 1) with only its first `cout` outputs.
fn deconv(wts: &Weights, name: &str, cout: usize) -> Result<Conv> {
    let w = get(wts, &format!("{name}.weight"))?;
    let b = get(wts, &format!("{name}.bias"))?;
    let [cin, cout0, 4, 4] = w.shape[..] else { return Err(Error::Other(format!("{name}: unexpected shape"))) };
    if cout > cout0 {
        return Err(Error::Other(format!("{name}: {cout0} outputs, fewer than {cout}")));
    }
    let wh = h16(&w.data);
    let mut out = vec![f16::ZERO; cin * cout * 16];
    for i in 0..cin {
        for o in 0..cout {
            let src = (i * cout0 + o) * 16;
            out[(i * cout + o) * 16..(i * cout + o + 1) * 16].copy_from_slice(&wh[src..src + 16]);
        }
    }
    Ok(Conv { cout, cin, k: 4, stride: 2, pad: 1, w: out, b: h16(&b.data[..cout]) })
}

/// One IFBlock's layers (metalrife.py _Block), with the input padded to `cin` channels. The last block makes only
/// flow and mask (20 channels before the PixelShuffle).
pub fn block_layers(wts: &Weights, i: usize, cin: usize, last: bool) -> Result<Vec<Layer>> {
    let p = format!("block{i}");
    let c0a = conv(wts, &format!("{p}.conv0.0.0"), 2, Some(cin), None)?;
    let c0b = conv(wts, &format!("{p}.conv0.1.0"), 2, None, None)?;
    let mut layers = vec![Layer::ConvLrelu(c0a), Layer::ConvLrelu(c0b)];
    for j in 0.. {
        let name = format!("{p}.convblock.{j}.conv");
        if get(wts, &format!("{name}.weight")).is_err() {
            break;
        }
        layers.push(Layer::Residual(conv(wts, &name, 1, None, Some(&format!("{p}.convblock.{j}.beta")))?));
    }
    let dc = deconv(wts, &format!("{p}.lastconv.0"), if last { 20 } else { 52 })?;
    layers.push(Layer::Deconv(dc));
    Ok(layers)
}

/// The block output layout the kernels read (kernels.rs BlockOut). It moves deconv_as_conv3's output channel
/// P * k4 + o, where P is the transposed convolution's output parity and o = c * 4 + q its channel before the
/// PixelShuffle, with c the channel after it and q the parity there. With sub = P * 4 + q, flow (c < 4) goes to
/// sub * 4 + c, mask (c = 4) to 64 + sub and features (c > 4) to 80 + sub * 8 + c - 5, so flow and features load as
/// half4s.
fn subpixel_layout(conv3: Conv, k4: usize) -> Conv {
    let kk = conv3.cin * conv3.k * conv3.k;
    let mut w = vec![f16::ZERO; conv3.w.len()];
    let mut b = vec![f16::ZERO; conv3.b.len()];
    for p in 0..4 {
        for o in 0..k4 {
            let (c, q) = (o / 4, o % 4);
            let sub = p * 4 + q;
            let new = if c < 4 { sub * 4 + c } else if c == 4 { 64 + sub } else { 80 + sub * 8 + c - 5 };
            let old = p * k4 + o;
            w[new * kk..(new + 1) * kk].copy_from_slice(&conv3.w[old * kk..(old + 1) * kk]);
            b[new] = conv3.b[old];
        }
    }
    Conv { w, b, ..conv3 }
}

/// The engine for one stream size: encoder, blocks, their buffers.
pub struct Net {
    pub h: usize,
    pub w: usize,
    pub ph: usize,
    pub pw: usize,
    pub scales: Vec<usize>,
    k: Kernels,
    enc: Stack,
    xin: TensorBuf,
    enc_a: TensorBuf,
    enc_w: Retained<Buffer>,
    enc_b: [f32; 4],
    blocks: Vec<Stack>,
    xs: Vec<TensorBuf>,
    ds: Vec<TensorBuf>,
    outs: [BlockOut; MAX_BLOCKS],
}

unsafe impl Send for Net {}
unsafe impl Sync for Net {}

impl Net {
    pub fn new(wts: &Weights, m: &Model, detail: f32, w: usize, h: usize) -> Result<Net> {
        let scales = block_scales(m, detail)?;
        let (pw, ph) = padded(m, detail, w, h);
        let gpu = Gpu::shared();
        let dev = &*gpu.device;
        let k = Kernels::new()?;
        let n = scales.len();
        if n > MAX_BLOCKS {
            return Err(Error::Other(format!("{n} blocks; the kernels take up to {MAX_BLOCKS}")));
        }

        // The encoder (framegen/rife Head). Its three convolutions run here and the transposed one as a kernel.
        let enc_layers = vec![
            Layer::ConvLrelu(conv(wts, "encode.cnn0", 2, Some(4), None)?),
            Layer::ConvLrelu(conv(wts, "encode.cnn1", 1, None, None)?),
            Layer::Conv(conv(wts, "encode.cnn2", 1, None, None)?),
        ];
        let enc = Stack::new(dev, &enc_layers, ph, pw, 4);
        if enc.out_shape[3] != 16 {
            return Err(Error::Other("expected RIFE 4.2x's encoder (16 channels)".into()));
        }
        let w3 = get(wts, "encode.cnn3.weight")?;
        let b3 = get(wts, "encode.cnn3.bias")?;
        if w3.shape != [16, 4, 4, 4] || b3.numel() != 4 {
            return Err(Error::Other("expected RIFE 4.2x's encoder (a 16 -> 4 transposed convolution)".into()));
        }
        // Lay (in, out, ky, kx) out as a float4 of the outputs per (ky, kx, in), from the fp16 weights as
        // metalrife.py does.
        let r = |v: f32| f16::from_f32(v).to_f32();
        let mut ew = vec![0f32; 16 * 16 * 4];
        for ky in 0..4 {
            for kx in 0..4 {
                for i in 0..16 {
                    for o in 0..4 {
                        ew[((ky * 4 + kx) * 16 + i) * 4 + o] = r(w3.data[((i * 4 + o) * 4 + ky) * 4 + kx]);
                    }
                }
            }
        }
        let enc_w = kernels::buffer_with(dev, &ew);
        let enc_b = [r(b3.data[0]), r(b3.data[1]), r(b3.data[2]), r(b3.data[3])];
        let xin = TensorBuf::new(dev, [1, ph, pw, 4]);
        let enc_a = TensorBuf::new(dev, enc.out_shape);

        let mut blocks = Vec::new();
        let mut xs = Vec::new();
        let mut ds = Vec::new();
        let mut outs = [BlockOut::default(); MAX_BLOCKS];
        for (i, &s) in scales.iter().enumerate() {
            let cin = if i == 0 { 16 } else { 32 };
            let real_in = get(wts, &format!("block{i}.conv0.0.0.weight"))?.shape[1];
            if real_in != if i == 0 { 15 } else { 28 } {
                return Err(Error::Other(format!("block{i} takes {real_in} channels: not a RIFE 4.2x block")));
            }
            let mut layers = block_layers(wts, i, cin, i == n - 1)?;
            // The transposed convolution as one 3x3 convolution making its four output parities (graph.rs
            // deconv_as_conv3). It does the same sums and is faster in MPSGraph than the transposed convolution.
            let Some(Layer::Deconv(dc)) = layers.pop() else { unreachable!() };
            let cout = dc.cout;
            layers.push(Layer::Conv(subpixel_layout(crate::graph::deconv_as_conv3(&dc), cout)));
            let (bh, bw) = (ph / s, pw / s);
            let st = Stack::new(dev, &layers, bh, bw, cin);
            let [_, oh, ow, oc] = st.out_shape;
            if oh * 4 != bh || ow * 4 != bw || oc != 4 * cout {
                return Err(Error::Other(format!("block{i}: output {oh}x{ow} doesn't match its input {bh}x{bw}")));
            }
            outs[i] = BlockOut {
                s: s as i32,
                h: bh as i32,
                w: bw as i32,
                pad: 0,
                sh: (ow * oc) as i32,
                sw: oc as i32,
            };
            xs.push(TensorBuf::new(dev, st.in_shape));
            ds.push(TensorBuf::new(dev, st.out_shape));
            blocks.push(st);
        }
        if get(wts, &format!("block{n}.conv0.0.0.weight")).is_ok() {
            return Err(Error::Other(format!("the checkpoint has more than {n} blocks: wrong model version?")));
        }
        Ok(Net { h, w, ph, pw, scales, k, enc, xin, enc_a, enc_w, enc_b, blocks, xs, ds, outs })
    }

    /// A frame slot: (PH, PW, 8) fp16 [r, g, b, f0, f1, f2, f3, 0].
    pub fn new_slot(&self) -> Retained<Buffer> {
        kernels::buffer(&Gpu::shared().device, self.ph * self.pw * 16)
    }

    /// A Steady HUD edge map (PH x PW bytes) or mask (PH x PW halves).
    pub fn new_plane(&self, bytes_per_pixel: usize) -> Retained<Buffer> {
        kernels::buffer(&Gpu::shared().device, self.ph * self.pw * bytes_per_pixel)
    }

    /// Encodes `frame` (BGRA8 or RGBA16F, h x w) into `slot`.
    pub fn encode_prepare(&self, cb: &MPSCommandBuffer, frame: &Frame, slot: &Buffer) {
        let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
        let p = [self.h as i32, self.w as i32, self.ph as i32, self.pw as i32];
        kernels::set_buffer(&enc, &self.xin.buf, 0);
        unsafe { enc.setTexture_atIndex(Some(&frame.texture), 0) };
        kernels::set_bytes(&enc, &p, 1);
        kernels::dispatch(&enc, &self.k.frame_in, self.pw, self.ph);
        enc.endEncoding();
        self.enc.encode(cb, &self.xin, &self.enc_a);
        let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
        kernels::set_buffer(&enc, slot, 0);
        kernels::set_buffer(&enc, &self.xin.buf, 1);
        kernels::set_buffer(&enc, &self.enc_a.buf, 2);
        kernels::set_buffer(&enc, &self.enc_w, 3);
        kernels::set_bytes(&enc, &self.enc_b, 4);
        kernels::set_bytes(&enc, &[self.ph as i32, self.pw as i32], 5);
        kernels::dispatch(&enc, &self.k.encode_out, self.pw, self.ph);
        enc.endEncoding();
    }

    fn params(&self, n: usize, so: usize, t: f32, steady: bool) -> Params {
        Params {
            ph: self.ph as i32,
            pw: self.pw as i32,
            n: n as i32,
            so: so as i32,
            h: self.h as i32,
            w: self.w as i32,
            steady: steady as i32,
            t,
            b: self.outs,
            ..Default::default()
        }
    }

    fn set_tmps(&self, enc: &kernels::Encoder, s0: &Buffer, p: &Params) {
        for j in 0..MAX_BLOCKS {
            kernels::set_buffer(enc, self.ds.get(j).map(|d| &*d.buf).unwrap_or(s0), 3 + j);
        }
        kernels::set_bytes(enc, p, 8);
    }

    /// Encodes the flow network between slots s0 (t = 0) and s1 (t = 1) and the frame at t into `out` (h x w,
    /// BGRA8 or RGBA16F), with Steady HUD's mask if given.
    pub fn encode_interpolate(&self, cb: &MPSCommandBuffer, s0: &Buffer, s1: &Buffer, t: f32, out: &Frame,
                              hud: Option<&Buffer>) {
        for (i, &s) in self.scales.iter().enumerate() {
            let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
            let p = self.params(i, s, t, false);
            kernels::set_buffer(&enc, &self.xs[i].buf, 0);
            kernels::set_buffer(&enc, s0, 1);
            kernels::set_buffer(&enc, s1, 2);
            self.set_tmps(&enc, s0, &p);
            kernels::dispatch(&enc, &self.k.block_input, self.pw / s, self.ph / s);
            enc.endEncoding();
            if i == 0 {
                // Start the GPU on the first block's input while MPSGraph encodes the rest.
                unsafe { cb.commitAndContinue() };
            }
            self.blocks[i].encode(cb, &self.xs[i], &self.ds[i]);
        }
        let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
        let p = self.params(self.scales.len(), 1, t, hud.is_some());
        unsafe { enc.setTexture_atIndex(Some(&out.texture), 0) };
        kernels::set_buffer(&enc, s0, 1);
        kernels::set_buffer(&enc, s1, 2);
        self.set_tmps(&enc, s0, &p);
        kernels::set_buffer(&enc, hud.unwrap_or(s0), 9);
        let hdr: i32 = (out.kind == PixelKind::Rgba16f) as i32;
        kernels::set_bytes(&enc, &hdr, 10);
        kernels::dispatch(&enc, &self.k.final_out, self.w, self.h);
        enc.endEncoding();
    }

    /// For Steady HUD, marks in `edges` where slot `s` has brightness edges.
    pub fn encode_edges(&self, cb: &MPSCommandBuffer, s: &Buffer, edges: &Buffer) {
        let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
        kernels::set_buffer(&enc, edges, 0);
        kernels::set_buffer(&enc, s, 1);
        kernels::set_bytes(&enc, &[self.ph as i32, self.pw as i32], 2);
        kernels::dispatch(&enc, &self.k.steady_edges, self.pw, self.ph);
        enc.endEncoding();
    }

    /// For Steady HUD, writes to `mask` where it acts between slots a and b (b the newest), from their edge maps.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_mask(&self, cb: &MPSCommandBuffer, a: &Buffer, b: &Buffer, ea: &Buffer, eb: &Buffer,
                       mask: &Buffer, threshold: f32) {
        let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
        kernels::set_buffer(&enc, mask, 0);
        kernels::set_buffer(&enc, a, 1);
        kernels::set_buffer(&enc, b, 2);
        kernels::set_buffer(&enc, ea, 3);
        kernels::set_buffer(&enc, eb, 4);
        kernels::set_bytes(&enc, &[self.ph as i32, self.pw as i32], 5);
        kernels::set_bytes(&enc, &threshold, 6);
        kernels::dispatch(&enc, &self.k.steady_mask, self.pw, self.ph);
        enc.endEncoding();
    }
}

/// Where the time goes (for the bench's breakdown).
pub mod profile {
    use super::*;
    use std::time::Instant;

    /// Milliseconds per run of `stage`, run `reps` times back to back in one command buffer after a warm-up. That
    /// shows the GPU's own time rather than the submission and wake-up around one call. MPSGraph commits command
    /// buffers as it goes, so GPU timestamps of one buffer would miss part of its work.
    fn per_run(reps: usize, stage: &dyn Fn(&MPSCommandBuffer)) -> f64 {
        let run = |n: usize| {
            let cb = unsafe { MPSCommandBuffer::commandBufferFromCommandQueue(&Gpu::shared().queue) };
            for _ in 0..n {
                stage(&cb);
            }
            cb.commit();
            unsafe { cb.commandBuffer() }.waitUntilCompleted();
        };
        run(2);
        let t = Instant::now();
        run(reps);
        t.elapsed().as_secs_f64() * 1e3 / reps as f64
    }

    impl Net {
        fn encode_frame_in(&self, cb: &MPSCommandBuffer, frame: &Frame) {
            let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
            let p = [self.h as i32, self.w as i32, self.ph as i32, self.pw as i32];
            kernels::set_buffer(&enc, &self.xin.buf, 0);
            unsafe { enc.setTexture_atIndex(Some(&frame.texture), 0) };
            kernels::set_bytes(&enc, &p, 1);
            kernels::dispatch(&enc, &self.k.frame_in, self.pw, self.ph);
            enc.endEncoding();
        }

        fn encode_encode_out(&self, cb: &MPSCommandBuffer, slot: &Buffer) {
            let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
            kernels::set_buffer(&enc, slot, 0);
            kernels::set_buffer(&enc, &self.xin.buf, 1);
            kernels::set_buffer(&enc, &self.enc_a.buf, 2);
            kernels::set_buffer(&enc, &self.enc_w, 3);
            kernels::set_bytes(&enc, &self.enc_b, 4);
            kernels::set_bytes(&enc, &[self.ph as i32, self.pw as i32], 5);
            kernels::dispatch(&enc, &self.k.encode_out, self.pw, self.ph);
            enc.endEncoding();
        }

        fn encode_block_input(&self, cb: &MPSCommandBuffer, i: usize, s0: &Buffer, s1: &Buffer, t: f32) {
            let s = self.scales[i];
            let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
            let p = self.params(i, s, t, false);
            kernels::set_buffer(&enc, &self.xs[i].buf, 0);
            kernels::set_buffer(&enc, s0, 1);
            kernels::set_buffer(&enc, s1, 2);
            self.set_tmps(&enc, s0, &p);
            kernels::dispatch(&enc, &self.k.block_input, self.pw / s, self.ph / s);
            enc.endEncoding();
        }

        fn encode_final(&self, cb: &MPSCommandBuffer, s0: &Buffer, s1: &Buffer, out: &Frame) {
            let enc = unsafe { cb.commandBuffer() }.computeCommandEncoder().expect("encoder");
            let p = self.params(self.scales.len(), 1, 0.5, false);
            unsafe { enc.setTexture_atIndex(Some(&out.texture), 0) };
            kernels::set_buffer(&enc, s0, 1);
            kernels::set_buffer(&enc, s1, 2);
            self.set_tmps(&enc, s0, &p);
            kernels::set_buffer(&enc, s0, 9);
            kernels::set_bytes(&enc, &((out.kind == PixelKind::Rgba16f) as i32), 10);
            kernels::dispatch(&enc, &self.k.final_out, self.w, self.h);
            enc.endEncoding();
        }

        /// Milliseconds of each stage of prepare and interpolate on the GPU, a whole interpolate the same way,
        /// and the CPU time to encode one. Writes slot s1 (from `frame`) and the network's buffers.
        pub fn profile(&self, frame: &Frame, s0: &Buffer, s1: &Buffer, out: &Frame) -> Vec<(String, f64)> {
            let reps = 20;
            let mut v = Vec::new();
            v.push(("prepare: frame_in".into(), per_run(reps, &|cb| self.encode_frame_in(cb, frame))));
            v.push((
                "prepare: encoder convolutions (MPSGraph)".into(),
                per_run(reps, &|cb| self.enc.encode(cb, &self.xin, &self.enc_a)),
            ));
            v.push(("prepare: encode_out".into(), per_run(reps, &|cb| self.encode_encode_out(cb, s1))));
            for (i, &s) in self.scales.iter().enumerate() {
                v.push((format!("block{i} input (scale {s})"), per_run(reps, &|cb| {
                    self.encode_block_input(cb, i, s0, s1, 0.5)
                })));
                v.push((format!("block{i} convolutions (MPSGraph)"), per_run(reps, &|cb| {
                    self.blocks[i].encode(cb, &self.xs[i], &self.ds[i])
                })));
            }
            v.push(("final (warp, blend, write)".into(), per_run(reps, &|cb| self.encode_final(cb, s0, s1, out))));
            let sum: f64 = v[3..].iter().map(|x| x.1).sum();
            v.push(("sum of the interpolate stages".into(), sum));
            v.push((
                "whole interpolate, back to back".into(),
                per_run(reps, &|cb| self.encode_interpolate(cb, s0, s1, 0.5, out, None)),
            ));
            let cb = unsafe { MPSCommandBuffer::commandBufferFromCommandQueue(&Gpu::shared().queue) };
            let t = Instant::now();
            self.encode_interpolate(&cb, s0, s1, 0.5, out, None);
            v.push(("CPU: encoding one interpolate".into(), t.elapsed().as_secs_f64() * 1e3));
            cb.commit();
            unsafe { cb.commandBuffer() }.waitUntilCompleted();
            v
        }
    }
}
