//! How long each block's convolutions take as MPSGraph executables at 1080p (each run 50 times back to back, so
//! dispatch gaps are amortised), layer by layer and as whole stacks. For choosing how to run the convolutions.

use std::time::Instant;

use objc2_metal::MTLCommandBuffer;
use objc2_metal_performance_shaders::MPSCommandBuffer;
use sp_core::Gpu;
use sp_rife::graph::{Layer, Stack, TensorBuf};
use sp_rife::{net, weights};

fn time(st: &Stack, reps: usize) -> f64 {
    let gpu = Gpu::shared();
    let a = TensorBuf::new(&gpu.device, st.in_shape);
    let b = TensorBuf::new(&gpu.device, st.out_shape);
    let run = |n: usize| {
        let cb = unsafe { MPSCommandBuffer::commandBufferFromCommandQueue(&gpu.queue) };
        for _ in 0..n {
            st.encode(&cb, &a, &b);
        }
        cb.commit();
        unsafe { cb.commandBuffer() }.waitUntilCompleted();
    };
    run(3);
    let t = Instant::now();
    run(reps);
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn main() {
    let scale: f32 = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let wts = weights::load(&weights::default_model_dir().join("flownet_v4.22.lite.pkl")).unwrap();
    let m = net::model("4.22.lite").unwrap();
    let scales = net::block_scales(&m, scale).unwrap();
    let (pw, ph) = net::padded(&m, scale, 1920, 1080);
    let dev = &Gpu::shared().device;
    let mut total = 0.0;
    for (i, &s) in scales.iter().enumerate() {
        let cin = if i == 0 { 16 } else { 32 };
        let layers = net::block_layers(&wts, i, cin, i == scales.len() - 1).unwrap();
        let (h, w) = (ph / s, pw / s);
        let whole = time(&Stack::new(dev, &layers, h, w, cin), 50);
        total += whole;
        println!("block{i} at {w}x{h}: whole stack {whole:.3} ms");
        let (mut ch, mut cw, mut cc) = (h, w, cin);
        let mut sum = 0.0;
        for (j, l) in layers.iter().enumerate() {
            let st = Stack::new(dev, std::slice::from_ref(l), ch, cw, cc);
            let ms = time(&st, 50);
            sum += ms;
            let kind = match l {
                Layer::ConvLrelu(_) => "conv+lrelu",
                Layer::Residual(_) => "residual",
                Layer::Conv(_) => "conv",
                Layer::Deconv(_) => "deconv",
                Layer::Subpixel(_) => "subpixel",
            };
            let [_, oh, ow, oc] = st.out_shape;
            let c = match l { Layer::ConvLrelu(c) | Layer::Residual(c) | Layer::Conv(c) | Layer::Deconv(c) | Layer::Subpixel(c) => c };
            let macs = if matches!(l, Layer::Deconv(_)) { oh * ow * oc * c.cin * 4 } else { oh * ow * oc * c.cin * c.k * c.k };
            println!("   {j:2} {kind:<11} {cw}x{ch}x{cc} -> {ow}x{oh}x{oc}: {ms:.3} ms  {:.1} TFLOPS", 2.0 * macs as f64 / ms / 1e9);
            (ch, cw, cc) = (oh, ow, oc);
        }
        println!("   layers one by one: {sum:.3} ms");
        // the transposed convolution three ways, and the whole stack with each
        let Layer::Deconv(dc) = layers.last().unwrap().clone() else { unreachable!() };
        let (rh, rw, rc) = (h / 4, w / 4, dc.cin);
        let t_sub = time(&Stack::new(dev, &[Layer::Subpixel(dc.clone())], rh, rw, rc), 50);
        let t_c3 = time(&Stack::new(dev, &[Layer::Conv(sp_rife::graph::deconv_as_conv3(&dc))], rh, rw, rc), 50);
        let mut l2 = layers.clone();
        *l2.last_mut().unwrap() = Layer::Subpixel(dc.clone());
        let whole_sub = time(&Stack::new(dev, &l2, h, w, cin), 50);
        *l2.last_mut().unwrap() = Layer::Conv(sp_rife::graph::deconv_as_conv3(&dc));
        let whole_c3 = time(&Stack::new(dev, &l2, h, w, cin), 50);
        println!("   deconv as 4 x 2x2 convs: {t_sub:.3} ms, as one 3x3 conv: {t_c3:.3} ms; whole stack with them: {whole_sub:.3} / {whole_c3:.3} ms");
        // CPU time to encode the stack once
        let st = Stack::new(dev, &layers, h, w, cin);
        let gpu = Gpu::shared();
        let a = TensorBuf::new(&gpu.device, st.in_shape);
        let b = TensorBuf::new(&gpu.device, st.out_shape);
        let mut v = Vec::new();
        for _ in 0..30 {
            let cb = unsafe { MPSCommandBuffer::commandBufferFromCommandQueue(&gpu.queue) };
            let t = Instant::now();
            st.encode(&cb, &a, &b);
            v.push(t.elapsed().as_secs_f64() * 1e3);
            cb.commit();
            unsafe { cb.commandBuffer() }.waitUntilCompleted();
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("   CPU encode of the stack: {:.3} ms (median)", v[15]);
    }
    println!("all blocks: {total:.3} ms");
}
