//! Does MPSGraph run the blocks' first convolution as fast on 28 input channels (exactly what RIFE uses) as on 32
//! (metalrife.py's padding)? Times block 1..3's whole stacks both ways at 1080p.

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
    let wts = weights::load(&weights::default_model_dir().join("flownet_v4.22.lite.pkl")).unwrap();
    let dev = &Gpu::shared().device;
    for scale in [1.0f32, 0.5] {
        let m = net::model("4.22.lite").unwrap();
        let scales = net::block_scales(&m, scale).unwrap();
        let (pw, ph) = net::padded(&m, scale, 1920, 1080);
        for (i, &s) in scales.iter().enumerate().skip(1) {
            let (h, w) = (ph / s, pw / s);
            let mut r = Vec::new();
            for cin in [32usize, 28] {
                let mut layers = net::block_layers(&wts, i, cin, i == 3).unwrap();
                let Some(Layer::Deconv(dc)) = layers.pop() else { unreachable!() };
                layers.push(Layer::Conv(sp_rife::graph::deconv_as_conv3(&dc)));
                let first = Stack::new(dev, &layers[..1], h, w, cin);
                r.push((time(&Stack::new(dev, &layers, h, w, cin), 50), time(&first, 50)));
            }
            println!("detail {scale} block{i} {w}x{h}: stack 32 ch {:.3} ms, 28 ch {:.3} ms; first conv {:.3} / {:.3}",
                     r[0].0, r[1].0, r[0].1, r[1].1);
        }
    }
}
