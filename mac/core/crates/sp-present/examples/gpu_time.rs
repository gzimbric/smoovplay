//! Measures the GPU time of the window's drawing, without a window. It runs FSR 1 (and bilinear) from 1920x1080 to a
//! 6016x3384 target, the size of this Mac's main display, then looks on top of it. The looks are the heaviest
//! built-in ones and one with every node type.
//!
//!     cargo run --release -p sp-present --example gpu_time [-- 6016x3384 [frames]]
//!
//! Each case draws `frames` frames (default 240) with one command buffer each, paced at 120 fps as the window draws
//! them. It reports GPUEndTime - GPUStartTime per frame, with each frame alone on the GPU, because frames committed
//! back to back queue behind each other and their start and end times overlap. It also reports the throughput of
//! the same frames back to back.

use std::sync::{Arc, Mutex};
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLPixelFormat};
use sp_core::{Frame, Gpu, PixelKind};
use sp_present::fill::Fill;
use sp_present::gpu;
use sp_present::looks::{self, Look, LutStore, Node};
use sp_present::presenter::Spread;
use sp_present::renderer::{Picture, Renderer};
use sp_present::Scaler;

fn every_node() -> Look {
    let n = |t: &str| Node::new(t).unwrap();
    Look {
        name: "Every node".into(),
        nodes: vec![n("color").with("contrast", 0.3), n("wheels").with("lift", 0.1), n("split").with("shadow_amount", 0.4),
                    n("bloom"), n("halation"), n("blur").with("mix", 0.3), n("clarity"), n("ca"), n("sharpen"),
                    n("grain"), n("vignette"), n("scanlines"), n("letterbox").with("aspect", 2.0),
                    n("assist"), n("mono").with("mix", 0.2)],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (tw, th) = args.first().and_then(|s| s.split_once('x')).map_or((6016, 3384), |(a, b)| (a.parse().unwrap(), b.parse().unwrap()));
    let frames: usize = args.get(1).map_or(240, |s| s.parse().unwrap());
    let (sw, sh) = (1920, 1080);
    let bgr: Vec<u8> = (0..sw * sh * 3).map(|i| ((i * 2654435761usize) >> 13) as u8).collect();
    let src = Frame::from_bgr(&bgr, sw, sh);
    let hdr_src = Frame::new(sw, sh, PixelKind::Rgba16f);
    let target = gpu::texture(tw, th, MTLPixelFormat::BGRA8Unorm);
    let target16 = gpu::texture(tw, th, MTLPixelFormat::RGBA16Float);
    let mut r = Renderer::new(Arc::new(LutStore::default())).unwrap();
    println!("{} drawing 1920x1080 frames into {tw}x{th}, {frames} frames per case:", Gpu::shared().name());
    let find = |id: &str| looks::builtin().iter().find(|(k, _)| *k == id).unwrap().1.clone();
    let cases: Vec<(&str, Scaler, Option<Look>, bool)> = vec![
        ("FSR 1", Scaler::Fsr, None, false),
        ("bilinear", Scaler::Bilinear, None, false),
        ("FSR 1, HDR10 (half floats)", Scaler::Fsr, None, true),
        ("FSR 1 + Cinematic (bloom)", Scaler::Fsr, Some(find("cinematic")), false),
        ("FSR 1 + Dreamy (blur + bloom)", Scaler::Fsr, Some(find("dreamy")), false),
        ("FSR 1 + Retro VHS", Scaler::Fsr, Some(find("vhs")), false),
        ("FSR 1 + every node type (15 nodes, 7 passes)", Scaler::Fsr, Some(every_node()), false),
    ];
    for (name, scaler, look, hdr) in cases {
        let look = look.map(Arc::new);
        let plan = look.as_ref().and_then(|l| r.looks.plan(l, &LutStore::default()));
        let times = Arc::new(Mutex::new(Vec::new()));
        let mut last = None;
        let mut next = sp_core::clock::now();
        for k in 0..frames + 20 {
            next += 1.0 / 120.0;
            autoreleasepool(|_| {
                let cmd = Gpu::shared().queue.commandBuffer().unwrap();
                let pic = Picture { src: if hdr { &hdr_src.texture } else { &src.texture }, src_size: (sw, sh), scaler,
                                    sharpness: 0.5, fill: Fill::Fit, look: plan.as_deref(), frame_no: k as u64, hdr, plain: false };
                r.draw(&cmd, &pic, if hdr { &target16 } else { &target }).unwrap();
                if k >= 20 {
                    let t = times.clone();
                    let done = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        let cb = unsafe { cb.as_ref() };
                        t.lock().unwrap().push((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3);
                    });
                    unsafe { cmd.addCompletedHandler(RcBlock::as_ptr(&done)) };
                }
                cmd.commit();
                cmd.waitUntilCompleted();
                last = Some(cmd);
            });
            let left = next - sp_core::clock::now();
            if left > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(left));
            }
        }
        last.unwrap().waitUntilCompleted();
        // throughput, with the same frames back to back
        let t0 = sp_core::clock::now();
        let mut lastb = None;
        for k in 0..frames {
            autoreleasepool(|_| {
                let cmd = Gpu::shared().queue.commandBuffer().unwrap();
                let pic = Picture { src: if hdr { &hdr_src.texture } else { &src.texture }, src_size: (sw, sh), scaler,
                                    sharpness: 0.5, fill: Fill::Fit, look: plan.as_deref(), frame_no: k as u64, hdr, plain: false };
                r.draw(&cmd, &pic, if hdr { &target16 } else { &target }).unwrap();
                cmd.commit();
                lastb = Some(cmd);
            });
        }
        lastb.unwrap().waitUntilCompleted();
        let per = (sp_core::clock::now() - t0) * 1e3 / frames as f64;
        std::thread::sleep(std::time::Duration::from_millis(50));
        let passes = plan.as_ref().map_or(String::new(), |p| format!(" [{} look passes]", p.passes()));
        println!("  {name}{passes}: {}; back to back {per:.2} ms a frame", Spread::of(&times.lock().unwrap()));
    }
}
