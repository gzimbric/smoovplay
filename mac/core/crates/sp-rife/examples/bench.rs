//! Timing of the Mac's RIFE engine at 1080p (RIFE 4.22.lite, fp16): load + prepare, one in-between frame, and a
//! whole 60 -> 90 Hz tick (a real frame prepared and two in-between frames), each call waited for (the GPU's work
//! finished), at full and half motion detail.
//!
//!     cargo run --release -p sp-rife --example bench [-- --n 300 --size 1920x1080 --model 4.22.lite]
//!
//! Frames are the parity fixtures' real frames if mac/core/fixtures/rife exists, else a moving pattern.

use std::path::Path;
use std::time::Instant;

use sp_core::{Frame, Gpu, Interpolator};
use sp_rife::{Options, RifeInterpolator};

fn pattern(w: usize, h: usize, shift: usize) -> Frame {
    let mut bgr = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let xf = (x + shift) as f32;
            let p = (y * w + x) * 3;
            bgr[p] = ((xf / 9.0).sin() * 70.0 + (y as f32 / 7.0).cos() * 50.0 + 128.0) as u8;
            bgr[p + 1] = ((xf / 13.0).cos() * 80.0 + 128.0) as u8;
            bgr[p + 2] = (((xf + y as f32) / 11.0).sin() * 90.0 + 128.0) as u8;
        }
    }
    Frame::from_bgr(&bgr, w, h)
}

fn frames(w: usize, h: usize) -> Vec<Frame> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/rife");
    let files = ["f0120.bgr", "f0122.bgr", "f0210.bgr", "f0213.bgr"];
    if (w, h) == (1920, 1080) && files.iter().all(|f| dir.join(f).exists()) {
        return files.iter().map(|f| Frame::from_bgr(&std::fs::read(dir.join(f)).unwrap(), w, h)).collect();
    }
    (0..4).map(|i| pattern(w, h, i * 3)).collect()
}

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    (q(0.5), q(0.95))
}

fn time<F: FnMut(usize)>(n: usize, mut f: F) -> (f64, f64) {
    for i in 0..20 {
        f(i);
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let t = Instant::now();
        f(i);
        v.push(t.elapsed().as_secs_f64() * 1e3);
    }
    stats(v)
}

fn main() {
    let mut n = 200usize;
    let (mut w, mut h) = (1920usize, 1080usize);
    let mut model = "4.22.lite".to_string();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => n = args[i + 1].parse().unwrap(),
            "--size" => {
                let (a, b) = args[i + 1].split_once('x').unwrap();
                (w, h) = (a.parse().unwrap(), b.parse().unwrap());
            }
            "--model" => model = args[i + 1].clone(),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    println!("{} | {w}x{h} | RIFE {model} fp16 | {n} calls each, every call waited for", Gpu::shared().name());
    let fr = frames(w, h);
    println!("{:<34} {:>16} {:>16}", "ms (p50 / p95)", "motion detail 1", "motion detail 0.5");
    let mut rows: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    let names = [
        "load + prepare",
        "interpolate (t = 0.5)",
        "interpolate, Steady HUD on",
        "tick: prepare + 1 frame (60 Hz)",
        "tick: prepare + 2 frames (90 Hz)",
    ];
    for name in names {
        rows.push((name.to_string(), Vec::new()));
    }
    for scale in [1.0f32, 0.5] {
        let t0 = Instant::now();
        let mut it = RifeInterpolator::new(w, h, Options { version: model.clone(), scale, ..Options::default() })
            .expect("RIFE");
        let build = t0.elapsed().as_secs_f64();
        eprintln!("built motion detail {scale} in {build:.2} s");
        for (s, f) in fr.iter().take(2).enumerate() {
            it.load(f);
            it.prepare(s);
        }
        it.wait();
        rows[0].1.push(time(n, |i| {
            it.load(&fr[i % 4]);
            it.prepare(i % 2);
            it.wait();
        }));
        rows[1].1.push(time(n, |i| {
            it.interpolate(i % 2, 0.5);
        }));
        it.set_steady(0.03);
        rows[2].1.push(time(n, |i| {
            it.interpolate(i % 2, 0.5);
        }));
        it.set_steady(0.0);
        let mut prof: Vec<(String, f64)> = Vec::new();
        for _ in 0..5 {
            for (k, (name, ms)) in it.profile().into_iter().enumerate() {
                if prof.len() <= k { prof.push((name, 0.0)); }
                prof[k].1 += ms / 5.0;
            }
        }
        println!("-- motion detail {scale}: where the time goes (ms per run, each stage run 20 times back to back)");
        for (name, ms) in &prof {
            println!("   {name:<44} {ms:>6.3}");
        }
        rows[3].1.push(time(n, |i| {
            it.load(&fr[i % 4]);
            it.prepare(i % 2);
            it.interpolate(i % 2, 0.5);
        }));
        rows[4].1.push(time(n, |i| {
            it.load(&fr[i % 4]);
            it.prepare(i % 2);
            it.interpolate(i % 2, 1.0 / 3.0);
            it.interpolate(i % 2, 2.0 / 3.0);
        }));
    }
    for (name, v) in rows {
        let cell = |(a, b): (f64, f64)| format!("{a:.2} / {b:.2}");
        println!("{name:<34} {:>16} {:>16}", cell(v[0]), cell(v[1]));
    }
}
