//! What the probe costs per call at 1080p: cargo run --release -p sp-sched --example probe_cost
use std::time::Instant;

use sp_core::Frame;
use sp_sched::{GpuProbe, Probe};

fn median(mut f: impl FnMut(), n: usize) -> f64 {
    f();
    let mut ms: Vec<f64> = (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms[n / 2]
}

fn main() {
    let (w, h) = (1920, 1080);
    let bgr: Vec<u8> = (0..w * h * 3).map(|i| ((i * 2654435761usize) >> 13) as u8).collect();
    let f = Frame::from_bgr(&bgr, w, h);
    let mut p = GpuProbe::new();
    println!("block means (sp-sched, exact):  {:.3} ms", median(|| drop(p.block_means(&f)), 200));
    println!("block means (sp-core):          {:.3} ms", median(|| drop(sp_core::stats::block_means(&f)), 200));
    println!("delay test blocks (small):      {:.3} ms", median(|| drop(p.small(&f)), 200));
    println!("score (four frames):            {:.3} ms", median(|| { std::hint::black_box(p.score(&f, &f, &f, &f)); }, 100));
}
