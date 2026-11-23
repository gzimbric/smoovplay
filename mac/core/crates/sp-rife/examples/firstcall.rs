//! How long building the interpolator and its first calls take (MPSGraph specialises executables on first use).

use std::time::Instant;

use sp_core::{Frame, Interpolator, PixelKind};
use sp_rife::{Options, RifeInterpolator};

fn main() {
    let (w, h) = (1920, 1080);
    let f = Frame::new(w, h, PixelKind::Bgra8);
    for scale in [1.0f32, 0.5] {
        let t = Instant::now();
        let mut it = RifeInterpolator::new(w, h, Options { scale, ..Options::default() }).unwrap();
        let built = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        it.load(&f);
        it.prepare(0);
        it.wait();
        let p1 = t.elapsed().as_secs_f64() * 1e3;
        it.prepare(1);
        let t = Instant::now();
        it.interpolate(1, 0.5);
        let i1 = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        it.interpolate(1, 0.5);
        let i2 = t.elapsed().as_secs_f64() * 1e3;
        println!("detail {scale}: new {built:.0} ms, first prepare {p1:.1} ms, first interpolate {i1:.1} ms, second {i2:.1} ms");
    }
}
