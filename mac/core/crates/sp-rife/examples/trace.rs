//! A steady loop of interpolate() calls for Instruments (Metal System Trace):
//!
//!     xcrun xctrace record --template 'Metal System Trace' --launch -- \
//!         target/release/examples/trace [scale] [calls]

use sp_core::{Frame, Interpolator};
use sp_rife::{Options, RifeInterpolator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scale: f32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(200);
    let (w, h) = (1920, 1080);
    let bgr: Vec<u8> = (0..w * h * 3).map(|i| ((i * 2654435761usize) >> 13) as u8).collect();
    let f = Frame::from_bgr(&bgr, w, h);
    let mut it = RifeInterpolator::new(w, h, Options { scale, ..Options::default() }).unwrap();
    for s in 0..2 {
        it.load(&f);
        it.prepare(s);
    }
    for i in 0..n {
        it.interpolate(i % 2, 0.5);
    }
}
