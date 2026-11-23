//! tests/test_hud_score.py, ported, and the GPU probe checked against the CPU reference for the scoring sums and the
//! block means (exact, as torch's avg_pool2d on uint8). It also measures what each costs at 1080p
//! (cargo test --release --test score -- --nocapture).

use std::time::Instant;

use sp_core::{Frame, PixelKind};
use sp_sched::score::{ScoreSums, add_scores, hud_mask, score_sums_bgr};
use sp_sched::{CpuProbe, GpuProbe, Probe, Stats};

/// A, the real in-between frame and B: a textured background scrolling 4 px a frame, a flat grey band that doesn't
/// move, and a white box with black text-like bars on top that never moves (the HUD).
fn frames() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>, usize, usize) {
    let (h, w) = (64usize, 96usize);
    let texture: Vec<f32> = (0..w + 16).map(|x| (128.0 + 100.0 * (x as f32 / 2.0).sin()).clamp(0.0, 255.0)).collect();
    let frame = |shift: usize| -> Vec<[f32; 3]> {
        let mut img = vec![[0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut v = texture[shift + x];
                if y >= 40 {
                    v = 90.0; // flat and still, so no edges
                }
                if (8..24).contains(&y) && (8..40).contains(&x) {
                    v = 255.0; // the label
                }
                if (12..20).contains(&y) && [(12..16), (20..24), (28..32)].iter().any(|r| r.contains(&x)) {
                    v = 0.0; // with text on it
                }
                img[y * w + x] = [v; 3];
            }
        }
        img
    };
    (frame(0), frame(4), frame(8), w, h)
}

fn sum(m: &[bool], w: usize, ys: std::ops::Range<usize>, xs: std::ops::Range<usize>) -> usize {
    ys.flat_map(|y| xs.clone().map(move |x| (y, x))).filter(|&(y, x)| m[y * w + x]).count()
}

#[test]
fn test_hud_mask_finds_still_edges_only() {
    let (a, gt, b, w, h) = frames();
    let m = hud_mask(&a, &b, &gt, w, h);
    assert_eq!(m.len(), 64 * 96);
    assert!(sum(&m, w, 12..20, 10..34) > 20); // the text's edges on the label
    assert_eq!(sum(&m, w, 44..60, 0..w), 0); // the flat still band has no edges
    assert_eq!(sum(&m, w, 28..36, 0..w), 0); // the scrolling background isn't still
}

#[test]
fn test_hud_mask_needs_the_in_between_frame_still_too() {
    let (a, mut gt, b, w, h) = frames();
    for y in 8..24 {
        for x in 8..40 {
            gt[y * w + x] = [40.0; 3]; // something crosses the label in the in-between frame only
        }
    }
    assert_eq!(sum(&hud_mask(&a, &b, &gt, w, h), w, 8..24, 8..40), 0);
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Four 8-bit pictures that score like a real one: b moves on from a, gt is in between with some error, out a bit
/// off gt, and a still label with text that all four share.
fn scene(w: usize, h: usize, seed: u64) -> [Vec<u8>; 4] {
    let mut g = Rng(seed | 1);
    let px = |x: usize, y: usize, s: usize| -> [u8; 3] {
        if (40..100).contains(&y) && (40..400).contains(&x) {
            return if (60..80).contains(&y) && x % 40 < 12 { [10, 10, 10] } else { [240, 240, 240] };
        }
        let v = ((x + 3 * s) * 7 + y * 3) % 200 + 20;
        [v as u8, (v + 15) as u8, (v + 30) as u8]
    };
    let mut out: [Vec<u8>; 4] = Default::default();
    for (k, s) in [(0, 0), (1, 2), (2, 1), (3, 1)] {
        let mut img = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let p = px(x, y, s);
                let noise = if k == 2 { (g.next() % 33) as i32 - 16 } else { 0 };
                img.extend(p.map(|c| (c as i32 + noise).clamp(0, 255) as u8));
            }
        }
        out[k] = img;
    }
    out // a, b, out, gt
}

fn close(a: &ScoreSums, b: &ScoreSums) -> bool {
    let v = |s: &ScoreSums| {
        [s.se_out, s.se_rep, s.se_blend, s.n_still, s.se_out_still, s.se_rep_still, s.n_off, s.n_off_rep, s.n_hud, s.se_out_hud, s.n_off_hud, s.pixels]
    };
    v(a) == v(b)
}

#[test]
fn gpu_scoring_matches_the_cpu_reference_exactly() {
    for (w, h) in [(1920, 1080), (130, 70)] {
        let [a, b, o, g] = scene(w, h, 7);
        let fr = |x: &Vec<u8>| Frame::from_bgr(x, w, h);
        let gpu = GpuProbe::new().score(&fr(&a), &fr(&b), &fr(&o), &fr(&g));
        let cpu = score_sums_bgr(&a, &b, &o, &g, w, h);
        assert!(close(&gpu, &cpu), "{w}x{h}\n gpu {gpu:?}\n cpu {cpu:?}");
        assert!(cpu.n_hud > 0.0 && cpu.n_still > 0.0 && cpu.n_off > 0.0); // every measure is exercised
        let mut st = Stats::default();
        add_scores(&mut st, &gpu);
        assert!(st.psnr_interp[0].is_finite() && !st.psnr_hud.is_empty() && st.off_interp[0] > 0.0);
    }
}

#[test]
fn gpu_block_means_are_exact() {
    // torch's avg_pool2d on uint8 sums integers and divides once. sp_core::stats::block_means adds normalised floats.
    let (w, h) = (1920, 1080);
    let mut g = Rng(42);
    let bgr: Vec<u8> = (0..w * h * 3).map(|_| (g.next() >> 24) as u8).collect();
    let f = Frame::from_bgr(&bgr, w, h);
    let ours = GpuProbe::new().block_means(&f);
    let mut exact = ours.clone();
    for by in 0..ours.rows {
        for bx in 0..ours.cols {
            let (x0, y0) = (bx * 16, by * 16);
            let (x1, y1) = ((x0 + 16).min(w), (y0 + 16).min(h));
            for c in 0..3 {
                let s: u32 = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (y, x))).map(|(y, x)| bgr[(y * w + x) * 3 + c] as u32).sum();
                exact.data[(by * ours.cols + bx) * 3 + c] = s as f32 / ((x1 - x0) * (y1 - y0)) as f32;
            }
        }
    }
    assert_eq!(ours, exact);
    let core = sp_core::stats::block_means(&f);
    let worst = core.data.iter().zip(&exact.data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("sp_core::stats::block_means is up to {worst:e} off the exact means");
}

#[test]
fn hdr_frames_are_scored_in_8_bit_levels() {
    // A PQ frame at SDR white (203 nits) reads as level 255, as hdr.py's level8 has it.
    let pq = sp_sched::score::pq_of_nits(203.0);
    let f = Frame::new(4, 4, PixelKind::Rgba16f);
    let half = |v: f32| -> [u8; 2] {
        // f32 to f16 by hand, for values in 0 to 1 (normal range)
        let b = v.to_bits();
        let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let m = (b >> 13) & 0x3ff;
        (((e as u32) << 10) | m).to_le_bytes()[..2].try_into().unwrap()
    };
    let mut raw = Vec::new();
    for _ in 0..16 {
        for v in [pq as f32, pq as f32, pq as f32, 1.0] {
            raw.extend(half(v));
        }
    }
    unsafe {
        use objc2_metal::MTLTexture;
        f.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            objc2_metal::MTLRegion {
                origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: objc2_metal::MTLSize { width: 4, height: 4, depth: 1 },
            },
            0,
            std::ptr::NonNull::new(raw.as_ptr() as *mut _).unwrap(),
            4 * 8,
        );
    }
    let m = GpuProbe::new().block_means(&f);
    assert!(m.data.iter().all(|&v| (v - 254.0).abs() <= 1.0), "{:?}", &m.data[..3]); // truncated, as to_u8
}

/// Prints what scoring costs at 1080p, both ways. Run it in release for the real figure.
#[test]
fn scoring_cost_at_1080p() {
    let (w, h) = (1920, 1080);
    let [a, b, o, g] = scene(w, h, 3);
    let fr = |x: &Vec<u8>| Frame::from_bgr(x, w, h);
    let (fa, fb, fo, fg) = (fr(&a), fr(&b), fr(&o), fr(&g));
    let time = |probe: &mut dyn Probe, n: usize| -> f64 {
        probe.score(&fa, &fb, &fo, &fg);
        let mut ms: Vec<f64> = (0..n)
            .map(|_| {
                let t = Instant::now();
                std::hint::black_box(probe.score(&fa, &fb, &fo, &fg));
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
        ms[n / 2]
    };
    let gpu = time(&mut GpuProbe::new(), 30);
    let cpu = time(&mut CpuProbe, 5);
    let t = Instant::now();
    for _ in 0..30 {
        GpuProbe::new().block_means(&fa);
    }
    let means = t.elapsed().as_secs_f64() * 1000.0 / 30.0;
    eprintln!("1080p scoring: GPU {gpu:.2} ms, CPU (to_bgr) {cpu:.1} ms; block means on the GPU {means:.2} ms (median, {})",
        if cfg!(debug_assertions) { "debug build" } else { "release build" });
    assert!(gpu < cpu);
}
