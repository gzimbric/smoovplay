//! Shared by sp-capture's tests: the reference data mac/core/tools/capture_fixtures.py makes, Python's random
//! numbers, and pictures to stand in for a card's.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use sp_media::pyrandom::PyRandom;

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/capture")
}

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf()
}

/// fixtures/capture/<name>, or None (and a note) when the fixtures haven't been made.
pub fn fixture(name: &str) -> Option<PathBuf> {
    let p = fixtures().join(name);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("(skipped: {name} missing; run .venv/bin/python mac/core/tools/capture_fixtures.py)");
        None
    }
}

pub fn manifest() -> Option<serde_json::Value> {
    let p = fixture("manifest.json")?;
    serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

/// Python's random.Random(seed) with gauss().
pub struct PyRng {
    r: PyRandom,
    next: Option<f64>,
}

impl PyRng {
    pub fn new(seed: u64) -> PyRng {
        PyRng { r: PyRandom::new(seed), next: None }
    }
    pub fn random(&mut self) -> f64 {
        self.r.random()
    }
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let z = match self.next.take() {
            Some(z) => z,
            None => {
                let x2pi = self.r.random() * std::f64::consts::TAU;
                let g2rad = (-2.0 * (1.0 - self.r.random()).ln()).sqrt();
                self.next = Some(x2pi.sin() * g2rad);
                x2pi.cos() * g2rad
            }
        };
        mu + z * sigma
    }
}

/// tests/test_audio.py _card. A card's 20 ms packets as the reader thread hands them over, and the output's 10 ms
/// blocks. The reader is held up for hold_ms every every_s, and the packets meanwhile all come at once at the end.
/// Returns (time, true for a packet in or false for a block out), in order.
pub fn card(secs: f64, hold_ms: f64, every_s: f64, seed: u64, first_at: Option<f64>) -> Vec<(f64, bool)> {
    let mut rnd = PyRng::new(seed);
    let (mut ev, mut held, mut nxt) = (Vec::new(), 0usize, first_at.unwrap_or(every_s));
    for i in 0..(secs / 0.02) as usize {
        let t = i as f64 * 0.02 + rnd.gauss(0.0, 0.002).abs();
        if hold_ms > 0.0 && nxt <= t && t < nxt + hold_ms / 1000.0 {
            held += 1;
            continue;
        }
        if held > 0 {
            ev.extend(std::iter::repeat_n((nxt + hold_ms / 1000.0, true), held));
            held = 0;
            nxt += every_s;
        }
        ev.push((t, true));
    }
    ev.extend((0..(secs / 0.01) as usize).map(|i| (i as f64 * 0.01, false)));
    ev.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ev
}

/// An image of flat 16x16 colour blocks (test_capture.py _blocks). Chroma differences stay at block edges, so
/// pictures are compared at block centres. BGR, h x w.
pub fn blocks(h: usize, w: usize, seed: u64) -> Vec<u8> {
    let mut rnd = PyRng::new(seed);
    let (bh, bw) = (h / 16, w / 16);
    let small: Vec<u8> = (0..bh * bw * 3).map(|_| 20 + (rnd.random() * 216.0) as u8).collect();
    let mut out = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let s = ((y / 16) * bw + x / 16) * 3;
            out[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&small[s..s + 3]);
        }
    }
    out
}

/// The block centres' values.
pub fn centres(bgr: &[u8], h: usize, w: usize) -> Vec<i32> {
    let mut out = Vec::new();
    for y in (8..h).step_by(16) {
        for x in (8..w).step_by(16) {
            out.extend(bgr[(y * w + x) * 3..(y * w + x) * 3 + 3].iter().map(|&v| v as i32));
        }
    }
    out
}

pub fn max_diff(a: &[i32], b: &[i32]) -> i32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).max().unwrap_or(0)
}

/// BGR to Y'CbCr in BT.601 limited range, as OpenCV's BGR2YUV_I420 makes it. Y, U and V are full size.
fn ycc(bgr: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = bgr.len() / 3;
    let (mut y, mut u, mut v) = (vec![0; n], vec![0; n], vec![0; n]);
    for i in 0..n {
        let (b, g, r) = (bgr[i * 3] as f64, bgr[i * 3 + 1] as f64, bgr[i * 3 + 2] as f64);
        let l = 0.299 * r + 0.587 * g + 0.114 * b;
        y[i] = (16.0 + l * 219.0 / 255.0).round() as u8;
        u[i] = (128.0 + (b - l) / 1.772 * 224.0 / 255.0).round() as u8;
        v[i] = (128.0 + (r - l) / 1.402 * 224.0 / 255.0).round() as u8;
    }
    (y, u, v)
}

/// A BGR picture as a card's NV12.
pub fn to_nv12(bgr: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (y, u, v) = ycc(bgr);
    let mut out = y;
    for r in 0..h / 2 {
        for c in 0..w / 2 {
            out.push(u[2 * r * w + 2 * c]);
            out.push(v[2 * r * w + 2 * c]);
        }
    }
    out
}

/// A BGR picture as a card's YUY2 (Y0 U Y1 V).
pub fn to_yuy2(bgr: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (y, u, v) = ycc(bgr);
    let mut out = Vec::with_capacity(h * w * 2);
    for r in 0..h {
        for c in 0..w / 2 {
            let i = r * w + 2 * c;
            out.extend([y[i], u[i], y[i + 1], v[i]]);
        }
    }
    out
}
