//! Parity with the Python Mac engine (framegen/interpolator.py RifeInterpolator on MPS) on real frames from the
//! test clip: mac/core/tools/rife_fixtures.py writes the frames, the Python outputs and their scores against the
//! real in-between frames into mac/core/fixtures/rife/. Skipped (with a note) when the fixtures or weights are
//! missing.
//!
//! Checks per case: PSNR of the Rust output against the Python output, and that the Rust output scores the same
//! against the real frame as the Python did (within 0.05 dB).

use std::path::{Path, PathBuf};

use half::f16;
use sp_core::{Frame, Interpolator, PixelKind};
use sp_rife::{Options, RifeInterpolator};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/rife")
}

fn psnr(a: &[f32], b: &[f32], peak: f64) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (peak * peak / mse).log10() }
}

/// A packed BGR file as floats (0..255 for .bgr, 0..1 for .bgr16).
fn read_bgr(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if path.extension().unwrap() == "bgr16" {
        b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
    } else {
        b.iter().map(|&v| v as f32).collect()
    }
}

/// A frame from a fixture file: BGRA8 from .bgr, RGBA16F from .bgr16 (framegen's pq16).
fn load_frame(path: &Path, w: usize, h: usize) -> Frame {
    let raw = std::fs::read(path).unwrap();
    if path.extension().unwrap() == "bgr16" {
        let px: Vec<f16> = raw.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])).collect();
        let mut rgba = Vec::with_capacity(w * h * 4);
        for p in px.chunks_exact(3) {
            rgba.extend_from_slice(&[p[2], p[1], p[0], f16::ONE]);
        }
        sp_rife::testing::frame_rgba16f(&rgba, w, h)
    } else {
        Frame::from_bgr(&raw, w, h)
    }
}

/// A frame's pixels as packed BGR floats (0..255 for BGRA8, 0..1 for RGBA16F).
fn frame_bgr(f: &Frame) -> Vec<f32> {
    match f.kind {
        PixelKind::Bgra8 => f.to_bgr().iter().map(|&v| v as f32).collect(),
        PixelKind::Rgba16f => {
            let raw = f.read_raw();
            let px: Vec<f32> = raw.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
            px.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0]]).collect()
        }
    }
}

#[test]
fn matches_the_python_engine_on_real_frames() {
    let dir = fixtures();
    let Ok(index) = std::fs::read_to_string(dir.join("index.json")) else {
        eprintln!("skipped: no fixtures (run mac/core/tools/rife_fixtures.py)");
        return;
    };
    if !sp_rife::weights::default_model_dir().join("flownet_v4.22.lite.pkl").exists() {
        eprintln!("skipped: RIFE 4.22.lite weights not downloaded");
        return;
    }
    let index: serde_json::Value = serde_json::from_str(&index).unwrap();
    let dim = |v: &serde_json::Value, k: &str, d: &serde_json::Value| v[k].as_u64().or(d[k].as_u64()).unwrap() as usize;
    let mut failures = Vec::new();
    println!("{:<28} {:>12} {:>12} {:>12} {:>8}", "case", "vs Python", "Rust/real", "Python/real", "delta");
    for case in index["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (w, h) = (dim(case, "width", &index), dim(case, "height", &index));
        let hdr = case["hdr"].as_bool().unwrap();
        let opts = Options {
            version: case["version"].as_str().unwrap_or("4.22.lite").to_string(),
            scale: case["scale"].as_f64().unwrap() as f32,
            hdr,
            model_dir: case["model_dir"].as_str().map(|d| dir.join(d)),
            ..Options::default()
        };
        let mut it = RifeInterpolator::new(w, h, opts).unwrap();
        it.set_steady(case["steady"].as_f64().unwrap() as f32);
        for (slot, key) in [(0, "a"), (1, "b")] {
            it.load(&load_frame(&dir.join(case[key].as_str().unwrap()), w, h));
            it.prepare(slot);
        }
        let got = frame_bgr(&it.interpolate(1, case["t"].as_f64().unwrap() as f32));
        let python = read_bgr(&dir.join(case["python"].as_str().unwrap()));
        let peak = if hdr { 1.0 } else { 255.0 };
        let vs_python = psnr(&got, &python, peak);
        let (mut vs_truth, mut py_truth, mut delta) = (f64::NAN, f64::NAN, 0.0);
        if let Some(truth) = case["truth"].as_str() {
            vs_truth = psnr(&got, &read_bgr(&dir.join(truth)), peak);
            py_truth = case["psnr_python_vs_truth"].as_f64().unwrap();
            delta = vs_truth - py_truth;
        }
        println!("{name:<28} {vs_python:>9.2} dB {vs_truth:>9.3} dB {py_truth:>9.3} dB {delta:>+8.4}");
        if vs_python < 50.0 || delta.abs() > 0.05 {
            failures.push(format!("{name}: {vs_python:.2} dB vs Python, {delta:+.4} dB vs real"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
