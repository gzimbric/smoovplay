//! The Metal passes held to the Python window's: FSR 1 and bilinear against what its OpenGL versions draw
//! (tools/present_fsr.py), every look against framegen/looks.py's NumPy render() (tools/present_looks.py, the cases
//! and tolerances of tests/test_looks.py), and Panorama against fill.py's curve (tests/test_fill.py's check).
//!
//! The reference data is generated, not checked in: run the two scripts first (from the repository root,
//! `.venv/bin/python mac/core/tools/present_fsr.py` and `present_looks.py`). Without it these tests say so and pass.

use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture,
};
use serde_json::Value;
use sp_core::{Frame, Gpu};
use sp_present::fill::{self, Fill, Rect};
use sp_present::gpu::{self, Load, Tex};
use sp_present::looks::{self, LutStore};
use sp_present::renderer::{Picture, Renderer};
use sp_present::Scaler;

fn fixtures(sub: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/present").join(sub);
    if p.join("cases.json").exists() {
        Some(p)
    } else {
        eprintln!("no reference data in {} (run mac/core/tools/present_{sub}.py): skipped", p.display());
        None
    }
}

fn read_bgr(t: &Tex) -> Vec<u8> {
    let (w, h) = (t.width(), t.height());
    let mut raw = vec![0u8; w * h * 4];
    unsafe {
        t.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            NonNull::new(raw.as_mut_ptr() as *mut _).unwrap(),
            w * 4,
            MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: w, height: h, depth: 1 } },
            0,
        );
    }
    raw.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
}

fn target(w: usize, h: usize) -> objc2::rc::Retained<Tex> {
    gpu::texture_with(w, h, MTLPixelFormat::BGRA8Unorm, MTLStorageMode::Shared)
}

fn run(f: impl FnOnce(&gpu::Cmd)) {
    let cmd = Gpu::shared().queue.commandBuffer().unwrap();
    f(&cmd);
    cmd.commit();
    cmd.waitUntilCompleted();
}

/// (mean, share above `over`, max) of |a - b|.
fn diff(a: &[u8], b: &[u8], over: i32) -> (f64, f64, i32) {
    assert_eq!(a.len(), b.len());
    let (mut sum, mut n_over, mut max) = (0i64, 0usize, 0i32);
    for (x, y) in a.iter().zip(b) {
        let d = (*x as i32 - *y as i32).abs();
        sum += d as i64;
        n_over += (d > over) as usize;
        max = max.max(d);
    }
    (sum as f64 / a.len() as f64, n_over as f64 / a.len() as f64, max)
}

#[test]
fn fsr_and_bilinear_draw_what_the_opengl_window_does() {
    let Some(dir) = fixtures("fsr") else { return };
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let mut r = Renderer::new(Arc::new(LutStore::default())).unwrap();
    for c in cases["cases"].as_array().unwrap() {
        let (w, h) = (c["w"].as_u64().unwrap() as usize, c["h"].as_u64().unwrap() as usize);
        let (dw, dh) = (c["dw"].as_u64().unwrap() as usize, c["dh"].as_u64().unwrap() as usize);
        let src = Frame::from_bgr(&std::fs::read(dir.join(c["input"].as_str().unwrap())).unwrap(), w, h);
        let want = std::fs::read(dir.join(c["file"].as_str().unwrap())).unwrap();
        let out = target(dw, dh);
        let what = c["what"].as_str().unwrap();
        run(|cmd| {
            let rect = Rect::new(0, 0, dw as i64, dh as i64);
            if what == "fsr" {
                r.fsr(cmd, &src.texture, w, h, &out, rect, Load::Clear, c["sharpness"].as_f64().unwrap() as f32, false).unwrap();
            } else {
                r.bilinear(cmd, &src.texture, &out, rect, Load::Clear).unwrap();
            }
        });
        let (mean, over, max) = diff(&read_bgr(&out), &want, 1);
        eprintln!("{}: mean {mean:.4}, >1 level {:.4}%, max {max}", c["file"].as_str().unwrap(), over * 100.0);
        // The same maths on the same GPU (Apple's OpenGL runs on Metal): within a level, almost everywhere.
        assert!(mean < 0.05 && over < 0.001 && max <= 3, "{}", c["file"]);
    }
}

#[test]
fn every_look_draws_what_the_reference_does() {
    let Some(dir) = fixtures("looks") else { return };
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let (w, h) = (cases["width"].as_u64().unwrap() as usize, cases["height"].as_u64().unwrap() as usize);
    let src = Frame::from_bgr(&std::fs::read(dir.join("input.bgr")).unwrap(), w, h);
    let luts = Arc::new(LutStore::new(Some(dir.join("luts"))));
    let mut r = Renderer::new(luts.clone()).unwrap();
    let mut worst = (0.0f64, 0.0f64, 0i32);
    for c in cases["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let look = Arc::new(looks::normalize(&c["look"]).unwrap());
        let want = std::fs::read(dir.join(c["file"].as_str().unwrap())).unwrap();
        let got = match r.looks.plan(&look, &luts) {
            None => {
                assert_eq!(c["passes"].as_u64(), Some(0), "{name}: a look that does something drew nothing");
                src.to_bgr()
            }
            Some(plan) => {
                assert_eq!(plan.passes() as u64, c["passes"].as_u64().unwrap(), "{name}");
                let out = target(w, h);
                run(|cmd| r.bake(cmd, &plan, &src.texture, &out, c["frame"].as_u64().unwrap()).unwrap());
                read_bgr(&out)
            }
        };
        let (mean, over, max) = diff(&got, &want, 2);
        worst = (worst.0.max(mean), worst.1.max(over), worst.2.max(max));
        // tests/test_looks.py's tolerances: half floats between passes, the GPU's filtering and maths.
        assert!(mean < 0.6 && over < 0.002 && max <= 12, "{name}: mean {mean:.3}, >2 {:.4}%, max {max}", over * 100.0);
    }
    eprintln!("{} looks; worst: mean {:.3}, >2 levels {:.4}%, max {}", cases["cases"].as_array().unwrap().len(),
              worst.0, worst.1 * 100.0, worst.2);
    assert!(r.looks.broken.is_none());
}

#[test]
fn the_panorama_pass_bends_the_picture_as_the_curve_says() {
    // tests/test_fill.py: a 21:9 screen, a red ramp rising left to right, green on the top half
    let (fw, fh) = (860usize, 360usize);
    let (sw, sh) = (1920usize, 1080usize);
    let mut bgr = vec![0u8; sw * sh * 3];
    for y in 0..sh {
        for x in 0..sw {
            let i = (y * sw + x) * 3;
            bgr[i + 2] = (x as f64 / (sw - 1) as f64 * 255.0).round() as u8;
            bgr[i + 1] = if y < sh / 2 { 200 } else { 0 };
        }
    }
    let src = Frame::from_bgr(&bgr, sw, sh);
    let mut r = Renderer::new(Arc::new(LutStore::default())).unwrap();
    let out = target(fw, fh);
    run(|cmd| {
        let pic = Picture { src: &src.texture, src_size: (sw, sh), scaler: Scaler::Bilinear, sharpness: 0.5,
                            fill: Fill::Panorama, look: None, frame_no: 0, hdr: false, plain: false };
        let d = r.draw(cmd, &pic, &out).unwrap();
        assert!(d.label.contains("panorama"), "{}", d.label);
        assert_eq!(d.ui, Rect::new(0, 0, fw as i64, fh as i64));
    });
    let img = read_bgr(&out);
    let px = |x: usize, y: usize, c: usize| img[(y * fw + x) * 3 + c] as f64;
    assert!((0..fw).all(|x| px(x, 10, 1) > 150.0) && (0..fw).all(|x| px(x, fh - 10, 1) < 50.0)); // upright
    let t = fill::panorama_t(fw as i64, fh as i64, sw as i64, sh as i64);
    for col in [0, fw / 8, fw / 4, fw / 2, 3 * fw / 4, fw - 1] {
        let x = (col as f64 + 0.5) / fw as f64 * 2.0 - 1.0;
        let want = (fill::panorama_u(x, t) + 1.0) / 2.0 * 255.0;
        assert!((px(col, fh / 4, 2) - want).abs() < 3.0, "{col}: {} vs {want}", px(col, fh / 4, 2));
    }
    assert!((1..fw).all(|x| px(x, fh / 4, 2) >= px(x - 1, fh / 4, 2) - 1.0)); // never runs backwards
}

#[test]
fn fit_and_zoom_place_the_picture() {
    let (sw, sh) = (192usize, 108usize);
    let src = Frame::from_bgr(&vec![200u8; sw * sh * 3], sw, sh);
    let mut r = Renderer::new(Arc::new(LutStore::default())).unwrap();
    for (fill, fw, fh) in [(Fill::Fit, 344usize, 144usize), (Fill::Zoom, 344, 144), (Fill::Stretch, 344, 144), (Fill::Fit, 192, 120)] {
        let out = target(fw, fh);
        let mut drawn = None;
        run(|cmd| {
            let pic = Picture { src: &src.texture, src_size: (sw, sh), scaler: Scaler::Fsr, sharpness: 0.5, fill,
                                look: None, frame_no: 0, hdr: false, plain: false };
            drawn = Some(r.draw(cmd, &pic, &out).unwrap());
        });
        let d = drawn.unwrap();
        let img = read_bgr(&out);
        let lit = |x: usize, y: usize| img[(y * fw + x) * 3] > 100;
        let ui = d.ui;
        for y in 0..fh {
            for x in 0..fw {
                let inside = (x as i64) >= ui.x && (x as i64) < ui.x + ui.w && (y as i64) >= ui.y && (y as i64) < ui.y + ui.h;
                assert_eq!(lit(x, y), inside, "{fill:?} {fw}x{fh} at {x},{y} ({ui:?})");
            }
        }
    }
}
