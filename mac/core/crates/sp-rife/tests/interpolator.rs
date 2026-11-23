//! RifeInterpolator's behaviour beyond the pixels, which tests/parity.rs checks. It covers options it refuses,
//! Steady HUD turned on mid-pair, the output ring, non-blocking calls, and fetching missing weights.

use sp_core::{Frame, Interpolator, PixelKind};
use sp_rife::{Options, RifeInterpolator, weights};

fn have_weights() -> bool {
    let ok = weights::default_model_dir().join("flownet_v4.22.lite.pkl").exists();
    if !ok {
        eprintln!("skipped: RIFE 4.22.lite weights not downloaded");
    }
    ok
}

/// A smooth coloured pattern moved `shift` pixels right (tests/test_metalrife.py two_frames), with a still,
/// sharp-edged box in the corner for Steady HUD.
fn frame(w: usize, h: usize, shift: f32) -> Frame {
    let mut bgr = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f32 - shift, y as f32);
            let p = (y * w + x) * 3;
            let v = [
                (xf / 9.0).sin() * 70.0 + (yf / 7.0).cos() * 50.0 + 128.0,
                (xf / 13.0).cos() * 80.0 + 128.0,
                ((xf + yf) / 11.0).sin() * 90.0 + 128.0,
            ];
            for c in 0..3 {
                bgr[p + c] = v[c].clamp(0.0, 255.0) as u8;
            }
            if (10..40).contains(&x) && (10..30).contains(&y) {
                bgr[p..p + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
    }
    Frame::from_bgr(&bgr, w, h)
}

fn err(opts: Options) -> String {
    match RifeInterpolator::new(320, 180, opts) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn refuses_what_it_cant_run() {
    let e = err(Options { version: "9.9".into(), ..Options::default() });
    assert!(e.contains("unknown RIFE model"), "{e}");
    let e = err(Options { scale: 2.0, ..Options::default() });
    assert!(e.contains("whole-number"), "{e}");
    let e = err(Options { scale: 0.3, ..Options::default() });
    assert!(e.contains("motion detail must be"), "{e}");
    let e = err(Options { fp16: false, ..Options::default() });
    assert!(e.contains("half precision"), "{e}");
    let e = err(Options { model_dir: Some("/nonexistent/sp-rife".into()), version: "4.26".into(), ..Options::default() });
    assert!(!e.is_empty());
}

fn pair(it: &mut RifeInterpolator, w: usize, h: usize) {
    it.load(&frame(w, h, 0.0));
    it.prepare(0);
    it.load(&frame(w, h, 3.0));
    it.prepare(1);
}

#[test]
fn steady_hud_turned_on_mid_pair_matches_on_from_the_start() {
    if !have_weights() {
        return;
    }
    let (w, h) = (320, 180);
    let mut a = RifeInterpolator::new(w, h, Options::default()).unwrap();
    a.set_steady(0.03);
    pair(&mut a, w, h);
    let mut b = RifeInterpolator::new(w, h, Options::default()).unwrap();
    pair(&mut b, w, h);
    let off = b.interpolate(1, 0.5).to_bgr();
    b.set_steady(0.03);
    assert_eq!(a.steady_mask(), b.steady_mask());
    assert!(a.steady_mask().iter().any(|&v| v > 0.0), "the still box's edges should be held");
    let (ga, gb) = (a.interpolate(1, 0.5).to_bgr(), b.interpolate(1, 0.5).to_bgr());
    assert_eq!(ga, gb);
    assert_ne!(ga, off, "Steady HUD changed nothing");
    b.set_steady(0.0);
    assert_eq!(b.interpolate(1, 0.5).to_bgr(), off);
}

#[test]
fn outputs_survive_the_next_calls_and_non_blocking_works() {
    if !have_weights() {
        return;
    }
    let (w, h) = (320, 180);
    let mut it = RifeInterpolator::new(w, h, Options::default()).unwrap();
    assert_eq!(it.label(), "RIFE 4.22.lite");
    assert_eq!(it.size(), (w, h));
    assert_eq!(it.padded_size(), (320, 192));
    pair(&mut it, w, h);
    let f1 = it.interpolate(1, 1.0 / 3.0);
    let k1 = f1.to_bgr();
    let f2 = it.interpolate(1, 2.0 / 3.0);
    assert_eq!(f1.to_bgr(), k1, "the first frame was overwritten by the second call");
    assert_ne!(f2.to_bgr(), k1);
    // the reverse direction (newest = slot 0) at 1 - t is the same picture up to rounding
    let back = it.interpolate(0, 2.0 / 3.0).to_bgr();
    let worst = back.iter().zip(&k1).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
    assert!(worst < 40, "slot order: worst difference {worst}");
    it.set_blocking(false);
    let f3 = it.interpolate(1, 1.0 / 3.0);
    it.wait();
    assert_eq!(f3.to_bgr(), k1);
}

#[test]
fn hdr_output_is_half_float() {
    if !have_weights() {
        return;
    }
    let (w, h) = (320, 180);
    let mut it = RifeInterpolator::new(w, h, Options { hdr: true, ..Options::default() }).unwrap();
    pair(&mut it, w, h);
    let f = it.interpolate(1, 0.5);
    assert_eq!(f.kind, PixelKind::Rgba16f);
    let px = f.read_raw();
    let v: Vec<f32> = px.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
    assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
}

#[test]
fn missing_weights_are_downloaded() {
    let tmp = std::env::temp_dir().join(format!("sp-rife-dl-{}", std::process::id()));
    let src = tmp.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("flownet_vX.pkl"), b"not really weights").unwrap();
    let url = format!("file://{}/flownet_v{{version}}.pkl", src.display());
    let dst = tmp.join("models");
    let got = weights::ensure_weights_from("X", &dst, &url).unwrap();
    assert_eq!(got, dst.join("flownet_vX.pkl"));
    assert_eq!(std::fs::read(&got).unwrap(), b"not really weights");
    assert!(weights::ensure_weights_from("Y", &dst, &url).is_err(), "a failed download is an error");
    assert!(!dst.join("flownet_vY.part").exists());
    std::fs::remove_dir_all(&tmp).unwrap();
}
