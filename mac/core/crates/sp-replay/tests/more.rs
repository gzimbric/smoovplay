//! What the Python's tests don't reach on a Mac: the GPU's NV12 against replay.py's sums (its CUDA check), HDR
//! frames, "Looks in clips", a before & after that has to scale, and saving while frames keep coming. The before &
//! after has to scale when it's side by side wider than 4096 or the game's frames are another size.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use objc2_core_video::{
    CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use sp_core::{Frame, Gpu};
use sp_media::{Converter, HdrOut, HdrSettings};
use sp_replay::vtenc::{VideoConfig, VtEncoder};
use sp_replay::{Kind, Layout, gpu, yuv};

fn solid(w: usize, h: usize, bgr: [u8; 3], bar: usize) -> Frame {
    let mut a = Vec::with_capacity(w * h * 3);
    for _ in 0..h {
        for x in 0..w {
            if x >= bar % w && x < bar % w + 8 { a.extend_from_slice(&[255, 255, 255]) } else { a.extend_from_slice(&bgr) }
        }
    }
    Frame::from_bgr(&a, w, h)
}

fn mean(img: &[u8], w: usize, (y0, y1): (usize, usize), (x0, x1): (usize, usize)) -> [f64; 3] {
    let mut s = [0.0; 3];
    for y in y0..y1 {
        for x in x0..x1 {
            for c in 0..3 {
                s[c] += img[(y * w + x) * 3 + c] as f64;
            }
        }
    }
    let n = ((y1 - y0) * (x1 - x0)) as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

#[test]
fn the_gpus_nv12_matches_replay_py() {
    let (w, h) = (161usize, 97usize); // odd, so it's cut to even
    let mut rng = Rng::new(1);
    let img: Vec<u8> = (0..w * h * 3).map(|_| (rng.next() * 256.0) as u8).collect();
    let f = Frame::from_bgr(&img, w, h);
    let enc = VtEncoder::new(
        VideoConfig { width: w & !1, height: h & !1, nominal: 60, bpp: 0.1, realtime: true, expected_fps: None },
        Box::new(|_| {}),
    )
    .unwrap();
    let buf = enc.pixel_buffer(4).unwrap();
    let cache = gpu::TexCache::new().unwrap();
    let planes = cache.planes(&buf).unwrap();
    let cb = Gpu::shared().queue.commandBuffer().unwrap();
    gpu::encode_to_nv12(&cb, &f.texture, w & !1, h & !1, &planes).unwrap();
    cb.commit();
    cb.waitUntilCompleted();
    let (ew, eh) = (w & !1, h & !1);
    let mut even = Vec::with_capacity(ew * eh * 3);
    for y in 0..eh {
        even.extend_from_slice(&img[y * w * 3..(y * w + ew) * 3]);
    }
    let want = yuv::bgr_to_i420(&even, ew, eh);
    let mut worst = 0i32;
    unsafe {
        assert_eq!(CVPixelBufferLockBaseAddress(&buf, CVPixelBufferLockFlags::ReadOnly), 0);
        let (y, yr) = (CVPixelBufferGetBaseAddressOfPlane(&buf, 0) as *const u8, CVPixelBufferGetBytesPerRowOfPlane(&buf, 0));
        let (uv, uvr) = (CVPixelBufferGetBaseAddressOfPlane(&buf, 1) as *const u8, CVPixelBufferGetBytesPerRowOfPlane(&buf, 1));
        for r in 0..eh {
            for c in 0..ew {
                worst = worst.max((*y.add(r * yr + c) as i32 - want[r * ew + c] as i32).abs());
            }
        }
        let (cw, ch) = (ew / 2, eh / 2);
        for r in 0..ch {
            for c in 0..cw {
                let u = *uv.add(r * uvr + 2 * c) as i32;
                let v = *uv.add(r * uvr + 2 * c + 1) as i32;
                worst = worst.max((u - want[ew * eh + r * cw + c] as i32).abs());
                worst = worst.max((v - want[ew * eh + cw * ch + r * cw + c] as i32).abs());
            }
        }
        CVPixelBufferUnlockBaseAddress(&buf, CVPixelBufferLockFlags::ReadOnly);
    }
    assert!(worst <= 1, "the GPU's NV12 is {worst} off replay.py's sums");
}

/// f32 to IEEE half bits. It handles normal numbers and zero, which is enough for 0 to 1.
fn half(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = (b >> 13) & 0x3ff;
    let round = (b >> 12) & 1;
    ((e as u32) << 10 | m) as u16 + round as u16
}

fn pq_of_nits(nits: f64) -> f32 {
    let (m1, m2, c1, c2, c3) = (0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875);
    let y = (nits / 10000.0).powf(m1);
    (((c1 + c2 * y) / (1.0 + c3 * y)).powf(m2)) as f32
}

#[test]
fn hdr_frames_are_tone_mapped_as_sdr8_would_show_them() {
    let tmp = TempDir::new("hdr");
    let (w, h) = (128usize, 64usize);
    // A dim grey on the left and a bright orange highlight on the right (PQ, BT.2020, 0 to 1)
    let mut px = Vec::with_capacity(w * h * 4);
    for _ in 0..h {
        for x in 0..w {
            let (r, g, b) = if x < w / 2 { (pq_of_nits(80.0), pq_of_nits(80.0), pq_of_nits(80.0)) } else { (pq_of_nits(900.0), pq_of_nits(300.0), pq_of_nits(30.0)) };
            px.extend_from_slice(&[half(r), half(g), half(b), half(1.0)]);
        }
    }
    let hdr = Frame::from_rgba16f_bits(&px, w, h);
    let settings = HdrSettings::new(HdrOut::Sdr8, None);
    let want = Converter::new().unwrap().pq16_to_8(&hdr, true, &settings).to_bgr();
    let (rec, said) = recorder(tmp.path(), 2);
    rec.set_hdr(Some(settings));
    for i in 0..120 {
        let t = 1000.0 + i as f64 / 60.0;
        rec.frame(&hdr, t);
        if i % 2 == 0 {
            rec.real_frame(&hdr, t - 0.04);
        }
    }
    assert!(rec.flush(10.0));
    let paths = rec.save(Some(Kind::Smoovplay), None, true);
    assert_eq!(paths.len(), 1, "{}", said.log());
    let pics = decode_bgr(&paths[0], w, h);
    let got = &pics[pics.len() / 2];
    for (name, xs) in [("grey", (8, w / 2 - 8)), ("highlight", (w / 2 + 8, w - 8))] {
        let (a, b) = (mean(got, w, (8, h - 8), xs), mean(&want, w, (8, h - 8), xs));
        let d = (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f64::max);
        println!("{name}: the clip has {a:?}, sdr8 {b:?}");
        assert!(d < 3.0, "{name}: the clip has {a:?}, sdr8 {b:?}");
        // and it's a real picture, a mid grey or an orange with its blue lowest
        assert!(if name == "grey" { b[1] > 60.0 && (b[0] - b[2]).abs() < 4.0 } else { b[2] > b[1] && b[1] > b[0] }, "{b:?}");
    }
    rec.close(0.0);
}

#[test]
fn looks_go_in_the_smoovplay_side_only() {
    let tmp = TempDir::new("looks");
    let (w, h) = (160usize, 96usize);
    let (rec, said) = recorder(tmp.path(), 2);
    let red = Arc::new(solid(w, h, [30, 30, 200], 1000));
    let r2 = red.clone();
    rec.set_look(Some(Box::new(move |_f: &Frame| Ok((*r2).clone()))));
    for i in 0..150 {
        let t = 1000.0 + i as f64 / 60.0;
        rec.frame(&solid(w, h, [40, 160, 40], i * 3), t);
        if i % 2 == 0 {
            rec.real_frame(&solid(w, h, [160, 40, 40], i * 3), t - 0.04);
        }
    }
    assert!(rec.flush(10.0));
    let paths = rec.save(Some(Kind::Both), None, true);
    assert_eq!(paths.len(), 2, "{}", said.log());
    let clip = decode_bgr(&paths[0], w, h);
    let m = mean(&clip[10], w, (40, 90), (0, w));
    assert!(m[2] > 150.0 && m[1] < 60.0, "the clip isn't the look's red: {m:?}");
    let both = decode_bgr(&paths[1], w, h);
    let (l, r) = (mean(&both[10], w, (60, 90), (4, 30)), mean(&both[10], w, (60, 90), (w - 30, w - 4)));
    assert!(l[0] > 120.0 && l[2] < 80.0, "the game's side has the look: {l:?}");
    assert!(r[2] > 150.0 && r[1] < 60.0, "SmoovPlay's side hasn't: {r:?}");
    rec.close(0.0);
}

#[test]
fn a_before_and_after_that_has_to_scale() {
    // Side by side wider than 4096, so both halves are shrunk to fit
    let tmp = TempDir::new("scale");
    let (w, h) = (2400usize, 128usize);
    let (rec, said) = recorder(tmp.path(), 2);
    for i in 0..120 {
        let t = 1000.0 + i as f64 / 60.0;
        rec.frame(&solid(w, h, [40, 160, 40], i * 3), t);
        if i % 2 == 0 {
            // The game's frames at half the size, grown to fit
            rec.real_frame(&solid(w / 2, h / 2, [160, 40, 40], i * 3), t - 0.04);
        }
    }
    assert!(rec.flush(10.0));
    let side = rec.save(Some(Kind::Compare), Some(Layout::Side), true);
    assert_eq!(side.len(), 1, "{}", said.log());
    let (v, _) = probe(&side[0]);
    let v = v.unwrap();
    let (ow, oh) = (2 * ((w as f64 * 4096.0 / (2.0 * w as f64)) as usize & !1), (h as f64 * 4096.0 / (2.0 * w as f64)) as usize & !1);
    assert_eq!((v.width as usize, v.height as usize), (ow, oh));
    let pic = &decode_bgr(&side[0], ow, oh)[20];
    let (l, r) = (mean(pic, ow, (oh / 2, oh - 4), (100, 600)), mean(pic, ow, (oh / 2, oh - 4), (ow / 2 + 100, ow / 2 + 600)));
    assert!(l[0] > 120.0 && l[1] < 80.0, "left {l:?}");
    assert!(r[1] > 120.0 && r[0] < 80.0, "right {r:?}");
    let split = rec.save(Some(Kind::Compare), Some(Layout::Split), true);
    let pic = &decode_bgr(&split[0], w, h)[20];
    let (l, r) = (mean(pic, w, (70, 120), (100, 600)), mean(pic, w, (70, 120), (w - 600, w - 100)));
    assert!(l[0] > 120.0 && l[1] < 80.0 && r[1] > 120.0 && r[0] < 80.0, "split: left {l:?}, right {r:?}");
    rec.close(0.0);
}

#[test]
fn saving_while_play_goes_on_doesnt_wait_for_it() {
    let tmp = TempDir::new("live");
    let (rec, said) = recorder(tmp.path(), 3);
    let rec = Arc::new(rec);
    let r2 = rec.clone();
    let start = sp_core::clock::now();
    let player = std::thread::spawn(move || {
        let frames: Vec<Frame> = (0..30).map(|k| solid(320, 192, [40, 160, 40], k * 10)).collect();
        for i in 0..(5.0 * 90.0) as usize {
            let due = start + i as f64 / 90.0;
            while sp_core::clock::now() < due {
                std::thread::sleep(Duration::from_micros(300));
            }
            r2.frame(&frames[i % 30], sp_core::clock::now());
            if i % 3 == 0 {
                r2.real_frame(&frames[i % 30], sp_core::clock::now() - 0.04);
            }
        }
    });
    std::thread::sleep(Duration::from_secs_f64(4.0));
    let t = Instant::now();
    let paths = rec.save(Some(Kind::Smoovplay), None, true);
    let took = t.elapsed().as_secs_f64();
    player.join().unwrap();
    assert_eq!(paths.len(), 1, "{}", said.log());
    assert!(took < 0.3, "saving took {took:.2} s while frames kept coming");
    let (v, _) = probe(&paths[0]);
    assert!(v.unwrap().duration >= 3.0);
    assert_eq!(rec.dropped(), 0);
    rec.close(0.0);
}
