//! The sources other than Remote Play's parity runs (tests/parity.rs): test mode, decoded frames through the tap
//! (TapSource, open_tap, Clock60), and video files (FileSource, Ps5SimSource), ported from tests/test_sources.py and
//! tests/test_tap.py where Python has the same test.

mod common;

use std::time::Duration;

use sp_core::{clock, Event, Frame, Source, Tick};
use sp_media::*;
use sp_tap::{FrameTap, CODEC_BGRA, CODEC_H265, CODEC_NV12, FEATURE_VARIABLE};

fn tick(i: u8) -> Tick {
    Tick::new(Frame::from_bgr(&[i; 4 * 4 * 3], 4, 4), clock::now(), i as u64)
}

fn px(f: &Frame) -> u8 {
    f.to_bgr()[0]
}

#[test]
fn test_mode_plays_60fps_as_30fps_and_keeps_the_left_out_frames() {
    let mut tm = TestMode::new(true);
    let ticks: Vec<Tick> = (0..12).map(|i| tm.apply(tick(i)).unwrap()).collect();
    assert_eq!(ticks.iter().map(|t| t.content.unwrap()).collect::<Vec<_>>(), vec![0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10]);
    assert_eq!(ticks.iter().map(|t| t.truth_new.unwrap()).collect::<Vec<_>>(), [true, false].repeat(6));
    assert_eq!(ticks.iter().map(|t| t.new_hint.unwrap()).collect::<Vec<_>>(), [true, false].repeat(6));
    for i in (1..12).step_by(2) {
        assert!(std::ptr::eq(&*ticks[i].frame.texture, &*ticks[i - 1].frame.texture), "the kept frame again");
    }
    assert_eq!(px(&tm.gt(11).unwrap()), 11);
    assert_eq!(px(&tm.gt(5).unwrap()), 5);
    assert!(tm.gt(1).is_none() && tm.gt(4).is_none());
    // game_new is what the source said. There's no size check here, so it's new_hint, which is None.
    assert_eq!(ticks[0].extra["game_new"], serde_json::Value::Null);
}

#[test]
fn test_mode_off_passes_ticks_through_and_switching_on_starts_clean() {
    let mut tm = TestMode::new(false);
    let t = tm.apply(tick(0)).unwrap();
    assert!(t.content.is_none() && t.new_hint.is_none());
    tm.set_on(true);
    assert_eq!(tm.apply(tick(1)).unwrap().content, Some(0));
    assert_eq!(tm.apply(tick(2)).unwrap().content, Some(0));
    tm.apply(tick(3));
    tm.set_on(false);
    tm.set_on(true);
    assert!(tm.apply(tick(4)).is_none()); // count is odd and nothing from before may be repeated
    assert_eq!(tm.apply(tick(5)).unwrap().content, Some(4));
    // A frame the game repeated (by the size check) offers no truth around it
    let mut tm = TestMode::new(true);
    for i in 0..6u8 {
        let mut t = tick(i);
        t.extra.insert("size_new".into(), serde_json::json!(i != 3));
        tm.apply(t);
    }
    assert!(tm.gt(3).is_none() && tm.gt(5).is_some());
}

fn name(tag: &str) -> String {
    format!("Local\\sp-rs-s{tag}{}", std::process::id() % 10000)
}

/// Push frames from a thread, a little apart, once the reader is listening.
fn send(n: String, frames: Vec<(Vec<u8>, i32, Option<bool>)>, gaps_ms: Vec<u64>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let tap = FrameTap::open(&n).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        for (k, (a, lost, new)) in frames.into_iter().enumerate() {
            tap.push(&a, lost, false, new);
            std::thread::sleep(Duration::from_millis(*gaps_ms.get(k).unwrap_or(&4)));
        }
    })
}

fn ticks(src: &mut dyn Source, n: usize) -> Vec<Tick> {
    let t_end = clock::now() + 5.0;
    let mut out = Vec::new();
    while out.len() < n {
        match src.next_event() {
            Some(Event::Tick(t)) => out.push(t),
            Some(Event::Waiting(_)) => assert!(clock::now() < t_end, "no frames came"),
            None => break,
        }
    }
    out
}

#[test]
fn tap_source_bgra_end_to_end() {
    let n = name("b");
    sp_tap::unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    let (w, h) = (64, 36);
    tap.begin(CODEC_BGRA, w, h, 60, 0, None).unwrap();
    let OpenedTap::Tap(mut src) = open_tap(&n, 1.0, None, None).unwrap() else { panic!("not a TapSource") };
    assert_eq!(src.info().kind, "tap");
    let frames = (0..6)
        .map(|i| {
            let px = [10 * i as u8, 100, 200, 255]; // B, G, R, A
            (px.repeat((w * h) as usize), (i == 4) as i32, [Some(true), Some(false), None][i % 3])
        })
        .collect();
    let th = send(n.clone(), frames, vec![]);
    let got = ticks(&mut src, 6);
    th.join().unwrap();
    assert_eq!(got.iter().map(|t| t.extra["seq"].as_u64().unwrap()).collect::<Vec<_>>(), (0..6).collect::<Vec<_>>());
    assert!(got.iter().all(|t| (t.frame.width, t.frame.height) == (64, 36)));
    assert_eq!(got.iter().map(|t| px(&t.frame)).collect::<Vec<_>>(), vec![0, 10, 20, 30, 40, 50]);
    assert_eq!(got[0].frame.to_bgr()[..3], [0, 100, 200]);
    assert_eq!(got.iter().map(|t| t.new_hint).collect::<Vec<_>>(), [Some(true), Some(false), None].repeat(2));
    assert_eq!(got.iter().map(|t| t.damaged).collect::<Vec<_>>(), vec![false, false, false, false, true, false]);
    assert_eq!(got.iter().map(|t| t.index).collect::<Vec<_>>(), (0..6).collect::<Vec<_>>());
    // The stream ends, and so do the ticks
    tap.end();
    let t_end = clock::now() + 3.0;
    loop {
        match src.next_event() {
            None => break,
            Some(Event::Tick(_)) => panic!("a tick after the end"),
            Some(Event::Waiting(_)) => assert!(clock::now() < t_end, "never ended"),
        }
    }
    src.close();
    sp_tap::unlink(&n);
}

#[test]
fn tap_source_nv12_colours() {
    let n = name("n");
    sp_tap::unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    let (w, h) = (64usize, 36usize);
    tap.begin(CODEC_NV12, w as u32, h as u32, 60, 0, None).unwrap();
    let OpenedTap::Tap(mut src) = open_tap(&n, 1.0, None, None).unwrap() else { panic!("not a TapSource") };
    // Mid grey with neutral chroma, then a red in BT.601 limited range (Y 81, Cb 90, Cr 240). With no colours in
    // the header, 36 lines means BT.601.
    let grey = vec![128u8; w * h * 3 / 2];
    let mut red = vec![81u8; w * h];
    red.extend([90u8, 240].repeat(w * h / 4));
    let th = send(n.clone(), vec![(grey, 0, None), (red, 0, None)], vec![]);
    let got = ticks(&mut src, 2);
    th.join().unwrap();
    let g = got[0].frame.to_bgr();
    assert!(g.iter().all(|&v| v.abs_diff(130) <= 3), "limited range 128 comes out about 130");
    let r = got[1].frame.to_bgr();
    let (b, gg, rr) = (r[0], r[1], r[2]);
    assert!(rr >= 250 && gg <= 3 && b <= 3, "red came out {b} {gg} {rr}");
    tap.end();
    src.close();
    sp_tap::unlink(&n);
}

#[test]
fn open_tap_variable_writer_gets_a_beat() {
    let n = name("v");
    sp_tap::unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    tap.begin(CODEC_BGRA, 16, 16, 60, 0, Some(FEATURE_VARIABLE)).unwrap();
    let opened = open_tap(&n, 1.0, Some(TestMode::shared(false)), None).unwrap();
    let OpenedTap::Variable(mut src) = opened else { panic!("not on a beat") };
    assert!(!src.info().test_mode && src.info().rate == "variable");
    src.close();
    assert!(open_tap(&n, 1.0, Some(TestMode::shared(true)), None).is_err(), "test mode needs every refresh");
    tap.end();
    sp_tap::unlink(&n);
}

/// framegen/sources.py's Clock60 test, with uneven frames from a variable writer on a 100 Hz beat.
#[test]
fn clock60_puts_uneven_frames_on_a_steady_beat() {
    let n = name("c");
    sp_tap::unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    tap.begin(CODEC_BGRA, 16, 16, 60, 0, Some(FEATURE_VARIABLE)).unwrap();
    let src = TapSource::new(FrameTap::open(&n).unwrap(), &n, tap.header(), None).unwrap();
    let mut clock = Clock60::new(src, 100.0);
    let gaps = vec![35u64, 4, 60, 12, 30, 200];
    let frames = (0..6u8).map(|i| ([i, 0, 0, 255].repeat(256), 0, None)).collect();
    let th = send(n.clone(), frames, gaps);
    // Waitings until the first frame
    let mut out = Vec::new();
    let t_end = clock::now() + 4.0;
    while clock::now() < t_end {
        match clock.next_event() {
            Some(Event::Tick(t)) => {
                out.push(t);
                if px(&out[out.len() - 1].frame) == 5 && out.len() > 30 {
                    break;
                }
            }
            Some(Event::Waiting(_)) => assert!(out.is_empty(), "a Waiting after the beat started"),
            None => break,
        }
    }
    th.join().unwrap();
    let fresh = out.iter().filter(|t| t.new_hint.is_none()).count();
    let repeats = out.iter().filter(|t| t.new_hint == Some(false)).count();
    assert_eq!(fresh as u64 + clock.overtaken(), 6, "every frame handed on, unless a newer one overtook it");
    assert!(repeats >= 5);
    let ids: Vec<u8> = out.iter().map(|t| px(&t.frame)).collect();
    assert!(ids.windows(2).all(|w| w[0] <= w[1]) && ids[0] == 0 && *ids.last().unwrap() == 5);
    for w in out.windows(2) {
        if w[1].new_hint == Some(false) {
            assert!(std::ptr::eq(&*w[1].frame.texture, &*w[0].frame.texture) && !w[1].damaged);
        }
    }
    let mut beats: Vec<f64> = out.windows(2).map(|w| w[1].t_ready.unwrap() - w[0].t_ready.unwrap()).collect();
    beats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = beats[beats.len() / 2];
    assert!((med - 0.01).abs() < 0.003, "median beat {med}");
    assert!(out.iter().all(|t| t.t_ready.unwrap() >= t.t_arrival));
    assert_eq!(out.iter().map(|t| t.index).collect::<Vec<_>>(), (0..out.len() as u64).collect::<Vec<_>>());
    clock.close();
    tap.end();
    sp_tap::unlink(&n);
}

#[test]
fn another_apps_compressed_frames_are_read_as_chiaki_ngs() {
    let n = name("h");
    sp_tap::unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    tap.begin(CODEC_H265, 1920, 1080, 60, 0, None).unwrap();
    let opened = open_tap(&n, 1.0, None, None).unwrap();
    assert_eq!(opened.kind(), "chiaki");
    let OpenedTap::Chiaki(mut src) = opened else { panic!() };
    assert_eq!(src.info().name, "Frame tap");
    assert_eq!(src.info().kind, "chiaki");
    // Nothing comes, so Waitings with the ring's hint
    let Some(Event::Waiting(w)) = src.next_event() else { panic!("no Waiting") };
    assert_eq!(w.what, "the PS5");
    std::thread::sleep(Duration::from_millis(1100));
    assert!(src.why().contains("isn't sending any frames"), "{}", src.why());
    tap.end();
    assert!(src.why().contains("isn't streaming"), "{}", src.why());
    src.close();
    sp_tap::unlink(&n);
}

fn clip() -> Option<std::path::PathBuf> {
    let p = common::repo().join("testdata/clip_1080p60.mp4");
    p.exists().then_some(p)
}

#[test]
fn file_source_frames_and_colours() {
    let Some(clip) = clip() else { return eprintln!("skipped: no testdata clip") };
    let mut src = FileSource::open(&clip, false, None, None).unwrap();
    assert_eq!(src.size(), (1920, 1080));
    assert_eq!(src.info().kind, "file");
    assert!((src.info().hz - 60.0).abs() < 0.01);
    let mut n = 0;
    let mut f30 = None;
    while let Some(Event::Tick(t)) = src.next_event() {
        assert_eq!(t.index, n);
        assert!((t.t_source.unwrap() - n as f64 / 60.0).abs() < 1e-9);
        if n == 30 {
            f30 = Some(t.frame.clone());
        }
        n += 1;
    }
    assert_eq!(n, 720, "every frame of the 12 s clip");
    let reference = common::fixtures().join("clip_30.bgr");
    if let Ok(r) = std::fs::read(reference) {
        let (max, mean, _) = common::diff(&f30.unwrap().to_bgr(), &r);
        eprintln!("FileSource frame 30 vs PyAV bgr24: max {max}, mean {mean:.4}");
        assert!(max <= 1);
    }
}

#[test]
fn ps5_sim_matches_python_and_a_file_in_test_mode() {
    let Some(clip) = clip() else { return eprintln!("skipped: no testdata clip") };
    // The hitches Python's Ps5SimSource makes for the same seed
    if let Ok(j) = std::fs::read_to_string(common::fixtures().join("sim_ticks.json")) {
        let want: Vec<(i64, bool)> = serde_json::from_str(&j).unwrap();
        let opts = SimOptions { hitch: 0.3, seed: 0, max_ticks: Some(120), ..Default::default() };
        let mut sim = Ps5SimSource::open(&clip, opts).unwrap();
        let mut got = Vec::new();
        while let Some(Event::Tick(t)) = sim.next_event() {
            got.push((t.content.unwrap(), t.truth_new.unwrap()));
        }
        assert_eq!(got, want, "hitch pattern differs from Python's");
        assert_eq!(sim.info().kind, "sim");
        assert!(!sim.info().test_mode);
    }
    // A raw file in test mode gives the same ticks as the simulation (without hitches)
    let mut sim = Ps5SimSource::open(&clip, SimOptions { max_ticks: Some(40), ..Default::default() }).unwrap();
    let test = TestMode::shared(true);
    let mut raw = FileSource::open(&clip, false, Some(40), Some(test.clone())).unwrap();
    let mut k = 0;
    while let (Some(Event::Tick(a)), Some(Event::Tick(b))) = (sim.next_event(), raw.next_event()) {
        assert_eq!((a.content, a.truth_new), (b.content, b.truth_new), "tick {k}");
        assert!(a.new_hint.is_none() && b.new_hint == Some(k % 2 == 0));
        if k % 7 == 0 {
            assert_eq!(a.frame.to_bgr(), b.frame.to_bgr(), "tick {k}: pictures differ");
        }
        if a.truth_new == Some(true) {
            let c = a.content.unwrap();
            let (ga, gb) = (sim.ground_truth(c + 1), raw.ground_truth(c + 1));
            if let (Some(ga), Some(gb)) = (ga, gb) {
                assert_eq!(ga.to_bgr()[..300], gb.to_bgr()[..300]);
            }
        }
        k += 1;
    }
    assert!(k >= 18, "{k} ticks");
}

#[test]
fn ps5_sim_noise_and_realtime() {
    let Some(clip) = clip() else { return eprintln!("skipped: no testdata clip") };
    let opts = SimOptions { noise: 4.0, realtime: true, max_ticks: Some(12), ..Default::default() };
    let mut sim = Ps5SimSource::open(&clip, opts).unwrap();
    let t0 = clock::now();
    let mut frames = Vec::new();
    while let Some(Event::Tick(t)) = sim.next_event() {
        frames.push(t.frame.to_bgr());
    }
    let took = clock::now() - t0;
    assert_eq!(frames.len(), 12);
    assert!(took > 11.0 / 60.0 - 0.01, "not paced: {took}");
    // Against the same tick without noise it's about sd 4, a little less because it's truncated and clipped at 0
    // and 255
    let mut clean = Ps5SimSource::open(&clip, SimOptions { max_ticks: Some(1), ..Default::default() }).unwrap();
    let Some(Event::Tick(c)) = clean.next_event() else { panic!() };
    let d: Vec<i32> = frames[0]
        .iter()
        .zip(&c.frame.to_bgr())
        .filter(|(_, b)| (16..240).contains(*b))
        .map(|(a, b)| *a as i32 - *b as i32)
        .collect();
    let sd = (d.iter().map(|v| (v * v) as f64).sum::<f64>() / d.len() as f64).sqrt();
    let mean = d.iter().sum::<i32>() as f64 / d.len() as f64;
    assert!(sd > 3.0 && sd < 4.5 && mean.abs() < 0.2, "noise sd {sd}, mean {mean}");
}

/// POSIX shared memory outlives chiaki-ng, so a crash leaves a live header behind. One whose newest frame is old
/// isn't taken as a stream (it gave the old size, then "stream size changed"); the next real stream is.
#[test]
fn a_header_left_by_a_dead_writer_is_waited_out() {
    let n = format!("Local\\sp-stale-{}", std::process::id() % 10000);
    sp_tap::unlink(&n);
    let tap = sp_tap::FrameTap::open(&n).unwrap();
    tap.begin(1, 1280, 720, 60, 0, None).unwrap();
    tap.push(b"\x00\x00\x01old-frame", 0, false, None);
    std::thread::sleep(std::time::Duration::from_millis(150));
    let opts = |stale_s| sp_media::ChiakiOptions { tap_name: n.clone(), wait_s: 0.5, stale_s, ..Default::default() };
    match sp_media::ChiakiSource::open(opts(0.1)) {
        Err(sp_media::OpenError::Timeout(_)) => {}
        other => panic!("a stale header was taken as a stream: {:?}", other.map(|_| ())),
    }
    tap.begin(1, 1920, 1080, 60, 0, None).unwrap(); // chiaki-ng started again: a new session, with a fresh frame
    tap.push(b"\x00\x00\x01new-frame", 0, false, None);
    let mut src = sp_media::ChiakiSource::open(opts(0.1)).expect("a fresh stream opens");
    sp_core::Source::close(&mut src);
    sp_tap::unlink(&n);
}
