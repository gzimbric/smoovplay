//! The whole source with stand-ins for a card (tests/test_capture.py's source tests, tests/test_audio.py's sound
//! tests): raw frames read as a card's are, and recordings played at their own rate through the same per-frame path,
//! checked against the Python's own frames (fixtures/capture/clip_*.bgr) and its sound timing (onset.mov).

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{blocks, centres, max_diff, to_nv12, to_yuy2};
use sp_capture::{CardOptions, CardSource, Colour, SoundOut, SoundStats};
use sp_core::{clock, Event, Source, Tick};
use sp_media::TestMode;

const H: usize = 64;
const W: usize = 96;
const LIMITED_601: Colour = Colour::Fixed { bt709: false, full_range: false };

fn ticks_of(src: &mut CardSource) -> Vec<Tick> {
    let mut out = Vec::new();
    while let Some(ev) = src.next_event() {
        if let Event::Tick(t) = ev {
            out.push(t);
        }
    }
    src.close();
    out
}

fn seq(t: &Tick) -> u64 {
    t.extra["seq"].as_u64().unwrap()
}

#[test]
fn source_plays_raw_frames_like_a_card() {
    let (a, b) = (blocks(H, W, 3), blocks(H, W, 4));
    let pics = [&a, &a, &b, &b, &b, &a];
    let frames = pics.iter().map(|p| Some(to_nv12(p, H, W))).collect();
    let mut src = CardSource::raw_frames(frames, W, H, "nv12", 240.0, CardOptions { colour: LIMITED_601, ..Default::default() }).unwrap();
    assert_eq!((src.width(), src.height(), src.fmt(), src.raw(), src.info().kind.as_str()), (W, H, "nv12", true, "capture"));
    let ticks = ticks_of(&mut src);
    assert_eq!(ticks.iter().map(|t| t.index).collect::<Vec<_>>(), (0..6).collect::<Vec<_>>());
    assert!(ticks.iter().all(|t| t.t_ready.unwrap() >= t.t_arrival));
    assert_eq!(ticks.iter().map(|t| (t.t_source.unwrap() * 240.0).round() as u64).collect::<Vec<_>>(), (0..6).collect::<Vec<_>>());
    assert_eq!(ticks.iter().map(seq).collect::<Vec<_>>(), (0..6).collect::<Vec<_>>()); // numbered as they came off the card
    let got: Vec<Vec<u8>> = ticks.iter().map(|t| t.frame.to_bgr()).collect();
    for (g, want) in got.iter().zip(pics) {
        assert!(max_diff(&centres(g, H, W), &centres(want, H, W)) <= 4);
    }
    assert_eq!(got[0], got[1]); // a repeat stays bit-identical through the conversion
    // paced like a card: 6 frames at 240 fps take 5/240 s from the first
    let span = ticks[5].t_arrival - ticks[0].t_arrival;
    assert!((span - 5.0 / 240.0).abs() < 0.01, "{span}");
}

#[test]
fn source_test_mode_and_yuy2() {
    let pics: Vec<Vec<u8>> = (10..18).map(|s| blocks(H, W, s)).collect();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("card.yuyv422");
    std::fs::write(&path, pics.iter().flat_map(|p| to_yuy2(p, H, W)).collect::<Vec<u8>>()).unwrap();
    let test = TestMode::shared(true);
    let mut src = CardSource::raw_file(&path, W, H, "yuyv422", 240.0, CardOptions { test: Some(test.clone()), ..Default::default() }).unwrap();
    let ticks = ticks_of(&mut src);
    assert_eq!(ticks.len(), 8); // test mode keeps every tick, repeating every other frame's content
    assert_eq!(ticks[0].frame.to_bgr(), ticks[1].frame.to_bgr());
    assert_eq!(ticks.iter().map(|t| t.content.unwrap()).collect::<Vec<_>>(), vec![0, 0, 2, 2, 4, 4, 6, 6]);
    for (t, p) in ticks.iter().step_by(2).zip(pics.iter().step_by(2)) {
        assert!(max_diff(&centres(&t.frame.to_bgr(), H, W), &centres(p, H, W)) <= 4);
    }
    // the frames test mode left out, for scoring
    let gt = src.ground_truth(7).unwrap().to_bgr();
    assert!(max_diff(&centres(&gt, H, W), &centres(&pics[7], H, W)) <= 4);
}

#[test]
fn a_damaged_frame_is_skipped_and_capture_goes_on() {
    // A cheap card's broken frame (often the first one) ended capture before a single frame was shown.
    let pics: Vec<Vec<u8>> = (40..44).map(|s| blocks(H, W, s)).collect();
    let frames = pics.iter().enumerate().map(|(i, p)| if i == 0 || i == 2 { None } else { Some(to_nv12(p, H, W)) }).collect();
    let mut src = CardSource::raw_frames(frames, W, H, "nv12", 60.0, CardOptions { colour: LIMITED_601, ..Default::default() }).unwrap();
    let ticks = ticks_of(&mut src);
    assert_eq!(ticks.iter().map(seq).collect::<Vec<_>>(), vec![1, 3]); // numbered as the card sent them, broken ones too
    assert!(max_diff(&centres(&ticks[0].frame.to_bgr(), H, W), &centres(&pics[1], H, W)) <= 4);
    let out = src.messages().join("\n");
    assert_eq!(src.damaged(), 2);
    assert!(!out.contains("stopped"));
    assert_eq!(out.matches("the card sent a damaged NV12 frame: skipped").count(), 1); // said once
    assert!(out.contains("dropped frames: 0, damaged frames from the card skipped: 2"), "{out}");
}

#[test]
fn capture_stops_saying_why_when_no_frame_can_be_read() {
    let frames = vec![None; 10];
    let opts = CardOptions { damaged_give_up_s: 0.03, ..Default::default() };
    let mut src = CardSource::raw_frames(frames, W, H, "nv12", 60.0, opts).unwrap();
    assert!(ticks_of(&mut src).is_empty());
    let out = src.messages().join("\n");
    assert!(out.contains("[capture] stopped: none of the card's NV12 frames could be read for 0.03 s"), "{out}");
}

#[test]
fn frames_keep_the_cards_numbers_when_the_queue_drops_some() {
    // A frame dropped from the queue (the Mac behind for a moment) leaves a gap in the numbers rather than shifting
    // them: Game frame rate 30 fps counts by them, and a shift made its beat move, a hitch.
    let frames = (30..50).map(|s| Some(to_nv12(&blocks(H, W, s), H, W))).collect();
    let mut src = CardSource::raw_frames(frames, W, H, "nv12", 240.0, CardOptions { queue_size: 1, ..Default::default() }).unwrap();
    let mut ticks = Vec::new();
    while let Some(ev) = src.next_event() {
        if let Event::Tick(t) = ev {
            ticks.push(t);
            std::thread::sleep(Duration::from_millis(20)); // slower than the "card" (240 fps): the queue drops frames
        }
    }
    let dropped = src.dropped();
    src.close();
    let seqs: Vec<u64> = ticks.iter().map(seq).collect();
    assert!(ticks.len() < 20 && dropped > 0);
    assert!(seqs.windows(2).all(|w| w[1] > w[0]));
    assert_eq!(seqs, ticks.iter().map(|t| (t.t_source.unwrap() * 240.0).round() as u64).collect::<Vec<_>>());
}

/// Stands in for the Player: records what it's given, and when.
#[derive(Default)]
struct Out {
    rate: u32,
    channels: u32,
    pushed: Vec<(f64, Vec<i16>)>,
    delays: Vec<f64>,
    stats: Option<SoundStats>,
}

struct OutHandle(Arc<Mutex<Out>>);

impl SoundOut for OutHandle {
    fn push(&mut self, pcm: &[i16]) {
        self.0.lock().unwrap().pushed.push((clock::now(), pcm.to_vec()));
    }
    fn set_delay(&mut self, seconds: f64) {
        self.0.lock().unwrap().delays.push(seconds);
    }
    fn stats(&self) -> Option<SoundStats> {
        self.0.lock().unwrap().stats.clone()
    }
}

fn recorder(out: &Arc<Mutex<Out>>) -> sp_capture::SoundFactory {
    let out = out.clone();
    Box::new(move |rate, channels| {
        let mut o = out.lock().unwrap();
        (o.rate, o.channels) = (rate, channels);
        Ok(Box::new(OutHandle(out.clone())) as Box<dyn SoundOut>)
    })
}

fn mean(bgr: &[u8]) -> f64 {
    bgr.iter().map(|&v| v as f64).sum::<f64>() / bgr.len() as f64
}

/// A recording's sound comes out with its picture: the tone that starts with the flash comes out within a frame of
/// the flash's tick (test_a_recordings_sound_comes_out_with_its_picture, measured to the sample).
#[test]
fn a_recordings_sound_comes_out_with_its_picture() {
    let Some(path) = common::fixture("onset.mov") else { return };
    let out = Arc::new(Mutex::new(Out::default()));
    let mut src = CardSource::video_file(&path, CardOptions { audio: Some(recorder(&out)), ..Default::default() }).unwrap();
    assert!(src.can_delay_audio() && !src.can_press());
    src.set_audio_delay(40.0);
    let ticks = ticks_of(&mut src);
    let o = out.lock().unwrap();
    assert_eq!((o.rate, o.channels), (48000, 2));
    assert_eq!(o.delays.len(), 1);
    assert!((o.delays[0] - 0.04).abs() < 1e-9);
    assert_eq!(ticks.len(), 120);
    let flash = ticks.iter().find(|t| mean(&t.frame.to_bgr()) > 120.0).unwrap();
    assert_eq!(seq(flash), 60);
    let (t_push, pcm) = o.pushed.iter().find(|(_, p)| p.iter().any(|v| v.abs() > 1000)).unwrap();
    let k = pcm.iter().position(|v| v.abs() > 1000).unwrap() / 2;
    let onset = t_push + k as f64 / 48000.0;
    let offset = onset - flash.t_arrival;
    println!("the tone came {:+.1} ms from the flash's tick", offset * 1000.0);
    assert!(offset.abs() <= 1.0 / 60.0 + 0.002, "{offset}"); // out with the picture, not when the file was read ahead
    let total: usize = o.pushed.iter().map(|(_, p)| p.len()).sum();
    let total = total as f64 / 2.0 / 48000.0;
    assert!((1.95..2.01).contains(&total), "{total} s of sound");
    // the Python's, on the same file
    if let Some(m) = common::manifest() {
        let py = m["onset"]["python_offset_s"].as_f64().unwrap();
        println!("(the Python's: {:+.1} ms)", py * 1000.0);
        assert!((offset - py).abs() <= 1.0 / 60.0 + 0.004);
    }
    // the same sound kept for instant replay
    drop(o);
    let replay = src.read_sound();
    let n: usize = replay.iter().map(|s| s.pcm.len()).sum();
    assert_eq!(n as f64 / 2.0 / 48000.0, total);
    assert!(replay.windows(2).all(|w| w[1].t >= w[0].t) && replay.iter().all(|s| s.rate == 48000 && s.channels == 2));
    assert!(src.read_sound().is_empty());
}

#[test]
fn no_sound_wanted_or_none_there() {
    let Some(path) = common::fixture("onset.mov") else { return };
    let mut src = CardSource::video_file(&path, CardOptions::default()).unwrap(); // audio None: the picture only
    assert!(!src.can_delay_audio());
    assert_eq!(ticks_of(&mut src).len(), 120);
    assert!(src.read_sound().is_empty());
    let Some(silent) = common::fixture("silent.mov") else { return };
    let out = Arc::new(Mutex::new(Out::default()));
    let mut src = CardSource::video_file(&silent, CardOptions { audio: Some(recorder(&out)), ..Default::default() }).unwrap();
    assert!(!src.can_delay_audio()); // nothing to play
    assert_eq!(ticks_of(&mut src).len(), 12);
    assert!(out.lock().unwrap().pushed.is_empty());
}

#[test]
fn the_capture_log_reports_once_a_minute_only_when_something_happened() {
    let Some(path) = common::fixture("silent.mov").and(common::fixture("onset.mov")) else { return };
    let out = Arc::new(Mutex::new(Out { stats: Some(SoundStats { longest_wait_ms: 21, kept_ms: 31, ..Default::default() }), ..Default::default() }));
    let mut src = CardSource::video_file(&path, CardOptions { audio: Some(recorder(&out)), ..Default::default() }).unwrap();
    // a second in: sound has been played, and the first report found nothing to say
    for _ in 0..60 {
        src.next_event();
    }
    assert!(!src.messages().iter().any(|l| l.contains("sound, last minute")));
    out.lock().unwrap().stats.as_mut().unwrap().ran_dry = 2;
    src.report_sound();
    src.report_sound();
    let said = src.messages().join("\n");
    assert_eq!(said.matches("[capture] sound, last minute: ran dry 2 times").count(), 1, "{said}");
    src.close();
    assert!(src.messages().iter().any(|l| l.starts_with("[capture] sound this session: ran dry 2 times")));
}

/// The recording standing in, frame for frame against the Python: PyAV's decode read with capture.py raw_to_bgr in
/// the file's own colours (within a level), and the Python stand-in's own frames (the same picture but where swscale
/// resamples chroma at colour edges). Played at its own rate: 12 s.
#[test]
fn a_recording_stands_in_through_the_raw_path() {
    let Some(m) = common::manifest() else { return };
    let clip = common::repo().join(m["clip"]["path"].as_str().unwrap());
    let refs: Vec<u64> = m["clip"]["frames"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let (w, h) = (m["clip"]["w"].as_u64().unwrap() as usize, m["clip"]["h"].as_u64().unwrap() as usize);
    let out = Arc::new(Mutex::new(Out::default()));
    // (a deep queue, so the frames compared aren't dropped while the ones before are read back)
    let opts = CardOptions { audio: Some(recorder(&out)), queue_size: 16, ..Default::default() };
    let mut src = CardSource::video_file(&clip, opts).unwrap();
    assert_eq!((src.width(), src.height(), src.fps(), src.fmt(), src.raw()), (w, h, 60.0, "nv12", true));
    assert_eq!((src.bt709(), src.full_range()), (m["clip"]["bt709"].as_bool().unwrap(), m["clip"]["full_range"].as_bool().unwrap()));
    let (mut n, mut last_seq, mut first, mut last) = (0u64, None, None, 0.0);
    let mut raw: Vec<(u64, Vec<u8>)> = Vec::new();
    while let Some(ev) = src.next_event() {
        let Event::Tick(t) = ev else { continue };
        let s = seq(&t);
        assert!(last_seq.is_none_or(|l| s > l)); // (a gap only where the queue dropped one)
        last_seq = Some(s);
        assert!((t.t_source.unwrap() - s as f64 / 60.0).abs() < 1e-6, "{:?}", t.t_source);
        first.get_or_insert(t.t_arrival);
        last = t.t_arrival;
        if refs.contains(&s) {
            raw.push((s, t.frame.read_raw())); // (only these kept: the converter reuses the rest)
        }
        n += 1;
    }
    let dropped = src.dropped();
    src.close();
    assert_eq!(last_seq, Some(719));
    assert_eq!(n + dropped, 720);
    let fps = 719.0 / (last - first.unwrap());
    assert!((fps - 60.0).abs() < 0.2, "played at {fps} fps");
    println!("720 frames at {fps:.3} fps, {dropped} dropped");
    assert_eq!(raw.len(), refs.len(), "a frame compared was dropped");
    let got: Vec<(u64, Vec<u8>)> =
        raw.into_iter().map(|(i, r)| (i, r.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect())).collect();
    for (i, frame) in &got {
        let py = std::fs::read(common::fixtures().join(format!("clip_{i}.bgr"))).unwrap();
        let worst = frame.iter().zip(&py).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        let off = frame.iter().zip(&py).filter(|(a, b)| a != b).count() as f64 / py.len() as f64;
        let stand = std::fs::read(common::fixtures().join(format!("clip_{i}.standin.bgr"))).unwrap();
        let mean_s = frame.iter().zip(&stand).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / stand.len() as f64;
        println!("frame {i}: within {worst} of PyAV + raw_to_bgr ({:.3}% of values off by one); {mean_s:.3} on average from the Python stand-in's", off * 100.0);
        assert!(worst <= 1, "frame {i}: {worst} levels from the Python");
        assert!(mean_s < 1.0, "frame {i}: {mean_s} on average from the Python stand-in");
    }
    // the clip's sound: all of it, out with the picture from the first frame on
    let o = out.lock().unwrap();
    let total: usize = o.pushed.iter().map(|(_, p)| p.len()).sum();
    let secs = total as f64 / o.channels as f64 / o.rate as f64;
    assert!((11.9..12.01).contains(&secs), "{secs} s of sound");
    assert!((o.pushed[0].0 - first.unwrap()).abs() < 0.01);
}
