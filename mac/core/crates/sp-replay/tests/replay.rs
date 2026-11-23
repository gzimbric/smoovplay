//! tests/test_replay.py's checks on the Rust recorder: the tracks it keeps, what it saves, the before & after video,
//! the sound's timing, the pipeline's side, the settings' hints. Small frames (160x96) on the media engine.

mod common;

use std::sync::Arc;
use std::time::Instant;

use common::*;
use sp_core::{Frame, Sound};
use sp_replay::sound::SoundClock;
use sp_replay::{Kind, Layout, Recorder, compose_main, yuv};

const W: usize = 160;
const H: usize = 96;
const SR: u32 = 48_000;

fn img(k: usize, bgr: [u8; 3]) -> Frame {
    let mut a = Vec::with_capacity(W * H * 3);
    let x0 = (k * 3) % W;
    for _y in 0..H {
        for x in 0..W {
            if x >= x0 && x < x0 + 8 { a.extend_from_slice(&[255, 255, 255]) } else { a.extend_from_slice(&bgr) } // a bar that moves
        }
    }
    Frame::from_bgr(&a, W, H)
}

fn tone(k: usize, n: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let s = (i + k * n) as f64;
        let v = ((2.0 * std::f64::consts::PI * 440.0 * s / SR as f64).sin() * 8000.0) as i16;
        out.push(v);
        out.push(v);
    }
    out
}

/// Plays for `seconds`, with a frame shown at `hz`, a new game frame every third and 100 ms of sound every 0.1 s.
fn feed(rec: &Recorder, seconds: f64, hz: usize, t0: f64, jitter: f64, seed: u64) {
    let mut rng = Rng::new(seed);
    rec.set_delay(40.0);
    let green = [40, 160, 40];
    let blue = [160, 40, 40];
    for i in 0..(seconds * hz as f64).round() as usize {
        let t = t0 + i as f64 / hz as f64;
        rec.frame(&img(i, green), t);
        if i % 3 == 0 {
            rec.real_frame(&img(i, blue), t - 0.04);
        }
        if i % (hz / 10) == 0 {
            let k = i / (hz / 10);
            rec.add_sound(vec![Sound {
                t: t0 + k as f64 * 0.1 - 0.04 + rng.uniform(0.0, jitter),
                pcm: tone(k, SR as usize / 10),
                rate: SR,
                channels: 2,
            }]);
        }
    }
    assert!(rec.flush(30.0));
}

#[test]
fn yuv_is_bt709_limited_range() {
    for (bgr, want) in [
        ([255, 255, 255], [235, 128, 128]),
        ([0, 0, 0], [16, 128, 128]),
        ([0, 0, 255], [63, 102, 240]),
        ([255, 0, 0], [32, 240, 118]),
    ] {
        assert_eq!(yuv::yuv_of(bgr), want, "{bgr:?}");
    }
}

#[test]
fn saves_what_was_shown_with_the_sound() {
    let tmp = TempDir::new("saves");
    let (rec, said) = recorder(tmp.path(), 3);
    feed(&rec, 6.0, 90, 1000.0, 0.02, 0);
    let paths = rec.save(None, None, true);
    assert!(paths.len() == 1 && paths[0].parent() == Some(tmp.path()) && paths[0].extension().unwrap() == "mp4", "{paths:?}");
    let (v, a) = probe(&paths[0]);
    let (v, a) = (v.unwrap(), a.expect("no sound in the clip"));
    // the last 3 s from the keyframe before, so 3 to 4 s, with every frame shown
    assert!((3.0..=4.1).contains(&v.duration) && v.width == W as u64, "{v:?}");
    assert!((v.frames as f64 / v.duration - 90.0).abs() < 2.0, "{v:?}");
    assert!((a.duration - v.duration).abs() < 0.15, "sound {a:?} video {v:?}"); // the sound covers the clip
    assert!(said.toasts().iter().any(|t| t.starts_with("Saved the last")), "{:?}", said.toasts());
    // A keyframe every second, the clip starting on one
    let f = frames(&paths[0]);
    assert!(f[0].1, "the clip doesn't start on a keyframe");
    let keys: Vec<f64> = f.iter().filter(|x| x.1).map(|x| x.0).collect();
    for k in keys.windows(2) {
        assert!(k[1] - k[0] <= 1.0 + 0.012, "keyframes {keys:?}");
    }
    rec.close(0.0);
}

#[test]
fn keeps_only_the_length_asked_for() {
    let tmp = TempDir::new("keeps");
    let (rec, _said) = recorder(tmp.path(), 3);
    feed(&rec, 9.0, 90, 1000.0, 0.02, 0);
    let held = rec.held();
    let limit = rec.length() as f64 + sp_replay::KEEP_EXTRA_S + sp_replay::KEY_EVERY_S + 0.1;
    assert!(held.all().iter().all(|(secs, _)| *secs <= limit), "{held:?}");
    assert!(held.after.0 >= rec.length() as f64, "{held:?}");
    rec.set_length(0); // off, so everything goes
    assert!(rec.flush(5.0) && held_nothing(&rec), "{:?}", rec.held());
    rec.close(0.0);
}

fn held_nothing(rec: &Recorder) -> bool {
    rec.held().all().iter().all(|(_, mb)| *mb == 0.0)
}

/// Mean BGR of rows y0..y1, columns x0..x1 of a packed BGR picture W wide.
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
fn before_and_after_split_and_side_by_side() {
    let tmp = TempDir::new("compare");
    let (rec, said) = recorder(tmp.path(), 3);
    feed(&rec, 4.0, 90, 1000.0, 0.02, 0);
    let paths = rec.save(Some(Kind::Both), None, true);
    assert!(paths.len() == 2 && paths[1].to_string_lossy().contains("before and after"), "{paths:?}\n{}", said.log());
    let (v, a) = probe(&paths[1]);
    let v = v.unwrap();
    let (v0, _) = probe(&paths[0]);
    assert!(v.width == W as u64 && (v.duration - v0.unwrap().duration).abs() < 0.05 && a.is_some(), "{v:?}");
    let pics = decode_bgr(&paths[1], W, H);
    let img = &pics[30];
    let (left, right) = (mean(img, W, (60, 90), (4, 30)), mean(img, W, (60, 90), (W - 30, W - 4)));
    assert!(left[0] > 120.0 && left[1] < 80.0, "left {left:?}"); // the game's own frames (blue) on the left
    assert!(right[1] > 120.0 && right[0] < 80.0, "right {right:?}"); // SmoovPlay's (green) on the right
    let side = rec.save(Some(Kind::Compare), Some(Layout::Side), true);
    assert!(side.len() == 1, "{side:?}\n{}", said.log());
    assert_eq!(probe(&side[0]).0.unwrap().width, 2 * W as u64);
    let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.ends_with(".part") || n.ends_with(".ts")
        })
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    rec.close(0.0);
}

#[test]
fn labels_name_the_frame_rates() {
    let tmp = TempDir::new("labels");
    let (rec, said) = recorder(tmp.path(), 3);
    let args = tmp.path().join("args.txt");
    // Stands in for the before & after maker, noting what it was asked
    rec.set_compose_cmd(Some(vec![
        "/bin/sh".into(),
        "-c".into(),
        format!("printf '%s\\n' \"$@\" > '{}'; exec \"$0\" \"$@\"", args.display()).into(),
        compose_bin(),
    ]));
    feed(&rec, 3.0, 90, 1000.0, 0.02, 0);
    let saved = rec.save(Some(Kind::Compare), None, true);
    assert_eq!(saved.len(), 1, "{}", said.log());
    let seen: Vec<String> = std::fs::read_to_string(&args).unwrap().lines().map(String::from).collect();
    let i = seen.iter().position(|a| a == "--labels").unwrap();
    assert_eq!(seen[i + 1..i + 3], ["Original 30 fps", "SmoovPlay 90 fps"]);
    rec.close(0.0);
}

/// On Esc the clip is finished first, but a before & after video carries on in its own process, so the video window
/// never waits on it.
#[test]
fn stopping_waits_for_the_clip_but_not_the_before_and_after() {
    let tmp = TempDir::new("stopping");
    let (rec, said) = recorder(tmp.path(), 3);
    rec.set_compose_cmd(Some(vec!["/bin/sh".into(), "-c".into(), "sleep 4".into(), "sh".into()]));
    feed(&rec, 3.0, 90, 1000.0, 0.02, 0);
    rec.save(Some(Kind::Both), None, false);
    let t0 = Instant::now();
    while rec.making().is_empty() && t0.elapsed().as_secs_f64() < 10.0 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let pids = rec.making();
    let t1 = Instant::now();
    rec.close(30.0);
    let took = t1.elapsed().as_secs_f64();
    assert!(!pids.is_empty() && took < 2.5, "took {took:.2} s, {pids:?}");
    let alive = unsafe { libc_kill(pids[0] as i32, 0) } == 0;
    assert!(alive, "the before & after process stopped");
    let clips: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with("SmoovPlay ") && n.ends_with(".mp4")
        })
        .collect();
    assert_eq!(clips.len(), 1); // the clip itself is there
    assert!(said.log().contains("still being made: they carry on in the background"), "{}", said.log());
    std::thread::sleep(std::time::Duration::from_secs(4));
}

unsafe extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

#[test]
fn the_before_and_after_process_leaves_word_for_the_launcher() {
    let tmp = TempDir::new("mark");
    let mark = tmp.path().join("saving");
    let out = tmp.path().join("x before and after.mp4");
    let none = tmp.path().join("none.ts").display().to_string();
    let code = compose_main(&[
        "--after".into(),
        none.clone(),
        "--before".into(),
        none,
        "--out".into(),
        out.display().to_string(),
        "--mark".into(),
        mark.display().to_string(),
    ]);
    assert_ne!(code, 0);
    let files: Vec<_> = std::fs::read_dir(&mark).unwrap().flatten().map(|e| e.path()).collect();
    assert!(files.len() == 1 && files[0].file_stem().unwrap().to_string_lossy() == std::process::id().to_string(), "{files:?}");
    let text = std::fs::read_to_string(&files[0]).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[0] == out.display().to_string() && lines[1].starts_with("failed: "), "{lines:?}");
    assert!(!out.exists() && !tmp.path().join(format!("{}.part", out.file_name().unwrap().to_string_lossy())).exists());
}

#[test]
fn a_finished_before_and_after_tidies_its_word_away() {
    let tmp = TempDir::new("tidies");
    let (rec, said) = recorder(tmp.path(), 3);
    let mark = tmp.path().join("saving");
    rec.set_mark_dir(Some(mark.clone()));
    feed(&rec, 3.0, 90, 1000.0, 0.02, 0);
    let paths = rec.save(Some(Kind::Compare), None, true);
    assert!(paths.len() == 1 && paths[0].exists(), "{paths:?}\n{}", said.log());
    assert_eq!(std::fs::read_dir(&mark).map(|d| d.count()).unwrap_or(0), 0);
    rec.close(0.0);
}

#[test]
fn off_says_so_and_takes_nothing() {
    let tmp = TempDir::new("off");
    let (r, said) = recorder(tmp.path(), 0);
    r.frame(&img(0, [0, 0, 0]), 1.0);
    assert!(r.save(None, None, true).is_empty() && said.toasts().last().unwrap().contains("off"));
    assert!(held_nothing(&r) && std::fs::read_dir(tmp.path()).unwrap().count() == 0);
    r.close(0.0);
}

#[test]
fn nothing_yet() {
    let tmp = TempDir::new("nothing");
    let (r, said) = recorder(tmp.path(), 3);
    assert!(r.save(None, None, true).is_empty() && said.toasts().last().unwrap().contains("nothing"), "{:?}", said.toasts());
    r.close(0.0);
}

// ---- the sound's timing. Arrival times jitter and the sound plays on, but gaps and running ahead are put right.
fn place(c: &mut SoundClock, t0: &mut Option<f64>, t: f64, delay: f64) {
    c.place(t, 480, SR, delay, t0);
}

#[test]
fn jitter_doesnt_move_the_sound() {
    let mut c = SoundClock::new();
    let mut t0 = None;
    let mut rng = Rng::new(2);
    for k in 0..300 {
        // 3 s of 10 ms chunks, each up to 40 ms late
        place(&mut c, &mut t0, 100.0 + k as f64 * 0.01 + rng.uniform(0.0, 0.04), 0.0);
    }
    // from the clock's zero, 5 s before the first
    let written = (c.a_pts - sp_replay::OFFSET * SR as i64 / sp_replay::TB) as f64 / SR as f64;
    assert!((written - 8.0).abs() < 0.03, "{written}"); // exactly the sound given, with nothing put in or skipped
}

#[test]
fn a_gap_becomes_silence_and_running_ahead_is_skipped() {
    let mut c = SoundClock::new();
    let mut t0 = None;
    let t = 100.0;
    for k in 0..100 {
        place(&mut c, &mut t0, t + k as f64 * 0.01, 0.2);
    }
    let t_gap = t + 1.5; // half a second with no sound
    for k in 0..50 {
        place(&mut c, &mut t0, t_gap + k as f64 * 0.01, 0.2);
    }
    let a = c.a_next.unwrap();
    assert!((a - (t_gap + 0.5 + 0.2)).abs() < 0.02, "{a}"); // silence put in, then the sound on time
    // SmoovPlay's delay drops by 100 ms and stays there. After a second the sound skips 100 ms to get back in step.
    for k in 0..150 {
        place(&mut c, &mut t0, t_gap + 0.5 + k as f64 * 0.01, 0.1);
    }
    let a = c.a_next.unwrap();
    assert!((a - (t_gap + 0.5 + 1.5 + 0.1)).abs() < 0.02, "{a}");
}

// ---- the pipeline's side, which reaches the recorder through sp_sched::Replay
#[test]
fn the_pipeline_feeds_it_through_the_replay_trait() {
    let tmp = TempDir::new("trait");
    let (rec, said) = recorder(tmp.path(), 3);
    rec.set_compose_cmd(Some(vec!["/bin/true".into()]));
    {
        let mut r: &Recorder = &rec;
        let replay: &mut dyn sp_sched::Replay = &mut r;
        replay.set_output_hz(120);
        replay.set_delay(40.0);
        for i in 0..240 {
            let t = 1000.0 + i as f64 / 120.0;
            replay.frame(&img(i, [40, 160, 40]), t);
            if i % 4 == 0 {
                replay.real_frame(&img(i, [160, 40, 40]), t - 0.04);
            }
        }
    }
    assert!(rec.flush(10.0));
    // At 120 Hz every other frame is kept, for 60 fps clips
    let paths = rec.save(Some(Kind::Smoovplay), None, true);
    let v = probe(&paths[0]).0.unwrap();
    assert!((v.frames as f64 / v.duration - 60.0).abs() < 2.0, "{v:?}\n{}", said.log());
    rec.close(0.0);
}

// ---- settings
#[test]
fn settings_and_their_memory_hints() {
    let s = sp_settings::settings::spec("replay").unwrap();
    assert_eq!(s.group, "Instant replay");
    for secs in &sp_replay::LENGTHS[1..] {
        let hint = s.hint(&sp_settings::Value::from(*secs));
        let after = hint.split("about").nth(1).unwrap().trim();
        let (n, unit) = after.split_once('\u{a0}').unwrap();
        let mb: f64 = n.parse::<f64>().unwrap() * if unit.starts_with("GB") { 1000.0 } else { 1.0 };
        let want = sp_replay::memory_mb_1080p90(*secs as f64);
        assert!((mb - want).abs() / want < 0.05, "{secs}: {hint} vs {want:.0} MB");
    }
}

#[test]
fn durations_and_names() {
    use sp_replay::{dur, snap_fps};
    assert!(dur(30.0) == "30 s" && dur(60.0) == "60 s" && dur(120.0) == "2 min");
    assert!(dur(300.0) == "5 min" && dur(150.0) == "2.5 min");
    assert!(snap_fps(29.6) == 30 && snap_fps(88.9) == 90 && snap_fps(37.0) == 37);
    assert!(sp_replay::stamp().len() == 19);
    let _ = Arc::new(());
}
