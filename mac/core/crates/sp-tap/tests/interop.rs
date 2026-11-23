//! sp-tap against the other two implementations of the frame tap: framegen/chiaki_tap.py (through
//! mac/core/tools/media_tap_peer.py) and chiaki_tap/frametap.c (the writer inside chiaki-ng, built with clang as
//! chiaki_tap/selftest.py builds it). Each test uses a tap name of its own and removes it afterwards.
//!
//! Needs the repo's Python environment (.venv) for the Python tests and clang for the C one; a test skips (passes
//! with a note) when its tool is missing.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use sp_tap::*;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf()
}

fn python() -> Option<PathBuf> {
    let p = repo().join(".venv/bin/python");
    p.exists().then_some(p)
}

fn peer() -> PathBuf {
    repo().join("mac/core/tools/media_tap_peer.py")
}

fn name(tag: &str) -> String {
    format!("Local\\sp-rs-{tag}{}", std::process::id() % 100000)
}

fn frame(i: usize) -> Vec<u8> {
    let size = if i % 2 == 1 { 1000 + i } else { 50000 + i * 1000 };
    (0..size).map(|j| ((i * 31 + j) & 0xFF) as u8).collect()
}

fn chunk(k: usize) -> Vec<u8> {
    (0..if k < 20 { 1920 } else { 20000 }).map(|j| ((k * 7 + j) & 0xFF) as u8).collect()
}

const FRAMES: usize = 24;

#[test]
fn rust_writer_python_reader() {
    let Some(py) = python() else { return eprintln!("skipped: no .venv") };
    let n = name("rw");
    unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    let sound = AudioTap::open(&n).unwrap();
    sound.begin(48000, 2);
    tap.begin(CODEC_H265, 1920, 1080, 60, 0, Some(FEATURE_PRESS | FEATURE_AUDIO_DELAY | FEATURE_AUDIO_TAP)).unwrap();
    for k in 0..21 {
        sound.push(&chunk(k), None);
    }
    for i in 0..FRAMES {
        tap.push(&frame(i), if i == 7 { 2 } else { 0 }, i == 8, Some(i % 2 == 0));
    }
    let out = Command::new(py).arg(peer()).args(["read", &n]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success() && text.starts_with("OK"), "{text}");
    // What the Python reader asked for, the other way
    assert!(tap.take_idr_request(), "no keyframe request");
    assert_eq!(tap.take_press_request(), Some((BUTTON_OPTIONS, 120)));
    assert_eq!(tap.audio_delay_ms(), 42);
    let beat = tap.shm().u64(OFF_HEARTBEAT);
    assert!(clock::qpc().abs_diff(beat) < 5_000_000_000, "heartbeat {beat} not from just now");
    tap.end();
    sound.end();
    drop((tap, sound));
    unlink(&n);
}

#[test]
fn python_writer_rust_reader() {
    let Some(py) = python() else { return eprintln!("skipped: no .venv") };
    let n = name("pw");
    unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    tap.end();
    let mut sound = AudioTap::open(&n).unwrap();
    let child = Command::new(py).arg(peer()).args(["write", &n]).stdout(Stdio::piped()).spawn().unwrap();
    // Read each frame as it comes, as ChiakiSource does
    let t_end = clock::now() + 15.0;
    let mut h = tap.header();
    for i in 0..FRAMES as u64 {
        while clock::now() < t_end && (!h.live() || h.write_seq <= i) {
            tap.wait(h.watch(), 0.1);
            h = tap.header();
        }
        assert!(h.live() && (h.version, h.codec, h.width, h.height, h.max_fps) == (1, 1, 1920, 1080, 60), "{h:?}");
        let p = tap.read(i, &h).unwrap_or_else(|| panic!("frame {i} gone"));
        assert_eq!(p.data, frame(i as usize), "frame {i}");
        assert_eq!((p.frames_lost, p.recovered, p.truncated), (if i == 7 { 2 } else { 0 }, i == 8, false), "{i}");
        assert_eq!(p.flags, if i % 2 == 0 { FLAG_NEW } else { FLAG_REPEAT });
        assert!((0.0..5.0).contains(&(clock::now() - p.t_arrival)), "frame {i}: arrival time off");
    }
    assert_eq!(tap.features(), 7);
    sound.reset(true);
    let chunks = sound.read_new();
    assert_eq!(chunks.iter().flat_map(|c| c.pcm.clone()).collect::<Vec<_>>(), (0..21).flat_map(chunk).collect::<Vec<_>>());
    assert!(chunks.len() == 23 && chunks.iter().all(|c| (c.rate, c.channels) == (48000, 2)));
    assert!(((chunks[22].t - chunks[21].t) - 2040.0 / 48000.0).abs() < 1e-3);
    // Asks the other way: a keyframe, a press (timed), the sound delay
    tap.request_idr();
    let t_ask = clock::now();
    let token = tap.press(BUTTON_OPTIONS, 120);
    let sent = loop {
        if let Some(s) = tap.press_sent(token, h.qpc_freq) {
            break s;
        }
        assert!(clock::now() - t_ask < 5.0, "press not sent");
        std::thread::sleep(Duration::from_millis(2));
    };
    assert!(t_ask < sent && sent < clock::now(), "press time off");
    tap.set_audio_delay(42);
    tap.heartbeat();
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success() && text.contains("press 4096 for 120 ms; audio delay 42 ms"), "{text}");
    assert!(!tap.header().live() && !sound.live(), "end not marked");
    drop((tap, sound));
    unlink(&n);
}

/// chiaki_tap/frametap.c (built as chiaki_tap/selftest.py builds it) to this reader: the checks selftest.py makes.
#[test]
fn c_writer_rust_reader() {
    let clang = ["/usr/bin/clang", "/usr/bin/cc"].into_iter().find(|p| Path::new(p).exists());
    let Some(clang) = clang else { return eprintln!("skipped: no clang") };
    let dir = std::env::temp_dir().join(format!("sp-tap-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("frametap_selftest");
    let src = repo().join("chiaki_tap");
    let st = Command::new(clang)
        .args(["-O2", "-Wall", "-o"])
        .arg(&exe)
        .arg(src.join("selftest.c"))
        .arg(src.join("frametap.c"))
        .status()
        .unwrap();
    assert!(st.success(), "clang failed");
    let n = name("c");
    unlink(&n);
    let tap = FrameTap::open(&n).unwrap();
    tap.end();
    let mut sound = AudioTap::open(&n).unwrap();
    sound.end();
    let child = Command::new(&exe)
        .env("CHIAKI_FRAMETAP_NAME", &n)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let t_end = clock::now() + 10.0;
    let mut ages = Vec::new();
    let mut h = tap.header();
    for i in 0..40u64 {
        while clock::now() < t_end && (!h.live() || h.write_seq <= i) {
            tap.wait(h.watch(), 0.1);
            h = tap.header();
        }
        assert!(h.live() && (h.version, h.codec, h.width, h.height, h.max_fps) == (1, 1, 1920, 1080, 60), "{h:?}");
        assert_eq!(h.qpc_freq, clock::qpc_freq());
        let p = tap.read(i, &h).unwrap_or_else(|| panic!("frame {i} gone"));
        ages.push(clock::now() - p.t_arrival);
        let size = if i % 2 == 1 { 1000 + i as usize } else { 50000 + i as usize * 1000 };
        let want: Vec<u8> = (0..size).map(|j| ((i as usize * 31 + j) & 0xFF) as u8).collect();
        assert_eq!(p.data, want, "frame {i}: data mismatch");
        assert_eq!((p.frames_lost, p.recovered, p.truncated), (if i == 17 { 2 } else { 0 }, i == 18, false), "{i}");
        assert!((0.0..5.0).contains(&(clock::now() - p.t_arrival)), "frame {i}: arrival time off");
    }
    tap.request_idr();
    assert!(sound.live(), "sound ring not started");
    sound.reset(true);
    let chunks = sound.read_new();
    let want: Vec<u8> =
        (0..21usize).flat_map(|k| (0..if k < 20 { 1920 } else { 20000 }).map(move |j| ((k * 7 + j) & 0xFF) as u8)).collect();
    assert_eq!(chunks.iter().flat_map(|c| c.pcm.clone()).collect::<Vec<_>>(), want, "sound data");
    assert!(chunks.len() == 23 && chunks.iter().all(|c| (c.rate, c.channels) == (48000, 2)));
    assert!(chunks.windows(2).all(|w| w[1].t >= w[0].t), "sound times go backwards");
    assert!(((chunks[22].t - chunks[21].t) - 2040.0 / 48000.0).abs() < 1e-3, "split chunk timing");
    assert_eq!(tap.features(), FEATURE_PRESS | FEATURE_AUDIO_DELAY | FEATURE_AUDIO_TAP);
    let t_ask = clock::now();
    let token = tap.press(BUTTON_OPTIONS, 120);
    let sent = loop {
        if let Some(s) = tap.press_sent(token, h.qpc_freq) {
            break s;
        }
        assert!(clock::now() - t_ask < 5.0, "press not sent");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(t_ask < sent && sent < clock::now(), "press time off");
    tap.set_audio_delay(42);
    for _ in 0..20 {
        tap.heartbeat();
        std::thread::sleep(Duration::from_millis(50));
    }
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "{text}{}", String::from_utf8_lossy(&out.stderr));
    assert!(text.contains(&format!("press {BUTTON_OPTIONS} for 120 ms sent; audio delay 42 ms")), "{text}");
    assert!(text.contains("audio delay lapsed: yes"), "{text}");
    assert!(!tap.header().live() && !sound.live(), "end not marked");
    ages.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("C writer: 40 frames and 21 chunks byte-identical; read {:.2} ms (median) after stamping", ages[20] * 1e3);
    drop((tap, sound));
    unlink(&n);
    let _ = std::fs::remove_dir_all(dir);
}

/// Either side may create the memory first, or both at once: both always see the same memory.
#[test]
fn create_race() {
    // Threads in this process opening a fresh name at the same moment
    for k in 0..40 {
        let n = name(&format!("t{k}-"));
        unlink(&n);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let b2 = barrier.clone();
        let n2 = n.clone();
        let th = std::thread::spawn(move || {
            b2.wait();
            FrameTap::open(&n2).unwrap()
        });
        barrier.wait();
        let a = FrameTap::open(&n).unwrap();
        let b = th.join().unwrap();
        a.begin(CODEC_H264, 64, 64, 60, 0, None).unwrap();
        a.push(b"same memory", 0, false, None);
        let h = b.header();
        assert!(h.live() && b.read(0, &h).unwrap().data == b"same memory", "round {k}");
        drop((a, b));
        unlink(&n);
    }
    let Some(py) = python() else { return eprintln!("skipped the Python part: no .venv") };
    // Rust first, then Python; Python first, then Rust; both at once (a few times)
    for (k, mode) in ["rust-first", "python-first", "same", "same", "same", "same"].iter().enumerate() {
        let n = name(&format!("x{k}-"));
        unlink(&n);
        let start = clock::now() + if *mode == "same" { 0.8 } else { 0.0 };
        let mut held = None;
        if *mode == "rust-first" {
            held = Some(FrameTap::open(&n).unwrap());
        }
        let child = Command::new(&py)
            .arg(peer())
            .args(["race", &n, &format!("{start:.6}")])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        if *mode == "python-first" {
            std::thread::sleep(Duration::from_millis(600));
        } else if *mode == "same" {
            while clock::now() < start {}
        }
        let tap = held.unwrap_or_else(|| FrameTap::open(&n).unwrap());
        let t_end = clock::now() + 10.0;
        loop {
            let h = tap.header();
            if h.live() && h.write_seq >= 1 {
                assert_eq!(tap.read(0, &h).unwrap().data, b"from-python");
                break;
            }
            assert!(clock::now() < t_end, "{mode}: never saw Python's frame");
            tap.wait(h.watch(), 0.05);
        }
        tap.request_idr();
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "{mode}: {}", String::from_utf8_lossy(&out.stdout));
        drop(tap);
        unlink(&n);
    }
}
