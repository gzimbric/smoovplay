//! Shared by sp-replay's tests: temp folders, a recorder that collects its messages, ffprobe and ffmpeg.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use sp_replay::{Options, Recorder};

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf()
}

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/replay")
}

pub fn python() -> Option<PathBuf> {
    let p = repo().join(".venv/bin/python");
    p.exists().then_some(p)
}

/// A new, empty folder for one test (removed when dropped).
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("sp-replay-test-{name}-{}-{}", std::process::id(), sp_core::clock::now_ns()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if std::env::var_os("SP_REPLAY_KEEP").is_none() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// What a recorder said: its toasts and its log.
#[derive(Clone, Default)]
pub struct Said {
    pub toasts: Arc<Mutex<Vec<String>>>,
    pub log: Arc<Mutex<Vec<String>>>,
}

impl Said {
    pub fn toasts(&self) -> Vec<String> {
        self.toasts.lock().unwrap().clone()
    }
    pub fn log(&self) -> String {
        self.log.lock().unwrap().join("\n")
    }
}

pub fn compose_bin() -> std::ffi::OsString {
    env!("CARGO_BIN_EXE_sp-replay-compose").into()
}

/// The before & after maker; with SP_REPLAY_COMPOSE_LOG set, its stderr goes to that file (the recorder only shows it
/// when it fails).
pub fn compose_cmd() -> Vec<std::ffi::OsString> {
    match std::env::var("SP_REPLAY_COMPOSE_LOG") {
        Ok(log) => vec!["/bin/sh".into(), "-c".into(), format!("exec \"$0\" \"$@\" 2>>'{log}'").into(), compose_bin()],
        Err(_) => vec![compose_bin()],
    }
}

/// replay.py's test fixture: 3 s kept, waiting for the encoder rather than dropping frames, clips in `dir`.
pub fn recorder(dir: &Path, length: u32) -> (Recorder, Said) {
    let said = Said::default();
    let (t, l) = (said.toasts.clone(), said.log.clone());
    let r = Recorder::new(Options {
        length_s: length,
        out_dir: Some(dir.to_path_buf()),
        drop_when_behind: false,
        notify: Some(Arc::new(move |s: &str, _| t.lock().unwrap().push(s.to_string()))),
        log: Some(Arc::new(move |s: &str| {
            println!("{s}");
            l.lock().unwrap().push(s.to_string())
        })),
        compose_cmd: Some(compose_cmd()),
        ..Default::default()
    });
    (r, said)
}

/// A stream as ffprobe sees it.
#[derive(Clone, Debug, Default)]
pub struct Stream {
    pub frames: u64,
    pub duration: f64,
    pub start: f64,
    pub width: u64,
    pub height: u64,
    pub fps: f64,
    pub profile: String,
}

pub fn ffprobe(path: &Path) -> serde_json::Value {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_streams", "-show_format", "-of", "json"])
        .arg(path)
        .output()
        .expect("ffprobe");
    assert!(out.status.success(), "ffprobe {}: {}", path.display(), String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
}

fn num(v: &serde_json::Value) -> f64 {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()).unwrap_or(0.0)
}

fn rate(v: &serde_json::Value) -> f64 {
    let s = v.as_str().unwrap_or("0/1");
    let (a, b) = s.split_once('/').unwrap_or((s, "1"));
    let (a, b): (f64, f64) = (a.parse().unwrap_or(0.0), b.parse().unwrap_or(1.0));
    if b == 0.0 { 0.0 } else { a / b }
}

/// (video, audio) streams.
pub fn probe(path: &Path) -> (Option<Stream>, Option<Stream>) {
    let j = ffprobe(path);
    let mut v = None;
    let mut a = None;
    for s in j["streams"].as_array().unwrap() {
        let st = Stream {
            frames: num(&s["nb_frames"]) as u64,
            duration: num(&s["duration"]),
            start: num(&s["start_time"]),
            width: s["width"].as_u64().unwrap_or(0),
            height: s["height"].as_u64().unwrap_or(0),
            fps: rate(&s["avg_frame_rate"]),
            profile: s["profile"].as_str().unwrap_or("").to_string(),
        };
        match s["codec_type"].as_str() {
            Some("video") => v = Some(st),
            Some("audio") => a = Some(st),
            _ => {}
        }
    }
    (v, a)
}

/// Each video frame's (pts seconds, keyframe), from ffprobe.
pub fn frames(path: &Path) -> Vec<(f64, bool)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time,key_frame", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .expect("ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut it = l.split(',');
            let a = it.next()?.trim();
            let b = it.next()?.trim();
            let (k, t) = if a == "0" || a == "1" { (a, b) } else { (b, a) };
            Some((t.parse().ok()?, k == "1"))
        })
        .collect()
}

/// Every frame of a video as packed BGR (FFmpeg's conversion, honouring the file's colour tags).
pub fn decode_bgr(path: &Path, w: usize, h: usize) -> Vec<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        .output()
        .expect("ffmpeg");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out.stdout.chunks_exact(w * h * 3).map(|c| c.to_vec()).collect()
}

/// A video's sound as mono f32 at 48 kHz, from its start (FFmpeg, edit lists applied).
pub fn decode_audio(path: &Path) -> Vec<f32> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"])
        .output()
        .expect("ffmpeg");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out.stdout.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// A small deterministic random source.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}
