//! Instant replay for SmoovPlay's macOS core (framegen/replay.py): the last 30 s to 5 min of what was shown, kept in
//! memory, saved as an MP4 in a second or two when asked; and the before & after video, made by a low-priority
//! process of its own.
//!
//! Three tracks are kept while it's on, a piece per encoded packet, so the oldest can be dropped and any stretch cut
//! out at a keyframe (one at least every [`KEY_EVERY_S`]):
//! - after: every frame SmoovPlay shows (every other one from 120 Hz), timed when it was shown;
//! - before: each new frame from the game, once, timed when it arrived plus SmoovPlay's delay;
//! - sound: the game's sound, timed the same way.
//!
//! How it's done on a Mac ([`Recorder`]):
//! - A frame is converted to the encoder's NV12 (BT.709, limited range: replay.py's sums) by a Metal kernel that
//!   writes straight into an IOSurface-backed pixel buffer from the VideoToolbox session's own pool (through a
//!   CVMetalTextureCache): no CPU copies. The kernel is queued on the core's shared Metal queue, after whatever drew
//!   the frame, and the pipeline's thread doesn't wait for it.
//! - The replay's own thread waits for the GPU and hands the buffer to a VTCompressionSession (H.264 on the media
//!   engine, real time, no B-frames, keyframes forced every second; replay.py's h264_videotoolbox settings), whose
//!   packets come back on VideoToolbox's thread and go into the ring.
//! - Sound is AAC (AudioToolbox), timed by replay.py's rules (jitter never moves it; gaps become silence; running
//!   ahead is skipped).
//! - Pieces are kept as the encoders give them (H.264 in AVCC form with its avcC, AAC frames with their
//!   AudioSpecificConfig), not as MPEG-TS: that's exactly what an MP4 holds, so saving writes them out as they are
//!   (our own muxer, [`mp4`], moov first), and nothing is packetised only to be unpacked again. It also saves TS's
//!   188-byte packet overhead (about 4% on video, a third on 430-byte AAC frames).
//! - HDR frames (RGBA16F, PQ) are copied on the GPU at once, then tone-mapped to SDR by sp-media's
//!   `Converter::pq16_to_8` ("sdr8"'s picture) on the replay's thread.
//! - Saving empties the encoders first, so a clip has every frame shown up to the press (FFmpeg's VideoToolbox
//!   wrapper, under the Python, still holds the last few).
//!
//! The before & after video ([`compose()`]) is `smoovplay replay-compose` (or this crate's `sp-replay-compose`) with
//! replay.py's `compose` arguments, at nice 10 and the utility QoS, writing replay.py's SAVING_DIR markers. Its
//! --after / --before / --sound files are MP4s the recorder writes (the tracks on one timeline), not TS.
//!
//! Using it from the app:
//! ```ignore
//! let rec = Arc::new(Recorder::new(Options { length_s: 60, size: Some((w, h)), mark_dir: Some(saving_dir()), ..Default::default() }));
//! rec.warm_up();                         // encoders open before play
//! rec.set_hdr(Some(hdr_settings));       // when the window shows HDR
//! let mut r: &Recorder = &rec;           // the pipeline: RunOptions { replay: Some(&mut r), .. }
//! rec.save(None, None, false);           // R from anywhere (B: Some(Kind::Compare))
//! rec.close(5.0);
//! ```

pub mod aac;
pub mod compose;
pub mod gpu;
pub mod label;
pub mod mp4;
pub mod recorder;
pub mod sound;
pub mod track;
pub mod vtenc;
pub mod yuv;

use std::path::{Path, PathBuf};

pub use compose::{ComposeArgs, compose, compose_main};
pub use recorder::{Held, Kind, Layout, Options, Recorder, Stats, Summary};

/// The 90 kHz clock every timestamp here is on (MPEG-TS's, as replay.py's).
pub const TB: i64 = 90_000;
/// Timestamps start 10 s in, so none is ever negative.
pub const OFFSET: i64 = 10 * TB;
/// settings.replay's choices, in seconds (0: off).
pub const LENGTHS: [u32; 5] = [0, 30, 60, 120, 300];
/// Kept beyond the length, so a cut can start at the keyframe before it.
pub const KEEP_EXTRA_S: f64 = 2.0;
/// A keyframe at least this often: where a saved clip can start.
pub const KEY_EVERY_S: f64 = 1.0;
/// Bits per pixel per frame, given for the encoder's nominal rate (replay.py's AFTER_BPP, BEFORE_BPP).
pub const AFTER_BPP: f64 = 0.14;
pub const BEFORE_BPP: f64 = 0.2;
pub const AFTER_NOMINAL: u32 = 120;
pub const BEFORE_NOMINAL: u32 = 60;
pub const MAX_BITRATE: f64 = 80_000_000.0;
pub const SOUND_RATE: u32 = 48_000;
pub const SOUND_BITRATE: u32 = 160_000;
/// Frames on their way to the encoder, per track (after, before); more and they're dropped.
pub const POOL: [usize; 2] = [10, 6];
/// MPEG-TS packet headers (replay.py keeps TS in memory; kept here so memory_mb is the settings' hint).
pub const TS_OVERHEAD: f64 = 1.04;
/// The line between before and after: amber (B, G, R).
pub const DIVIDER_BGR: [u8; 3] = [0, 176, 255];
/// The name replay.py gives the encoder it picks on a Mac (passed to the before & after process as --encoder).
pub const ENCODER: &str = "h264_videotoolbox";

/// Roughly how much memory a replay of this length holds, in MB (replay.py memory_mb; the settings' hints quote it
/// for 1080p at 90 fps).
pub fn memory_mb(length_s: f64, w: u32, h: u32, hz: u32, game_fps: f64) -> f64 {
    let bits = (AFTER_BPP * hz as f64 + BEFORE_BPP * game_fps) * w as f64 * h as f64 + SOUND_BITRATE as f64;
    bits * TS_OVERHEAD * (length_s + KEEP_EXTRA_S) / 8.0 / 1e6
}

/// memory_mb with replay.py's defaults (1080p, 90 Hz, a 30 fps game).
pub fn memory_mb_1080p90(length_s: f64) -> f64 {
    memory_mb(length_s, 1920, 1080, 90, 30.0)
}

/// Where clips go: SMOOVPLAY_REPLAY_DIR if set, else ~/Movies/SmoovPlay (replay.py videos_folder on a Mac).
pub fn videos_folder() -> PathBuf {
    if let Some(d) = std::env::var_os("SMOOVPLAY_REPLAY_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    sp_settings::paths::home().join("Movies").join("SmoovPlay")
}

/// Where a before & after video being made leaves word for the launcher (replay.py SAVING_DIR): `<pid>.txt` holding
/// the video's path.
pub fn saving_dir() -> PathBuf {
    sp_settings::paths::data_dir().join("saving")
}

/// `path`, or "name (2).mp4", "name (3).mp4", ... if it's taken.
pub fn unique(path: &Path) -> PathBuf {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = path.extension().map(|s| format!(".{}", s.to_string_lossy())).unwrap_or_default();
    let mut out = path.to_path_buf();
    let mut n = 2;
    while out.exists() {
        out = path.with_file_name(format!("{stem} ({n}){ext}"));
        n += 1;
    }
    out
}

/// "30 s", "2 min", "2.5 min" (replay.py _dur).
pub fn dur(secs: f64) -> String {
    if secs < 90.0 {
        return format!("{} s", py_round(secs));
    }
    let m = format!("{:.1}", secs / 60.0);
    let m = m.trim_end_matches('0').trim_end_matches('.');
    format!("{m} min")
}

/// Python's round(): halves to even.
pub fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The folder above, the folder and the name, for messages: "Movies/SmoovPlay/SmoovPlay ....mp4" (replay.py _short,
/// with a Mac's slashes).
pub fn short(path: &Path) -> String {
    let name = |p: Option<&Path>| p.and_then(|p| p.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let parent = path.parent();
    format!("{}/{}/{}", name(parent.and_then(|p| p.parent())), name(parent), name(Some(path)))
}

/// A measured frame rate as the usual one it's close to (within 5%), else rounded (replay.py _snap_fps).
pub fn snap_fps(fps: f64) -> i64 {
    for f in [24.0, 30.0, 40.0, 45.0, 50.0, 60.0, 90.0, 120.0] {
        if (fps - f).abs() <= 0.05 * f {
            return f as i64;
        }
    }
    py_round(fps)
}

/// The local time as "2026-11-22 17-05-09" (replay.py's time.strftime("%Y-%m-%d %H-%M-%S")).
pub fn stamp() -> String {
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut buf = [0u8; 64];
        let n = libc::strftime(buf.as_mut_ptr() as *mut _, buf.len(), c"%Y-%m-%d %H-%M-%S".as_ptr(), &tm);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }
}

/// What can go wrong here.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{what}: OSStatus {status}")]
    Status { what: &'static str, status: i32 },
    #[error("{0}")]
    Core(#[from] sp_core::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn status(what: &'static str, status: i32) -> Result<()> {
    if status == 0 { Ok(()) } else { Err(Error::Status { what, status }) }
}
