//! What a source yields, mirroring framegen/io.py's Tick and Waiting and framegen/sources.py's SourceInfo.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::frame::Frame;

/// One 60 Hz tick of a source: a frame and what's known about it.
#[derive(Clone, Debug)]
pub struct Tick {
    pub frame: Frame,
    /// When the frame reached this Mac, on [`crate::clock`]. For Remote Play that's when chiaki-ng got it, before
    /// decoding.
    pub t_arrival: f64,
    /// Tick number, from 0.
    pub index: u64,
    /// The source frame shown on this tick. Ground truth, set only for simulated input and test mode.
    pub content: Option<i64>,
    pub truth_new: Option<bool>,
    /// New content (true) or a repeat (false), by the source's own account. With None the pipeline compares pictures.
    pub new_hint: Option<bool>,
    /// Known to be corrupt because frames were lost before it. Held like a scene cut, not blended.
    pub damaged: bool,
    /// When the pipeline could first have it, after decoding. None means t_arrival.
    pub t_ready: Option<f64>,
    /// The source's own time for the frame in seconds (a file's position), for timing by the source's clock.
    pub t_source: Option<f64>,
    /// Anything else, for logs. Remote Play puts its per-frame facts here, such as size, decode_ms, size_new and
    /// game_new.
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Tick {
    pub fn new(frame: Frame, t_arrival: f64, index: u64) -> Tick {
        Tick {
            frame,
            t_arrival,
            index,
            content: None,
            truth_new: None,
            new_hint: None,
            damaged: false,
            t_ready: None,
            t_source: None,
            extra: BTreeMap::new(),
        }
    }
}

/// What a live source yields instead of blocking when no frame has come for a moment.
#[derive(Clone, Debug, PartialEq)]
pub struct Waiting {
    /// Seconds since the last frame.
    pub since: f64,
    /// Where frames come from, for the message on screen ("the PS5").
    pub what: String,
    /// What the source knows about it, as a hint for the player ("" if nothing).
    pub why: String,
}

/// How long a source waits for a frame before yielding a [`Waiting`] (framegen/io.py WAIT_POLL_S).
pub const WAIT_POLL_S: f64 = 0.1;

#[derive(Clone, Debug)]
pub enum Event {
    Tick(Tick),
    Waiting(Waiting),
}

/// A chunk of the game's sound, for instant replay.
#[derive(Clone, Debug)]
pub struct Sound {
    /// When its first sample reached this Mac, on [`crate::clock`].
    pub t: f64,
    /// Interleaved signed 16-bit samples.
    pub pcm: Vec<i16>,
    pub rate: u32,
    pub channels: u32,
}

/// What a source is (framegen/sources.py SourceInfo). The fields mean what they do there.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceInfo {
    /// "chiaki", "capture", "window", "tap", "file" or "sim".
    pub kind: String,
    pub name: String,
    pub live: bool,
    /// "fixed" (a frame every refresh, repeats included) or "variable" (frames when the sender likes).
    pub rate: String,
    pub hz: f64,
    /// "hint" (the source says which frames are repeats) or "pixels" (pictures compared).
    pub repeats: String,
    pub test_mode: bool,
    pub full_range: bool,
    pub audio: String,
    pub controls: String,
}

impl SourceInfo {
    /// framegen/sources.py KINDS[kind], before a running source fills in what only it knows.
    pub fn of(kind: &str) -> SourceInfo {
        let base = |name: &str| SourceInfo {
            kind: kind.to_string(),
            name: name.to_string(),
            live: true,
            rate: "fixed".into(),
            hz: 60.0,
            repeats: "pixels".into(),
            test_mode: true,
            full_range: true,
            audio: String::new(),
            controls: String::new(),
        };
        match kind {
            "chiaki" => SourceInfo {
                audio: "chiaki-ng plays the sound".into(),
                controls: "Your controller connects to this Mac; chiaki-ng sends it to the PS5".into(),
                ..base("PS5 Remote Play")
            },
            "capture" => SourceInfo {
                audio: "SmoovPlay plays the card's sound, in step with the picture".into(),
                controls: "Your controller connects to the console as usual".into(),
                ..base("Capture card")
            },
            "window" => SourceInfo {
                rate: "variable".into(),
                test_mode: false,
                audio: "The app plays its own sound, a little ahead of the picture".into(),
                controls: "Your controller connects to the app as usual".into(),
                ..base("App window")
            },
            "tap" => SourceInfo {
                audio: "The app plays its own sound".into(),
                controls: "Your controller connects to the app as usual".into(),
                ..base("Frame tap (for developers)")
            },
            "file" => SourceInfo { live: false, ..base("Video file") },
            "sim" => SourceInfo { live: false, test_mode: false, ..base("Video file as a PS5 30 fps game") },
            _ => base(kind),
        }
    }
}
