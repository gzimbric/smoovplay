//! The card as a source (capture.py CardSource), and run.py's `--capture` (open_capture).

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;

use sp_core::{Event, Frame, Source, SourceInfo, Sound};
use sp_media::queue::Ticker;
use sp_media::test_mode::SharedTestMode;

use crate::devices::{self, g, pick_audio, pick_mode, Device, Mode};
use crate::pipe::{Pipe, SoundFactory, DAMAGED_GIVE_UP_S};
use crate::session::Live;

/// How to read the card's colours, auto or as given. Auto is limited range unless the card says full, with BT.709
/// from 720 lines up and BT.601 below, as video players assume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Colour {
    #[default]
    Auto,
    Fixed { bt709: bool, full_range: bool },
}

impl Colour {
    /// --capture-colour: auto, 709-limited, 709-full, 601-limited or 601-full.
    pub fn parse(s: &str) -> Option<Colour> {
        Some(match s {
            "auto" => Colour::Auto,
            "709-limited" => Colour::Fixed { bt709: true, full_range: false },
            "709-full" => Colour::Fixed { bt709: true, full_range: true },
            "601-limited" => Colour::Fixed { bt709: false, full_range: false },
            "601-full" => Colour::Fixed { bt709: false, full_range: true },
            _ => return None,
        })
    }

    /// (bt709, full_range) for a picture `h` lines high whose card says `range` ("pc" full, else limited).
    pub fn resolve(self, h: usize, range: &str) -> (bool, bool) {
        match self {
            Colour::Auto => (h >= 720, range == "pc"),
            Colour::Fixed { bt709, full_range } => (bt709, full_range),
        }
    }
}

/// What every CardSource takes besides where its frames come from.
pub struct CardOptions {
    /// Test mode ("Simulate a 30 fps game"), applied before the queue.
    pub test: Option<SharedTestMode>,
    pub colour: Colour,
    /// Ticks queued for the pipeline. When it falls behind the oldest goes.
    pub queue_size: usize,
    /// What plays the sound (a card's, or a file's standing in), or None for no sound.
    pub audio: Option<SoundFactory>,
    /// No frame readable for this long and capture stops ([`DAMAGED_GIVE_UP_S`]).
    pub damaged_give_up_s: f64,
}

impl Default for CardOptions {
    fn default() -> CardOptions {
        CardOptions { test: None, colour: Colour::Auto, queue_size: 3, audio: None, damaged_give_up_s: DAMAGED_GIVE_UP_S }
    }
}

/// Frames from a capture card as Ticks. They're delivered on a thread of their own (the capture session's queue, or
/// the stand-in's reader) and queued, because keeping only the latest would break repeat detection. When the pipeline
/// falls behind the oldest goes and is counted in [`CardSource::dropped`].
///
/// Open a card with [`CardSource::open`], or with [`open_capture`] as run.py does. Files stand in for one, played at
/// their own rate from when the pipeline starts reading. [`CardSource::video_file`] takes any video, its sound too.
/// [`CardSource::raw_file`] and [`CardSource::raw_frames`] read raw frames exactly as a card's are read.
pub struct CardSource {
    pub(crate) info: SourceInfo,
    pub(crate) pipe: Arc<Pipe>,
    pub(crate) ticker: Ticker,
    pub(crate) started: bool,
    pub(crate) closed: bool,
    pub(crate) live: Option<Live>,
    pub(crate) thread: Option<JoinHandle<()>>,
    /// The frames come uncompressed (a card's raw format, or a stand-in's).
    pub(crate) raw: bool,
    pub(crate) name: String,
}

impl CardSource {
    pub(crate) fn new(pipe: Arc<Pipe>, name: &str, raw: bool, live: Option<Live>, thread: Option<JoinHandle<()>>) -> CardSource {
        let mut info = SourceInfo::of("capture");
        info.hz = pipe.fps;
        info.name = name.to_string();
        let how = if raw { "decoded and converted on the GPU" } else { "decoded by macOS (AVFoundation), converted on the GPU" };
        pipe.say(format!(
            "[capture] {name}: {}x{} {} at {} fps; {how} ({}, {} range)",
            pipe.w,
            pipe.h,
            pipe.fmt,
            g(pipe.fps),
            if pipe.bt709 { "BT.709" } else { "BT.601" },
            if pipe.full_range { "full" } else { "limited" }
        ));
        CardSource { info, pipe, ticker: Ticker::new("the capture card"), started: false, closed: false, live, thread, raw, name: name.into() }
    }

    pub fn width(&self) -> usize {
        self.pipe.w
    }
    pub fn height(&self) -> usize {
        self.pipe.h
    }
    pub fn fps(&self) -> f64 {
        self.pipe.fps
    }
    /// The card's own format ("nv12", "yuyv422", "mjpeg").
    pub fn fmt(&self) -> &str {
        &self.pipe.fmt
    }
    /// A card (timed by its own clock), not something standing in for one.
    pub fn is_card(&self) -> bool {
        self.pipe.live
    }
    /// Frames come uncompressed (run.py's clean_source).
    pub fn raw(&self) -> bool {
        self.raw
    }
    pub fn bt709(&self) -> bool {
        self.pipe.bt709
    }
    pub fn full_range(&self) -> bool {
        self.pipe.full_range
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Ticks dropped because the pipeline fell behind (since it started reading).
    pub fn dropped(&self) -> u64 {
        self.pipe.dropped.load(Ordering::Relaxed)
    }
    /// Frames the card sent that couldn't be read, skipped.
    pub fn damaged(&self) -> u64 {
        self.pipe.damaged.load(Ordering::Relaxed)
    }
    /// Frames macOS dropped before handing them over.
    pub fn os_dropped(&self) -> u64 {
        self.pipe.os_dropped.load(Ordering::Relaxed)
    }
    /// What the source has said in the log so far.
    pub fn messages(&self) -> Vec<String> {
        self.pipe.messages()
    }
    /// The delay test needs Remote Play.
    pub fn can_press(&self) -> bool {
        false
    }
    /// There's sound and something plays it, so the pipeline holds it back to meet the picture (Sound in sync).
    pub fn can_delay_audio(&self) -> bool {
        self.pipe.can_delay_audio()
    }
    /// How much later the sound arrives than the picture, going by their own timestamps (0 if not known).
    pub fn sound_late_s(&self) -> f64 {
        self.pipe.sound_late_s()
    }
    /// Says the once-a-minute sound line now, for tests. It doesn't tell whether there was anything to say.
    pub fn report_sound(&self) {
        self.pipe.report_sound(false)
    }

}

/// Why no frames are coming, if the card says (asked only while waiting).
fn why(live: &Option<Live>) -> String {
    match live {
        Some(l) if !l.connected() => "The capture card was disconnected".into(),
        Some(l) if !l.running() => "macOS stopped reading the capture card".into(),
        Some(l) if l.in_use_elsewhere() => "Another app is using the capture card".into(),
        _ => String::new(),
    }
}

impl Source for CardSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        (self.pipe.w, self.pipe.h)
    }
    fn next_event(&mut self) -> Option<Event> {
        if !self.started {
            self.started = true;
            self.pipe.dropped.store(0, Ordering::Relaxed); // frames dropped while the pipeline was loading don't count
            self.pipe.start();
        }
        let live = &self.live;
        let ev = self.ticker.next(&self.pipe.q, || why(live));
        if let (Some(Event::Waiting(_)), Some(l)) = (&ev, &self.live)
            && !l.connected()
        {
            // the card went away, so end the stream, as the Python's reader does when FFmpeg's read fails
            self.pipe.stop_with("the capture card was disconnected");
            return None;
        }
        ev
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.pipe.test.as_ref()?.lock().unwrap().gt(content)
    }
    fn read_sound(&mut self) -> Vec<Sound> {
        self.pipe.read_sound()
    }
    fn set_audio_delay(&mut self, ms: f64) {
        self.pipe.set_audio_delay(ms)
    }
    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.pipe.stop.store(true, Ordering::SeqCst);
        self.pipe.start(); // a stand-in still waiting for the pipeline
        if let Some(l) = self.live.take() {
            l.stop();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.pipe.end();
        self.pipe.close_sound();
        let mut line = format!("[capture] dropped frames: {}", self.dropped());
        if self.damaged() > 0 {
            line += &format!(", damaged frames from the card skipped: {}", self.damaged());
        }
        if self.os_dropped() > 0 {
            line += &format!(", frames macOS dropped before SmoovPlay got them: {}", self.os_dropped());
        }
        self.pipe.say(line);
    }
}

impl Drop for CardSource {
    fn drop(&mut self) {
        self.close();
    }
}

// ----------------------------------------------------------------------------- run.py's --capture
/// run.py's capture flags.
pub struct CaptureArgs {
    /// A card by name, unique ID or number (from --list-devices), or a video file standing in for one.
    pub capture: String,
    /// The size to ask the card for.
    pub w: u32,
    pub h: u32,
    pub fps: f64,
    /// "auto" (NV12, then YUY2, then MJPEG) or one the card lists.
    pub format: String,
    /// "auto", "709-limited", "709-full", "601-limited" or "601-full".
    pub colour: String,
    /// "auto" (the sound device named like the card), "off", or a sound device by name, unique ID or number.
    pub audio: String,
    pub test: Option<SharedTestMode>,
}

impl Default for CaptureArgs {
    fn default() -> CaptureArgs {
        CaptureArgs {
            capture: String::new(),
            w: 1920,
            h: 1080,
            fps: 60.0,
            format: "auto".into(),
            colour: "auto".into(),
            audio: "auto".into(),
            test: None,
        }
    }
}

/// Which sound device among `devices` goes with the card (run.py open_capture). `audio` is as --capture-audio gives
/// it. "auto" picks by name, and failing that tries the card's own sound. None means no sound.
pub fn pick_sound_device(card: &str, audio: &str, devices: &[Device], say: &mut dyn FnMut(String)) -> Option<String> {
    let names: Vec<&str> = devices.iter().filter(|d| d.kind == "audio").map(|d| d.name.as_str()).collect();
    match audio {
        "off" => None,
        "auto" => pick_audio(card, devices).or_else(|| {
            // Some cards carry their sound on the video device itself, not as a sound device of their own, so try
            // that. CardSource::open plays the picture alone if not.
            say("[capture] no sound device named like the card: trying the card's own sound".into());
            Some(card.to_string())
        }),
        n if n.parse::<usize>().is_ok_and(|i| i < names.len()) => Some(names[n.parse::<usize>().unwrap()].to_string()),
        other => Some(other.to_string()),
    }
}

/// run.py open_capture. Opens a card by name or number, or a file standing in for one. `sound` is what plays the
/// card's sound. None, or --capture-audio off, gives the picture only.
pub fn open_capture(a: &CaptureArgs, sound: Option<SoundFactory>) -> Result<CardSource, String> {
    let colour = Colour::parse(&a.colour).ok_or_else(|| format!("--capture-colour {}: not a choice", a.colour))?;
    let sound = if a.audio == "off" { None } else { sound };
    let opts = |audio| CardOptions { test: a.test.clone(), colour, audio, ..Default::default() };
    if Path::new(&a.capture).is_file() {
        // a recording standing in for a card, decoded and sent through the raw NV12 path
        return CardSource::video_file(Path::new(&a.capture), opts(sound)).map_err(|e| e.to_string());
    }
    let devices = devices::list_devices();
    // Opened by unique ID once known, as two devices can share a name. The sound device is picked by the name.
    let (mut name, mut card) = (a.capture.clone(), a.capture.clone());
    if let Ok(i) = name.parse::<usize>() {
        let video: Vec<&Device> = devices.iter().filter(|d| d.kind == "video" || d.kind == "unknown").collect();
        let d = video
            .get(i)
            .ok_or_else(|| format!("no capture device {i}: macOS lists {} (see --list-devices)", video.len()))?;
        (name, card) = (d.name.clone(), d.alt.clone());
    } else if let Some(d) = devices.iter().find(|d| d.alt == name) {
        name = d.name.clone();
    }
    let mut said = Vec::new();
    let audio_device =
        if sound.is_some() { pick_sound_device(&name, &a.audio, &devices, &mut |l| said.push(l)) } else { None };
    for l in said {
        println!("{l}");
    }
    let mode = if a.format != "auto" {
        let modes: Vec<Mode> = devices::list_modes(&card)?.into_iter().filter(|m| m.fmt == a.format).collect();
        let (mode, note) = pick_mode(&modes, a.w, a.h, a.fps);
        Some(mode.ok_or_else(|| format!("{name} has no {} mode ({note})", a.format))?)
    } else {
        None
    };
    CardSource::open(&card, a.w, a.h, a.fps, mode, audio_device.as_deref(), opts(sound))
}
