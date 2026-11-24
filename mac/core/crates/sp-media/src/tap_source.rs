//! Decoded frames (NV12 or BGRA, layout version 2) another app hands over through the frame tap
//! (framegen/tap.py TapSource), and [`open_tap`], which picks the reader for whatever the writer sends.
//!
//! There's nothing to decode. Each frame is copied out of its slot into a GPU buffer and converted, NV12 with the
//! header's colours and bilinear chroma as capture.py raw_to_bgr does. Repeats are as the writer flags them (FLAG_NEW
//! or FLAG_REPEAT, in `new_hint`) when it does, else the pipeline compares pictures. A writer that sends a frame only
//! when the picture changes sets FEATURE_VARIABLE, and open_tap puts it on a 60 Hz beat with [`crate::Clock60`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde_json::json;
use sp_core::clock::now;
use sp_core::{Event, Frame, Source, SourceInfo, Sound, Tick};
use sp_tap::{
    codec_name, colour_of, AudioTap, Codec, FrameTap, Header, RawFormat, FEATURE_VARIABLE, FLAG_NEW, FLAG_REPEAT,
};

use crate::chiaki::{ChiakiOptions, ChiakiSource, OpenError, STALE_S};
use crate::clock60::{Clock60, QueuedSource};
use crate::convert::Converter;
use crate::queue::{TickQueue, Ticker};
use crate::test_mode::{self, SharedTestMode};

struct Shared {
    tap: FrameTap,
    q: Arc<TickQueue>,
    go: AtomicBool,
    stop: AtomicBool,
    beat: Mutex<f64>,
    ring: Mutex<(u64, f64)>,
    dropped: AtomicU64,
    skipped: AtomicU64,
}

pub struct TapSource {
    info: SourceInfo,
    w: usize,
    h: usize,
    pub format: RawFormat,
    sh: Arc<Shared>,
    sound: AudioTap,
    test: Option<SharedTestMode>,
    ticker: Ticker,
    thread: Option<JoinHandle<()>>,
    closed: bool,
}

impl TapSource {
    /// A reader for the decoded frames the writer on `tap` has started sending, as `header` describes them. `name` is
    /// the tap's name.
    pub fn new(tap: FrameTap, name: &str, header: Header, test: Option<SharedTestMode>) -> Result<TapSource, OpenError> {
        let format = RawFormat::of_codec(header.codec).ok_or_else(|| OpenError::Timeout("not decoded frames".into()))?;
        let variable = header.features & FEATURE_VARIABLE != 0;
        let mut info = SourceInfo::of("tap");
        info.hz = if header.max_fps > 0 { header.max_fps as f64 } else { 60.0 };
        info.rate = if variable { "variable" } else { "fixed" }.into();
        info.test_mode = !variable;
        println!(
            "[tap] frames: {} {}x{}, {}",
            codec_name(header.codec),
            header.width,
            header.height,
            if variable { "frames when the picture changes".to_string() } else { format!("{} a second", header.max_fps) }
        );
        let sound = AudioTap::open(name)?;
        let conv = Converter::new()?;
        let t = now();
        let sh = Arc::new(Shared {
            tap,
            q: Arc::new(TickQueue::new(3)),
            go: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            beat: Mutex::new(t),
            ring: Mutex::new((u64::MAX, t)),
            dropped: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        });
        let (sh2, test2) = (sh.clone(), test.clone());
        let thread = std::thread::Builder::new()
            .name("tap-reader".into())
            .spawn(move || reader(sh2, conv, header, format, test2))
            .map_err(sp_core::Error::from)?;
        Ok(TapSource {
            info,
            w: header.width as usize,
            h: header.height as usize,
            format,
            sh,
            sound,
            test,
            ticker: Ticker::new("the app"),
            thread: Some(thread),
            closed: false,
        })
    }

    /// With `on`, stop reading while the pipeline restarts. Without it, start again from the newest frame.
    pub fn hold(&mut self, on: bool) {
        if on {
            self.sh.go.store(false, Ordering::SeqCst);
            return;
        }
        self.sh.q.clear();
        self.sh.go.store(true, Ordering::SeqCst);
    }

    pub fn why(&self) -> String {
        why(&self.sh)
    }

    pub fn has_sound(&self) -> bool {
        self.sound.live()
    }

    /// Frames dropped because the pipeline fell behind, and frames the writer overwrote before they were read.
    pub fn counts(&self) -> (u64, u64) {
        (self.sh.dropped.load(Ordering::Relaxed), self.sh.skipped.load(Ordering::Relaxed))
    }
}

fn why(sh: &Shared) -> String {
    let t = now();
    let h = sh.tap.header();
    {
        let mut r = sh.ring.lock().unwrap();
        if h.write_seq != r.0 {
            *r = (h.write_seq, t);
        }
    }
    if !h.live() {
        return "The app stopped sending frames".into();
    }
    if t - *sh.beat.lock().unwrap() > 3.0 {
        return "SmoovPlay stopped reading frames. Press Q, then Play again".into();
    }
    if t - sh.ring.lock().unwrap().1 > 1.0 {
        return "The app isn't sending any frames. Is it still running?".into();
    }
    String::new()
}

impl QueuedSource for TapSource {
    fn queue(&self) -> Arc<TickQueue> {
        self.sh.q.clone()
    }
    fn start(&mut self) {
        self.sh.go.store(true, Ordering::SeqCst);
    }
    fn hint(&self) -> String {
        self.why()
    }
}

impl Source for TapSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
    fn next_event(&mut self) -> Option<Event> {
        self.sh.go.store(true, Ordering::SeqCst);
        let sh = &self.sh;
        self.ticker.next(&sh.q, || why(sh))
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.test.as_ref()?.lock().unwrap().gt(content)
    }
    fn read_sound(&mut self) -> Vec<Sound> {
        self.sound
            .read_new()
            .into_iter()
            .map(|c| Sound { t: c.t, pcm: c.samples(), rate: c.rate, channels: c.channels })
            .collect()
    }
    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.sh.stop.store(true, Ordering::SeqCst);
        self.sh.go.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let (d, s) = self.counts();
        println!("[tap] dropped frames: {d}, overwritten before they were read: {s}");
    }
}

impl Drop for TapSource {
    fn drop(&mut self) {
        self.close();
    }
}

fn reader(sh: Arc<Shared>, mut conv: Converter, first: Header, format: RawFormat, test: Option<SharedTestMode>) {
    let (w, hgt) = (first.width as usize, first.height as usize);
    let n = format.frame_bytes(w, hgt);
    let (bt709, full_range) = colour_of(first.colour).unwrap_or((hgt >= 720, false));
    let (codec, mut session) = (first.codec, first.session);
    let mut seq: Option<u64> = None; // the next frame to read, or None for the writer's newest
    while !sh.stop.load(Ordering::SeqCst) {
        *sh.beat.lock().unwrap() = now();
        let h = sh.tap.header();
        if !h.live() {
            println!("[tap] the app stopped sending frames");
            break;
        }
        if h.session != session {
            // the writer started over
            if (h.codec, h.width, h.height) != (codec, first.width, first.height) {
                eprintln!(
                    "[tap] the frames changed to {} {}x{}; restart SmoovPlay",
                    codec_name(h.codec),
                    h.width,
                    h.height
                );
                break;
            }
            (session, seq) = (h.session, None);
        }
        let go = sh.go.load(Ordering::SeqCst);
        let Some(mut s) = seq.filter(|_| go) else {
            // the pipeline isn't ready, so read nothing and do no GPU work
            seq = if go { Some(h.write_seq) } else { None };
            sh.tap.heartbeat();
            sh.tap.wait(h.watch(), 0.05);
            continue;
        };
        if h.write_seq == s {
            sh.tap.heartbeat();
            sh.tap.wait(h.watch(), 0.1);
            continue;
        }
        if h.write_seq - s >= h.slots as u64 - 1 {
            // fell behind, and the oldest slots are being rewritten
            sh.skipped.fetch_add(h.write_seq - 1 - s, Ordering::Relaxed);
            s = h.write_seq - 1;
        }
        let p = sh.tap.read_into(s, &h, conv.staging(n));
        seq = Some(s + 1);
        let Some(p) = p.filter(|p| p.size >= n) else {
            sh.skipped.fetch_add(1, Ordering::Relaxed); // overwritten while we read it, or a short frame
            continue;
        };
        let frame = match format {
            RawFormat::Bgra => conv.bgra_staged(w, hgt),
            RawFormat::Nv12 => conv.nv12_staged(w, hgt, bt709, full_range),
        };
        let mut tick = Tick::new(frame, p.t_arrival, 0);
        tick.new_hint = if p.flags & FLAG_NEW != 0 {
            Some(true)
        } else if p.flags & FLAG_REPEAT != 0 {
            Some(false)
        } else {
            None
        };
        tick.damaged = p.frames_lost > 0 || p.recovered || p.truncated;
        tick.t_ready = Some(now());
        tick.extra.insert("seq".into(), json!(p.seq));
        let Some(tick) = test_mode::apply(&test, tick) else { continue };
        if sh.q.put(tick) {
            sh.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    sh.q.end();
}

/// What a writer on the tap sends, read the right way.
pub enum OpenedTap {
    /// Compressed frames (H.264 or H.265), read by chiaki-ng's reader.
    Chiaki(ChiakiSource),
    /// Decoded frames, one every refresh.
    Tap(TapSource),
    /// Decoded frames sent only when the picture changes, put on a 60 Hz beat.
    Variable(Clock60<TapSource>),
}

impl OpenedTap {
    pub fn kind(&self) -> &'static str {
        match self {
            OpenedTap::Chiaki(_) => "chiaki",
            _ => "tap",
        }
    }
    pub fn into_source(self) -> Box<dyn Source> {
        match self {
            OpenedTap::Chiaki(s) => Box::new(s),
            OpenedTap::Tap(s) => Box::new(s),
            OpenedTap::Variable(s) => Box::new(s),
        }
    }
}

/// framegen/tap.py open_tap. Waits up to wait_s for a writer on `name`, then reads what it sends. `test` is test
/// mode, for a writer that sends every refresh. It's an error for a variable writer with test mode on. log_path is
/// the per-tick CSV for compressed frames.
pub fn open_tap(
    name: &str,
    wait_s: f64,
    test: Option<SharedTestMode>,
    log_path: Option<std::path::PathBuf>,
) -> Result<OpenedTap, OpenError> {
    let tap = FrameTap::open(name)?;
    let t_end = now() + wait_s;
    let mut said = false;
    let h = loop {
        let h = tap.header();
        if h.live() {
            break h;
        }
        if !said {
            println!("[tap] waiting for an app to start sending frames...");
            said = true;
        }
        if now() > t_end {
            return Err(OpenError::Timeout("no app is sending frames to the frame tap".into()));
        }
        tap.wait(h.watch(), 0.25);
    };
    if RawFormat::of_codec(h.codec).is_some() {
        let variable = h.features & FEATURE_VARIABLE != 0;
        if variable && test.as_ref().is_some_and(|t| t.lock().unwrap().on) {
            return Err(OpenError::Timeout(
                "test mode needs an app that sends every refresh; this one sends only new pictures".into(),
            ));
        }
        let src = TapSource::new(tap, name, h, if variable { None } else { test })?;
        return Ok(if variable { OpenedTap::Variable(Clock60::new(src, 60.0)) } else { OpenedTap::Tap(src) });
    }
    drop(tap);
    if Codec::of_tap(h.codec).is_none() {
        return Err(OpenError::Timeout(format!("the app sends frames in a format SmoovPlay doesn't know (codec {})", h.codec)));
    }
    let mut src = ChiakiSource::open(ChiakiOptions { tap_name: name.into(), wait_s, log_path, test, app: true, stale_s: STALE_S })?;
    let mut info = src.info().clone();
    info.name = "Frame tap".into();
    info.audio = "The app plays its own sound".into();
    info.controls = "Your controller connects to the app as usual".into();
    src.set_info(info);
    Ok(OpenedTap::Chiaki(src))
}
