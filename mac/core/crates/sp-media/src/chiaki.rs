//! Frames from chiaki-ng (PS5 Remote Play) through its frame tap, decoded by VideoToolbox, as Ticks
//! (framegen/chiaki_tap.py ChiakiSource, step for step). Also any other app that writes chiaki-ng's layout of
//! compressed frames, through [`crate::open_tap`].
//!
//! Joins the live stream and never replays old frames. It asks for a keyframe whenever it can't decode: at start,
//! after falling more than the ring's 32 frames behind, or after a decode error. The reader starts at once, but until
//! the pipeline asks for frames (the first [`Source::next_event`]) it only watches for the stream's codec header,
//! which a new stream sends once, as its first packet. It waits at the stream's first picture, leaving it in the
//! ring, and starts decoding there when the pipeline is ready. Each frame's facts (size, loss, the size check's
//! call, decode time) go in the tick's `extra`, and [`ChiakiSource::log_tick`] writes Python's per-tick CSV.

use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde_json::{json, Value};
use sp_core::clock::now;
use sp_core::{Event, Frame, Source, SourceInfo, Sound, Tick};
use sp_tap::{nal, saved, AudioTap, Codec, FrameTap, SizeRepeatDetector, Watch};
use sp_tap::{codec_name, CODEC_H265_HDR, DAMAGE_HOLD, FEATURE_AUDIO_DELAY, FEATURE_AUDIO_TAP, FEATURE_PRESS, NAME, SLOTS};

use crate::convert::{Converter, HdrSettings};
use crate::queue::{TickQueue, Ticker};
use crate::test_mode::{self, SharedTestMode};
use crate::vt::{DecodeError, VtDecoder};

/// How to open a ChiakiSource.
#[derive(Clone)]
pub struct ChiakiOptions {
    /// The tap's name. chiaki-ng's is sp_tap::NAME.
    pub tap_name: String,
    /// How long to wait for a stream before giving up, seconds.
    pub wait_s: f64,
    /// Write one CSV row per tick here (log_tick).
    pub log_path: Option<PathBuf>,
    /// "Simulate a 30 fps game", applied as frames are decoded.
    pub test: Option<SharedTestMode>,
    /// Another app writes chiaki-ng's layout, so messages say "[tap]" and "the app".
    pub app: bool,
    /// A live header whose newest frame is older than this is a dead writer's, not a stream (STALE_S; 0 = never).
    pub stale_s: f64,
}

/// A live header whose newest frame is older than this was left by a writer that stopped without `end()`: POSIX
/// shared memory outlives chiaki-ng, where a Windows mapping died with it. A stream sends a frame every 33 ms or less.
pub const STALE_S: f64 = 5.0;

/// framegen/chiaki_tap.py ChiakiSource._stale. After a chiaki-ng crash the header still read live, and taking it
/// meant the old size and "stream size changed" once the real stream began. A header with no frames yet can't be
/// told from a writer about to send, so it counts as live.
fn stale_header(tap: &FrameTap, h: &sp_tap::Header, stale_s: f64) -> bool {
    if h.write_seq == 0 || stale_s <= 0.0 {
        return false;
    }
    tap.read(h.write_seq - 1, h).is_some_and(|p| now() - p.t_arrival > stale_s)
}

impl Default for ChiakiOptions {
    fn default() -> Self {
        ChiakiOptions { tap_name: NAME.into(), wait_s: 120.0, log_path: None, test: None, app: false, stale_s: STALE_S }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error("{0}")]
    Tap(#[from] sp_tap::Error),
    #[error("{0}")]
    Timeout(String),
    #[error("{0}")]
    Metal(#[from] sp_core::Error),
}

struct F64(AtomicU64);

impl F64 {
    fn new(v: f64) -> F64 {
        F64(AtomicU64::new(v.to_bits()))
    }
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
    fn set(&self, v: f64) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
}

/// What the reader thread and the pipeline's side share.
struct Shared {
    tap: FrameTap,
    q: TickQueue,
    go: AtomicBool,
    stop: AtomicBool,
    beat: F64,
    no_header: AtomicBool,
    ring: Mutex<(u64, f64)>,
    last_idr: Mutex<f64>,
    dropped: AtomicU64,
    skipped: AtomicU64,
    idr_requests: AtomicU64,
    decode_errors: AtomicU64,
    hdr: Mutex<Option<HdrSettings>>,
    hardware: AtomicBool,
}

impl Shared {
    fn ask_idr(&self) {
        let t = now();
        let mut last = self.last_idr.lock().unwrap();
        if t - *last < 0.5 {
            return; // chiaki-ng re-asks by itself, so don't flood the PS5
        }
        *last = t;
        self.tap.request_idr();
        self.idr_requests.fetch_add(1, Ordering::Relaxed);
    }
}

/// The reader thread's view of the stream it was opened on.
struct Cfg {
    tap_name: String,
    tag: &'static str,
    app: bool,
    w: u32,
    h: u32,
    test: Option<SharedTestMode>,
}

pub struct ChiakiSource {
    info: SourceInfo,
    w: usize,
    h: usize,
    /// The tap's codec (0 H.264, 1 H.265, 2 H.265 HDR).
    pub codec_id: u32,
    /// The stream uses chiaki-ng's H.265 HDR codec, 10-bit HDR10 from the PS5.
    pub hdr_stream: bool,
    sh: Arc<Shared>,
    sound: AudioTap,
    sound_session: Option<u32>,
    test: Option<SharedTestMode>,
    ticker: Ticker,
    started: bool,
    hdr_told: bool,
    log: Option<File>,
    t0: f64,
    tag: &'static str,
    app: bool,
    thread: Option<JoinHandle<()>>,
    closed: bool,
}

impl ChiakiSource {
    /// Opens the tap and waits (up to opts.wait_s) for a stream, then starts the reader.
    pub fn open(opts: ChiakiOptions) -> Result<ChiakiSource, OpenError> {
        let tag = if opts.app { "[tap]" } else { "[chiaki]" };
        let tap = FrameTap::open(&opts.tap_name)?;
        let sound = AudioTap::open(&opts.tap_name)?;
        let t_end = now() + opts.wait_s;
        let (mut printed, mut stale_said) = (false, false);
        let h = loop {
            let h = tap.header();
            let stale = h.live() && !opts.app && stale_header(&tap, &h, opts.stale_s);
            if h.live() && !stale {
                break h;
            }
            if stale {
                if !stale_said {
                    stale_said = true;
                    println!(
                        "[chiaki] the frame tap holds a stream header left by a chiaki-ng that stopped without ending its \
                         stream (its last frame is old): waiting for a new stream"
                    );
                }
            } else if !printed {
                println!(
                    "{}",
                    if opts.app {
                        "[tap] waiting for an app to start sending frames..."
                    } else {
                        "[chiaki] waiting for chiaki-ng to start a stream (run it with CHIAKI_FRAMETAP=1)..."
                    }
                );
                printed = true;
            }
            if now() > t_end {
                return Err(OpenError::Timeout(
                    if opts.app { "no app is sending frames to the frame tap" } else { "no chiaki-ng stream on the frame tap" }
                        .into(),
                ));
            }
            tap.wait(h.watch(), 0.25);
        };
        let mut info = SourceInfo::of("chiaki");
        info.hz = h.max_fps as f64;
        let what = format!("{} {}x{}, up to {} fps", codec_name(h.codec), h.width, h.height, h.max_fps);
        println!("{}", if opts.app { format!("[tap] frames: {what}") } else { format!("[chiaki] stream: {what}") });
        if h.max_fps < 60 {
            println!(
                "{}",
                if opts.app {
                    format!(
                        "[tap] WARNING: the stream is set to {} fps. Send 60 so the pipeline sees the console's 60 Hz \
                         output (it expects repeats, not every other frame missing).",
                        h.max_fps
                    )
                } else {
                    "[chiaki] WARNING: the stream is set to 30 fps. Set chiaki-ng's frame rate to 60 so the pipeline \
                     sees the PS5's 60 Hz output (it expects repeats, not every other frame missing)."
                        .to_string()
                }
            );
        }
        let t0 = now();
        let log = match &opts.log_path {
            Some(p) => {
                let mut f = File::create(p).map_err(sp_core::Error::from)?;
                writeln!(
                    f,
                    "tick,seq,t_arrival_ms,size,frames_lost,recovered,keyframe,decode_ms,queue_ms,size_says_new,\
                     image_says_new,max_diff,mean_diff,action,proc_ms\r"
                )
                .map_err(sp_core::Error::from)?;
                Some(f)
            }
            None => None,
        };
        let sh = Arc::new(Shared {
            tap,
            q: TickQueue::new(3),
            go: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            beat: F64::new(t0),
            no_header: AtomicBool::new(false),
            ring: Mutex::new((u64::MAX, t0)),
            last_idr: Mutex::new(f64::NEG_INFINITY),
            dropped: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            idr_requests: AtomicU64::new(0),
            decode_errors: AtomicU64::new(0),
            hdr: Mutex::new(None),
            hardware: AtomicBool::new(false),
        });
        let conv = Converter::new()?;
        let cfg = Cfg { tap_name: opts.tap_name.clone(), tag, app: opts.app, w: h.width, h: h.height, test: opts.test.clone() };
        let sh2 = sh.clone();
        let thread = std::thread::Builder::new()
            .name("chiaki-reader".into())
            .spawn(move || reader(sh2, conv, cfg))
            .map_err(sp_core::Error::from)?;
        Ok(ChiakiSource {
            info,
            w: h.width as usize,
            h: h.height as usize,
            codec_id: h.codec,
            hdr_stream: h.codec == CODEC_H265_HDR,
            sh,
            sound,
            sound_session: None,
            test: opts.test,
            ticker: Ticker::new("the PS5"),
            started: false,
            hdr_told: false,
            log,
            t0,
            tag,
            app: opts.app,
            thread: Some(thread),
            closed: false,
        })
    }

    /// Gives another app writing chiaki-ng's layout its own description.
    pub fn set_info(&mut self, info: SourceInfo) {
        self.info = info;
    }

    /// SmoovPlay's HDR setting (framegen/hdr.py), given before the frames start. Some(settings) converts an HDR
    /// stream, "sdr8" tone-mapped to SDR or "pq16" kept HDR10. None leaves it washed out as before, and `why` gives
    /// the reason if HDR is on but can't be used. Only an HDR stream is converted.
    pub fn set_hdr(&mut self, settings: Option<HdrSettings>, why: &str) {
        self.hdr_told = true;
        let Some(s) = settings else {
            if self.hdr_stream && !why.is_empty() {
                println!(
                    "{} WARNING: this stream is HDR, but {why}, so its colours look washed out. {}",
                    self.tag,
                    if self.app { "Have the app send it without HDR." } else { "Pick a codec without HDR in chiaki-ng's Video settings." }
                );
            } else if self.hdr_stream {
                println!(
                    "{} WARNING: this stream is HDR, and HDR is off in SmoovPlay's settings, so its colours look washed \
                     out. Turn on HDR in SmoovPlay's Settings (Picture), or {}",
                    self.tag,
                    if self.app { "have the app send it without HDR." } else { "pick a codec without HDR in chiaki-ng." }
                );
            }
            *self.sh.hdr.lock().unwrap() = None;
            return;
        };
        if !self.hdr_stream {
            println!(
                "{} HDR is on in SmoovPlay's settings, but this stream isn't HDR: shown as usual.{}",
                self.tag,
                if self.app { "" } else { " (For HDR, pick chiaki-ng's H.265 HDR codec.)" }
            );
            return;
        }
        *self.sh.hdr.lock().unwrap() = Some(s);
        println!(
            "{} HDR stream: {} (HDR is Experimental)",
            self.tag,
            if s.out == crate::HdrOut::Pq16 { "kept HDR for the HDR display" } else { "shown in SDR" }
        );
    }

    // The delay test and the sound delay go through chiaki-ng (frametap.h). A chiaki-ng from an older patch can't.
    pub fn can_press(&self) -> bool {
        self.sh.tap.features() & FEATURE_PRESS != 0
    }
    pub fn can_delay_audio(&self) -> bool {
        self.sh.tap.features() & FEATURE_AUDIO_DELAY != 0
    }
    /// Ask chiaki-ng to press `buttons` (a ChiakiControllerButton mask). Returns a token for press_sent.
    pub fn press(&self, buttons: u32, hold_ms: u32) -> u32 {
        self.sh.tap.press(buttons, hold_ms)
    }
    /// When chiaki-ng sent the press to the PS5 (on sp_core::clock), or None if it hasn't yet.
    pub fn press_sent(&self, token: u32) -> Option<f64> {
        self.sh.tap.press_sent(token, self.sh.tap.header().qpc_freq)
    }
    /// The game's sound is in the sound ring for instant replay. A chiaki-ng from before it leaves replays silent.
    pub fn has_sound(&self) -> bool {
        self.sh.tap.features() & FEATURE_AUDIO_TAP != 0 && self.sound.live()
    }

    /// With `on`, stop decoding while the pipeline restarts. Without it, start again from a fresh frame.
    pub fn hold(&mut self, on: bool) {
        if on {
            self.sh.go.store(false, Ordering::SeqCst);
            return;
        }
        self.sh.q.clear(); // frames from before the restart are stale by now
        self.sh.go.store(true, Ordering::SeqCst);
    }

    /// What the ring says about a wait for frames, as a hint under the window's "No picture" note. Empty if none.
    pub fn why(&self) -> String {
        why(&self.sh, self.app)
    }

    /// Decoded frames dropped because the pipeline fell behind, stream frames never decoded, and keyframe requests.
    pub fn counts(&self) -> (u64, u64, u64) {
        (
            self.sh.dropped.load(Ordering::Relaxed),
            self.sh.skipped.load(Ordering::Relaxed),
            self.sh.idr_requests.load(Ordering::Relaxed),
        )
    }

    /// Decode errors so far, and whether VideoToolbox decodes on the media engine.
    pub fn decoder_state(&self) -> (u64, bool) {
        (self.sh.decode_errors.load(Ordering::Relaxed), self.sh.hardware.load(Ordering::Relaxed))
    }

    /// The per-tick CSV row in Python's columns, with stream facts next to the pipeline's decisions. `image_new`,
    /// `max_diff` and `mean_diff` are the picture check's. `action` names what the pipeline did, and `t_start` is
    /// when the pipeline started on the tick.
    #[allow(clippy::too_many_arguments)]
    pub fn log_tick(&mut self, tick: &Tick, image_new: bool, max_diff: f64, mean_diff: f64, action: &str, proc_ms: f64, t_start: f64) {
        let Some(f) = self.log.as_mut() else { return };
        let e = &tick.extra;
        let s = |k: &str| match e.get(k) {
            None | Some(Value::Null) => String::new(),
            Some(Value::Bool(b)) => (*b as u8).to_string(),
            Some(v) => v.to_string(),
        };
        let b = |k: &str| (e.get(k).and_then(|v| v.as_bool()).unwrap_or(false) as u8).to_string();
        let decode_ms = e.get("decode_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let _ = write!(
            f,
            "{},{},{:.2},{},{},{},{},{:.2},{:.2},{},{},{:.2},{:.3},{},{:.2}\r\n",
            tick.index,
            s("seq"),
            (tick.t_arrival - self.t0) * 1000.0,
            s("size"),
            s("frames_lost"),
            b("recovered"),
            b("keyframe"),
            decode_ms,
            (t_start - tick.t_arrival) * 1000.0,
            s("size_new"),
            image_new as u8,
            max_diff,
            mean_diff,
            action,
            proc_ms
        );
    }
}

fn why(sh: &Shared, app: bool) -> String {
    let t = now();
    let h = sh.tap.header();
    {
        let mut ring = sh.ring.lock().unwrap();
        if h.write_seq != ring.0 {
            *ring = (h.write_seq, t);
        }
    }
    if !h.live() {
        return if app {
            "The app isn't streaming. Start its stream (or wait while it reconnects)".into()
        } else {
            "chiaki-ng isn't streaming. Connect to the PS5 in chiaki-ng (or wait while it reconnects)".into()
        };
    }
    if t - sh.beat.get() > 3.0 {
        return "SmoovPlay's video decoder has stopped. Press Q, then Play again".into();
    }
    if t - sh.ring.lock().unwrap().1 > 1.0 {
        return format!("{} isn't sending any frames. Is its stream still running?", if app { "The app" } else { "chiaki-ng" });
    }
    if sh.no_header.load(Ordering::Relaxed) {
        return format!(
            "This stream started before SmoovPlay, so its start is missing.\n{}",
            if app {
                "Restart the stream in the app"
            } else {
                "In chiaki-ng press Ctrl+O, click the x to end the stream, then connect again"
            }
        );
    }
    String::new()
}

impl Source for ChiakiSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }
    fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
    fn next_event(&mut self) -> Option<Event> {
        if !self.started {
            if !self.hdr_told {
                self.set_hdr(None, ""); // nobody said what to do with HDR, so act as with the setting off
            }
            self.sh.dropped.store(0, Ordering::Relaxed); // frames dropped while the pipeline was loading don't count
            self.sh.go.store(true, Ordering::SeqCst); // the pipeline is ready, so start decoding
            self.started = true;
        }
        let (sh, app) = (&self.sh, self.app);
        self.ticker.next(&sh.q, || why(sh, app))
    }
    fn ground_truth(&mut self, content: i64) -> Option<Frame> {
        self.test.as_ref()?.lock().unwrap().gt(content)
    }
    fn read_sound(&mut self) -> Vec<Sound> {
        let h = self.sh.tap.header();
        if !h.live() {
            return Vec::new();
        }
        if Some(h.session) != self.sound_session {
            self.sound_session = Some(h.session); // a new stream, so start from its current sound
            self.sound.reset(false);
        }
        self.sound
            .read_new()
            .into_iter()
            .map(|c| Sound { t: c.t, pcm: c.samples(), rate: c.rate, channels: c.channels })
            .collect()
    }
    fn request_keyframe(&mut self) {
        self.sh.ask_idr();
    }
    fn set_audio_delay(&mut self, ms: f64) {
        self.sh.tap.set_audio_delay(ms.round() as i64);
    }
    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.sh.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        if let Some(f) = self.log.as_mut() {
            let _ = f.flush();
        }
        let (d, s, i) = self.counts();
        println!("{} decoded frames dropped: {d}, stream frames skipped: {s}, keyframe requests: {i}", self.tag);
    }
}

impl Drop for ChiakiSource {
    fn drop(&mut self) {
        self.close();
    }
}

// --------------------------------------------------------------------------------------------- the reader thread
fn reader(sh: Arc<Shared>, mut conv: Converter, cfg: Cfg) {
    let tag = cfg.tag;
    let mut session: Option<u32> = None;
    let mut seq: u64 = 0;
    let mut dec: Option<VtDecoder> = None;
    let mut codec = Codec::H264;
    let mut codec_id = 0u32;
    let mut params: Vec<u8> = Vec::new(); // this stream's codec header, once seen
    let mut pending: Vec<u8> = Vec::new(); // the header, still to go into the decoder
    let mut decoded = false; // a picture has come out of this stream
    let mut damaged_left: u32 = 0; // frames still treated as damaged after a gap
    let mut errors = 0u32; // decode errors in a row
    let mut passed = false; // frames went by undecoded (joined late, the pipeline not ready yet, or held)
    let mut held: Option<u64> = None; // the frame the reader waits at for the pipeline to be ready
    let mut late = false;
    let (mut t_join, mut told) = (0.0, false);
    let mut sizes = SizeRepeatDetector::default();
    let mut hdr_said: Option<String> = None;
    let mut said_vt = false;
    let wait = |w: Watch, s: f64| sh.tap.wait(w, s);
    while !sh.stop.load(Ordering::SeqCst) {
        sh.beat.set(now());
        let h = sh.tap.header();
        if !h.live() {
            if decoded {
                // the stream we were showing ended
                println!("{}", if cfg.app { "[tap] the app stopped sending frames" } else { "[chiaki] stream ended" });
                break;
            }
            wait(h.watch(), 0.25); // not started yet, or being reconnected
            continue;
        }
        if Some(h.session) != session {
            // a new stream
            session = Some(h.session);
            codec_id = h.codec;
            codec = Codec::of_tap(h.codec).unwrap_or(Codec::H264);
            if (h.width, h.height) != (cfg.w, cfg.h) {
                eprintln!("{tag} stream size changed to {}x{}; restart SmoovPlay", h.width, h.height);
                break;
            }
            dec = None;
            params.clear();
            pending.clear();
            (decoded, damaged_left, errors, held) = (false, 0, 0, None);
            sizes = SizeRepeatDetector::default();
            // Read it from its first packet (the codec header) while that's still in the ring.
            seq = if h.write_seq <= (SLOTS - 4) as u64 { 0 } else { h.write_seq };
            late = seq > 0;
            passed = late;
            if late {
                // joining a running stream, whose header an earlier run may have saved
                params = saved::load(&cfg.tap_name, h.session, h.codec);
                if !params.is_empty() {
                    println!("{tag} joined a running stream, using its saved codec header");
                } else {
                    sh.ask_idr();
                }
            }
            sh.no_header.store(late && params.is_empty(), Ordering::Relaxed);
            (t_join, told) = (now(), false);
        }
        let going = sh.go.load(Ordering::SeqCst);
        if going && dec.is_none() {
            dec = Some(VtDecoder::new(codec));
            pending = params.clone();
        }
        if going && passed && !params.is_empty() {
            // Frames went by undecoded. Ask the PS5 for a refresh frame and hold what comes out until the picture
            // is whole again.
            (passed, damaged_left) = (false, 60);
            sh.ask_idr();
        }
        if late && params.is_empty() && !told && now() - t_join > 3.0 {
            println!(
                "{tag} this stream was already running, so its codec header went past before the pipeline started. {}: \
                 the pipeline picks up the new stream from its start.",
                if cfg.app {
                    "Restart the stream in the app"
                } else {
                    "In chiaki-ng press Ctrl+O, click the x to end the stream, then connect again"
                }
            );
            told = true;
        }
        if h.write_seq == seq {
            sh.tap.heartbeat();
            wait(h.watch(), 0.1);
            continue;
        }
        if h.write_seq - seq > (SLOTS - 4) as u64 {
            // fell behind, and the oldest slots are being overwritten
            if going {
                sh.skipped.fetch_add(h.write_seq - seq, Ordering::Relaxed);
                damaged_left = DAMAGE_HOLD;
                sh.ask_idr();
            } else {
                passed = true;
            }
            seq = h.write_seq;
            continue;
        }
        if !going && held == Some(seq) {
            // at the stream's first picture, waiting for the pipeline
            sh.tap.heartbeat();
            wait(h.watch(), 0.1);
            continue;
        }
        let p = sh.tap.read(seq, &h);
        seq += 1;
        let Some(p) = p else {
            if going {
                sh.skipped.fetch_add(1, Ordering::Relaxed);
                damaged_left = DAMAGE_HOLD;
                sh.ask_idr();
            } else {
                passed = true;
            }
            continue;
        };
        let ps = nal::param_sets(&p.data, codec);
        if !ps.is_empty() {
            if ps != params {
                saved::save(&cfg.tap_name, h.session, codec_id, &ps);
            }
            params = ps.clone();
            sh.no_header.store(false, Ordering::Relaxed);
            if nal::only_params(&p.data, codec) {
                // the header on its own
                pending = ps;
                continue;
            }
        } else if params.is_empty() {
            // nothing is decodable without the header
            if going {
                sh.skipped.fetch_add(1, Ordering::Relaxed);
            }
            continue;
        }
        if !going {
            // the pipeline isn't ready, so only the header was wanted
            if !passed {
                // A picture. Leave it in the ring and start decoding there once the pipeline is ready. If that takes
                // longer than the ring holds, it moves on (passed) and the PS5 is asked for a refresh frame instead.
                seq = p.seq;
                held = Some(p.seq);
            }
            continue;
        }
        let key = nal::is_keyframe(&p.data, codec);
        let t = now();
        let d = dec.as_mut().unwrap();
        if !pending.is_empty() {
            d.set_params(&pending);
        }
        let result = d.decode(&p.data);
        sh.hardware.store(d.hardware, Ordering::Relaxed);
        let pictures = match result {
            Ok(pics) => {
                pending.clear();
                errors = 0;
                if !said_vt && !pics.is_empty() {
                    said_vt = true;
                    println!("{tag} decoding with VideoToolbox{}", if d.hardware { "" } else { " (in software)" });
                }
                pics
            }
            Err(e) => {
                // Keep the decoder, which holds the header and reference frames, and hold the next few frames. The
                // PS5 answers a keyframe request with a refresh frame.
                eprintln!("{tag} decode error at frame {}: {e}", p.seq);
                sh.decode_errors.fetch_add(1, Ordering::Relaxed);
                // max, because VideoToolbox fails the pictures before the first keyframe after a join, where FFmpeg
                // drops them silently. Those errors mustn't cut the longer hold that the join started.
                damaged_left = damaged_left.max(DAMAGE_HOLD);
                errors += 1;
                if errors >= 30 || e == DecodeError::NoParams {
                    // stuck, so start a decoder afresh from the saved header
                    dec = Some(VtDecoder::new(codec));
                    (errors, pending) = (0, params.clone());
                }
                sh.ask_idr();
                continue;
            }
        };
        if pictures.is_empty() {
            continue; // a decoder that holds a frame back, or dropped it
        }
        decoded = true;
        let size_new = sizes.update(p.data.len() as u64, key);
        let damaged = p.frames_lost > 0 || p.recovered || p.truncated || damaged_left > 0;
        let hdr = *sh.hdr.lock().unwrap();
        for pic in pictures {
            let image = match hdr {
                Some(s) => conv.hdr(&pic, &s).map(|(f, info)| {
                    let d = info.describe();
                    if hdr_said.as_deref() != Some(&d) {
                        println!("{tag} HDR: the stream is {d}");
                        hdr_said = Some(d);
                    }
                    f
                }),
                None => conv.sdr(&pic),
            };
            let image = match image {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("{tag} couldn't convert frame {}: {e}", p.seq);
                    continue;
                }
            };
            let decode_ms = (now() - t) * 1000.0;
            // Ready is arrival plus decode time, which the pipeline's behind check has always measured from.
            let mut tick = Tick::new(image, p.t_arrival, 0);
            tick.damaged = damaged;
            tick.t_ready = Some(p.t_arrival + decode_ms / 1000.0);
            let e = &mut tick.extra;
            e.insert("seq".into(), json!(p.seq));
            e.insert("size".into(), json!(p.data.len()));
            e.insert("frames_lost".into(), json!(p.frames_lost));
            e.insert("recovered".into(), json!(p.recovered));
            e.insert("keyframe".into(), json!(key));
            e.insert("decode_ms".into(), json!(decode_ms));
            // the size check's call, for the stream check, test mode and the frame log
            e.insert("size_new".into(), json!(size_new));
            e.insert("size_ref".into(), json!(sizes.r#ref));
            e.insert("big".into(), json!(sizes.big));
            let Some(tick) = test_mode::apply(&cfg.test, tick) else { continue };
            if sh.q.put(tick) {
                sh.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        damaged_left = damaged_left.saturating_sub(1);
    }
    sh.q.end();
}
