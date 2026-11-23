//! The recorder (replay.py Recorder). Through sp_sched::Replay, the pipeline calls [`Recorder::frame`] for each frame
//! it shows, [`Recorder::real_frame`] for each new frame from the game and [`Recorder::set_delay`] with its delay.
//! The game's sound comes from [`Recorder::attach_sound`] (polled) or [`Recorder::add_sound`]. [`Recorder::save`]
//! writes a clip.
//!
//! The pipeline's thread only queues GPU work and a message. It never waits for the GPU or the encoder, and drops a
//! frame instead when the encoder is behind or not open yet. The replay's own thread waits for the GPU, feeds
//! VideoToolbox and AudioToolbox, and opens encoders. VideoToolbox's thread puts packets in the ring. A save runs on
//! a thread of its own, and a before & after video in a process of its own.

use std::collections::{HashSet, VecDeque};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use objc2::rc::{Retained, autoreleasepool};
use objc2_core_foundation::CFRetained;
use objc2_core_video::CVPixelBuffer;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use sp_core::{Frame, Gpu, PixelKind, Sound, clock};
use sp_media::{Converter, HdrOut, HdrSettings};

use crate::aac::{AacEncoder, ToStereo48};
use crate::gpu::{self, CommandBuffer, Planes, TexCache};
use crate::mp4::{self, Sample, TrackSpec};
use crate::sound::{Part, SoundClock};
use crate::track::{Format, Piece, Track, cut};
use crate::vtenc::{Encoded, VideoConfig, VtEncoder};
use crate::{
    AFTER_BPP, AFTER_NOMINAL, BEFORE_BPP, BEFORE_NOMINAL, ENCODER, Error, KEEP_EXTRA_S, KEY_EVERY_S, OFFSET, POOL, Result,
    SOUND_BITRATE, SOUND_RATE, TB, aac, dur, py_round, short, snap_fps, stamp, unique, videos_folder,
};

const AFTER: usize = 0;
const BEFORE: usize = 1;
const SOUND: usize = 2;

/// What save() writes: what was shown, a before & after video, or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Smoovplay,
    Compare,
    Both,
}

impl Kind {
    /// settings.replay_save's values ("smoovplay", "compare", "both").
    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "smoovplay" => Some(Kind::Smoovplay),
            "compare" => Some(Kind::Compare),
            "both" => Some(Kind::Both),
            _ => None,
        }
    }
    fn video(self) -> bool {
        self != Kind::Compare
    }
    fn compare(self) -> bool {
        self != Kind::Smoovplay
    }
}

/// A before & after video's layout (settings.replay_layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Split,
    Side,
}

impl Layout {
    pub fn parse(s: &str) -> Option<Layout> {
        match s {
            "split" => Some(Layout::Split),
            "side" => Some(Layout::Side),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Layout::Split => "split",
            Layout::Side => "side",
        }
    }
}

/// Messages for the user (the window's toasts), as (text, seconds).
pub type Notify = Arc<dyn Fn(&str, f64) + Send + Sync>;
/// Where the `[replay] ...` lines go (Python's print).
pub type Log = Arc<dyn Fn(&str) + Send + Sync>;
/// Settings "Looks in clips". It gives the frame with the look drawn on (the video window's baked()), for the
/// SmoovPlay video only. Err means the clip goes without the look, and that's said once.
pub type LookFn = Box<dyn FnMut(&Frame) -> std::result::Result<Frame, String> + Send>;
/// The game's sound since the last call (ChiakiSource or CardSource read_sound), polled every 20 ms.
pub type SoundFn = Box<dyn FnMut() -> Vec<Sound> + Send>;

/// The recorder's settings (replay.py Recorder's arguments that apply on a Mac).
pub struct Options {
    /// How much to keep, in seconds. 0 is off, and set_length changes it.
    pub length_s: u32,
    /// What save() writes by default.
    pub clip: Kind,
    pub layout: Layout,
    /// Where clips go, or None for [`videos_folder`].
    pub out_dir: Option<PathBuf>,
    pub notify: Option<Notify>,
    /// None prints to stdout.
    pub log: Option<Log>,
    /// When false, as in tests, it waits for the encoder instead of dropping a frame.
    pub drop_when_behind: bool,
    /// The stream's size, so warm_up() can open the encoders before play.
    pub size: Option<(usize, usize)>,
    /// [`crate::saving_dir`] when SmoovPlay runs it. Before & after videos say there that they're being made.
    pub mark_dir: Option<PathBuf>,
    /// The before & after maker and its first arguments, or None for this program with `replay-compose`. The compose
    /// arguments follow.
    pub compose_cmd: Option<Vec<OsString>>,
    /// The keys that work from anywhere, for the message that replay is on, for example "Control-Option-R and
    /// Control-Option-B from anywhere". None if the app has none.
    pub hotkeys: Option<String>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            length_s: 0,
            clip: Kind::Smoovplay,
            layout: Layout::Split,
            out_dir: None,
            notify: None,
            log: None,
            drop_when_behind: true,
            size: None,
            mark_dir: None,
            compose_cmd: None,
            hotkeys: None,
        }
    }
}

/// What's held, as (seconds, megabytes) per track.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Held {
    pub after: (f64, f64),
    pub before: (f64, f64),
    pub sound: (f64, f64),
}

impl Held {
    pub fn all(&self) -> [(f64, f64); 3] {
        [self.after, self.before, self.sound]
    }
    pub fn mb(&self) -> f64 {
        self.after.1 + self.before.1 + self.sound.1
    }
}

/// A distribution's summary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
}

impl Summary {
    fn of(v: &VecDeque<f64>) -> Summary {
        let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        if s.is_empty() {
            return Summary::default();
        }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        Summary { n, mean: s.iter().sum::<f64>() / n as f64, p50: s[n / 2], p99: s[(n * 99 / 100).min(n - 1)], max: s[n - 1] }
    }
}

/// How the recorder is doing over the last 20 000 frames, in what it costs the pipeline's thread, the GPU and the
/// encoder.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// Time on the pipeline's thread per frame taken, in microseconds.
    pub put_us: Summary,
    /// GPU time of the conversion per frame, in microseconds.
    pub gpu_us: Summary,
    /// Time from handing a frame to VideoToolbox to its packet, in milliseconds.
    pub encode_ms: Summary,
    /// Video packets kept.
    pub packets: u64,
}

#[derive(Default)]
struct Acc {
    put_us: VecDeque<f64>,
    gpu_us: VecDeque<f64>,
    encode_ms: VecDeque<f64>,
    packets: u64,
}

fn push(v: &mut VecDeque<f64>, x: f64) {
    if v.len() >= 20_000 {
        v.pop_front();
    }
    v.push_back(x);
}

// ---- what VideoToolbox's thread touches -------------------------------------------------------------------------
struct Shared {
    tracks: Mutex<[Track; 3]>,
    length: AtomicU32,
    in_flight: [AtomicUsize; 2],
    /// Frames handed on and frames done with (encoded, dropped or failed), per video track, for flush().
    submitted: [AtomicU64; 2],
    completed: [AtomicU64; 2],
    progress: (Mutex<()>, Condvar),
    log: Log,
    warned: Mutex<HashSet<String>>,
    acc: Mutex<Acc>,
}

impl Shared {
    fn warn(&self, key: &str, text: &str) {
        if self.warned.lock().unwrap().insert(key.to_string()) {
            (self.log)(text);
        }
    }

    fn poke(&self) {
        let _g = self.progress.0.lock().unwrap();
        self.progress.1.notify_all();
    }

    fn done_one(&self, track: usize) {
        self.completed[track].fetch_add(1, Ordering::SeqCst);
        let n = &self.in_flight[track];
        let mut cur = n.load(Ordering::SeqCst);
        while cur > 0 {
            match n.compare_exchange(cur, cur - 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
        self.poke();
    }

    /// A packet from a video track's encoder.
    fn on_packet(&self, track: usize, generation: u64, r: std::result::Result<Encoded, i32>) {
        match r {
            Ok(e) => {
                {
                    let mut a = self.acc.lock().unwrap();
                    push(&mut a.encode_ms, e.latency * 1000.0);
                    a.packets += 1;
                }
                let length = self.length.load(Ordering::SeqCst);
                let mut t = self.tracks.lock().unwrap();
                if length > 0 && t[track].generation == generation {
                    if t[track].format.as_ref() != Some(&e.format) {
                        if t[track].format.is_some() {
                            // A new codec header mid-stream, so what's held can't be cut with it
                            t[track].ring.clear();
                            t[track].keys.clear();
                            t[track].nbytes = 0;
                        }
                        t[track].format = Some(e.format.clone());
                    }
                    t[track].add(Piece { pts: e.pts, ts: e.pts, key: e.key, data: Arc::from(e.data) });
                    trim(&mut t, length);
                }
            }
            Err(-1) => {}
            Err(st) => self.warn(&format!("encode{st}"), &format!("[replay] the encoder failed a frame ({st})")),
        }
        self.done_one(track);
    }
}

fn trim(tracks: &mut [Track; 3], length: u32) {
    let newest = tracks.iter().filter_map(|t| t.newest()).max();
    if let Some(n) = newest {
        let cut = n - py_round((length as f64 + KEEP_EXTRA_S) * TB as f64);
        for t in tracks.iter_mut() {
            t.trim(cut);
        }
    }
}

// ---- the replay's thread's messages ---------------------------------------------------------------------------
struct VideoJob {
    track: usize,
    t: f64,
    generation: u64,
    buf: CFRetained<CVPixelBuffer>,
    cb: Retained<CommandBuffer>,
    // Held until the GPU has read the frame and written the buffer
    keep: Option<(Frame, Planes)>,
}

struct HdrJob {
    track: usize,
    t: f64,
    copy: Frame,
    cb: Retained<CommandBuffer>,
    src: Option<Frame>,
}

enum Msg {
    Prepare,
    Open(usize, usize, usize),
    Video(VideoJob),
    Hdr(HdrJob),
    Sound(Vec<Sound>),
    Complete(u64),
    Reset,
    Stop,
}

// Command buffers, pixel buffers and frames are safe to hand to another thread.
unsafe impl Send for Msg {}

struct VideoEnc {
    enc: VtEncoder,
    generation: u64,
}

struct Cfg {
    clip: Kind,
    layout: Layout,
    out_dir: Option<PathBuf>,
    mark_dir: Option<PathBuf>,
    compose_cmd: Option<Vec<OsString>>,
    size: Option<(usize, usize)>,
}

struct Inner {
    sh: Arc<Shared>,
    notify: Option<Notify>,
    block: bool,
    hotkeys: Option<String>,
    cfg: Mutex<Cfg>,
    t0: Mutex<Option<f64>>,
    delay: AtomicU64,
    after_gap: AtomicU64,
    after_last: Mutex<Option<f64>>,
    ready: AtomicBool,
    encs: Mutex<[Option<Arc<VideoEnc>>; 2]>,
    opening: [AtomicBool; 2],
    cache: OnceLock<TexCache>,
    tx: Mutex<Sender<Msg>>,
    flush_seq: AtomicU64,
    flushed: AtomicU64,
    look: Mutex<Option<LookFn>>,
    hdr: Mutex<Option<HdrSettings>>,
    sound_fn: Mutex<Option<SoundFn>>,
    dropped: AtomicU64,
    quick: (Mutex<usize>, Condvar),
    making: Mutex<Vec<u32>>,
    prep: Mutex<()>,
    hdr_pool: Mutex<Vec<Frame>>,
    said_encoder: AtomicBool,
}

fn f64_of(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::SeqCst))
}

/// Instant replay (see the module docs).
pub struct Recorder {
    inner: Arc<Inner>,
    worker: Mutex<Option<(JoinHandle<()>, Receiver<()>)>>,
}

impl Recorder {
    pub fn new(opts: Options) -> Recorder {
        let log: Log = opts.log.unwrap_or_else(|| Arc::new(|s: &str| println!("{s}")));
        let sh = Arc::new(Shared {
            tracks: Mutex::new([Track::new("after"), Track::new("before"), Track::new("sound")]),
            length: AtomicU32::new(0),
            in_flight: [AtomicUsize::new(0), AtomicUsize::new(0)],
            submitted: [AtomicU64::new(0), AtomicU64::new(0)],
            completed: [AtomicU64::new(0), AtomicU64::new(0)],
            progress: (Mutex::new(()), Condvar::new()),
            log,
            warned: Mutex::new(HashSet::new()),
            acc: Mutex::new(Acc::default()),
        });
        let (tx, rx) = channel();
        let inner = Arc::new(Inner {
            sh,
            notify: opts.notify,
            block: !opts.drop_when_behind,
            hotkeys: opts.hotkeys,
            cfg: Mutex::new(Cfg {
                clip: opts.clip,
                layout: opts.layout,
                out_dir: opts.out_dir,
                mark_dir: opts.mark_dir,
                compose_cmd: opts.compose_cmd,
                size: opts.size.map(|(w, h)| (w & !1, h & !1)),
            }),
            t0: Mutex::new(None),
            delay: AtomicU64::new(0.04f64.to_bits()), // until the pipeline says
            after_gap: AtomicU64::new(0f64.to_bits()),
            after_last: Mutex::new(None),
            ready: AtomicBool::new(false),
            encs: Mutex::new([None, None]),
            opening: [AtomicBool::new(false), AtomicBool::new(false)],
            cache: OnceLock::new(),
            tx: Mutex::new(tx),
            flush_seq: AtomicU64::new(0),
            flushed: AtomicU64::new(0),
            look: Mutex::new(None),
            hdr: Mutex::new(None),
            sound_fn: Mutex::new(None),
            dropped: AtomicU64::new(0),
            quick: (Mutex::new(0), Condvar::new()),
            making: Mutex::new(Vec::new()),
            prep: Mutex::new(()),
            hdr_pool: Mutex::new(Vec::new()),
            said_encoder: AtomicBool::new(false),
        });
        let (done_tx, done_rx) = channel();
        let w = inner.clone();
        let handle = std::thread::Builder::new()
            .name("replay".into())
            .spawn(move || {
                Worker::new(w).run(rx);
                let _ = done_tx.send(());
            })
            .expect("replay thread");
        let r = Recorder { inner, worker: Mutex::new(Some((handle, done_rx))) };
        r.set_length(opts.length_s);
        r
    }

    // ---- settings
    pub fn on(&self) -> bool {
        self.length() > 0
    }

    pub fn length(&self) -> u32 {
        self.inner.sh.length.load(Ordering::SeqCst)
    }

    pub fn set_length(&self, seconds: u32) {
        let i = &self.inner;
        let was = i.sh.length.swap(seconds, Ordering::SeqCst);
        if was > 0 && seconds == 0 {
            i.send(Msg::Reset);
            i.log("[replay] off");
        } else if seconds > 0 && was == 0 {
            i.send(Msg::Prepare);
            i.log(&format!("[replay] on: keeping the last {}", dur(seconds as f64)));
            let keys = i.hotkeys.as_ref().map(|k| format!(" ({k})")).unwrap_or_default();
            i.notify(&format!("Instant replay on: R saves the last {}, B a before & after{keys}", dur(seconds as f64)), 6.0);
        } else if seconds != was {
            i.log(&format!("[replay] keeping the last {}", dur(seconds as f64)));
        }
    }

    pub fn set_clip(&self, clip: Kind) {
        self.inner.cfg.lock().unwrap().clip = clip;
    }

    pub fn set_layout(&self, layout: Layout) {
        self.inner.cfg.lock().unwrap().layout = layout;
    }

    pub fn set_mark_dir(&self, dir: Option<PathBuf>) {
        self.inner.cfg.lock().unwrap().mark_dir = dir;
    }

    pub fn set_out_dir(&self, dir: Option<PathBuf>) {
        self.inner.cfg.lock().unwrap().out_dir = dir;
    }

    pub fn set_compose_cmd(&self, cmd: Option<Vec<OsString>>) {
        self.inner.cfg.lock().unwrap().compose_cmd = cmd;
    }

    /// The stream's size, for warm_up().
    pub fn set_size(&self, size: Option<(usize, usize)>) {
        self.inner.cfg.lock().unwrap().size = size.map(|(w, h)| (w & !1, h & !1));
    }

    /// "Looks in clips", or None for clips as the game looked.
    pub fn set_look(&self, look: Option<LookFn>) {
        *self.inner.look.lock().unwrap() = look;
    }

    /// For HDR output, how HDR10 frames (RGBA16F) become the 8-bit SDR frames clips are made of. It's sp-media's
    /// pq16_to_8, as "shown in SDR" would show them. None means the default HDR settings.
    pub fn set_hdr(&self, hdr: Option<HdrSettings>) {
        *self.inner.hdr.lock().unwrap() = hdr;
    }

    /// SmoovPlay's delay, from a real frame's arrival to the screen. The before frames and the sound move by it.
    pub fn set_delay(&self, ms: f64) {
        self.inner.delay.store((ms / 1000.0).max(0.0).to_bits(), Ordering::SeqCst);
    }

    /// The rate frames are shown at. From 120 a second, every other one is kept, for 60 fps clips.
    pub fn set_output_hz(&self, hz: u32) {
        let gap: f64 = if hz >= 120 { 0.75 / 60.0 } else { 0.0 }; // 1.5 frames at 120 keeps alternate ones despite jitter
        self.inner.after_gap.store(gap.to_bits(), Ordering::SeqCst);
    }

    /// A function that returns the new sound chunks since the last call, polled on the replay's thread.
    pub fn attach_sound(&self, f: Option<SoundFn>) {
        *self.inner.sound_fn.lock().unwrap() = f;
    }

    /// Sound chunks, for a source that pushes them.
    pub fn add_sound(&self, chunks: Vec<Sound>) {
        if self.on() && !chunks.is_empty() {
            self.inner.zero(chunks[0].t);
            self.inner.send(Msg::Sound(chunks));
        }
    }

    // ---- frames
    /// A frame just shown, at t on the core's clock.
    pub fn frame(&self, frame: &Frame, t: f64) {
        let i = &self.inner;
        if !self.on() {
            return;
        }
        let gap = f64_of(&i.after_gap);
        {
            let mut last = i.after_last.lock().unwrap();
            if gap > 0.0 && last.is_some_and(|l| t - l < gap) {
                return;
            }
            *last = Some(t);
        }
        let mut f = frame.clone();
        if let Some(look) = i.look.lock().unwrap().as_mut() {
            match look(&f) {
                Ok(x) => f = x,
                Err(e) => i.sh.warn("look", &format!("[replay] can't put the look in clips ({})", trunc(&e, 200))),
            }
        }
        i.put(AFTER, &f, t);
    }

    /// A new frame from the game, which arrived at t_arrival.
    pub fn real_frame(&self, frame: &Frame, t_arrival: f64) {
        if self.on() {
            let t = t_arrival + f64_of(&self.inner.delay);
            self.inner.put(BEFORE, frame, t);
        }
    }

    /// Opens the encoders and compiles the kernels now (before play), if it's on.
    pub fn warm_up(&self) {
        if self.on() {
            let t0 = Instant::now();
            if let Err(e) = self.inner.prepare() {
                self.inner.sh.warn("prepare", &format!("[replay] can't start: {e}"));
            }
            self.inner.log(&format!("[replay] ready in {:.2} s", t0.elapsed().as_secs_f64()));
        }
    }

    /// Waits until every frame and chunk given so far is encoded.
    pub fn flush(&self, timeout: f64) -> bool {
        self.inner.flush(timeout)
    }

    /// Frames dropped because the encoder was behind (or not open yet).
    pub fn dropped(&self) -> u64 {
        self.inner.dropped.load(Ordering::SeqCst)
    }

    /// What's held, in seconds and megabytes per track.
    pub fn held(&self) -> Held {
        let t = self.inner.sh.tracks.lock().unwrap();
        let f = |k: usize| (t[k].seconds(), t[k].nbytes as f64 / 1e6);
        Held { after: f(AFTER), before: f(BEFORE), sound: f(SOUND) }
    }

    /// What it costs (see [`Stats`]).
    pub fn stats(&self) -> Stats {
        let a = self.inner.sh.acc.lock().unwrap();
        Stats { put_us: Summary::of(&a.put_us), gpu_us: Summary::of(&a.gpu_us), encode_ms: Summary::of(&a.encode_ms), packets: a.packets }
    }

    /// Before & after videos still being made (their processes' ids).
    pub fn making(&self) -> Vec<u32> {
        self.inner.making.lock().unwrap().clone()
    }

    // ---- saving
    /// Saves the replay, which is what was shown, a before & after, or both. The default is the clip setting. It
    /// saves in the background unless `wait`, which returns the files written.
    pub fn save(&self, kind: Option<Kind>, layout: Option<Layout>, wait: bool) -> Vec<PathBuf> {
        let i = &self.inner;
        let (kind, layout) = {
            let c = i.cfg.lock().unwrap();
            (kind.unwrap_or(c.clip), layout.unwrap_or(c.layout))
        };
        if !self.on() {
            i.notify("Instant replay is off: turn it on in Settings > Instant replay", 5.0);
            return Vec::new();
        }
        if wait {
            return i.save(kind, layout, false);
        }
        // close() waits for the clip itself. A before & after video carries on in its own process.
        *i.quick.0.lock().unwrap() += 1;
        let i2 = i.clone();
        let spawned = std::thread::Builder::new().name("replay-save".into()).spawn(move || {
            i2.save(kind, layout, true);
        });
        if spawned.is_err() {
            *i.quick.0.lock().unwrap() -= 1;
        }
        Vec::new()
    }

    /// Stops recording. A clip being saved is finished first, waiting up to `wait_saves` seconds. A before & after
    /// video being made carries on in its own process and isn't waited for.
    pub fn close(&self, wait_saves: f64) {
        let i = &self.inner;
        let Some((handle, done)) = self.worker.lock().unwrap().take() else { return };
        i.sh.length.store(0, Ordering::SeqCst);
        i.send(Msg::Stop);
        if done.recv_timeout(Duration::from_secs(5)).is_ok() {
            let _ = handle.join();
        }
        let end = Instant::now() + Duration::from_secs_f64(wait_saves.max(0.0));
        let mut q = i.quick.0.lock().unwrap();
        while *q > 0 {
            let now = Instant::now();
            if now >= end {
                break;
            }
            q = i.quick.1.wait_timeout(q, end - now).unwrap().0;
        }
        let unfinished = *q;
        drop(q);
        if unfinished > 0 {
            i.log(&format!("[replay] {unfinished} clip(s) weren't finished saving after {wait_saves:.0} s"));
        }
        let making = i.making.lock().unwrap().len();
        if making > 0 {
            i.log(&format!("[replay] {making} before & after video(s) still being made: they carry on in the background"));
        }
        let dropped = self.dropped();
        if dropped > 0 {
            i.log(&format!("[replay] frames dropped (the encoder fell behind): {dropped}"));
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.close(0.0);
    }
}

/// The pipeline's side (sp_sched::Replay).
impl sp_sched::Replay for Recorder {
    fn set_output_hz(&mut self, hz: u32) {
        Recorder::set_output_hz(self, hz);
    }
    fn set_delay(&mut self, ms: f64) {
        Recorder::set_delay(self, ms);
    }
    fn frame(&mut self, frame: &Frame, t: f64) {
        Recorder::frame(self, frame, t);
    }
    fn real_frame(&mut self, frame: &Frame, t: f64) {
        Recorder::real_frame(self, frame, t);
    }
    fn save(&mut self, kind: Option<&str>) {
        Recorder::save(self, kind.and_then(Kind::parse), None, false);
    }
}

/// The same through a shared reference, so the app can keep the recorder (in an Arc) while the pipeline runs.
impl sp_sched::Replay for &Recorder {
    fn set_output_hz(&mut self, hz: u32) {
        Recorder::set_output_hz(self, hz);
    }
    fn set_delay(&mut self, ms: f64) {
        Recorder::set_delay(self, ms);
    }
    fn frame(&mut self, frame: &Frame, t: f64) {
        Recorder::frame(self, frame, t);
    }
    fn real_frame(&mut self, frame: &Frame, t: f64) {
        Recorder::real_frame(self, frame, t);
    }
    fn save(&mut self, kind: Option<&str>) {
        Recorder::save(self, kind.and_then(Kind::parse), None, false);
    }
}

fn trunc(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

impl Inner {
    fn log(&self, s: &str) {
        (self.sh.log)(s);
    }

    fn notify(&self, text: &str, seconds: f64) {
        if let Some(n) = &self.notify {
            n(text, seconds);
        }
    }

    fn send(&self, m: Msg) {
        let _ = self.tx.lock().unwrap().send(m);
    }

    fn length(&self) -> u32 {
        self.sh.length.load(Ordering::SeqCst)
    }

    /// Sets the clock's zero a little before the first frame or sound. OFFSET leaves 10 s more.
    fn zero(&self, t: f64) {
        let mut z = self.t0.lock().unwrap();
        if z.is_none() {
            *z = Some(t - 5.0);
        }
    }

    /// Waits (up to `timeout`) for `done`, woken as packets come out and encoders open.
    fn wait_for(&self, timeout: f64, done: impl Fn() -> bool) -> bool {
        let end = Instant::now() + Duration::from_secs_f64(timeout);
        let mut g = self.sh.progress.0.lock().unwrap();
        while !done() {
            let now = Instant::now();
            if now >= end {
                return false;
            }
            g = self.sh.progress.1.wait_timeout(g, (end - now).min(Duration::from_millis(5))).unwrap().0;
        }
        true
    }

    fn cache(&self) -> Result<&TexCache> {
        if let Some(c) = self.cache.get() {
            return Ok(c);
        }
        let c = TexCache::new()?;
        Ok(self.cache.get_or_init(|| c))
    }

    fn prepare(&self) -> Result<()> {
        let _g = self.prep.lock().unwrap();
        gpu::kernels()?;
        self.cache()?;
        let size = self.cfg.lock().unwrap().size;
        if let Some((w, h)) = size {
            for track in [AFTER, BEFORE] {
                let have = self.encs.lock().unwrap()[track].as_ref().map(|e| (e.enc.width, e.enc.height));
                if have != Some((w, h)) {
                    self.open_track(track, w, h)?;
                }
            }
        }
        self.ready.store(true, Ordering::SeqCst);
        self.sh.poke();
        Ok(())
    }

    /// Starts a video track with a new encoder for this size. What it held goes, since it can't be cut with the new
    /// one.
    fn open_track(&self, track: usize, w: usize, h: usize) -> Result<()> {
        let (nominal, bpp) = if track == BEFORE { (BEFORE_NOMINAL, BEFORE_BPP) } else { (AFTER_NOMINAL, AFTER_BPP) };
        let config = VideoConfig { width: w, height: h, nominal, bpp, realtime: true, expected_fps: None };
        let generation = {
            let mut t = self.sh.tracks.lock().unwrap();
            t[track].clear();
            t[track].generation
        };
        let sh = self.sh.clone();
        let enc = VtEncoder::new(config, Box::new(move |r| sh.on_packet(track, generation, r)))?;
        if !self.said_encoder.swap(true, Ordering::SeqCst) {
            self.log(&format!(
                "[replay] encoding with VideoToolbox (H.264{})",
                if enc.hardware { ", on the media engine" } else { "" }
            ));
        }
        let old = self.encs.lock().unwrap()[track].replace(Arc::new(VideoEnc { enc, generation }));
        if let Some(old) = old {
            old.enc.invalidate();
            if let Some(c) = self.cache.get() {
                c.flush(); // the old pool's textures
            }
        }
        self.sh.poke();
        Ok(())
    }

    /// The track's encoder for frames of this size. None while one is being opened, which this asks for.
    fn encoder_for(&self, track: usize, w: usize, h: usize) -> Option<Arc<VideoEnc>> {
        let get = || {
            self.encs.lock().unwrap()[track].as_ref().filter(|e| (e.enc.width, e.enc.height) == (w, h)).cloned()
        };
        if let Some(e) = get() {
            return Some(e);
        }
        if !self.opening[track].swap(true, Ordering::SeqCst) {
            self.send(Msg::Open(track, w, h));
        }
        if self.block && self.wait_for(10.0, || get().is_some()) {
            return get();
        }
        None
    }

    /// Room for another frame on its way to the encoder (waiting for it when not dropping frames).
    fn slot(&self, track: usize) -> bool {
        let free = || self.sh.in_flight[track].load(Ordering::SeqCst) < POOL[track];
        free() || (self.block && self.wait_for(10.0, free))
    }

    fn put(&self, track: usize, frame: &Frame, t: f64) {
        self.zero(t);
        if !self.ready.load(Ordering::SeqCst) && !(self.block && self.wait_for(10.0, || self.ready.load(Ordering::SeqCst))) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
            return;
        }
        let t_in = Instant::now();
        let r = autoreleasepool(|_| {
            if frame.kind == PixelKind::Rgba16f { self.put_hdr(track, frame, t) } else { self.put_sdr(track, frame, t) }
        });
        push(&mut self.sh.acc.lock().unwrap().put_us, t_in.elapsed().as_secs_f64() * 1e6);
        match r {
            Ok(true) => {}
            Ok(false) => {
                self.dropped.fetch_add(1, Ordering::SeqCst);
            }
            Err(e) => self.sh.warn("convert", &format!("[replay] can't take frames ({})", trunc(&e.to_string(), 200))),
        }
    }

    /// Takes an SDR frame. It's converted on the GPU into a buffer from the encoder's pool, then goes to the replay's
    /// thread.
    fn put_sdr(&self, track: usize, frame: &Frame, t: f64) -> Result<bool> {
        let (w, h) = (frame.width & !1, frame.height & !1);
        let Some(enc) = self.encoder_for(track, w, h) else { return Ok(false) };
        if !self.slot(track) {
            return Ok(false);
        }
        let Some(buf) = enc.enc.pixel_buffer(POOL[track] + 8) else { return Ok(false) };
        let planes = self.cache()?.planes(&buf)?;
        let cb = Gpu::shared().queue.commandBuffer().ok_or_else(|| Error::Other("no command buffer".into()))?;
        gpu::encode_to_nv12(&cb, &frame.texture, w, h, &planes)?;
        cb.commit();
        self.sh.in_flight[track].fetch_add(1, Ordering::SeqCst);
        self.sh.submitted[track].fetch_add(1, Ordering::SeqCst);
        self.send(Msg::Video(VideoJob { track, t, generation: enc.generation, buf, cb, keep: Some((frame.clone(), planes)) }));
        Ok(true)
    }

    /// Takes an HDR frame. It's copied on the GPU at once, since the frame's texture may be drawn over soon, then
    /// tone-mapped and converted on the replay's thread.
    fn put_hdr(&self, track: usize, frame: &Frame, t: f64) -> Result<bool> {
        if !self.slot(track) {
            return Ok(false);
        }
        let copy = {
            let mut pool = self.hdr_pool.lock().unwrap();
            let free = pool.iter().find(|f| {
                (f.width, f.height) == (frame.width, frame.height) && f.owner.as_ref().is_some_and(|o| Arc::strong_count(o) == 1)
            });
            match free {
                Some(f) => f.clone(),
                None => {
                    let mut f = Frame::new(frame.width, frame.height, PixelKind::Rgba16f);
                    f.owner = Some(Arc::new(()));
                    if pool.len() < POOL[0] + POOL[1] + 4 {
                        pool.push(f.clone());
                    }
                    f
                }
            }
        };
        let cb = Gpu::shared().queue.commandBuffer().ok_or_else(|| Error::Other("no command buffer".into()))?;
        gpu::encode_copy(&cb, &frame.texture, &copy.texture)?;
        cb.commit();
        self.sh.in_flight[track].fetch_add(1, Ordering::SeqCst);
        self.sh.submitted[track].fetch_add(1, Ordering::SeqCst);
        self.send(Msg::Hdr(HdrJob { track, t, copy, cb, src: Some(frame.clone()) }));
        Ok(true)
    }

    /// Waits for what was handed on before this call (not what comes meanwhile, so it ends while play goes on).
    fn flush(&self, timeout: f64) -> bool {
        let want = [0, 1].map(|k| self.sh.submitted[k].load(Ordering::SeqCst));
        let seq = self.flush_seq.fetch_add(1, Ordering::SeqCst) + 1;
        self.send(Msg::Complete(seq));
        self.wait_for(timeout, || {
            self.flushed.load(Ordering::SeqCst) >= seq
                && (0..2).all(|k| self.sh.completed[k].load(Ordering::SeqCst) >= want[k])
        })
    }

    // ---- saving
    fn save(self: &Arc<Self>, kind: Kind, layout: Layout, counted: bool) -> Vec<PathBuf> {
        let handed = AtomicBool::new(false);
        let handoff = || {
            if counted && !handed.swap(true, Ordering::SeqCst) {
                *self.quick.0.lock().unwrap() -= 1;
                self.quick.1.notify_all();
            }
        };
        let out = match self.save_inner(kind, layout, &handoff) {
            Ok(v) => v,
            Err(e) => {
                self.log(&format!("[replay] save failed: {e}"));
                self.notify(&format!("Couldn't save the replay: {}", trunc(&e.to_string(), 120)), 8.0);
                Vec::new()
            }
        };
        handoff();
        out
    }

    fn save_inner(self: &Arc<Self>, kind: Kind, layout: Layout, handoff: &dyn Fn()) -> Result<Vec<PathBuf>> {
        self.flush(1.0);
        let (snap, length) = {
            let t = self.sh.tracks.lock().unwrap();
            let snap: Vec<(Option<Arc<Format>>, Vec<Piece>)> =
                t.iter().map(|tr| (tr.format.clone(), tr.ring.iter().cloned().collect())).collect();
            (snap, self.length())
        };
        let (afmt, ring) = &snap[AFTER];
        let Some(afmt) = afmt.clone().filter(|_| ring.len() >= 2) else {
            self.notify("Instant replay has nothing to save yet", 4.0);
            return Ok(Vec::new());
        };
        let end = ring.last().unwrap().pts;
        let after = cut(ring, end - length as i64 * TB, end, true);
        let base = after[0].pts;
        let secs = (end - base) as f64 / TB as f64;
        let (sfmt, sring) = &snap[SOUND];
        let sound = match sfmt {
            Some(f) if !sring.is_empty() => Some((f.clone(), cut(sring, base, end, false))),
            _ => None,
        };
        let (out_dir, mark_dir, compose_cmd) = {
            let c = self.cfg.lock().unwrap();
            (c.out_dir.clone(), c.mark_dir.clone(), c.compose_cmd.clone())
        };
        let folder = out_dir.unwrap_or_else(videos_folder);
        std::fs::create_dir_all(&folder)?;
        let stamp = stamp();
        let mut saved = Vec::new();
        if kind.video() {
            let path = unique(&folder.join(format!("SmoovPlay {stamp}.mp4")));
            let t0 = Instant::now();
            write_clip(&path, &afmt, &after, sound.as_ref(), base, end)?;
            let mb = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) as f64 / 1e6;
            self.log(&format!("[replay] saved {secs:.1} s -> {} ({mb:.0} MB, {:.1} s)", path.display(), t0.elapsed().as_secs_f64()));
            self.notify(&format!("Saved the last {}: {}", dur(secs), short(&path)), 6.0);
            saved.push(path);
        }
        if kind.compare() {
            let (bfmt, bring) = &snap[BEFORE];
            let before = cut(bring, base, end, true);
            let Some(bfmt) = bfmt.clone().filter(|_| before.len() >= 2) else {
                self.notify("No game frames to compare yet", 4.0);
                return Ok(saved);
            };
            let span_b = ((before.last().unwrap().pts - base.max(before[0].pts)) as f64 / TB as f64).max(1e-6);
            let n_b = before.iter().filter(|p| p.pts >= base).count();
            let labels = [
                format!("Original {} fps", snap_fps((n_b.saturating_sub(1)).max(1) as f64 / span_b)),
                format!("SmoovPlay {} fps", snap_fps((after.len() - 1) as f64 / secs.max(1e-6))),
            ];
            let path = unique(&folder.join(format!("SmoovPlay {stamp} before and after.mp4")));
            let tmp = temp_dir("smoovplay-replay-")?;
            // The three tracks on one timeline, as the compose step reads them
            let origin = [Some(after[0].pts), Some(before[0].pts), sound.as_ref().and_then(|s| s.1.first().map(|p| p.pts))]
                .into_iter()
                .flatten()
                .min()
                .unwrap();
            write_track(&tmp.join("after.mp4"), &afmt, &after, origin)?;
            write_track(&tmp.join("before.mp4"), &bfmt, &before, origin)?;
            let mut cmd: Vec<OsString> = match compose_cmd {
                Some(c) => c,
                None => vec![std::env::current_exe()?.into_os_string(), "replay-compose".into()],
            };
            let arg = |p: &Path| p.as_os_str().to_os_string();
            cmd.extend([
                "--after".into(),
                arg(&tmp.join("after.mp4")),
                "--before".into(),
                arg(&tmp.join("before.mp4")),
                "--out".into(),
                arg(&path),
                "--layout".into(),
                layout.name().into(),
                "--labels".into(),
                labels[0].clone().into(),
                labels[1].clone().into(),
                "--encoder".into(),
                ENCODER.into(),
                "--cleanup".into(),
                arg(&tmp),
            ]);
            if let Some((sfmt, spieces)) = &sound
                && !spieces.is_empty() {
                    write_track(&tmp.join("sound.m4a"), sfmt, spieces, origin)?;
                    cmd.extend(["--sound".into(), arg(&tmp.join("sound.m4a"))]);
                }
            if let Some(m) = &mark_dir {
                cmd.extend(["--mark".into(), arg(m)]);
            }
            self.notify(&format!("Making a before & after video of the last {} in the background. Keep playing.", dur(secs)), 6.0);
            let t0 = Instant::now();
            // A process of its own at nice 10, which carries on if play stops or SmoovPlay closes. Only its result
            // waits here.
            let mut c = std::process::Command::new(&cmd[0]);
            c.args(&cmd[1..])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            unsafe {
                use std::os::unix::process::CommandExt;
                c.pre_exec(|| {
                    libc::nice(10);
                    Ok(())
                });
            }
            let child = match c.spawn() {
                Ok(ch) => ch,
                Err(e) => {
                    let _ = std::fs::remove_dir_all(&tmp);
                    return Err(e.into());
                }
            };
            let pid = child.id();
            self.making.lock().unwrap().push(pid);
            self.log(&format!("[replay] before & after of {secs:.1} s: being made in the background (process {pid})"));
            handoff();
            let out = child.wait_with_output();
            self.making.lock().unwrap().retain(|&p| p != pid);
            if let Some(m) = &mark_dir {
                let _ = std::fs::remove_file(m.join(format!("{pid}.txt"))); // reported here, so the launcher needn't
            }
            let ok = out.as_ref().is_ok_and(|o| o.status.success());
            if ok && path.exists() {
                self.log(&format!("[replay] saved before & after -> {} ({:.0} s)", path.display(), t0.elapsed().as_secs_f64()));
                self.notify(&format!("Saved before & after: {}", short(&path)), 8.0);
                saved.push(path);
            } else {
                let (code, text) = match &out {
                    Ok(o) => (
                        o.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
                        format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)),
                    ),
                    Err(e) => ("?".into(), e.to_string()),
                };
                let tail: String = text.chars().rev().take(800).collect::<Vec<_>>().into_iter().rev().collect();
                self.log(&format!("[replay] before & after failed ({code}): {tail}"));
                self.notify("Couldn't make the before & after video (see the log)", 8.0);
                let _ = std::fs::remove_dir_all(&tmp);
            }
        }
        Ok(saved)
    }
}

/// A new, empty folder in the temp folder.
fn temp_dir(prefix: &str) -> Result<PathBuf> {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir();
    for _ in 0..100 {
        let n = N.fetch_add(1, Ordering::SeqCst);
        let p = base.join(format!("{prefix}{}-{}-{n}", std::process::id(), clock::now_ns() % 1_000_000_000));
        match std::fs::create_dir(&p) {
            Ok(()) => return Ok(p),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(Error::Other("couldn't make a temporary folder".into()))
}

fn spec_of(format: &Format, start90: i64, first_ts: i64) -> TrackSpec {
    match format {
        Format::H264 { .. } => TrackSpec { format: format.clone(), timescale: TB as u32, start: start90 },
        Format::Aac { rate, .. } => TrackSpec { format: format.clone(), timescale: *rate, start: first_ts },
    }
}

fn to48(pts90: i64) -> i64 {
    (pts90 as i128 * SOUND_RATE as i128 / TB as i128) as i64
}

fn samples(pieces: &[Piece], timescale_is_90k: bool) -> Vec<(Sample, &[u8])> {
    pieces
        .iter()
        .map(|p| (Sample { dts: if timescale_is_90k { p.pts } else { p.ts }, cto: 0, size: p.data.len() as u32, key: p.key }, &p.data[..]))
        .collect()
}

/// Writes a saved clip without encoding again (replay.py to_mp4). It has the video's pieces from `base` (a keyframe)
/// to `end` and the sound's, timed from `base`.
pub(crate) fn write_clip(
    path: &Path,
    video: &Format,
    after: &[Piece],
    sound: Option<&(Arc<Format>, Vec<Piece>)>,
    base: i64,
    end: i64,
) -> Result<()> {
    let tmp = path.with_file_name(format!("{}.part", path.file_name().unwrap().to_string_lossy()));
    let v: Vec<Piece> = after.iter().filter(|p| base <= p.pts && p.pts <= end).cloned().collect();
    let mut tracks = vec![(spec_of(video, v[0].pts - base, 0), samples(&v, true))];
    let s: Vec<Piece>;
    if let Some((sfmt, pieces)) = sound {
        s = pieces.iter().filter(|p| base <= p.pts && p.pts <= end).cloned().collect();
        if let Some(first) = s.first() {
            tracks.push((spec_of(sfmt, 0, first.ts - to48(base)), samples(&s, false)));
        }
    }
    let r = mp4::write_file(&tmp, tracks);
    if let Err(e) = r {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// One track on its own, placed on the timeline from `origin` (the before & after process's input).
fn write_track(path: &Path, format: &Format, pieces: &[Piece], origin: i64) -> Result<()> {
    let first = &pieces[0];
    let spec = spec_of(format, first.pts - origin, first.ts - to48(origin));
    let is_video = matches!(format, Format::H264 { .. });
    mp4::write_file(path, vec![(spec, samples(pieces, is_video))])?;
    Ok(())
}

// ---- the replay's thread ----------------------------------------------------------------------------------------
struct Worker {
    i: Arc<Inner>,
    aac: Option<AacEncoder>,
    sound_fmt: Option<(u32, u32)>,
    sound_gen: u64,
    clock: SoundClock,
    stereo: ToStereo48,
    converter: Option<Converter>,
}

impl Worker {
    fn new(i: Arc<Inner>) -> Worker {
        Worker { i, aac: None, sound_fmt: None, sound_gen: 0, clock: SoundClock::new(), stereo: ToStereo48::default(), converter: None }
    }

    fn run(mut self, rx: Receiver<Msg>) {
        let mut next_poll = 0.0;
        loop {
            let msg = match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(m) => Some(m),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            let got = msg.is_some();
            let stop = matches!(msg, Some(Msg::Stop));
            if let Some(m) = msg
                && !stop {
                    autoreleasepool(|_| {
                        // One bad frame or chunk mustn't stop the replay
                        if let Err(e) = self.handle(m) {
                            let key = format!("work{}", trunc(&e.to_string(), 40));
                            self.i.sh.warn(&key, &format!("[replay] {}", trunc(&e.to_string(), 200)));
                        }
                    });
                }
            if got {
                self.i.sh.poke();
            }
            if stop {
                break;
            }
            let now = clock::now();
            if self.i.length() > 0 && now >= next_poll {
                next_poll = now + 0.02;
                let chunks = match self.i.sound_fn.lock().unwrap().as_mut() {
                    Some(f) => f(),
                    None => Vec::new(),
                };
                if let Some(c) = chunks.first() {
                    self.i.zero(c.t);
                }
                for c in &chunks {
                    if let Err(e) = self.encode_sound(c) {
                        self.i.sh.warn("sound", &format!("[replay] sound: {e}"));
                    }
                }
            }
        }
        self.reset();
        self.i.sh.poke();
    }

    fn handle(&mut self, m: Msg) -> Result<()> {
        match m {
            Msg::Prepare => self.i.prepare(),
            Msg::Open(track, w, h) => {
                let r = self.i.open_track(track, w, h);
                self.i.opening[track].store(false, Ordering::SeqCst);
                self.i.sh.poke();
                r
            }
            Msg::Video(job) => self.video(job),
            Msg::Hdr(job) => self.hdr(job),
            Msg::Sound(chunks) => {
                for c in &chunks {
                    self.encode_sound(c)?;
                }
                Ok(())
            }
            Msg::Complete(seq) => {
                let encs: Vec<Arc<VideoEnc>> = self.i.encs.lock().unwrap().iter().flatten().cloned().collect();
                for e in encs {
                    e.enc.complete();
                }
                self.i.flushed.fetch_max(seq, Ordering::SeqCst);
                Ok(())
            }
            Msg::Reset => {
                self.reset();
                Ok(())
            }
            Msg::Stop => Ok(()),
        }
    }

    fn reset(&mut self) {
        let encs: Vec<Arc<VideoEnc>> = self.i.encs.lock().unwrap().iter_mut().filter_map(|e| e.take()).collect();
        for e in &encs {
            e.enc.invalidate();
        }
        drop(encs);
        let mut t = self.i.sh.tracks.lock().unwrap();
        for tr in t.iter_mut() {
            tr.clear();
        }
        drop(t);
        self.aac = None;
        self.sound_fmt = None;
        self.clock.reset();
        self.i.hdr_pool.lock().unwrap().clear();
    }

    fn wait_gpu(cb: &CommandBuffer) -> Result<()> {
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            let why = cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_default();
            return Err(Error::Other(format!("the GPU failed a frame: {why}")));
        }
        Ok(())
    }

    fn video(&mut self, mut job: VideoJob) -> Result<()> {
        let r = Self::wait_gpu(&job.cb);
        job.keep = None;
        let gpu = job.cb.GPUEndTime() - job.cb.GPUStartTime();
        push(&mut self.i.sh.acc.lock().unwrap().gpu_us, gpu * 1e6);
        if let Err(e) = r {
            self.i.sh.done_one(job.track);
            return Err(e);
        }
        self.encode(job.track, job.t, job.generation, &job.buf)
    }

    fn encode(&mut self, track: usize, t: f64, generation: u64, buf: &CVPixelBuffer) -> Result<()> {
        let i = &self.i;
        let enc = i.encs.lock().unwrap()[track].clone();
        let Some(enc) = enc.filter(|e| e.generation == generation && i.length() > 0) else {
            i.sh.done_one(track);
            return Ok(());
        };
        let t0 = i.t0.lock().unwrap().unwrap_or(t - 5.0);
        let (pts, key) = {
            let mut tr = i.sh.tracks.lock().unwrap();
            let tr = &mut tr[track];
            let mut pts = (OFFSET + py_round((t - t0) * TB as f64)).max(0);
            if let Some(last) = tr.last_pts
                && pts <= last {
                    pts = last + 1;
                }
            tr.last_pts = Some(pts);
            let key = tr.next_key.is_none_or(|k| pts >= k);
            if key {
                tr.next_key = Some(pts + py_round(KEY_EVERY_S * TB as f64));
            }
            (pts, key)
        };
        if let Err(e) = enc.enc.encode(buf, pts, key) {
            i.sh.done_one(track);
            return Err(e);
        }
        Ok(())
    }

    fn hdr(&mut self, mut job: HdrJob) -> Result<()> {
        let track = job.track;
        let r = self.hdr_inner(&mut job);
        if r.is_err() {
            self.i.sh.done_one(track);
        }
        r
    }

    fn hdr_inner(&mut self, job: &mut HdrJob) -> Result<()> {
        Self::wait_gpu(&job.cb)?;
        job.src = None;
        if self.converter.is_none() {
            self.converter = Some(Converter::new()?);
        }
        let settings = self.i.hdr.lock().unwrap().unwrap_or_else(|| HdrSettings::new(HdrOut::Sdr8, None));
        let sdr = self.converter.as_mut().unwrap().pq16_to_8(&job.copy, true, &settings);
        let (w, h) = (sdr.width & !1, sdr.height & !1);
        let have = self.i.encs.lock().unwrap()[job.track].as_ref().map(|e| (e.enc.width, e.enc.height));
        if have != Some((w, h)) {
            self.i.open_track(job.track, w, h)?;
        }
        let enc = self.i.encs.lock().unwrap()[job.track].clone().unwrap();
        let buf = enc.enc.pixel_buffer(POOL[job.track] + 8).ok_or_else(|| Error::Other("no buffer for an HDR frame".into()))?;
        let planes = self.i.cache()?.planes(&buf)?;
        let cb = Gpu::shared().queue.commandBuffer().ok_or_else(|| Error::Other("no command buffer".into()))?;
        gpu::encode_to_nv12(&cb, &sdr.texture, w, h, &planes)?;
        cb.commit();
        Self::wait_gpu(&cb)?;
        drop(planes);
        self.encode(job.track, job.t, enc.generation, &buf)
    }

    fn encode_sound(&mut self, s: &Sound) -> Result<()> {
        if self.i.length() == 0 || s.pcm.is_empty() || s.channels < 1 || s.rate == 0 {
            return Ok(());
        }
        if self.sound_fmt != Some((s.rate, s.channels)) || self.aac.is_none() {
            let aac = AacEncoder::new()?;
            let mut t = self.i.sh.tracks.lock().unwrap();
            t[SOUND].clear();
            t[SOUND].format = Some(Arc::new(Format::Aac { rate: SOUND_RATE, channels: 2, asc: aac::ASC.to_vec(), bitrate: SOUND_BITRATE }));
            self.sound_gen = t[SOUND].generation;
            drop(t);
            self.aac = Some(aac);
            self.sound_fmt = Some((s.rate, s.channels));
            self.clock.reset();
            self.stereo = ToStereo48::default();
        }
        let ch = s.channels as usize;
        let n = s.pcm.len() / ch;
        let delay = f64_of(&self.i.delay);
        let parts = {
            let mut t0 = self.i.t0.lock().unwrap();
            self.clock.place(s.t, n, s.rate, delay, &mut t0)
        };
        let mut packets = Vec::new();
        for part in parts {
            let (pts, pcm) = match part {
                Part::Silence { pts, frames } => (pts, self.stereo.convert(&vec![0i16; frames * ch], s.channels, s.rate)),
                Part::Chunk { pts, from, frames } => (pts, self.stereo.convert(&s.pcm[from * ch..(from + frames) * ch], s.channels, s.rate)),
            };
            packets.extend(self.aac.as_mut().unwrap().write(&pcm, pts));
        }
        if !packets.is_empty() {
            let length = self.i.length();
            let mut t = self.i.sh.tracks.lock().unwrap();
            if t[SOUND].generation == self.sound_gen && length > 0 {
                for p in packets {
                    let pts = py_round(p.pts as f64 * TB as f64 / SOUND_RATE as f64);
                    t[SOUND].add(Piece { pts, ts: p.pts, key: true, data: Arc::from(p.data) });
                }
                trim(&mut t, length);
            }
        }
        Ok(())
    }
}
