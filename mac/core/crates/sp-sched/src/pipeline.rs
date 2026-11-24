//! The source, repeat detection, the scheduler, the interpolator and the presenter in one loop, with latency and
//! quality stats (framegen/pipeline.py Pipeline.run). The loop follows the Python line by line, so its docstring
//! describes this one too.
//!
//! At hz 120 or 90, which needs a presenter, more frames are generated per 30 fps frame and shown at least 1/hz
//! apart. At 120, the tick a new frame B arrives after A shows the frames 1/4 and then 1/2 of the way from A to B,
//! and its repeat tick shows 3/4 and then B. At 90, the new frame's tick shows 1/3 and then 2/3, and its repeat tick
//! shows B. When the pipeline is running behind, a tick skips the extra frames and shows what 60 Hz would, the 1/2
//! frame or B, so it never builds a backlog. A 40 to 60 fps game is evened out (FrameTimeline). A 30 fps game that
//! dips is timed as its frames arrived (Dip smoothing).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use sp_core::{Event, Frame, Interpolator, Overlay, Presenter, SinkReply, Source, Tick, Waiting};

use crate::cadence::{
    Action, CadenceScheduler, CutDetector, DipWatch, DuplicateDetector, FrameDiff, FrameTimeline, RhythmLock, TimedFrame,
};
use crate::clock::{Clock, SystemClock};
use crate::delaytest::{self, DelayTest, Press};
use crate::np;
use crate::probe::{GpuProbe, Probe};
use crate::score;
use crate::settings::{Live, LiveSettings, OverlayStyle};
use crate::stats::{self, Exit, GameRate, RATE_NOTE_S, RATE_TOAST_S, ShownRate, Stats, TICK, Worst};

/// The largest real-frame difference, from 0 to 1, that Steady HUD counts as still (framegen/steady.py).
pub const STEADY_HUD_THRESHOLD: f32 = 0.03;

/// What the pipeline asks of chiaki-ng or a capture card in return: presses for the delay test, and holding the
/// sound back to meet the picture (framegen/chiaki_tap.py ChiakiSource, capture.CardSource). In the Python this is
/// the source itself. Here it is a handle to it, since the pipeline already holds the source mutably.
pub trait Remote: Press {
    /// It can press a button now (the chiaki-ng that comes with SmoovPlay, connected).
    fn can_press(&self) -> bool {
        false
    }
    /// It presses buttons at all, like Python's hasattr(remote, "press"). This picks the message to give when it
    /// can't press now.
    fn has_press(&self) -> bool {
        true
    }
    fn can_delay_audio(&self) -> bool {
        false
    }
    fn set_audio_delay(&mut self, _ms: i64) {}
}

/// Instant replay (framegen/replay.py Recorder). It gets every frame shown, each new frame from the game and the
/// delay.
pub trait Replay {
    fn set_output_hz(&mut self, _hz: u32) {}
    fn set_delay(&mut self, _ms: f64) {}
    /// A frame just put on screen, and when.
    fn frame(&mut self, frame: &Frame, t: f64);
    /// The game's own new frame (for a before & after video), and when it arrived.
    fn real_frame(&mut self, frame: &Frame, t: f64);
    /// Saves a replay. R passes None and B passes "compare".
    fn save(&mut self, kind: Option<&str>);
}

/// What one run plays with, besides the source and presenter (framegen/pipeline.py run()'s keywords).
pub struct RunOptions<'a> {
    /// Frame generation on (followed from `settings` when given).
    pub enabled: bool,
    /// With --score, each generated frame is scored against the source's ground truth and the worst is kept.
    pub score: bool,
    /// The engine's name for the overlay ("RIFE").
    pub label: String,
    /// Output rate (followed from `settings` when given).
    pub hz: u32,
    /// Followed live: interp, hz, simulate_30, steady_hud, overlay and its style, even_out, even_dips, game_fps,
    /// audio_sync.
    pub settings: Option<&'a dyn LiveSettings>,
    pub remote: Option<&'a mut dyn Remote>,
    pub replay: Option<&'a mut dyn Replay>,
    /// Called as on_tick(tick, diff, action, proc_ms, t_start) after each tick is processed, for the stream check
    /// and frame logs. It runs while the tick holds Status::stats, so neither it nor the presenter may read
    /// status().stats().
    pub on_tick: Option<&'a mut dyn FnMut(&Tick, &FrameDiff, Action, f64, f64)>,
    /// The real frame left out at a content index, for scoring. Defaults to `source.ground_truth`.
    pub gt: Option<&'a mut dyn FnMut(i64) -> Option<Frame>>,
}

impl Default for RunOptions<'_> {
    fn default() -> Self {
        RunOptions {
            enabled: true,
            score: false,
            label: String::new(),
            hz: 60,
            settings: None,
            remote: None,
            replay: None,
            on_tick: None,
            gt: None,
        }
    }
}

/// Times the Python reads from module constants, which its tests change. They live here so tests can change them
/// too.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    pub rate_note_s: f64,
    pub rate_toast_s: f64,
    pub delay_settle_s: f64,
    pub delay_gap_s: f64,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            rate_note_s: RATE_NOTE_S,
            rate_toast_s: RATE_TOAST_S,
            delay_settle_s: delaytest::SETTLE_S,
            delay_gap_s: delaytest::GAP_S,
        }
    }
}

/// What another thread can read while run() plays: when its loop last went round (for run.py's watchdog), the
/// source's Waiting while no frame comes, and the running session's Stats for reports written during play. Stats is
/// locked a tick at a time, so a reader waits at most about a tick.
#[derive(Debug, Default)]
pub struct Status {
    beat: AtomicU64,
    waiting: Mutex<Option<Waiting>>,
    pub stats: Mutex<Stats>,
}

impl Status {
    pub fn beat(&self) -> f64 {
        f64::from_bits(self.beat.load(Ordering::Relaxed))
    }
    fn set_beat(&self, t: f64) {
        self.beat.store(t.to_bits(), Ordering::Relaxed);
    }
    pub fn waiting(&self) -> Option<Waiting> {
        self.waiting.lock().unwrap().clone()
    }
    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }
}

pub type Log = Box<dyn FnMut(&str) + Send>;

pub struct Pipeline<I: Interpolator> {
    pub interp: I,
    pub detector: DuplicateDetector,
    /// See CutDetector.
    pub cut_threshold: f64,
    pub probe: Box<dyn Probe>,
    pub clock: Arc<dyn Clock>,
    pub tuning: Tuning,
    /// The engine's numbers mustn't be shared (NVIDIA VFG in the Python), so summary() leaves them out.
    pub private: bool,
    /// Where the [pipeline] lines go (Python's print).
    pub log: Log,
    status: Arc<Status>,
}

/// What showing a frame touches: the presenter, instant replay and the frame pacer.
struct Out<'s, 'r> {
    sink: Option<&'s mut dyn Presenter>,
    replay: Option<&'r mut dyn Replay>,
    clock: Arc<dyn Clock>,
    next_due: f64,
    shown_rate: ShownRate,
    overlay_on: bool,
}

impl Out<'_, '_> {
    /// The frame pacer. Each present is due one gap after the previous one was due, so sleep overshoot and present
    /// time don't add up. When late it goes straight away and the schedule restarts from now.
    fn present(&mut self, frame: &Frame, ov: &Overlay, hz: u32, gap: f64) -> SinkReply {
        if hz > 60 {
            let now = self.clock.now();
            if self.next_due > now {
                self.clock.sleep(self.next_due - now);
            }
            self.next_due = self.next_due.max(now) + gap;
        }
        let cmd = self.sink.as_deref_mut().unwrap().write(frame, self.overlay_on.then_some(ov));
        let t_now = self.clock.now();
        self.shown_rate.add(t_now);
        if let Some(r) = self.replay.as_deref_mut() {
            r.frame(frame, t_now); // after it's on screen, so it adds no delay
        }
        cmd
    }
    fn toast(&mut self, text: &str, seconds: f64) {
        if let Some(s) = self.sink.as_deref_mut() {
            s.toast(text, seconds);
        }
    }
}

fn or(cmd: SinkReply, other: SinkReply) -> SinkReply {
    if cmd != SinkReply::Continue { cmd } else { other }
}

fn opt_i64(v: Option<&serde_json::Value>) -> Option<i64> {
    v.and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
}

/// Python's bool(x) for a JSON value.
fn truthy(v: Option<&serde_json::Value>) -> bool {
    match v {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(serde_json::Value::String(s)) => !s.is_empty(),
        Some(serde_json::Value::Array(a)) => !a.is_empty(),
        Some(serde_json::Value::Object(o)) => !o.is_empty(),
    }
}

/// extra["game_new"]: True, False or None.
fn opt_bool(v: &serde_json::Value) -> Option<bool> {
    match v {
        serde_json::Value::Null => None,
        other => Some(truthy(Some(other))),
    }
}

fn fps_str(f: Option<i64>) -> String {
    f.map_or("None".into(), |v| v.to_string())
}

impl<I: Interpolator> Pipeline<I> {
    /// A `detector` of None gives the default, a fixed threshold of 2 against the previous frame. make_detector
    /// picks the right one for a source.
    pub fn new(interp: I, detector: Option<DuplicateDetector>) -> Pipeline<I> {
        Pipeline {
            interp,
            detector: detector.unwrap_or_default(),
            cut_threshold: 25.0,
            probe: Box::new(GpuProbe::new()),
            clock: Arc::new(SystemClock),
            tuning: Tuning::default(),
            private: false,
            log: Box::new(|line| println!("{line}")),
            status: Arc::new(Status::default()),
        }
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }
    pub fn with_probe(mut self, probe: Box<dyn Probe>) -> Self {
        self.probe = probe;
        self
    }
    pub fn with_log(mut self, log: Log) -> Self {
        self.log = log;
        self
    }

    /// For the watchdog and the results writer, on another thread.
    pub fn status(&self) -> Arc<Status> {
        self.status.clone()
    }

    /// Plays `source` to `sink` until the source ends or the presenter says quit or restart. A `sink` of None runs
    /// headless, with no window. Returns the session's Stats, which are also left in status().
    pub fn run<S: Source + ?Sized>(&mut self, source: &mut S, sink: Option<&mut dyn Presenter>, opts: RunOptions<'_>) -> Stats {
        let RunOptions { mut enabled, score, label, mut hz, settings, mut remote, replay, mut on_tick, mut gt } = opts;
        let clock = self.clock.clone();
        let tuning = self.tuning;
        let has_sink = sink.is_some();
        let mut live: Option<Live> = settings.map(|s| s.live());
        if let Some(l) = &live {
            enabled = l.interp;
            hz = if has_sink { l.hz } else { 60 };
        }
        let mut sched = CadenceScheduler::new();
        let status = self.status.clone();
        *status.stats.lock().unwrap() = Stats { private: self.private, ..Default::default() };
        let mut steady: Option<bool> = None; // Steady HUD as last applied to the interpolator
        let mut quarter_due = false; // at 120 Hz the previous tick interpolated, so this repeat tick owes the 3/4 frame
        let behind_ms = 1000.0 * TICK * 0.6; // start-of-tick lag above which the extra frames are skipped
        let mut gap = 0.95 / hz as f64; // time between two presents, 5% short so the schedule has slack
        let mut cuts = CutDetector::new(self.cut_threshold, true);
        let mut slot = 0usize;
        let mut key_frame: Option<Frame> = None; // last frame that arrived as new content
        let mut key_content: Option<i64> = None;
        // Scoring is on with --score, or live while "Simulate a 30 fps game" is on (the source keeps the left-out
        // frames).
        let mut scoring = score || live.as_ref().is_some_and(|l| l.simulate_30);
        status.stats.lock().unwrap().hz_used.insert(hz);
        let mut out_dev = Out {
            sink,
            replay,
            clock: clock.clone(),
            next_due: 0.0,
            shown_rate: ShownRate::new(1.0),
            overlay_on: live.as_ref().is_none_or(|l| l.overlay),
        };
        if let Some(r) = out_dev.replay.as_deref_mut() {
            r.set_output_hz(hz);
        }
        let mut key_gpu: Option<Frame> = None; // the previous real frame, for scoring
        let mut unshown: Option<f64> = None; // arrival time of the newest real frame not yet presented as itself
        let t_start = clock.now();
        let mut ov = Overlay { text: label.clone(), ..Default::default() };
        let mut ov_next = 0.0;
        let mut ov_mark = (0usize, 0u64, 0u64);
        let (mut seen_rate, mut game_rate) = (GameRate::new(120), GameRate::new(120));
        let mut delay_test: Option<DelayTest> = None;
        let mut test_mark = 0usize;
        let mut timeline: FrameTimeline<Frame> = FrameTimeline::new(12);
        let mut timed = false;
        let mut in_slot = [-1i64, -1i64];
        let mut dip_watch = DipWatch::new();
        let mut reached: Vec<TimedFrame<Frame>> = Vec::new();
        let (mut audio_next, mut audio_ms) = (0.0, None::<i64>);
        let mut replay_next = 0.0;
        let (mut rhythm, mut rhythm_moves, mut rhythm_note) = (None::<RhythmLock>, 0u64, 0.0);
        let mut off_since: Option<f64> = None;
        let mut note = String::new();
        let (mut toasted_dips, mut toasted_even, mut toasted_test) = (false, false, [false, false]);
        let (mut test_at_60, mut at30_since) = (false, None::<f64>);
        let mut last_new_at: Option<f64> = None;
        let mut out: Option<Frame> = None;
        let log = &mut self.log;
        let it = &mut self.interp;
        let probe = &mut self.probe;
        let detector = &mut self.detector;

        loop {
            let Some(event) = source.next_event() else { break };
            status.set_beat(clock.now());
            let mut st = status.stats.lock().unwrap();
            let tick = match event {
                Event::Waiting(w) => {
                    // no frame for a moment, so keep the window answering until one comes
                    *status.waiting.lock().unwrap() = Some(w.clone());
                    if let Some(s) = out_dev.sink.as_deref_mut() {
                        match s.wait(&w) {
                            SinkReply::Quit => {
                                st.exit = Exit::Quit;
                                break;
                            }
                            SinkReply::Restart => {
                                st.exit = Exit::Restart;
                                break;
                            }
                            _ => {}
                        }
                    }
                    continue;
                }
                Event::Tick(t) => t,
            };
            *status.waiting.lock().unwrap() = None;
            if tick.t_arrival < t_start {
                st.stale += 1; // queued while the previous session was restarting, and only live frames are shown
                continue;
            }
            if let Some(s) = settings {
                // pick up anything changed in the menu or by a hotkey since the last tick
                let l = s.live();
                out_dev.overlay_on = l.overlay;
                let want_hz = if has_sink { l.hz } else { 60 };
                if want_hz != hz {
                    hz = want_hz;
                    gap = 0.95 / want_hz as f64;
                    quarter_due = false;
                    ov_next = 0.0;
                    st.hz_used.insert(hz);
                    if let Some(r) = out_dev.replay.as_deref_mut() {
                        r.set_output_hz(hz);
                    }
                    log(&format!("[pipeline] output {hz} Hz"));
                }
                if (score || l.simulate_30) != scoring {
                    scoring = !scoring;
                    key_gpu = None;
                    ov_next = 0.0;
                    seen_rate.clear(); // test mode changes what the pipeline sees, but the game's own rate stays
                    off_since = None;
                    note.clear();
                    log(&format!(
                        "[pipeline] test mode (simulated 30 fps game) {}",
                        if l.simulate_30 { "ON" } else { "OFF" }
                    ));
                }
                if l.simulate_30 {
                    st.simulated += 1;
                }
                if l.interp != enabled {
                    enabled = l.interp;
                    sched = CadenceScheduler::new();
                    timeline = FrameTimeline::new(12); // afresh too: its frames are from before interpolation went off
                    quarter_due = false;
                    ov_next = 0.0;
                    log(&format!("[pipeline] interpolation {}", if enabled { "ON" } else { "OFF" }));
                }
                if steady != Some(l.steady_hud) {
                    steady = Some(l.steady_hud);
                    it.set_steady(if l.steady_hud { STEADY_HUD_THRESHOLD } else { 0.0 });
                    log(&format!("[pipeline] steady HUD {}", if l.steady_hud { "ON" } else { "OFF" }));
                }
                live = Some(l);
            }
            let gt_on = scoring;
            let t0 = clock.now();
            st.queue_ms.push((t0 - tick.t_arrival) * 1000.0);
            it.load(&tick.frame);
            let diff = detector.update(&probe.block_means(&tick.frame));
            // A source that knows better (the frame tap's flags, test mode) overrides the image check.
            let mut is_new = tick.new_hint.unwrap_or(diff.is_new);
            // Dip smoothing goes by the picture's own call. Not in test mode, nor without a window.
            dip_watch.update(is_new);
            let game_new_in = tick.extra.contains_key("game_new");
            let dipping = dip_watch.dipping
                && has_sink
                && !gt_on
                && !game_new_in
                && live.as_ref().is_some_and(|l| l.even_dips);
            let seen_new = is_new;
            // With the game frame rate set to 30 fps, every other tick is new (not in test mode)
            let lock30 = live.as_ref().is_some_and(|l| l.game_fps_30) && !game_new_in;
            if lock30 != rhythm.is_some() {
                rhythm = lock30.then(RhythmLock::new);
                log(&format!(
                    "[pipeline] game frame rate: {}",
                    if lock30 { "30 fps (every other frame is new)" } else { "Auto" }
                ));
            }
            if let Some(r) = rhythm.as_mut() {
                is_new = r.update(
                    Some(is_new),
                    opt_i64(tick.extra.get("size")),
                    truthy(tick.extra.get("keyframe")),
                    opt_i64(tick.extra.get("seq")),
                );
                if dipping {
                    is_new = seen_new; // its frames timed as they arrived, which the lock can't follow
                }
                if r.moves != rhythm_moves {
                    rhythm_moves = r.moves;
                    if t0 >= rhythm_note {
                        rhythm_note = t0 + 10.0;
                        log(&format!("[pipeline] 30 fps: the game's frames moved by a tick ({rhythm_moves} so far)"));
                    }
                }
            }
            seen_rate.add(Some(is_new));
            if is_new {
                if let Some(l) = last_new_at {
                    st.new_gaps_ms.push((tick.t_arrival - l) * 1000.0);
                }
                last_new_at = Some(tick.t_arrival);
            }
            // In test mode the source says whether the game's frame is new
            let game_new = if game_new_in { opt_bool(&tick.extra["game_new"]) } else { Some(is_new) };
            game_rate.add(game_new);
            if game_new_in {
                match game_new {
                    Some(true) => st.test_game_new += 1,
                    Some(false) => st.test_game_repeat += 1,
                    None => {}
                }
            }
            if let Some(dt) = delay_test.as_mut() {
                let blocks = probe.small(&tick.frame);
                if let Some(r) = remote.as_deref_mut() {
                    dt.feed(r, &blocks, tick.t_arrival, clock.now());
                }
                if dt.done() {
                    let ours = &st.real_delay_ms[test_mark.min(st.real_delay_ms.len())..];
                    let result = dt.result(if ours.is_empty() { None } else { Some(np::median(ours)) });
                    st.delay_tests.push(result.clone());
                    if !st.private {
                        log(&format!("[pipeline] {result}"));
                    }
                    out_dev.toast(&result.replacen("Delay test: ", "", 1), 15.0);
                    delay_test = None;
                    ov_next = 0.0;
                }
            }
            // Every 2 s, chiaki-ng is told to hold the sound back by the picture's time in SmoovPlay.
            let sync = live.as_ref().is_none_or(|l| l.audio_sync);
            if let Some(r) = remote.as_deref_mut()
                && t0 >= audio_next && (!st.real_delay_ms.is_empty() || !sync) {
                    audio_next = t0 + 2.0;
                    let want = if sync { np::round(np::median(np::tail(&st.real_delay_ms, 120))) } else { 0 };
                    if audio_ms.is_none_or(|a| (want - a).abs() >= 4) && r.can_delay_audio() {
                        if want != 0 && audio_ms.is_none_or(|a| (want - a).abs() >= 10) {
                            log(&format!("[pipeline] sound held back {want} ms to meet the picture"));
                        } else if want == 0 && audio_ms.is_some_and(|a| a != 0) {
                            log("[pipeline] sound no longer held back (Sound in sync is off)");
                        }
                        r.set_audio_delay(want);
                        audio_ms = Some(want);
                    }
                }
            // Instant replay times the game's own frames and its sound by the same delay.
            if let Some(rp) = out_dev.replay.as_deref_mut()
                && t0 >= replay_next && !st.real_delay_ms.is_empty() {
                    replay_next = t0 + 2.0;
                    rp.set_delay(np::median(np::tail(&st.real_delay_ms, 120)));
                }
            if let Some(truth) = tick.truth_new {
                let key = format!("{}{}", if is_new == truth { "t" } else { "f" }, if is_new { "p" } else { "n" });
                st.dedup.add(&key);
            }

            let mut to_score: Option<(Option<Frame>, Option<i64>)> = None;
            let mut early_cmd = SinkReply::Continue; // at 90 or 120 Hz, the first of this tick's frames, already shown
            let mut mid_t = 0.5; // the generated frame this tick ends on
            // Behind means waiting frames, not decode time.
            let t_avail = tick.t_ready.unwrap_or(tick.t_arrival);
            let behind = (t0 - t_avail) * 1000.0 >= behind_ms;
            st.behind += behind as u64;
            let extra_ok = hz > 60 && has_sink && !behind;

            macro_rules! show_early {
                ($t:expr) => {{
                    if !extra_ok {
                        st.quarter_skipped += 1;
                        false
                    } else {
                        let f = it.interpolate(slot, $t as f32);
                        early_cmd = out_dev.present(&f, &ov, hz, gap);
                        st.quarter_shown += 1;
                        true
                    }
                }};
            }

            let mut action;
            if !enabled {
                action = Action::Current;
                out = Some(tick.frame.clone());
                quarter_due = false;
                timed = false;
            } else {
                // A damaged frame is held like a cut. The `|` lets the cut detector still see every new frame.
                let is_cut = is_new && (cuts.update(diff.mean_diff, diff.cut.as_ref()) | tick.damaged);
                action = sched.step(is_new, is_cut);
                timeline.set_at_arrival(dipping);
                let added = timeline.tick(is_new, is_cut, Some(tick.frame.clone()), tick.t_arrival);
                let was_timed = timed;
                // Not before the timeline has a frame (framegen/pipeline.py): interpolation turned on at a repeat
                // tick, while the game was already dipping, asked place() for frames it didn't have.
                timed = !timeline.frames.is_empty()
                    && (dipping || (timeline.uneven && has_sink && !gt_on && live.as_ref().is_none_or(|l| l.even_out)));
                if timed != was_timed {
                    if timed && dipping {
                        unshown = None;
                        log("[pipeline] game dipping below 30 fps: timing its frames as they arrived");
                        if !toasted_dips {
                            toasted_dips = true;
                            out_dev.toast(
                                "Game dipping below 30 fps: SmoovPlay is spacing its frames by when they arrived (Dip \
                                 smoothing)",
                                6.0,
                            );
                        }
                    } else if timed {
                        unshown = None;
                        log(&format!("[pipeline] game at about {} fps: evening out its frames", fps_str(timeline.fps())));
                        if !toasted_even {
                            toasted_even = true;
                            out_dev.toast(
                                &format!(
                                    "Game at about {} fps: SmoovPlay is spacing its frames evenly (Even out 40-60 fps)",
                                    fps_str(timeline.fps())
                                ),
                                6.0,
                            );
                        }
                    } else {
                        sched = CadenceScheduler::new(); // starts afresh, from the next new frame
                        action = sched.step(is_new, is_cut);
                        log("[pipeline] even timing off");
                    }
                }
                if is_new && !timed {
                    slot = 1 - slot;
                    it.prepare(slot); // the input holds this tick's frame
                    in_slot[slot] = added.as_ref().map_or(-1, |a| a.id);
                }
                if timed {
                    if dipping {
                        st.dip_ticks += 1;
                    } else {
                        st.uneven_ticks += 1;
                        if is_new {
                            st.uneven_fps.push(timeline.fps().unwrap_or(0));
                        }
                    }
                    let mut phases = timeline.slots(hz); // quarter_due is set below, as for any tick
                    if phases.len() > 1 && !extra_ok {
                        st.quarter_skipped += (phases.len() - 1) as u64;
                        phases = vec![*phases.last().unwrap()];
                    } else {
                        st.quarter_shown += phases.len().saturating_sub(1) as u64;
                    }
                    let last = phases.len().saturating_sub(1);
                    for (k, &phi) in phases.iter().enumerate() {
                        let pl = timeline.place(phi).expect("even timing has frames");
                        let frame = match &pl.b {
                            None => pl.a.frame.clone().expect("timed frames carry theirs"),
                            Some(b) => {
                                slot = Self::timed_pair(it, &pl.a, b, slot, &mut in_slot);
                                it.interpolate(slot, pl.t as f32)
                            }
                        };
                        st.uneven_waits += pl.waiting as u64;
                        if k < last {
                            early_cmd = or(out_dev.present(&frame, &ov, hz, gap), early_cmd);
                            let t_now = clock.now();
                            st.real_delay_ms.extend(pl.reached.iter().map(|f| (t_now - f.t_arrival) * 1000.0));
                        } else {
                            out = Some(frame);
                            action = if pl.b.is_some() { Action::Interp } else { Action::Current };
                            reached = pl.reached;
                        }
                    }
                } else if hz == 120 && action == Action::Interp {
                    show_early!(0.25);
                } else if hz == 120 && quarter_due && action == Action::Current && !is_new {
                    show_early!(0.75);
                } else if hz == 90 && action == Action::Interp && show_early!(1.0 / 3.0) {
                    mid_t = 2.0 / 3.0;
                }
                quarter_due = action == Action::Interp;
                if timed {
                    // shown above
                } else if action == Action::Interp {
                    out = Some(it.interpolate(slot, mid_t as f32));
                    // --score has ground truth only at the midpoint
                    if gt_on && key_gpu.is_some() && mid_t == 0.5 {
                        to_score = Some((key_frame.clone(), key_content)); // scored once the frame is on screen
                    }
                } else if matches!(action, Action::Pending | Action::Hold) {
                    out = key_frame.clone().or_else(|| Some(tick.frame.clone()));
                } else if rhythm.is_some() && key_frame.is_some() {
                    // Game frame rate is 30 fps and the lock called this tick a repeat. Showing the frame the lock
                    // last called new keeps the order, a tick late, after a game held a frame for a third tick.
                    out = key_frame.clone();
                } else {
                    out = Some(tick.frame.clone());
                }
            }
            let prev_key = key_gpu.clone();
            if is_new {
                key_frame = Some(tick.frame.clone());
                key_content = tick.content;
                if gt_on {
                    key_gpu = Some(tick.frame.clone());
                }
            }

            let t_ready = clock.now();
            let last_proc = (t_ready - t0) * 1000.0;
            if action == Action::Interp {
                st.proc_interp_ms.push(last_proc);
            } else {
                st.proc_other_ms.push(last_proc);
            }
            st.actions.add(action.as_str());
            if let Some(cb) = on_tick.as_deref_mut() {
                cb(&tick, &diff, action, last_proc, t0);
            }

            let mut cmd = SinkReply::Continue;
            if has_sink {
                if t_ready >= ov_next {
                    let mut test = live.as_ref().is_some_and(|l| l.simulate_30);
                    let even = live.as_ref().is_none_or(|l| l.even_out);
                    let game_fps = game_rate.fps();
                    // Simulate on a game that already runs at about 30 fps plays it at 15, so after RATE_TOAST_S it
                    // goes off by itself. Not when the game has been at 60 since Simulate went on, because then it's
                    // a 30 fps cutscene.
                    test_at_60 = test && (test_at_60 || game_fps.is_some_and(|g| g >= 50.0));
                    let at30 = test && !test_at_60 && game_fps.is_some_and(|g| (24.0..38.5).contains(&g));
                    at30_since = if !at30 { None } else { Some(at30_since.unwrap_or(t_ready)) };
                    if at30 && t_ready - at30_since.unwrap() >= tuning.rate_toast_s {
                        if let Some(s) = settings {
                            s.set_simulate_30(false);
                        }
                        if let Some(l) = live.as_mut() {
                            l.simulate_30 = false;
                        }
                        test = false;
                        at30_since = None;
                        toasted_test[1] = true; // its own message instead of the advice
                        let said = stats::simulate_off(np::round(game_fps.unwrap()));
                        log(&format!("[pipeline] {said}"));
                        out_dev.toast(&said, 8.0);
                    }
                    let advice = if enabled || test { stats::rate_advice(game_fps, test, even) } else { String::new() };
                    if advice.is_empty() {
                        off_since = None;
                        note.clear();
                    } else {
                        let since = *off_since.get_or_insert(t_ready);
                        note = if t_ready - since >= tuning.rate_note_s { advice.clone() } else { String::new() };
                        if t_ready - since >= tuning.rate_toast_s && !toasted_test[test as usize] {
                            toasted_test[test as usize] = true; // once a session for each
                            log(&format!("[pipeline] {advice}"));
                            out_dev.toast(&advice, 8.0);
                        }
                    }
                    let ctx = OverlayCtx {
                        st: &st,
                        live: live.as_ref(),
                        label: &label,
                        enabled,
                        hz,
                        timed,
                        note: &note,
                        delay_test: delay_test.as_ref(),
                        seen_fps: seen_rate.fps(),
                        shown_fps: out_dev.shown_rate.fps(),
                    };
                    ov = ctx.overlay(&mut ov_mark);
                    ov_next = t_ready + 0.25;
                }
                let frame = out.clone().unwrap_or_else(|| tick.frame.clone());
                cmd = or(out_dev.present(&frame, &ov, hz, gap), early_cmd);
            }
            let t_shown = clock.now();
            st.busy_ms.push((t_shown - t0) * 1000.0);
            if has_sink {
                st.sink_ms.push((t_shown - t_ready) * 1000.0);
            }

            // When each real frame first reaches the screen. With even timing, that's when the game time shown
            // reaches it.
            if timed {
                let r = std::mem::take(&mut reached);
                st.real_delay_ms.extend(r.iter().map(|f| (t_shown - f.t_arrival) * 1000.0));
            } else if action == Action::Pending && unshown.is_some() {
                st.real_delay_ms.push((t_shown - unshown.unwrap()) * 1000.0);
                unshown = None;
            } else if is_new {
                if action == Action::Current {
                    st.real_delay_ms.push((t_shown - tick.t_arrival) * 1000.0);
                } else {
                    unshown = Some(tick.t_arrival);
                }
            } else if let Some(u) = unshown {
                st.real_delay_ms.push((t_shown - u) * 1000.0);
                unshown = None;
            }

            if is_new
                && let Some(r) = out_dev.replay.as_deref_mut() {
                    r.real_frame(&tick.frame, tick.t_arrival); // the game's own frame, for a before & after video
                }

            if let Some((prev, prev_content)) = to_score {
                let generated = out.clone().unwrap();
                let mut ground = |c: i64| match gt.as_deref_mut() {
                    Some(g) => g(c),
                    None => source.ground_truth(c),
                };
                score_frame(probe.as_mut(), &mut st, &tick, prev_key, prev, prev_content, &generated, &mut ground, score);
                st.score_ms.push((clock.now() - t_shown) * 1000.0);
            }

            if has_sink {
                match cmd {
                    SinkReply::Quit => {
                        st.exit = Exit::Quit;
                        break;
                    }
                    SinkReply::Restart => {
                        st.exit = Exit::Restart;
                        break;
                    }
                    SinkReply::DelayTest if delay_test.is_none() => match remote.as_deref_mut() {
                        Some(r) if r.can_press() => {
                            let mut dt = DelayTest::new(delaytest::PRESSES, delaytest::BUTTON_OPTIONS, clock.now());
                            dt.settle_s = tuning.delay_settle_s;
                            dt.gap_s = tuning.delay_gap_s;
                            delay_test = Some(dt);
                            test_mark = st.real_delay_ms.len();
                            ov_next = 0.0;
                            log("[pipeline] delay test started");
                        }
                        r => {
                            let msg = if r.as_ref().is_none_or(|r| !r.has_press()) {
                                "The delay test works with PS5 Remote Play, through the chiaki-ng that comes with this \
                                 SmoovPlay"
                            } else {
                                "The delay test needs the chiaki-ng that comes with this SmoovPlay: open it from there"
                            };
                            log(&format!("[pipeline] {msg}"));
                            out_dev.toast(msg, 6.0);
                        }
                    },
                    SinkReply::SaveReplay | SinkReply::SaveCompare => match out_dev.replay.as_deref_mut() {
                        Some(r) => r.save(if cmd == SinkReply::SaveCompare { Some("compare") } else { None }),
                        None => out_dev.toast("Instant replay isn't available here", 4.0),
                    },
                    _ => {}
                }
            }
        }

        let mut st = status.stats.lock().unwrap();
        st.wall_s = clock.now() - t_start;
        st.clone()
    }

    /// For even timing, puts frames a and b in the interpolator's two slots, b the newer, loading whichever isn't
    /// there yet. Returns b's slot. It overwrites the input, so call it only after this tick's frame has been looked
    /// at.
    fn timed_pair(it: &mut I, a: &TimedFrame<Frame>, b: &TimedFrame<Frame>, mut slot: usize, in_slot: &mut [i64; 2]) -> usize {
        if in_slot[slot] == b.id && in_slot[1 - slot] == a.id {
            return slot;
        }
        if in_slot[slot] != a.id {
            if in_slot[1 - slot] == a.id {
                slot = 1 - slot; // a is already the older of the two, so b goes in after it
            } else {
                slot = 1 - slot;
                it.load(a.frame.as_ref().unwrap());
                it.prepare(slot);
                in_slot[slot] = a.id;
            }
        }
        slot = 1 - slot;
        it.load(b.frame.as_ref().unwrap());
        it.prepare(slot);
        in_slot[slot] = b.id;
        slot
    }
}

/// Scores one generated frame (pipeline.py _score). `a` is the previous real frame and `tick.frame` this tick's.
/// Only clean 2-tick steps are scored, where the in-between source frame exists.
#[allow(clippy::too_many_arguments)]
fn score_frame(
    probe: &mut dyn Probe,
    st: &mut Stats,
    tick: &Tick,
    a: Option<Frame>,
    prev: Option<Frame>,
    prev_content: Option<i64>,
    generated: &Frame,
    gt: &mut dyn FnMut(i64) -> Option<Frame>,
    keep_worst: bool,
) {
    let (Some(pc), Some(c)) = (prev_content, tick.content) else { return };
    if c - pc != 2 {
        return;
    }
    let Some(truth) = gt(pc + 1) else { return };
    let Some(a) = a else { return };
    let sums = probe.score(&a, &tick.frame, generated, &truth);
    let p = score::add_scores(st, &sums);
    if keep_worst && st.worst.as_ref().is_none_or(|w| p < w.psnr) {
        let sdr = [prev.as_ref(), Some(generated), Some(&truth), Some(&tick.frame)]
            .iter()
            .all(|f| f.is_some_and(|f| f.kind == sp_core::PixelKind::Bgra8));
        let images = if sdr {
            vec![prev.unwrap().to_bgr(), generated.to_bgr(), truth.to_bgr(), tick.frame.to_bgr()]
        } else {
            Vec::new()
        };
        st.worst = Some(Worst { psnr: p, tick: tick.index, width: truth.width, height: truth.height, images });
    }
}

/// What the overlay is made from (pipeline.py overlay_text, mini_text and test_line).
struct OverlayCtx<'a> {
    st: &'a Stats,
    live: Option<&'a Live>,
    label: &'a str,
    enabled: bool,
    hz: u32,
    timed: bool,
    note: &'a str,
    delay_test: Option<&'a DelayTest>,
    seen_fps: Option<f64>,
    shown_fps: Option<f64>,
}

impl OverlayCtx<'_> {
    /// Test mode's badge, and quality over up to the last 30 scored frames.
    fn test_line(&self) -> String {
        let mut text = String::from("TEST: 60 fps game played as 30 fps");
        let ai = np::tail(&self.st.off_interp, 30);
        let rep = np::tail(&self.st.off_repeat, 30);
        let n = ai.len().min(rep.len());
        if self.enabled && self.hz == 90 {
            text += "  quality: measured at 60 and 120 fps";
        } else if self.enabled && n > 0 {
            text += &format!(
                "  quality: {} visibly off from the real frames (no AI {})",
                stats::pct(np::mean(&ai[..n])),
                stats::pct(np::mean(&rep[..n]))
            );
        }
        text
    }

    fn overlay(&self, ov_mark: &mut (usize, u64, u64)) -> Overlay {
        let st = self.st;
        let (n0, s0, k0) = *ov_mark;
        *ov_mark = (st.proc_interp_ms.len(), st.quarter_shown, st.quarter_skipped);
        let (g, s) = (self.seen_fps, self.shown_fps);
        let proc = &st.proc_interp_ms[n0.min(st.proc_interp_ms.len())..];
        let (shown, skipped) = (st.quarter_shown - s0, st.quarter_skipped - k0);
        let testing = self.live.is_some_and(|l| l.simulate_30);
        if self.live.is_some_and(|l| l.overlay_style == OverlayStyle::Minimal) {
            return self.mini(g, s, proc, shown, skipped, testing);
        }
        let rates = match (g, s) {
            (Some(g), Some(s)) => format!(
                "game {} fps{}, showing {}",
                np::f(g, 0),
                if self.timed { " evened out" } else { "" },
                np::f(s, 0)
            ),
            _ => String::new(),
        };
        let mut text = if !self.enabled {
            format!("{}  INTERP OFF{}", self.label, if rates.is_empty() { String::new() } else { format!("  {rates}") })
        } else {
            let mut t = format!(
                "{}  INTERP ON  {}  interp {} ms",
                self.label,
                if rates.is_empty() { format!("{} Hz", self.hz) } else { rates },
                np::fw(if proc.is_empty() { 0.0 } else { np::mean(proc) }, 5, 1)
            );
            if self.hz > 60 && shown + skipped > 0 {
                t += &format!("  extra frames {}%", np::fw(100.0 * shown as f64 / (shown + skipped) as f64, 3, 0));
            }
            t
        };
        if let Some(dt) = self.delay_test {
            text += &format!("\n{}", dt.status()); // the delay test, while it runs, is the second line
        } else if !self.note.is_empty() {
            text += &format!("\n{}", self.note);
        } else if testing {
            text += &format!("\n{}", self.test_line());
        }
        Overlay { text, minimal: false, ..Default::default() }
    }

    /// Stats overlay style Minimal: frames on screen, the game's rate (amber while it's off the rate that's needed),
    /// SmoovPlay's delay, and how well the Mac keeps up.
    fn mini(&self, g: Option<f64>, s: Option<f64>, proc: &[f64], shown: u64, skipped: u64, testing: bool) -> Overlay {
        let mut parts = vec![(
            match s {
                Some(s) => format!("{} fps", np::f(s, 0)),
                None => format!("{} fps", self.hz),
            },
            "main".to_string(),
        )];
        if let Some(g) = g {
            parts.push((format!("game {}", np::f(g, 0)), if self.note.is_empty() { "dim" } else { "warn" }.into()));
        }
        if !self.st.real_delay_ms.is_empty() {
            parts.push((format!("delay {} ms", np::f(np::median(np::tail(&self.st.real_delay_ms, 120)), 0)), "dim".into()));
        }
        let health = if !self.enabled {
            "off"
        } else if self.hz > 60 && shown + skipped > 0 {
            let f = shown as f64 / (shown + skipped) as f64;
            if f >= 0.95 { "good" } else if f >= 0.8 { "tight" } else { "behind" }
        } else if !proc.is_empty() {
            let load = np::mean(proc) / (TICK * 1000.0);
            if load < 0.75 { "good" } else if load < 0.95 { "tight" } else { "behind" }
        } else {
            ""
        };
        let second = match self.delay_test {
            Some(dt) => dt.status(),
            None if testing => {
                if self.note.is_empty() { self.test_line() } else { self.note.to_string() }
            }
            None => String::new(),
        };
        let mut text = parts.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("  ");
        if !second.is_empty() {
            text += &format!("\n{second}");
        }
        Overlay { text, parts, health: health.into(), second, minimal: true }
    }
}
