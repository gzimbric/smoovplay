//! Repeat and cut detection and output scheduling for a 60 Hz signal carrying 30 fps content (framegen/cadence.py).
//!
//! A console's video runs at 60 Hz, so a 30 fps game arrives as A A B B C C. At 60 Hz output we emit
//! A A mid(A,B) B mid(B,C) C, one tick behind the input. On the tick a new frame arrives we show the midpoint between
//! it and the previous real frame, and on its repeat tick the real frame itself. At 90 and 120 Hz two or three frames
//! are drawn between each pair (see the pipeline).
//!
//! The scheduler also handles breaks in the cadence. A game hitch's extra repeat is shown again. Real 60 fps content
//! passes through a tick late. Scene cuts are held, because interpolating across a cut ghosts two shots together.
//!
//! Names, constants and arithmetic follow the Python line by line, so its docs and tests describe these too.

use std::collections::VecDeque;

use sp_core::stats::BlockMeans;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Show the midpoint of the previous real frame and the current one.
    Interp,
    /// Show the incoming frame as it is.
    Current,
    /// Show the previous new frame, not yet displayed. The current one becomes pending.
    Pending,
    /// Show the previous real frame again (scene cut).
    Hold,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::Interp => "interp",
            Action::Current => "current",
            Action::Pending => "pending",
            Action::Hold => "hold",
        }
    }
}

/// How differently two frames are built, whatever their brightness. This catches cuts between dark shots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutFeatures {
    /// 1 - correlation of the two frames' block brightness, from 0 (same layout) to 2 (inverted).
    pub corr_dist: f64,
    /// Half the L1 distance between brightness histograms, from 0 (same) to 1 (no overlap).
    pub hist_dist: f64,
    /// This frame's mean brightness, 0 to 255.
    pub luma: f64,
    /// This frame's brightness standard deviation over blocks, 0 to 255.
    pub contrast: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameDiff {
    pub is_new: bool,
    /// Max 16x16 block-mean difference, 0 to 255.
    pub max_diff: f64,
    /// Mean absolute difference over blocks, 0 to 255.
    pub mean_diff: f64,
    /// None on the first frame.
    pub cut: Option<CutFeatures>,
}

const LUMA: [f32; 3] = [0.114, 0.587, 0.299]; // BGR to Y (BT.601)
pub const HIST_BINS: usize = 32;

fn lumas(m: &BlockMeans) -> Vec<f32> {
    m.data.as_chunks::<3>().0.iter().map(|p| p[0] * LUMA[0] + p[1] * LUMA[1] + p[2] * LUMA[2]).collect()
}

fn mean32(xs: &[f32]) -> f32 {
    (xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len().max(1) as f64) as f32
}

fn histc(ys: &[f32]) -> [f32; HIST_BINS] {
    // As torch.histc(y, 32, 0, 255). Values outside [0, 255] are left out and 255 itself goes in the last bin.
    let mut h = [0f32; HIST_BINS];
    for &y in ys {
        if !(0.0..=255.0).contains(&y) {
            continue;
        }
        let pos = ((y * HIST_BINS as f32 / 255.0) as usize).min(HIST_BINS - 1);
        h[pos] += 1.0;
    }
    let n = ys.len().max(1) as f32;
    h.map(|c| c / n)
}

/// CutFeatures from two block-mean images (framegen/cadence.py _cut_stats), in float32 as torch has them.
pub fn cut_features(prev: &BlockMeans, cur: &BlockMeans) -> CutFeatures {
    let (yp, yc) = (lumas(prev), lumas(cur));
    let (mp, mc) = (mean32(&yp), mean32(&yc));
    let dp: Vec<f32> = yp.iter().map(|y| y - mp).collect();
    let dc: Vec<f32> = yc.iter().map(|y| y - mc).collect();
    let sp = mean32(&dp.iter().map(|d| d * d).collect::<Vec<_>>()).sqrt();
    let sc = mean32(&dc.iter().map(|d| d * d).collect::<Vec<_>>()).sqrt();
    // Correlation is undefined for a flat frame (a black screen). Two flat frames count as alike, and a flat frame
    // next to a detailed one as fully different.
    let denom = (sp * sc).max(1e-6);
    let mut corr = mean32(&dp.iter().zip(&dc).map(|(a, b)| a * b).collect::<Vec<_>>()) / denom;
    if sp < 1.0 || sc < 1.0 {
        corr = if sp < 1.0 && sc < 1.0 { 1.0 } else { 0.0 };
    }
    let (hp, hc) = (histc(&yp), histc(&yc));
    let hist: f32 = hp.iter().zip(&hc).map(|(a, b)| (a - b).abs()).sum();
    CutFeatures { corr_dist: (1.0 - corr) as f64, hist_dist: (0.5 * hist) as f64, luma: mc as f64, contrast: sc as f64 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// Each frame is compared with the frame before it.
    Previous,
    /// Each frame is compared with the last frame found to be new content (capture cards).
    New,
}

/// Classifies each incoming frame as new content or a repeat of the previous one, from its 16x16 block means.
/// This is framegen/cadence.py DuplicateDetector, and the reference, learn and rhythm work as documented there.
#[derive(Clone, Debug)]
pub struct DuplicateDetector {
    pub threshold: f64,
    pub reference: Reference,
    pub learn: bool,
    means: VecDeque<f64>,
    prev: Option<BlockMeans>,
    calls: VecDeque<bool>,
    /// (period, where the next new frame falls in it), once steady.
    pub rhythm: Option<(usize, usize)>,
    /// Calls the rhythm made.
    pub held: u64,
}

impl Default for DuplicateDetector {
    fn default() -> Self {
        Self::new(2.0, Reference::Previous, false)
    }
}

impl DuplicateDetector {
    pub const LEARN_N: usize = 16;
    pub const LEARN_MIN: usize = 8;
    pub const LEARN_RATIO: f64 = 4.0;
    pub const RHYTHM_N: usize = 24;
    pub const RHYTHM_PERIODS: [usize; 2] = [2, 3];
    pub const SURE: f64 = 0.5;

    pub fn new(threshold: f64, reference: Reference, learn: bool) -> Self {
        DuplicateDetector {
            threshold,
            reference,
            learn,
            means: VecDeque::with_capacity(Self::LEARN_N),
            prev: None,
            calls: VecDeque::with_capacity(Self::RHYTHM_N),
            rhythm: None,
            held: 0,
        }
    }

    /// The two-band rule's call on a frame's mean difference and how sure it is, or None without two bands.
    fn learned(&self, mean: f64) -> Option<(bool, f64)> {
        if self.means.len() < Self::LEARN_MIN {
            return None;
        }
        let mut xs: Vec<f64> = self.means.iter().copied().collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = (xs.len() - 1) as f64;
        let lo = xs[(0.2 * k).round_ties_even() as usize].max(1e-3);
        let hi = xs[(0.8 * k).round_ties_even() as usize];
        if hi <= Self::LEARN_RATIO * lo {
            return None;
        }
        let line = (lo * hi).sqrt();
        Some((mean > line, (mean.max(1e-6) / line).ln().abs() / (hi / line).ln()))
    }

    /// The call after the game's rhythm has had its say (learn only), and the rhythm brought up to date.
    fn keep_rhythm(&mut self, mut is_new: bool, sure: Option<f64>) -> bool {
        if let Some((period, phase)) = self.rhythm {
            let expected = phase == 0;
            if sure.is_none() || (expected != is_new && sure.unwrap() >= Self::SURE) {
                self.rhythm = None; // no bands, or a clear exception, so go by the pictures alone for a while
                self.calls.clear();
            } else {
                if expected != is_new {
                    self.held += 1;
                    is_new = expected;
                }
                self.rhythm = Some((period, (phase + 1) % period));
            }
        }
        if self.calls.len() == Self::RHYTHM_N {
            self.calls.pop_front();
        }
        self.calls.push_back(is_new);
        if self.rhythm.is_none() && sure.is_some() && self.calls.len() == Self::RHYTHM_N {
            let calls: Vec<bool> = self.calls.iter().copied().collect();
            for period in Self::RHYTHM_PERIODS {
                let first = calls.iter().position(|&c| c);
                if let Some(first) = first
                    && first < period
                        && calls.iter().enumerate().all(|(i, &c)| c == ((i as i64 - first as i64).rem_euclid(period as i64) == 0))
                    {
                        self.rhythm = Some((period, (calls.len() - first) % period));
                        break;
                    }
            }
        }
        is_new
    }

    /// One frame's block means (from [`crate::probe::Probe::block_means`]).
    pub fn update(&mut self, small: &BlockMeans) -> FrameDiff {
        let prev = match &self.prev {
            Some(p) if p.rows == small.rows && p.cols == small.cols => p,
            _ => {
                self.prev = Some(small.clone());
                return FrameDiff { is_new: true, max_diff: f64::INFINITY, mean_diff: f64::INFINITY, cut: None };
            }
        };
        let mut amax = 0f32;
        let mut total = 0f64;
        for (a, b) in small.data.iter().zip(&prev.data) {
            let d = (a - b).abs();
            amax = amax.max(d);
            total += d as f64;
        }
        let max_diff = amax as f64;
        let mean_diff = (total / small.data.len().max(1) as f64) as f32 as f64; // torch's float32 mean
        let cut = cut_features(prev, small);
        let mut is_new = max_diff > self.threshold;
        if self.learn {
            let learned = self.learned(mean_diff);
            if self.means.len() == Self::LEARN_N {
                self.means.pop_front();
            }
            self.means.push_back(mean_diff);
            if let Some((call, _)) = learned {
                is_new = call;
            }
            is_new = self.keep_rhythm(is_new, learned.map(|l| l.1));
        }
        if is_new || self.reference == Reference::Previous {
            self.prev = Some(small.clone());
        }
        FrameDiff { is_new, max_diff, mean_diff, cut: Some(cut) }
    }
}

/// The repeat detector for a source (framegen/sources.py make_detector). A capture card compares with the last new
/// frame. PS5 Remote Play, an app's window and the frame tap learn where the line between repeats and new frames is.
pub fn make_detector(info: &sp_core::SourceInfo, threshold: f64) -> DuplicateDetector {
    DuplicateDetector::new(
        threshold,
        if info.kind == "capture" { Reference::New } else { Reference::Previous },
        matches!(info.kind.as_str(), "chiaki" | "window" | "tap"),
    )
}

/// For a game frame rate set to 30 fps. Every other tick is new, and the checks only vote on which of the two ticks
/// it is (framegen/cadence.py RhythmLock).
#[derive(Clone, Debug, Default)]
pub struct RhythmLock {
    votes: VecDeque<(i64, bool)>,
    n: i64,
    last: Option<(i64, i64)>,
    /// New frames fall on ticks whose number % 2 == phase.
    pub phase: Option<i64>,
    /// Times the phase moved after it was first found.
    pub moves: u64,
}

impl RhythmLock {
    pub const WINDOW: usize = 32;
    pub const MARGIN: i64 = 10;
    pub const MIN: usize = 8;

    pub fn new() -> Self {
        Self::default()
    }

    /// `call` is the checks' new-or-repeat call, `size` the compressed size (Remote Play) and `seq` the stream's
    /// frame number.
    pub fn update(&mut self, call: Option<bool>, size: Option<i64>, keyframe: bool, seq: Option<i64>) -> bool {
        let n = seq.unwrap_or(self.n);
        self.n = n + 1;
        let looks = if let Some(size) = size {
            let prev = self.last;
            let looks = match prev {
                Some((pn, ps)) if !keyframe && pn == n - 1 => Some(size > ps),
                _ => None,
            };
            self.last = if keyframe { None } else { Some((n, size)) };
            looks
        } else {
            call
        };
        if let Some(l) = looks {
            if self.votes.len() == Self::WINDOW {
                self.votes.pop_front();
            }
            self.votes.push_back((n, l));
        }
        if self.votes.len() >= Self::MIN {
            let agree: Vec<i64> =
                (0..2).map(|p| self.votes.iter().filter(|&&(i, v)| v == (i.rem_euclid(2) == p)).count() as i64).collect();
            let best = if agree[0] >= agree[1] { 0 } else { 1 };
            match self.phase {
                None => self.phase = Some(best),
                Some(ph) if best != ph && agree[best as usize] - agree[ph as usize] >= Self::MARGIN => {
                    self.phase = Some(best);
                    self.moves += 1;
                }
                _ => {}
            }
        }
        match self.phase {
            None => call.unwrap_or(true), // the first few ticks go by the checks
            Some(p) => n.rem_euclid(2) == p,
        }
    }
}

/// Flags scene cuts among new frames (framegen/cadence.py CutDetector).
#[derive(Clone, Debug)]
pub struct CutDetector {
    pub threshold: f64,
    pub use_structure: bool,
    pub prev: f64,
    pub prev_corr: f64,
}

impl Default for CutDetector {
    fn default() -> Self {
        Self::new(25.0, true)
    }
}

impl CutDetector {
    /// The corr_dist jump that marks a cut, when the frame also changed by more than MIN_DIFF.
    pub const STRUCTURE: f64 = 0.75;
    /// The mean_diff a frame needs on top of a STRUCTURE jump to count as a cut, so noise alone never cuts.
    pub const MIN_DIFF: f64 = 6.0;

    pub fn new(threshold: f64, use_structure: bool) -> Self {
        CutDetector { threshold, use_structure, prev: 0.0, prev_corr: 0.0 }
    }

    pub fn update(&mut self, mean_diff: f64, f: Option<&CutFeatures>) -> bool {
        let score = mean_diff.min(mean_diff - self.prev);
        self.prev = mean_diff;
        let mut cut = score > self.threshold;
        if let (true, Some(f)) = (self.use_structure, f) {
            let jump = f.corr_dist.min(f.corr_dist - self.prev_corr);
            self.prev_corr = f.corr_dist;
            cut = cut || (jump > Self::STRUCTURE && mean_diff > Self::MIN_DIFF);
        }
        cut
    }
}

/// A pure state machine that turns each tick's (is_new, is_cut) into an Action.
#[derive(Clone, Debug, Default)]
pub struct CadenceScheduler {
    pub started: bool,
    /// A new frame has been received but not yet displayed as itself.
    pub pending: bool,
}

impl CadenceScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step(&mut self, is_new: bool, is_cut: bool) -> Action {
        if !is_new {
            self.pending = false;
            return Action::Current; // duplicate of the latest real frame
        }
        if !self.started {
            self.started = true;
            return Action::Current;
        }
        if self.pending {
            return Action::Pending; // back-to-back new frames, so the source is over 30 fps here
        }
        self.pending = true;
        if is_cut { Action::Hold } else { Action::Interp }
    }
}

/// New frames in the last second at which even timing switches on.
pub const UNEVEN_ON: (i64, i64) = (39, 56);
/// New frames in the last second outside which even timing switches off again.
pub const UNEVEN_OFF: (i64, i64) = (35, 58);
/// Ticks the rate has to stay there first, so a 30 fps game's 60 fps menu doesn't flip it on and off.
pub const UNEVEN_DWELL: u32 = 60;
pub const UNEVEN_MAX_DELAY: f64 = 2.0;

/// Ticks a frame stays up to count as a dip. Longer is a pause, a load or a still scene.
pub const DIP_HOLD: (i64, i64) = (3, 5);
/// Dips in the last second that switch Dip smoothing on, while the game's rate is in DIP_FPS.
pub const DIP_ON_HOLDS: usize = 2;
/// The game rates at which Dip smoothing can switch on.
pub const DIP_FPS: (i64, i64) = (15, 34);
/// Ticks without a dip, or with the rate outside DIP_FPS, that switch it off again.
pub const DIP_OFF_TICKS: i64 = 180;
pub const DIP_MAX_DELAY: f64 = 3.0;

/// A new frame on the timeline (framegen/cadence.py TimedFrame).
#[derive(Clone, Debug)]
pub struct TimedFrame<T> {
    pub id: i64,
    /// Its time in the game, in ticks.
    pub t: f64,
    /// The tick it arrived on.
    pub arrival: i64,
    /// A scene cut (or a damaged frame) comes just before it.
    pub cut: bool,
    pub frame: Option<T>,
    pub t_arrival: f64,
    /// It has been on screen (drawn as itself, or its time has passed).
    pub reached: bool,
}

/// What to show at one display time, either frame `a` itself or, when `b` is Some, the frame at fraction `t` from
/// `a` to `b`.
#[derive(Clone, Debug)]
pub struct Place<T> {
    pub a: TimedFrame<T>,
    pub b: Option<TimedFrame<T>>,
    pub t: f64,
    /// The game time shown.
    pub c: f64,
    /// Frames on screen for the first time with this one.
    pub reached: Vec<TimedFrame<T>>,
    /// The game time shown is past the newest frame, so the next one is late.
    pub waiting: bool,
}

/// Whether a game that runs at up to 30 fps is dipping below it right now (Dip smoothing).
#[derive(Clone, Debug)]
pub struct DipWatch {
    pub n: i64,
    pub dipping: bool,
    last_new: Option<i64>,
    holds: VecDeque<i64>,
    recent: VecDeque<u8>,
    out: i64,
}

impl Default for DipWatch {
    fn default() -> Self {
        DipWatch { n: -1, dipping: false, last_new: None, holds: VecDeque::new(), recent: VecDeque::new(), out: 0 }
    }
}

impl DipWatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, is_new: bool) -> bool {
        self.n += 1;
        if self.recent.len() == 60 {
            self.recent.pop_front();
        }
        self.recent.push_back(is_new as u8);
        if is_new {
            if let Some(l) = self.last_new
                && DIP_HOLD.0 <= self.n - l && self.n - l <= DIP_HOLD.1 {
                    self.holds.push_back(self.n);
                }
            self.last_new = Some(self.n);
        }
        while self.holds.front().is_some_and(|&h| h <= self.n - DIP_OFF_TICKS) {
            self.holds.pop_front();
        }
        let fps = (self.recent.len() == 60).then(|| self.recent.iter().map(|&x| x as i64).sum::<i64>());
        let in_range = fps.is_some_and(|f| DIP_FPS.0 <= f && f <= DIP_FPS.1);
        self.out = if in_range { 0 } else { self.out + 1 };
        if !self.dipping {
            self.dipping = in_range && self.holds.iter().filter(|&&h| h > self.n - 60).count() >= DIP_ON_HOLDS;
        } else if self.holds.is_empty() || self.out >= DIP_OFF_TICKS {
            self.dipping = false;
        }
        self.dipping
    }
}

/// Even timing for games at about 40 to 60 fps, and frames timed as they arrived for Dip smoothing
/// (framegen/cadence.py FrameTimeline). `T` is whatever the caller keeps with each frame. The pipeline keeps the Frame.
#[derive(Clone, Debug)]
pub struct FrameTimeline<T> {
    /// Ticks seen.
    pub n: i64,
    /// The frame at or before the last game time shown, and later ones.
    pub frames: VecDeque<TimedFrame<T>>,
    /// Ticks between the display's time and the game time it shows.
    pub delay: f64,
    pub uneven: bool,
    ids: i64,
    k: usize,
    arrivals: VecDeque<i64>,
    need: VecDeque<f64>,
    last: Option<(f64, i64)>, // the last frame's (t, arrival)
    recent: VecDeque<u8>,
    dwell: u32,
    hz: u32,
    j: i64,
    /// Frames are timed as they arrived, for Dip smoothing.
    pub at_arrival: bool,
}

impl<T: Clone> Default for FrameTimeline<T> {
    fn default() -> Self {
        Self::new(12)
    }
}

fn push_max<X>(q: &mut VecDeque<X>, max: usize, x: X) {
    if q.len() == max {
        q.pop_front();
    }
    q.push_back(x);
}

impl<T: Clone> FrameTimeline<T> {
    pub fn new(k: usize) -> Self {
        FrameTimeline {
            n: -1,
            frames: VecDeque::new(),
            delay: 1.0,
            uneven: false,
            ids: 0,
            k,
            arrivals: VecDeque::new(),
            need: VecDeque::new(),
            last: None,
            recent: VecDeque::new(),
            dwell: 0,
            hz: 0,
            j: 0,
            at_arrival: false,
        }
    }

    pub fn set_at_arrival(&mut self, on: bool) {
        if on == self.at_arrival {
            return;
        }
        self.at_arrival = on;
        if on {
            // the delay the last few frames would have needed, timed as they arrived
            let a: Vec<i64> = self.arrivals.iter().copied().collect();
            self.need.clear();
            for w in a.windows(2) {
                if w[1] - w[0] <= DIP_HOLD.1 {
                    push_max(&mut self.need, self.k, (w[1] - w[0] - 1) as f64);
                }
            }
            self.delay = self.delay.max(self.want_delay());
        }
    }

    /// The recent average spacing of new frames, in ticks (a gap over 3 ticks counts as 3).
    pub fn period(&self) -> f64 {
        let a: Vec<i64> = self.arrivals.iter().copied().collect();
        let d: Vec<i64> = a.windows(2).map(|w| 3.min(w[1] - w[0])).collect();
        if d.is_empty() { 1.0 } else { d.iter().sum::<i64>() as f64 / d.len() as f64 }
    }

    /// New frames in the last second (the last 60 ticks), or None until there's a second of them.
    pub fn fps(&self) -> Option<i64> {
        (self.recent.len() == 60).then(|| self.recent.iter().map(|&x| x as i64).sum())
    }

    fn want_delay(&self) -> f64 {
        let cap = if self.at_arrival { DIP_MAX_DELAY } else { UNEVEN_MAX_DELAY };
        let need = self.need.iter().copied().fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| if x > m { x } else { m })));
        cap.min(need.unwrap_or(1.0) + 0.05)
    }

    /// One tick of the stream. A new frame goes on the timeline and a copy is returned.
    pub fn tick(&mut self, is_new: bool, is_cut: bool, frame: Option<T>, t_arrival: f64) -> Option<TimedFrame<T>> {
        self.n += 1;
        push_max(&mut self.recent, 60, is_new as u8);
        let mut added = None;
        if is_new {
            push_max(&mut self.arrivals, self.k + 1, self.n);
            let n = self.n as f64;
            let t = match self.last {
                None => n,
                Some((lt, la)) if self.at_arrival && self.n - la <= DIP_HOLD.1 => {
                    push_max(&mut self.need, self.k, (self.n - 1) as f64 - lt);
                    n.max(lt + 0.05) // as it arrived, a held frame's wait included
                }
                Some((lt, la)) if self.n - la > 3 => (n - 0.95).max(lt + 0.05), // after a hitch, as it comes
                Some((lt, _)) => {
                    // It finished in (n - 1, n], the tick before it arrived. The 0.05 keeps two frames' times apart.
                    let t = n.min((n - 0.95).max(lt + self.period()));
                    push_max(&mut self.need, self.k, (self.n - 1) as f64 - lt);
                    t
                }
            };
            let tf = TimedFrame { id: self.ids, t, arrival: self.n, cut: is_cut, frame, t_arrival, reached: false };
            self.ids += 1;
            self.frames.push_back(tf.clone());
            self.last = Some((t, self.n));
            while self.frames.len() > 4 {
                self.frames.pop_front();
            }
            let want = self.want_delay();
            self.delay += (if want > self.delay { 0.3 } else { 0.05 }) * (want - self.delay);
            added = Some(tf);
        }
        let fps = self.fps();
        let mut want_on = self.uneven;
        if let Some(fps) = fps {
            if UNEVEN_ON.0 <= fps && fps <= UNEVEN_ON.1 {
                want_on = true;
            } else if fps <= UNEVEN_OFF.0 || fps >= UNEVEN_OFF.1 {
                want_on = false;
            }
        }
        self.dwell = if want_on != self.uneven { self.dwell + 1 } else { 0 };
        if self.dwell >= UNEVEN_DWELL {
            self.uneven = want_on;
            self.dwell = 0;
            if want_on {
                self.delay = self.want_delay();
            }
        }
        added
    }

    /// The display times (in ticks) of the output frames due this tick: one at 60 fps, one or two at 90, two at 120.
    pub fn slots(&mut self, hz: u32) -> Vec<f64> {
        let step = 60.0 / hz as f64;
        if hz != self.hz || self.j as f64 * step <= (self.n - 1) as f64 {
            self.hz = hz;
            self.j = ((self.n - 1) as f64 / step + 1e-9).floor() as i64 + 1;
        }
        let mut out = Vec::new();
        while self.j as f64 * step <= self.n as f64 + 1e-9 {
            out.push(self.j as f64 * step);
            self.j += 1;
        }
        out
    }

    /// What to show at display time phi, in ticks. None before the first frame.
    pub fn place(&mut self, phi: f64) -> Option<Place<T>> {
        let c = phi - self.delay;
        if self.frames.is_empty() {
            return None;
        }
        while self.frames.len() >= 2 && self.frames[1].t <= c {
            self.frames.pop_front();
        }
        let mut reached = Vec::new();
        for f in self.frames.iter_mut() {
            if !f.reached && f.t <= c {
                f.reached = true;
                reached.push(f.clone());
            }
        }
        let a = self.frames[0].clone();
        if self.frames.len() == 1 || c <= a.t {
            let waiting = self.frames.len() == 1 && c > a.t + 1e-6;
            return Some(Place { a, b: None, t: 0.0, c, reached, waiting });
        }
        let b = self.frames[1].clone();
        let t = (c - a.t) / (b.t - a.t);
        if b.cut || t < 0.04 {
            // across a cut, the frame before it until the new one's time comes
            return Some(Place { a, b: None, t: 0.0, c, reached, waiting: false });
        }
        if t > 0.96 {
            // b itself is on screen from now, so its delay counts from here
            if !self.frames[1].reached {
                self.frames[1].reached = true;
                reached.push(self.frames[1].clone());
            }
            return Some(Place { a: self.frames[1].clone(), b: None, t: 0.0, c, reached, waiting: false });
        }
        Some(Place { a, b: Some(b), t, c, reached, waiting: false })
    }
}
