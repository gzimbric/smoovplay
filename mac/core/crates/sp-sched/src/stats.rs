//! What a session measured, and the two reports made from it (framegen/pipeline.py Stats, GameRate, ShownRate,
//! rate_advice). summary() is for the log and results() for test mode's plain-language report. The text is the
//! Python's, character for character, and tests/parity.rs checks it against the reference.

use std::collections::{BTreeSet, VecDeque};

use crate::np;

/// One 60 Hz tick, in seconds.
pub const TICK: f64 = 1.0 / 60.0;

/// --score's still areas are pixels within this many levels (of 255) in both real frames, like a HUD or map.
pub const STILL_LEVELS: f32 = 3.0;
/// --score's HUD-like areas are still in both real frames and the real in-between one, and on an edge, where the
/// Sobel L1 of the real frame's brightness is above this.
pub const HUD_EDGE: f32 = 128.0;
/// A HUD-like pixel is visibly off when the AI frame misses the real one by more than this many levels.
pub const HUD_OFF_LEVELS: f32 = 12.0;
/// Test mode's quality is the share of a frame's pixels that miss the real frame by more than this.
pub const VISIBLE_LEVELS: f32 = HUD_OFF_LEVELS;

/// How long the game has to stay off the rate that's needed before the overlay says so, and before the message.
pub const RATE_NOTE_S: f64 = 3.0;
pub const RATE_TOAST_S: f64 = 10.0;

/// Said when "Simulate a 30 fps game" goes off by itself.
pub fn simulate_off(g: i64) -> String {
    format!("Simulate turned off: this game is at about {g} fps already, and Simulate halved that")
}

/// 0.062 gives "6.2%" and 0.12 gives "12%".
pub fn pct(share: f64) -> String {
    let v = 100.0 * share;
    if v < 10.0 { format!("{}%", np::f(v, 1)) } else { format!("{}%", np::f(v, 0)) }
}

/// One line for someone whose game isn't at the rate that's needed, or "" when it is or the rate isn't known yet.
pub fn rate_advice(game_fps: Option<f64>, test: bool, even: bool) -> String {
    let Some(g) = game_fps else { return String::new() };
    let g = np::round(g);
    if test {
        if g >= 50 {
            return String::new();
        }
        if g <= 38 {
            return format!("Simulate is for 60 fps games. This one is at {g} fps already: turn Simulate off (menu: M)");
        }
        return format!("Simulate needs a steady 60 fps game. This one is at about {g}, so quality isn't measured");
    }
    if g <= 38 || (even && g <= 56) {
        return String::new();
    }
    if g >= 55 {
        return format!("Game at {g} fps: no gaps to fill. Frame generation is for 30 fps modes (Quality/Fidelity)");
    }
    format!("Game at about {g} fps, not 30: few gaps to fill. Its 30 fps mode (Quality/Fidelity) is smoother")
}

/// The game's own frame rate, from new frames among the last `n` ticks of a 60 Hz stream.
#[derive(Clone, Debug)]
pub struct GameRate {
    new: VecDeque<bool>,
    n: usize,
}

impl Default for GameRate {
    fn default() -> Self {
        Self::new(120)
    }
}

impl GameRate {
    pub fn new(n: usize) -> Self {
        GameRate { new: VecDeque::with_capacity(n), n }
    }
    /// One tick, Some(true) for new content and Some(false) for a repeat. None, when the source can't say, is left
    /// out.
    pub fn add(&mut self, new: Option<bool>) {
        if let Some(v) = new {
            if self.new.len() == self.n {
                self.new.pop_front();
            }
            self.new.push_back(v);
        }
    }
    pub fn clear(&mut self) {
        self.new.clear();
    }
    pub fn fps(&self) -> Option<f64> {
        let n = self.new.len();
        (n > 0 && n == self.n).then(|| self.new.iter().filter(|&&v| v).count() as f64 / n as f64 / TICK)
    }
}

/// Frames put on screen per second over the last `window` seconds.
#[derive(Clone, Debug)]
pub struct ShownRate {
    pub window: f64,
    t: VecDeque<f64>,
    start: Option<f64>,
}

impl Default for ShownRate {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ShownRate {
    pub fn new(window: f64) -> Self {
        ShownRate { window, t: VecDeque::new(), start: None }
    }
    pub fn add(&mut self, t: f64) {
        self.start.get_or_insert(t);
        self.t.push_back(t);
        while self.t[0] <= t - self.window {
            self.t.pop_front();
        }
    }
    pub fn fps(&self) -> Option<f64> {
        let start = self.start?;
        if self.t.back().unwrap() - start < self.window {
            return None;
        }
        Some(self.t.len() as f64 / self.window)
    }
}

/// A Counter, in the order keys first appeared (as Python's dict shows it).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Counts(pub Vec<(String, u64)>);

impl Counts {
    pub fn add(&mut self, key: &str) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some((_, n)) => *n += 1,
            None => self.0.push((key.to_string(), 1)),
        }
    }
    pub fn get(&self, key: &str) -> u64 {
        self.0.iter().find(|(k, _)| k == key).map_or(0, |(_, n)| *n)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Python's repr of the dict, as in {'current': 3, 'interp': 2}.
    pub fn repr(&self) -> String {
        let items: Vec<String> = self.0.iter().map(|(k, n)| format!("'{k}': {n}")).collect();
        format!("{{{}}}", items.join(", "))
    }
}

/// Why run() returned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Exit {
    /// The source finished.
    #[default]
    End,
    Quit,
    Restart,
}

impl Exit {
    pub fn as_str(self) -> &'static str {
        match self {
            Exit::End => "end",
            Exit::Quit => "quit",
            Exit::Restart => "restart",
        }
    }
}

/// The worst-scoring frame (--save-worst): its PSNR, tick, and the four pictures as packed BGR (SDR frames only).
#[derive(Clone, Debug)]
pub struct Worst {
    pub psnr: f64,
    pub tick: u64,
    pub width: usize,
    pub height: usize,
    /// prev, interp, gt, next (empty for HDR frames).
    pub images: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Tick processing when we interpolate.
    pub proc_interp_ms: Vec<f64>,
    /// Tick processing otherwise.
    pub proc_other_ms: Vec<f64>,
    /// Time spent in display.
    pub sink_ms: Vec<f64>,
    /// From arrival until processing starts (backlog).
    pub queue_ms: Vec<f64>,
    /// Scoring, done after the frame is shown.
    pub score_ms: Vec<f64>,
    /// Processing and display per tick.
    pub busy_ms: Vec<f64>,
    /// From the arrival of each real (new) frame until it is first presented.
    pub real_delay_ms: Vec<f64>,
    pub actions: Counts,
    /// True and false positives and negatives (tp, tn, fp, fn) against ground truth.
    pub dedup: Counts,
    pub psnr_interp: Vec<f64>,
    pub psnr_repeat: Vec<f64>,
    pub psnr_blend: Vec<f64>,
    pub off_interp: Vec<f64>,
    pub off_repeat: Vec<f64>,
    pub psnr_still: Vec<f64>,
    pub psnr_still_repeat: Vec<f64>,
    pub still_frac: Vec<f64>,
    pub psnr_hud: Vec<f64>,
    pub hud_off: Vec<f64>,
    pub hud_frac: Vec<f64>,
    pub worst: Option<Worst>,
    /// At 90 or 120 Hz, the extra in-between frames shown, and ones skipped because the pipeline was behind.
    pub quarter_shown: u64,
    pub quarter_skipped: u64,
    /// Ticks whose frame waited for the pipeline to finish earlier ones (more than 60% of a tick).
    pub behind: u64,
    pub hz_used: BTreeSet<u32>,
    /// Ticks played with "Simulate a 30 fps game" on.
    pub simulated: u64,
    pub test_game_new: u64,
    pub test_game_repeat: u64,
    /// The delay test's results, a line each.
    pub delay_tests: Vec<String>,
    pub uneven_ticks: u64,
    pub uneven_fps: Vec<i64>,
    pub uneven_waits: u64,
    pub dip_ticks: u64,
    /// Time between the game's new frames as they reached SmoovPlay.
    pub new_gaps_ms: Vec<f64>,
    pub wall_s: f64,
    pub exit: Exit,
    /// Frames skipped because they arrived before run() started.
    pub stale: u64,
    /// The engine's numbers mustn't be shared (NVIDIA VFG), so summary() leaves out timings and scores.
    pub private: bool,
}

pub const PRIVATE_NOTE: &str = "NVIDIA VFG: timings and quality scores are left out of the log, because NVIDIA's licence \
doesn't allow sharing them and the log goes into Copy diagnostics. The stats overlay (S) and test mode's report show them \
on this PC.";

fn finite(xs: &[f64]) -> Vec<f64> {
    xs.iter().copied().filter(|x| x.is_finite()).collect()
}

fn fmax(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, |m, x| if x > m { x } else { m })
}

fn fmin(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::INFINITY, |m, x| if x < m { x } else { m })
}

impl Stats {
    /// How evenly the game's new frames reached SmoovPlay, or "" for a short session.
    pub fn arrival_line(&self) -> String {
        let gaps = &self.new_gaps_ms;
        if gaps.len() < 20 {
            return String::new();
        }
        let usual = np::median(gaps);
        let late = gaps.iter().filter(|&&g| g > 1.5 * usual).count() as f64 / gaps.len() as f64;
        let bunched = gaps.iter().filter(|&&g| g < 0.5 * usual).count() as f64 / gaps.len() as f64;
        format!(
            "game frames as they arrived: {} new, usually {} ms apart; {}% late (over 1.5x that), {}% bunched \
             (under half); longest gap {} ms",
            gaps.len() + 1,
            np::f(usual, 1),
            np::f(100.0 * late, 1),
            np::f(100.0 * bunched, 1),
            np::f(fmax(gaps), 0)
        )
    }

    pub fn summary(&self) -> String {
        let n = self.proc_interp_ms.len() + self.proc_other_ms.len();
        let mut lines = vec![format!("ticks={n}  wall={}s", np::f(self.wall_s, 1)), format!("actions: {}", self.actions.repr())];
        let arrival = self.arrival_line();
        if !arrival.is_empty() {
            lines.push(arrival);
        }
        let tg = self.test_game_new + self.test_game_repeat;
        if tg > 0 {
            lines.push(format!(
                "test mode: {} of {tg} frames were repeats in the game itself (a steady 60 fps game has next to none); \
                 only frames with new ones on both sides are scored",
                self.test_game_repeat
            ));
        }
        if self.uneven_ticks > 0 {
            let fps: Vec<f64> = self.uneven_fps.iter().map(|&x| x as f64).collect();
            lines.push(format!(
                "even timing (a game at about {} fps): {} ticks, {} output frames waited for a late frame",
                np::f(np::median(&fps), 0),
                self.uneven_ticks,
                self.uneven_waits
            ));
        }
        if self.dip_ticks > 0 {
            lines.push(format!("dip timing (a 30 fps game dipping below it): {} ticks", self.dip_ticks));
        }
        if self.private {
            if !self.dedup.is_empty() {
                lines.push(format!("dedup vs truth: {}", self.dedup.repr()));
            }
            lines.push(PRIVATE_NOTE.to_string());
            return lines.join("\n");
        }
        lines.extend(self.delay_tests.iter().cloned());
        lines[0] += &format!("  ({} ticks/s)", np::f(n as f64 / self.wall_s.max(1e-9), 1));
        let rows: [(&str, &Vec<f64>); 6] = [
            ("processing, interp ticks", &self.proc_interp_ms),
            ("processing, other ticks", &self.proc_other_ms),
            ("display/encode", &self.sink_ms),
            ("scoring, after display", &self.score_ms),
            ("queueing (backlog)", &self.queue_ms),
            ("real frame arrival->presented", &self.real_delay_ms),
        ];
        for (name, xs) in rows {
            if !xs.is_empty() {
                lines.push(format!(
                    "{:<30} ms  p50={}  p95={}  p99={}  max={}",
                    name,
                    np::fw(np::percentile(xs, 50.0), 6, 2),
                    np::fw(np::percentile(xs, 95.0), 6, 2),
                    np::fw(np::percentile(xs, 99.0), 6, 2),
                    np::fw(fmax(xs), 6, 2)
                ));
            }
        }
        if self.stale > 0 {
            lines.push(format!("stale frames skipped at start (queued before this run): {}", self.stale));
        }
        lines.push(format!(
            "ticks over the {} ms budget (processing + display): {}",
            np::f(TICK * 1000.0, 1),
            self.busy_ms.iter().filter(|&&b| b > TICK * 1000.0).count()
        ));
        if self.quarter_shown > 0 || self.quarter_skipped > 0 {
            let due = self.quarter_shown + self.quarter_skipped;
            lines.push(format!(
                "extra in-between frames (90/120 Hz) shown {} of {due} ({}%); the rest fell back to 60 Hz because it was behind",
                self.quarter_shown,
                np::f(100.0 * self.quarter_shown as f64 / due as f64, 0)
            ));
        }
        if !self.dedup.is_empty() {
            lines.push(format!("dedup vs truth: {}", self.dedup.repr()));
        }
        if !self.psnr_interp.is_empty() {
            lines.push(format!(
                "PSNR vs real in-between frame (dB, higher=better): RIFE {} | blend {} | repeat {}  (n={}, RIFE worst {})",
                np::f(np::mean(&finite(&self.psnr_interp)), 2),
                np::f(np::mean(&finite(&self.psnr_blend)), 2),
                np::f(np::mean(&finite(&self.psnr_repeat)), 2),
                self.psnr_interp.len(),
                np::f(fmin(&self.psnr_interp), 2)
            ));
        }
        if !self.off_interp.is_empty() {
            lines.push(format!(
                "visibly off (more than {} levels): RIFE {}% | repeat {}%  (worst {}%)",
                VISIBLE_LEVELS as i64,
                np::f(100.0 * np::mean(&self.off_interp), 2),
                np::f(100.0 * np::mean(&self.off_repeat), 2),
                np::f(100.0 * fmax(&self.off_interp), 2)
            ));
        }
        if !self.psnr_still.is_empty() {
            lines.push(format!(
                "  still areas ({}% of pixels): RIFE {} | repeat {}",
                np::f(100.0 * np::mean(&self.still_frac), 0),
                np::f(np::mean(&finite(&self.psnr_still)), 2),
                np::f(np::mean(&finite(&self.psnr_still_repeat)), 2)
            ));
        }
        if !self.psnr_hud.is_empty() {
            let fin = finite(&self.psnr_hud);
            lines.push(format!(
                "  HUD-like areas ({}% of pixels): RIFE {} dB, {}% visibly off",
                np::f(100.0 * np::mean(&self.hud_frac), 1),
                np::f(if fin.is_empty() { f64::INFINITY } else { np::mean(&fin) }, 2),
                np::f(100.0 * np::mean(&self.hud_off), 2)
            ));
        }
        lines.join("\n")
    }

    /// Test mode's plain-language report. `full_motion` says whether motion ran at full resolution, or is None when
    /// unknown.
    pub fn results(&self, full_motion: Option<bool>) -> Vec<String> {
        let budget = TICK * 1000.0;
        let (interp, busy, delay) = (&self.proc_interp_ms, &self.busy_ms, &self.real_delay_ms);
        let (ai, repeat) = (&self.off_interp, &self.off_repeat);
        let rates: Vec<u32> = if self.hz_used.is_empty() { vec![60] } else { self.hz_used.iter().copied().collect() };
        let mut lines =
            vec![format!("Output: {}", rates.iter().map(|h| format!("{h} fps")).collect::<Vec<_>>().join(" and "))];
        if !interp.is_empty() {
            lines.push(format!(
                "AI frame time: {} ms on average, {} ms for the slowest 5% (a 60 fps frame lasts {} ms)",
                np::f(np::mean(interp), 1),
                np::f(np::percentile(interp, 95.0), 1),
                np::f(budget, 1)
            ));
        }
        let due = self.quarter_shown + self.quarter_skipped;
        let (n, behind) = (busy.len(), self.behind);
        let made: Vec<f64> = interp.iter().chain(&self.proc_other_ms).copied().collect();
        let slow = made.iter().filter(|&&ms| ms > budget).count();
        if due > 0 && n > 0 {
            lines.push(format!(
                "Running behind: {}% of frames ({behind} of {n}) arrived while the PC was still busy with earlier ones",
                np::f(100.0 * behind as f64 / n as f64, 1)
            ));
        } else if !made.is_empty() {
            lines.push(format!(
                "Slow frames: {}% ({slow} of {}) took longer to make than a 60 fps frame lasts",
                np::f(100.0 * slow as f64 / made.len() as f64, 1),
                made.len()
            ));
        }
        if due > 0 {
            lines.push(format!(
                "Extra frames for 90/120 fps: {}% shown; the rest fell back to 60 fps because the PC was behind",
                np::f(100.0 * self.quarter_shown as f64 / due as f64, 0)
            ));
        }
        if !delay.is_empty() {
            lines.push(format!(
                "Delay: a real frame reached the screen {} ms after it arrived (median)",
                np::f(np::median(delay), 0)
            ));
        }
        let tg = self.test_game_new + self.test_game_repeat;
        let not_60 = tg >= 120 && self.test_game_repeat as f64 / tg as f64 > 0.1;
        if not_60 {
            let fps = 60.0 * self.test_game_new as f64 / tg as f64;
            lines.push(format!(
                "Game: it ran at about {} fps, not a steady 60. Simulate a 30 fps game is for 60 fps games; a 30 fps \
                 game doesn't need it, as frame generation works on it as it is",
                np::f(fps, 0)
            ));
        }
        let only_90 = !self.hz_used.is_empty() && self.hz_used.iter().all(|&h| h == 90);
        if ai.len() >= (if not_60 || only_90 { 300 } else { 1 }) && !repeat.is_empty() {
            lines.push(format!(
                "Quality: {} of each AI frame was visibly off from the real frame it stood in for, against {} when the \
                 last frame is repeated instead (no AI). Lower is closer to the real thing; visibly off means more than \
                 about 5% brighter or darker. Slow scenes score well either way; the gap shows in motion ({} frames \
                 measured, worst {}{}",
                pct(np::mean(ai)),
                pct(np::mean(repeat)),
                ai.len(),
                pct(fmax(ai)),
                if not_60 { "; only where the game drew every frame)" } else { ")" }
            ));
        } else if not_60 {
            lines.push("Quality: not measured, as the game wasn't at a steady 60 fps".into());
        } else if !interp.is_empty() {
            lines.push(
                "Quality: not measured yet (it's measured at 60 and 120 fps output; 90 fps has no real frame to compare \
                 with)"
                    .into(),
            );
        }
        if self.uneven_ticks > 0 {
            lines.push(format!(
                "Game: for {} s it ran between 30 and 60 fps, and its frames were evened out (Even out 40-60 fps)",
                np::f(self.uneven_ticks as f64 / 60.0, 0)
            ));
        }
        if self.dip_ticks > 0 {
            lines.push(format!(
                "Game: for {} s it dipped below 30 fps, and its frames were spaced by when they arrived (Dip smoothing)",
                np::f(self.dip_ticks as f64 / 60.0, 0)
            ));
        }
        lines.extend(self.delay_tests.iter().cloned());
        if interp.is_empty() {
            lines.push("Verdict: frame generation was off, so there's nothing to judge".into());
        } else {
            let late = slow as f64 / made.len().max(1) as f64;
            let shown = if due > 0 { self.quarter_shown as f64 / due as f64 } else { 1.0 };
            let top_rate = *rates.iter().max().unwrap();
            let top = format!("{top_rate} fps");
            let mut lighter: Vec<&str> = Vec::new();
            if top_rate > 60 {
                lighter.push("a lower output rate");
            }
            if full_motion != Some(false) {
                lighter.push("Motion detail: Half");
            }
            let either = lighter.join(" or ");
            let (ok, mostly) = if due > 0 { (shown >= 0.95, shown >= 0.8) } else { (late < 0.01, late < 0.05) };
            if ok {
                lines.push(format!("Verdict: this PC keeps up at {top}"));
            } else if mostly {
                let mut s = format!("Verdict: this PC mostly keeps up at {top}, but falls behind now and then");
                if !lighter.is_empty() {
                    let mut c = either.chars();
                    let first: String = c.next().map(|ch| ch.to_uppercase().collect()).unwrap_or_default();
                    s += &format!(". {first}{} would smooth that out", c.as_str());
                }
                lines.push(s);
            } else {
                lines.push(format!(
                    "Verdict: this PC is falling behind at {top}{}",
                    if lighter.is_empty() { ", even on the lightest settings".to_string() } else { format!(". Try {either}") }
                ));
            }
        }
        lines
    }
}
