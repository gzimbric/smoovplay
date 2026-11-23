//! The delay test (L in the video window, PS5 Remote Play; framegen/delaytest.py): chiaki-ng presses Options on the
//! PS5 a few times, and each time the pipeline times how long the answer takes to reach this Mac, from the moment
//! chiaki-ng sent the press to the first frame that shows it. The picture is compared in 32x18 blocks of brightness:
//! a press is answered when at least a quarter of the blocks change by much more than the picture moved on its own
//! in the second before the test.

use crate::np;

pub const BUTTON_OPTIONS: u32 = 1 << 12; // ChiakiControllerButton
/// An even number, so a menu that opens and closes ends closed.
pub const PRESSES: u32 = 6;
pub const HOLD_MS: u32 = 100;
/// Watch the picture this long first, to learn how much it moves on its own...
pub const SETTLE_S: f64 = 1.0;
/// ...comparing frames this many ticks apart.
pub const SPAN: usize = 10;
/// The longest wait for an answer.
pub const WAIT_S: f64 = 1.0;
/// The longest wait for chiaki-ng to send the press.
pub const SEND_S: f64 = 0.5;
/// From an answer (or giving up) to the next press.
pub const GAP_S: f64 = 1.2;
/// Brightness levels: the smallest change that counts as an answer, however still the picture.
pub const MIN_CHANGE: f64 = 6.0;

/// Blocks down and across.
pub const BLOCKS: (usize, usize) = (18, 32);
/// Pixels looked at across each block (and down it): every 4th at 1080p.
pub const SAMPLES: usize = 15;

/// How far apart small() reads pixels: 4 at 1080p, 8 at 4K, every pixel in a frame too small to spare any.
pub fn step(h: usize, w: usize) -> usize {
    1.max((h / (BLOCKS.0 * SAMPLES)).min(w / (BLOCKS.1 * SAMPLES)))
}

/// small() on packed 8-bit BGR, on the CPU: the reference the GPU probe is checked against. 18x32 blocks of
/// brightness, row-major, every step()th pixel read, averaged like torch's adaptive_avg_pool2d.
pub fn small_bgr(bgr: &[u8], w: usize, h: usize) -> Vec<f32> {
    let s = step(h, w);
    let (hs, ws) = (h.div_ceil(s), w.div_ceil(s));
    let (by, bx) = BLOCKS;
    let mut out = vec![0f32; by * bx];
    for i in 0..by {
        let (y0, y1) = (i * hs / by, ((i + 1) * hs).div_ceil(by));
        for j in 0..bx {
            let (x0, x1) = (j * ws / bx, ((j + 1) * ws).div_ceil(bx));
            let mut sum = 0f32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let p = ((yy * s) * w + xx * s) * 3;
                    sum += bgr[p] as f32 * 0.114 + bgr[p + 1] as f32 * 0.587 + bgr[p + 2] as f32 * 0.299;
                }
            }
            out[i * bx + j] = sum / ((y1 - y0) * (x1 - x0)) as f32;
        }
    }
    out
}

/// How much of the picture changed: the 75th percentile of the blocks' change, so at least a quarter of it.
pub fn change(a: &[f32], b: &[f32]) -> f64 {
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    np::percentile_f32(&d, 75.0) as f64
}

/// chiaki-ng's side of a press: press(buttons, hold_ms) -> token, press_sent(token) -> when it went out, or None.
pub trait Press {
    fn press(&mut self, buttons: u32, hold_ms: u32) -> u64;
    fn press_sent(&mut self, token: u64) -> Option<f64>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Settle,
    Gap,
    Sending,
    Waiting,
    Done,
}

/// Feed it every frame with feed(); it presses, watches and times by itself until done().
#[derive(Clone, Debug)]
pub struct DelayTest {
    pub presses: u32,
    pub button: u32,
    pub state: State,
    pub t_state: f64,
    pub recent: Vec<Vec<f32>>,
    pub motion: Vec<f64>,
    pub threshold: f64,
    /// Each answered press: sent -> the first frame showing it arrived.
    pub ms: Vec<f64>,
    pub pressed: u32,
    pub missed: u32,
    pub error: String,
    /// SETTLE_S and GAP_S, for tests that hurry it.
    pub settle_s: f64,
    pub gap_s: f64,
    reference: Option<Vec<f32>>,
    token: u64,
    t_sent: f64,
}

impl DelayTest {
    pub fn new(presses: u32, button: u32, now: f64) -> DelayTest {
        DelayTest {
            presses,
            button,
            state: State::Settle,
            t_state: now,
            recent: Vec::new(),
            motion: Vec::new(),
            threshold: MIN_CHANGE,
            ms: Vec::new(),
            pressed: 0,
            missed: 0,
            error: String::new(),
            settle_s: SETTLE_S,
            gap_s: GAP_S,
            reference: None,
            token: 0,
            t_sent: 0.0,
        }
    }

    pub fn done(&self) -> bool {
        self.state == State::Done
    }

    fn to(&mut self, state: State, now: f64) {
        self.state = state;
        self.t_state = now;
    }

    /// One frame (small()), when it reached this Mac, and the time now.
    pub fn feed(&mut self, remote: &mut dyn Press, blocks: &[f32], t_arrival: f64, now: f64) {
        self.recent.push(blocks.to_vec());
        if self.recent.len() > SPAN + 1 {
            self.recent.remove(0);
        }
        if self.state == State::Settle {
            if self.recent.len() > SPAN {
                self.motion.push(change(&self.recent[0], blocks));
            }
            if now - self.t_state >= self.settle_s && !self.motion.is_empty() {
                self.threshold = MIN_CHANGE.max(2.5 * np::percentile(&self.motion, 95.0));
                self.to(State::Gap, now - self.gap_s);
            }
        }
        if self.state == State::Gap && now - self.t_state >= self.gap_s {
            if self.pressed >= self.presses {
                self.to(State::Done, now);
                return;
            }
            self.reference = Some(blocks.to_vec());
            self.pressed += 1;
            self.token = remote.press(self.button, HOLD_MS);
            self.to(State::Sending, now);
        }
        if self.state == State::Sending {
            match remote.press_sent(self.token) {
                Some(sent) => {
                    self.t_sent = sent;
                    self.to(State::Waiting, now);
                }
                None if now - self.t_state > SEND_S => {
                    self.error = "chiaki-ng didn't send the press".into();
                    self.to(State::Done, now);
                    return;
                }
                None => {}
            }
        }
        if self.state == State::Waiting {
            if t_arrival > self.t_sent && change(self.reference.as_deref().unwrap_or(blocks), blocks) > self.threshold {
                self.ms.push((t_arrival - self.t_sent) * 1000.0);
                self.to(State::Gap, now);
            } else if now - self.t_sent > WAIT_S {
                self.missed += 1;
                if self.missed >= 2 && self.ms.is_empty() {
                    self.error = "nothing changed on screen after pressing Options".into();
                    self.to(State::Done, now);
                    return;
                }
                self.to(State::Gap, now);
            }
        }
    }

    /// The overlay's line while it runs.
    pub fn status(&self) -> String {
        if self.state == State::Settle {
            return "DELAY TEST: stand still; it presses Options a few times (a menu opening is what it times)".into();
        }
        let mut s = format!("DELAY TEST: press {} of {}", self.pressed, self.presses);
        if !self.ms.is_empty() {
            s += &format!("  PS5 answered in {} ms", np::f(np::median(&self.ms), 0));
        }
        s
    }

    /// The answer, in a line: the PS5 and Remote Play's part, SmoovPlay's, and the two together.
    pub fn result(&self, smoovplay_ms: Option<f64>) -> String {
        if !self.error.is_empty() && self.ms.is_empty() {
            return if self.error.contains("nothing changed") {
                format!("Delay test: {}. Try where Options opens a menu, standing still", self.error)
            } else {
                format!("Delay test: {}. Close chiaki-ng and open it from SmoovPlay", self.error)
            };
        }
        let remote = np::median(&self.ms);
        let mut text = format!(
            "Delay test: PS5 and Remote Play {} ms (button sent to the answer arriving, median of {})",
            np::f(remote, 0),
            self.ms.len()
        );
        if let Some(ours) = smoovplay_ms {
            text += &format!(
                ", SmoovPlay {} ms: {} ms from button to SmoovPlay's screen, plus your controller's link and your monitor",
                np::f(ours, 0),
                np::f(remote + ours, 0)
            );
        }
        text
    }
}
