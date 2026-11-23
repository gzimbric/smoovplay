//! The sound's timing (replay.py Recorder._encode_sound). The sound plays on, sample after sample, while arrival
//! times jitter. Only a gap (silence put in) or getting ahead (sound skipped) that's large, or that's lasted a
//! second, moves it, so jitter never makes a click. A pause of more than 30 s starts its timeline again.

use std::collections::VecDeque;

use crate::{OFFSET, SOUND_RATE, TB, py_round};

/// What to write for a chunk, in order. Each part is `frames` of silence, or of the chunk from frame `from`, at `pts`
/// in 48 kHz samples on the replay's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Silence { pts: i64, frames: usize },
    Chunk { pts: i64, from: usize, frames: usize },
}

#[derive(Debug, Default)]
pub struct SoundClock {
    /// When (on the pipeline's clock) the next sample written is heard.
    pub a_next: Option<f64>,
    /// Its timestamp, in 48 kHz samples from the replay's zero (replay.py _a_pts).
    pub a_pts: i64,
    skip_s: f64,
    drifts: VecDeque<(f64, f64)>,
    rem: u64,
}

impl SoundClock {
    pub fn new() -> SoundClock {
        SoundClock::default()
    }

    /// Starts over (a new sound format, or off).
    pub fn reset(&mut self) {
        *self = SoundClock::default();
    }

    fn advance(&mut self, frames: usize, rate: u32) {
        let total = self.rem + frames as u64 * SOUND_RATE as u64;
        self.a_pts += (total / rate as u64) as i64;
        self.rem = total % rate as u64;
    }

    /// Places a chunk of `n` frames at `rate` that reached the Mac at `t`, with SmoovPlay's delay `delay` in seconds.
    /// `t0` is the replay's clock zero, set from this chunk if it isn't yet.
    pub fn place(&mut self, t: f64, n: usize, rate: u32, delay: f64, t0: &mut Option<f64>) -> Vec<Part> {
        let mut out = Vec::new();
        let t_due = t + delay; // when the picture it goes with is shown
        if self.a_next.is_none() {
            self.a_next = Some(t_due);
            self.skip_s = 0.0;
            let zero = *t0.get_or_insert(t - 5.0);
            self.a_pts = (OFFSET * SOUND_RATE as i64 / TB + py_round((t_due - zero) * SOUND_RATE as f64)).max(0);
            self.rem = 0;
            self.drifts.clear();
        }
        let drift = t_due - self.a_next.unwrap();
        self.drifts.push_back((t, drift));
        while self.drifts.front().is_some_and(|d| t - d.0 > 1.0) {
            self.drifts.pop_front();
        }
        let steady = self.drifts.len() > 1 && t - self.drifts[0].0 >= 0.9;
        let min = self.drifts.iter().map(|d| d.1).fold(f64::INFINITY, f64::min);
        let max = self.drifts.iter().map(|d| d.1).fold(f64::NEG_INFINITY, f64::max);
        if drift > 0.25 || (steady && min > 0.06) {
            if drift > 30.0 {
                // After a long pause, start the sound's timeline again. The clip will have a gap.
                self.a_next = Some(t_due);
                self.a_pts += py_round(drift * SOUND_RATE as f64);
            } else {
                let frames = py_round(drift * rate as f64).max(0) as usize;
                if frames > 0 {
                    out.push(Part::Silence { pts: self.a_pts, frames });
                    self.advance(frames, rate);
                }
                *self.a_next.as_mut().unwrap() += drift;
            }
            self.drifts.clear();
            self.skip_s = 0.0;
        } else if drift < -0.25 || (steady && max < -0.06) {
            self.skip_s = -drift;
            self.drifts.clear();
        }
        let mut from = 0;
        let mut n = n;
        if self.skip_s > 0.0 {
            let drop = n.min(py_round(self.skip_s * rate as f64).max(0) as usize);
            self.skip_s = (self.skip_s - drop as f64 / rate as f64).max(0.0);
            from = drop;
            n -= drop;
        }
        if n > 0 {
            out.push(Part::Chunk { pts: self.a_pts, from, frames: n });
            self.advance(n, rate);
            *self.a_next.as_mut().unwrap() += n as f64 / rate as f64;
        }
        out
    }
}
