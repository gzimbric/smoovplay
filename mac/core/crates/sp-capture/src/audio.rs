//! A capture card's sound, played in step with the picture (framegen/audio.py).
//!
//! [`DelayLine`] is the part that matters, and it needs no sound device. PCM goes in as it arrives and comes out at
//! the output's pace, with the queue kept at the delay the pipeline asks for. Near it the sound plays half a percent
//! faster or slower until it's back, which nobody hears, so the card's clock and the output's can drift apart
//! without the delay creeping. Further off it's one percent. Only far off, over [`JUMP_S`], does it jump, to
//! silence until it has filled or by skipping the oldest sound. Wherever the sound doesn't follow on it's eased over
//! [`FADE_S`], because a step is heard as a click. The queue never goes below the longest recent packet, or the
//! longest recent wait between packets, plus [`CUSHION_S`]. In issue 17, with less queued than the wait for the
//! next packet, the output ran dry every packet or so.
//!
//! The maths is audio.py's step for step, numpy's float32 and float64 included, so the same pushes and pulls give
//! the same bytes (tests/delay_line.rs checks that against the Python).
//!
//! [`PlayerCore`] is the output's side, which [`crate::Player`] plays through Core Audio. It keeps the delay less the
//! output's own latency, and the counts for the log. [`sound_report`] turns them into its line.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::Serialize;

pub const JUMP_S: f64 = 0.120; // further than this from the delay, jump to it
pub const NUDGE_S: f64 = 0.004; // further than this, play half a percent faster or slower until it's back
pub const NUDGE: f64 = 0.005;
pub const FAR_S: f64 = 0.020; // further than this, one percent (100 ms takes 10 s)
pub const FAR_NUDGE: f64 = 0.01;
pub const MAX_S: f64 = 1.5; // never queue more than this
pub const CUSHION_S: f64 = 0.010; // always queued on top of the longest recent packet or wait
pub const GAP_WINDOW_S: f64 = 60.0; // the longest wait between packets this long ago or less sets the cushion
pub const MAX_WAIT_KEEP_S: f64 = 0.150; // but never more than this
pub const STOP_S: f64 = 0.5; // a longer wait is the sound stopping (the card lost the picture, say), not a hold-up
pub const FADE_S: f64 = 0.002; // where the sound doesn't follow on, eased over this
/// Windows' buffer in audio.py (sounddevice's "low" was too little while Python was busy). On the Mac the output's
/// callback is Rust on Core Audio's real-time thread, with nothing to wait for, so [`crate::Player`] keeps the
/// device's own buffer (usually 512 frames, about 11 ms) and takes whatever the output's latency is off the delay.
pub const OUTPUT_S: f64 = 0.040;

/// Python's round() of a float (half to even), as an integer.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// numpy's linspace(1.0, 0.0, k, endpoint=False, dtype=float32)[i], worked out in float64, then made float32.
fn ramp(i: usize, k: usize) -> f32 {
    (i as f64 * (-1.0 / k as f64) + 1.0) as f32
}

/// PCM in as it arrives, out at the output's pace, held back by the delay asked for. It isn't thread-safe by itself,
/// so [`PlayerCore`] holds it behind a lock, with push from the reader's thread and pull from the output's. All sizes
/// are in samples (i16), where audio.py counts bytes. A sample frame is `channels` samples.
pub struct DelayLine {
    pub rate: u32,
    pub channels: usize,
    buf: VecDeque<i16>,
    /// Samples to keep queued, as asked for. See `keep`.
    target: usize,
    /// The last packets' sizes, in samples (a second of a card's).
    packets: VecDeque<usize>,
    /// (time, wait since the packet before), falling, so the first is the longest.
    waits: VecDeque<(f64, f64)>,
    last_in: Option<f64>,
    /// Times it ran dry because nothing had arrived. Each is a gap, eased.
    pub underruns: u64,
    /// Times far too much was queued, so the oldest sound was skipped.
    pub skips: u64,
    /// Times far too little was queued (a long hold-up, or a longer delay), so it went silent until it was back.
    pub refills: u64,
    /// Waits of over STOP_S between packets.
    pub stops: u64,
    /// The longest wait between packets (short of a stop), all session.
    pub longest_wait_s: f64,
    filling: bool,
    seam: bool,
    /// The last sample frame played.
    last: Vec<f32>,
    fade: usize,
    // reused between pulls, so the output's thread doesn't allocate once they've grown
    chunk: Vec<i16>,
    a: Vec<f32>,
    stretched: Vec<f64>,
}

impl DelayLine {
    pub fn new(rate: u32, channels: usize) -> DelayLine {
        DelayLine {
            rate,
            channels,
            buf: VecDeque::with_capacity((MAX_S * rate as f64) as usize * channels + 1),
            target: 0,
            packets: VecDeque::with_capacity(51),
            waits: VecDeque::new(),
            last_in: None,
            underruns: 0,
            skips: 0,
            refills: 0,
            stops: 0,
            longest_wait_s: 0.0,
            filling: true,
            seam: false,
            last: vec![0.0; channels],
            fade: (py_round(FADE_S * rate as f64)).max(1) as usize,
            chunk: Vec::with_capacity(8192 * channels),
            a: Vec::with_capacity(8192 * channels),
            stretched: Vec::with_capacity(8192 * channels),
        }
    }

    pub fn jumps(&self) -> u64 {
        self.skips + self.refills
    }

    /// audio.py `_bytes` in samples, as whole sample frames.
    fn samples(&self, seconds: f64) -> usize {
        py_round(seconds * self.rate as f64).max(0) as usize * self.channels
    }

    pub fn set_delay(&mut self, seconds: f64) {
        self.target = self.samples(seconds).min(self.samples(MAX_S - JUMP_S));
    }

    /// Samples to keep queued.
    pub fn target(&self) -> usize {
        self.target
    }

    pub fn queued_s(&self) -> f64 {
        self.buf.len() as f64 / self.channels as f64 / self.rate as f64
    }

    /// Samples to keep queued. That's the delay asked for, but never less than the longest recent packet or wait
    /// between packets, plus a cushion.
    fn keep(&self) -> usize {
        let packet = (self.packets.iter().copied().max().unwrap_or(0) + self.samples(CUSHION_S)).min(self.samples(MAX_S / 2.0));
        let wait = match self.waits.front() {
            Some(&(_, w)) => self.samples((w + CUSHION_S).min(MAX_WAIT_KEEP_S)),
            None => 0,
        };
        self.target.max(packet).max(wait)
    }

    /// The delay the queue is kept at, in seconds (more than asked for when that's too short to play smoothly).
    pub fn kept_s(&self) -> f64 {
        self.keep() as f64 / self.channels as f64 / self.rate as f64
    }

    /// A packet, as it arrives now (sp_core::clock).
    pub fn push(&mut self, pcm: &[i16]) {
        self.push_at(sp_core::clock::now(), pcm)
    }

    /// A packet that arrived at `now`, in seconds on any clock that only moves forward. Tests simulate one.
    pub fn push_at(&mut self, now: f64, pcm: &[i16]) {
        if let Some(last) = self.last_in {
            let wait = now - last;
            if wait > STOP_S {
                self.stops += 1;
            } else {
                self.longest_wait_s = self.longest_wait_s.max(wait);
                while self.waits.back().is_some_and(|w| w.1 <= wait) {
                    self.waits.pop_back(); // a shorter, older wait never counts again
                }
                self.waits.push_back((now, wait));
            }
        }
        self.last_in = Some(now);
        while self.waits.front().is_some_and(|w| w.0 < now - GAP_WINDOW_S) {
            self.waits.pop_front();
        }
        if self.packets.len() == 50 {
            self.packets.pop_front();
        }
        self.packets.push_back(pcm.len() - pcm.len() % self.channels);
        self.buf.extend(pcm.iter().copied());
        let over = self.buf.len() as i64 - self.samples(MAX_S) as i64;
        if over > 0 {
            let n = over + (-over).rem_euclid(self.channels as i64); // whole sample frames
            self.buf.drain(..(n as usize).min(self.buf.len()));
        }
    }

    /// `n` sample frames for the output, as a new Vec (audio.py's pull, in frames rather than bytes).
    pub fn pull(&mut self, frames: usize) -> Vec<i16> {
        let mut out = vec![0i16; frames * self.channels];
        self.pull_into(&mut out);
        out
    }

    /// Fills `out` (whole sample frames) for the output.
    pub fn pull_into(&mut self, out: &mut [i16]) {
        let ch = self.channels;
        let nf = out.len() / ch;
        if nf == 0 {
            return;
        }
        let out = &mut out[..nf * ch];
        let mut err = self.buf.len() as i64 - self.keep() as i64; // how far ahead of the delay the queue is
        let jump = self.samples(JUMP_S) as i64;
        if err < -jump && !self.filling {
            // too little queued (a long hold-up, or a longer delay), so refill
            self.filling = true;
            self.refills += 1;
        }
        if self.filling {
            if err < 0 {
                self.silence(out);
                return;
            }
            self.filling = false;
            self.seam = true;
        }
        if err > jump {
            // too much, so skip the oldest sound
            self.buf.drain(..err as usize);
            self.skips += 1;
            self.seam = true;
            err = 0;
        }
        let e = err.unsigned_abs() as usize;
        let speed = if e > self.samples(FAR_S) {
            FAR_NUDGE
        } else if e > self.samples(NUDGE_S) {
            NUDGE
        } else {
            0.0
        };
        let take = if speed == 0.0 {
            nf as i64
        } else if err > 0 {
            (nf as i64 + 1).max(py_round(nf as f64 * (1.0 + speed)))
        } else {
            (nf as i64 - 1).min(py_round(nf as f64 * (1.0 - speed)))
        };
        let take = take.max(1) as usize;
        let n = (take * ch).min(self.buf.len());
        self.chunk.clear();
        self.chunk.extend(self.buf.drain(..n));
        if n < take * ch {
            // Ran dry. One gap, then silent until it's back to the delay. Playing on from almost nothing, as it used
            // to, ran dry again and again.
            self.underruns += 1;
            self.filling = true;
        }
        let seam = std::mem::replace(&mut self.seam, false);
        let got = n / ch;
        if got == take && take == nf && !seam {
            // nearly always
            out.copy_from_slice(&self.chunk);
            for c in 0..ch {
                self.last[c] = self.chunk[(nf - 1) * ch + c] as f32;
            }
            return;
        }
        let a = &mut self.a;
        a.clear();
        a.extend(self.chunk[..got * ch].iter().map(|&s| s as f32));
        if seam && got > 0 {
            // from where the last block ended into this one
            let k = self.fade.min(got);
            let first: Vec<f32> = (0..ch).map(|c| self.last[c] - a[c]).collect();
            for i in 0..k {
                let r = ramp(i, k);
                for c in 0..ch {
                    a[i * ch + c] += first[c] * r;
                }
            }
        }
        if got < take {
            // ran dry, so ease down into silence, and up from it when the sound's back (seam)
            let from: Vec<f32> = if got > 0 { a[(got - 1) * ch..got * ch].to_vec() } else { self.last.clone() };
            let k = self.fade.min(take - got);
            for i in 0..take - got {
                for f in &from {
                    a.push(if i < k { f * ramp(i, k) } else { 0.0 });
                }
            }
        }
        if take != nf {
            // played a little faster (more in than out) or slower, stretched to nf frames with np.interp on float64
            let s = &mut self.stretched;
            s.clear();
            let step = if nf > 1 { (take - 1) as f64 / (nf - 1) as f64 } else { 0.0 };
            for i in 0..nf {
                let x = if nf > 1 && i == nf - 1 { (take - 1) as f64 } else { i as f64 * step };
                let j = x.floor() as usize;
                for c in 0..ch {
                    let v = if j >= take - 1 {
                        a[(take - 1) * ch + c] as f64
                    } else if x == j as f64 {
                        a[j * ch + c] as f64
                    } else {
                        let (y0, y1) = (a[j * ch + c] as f64, a[(j + 1) * ch + c] as f64);
                        (y1 - y0) * (x - j as f64) + y0
                    };
                    s.push(v);
                }
            }
            for c in 0..ch {
                self.last[c] = s[(nf - 1) * ch + c] as f32;
            }
            for (o, v) in out.iter_mut().zip(s.iter()) {
                *o = v.round_ties_even().clamp(-32768.0, 32767.0) as i16;
            }
        } else {
            for c in 0..ch {
                self.last[c] = a[(nf - 1) * ch + c];
            }
            for (o, v) in out.iter_mut().zip(a.iter()) {
                *o = v.round_ties_even().clamp(-32768.0, 32767.0) as i16;
            }
        }
    }

    /// Silent while filling, eased down from the last sound played.
    fn silence(&mut self, out: &mut [i16]) {
        let ch = self.channels;
        let nf = out.len() / ch;
        if !self.last.iter().any(|v| *v != 0.0) {
            out.fill(0);
            return;
        }
        let k = self.fade.min(nf);
        let mut last = vec![0.0f32; ch];
        for i in 0..nf {
            for c in 0..ch {
                let v = if i < k { self.last[c] * ramp(i, k) } else { 0.0 };
                out[i * ch + c] = v.round_ties_even() as i16;
                if i == nf - 1 {
                    last[c] = v;
                }
            }
        }
        if nf > 0 {
            self.last = last;
        }
    }

    /// Counts for the log ([`sound_report`]).
    pub fn stats(&self) -> SoundStats {
        SoundStats {
            ran_dry: self.underruns,
            skips: self.skips,
            refills: self.refills,
            stops: self.stops,
            longest_wait_ms: py_round(self.longest_wait_s * 1000.0),
            kept_ms: py_round(self.kept_s() * 1000.0),
            ..Default::default()
        }
    }
}

/// How the sound played, for the log, as audio.py stats() gives it. That's DelayLine's, and the Player's three when
/// it's a Player.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SoundStats {
    pub ran_dry: u64,
    pub output_short: u64,
    pub skips: u64,
    pub refills: u64,
    pub stops: u64,
    pub longest_wait_ms: i64,
    pub kept_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub longest_call_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_ms: Option<i64>,
}

impl SoundStats {
    fn counts(&self) -> [u64; 5] {
        [self.ran_dry, self.output_short, self.skips, self.refills, self.stops]
    }
}

/// A line for the log from the stats. It covers all of it when `since` is None, or what happened after `since`.
/// None if nothing happened.
pub fn sound_report(stats: &SoundStats, since: Option<&SoundStats>) -> Option<String> {
    let base = since.cloned().unwrap_or_default().counts();
    let now = stats.counts();
    let d: Vec<u64> = now.iter().zip(base.iter()).map(|(a, b)| a.saturating_sub(*b)).collect();
    if since.is_some() && d.iter().all(|v| *v == 0) {
        return None;
    }
    let mut out = format!(
        "ran dry {} times, the Mac's output ran short {} times, skipped ahead {} times, went quiet to catch up {} times",
        d[0], d[1], d[2], d[3]
    );
    if d[4] > 0 {
        out += &format!(", stopped coming {} times", d[4]);
    }
    out += &format!("; longest wait between the card's sound packets {} ms", stats.longest_wait_ms);
    if let Some(ms) = stats.longest_call_ms {
        out += &format!(", for the Mac's calls {ms} ms");
    }
    out += &format!("; held back {} ms", stats.kept_ms);
    if let Some(ms) = stats.output_ms {
        out += &format!(" (and the Mac's output {ms} ms)");
    }
    Some(out)
}

/// The output's side of audio.py's Player, without the device. It has a DelayLine behind a lock, the delay less the
/// output's latency, and the calls counted. [`crate::Player`] hands its callback to Core Audio. Tests call
/// [`PlayerCore::render`] themselves.
pub struct PlayerCore {
    pub line: Mutex<DelayLine>,
    latency_bits: AtomicU64,
    short: AtomicU64,
    longest_call_ns: AtomicU64,
    called_ns: AtomicU64,
    calls: AtomicU64,
}

impl PlayerCore {
    /// `latency` is the output's own delay in seconds, which set_delay takes off so the total is what was asked for.
    pub fn new(rate: u32, channels: usize, latency: f64) -> PlayerCore {
        PlayerCore {
            line: Mutex::new(DelayLine::new(rate, channels)),
            latency_bits: AtomicU64::new(latency.to_bits()),
            short: AtomicU64::new(0),
            longest_call_ns: AtomicU64::new(0),
            called_ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }

    pub fn latency(&self) -> f64 {
        f64::from_bits(self.latency_bits.load(Ordering::Relaxed))
    }

    pub fn set_latency(&self, seconds: f64) {
        self.latency_bits.store(seconds.to_bits(), Ordering::Relaxed);
    }

    /// The output asks for `out` (whole sample frames). `flagged` means the output says it ran short.
    pub fn render(&self, out: &mut [i16], flagged: bool) {
        // A late call counts too, because a flag isn't always given. audio.py found WASAPI's never came.
        let now = sp_core::clock::now_ns().max(1);
        let prev = self.called_ns.swap(now, Ordering::Relaxed);
        let mut late = false;
        if prev != 0 {
            let wait = now.saturating_sub(prev);
            self.longest_call_ns.fetch_max(wait, Ordering::Relaxed);
            late = wait as f64 * 1e-9 > self.latency();
        }
        if late || flagged {
            self.short.fetch_add(1, Ordering::Relaxed);
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        match self.line.lock() {
            Ok(mut line) => line.pull_into(out),
            Err(_) => out.fill(0),
        }
    }

    /// The output ran short outside a call (Core Audio's overload notice).
    pub fn ran_short(&self) {
        self.short.fetch_add(1, Ordering::Relaxed);
    }

    /// How many times the output has asked for sound.
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    pub fn push(&self, pcm: &[i16]) {
        self.line.lock().unwrap().push(pcm);
    }

    pub fn set_delay(&self, seconds: f64) {
        self.line.lock().unwrap().set_delay(seconds - self.latency());
    }

    pub fn stats(&self) -> SoundStats {
        SoundStats {
            output_short: self.short.load(Ordering::Relaxed),
            longest_call_ms: Some(py_round(self.longest_call_ns.load(Ordering::Relaxed) as f64 * 1e-6)),
            output_ms: Some(py_round(self.latency() * 1000.0)),
            ..self.line.lock().unwrap().stats()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_is_numpys() {
        // np.linspace(1.0, 0.0, 96, endpoint=False, dtype=np.float32)[[0, 1, 95]]
        assert_eq!(ramp(0, 96), 1.0);
        assert_eq!(ramp(1, 96), (1.0f64 - 1.0 / 96.0) as f32);
        assert_eq!(ramp(95, 96), (95.0 * (-1.0f64 / 96.0) + 1.0) as f32);
    }
}
