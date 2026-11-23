//! New frame or repeat from compressed size alone (framegen/chiaki_tap.py SizeRepeatDetector, line for line). Not
//! the pipeline's call, which is the picture's; kept for the stream check, test mode and the frame log.

use std::collections::VecDeque;

/// numpy.percentile's default (linear between the closest ranks).
pub fn percentile(values: impl IntoIterator<Item = u64>, q: f64) -> f64 {
    let mut v: Vec<f64> = values.into_iter().map(|x| x as f64).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let f = pos - lo as f64;
    v[lo] + (v[hi] - v[lo]) * f
}

/// A frame is new if it's at least `ratio` of a typical recent new frame (the 75th percentile of the last 16
/// non-keyframe sizes). Keyframes (and frames far bigger than anything recent) get the opposite of the frame before
/// if recent frames alternate (a 30 fps game), else None. See the Python docstring for the reasoning.
#[derive(Clone, Debug)]
pub struct SizeRepeatDetector {
    pub ratio: f64,
    window: usize,
    recent: VecDeque<(u64, bool)>,
    sizes: VecDeque<u64>,
    decisions: VecDeque<bool>,
    /// The last update's typical new frame size (None: keyframe or warming up).
    pub r#ref: Option<f64>,
    /// The last update was treated as a keyframe for its size.
    pub big: bool,
}

impl Default for SizeRepeatDetector {
    fn default() -> Self {
        Self::new(0.4, 16)
    }
}

impl SizeRepeatDetector {
    /// This many "big" frames among the last 2 * BIG_RUN aren't refresh frames but a busier scene.
    pub const BIG_RUN: usize = 6;

    pub fn new(ratio: f64, window: usize) -> Self {
        SizeRepeatDetector {
            ratio,
            window,
            recent: VecDeque::with_capacity(2 * Self::BIG_RUN),
            sizes: VecDeque::with_capacity(window),
            decisions: VecDeque::with_capacity(6),
            r#ref: None,
            big: false,
        }
    }

    fn push_size(&mut self, s: u64) {
        if self.sizes.len() == self.window {
            self.sizes.pop_front();
        }
        self.sizes.push_back(s);
    }

    pub fn update(&mut self, size: u64, keyframe: bool) -> Option<bool> {
        self.big = !keyframe
            && self.sizes.len() >= 8
            && (size as f64) > 4.0 * percentile(self.sizes.iter().copied(), 75.0);
        if !keyframe {
            if self.recent.len() == 2 * Self::BIG_RUN {
                self.recent.pop_front();
            }
            self.recent.push_back((size, self.big));
        }
        if self.recent.iter().filter(|r| r.1).count() >= Self::BIG_RUN {
            self.sizes.clear();
            let keep: Vec<u64> = self.recent.iter().take(self.recent.len() - 1).map(|r| r.0).collect();
            for s in keep {
                self.push_size(s);
            }
            self.recent.clear();
            self.big = false;
        }
        self.r#ref = None;
        let decision = if keyframe || self.big {
            let d: Vec<bool> = self.decisions.iter().copied().collect();
            let alternating = d.len() >= 4 && d.windows(2).all(|w| w[0] != w[1]);
            if alternating { Some(!d[d.len() - 1]) } else { None }
        } else {
            self.push_size(size);
            if self.sizes.len() >= 4 {
                self.r#ref = Some(percentile(self.sizes.iter().copied(), 75.0));
            }
            self.r#ref.map(|r| size as f64 >= self.ratio * r)
        };
        if let Some(d) = decision {
            if self.decisions.len() == 6 {
                self.decisions.pop_front();
            }
            self.decisions.push_back(d);
        }
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy() {
        assert_eq!(percentile([1, 2, 3, 4], 75.0), 3.25);
        assert_eq!(percentile([10], 75.0), 10.0);
        assert_eq!(percentile([5, 1, 9, 3, 7], 75.0), 7.0);
    }

    #[test]
    fn cadence_30fps() {
        let mut d = SizeRepeatDetector::default();
        let out: Vec<_> = [60000u64, 3000].repeat(10).into_iter().map(|s| d.update(s, false)).collect();
        assert_eq!(out[..3], [None, None, None]);
        let want: Vec<_> = [Some(true), Some(false)].repeat(8);
        assert_eq!(out[4..], want[..]);
        assert_eq!(d.update(150000, true), Some(true));
        assert_eq!(d.update(2500, false), Some(false));
    }

    #[test]
    fn all_new_at_60() {
        let mut d = SizeRepeatDetector::default();
        let out: Vec<_> =
            [50000u64, 52000, 47000, 61000, 55000, 49000, 58000, 51000].into_iter().map(|s| d.update(s, false)).collect();
        assert!(out[4..].iter().all(|o| *o == Some(true)));
    }

    #[test]
    fn recovers_after_a_still_stretch() {
        // As the Python test, with a simple deterministic jitter in place of numpy's normal draws.
        let mut x = 12345u64;
        let mut jit = |amp: f64| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0 * amp
        };
        let mut d = SizeRepeatDetector::default();
        for _ in 0..180 {
            d.update((600.0 + jit(300.0)).max(50.0) as u64, false);
        }
        let out: Vec<_> = (0..300)
            .map(|i| {
                let s = if i % 2 == 0 { 50000.0 + jit(12000.0) } else { 2000.0 + jit(450.0) };
                d.update(s as u64, false)
            })
            .collect();
        let want: Vec<_> = [Some(true), Some(false)].repeat(140);
        assert_eq!(out[20..], want[..]);
    }
}
