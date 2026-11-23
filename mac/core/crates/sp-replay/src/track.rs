//! One stream of the replay: encoded packets as pieces (pts, keyframe, bytes), oldest first (replay.py _Track), and
//! cutting a stretch out of one (_cut).

use std::collections::VecDeque;
use std::sync::Arc;

/// What a track's pieces are: the format an MP4 (or VideoToolbox's decoder) needs to read them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Format {
    /// H.264, each piece an access unit in AVCC form (4-byte big-endian lengths); `avcc` is the avcC box's payload.
    H264 { width: u32, height: u32, avcc: Vec<u8> },
    /// AAC-LC, each piece one raw frame of 1024 samples; `asc` is the AudioSpecificConfig.
    Aac { rate: u32, channels: u32, asc: Vec<u8>, bitrate: u32 },
}

/// An encoded packet.
#[derive(Clone, Debug)]
pub struct Piece {
    /// When it's shown, on the 90 kHz clock (every track on the same one).
    pub pts: i64,
    /// The same in the track's own timescale (video: 90 kHz, so `pts`; sound: 48 kHz samples), exact.
    pub ts: i64,
    pub key: bool,
    pub data: Arc<[u8]>,
}

#[derive(Debug)]
pub struct Track {
    pub name: &'static str,
    pub ring: VecDeque<Piece>,
    /// pts of the keyframes in ring (every piece for sound).
    pub keys: VecDeque<i64>,
    pub nbytes: usize,
    pub format: Option<Arc<Format>>,
    /// Bumped whenever the track starts over (a new encoder, or off), so a late packet from before is left out.
    pub generation: u64,
    pub last_pts: Option<i64>,
    pub next_key: Option<i64>,
}

impl Track {
    pub fn new(name: &'static str) -> Track {
        Track {
            name,
            ring: VecDeque::new(),
            keys: VecDeque::new(),
            nbytes: 0,
            format: None,
            generation: 0,
            last_pts: None,
            next_key: None,
        }
    }

    pub fn is_sound(&self) -> bool {
        self.name == "sound"
    }

    /// Empties it and starts a new generation.
    pub fn clear(&mut self) {
        self.ring.clear();
        self.keys.clear();
        self.nbytes = 0;
        self.format = None;
        self.last_pts = None;
        self.next_key = None;
        self.generation += 1;
    }

    pub fn add(&mut self, p: Piece) {
        if p.key || self.is_sound() {
            self.keys.push_back(p.pts);
        }
        self.nbytes += p.data.len();
        self.ring.push_back(p);
    }

    pub fn newest(&self) -> Option<i64> {
        self.ring.back().map(|p| p.pts)
    }

    /// Drop what's older than `cut`, keeping the keyframe at or before it.
    pub fn trim(&mut self, cut: i64) {
        while self.keys.len() >= 2 && self.keys[1] <= cut {
            self.keys.pop_front();
            let k = self.keys[0];
            while self.ring.front().is_some_and(|p| p.pts < k) {
                let p = self.ring.pop_front().unwrap();
                self.nbytes -= p.data.len();
            }
        }
    }

    /// Seconds from the oldest piece to the newest.
    pub fn seconds(&self) -> f64 {
        match (self.ring.front(), self.ring.back()) {
            (Some(a), Some(b)) if self.ring.len() > 1 => (b.pts - a.pts) as f64 / crate::TB as f64,
            _ => 0.0,
        }
    }
}

/// The pieces of a stretch: from the last keyframe at or before `start` (or with `at_key` false, the first piece at
/// or after it), up to `end`.
pub fn cut<'a>(ring: impl IntoIterator<Item = &'a Piece>, start: i64, end: i64, at_key: bool) -> Vec<Piece> {
    let ring: Vec<&Piece> = ring.into_iter().collect();
    let mut i = 0;
    for (j, p) in ring.iter().enumerate() {
        if p.pts > start {
            break;
        }
        if p.key || !at_key {
            i = j;
        }
    }
    if !at_key {
        while i < ring.len() && ring[i].pts < start {
            i += 1;
        }
    }
    ring[i.min(ring.len())..].iter().filter(|p| p.pts <= end).map(|p| (*p).clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(pts: i64, key: bool) -> Piece {
        Piece { pts, ts: pts, key, data: Arc::from(vec![0u8; 10]) }
    }

    #[test]
    fn trims_to_the_keyframe_before_the_cut() {
        let mut t = Track::new("after");
        for i in 0..100 {
            t.add(piece(i * 1000, i % 10 == 0));
        }
        t.trim(35_000);
        assert_eq!(t.ring.front().unwrap().pts, 30_000);
        assert_eq!(t.nbytes, 70 * 10);
        let c = cut(&t.ring, 55_000, 80_000, true);
        assert_eq!((c[0].pts, c.last().unwrap().pts), (50_000, 80_000));
        let c = cut(&t.ring, 55_500, 80_000, false);
        assert_eq!(c[0].pts, 56_000);
    }
}
