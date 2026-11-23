//! The sound ring (frametap.h "Sound ring", framegen/chiaki_tap.py AudioTap). It holds 16-bit PCM chunks timed by
//! their arrival, for instant replay. A chiaki-ng from before this ring never writes it, and its `features` lack
//! FEATURE_AUDIO_TAP.

use std::sync::atomic::Ordering;

use crate::clock;
use crate::layout::*;
use crate::shm::{fence, Shm};
use crate::Result;

/// A chunk of sound, with its samples and when the first of them arrived (seconds, on perf_counter's timebase).
#[derive(Clone, Debug, PartialEq)]
pub struct SoundChunk {
    pub t: f64,
    /// Interleaved signed 16-bit samples, as the ring holds them (little-endian bytes).
    pub pcm: Vec<u8>,
    pub rate: u32,
    pub channels: u32,
}

impl SoundChunk {
    /// The samples as i16.
    pub fn samples(&self) -> Vec<i16> {
        self.pcm.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
    }
}

pub struct AudioTap {
    shm: Shm,
    /// The next chunk to read. None starts from wherever the writer is now.
    pub next: Option<u64>,
}

impl AudioTap {
    /// The sound ring of the tap `tap_name` (the tap's name + "-audio"), created if it isn't there.
    pub fn open(tap_name: &str) -> Result<AudioTap> {
        let shm = Shm::open(&format!("{tap_name}{AUDIO_SUFFIX}"), AUDIO_TOTAL)?;
        Ok(AudioTap { shm, next: None })
    }

    pub fn live(&self) -> bool {
        self.shm.u32(0) == AUDIO_MAGIC
    }

    /// Forget the read position, for a new stream. The next read starts at the writer's position, or at its first
    /// chunk with `from_start`.
    pub fn reset(&mut self, from_start: bool) {
        self.next = if from_start { Some(0) } else { None };
    }

    /// Every chunk written since the last call (at most the ring's worth).
    pub fn read_new(&mut self) -> Vec<SoundChunk> {
        let m = &self.shm;
        let magic = m.u32(0);
        let write_seq = m.atomic_u64(24).load(Ordering::Acquire);
        let freq = m.u64(32);
        if magic != AUDIO_MAGIC || freq == 0 {
            self.next = None;
            return Vec::new();
        }
        let mut next = match self.next {
            Some(n) if n <= write_seq => n,
            _ => write_seq, // first read, or the writer started over
        };
        next = next.max((write_seq + 1).saturating_sub(AUDIO_SLOTS as u64)); // a whole ring behind, take what's left
        let mut out = Vec::new();
        while next < write_seq {
            let seq = next;
            next += 1;
            let off = AUDIO_HEADER + (seq % AUDIO_SLOTS as u64) as usize * AUDIO_SLOT_SIZE;
            fence();
            let s0 = m.u64(off);
            let q = m.u64(off + 8);
            let size = m.u32(off + 16) as usize;
            let rate = m.u32(off + 20);
            let channels = m.u32(off + 24);
            if s0 != seq || size > AUDIO_SLOT_SIZE - AUDIO_SLOT_HEADER {
                continue;
            }
            fence();
            let mut pcm = vec![0u8; size];
            m.copy_out(off + AUDIO_SLOT_HEADER, &mut pcm);
            fence();
            if m.u64(off) != seq {
                continue; // rewritten while we copied
            }
            out.push(SoundChunk { t: clock::to_seconds(q, freq), pcm, rate, channels });
        }
        self.next = Some(next);
        out
    }

    // Writer side, for tests and stand-in writers. chiaki-ng uses frametap.c.
    pub fn begin(&self, rate: u32, channels: u32) {
        let m = &self.shm;
        m.xchg_u32(0, 0);
        m.set_u32(4, 1);
        m.set_u32(8, AUDIO_SLOTS as u32);
        m.set_u32(12, AUDIO_SLOT_SIZE as u32);
        m.set_u32(16, rate);
        m.set_u32(20, channels);
        m.set_u64(32, clock::qpc_freq());
        m.xchg_u64(24, 0);
        m.xchg_u32(0, AUDIO_MAGIC);
    }

    /// Interleaved 16-bit samples (bytes), the first of them arriving at `t_qpc` (now if None). A chunk too big for
    /// a slot goes in several, each timed by the samples before it.
    pub fn push(&self, pcm: &[u8], t_qpc: Option<u64>) {
        let m = &self.shm;
        let (rate, channels) = (m.u32(16).max(1) as usize, m.u32(20).max(1) as usize);
        let frame = 2 * channels;
        let cap = (AUDIO_SLOT_SIZE - AUDIO_SLOT_HEADER) / frame * frame;
        let q0 = t_qpc.unwrap_or_else(clock::qpc);
        let freq = m.u64(32);
        for (k, part) in pcm.chunks(cap).enumerate() {
            let i = k * cap;
            let seq = m.atomic_u64(24).load(Ordering::SeqCst);
            let off = AUDIO_HEADER + (seq % AUDIO_SLOTS as u64) as usize * AUDIO_SLOT_SIZE;
            m.set_u64(off, u64::MAX);
            fence();
            m.set_u64(off + 8, q0 + ((i / frame) as f64 * freq as f64 / rate as f64).round() as u64);
            m.set_u32(off + 16, part.len() as u32);
            m.set_u32(off + 20, rate as u32);
            m.set_u32(off + 24, channels as u32);
            m.copy_in(off + AUDIO_SLOT_HEADER, part);
            fence();
            m.set_u64(off, seq);
            m.xchg_u64(24, seq + 1);
        }
    }

    pub fn end(&self) {
        self.shm.xchg_u32(0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shm::unlink;

    #[test]
    fn roundtrip_split_and_times() {
        let name = format!("sp-rs-a{}", std::process::id());
        unlink(&name);
        let mut a = AudioTap::open(&name).unwrap();
        a.end();
        assert!(!a.live() && a.read_new().is_empty());
        a.begin(48000, 2);
        assert!(a.live());
        a.reset(true);
        let big: Vec<u8> = (0..20480).map(|i| (i % 256) as u8).collect();
        let q0 = clock::qpc();
        a.push(&[1, 2, 3, 4].repeat(480), Some(q0));
        a.push(&big, Some(q0));
        let got = a.read_new();
        let cap = (AUDIO_SLOT_SIZE - AUDIO_SLOT_HEADER) / 4 * 4;
        assert_eq!(got.iter().map(|c| c.pcm.len()).collect::<Vec<_>>(), vec![1920, cap, cap, big.len() - 2 * cap]);
        assert_eq!(got[1..].iter().flat_map(|c| c.pcm.clone()).collect::<Vec<_>>(), big);
        assert!((got[2].t - got[1].t - cap as f64 / 4.0 / 48000.0).abs() < 1e-4);
        assert!((clock::now() - got[0].t).abs() < 1.0);
        assert!(a.read_new().is_empty());
        for _ in 0..AUDIO_SLOTS + 10 {
            a.push(&[0u8; 64], None);
        }
        assert_eq!(a.read_new().len(), AUDIO_SLOTS - 1);
        a.begin(48000, 2); // the writer started over, and the reader follows it
        a.push(&[5u8; 64], None);
        assert!(a.read_new().is_empty() && a.next == Some(1));
        a.end();
        assert!(a.read_new().is_empty() && a.next.is_none());
        drop(a);
        unlink(&name);
    }
}
