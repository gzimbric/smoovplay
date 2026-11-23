//! The header and the frame slots (framegen/chiaki_tap.py FrameTap). chiaki_tap/frametap.c writes the same.

use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::clock;
use crate::layout::*;
use crate::shm::{check_name, fence, Shm};
use crate::Result;

/// The header as a reader sees it. `slots` and `slot_size` are the writer's layout. A version 1 writer always uses
/// SLOTS and SLOT_SIZE, and a version 2 writer uses its own if they fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub magic: u32,
    pub version: u32,
    pub codec: u32,
    pub width: u32,
    pub height: u32,
    pub max_fps: u32,
    pub write_seq: u64,
    pub session: u32,
    pub qpc_freq: u64,
    pub slots: usize,
    pub slot_size: usize,
    /// NV12's colours, from a version 2 writer (layout::colour_of).
    pub colour: u32,
    pub features: u32,
}

impl Header {
    /// A stream is on.
    pub fn live(&self) -> bool {
        self.magic == MAGIC
    }
    /// What changes when a frame comes or a stream starts or ends, which [`FrameTap::wait`] watches.
    pub fn watch(&self) -> Watch {
        Watch { magic: self.magic, write_seq: self.write_seq, session: self.session }
    }
}

/// The (magic, write_seq, session) that [`Header::watch`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Watch {
    pub magic: u32,
    pub write_seq: u64,
    pub session: u32,
}

/// One frame copied out of its slot.
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    pub seq: u64,
    /// When the frame reached this Mac, in seconds on the clock Python's `time.perf_counter()` reads.
    pub t_arrival: f64,
    /// The frame's bytes (empty when read with [`FrameTap::read_into`]).
    pub data: Vec<u8>,
    pub frames_lost: i32,
    pub recovered: bool,
    pub truncated: bool,
    /// The slot's flags (FLAG_*).
    pub flags: u32,
    /// Bytes in the slot.
    pub size: usize,
}

/// The shared ring. Either side may create it first, and both see the same memory. A reader and a writer may share
/// one FrameTap between threads, since every access is volatile or atomic.
pub struct FrameTap {
    shm: Shm,
}

impl FrameTap {
    /// Opens (or creates) the tap `name`. Err if the name doesn't fit this system's shared memory with the sound ring's
    /// suffix ([`check_name`]), before any memory is made.
    pub fn open(name: &str) -> Result<FrameTap> {
        check_name(name)?;
        let shm = Shm::open(name, TOTAL)?;
        Ok(FrameTap { shm })
    }

    pub fn shm(&self) -> &Shm {
        &self.shm
    }

    pub fn header(&self) -> Header {
        let m = &self.shm;
        let magic = m.u32(OFF_MAGIC);
        let version = m.u32(OFF_VERSION);
        let mut slots = m.u32(OFF_SLOT_COUNT) as usize;
        let mut slot_size = m.u32(OFF_SLOT_SIZE) as usize;
        let codec = m.u32(OFF_CODEC);
        let width = m.u32(OFF_WIDTH);
        let height = m.u32(OFF_HEIGHT);
        let max_fps = m.u32(OFF_MAX_FPS);
        let write_seq = m.atomic_u64(OFF_WRITE_SEQ).load(Ordering::Acquire);
        let session = m.u32(OFF_SESSION);
        let qpc_freq = m.u64(OFF_QPC_FREQ);
        let features = m.u32(OFF_FEATURES);
        let mut colour = m.u32(OFF_COLOUR);
        if version < VERSION_RAW
            || !((4..=SLOTS).contains(&slots) && SLOT_HEADER < slot_size && slots * slot_size <= CAPACITY)
        {
            (slots, slot_size, colour) = (SLOTS, SLOT_SIZE, 0); // version 1 (chiaki-ng) has the fixed layout
        }
        Header {
            magic,
            version,
            codec,
            width,
            height,
            max_fps,
            write_seq,
            session,
            qpc_freq,
            slots,
            slot_size,
            colour,
            features,
        }
    }

    /// The watched fields only (cheaper than a whole header).
    pub fn watch(&self) -> Watch {
        Watch {
            magic: self.shm.u32(OFF_MAGIC),
            write_seq: self.shm.atomic_u64(OFF_WRITE_SEQ).load(Ordering::Acquire),
            session: self.shm.u32(OFF_SESSION),
        }
    }

    // ------------------------------------------------------------------ reader side
    /// Ask the writer for a keyframe.
    pub fn request_idr(&self) {
        self.shm.atomic_u32(OFF_IDR_REQUEST).store(1, Ordering::SeqCst);
    }

    /// Tell the writer the reader is still there (it drops the sound delay once this is over 2 s old).
    pub fn heartbeat(&self) {
        self.shm.atomic_u64(OFF_HEARTBEAT).store(clock::qpc(), Ordering::SeqCst);
    }

    /// The writer's `features` while a stream is on, else 0. A chiaki-ng from an older patch leaves them 0.
    pub fn features(&self) -> u32 {
        if self.header().live() { self.shm.u32(OFF_FEATURES) } else { 0 }
    }

    /// Ask the writer to press `buttons` (a ChiakiControllerButton mask) for hold_ms. Returns a token for
    /// [`FrameTap::press_sent`].
    pub fn press(&self, buttons: u32, hold_ms: u32) -> u32 {
        let count = self.shm.atomic_u32(OFF_PRESS_COUNT).load(Ordering::SeqCst);
        self.shm.set_u32(OFF_PRESS_HOLD_MS, hold_ms);
        fence(); // the hold before the buttons, because the writer reads them in the other order
        self.shm.atomic_u32(OFF_PRESS_BUTTONS).store(buttons, Ordering::SeqCst);
        count
    }

    /// When the writer sent the press to the PS5 (seconds, perf_counter's timebase), or None if it hasn't yet.
    pub fn press_sent(&self, token: u32, qpc_freq: u64) -> Option<f64> {
        let count = self.shm.atomic_u32(OFF_PRESS_COUNT).load(Ordering::SeqCst);
        if count == token || qpc_freq == 0 {
            return None;
        }
        let q = self.shm.atomic_u64(OFF_PRESS_QPC).load(Ordering::SeqCst);
        Some(clock::to_seconds(q, qpc_freq))
    }

    /// Ask the writer to hold its sound back this long (0 to 500 ms).
    pub fn set_audio_delay(&self, ms: i64) {
        self.shm.atomic_u32(OFF_AUDIO_DELAY_MS).store(ms.clamp(0, 500) as u32, Ordering::SeqCst);
    }

    /// Waits until the writer sends a frame or starts or ends a stream (anything different from `seen`), or for
    /// timeout_s. There's no event on macOS, so this looks at the header every POLL_S (half a millisecond).
    pub fn wait(&self, seen: Watch, timeout_s: f64) {
        let t_end = clock::now() + timeout_s;
        loop {
            if self.watch() != seen {
                return;
            }
            let left = t_end - clock::now();
            if left <= 0.0 {
                return;
            }
            std::thread::sleep(Duration::from_secs_f64(left.min(POLL_S)));
        }
    }

    fn slot_off(seq: u64, slots: usize, slot_size: usize) -> usize {
        HEADER + (seq % slots as u64) as usize * slot_size
    }

    /// Frame `seq` copied out of its slot, or None if the writer has already reused the slot or rewrote it while it
    /// was being copied. `h` is the header, for the writer's layout and clock.
    pub fn read(&self, seq: u64, h: &Header) -> Option<Packet> {
        self.read_with(seq, h, None)
    }

    /// As [`FrameTap::read`], but into `into` (a decoded frame, megabytes) instead of a new Vec. The Packet's data is
    /// then empty and its size is in `size`. Also None if the frame doesn't fit `into`.
    pub fn read_into(&self, seq: u64, h: &Header, into: &mut [u8]) -> Option<Packet> {
        self.read_with(seq, h, Some(into))
    }

    fn read_with(&self, seq: u64, h: &Header, into: Option<&mut [u8]>) -> Option<Packet> {
        let m = &self.shm;
        let off = Self::slot_off(seq, h.slots, h.slot_size);
        fence(); // after the header's write_seq was read
        let s0 = m.u64(off);
        let q = m.u64(off + 8);
        let size = m.u32(off + 16) as usize;
        let lost = m.i32(off + 20);
        let rec = m.u32(off + 24);
        let flags = m.u32(off + 28);
        if s0 != seq || size > h.slot_size - SLOT_HEADER {
            return None;
        }
        fence();
        let data = match into {
            None => {
                let mut v = vec![0u8; size];
                m.copy_out(off + SLOT_HEADER, &mut v);
                v
            }
            Some(buf) => {
                if size > buf.len() {
                    return None;
                }
                m.copy_out(off + SLOT_HEADER, &mut buf[..size]);
                Vec::new()
            }
        };
        fence();
        if m.u64(off) != seq {
            return None; // rewritten while we copied
        }
        Some(Packet {
            seq,
            t_arrival: clock::to_seconds(q, h.qpc_freq),
            data,
            frames_lost: lost,
            recovered: rec != 0,
            truncated: flags & FLAG_TRUNCATED != 0,
            flags,
            size,
        })
    }

    // ------------------------------------------------------------------ writer side
    // For tests, stand-in writers and apps. chiaki-ng uses frametap.c.

    /// Start a stream. A decoded format (CODEC_NV12, CODEC_BGRA) gets version 2 and slots of its own
    /// (raw_layout). `features`, if given, replaces the header's features field.
    pub fn begin(&self, codec: u32, width: u32, height: u32, max_fps: u32, colour: u32, features: Option<u32>) -> Result<()> {
        let m = &self.shm;
        let (version, slots, slot_size) = match RawFormat::of_codec(codec) {
            Some(fmt) => {
                let (n, s) = raw_layout(fmt, width as usize, height as usize)?;
                (VERSION_RAW, n, s)
            }
            None => (VERSION, SLOTS, SLOT_SIZE),
        };
        m.xchg_u32(OFF_MAGIC, 0); // the header is being rewritten
        m.set_u32(OFF_VERSION, version);
        m.set_u32(OFF_SLOT_COUNT, slots as u32);
        m.set_u32(OFF_SLOT_SIZE, slot_size as u32);
        m.set_u32(OFF_CODEC, codec);
        m.set_u32(OFF_WIDTH, width);
        m.set_u32(OFF_HEIGHT, height);
        m.set_u32(OFF_MAX_FPS, max_fps);
        m.set_u32(OFF_IDR_REQUEST, 0);
        m.set_u64(OFF_QPC_FREQ, clock::qpc_freq());
        m.set_u32(OFF_COLOUR, if version == VERSION_RAW { colour } else { 0 });
        if let Some(f) = features {
            m.set_u32(OFF_FEATURES, f);
        }
        m.xchg_u64(OFF_WRITE_SEQ, 0);
        m.atomic_u32(OFF_SESSION).fetch_add(1, Ordering::SeqCst);
        m.xchg_u32(OFF_MAGIC, MAGIC); // last, once the header is complete
        Ok(())
    }

    /// Writes one frame, as compressed bytes or a decoded frame (rows packed tightly). `new` is the writer's own word
    /// on it: Some(true) for new content, Some(false) for a repeat of the frame before, None if unknown. Never waits
    /// for the reader. The slots are the ones the header gives, from whichever FrameTap started the stream.
    pub fn push(&self, data: &[u8], frames_lost: i32, recovered: bool, new: Option<bool>) {
        let m = &self.shm;
        let h = self.header();
        let (slots, slot_size) = (h.slots, h.slot_size);
        let seq = m.atomic_u64(OFF_WRITE_SEQ).load(Ordering::SeqCst);
        let off = Self::slot_off(seq, slots, slot_size);
        let size = data.len().min(slot_size - SLOT_HEADER);
        let flags = if size < data.len() { FLAG_TRUNCATED } else { 0 }
            | match new {
                Some(true) => FLAG_NEW,
                Some(false) => FLAG_REPEAT,
                None => 0,
            };
        m.set_u64(off, u64::MAX); // being rewritten, so a reader copying it sees the change and drops it
        fence();
        m.set_u64(off + 8, clock::qpc());
        m.set_u32(off + 16, size as u32);
        m.set_i32(off + 20, frames_lost);
        m.set_u32(off + 24, recovered as u32);
        m.set_u32(off + 28, flags);
        m.copy_in(off + SLOT_HEADER, &data[..size]);
        fence();
        m.set_u64(off, seq);
        m.xchg_u64(OFF_WRITE_SEQ, seq + 1);
    }

    /// True once per reader request for a keyframe.
    pub fn take_idr_request(&self) -> bool {
        self.shm.atomic_u32(OFF_IDR_REQUEST).compare_exchange(1, 0, Ordering::SeqCst, Ordering::SeqCst).is_ok()
    }

    pub fn set_features(&self, features: u32) {
        self.shm.atomic_u32(OFF_FEATURES).store(features, Ordering::SeqCst);
    }

    /// The buttons the reader asked to press and how long to hold them (20 to 2000 ms), once per request, or None if
    /// there's no request.
    pub fn take_press_request(&self) -> Option<(u32, u32)> {
        let buttons = self.shm.xchg_u32(OFF_PRESS_BUTTONS, 0);
        if buttons == 0 {
            return None;
        }
        Some((buttons, self.shm.u32(OFF_PRESS_HOLD_MS).clamp(20, 2000)))
    }

    /// The press went to the (fake) PS5 now.
    pub fn press_done(&self) {
        self.shm.xchg_u64(OFF_PRESS_QPC, clock::qpc());
        self.shm.atomic_u32(OFF_PRESS_COUNT).fetch_add(1, Ordering::SeqCst);
    }

    /// The sound delay the reader asked for, as it wrote it (frametap.c's version also lapses without a heartbeat).
    pub fn audio_delay_ms(&self) -> u32 {
        self.shm.u32(OFF_AUDIO_DELAY_MS)
    }

    /// The stream has ended.
    pub fn end(&self) {
        self.shm.xchg_u32(OFF_MAGIC, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shm::unlink;

    fn tap(name: &str) -> FrameTap {
        unlink(name);
        FrameTap::open(name).unwrap()
    }

    #[test]
    fn roundtrip_overwrite_session_and_requests() {
        let name = format!("sp-rs-u{}", std::process::id());
        let t = tap(&name);
        t.begin(CODEC_H265, 1920, 1080, 60, 0, None).unwrap();
        let h = t.header();
        assert!(h.live() && (h.codec, h.width, h.height, h.max_fps, h.write_seq) == (1, 1920, 1080, 60, 0));
        t.push(b"\x00\x00\x01frame-zero", 0, false, None);
        t.push(&b"second".repeat(1000), 2, true, Some(true));
        let h = t.header();
        assert_eq!(h.write_seq, 2);
        let (p0, p1) = (t.read(0, &h).unwrap(), t.read(1, &h).unwrap());
        assert_eq!(p0.data, b"\x00\x00\x01frame-zero");
        assert!(p0.frames_lost == 0 && !p0.recovered && p0.flags == 0);
        assert_eq!(p1.data, b"second".repeat(1000));
        assert!(p1.frames_lost == 2 && p1.recovered && !p1.truncated && p1.flags == FLAG_NEW);
        assert!((clock::now() - p0.t_arrival).abs() < 1.0);
        // A whole ring later, slot 0 holds frame SLOTS
        for i in 2..=SLOTS as u64 {
            t.push(&[i as u8; 10], 0, false, None);
        }
        let h = t.header();
        assert!(t.read(0, &h).is_none());
        assert_eq!(t.read(SLOTS as u64, &h).unwrap().data, vec![SLOTS as u8; 10]);
        // Keyframe requests, sessions
        let s = h.session;
        t.request_idr();
        assert!(t.take_idr_request() && !t.take_idr_request());
        t.end();
        assert!(!t.header().live());
        t.begin(CODEC_H264, 64, 64, 60, 0, None).unwrap();
        assert!(t.header().session == s + 1 && t.header().write_seq == 0);
        // Presses and the sound delay
        let token = t.press(BUTTON_OPTIONS, 5000);
        assert_eq!(t.press_sent(token, h.qpc_freq), None);
        assert_eq!(t.take_press_request(), Some((BUTTON_OPTIONS, 2000)));
        assert_eq!(t.take_press_request(), None);
        t.press_done();
        let sent = t.press_sent(token, h.qpc_freq).unwrap();
        assert!((clock::now() - sent).abs() < 0.5);
        t.set_audio_delay(900);
        assert_eq!(t.audio_delay_ms(), 500);
        // Decoded frames get version 2 and their own slots
        t.begin(CODEC_NV12, 1920, 1080, 60, 1, Some(FEATURE_VARIABLE)).unwrap();
        let h = t.header();
        assert_eq!((h.version, h.slots, h.slot_size, h.colour, h.features), (2, 21, 3 << 20, 1, FEATURE_VARIABLE));
        let frame = vec![7u8; 1920 * 1080 * 3 / 2];
        t.push(&frame, 0, false, Some(false));
        let mut buf = vec![0u8; frame.len()];
        let p = t.read_into(0, &t.header(), &mut buf).unwrap();
        assert!(p.data.is_empty() && p.size == frame.len() && buf == frame && p.flags == FLAG_REPEAT);
        drop(t);
        unlink(&name);
    }

    #[test]
    fn wait_wakes_on_a_frame_and_times_out() {
        let name = format!("sp-rs-w{}", std::process::id());
        let t = tap(&name);
        t.begin(CODEC_H265, 64, 64, 60, 0, None).unwrap();
        let writer = FrameTap::open(&name).unwrap();
        let seen = t.header().watch();
        let t0 = clock::now();
        t.wait(seen, 0.05);
        let waited = clock::now() - t0;
        assert!(waited > 0.045 && waited < 0.2, "{waited}");
        let th = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            writer.push(b"frame", 0, false, None);
            writer
        });
        let t0 = clock::now();
        t.wait(seen, 2.0);
        let waited = clock::now() - t0;
        let writer = th.join().unwrap();
        assert!(t.header().write_seq == 1 && waited < 0.5, "{waited}");
        writer.push(b"another", 0, false, None);
        let t0 = clock::now();
        t.wait(seen, 2.0); // changed since `seen`, so it returns at once
        assert!(clock::now() - t0 < 0.05);
        drop((t, writer));
        unlink(&name);
    }
}
