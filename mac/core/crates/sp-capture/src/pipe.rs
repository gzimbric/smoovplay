//! What every kind of [`crate::CardSource`] does with each frame and each packet of sound, whoever delivers them (a
//! card's capture session, a file standing in, raw frames in a test). It's capture.py's `_reader`, `_frames`,
//! `_damaged`, `_sound`, `_play` and `_report_sound` in one place, so a stand-in runs the same lines as a card.
//!
//! - Frames are numbered as they come off the card (`extra["seq"]`), damaged ones and ones macOS dropped included.
//!   A frame dropped from the queue then leaves a gap rather than shifting the count, which Game frame rate 30 fps
//!   counts by. Test mode is applied before the queue. When the pipeline falls behind the oldest tick goes, counted
//!   in `dropped`.
//! - A frame that can't be read is skipped and counted, and said once. Capture stops only when none can be read for
//!   [`DAMAGED_GIVE_UP_S`].
//! - A card's sound is played as it comes, held back by the pipeline's delay less however much later the sound
//!   arrives than the picture. Both are timed by the session's clock. A file's sound waits until the picture gets to
//!   it. Everything played is kept for instant replay (`read_sound`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};
use serde_json::json;
use sp_core::clock::now;
use sp_core::{Frame, Sound, Tick};
use sp_media::queue::TickQueue;
use sp_media::test_mode::{self, SharedTestMode};
use sp_media::Converter;

use crate::audio::{sound_report, SoundStats};
use crate::devices::g;
use crate::{fourcc, fourcc_name};

/// No frame readable for this long means the card's frames aren't readable at all, and capture stops. It's generous
/// because a stick can send broken frames for a moment while it settles on the signal.
pub const DAMAGED_GIVE_UP_S: f64 = 5.0;
/// Sound chunks kept for instant replay until it reads them.
const REPLAY_KEPT: usize = 2000;
/// Recent arrivals kept to tell how late each stream comes.
const LATE_KEPT: usize = 120;

/// What plays the card's sound (framegen/audio.py Player, or a stand-in in tests). It's made when the first sound
/// comes, with its rate and channels.
pub trait SoundOut: Send {
    /// Interleaved 16-bit PCM, as it arrives.
    fn push(&mut self, pcm: &[i16]);
    /// How long to hold the sound back, seconds.
    fn set_delay(&mut self, seconds: f64);
    /// How it's played, for the log. None for a stand-in that doesn't say.
    fn stats(&self) -> Option<SoundStats> {
        None
    }
    fn close(&mut self) {}
}

/// Makes a [`SoundOut`] for (rate, channels). An error plays the picture without sound.
pub type SoundFactory = Box<dyn FnMut(u32, u32) -> Result<Box<dyn SoundOut>, String> + Send>;

/// Python's `str(e)[:n]`.
pub(crate) fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

struct Frames {
    conv: Converter,
    /// Frames numbered so far (as the card sent them).
    n: u64,
    /// When the current run of damaged frames began.
    damaged_since: Option<f64>,
}

struct SoundState {
    out: Option<SoundFactory>,
    player: Option<Box<dyn SoundOut>>,
    /// There's sound coming (a track, or the session's sound output).
    has: bool,
    delay_s: Option<f64>,
    replay: VecDeque<Sound>,
    /// A file's sound as (time, pcm, rate, channels), held until the picture gets to it.
    wait: VecDeque<(f64, Vec<i16>, u32, u32)>,
    report_t: f64,
    seen: Option<SoundStats>,
}

pub(crate) struct Pipe {
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub bt709: bool,
    pub full_range: bool,
    /// The card's own format, for messages ("nv12", "mjpeg").
    pub fmt: String,
    /// A card (timed by its own clock) rather than something standing in (timed here).
    pub live: bool,
    pub q: Arc<TickQueue>,
    pub test: Option<SharedTestMode>,
    pub give_up_s: f64,
    frames: Mutex<Frames>,
    sound: Mutex<SoundState>,
    /// How late each stream arrives after its own timestamps, as (picture, sound).
    late: Mutex<(VecDeque<f64>, VecDeque<f64>)>,
    pub dropped: AtomicU64,
    pub damaged: AtomicU64,
    /// Frames macOS dropped before they were handed over (late, or out of buffers).
    pub os_dropped: AtomicU64,
    pub stop: AtomicBool,
    go: Mutex<bool>,
    go_cv: Condvar,
    messages: Mutex<Vec<String>>,
}

pub(crate) struct PipeConfig {
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub bt709: bool,
    pub full_range: bool,
    pub fmt: String,
    pub live: bool,
    pub queue_size: usize,
    pub test: Option<SharedTestMode>,
    pub give_up_s: f64,
    pub sound_out: Option<SoundFactory>,
    pub has_sound: bool,
}

impl Pipe {
    pub fn new(c: PipeConfig) -> sp_core::Result<Pipe> {
        Ok(Pipe {
            w: c.w,
            h: c.h,
            fps: c.fps,
            bt709: c.bt709,
            full_range: c.full_range,
            fmt: c.fmt,
            live: c.live,
            q: Arc::new(TickQueue::new(c.queue_size.max(1))),
            test: c.test,
            give_up_s: c.give_up_s,
            frames: Mutex::new(Frames { conv: Converter::new()?, n: 0, damaged_since: None }),
            sound: Mutex::new(SoundState {
                has: c.has_sound && c.sound_out.is_some(),
                out: c.sound_out,
                player: None,
                delay_s: None,
                replay: VecDeque::new(),
                wait: VecDeque::new(),
                report_t: 0.0,
                seen: None,
            }),
            late: Mutex::new((VecDeque::new(), VecDeque::new())),
            dropped: AtomicU64::new(0),
            damaged: AtomicU64::new(0),
            os_dropped: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            go: Mutex::new(false),
            go_cv: Condvar::new(),
            messages: Mutex::new(Vec::new()),
        })
    }

    /// A line for the log. It's printed, and kept for tests and the launcher's diagnostics.
    pub fn say(&self, line: String) {
        println!("{line}");
        self.messages.lock().unwrap().push(line);
    }

    pub fn messages(&self) -> Vec<String> {
        self.messages.lock().unwrap().clone()
    }

    /// The pipeline started reading.
    pub fn start(&self) {
        *self.go.lock().unwrap() = true;
        self.go_cv.notify_all();
    }

    /// Something standing in waits for the pipeline before it plays (it's timed from then).
    pub fn wait_go(&self) {
        let mut go = self.go.lock().unwrap();
        while !*go && !self.stop.load(Ordering::SeqCst) {
            go = self.go_cv.wait_timeout(go, std::time::Duration::from_millis(50)).unwrap().0;
        }
    }

    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// No more frames. What's queued goes out, then the end.
    pub fn end(&self) {
        self.q.end();
    }

    /// Capture stops, saying why (the card went away, the file ended badly, nothing readable).
    pub fn stop_with(&self, why: &str) {
        if !self.stop.swap(true, Ordering::SeqCst) {
            self.say(format!("[capture] stopped: {}", clip(why, 200)));
        }
        self.q.end();
    }

    /// One frame as it came, with `make` reading it into a Frame. `make` gives Ok(None) for a short one, which is
    /// skipped and not numbered, and Err for one that can't be read, which is damaged, skipped and numbered. Returns
    /// Err when capture must stop.
    pub fn frame(
        &self,
        t_arrival: f64,
        t_src: Option<f64>,
        make: impl FnOnce(&mut Converter) -> Result<Option<Frame>, String>,
    ) -> Result<(), String> {
        if self.stopped() {
            return Ok(());
        }
        let mut st = self.frames.lock().unwrap();
        let frame = match make(&mut st.conv) {
            Ok(None) => return Ok(()),
            Err(why) => return self.damaged_frame(&mut st, &why),
            Ok(Some(f)) => f,
        };
        st.damaged_since = None;
        if let (true, Some(t)) = (self.live, t_src) {
            let mut late = self.late.lock().unwrap();
            late.0.push_back(t_arrival - t);
            if late.0.len() > LATE_KEPT {
                late.0.pop_front();
            }
        }
        let mut tick = Tick::new(frame, t_arrival, 0);
        tick.t_source = t_src;
        tick.t_ready = Some(now());
        // Numbered as they come off the card, so a frame dropped from the queue below (the Mac behind for a moment)
        // doesn't shift the count. Game frame rate 30 fps takes every other one as new (cadence.RhythmLock), and a
        // shift by one made it move its beat later, a visible hitch.
        tick.extra.insert("seq".into(), json!(st.n));
        st.n += 1;
        drop(st);
        let Some(tick) = test_mode::apply(&self.test, tick) else { return Ok(()) };
        if self.q.put(tick) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// A frame macOS reported it couldn't deliver because of a break in the card's stream, taken as a damaged one.
    pub fn os_damaged(&self, why: &str) -> Result<(), String> {
        let mut st = self.frames.lock().unwrap();
        self.damaged_frame(&mut st, why)
    }

    /// A frame macOS dropped before handing it over because SmoovPlay was late taking the one before. It's numbered
    /// all the same.
    pub fn os_dropped(&self) {
        self.frames.lock().unwrap().n += 1;
        self.os_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// capture.py _damaged. A frame the card sent that can't be read is skipped and counted. Cheap cards send the odd
    /// broken JPEG, often the first one.
    fn damaged_frame(&self, st: &mut Frames, why: &str) -> Result<(), String> {
        let t = now();
        let n = self.damaged.fetch_add(1, Ordering::Relaxed) + 1;
        let fmt = self.fmt.to_uppercase();
        if n == 1 {
            self.say(format!("[capture] the card sent a damaged {fmt} frame: skipped ({})", clip(why, 120)));
        }
        match st.damaged_since {
            None => st.damaged_since = Some(t),
            Some(since) if t - since >= self.give_up_s => {
                return Err(format!(
                    "none of the card's {fmt} frames could be read for {} s ({n} damaged)",
                    g(self.give_up_s)
                ));
            }
            _ => {}
        }
        st.n += 1; // numbered all the same, as a frame the card sent
        Ok(())
    }

    // ---- sound
    /// A card's sound packet as it arrives. `t_end_src` is its own time for its end on the session's clock, which tells
    /// how late it comes.
    pub fn sound_live(&self, pcm: Vec<i16>, rate: u32, channels: u32, t_end_src: Option<f64>, t_arrival: f64) {
        if pcm.is_empty() || self.stopped() {
            return;
        }
        if let Some(t) = t_end_src {
            let mut late = self.late.lock().unwrap();
            late.1.push_back(t_arrival - t);
            if late.1.len() > LATE_KEPT {
                late.1.pop_front();
            }
        }
        self.play(pcm, rate, channels);
    }

    /// A file's sound, held until the picture gets to `t` (its time in the file).
    pub fn hold_sound(&self, t: f64, pcm: Vec<i16>, rate: u32, channels: u32) {
        if !pcm.is_empty() {
            self.sound.lock().unwrap().wait.push_back((t, pcm, rate, channels));
        }
    }

    /// The held sound up to `until` (the picture's time, plus a frame) goes out.
    pub fn release_sound(&self, until: f64) {
        loop {
            let next = {
                let mut s = self.sound.lock().unwrap();
                match s.wait.front() {
                    Some(w) if w.0 <= until => s.wait.pop_front(),
                    _ => None,
                }
            };
            match next {
                Some((_, pcm, rate, ch)) => self.play(pcm, rate, ch),
                None => break,
            }
        }
    }

    fn play(&self, pcm: Vec<i16>, rate: u32, channels: u32) {
        // Timed like the picture it goes with, since a card's sound comes this much later than its picture.
        let t = now() - self.sound_late_s();
        let mut lines: Vec<String> = Vec::new();
        {
            let mut guard = self.sound.lock().unwrap();
            let s = &mut *guard;
            if s.replay.len() >= REPLAY_KEPT {
                s.replay.pop_front();
            }
            if let (None, Some(out)) = (&s.player, s.out.as_mut()) {
                match out(rate, channels) {
                    Ok(mut p) => {
                        if let Some(d) = s.delay_s {
                            p.set_delay(d);
                        }
                        s.player = Some(p);
                    }
                    Err(e) => {
                        lines.push(format!("[capture] can't play the sound ({}); playing without it", clip(&e, 120)));
                        (s.out, s.has) = (None, false);
                    }
                }
            }
            if let Some(p) = s.player.as_mut() {
                p.push(&pcm);
                let t = now();
                if t >= s.report_t {
                    s.report_t = t + 60.0;
                    lines.extend(Self::report_line(s, false));
                }
            }
            s.replay.push_back(Sound { t, pcm, rate, channels });
        }
        for l in lines {
            self.say(l);
        }
    }

    /// capture.py _report_sound. Says how the sound played, once a minute when something went wrong, and for the
    /// whole session at the end.
    fn report_line(s: &mut SoundState, last: bool) -> Option<String> {
        let now = s.player.as_ref()?.stats()?;
        let since = if last { None } else { Some(s.seen.clone().unwrap_or_default()) };
        let line = sound_report(&now, since.as_ref());
        if !last {
            s.seen = Some(now);
        }
        line.map(|l| format!("[capture] sound{}: {l}", if last { " this session" } else { ", last minute" }))
    }

    /// The once-a-minute report, now, for tests.
    pub fn report_sound(&self, last: bool) {
        let line = Self::report_line(&mut self.sound.lock().unwrap(), last);
        if let Some(l) = line {
            self.say(l);
        }
    }

    /// How much later the sound arrives than the picture, going by their own timestamps (0 if not known).
    pub fn sound_late_s(&self) -> f64 {
        let late = self.late.lock().unwrap();
        if late.0.len() < 10 || late.1.len() < 10 {
            return 0.0;
        }
        median(&late.1) - median(&late.0)
    }

    /// No sound after all because its device wouldn't open, so the picture plays alone.
    pub fn sound_off(&self) {
        let mut s = self.sound.lock().unwrap();
        (s.has, s.out) = (false, None);
    }

    pub fn can_delay_audio(&self) -> bool {
        let s = self.sound.lock().unwrap();
        s.has && s.out.is_some()
    }

    /// The pipeline's delay. The sound is held back this much less however much later it already comes.
    pub fn set_audio_delay(&self, ms: f64) {
        let d = (ms / 1000.0 - self.sound_late_s()).max(0.0);
        let mut s = self.sound.lock().unwrap();
        s.delay_s = Some(d);
        if let Some(p) = s.player.as_mut() {
            p.set_delay(d);
        }
    }

    /// The sound played since the last call, for instant replay.
    pub fn read_sound(&self) -> Vec<Sound> {
        self.sound.lock().unwrap().replay.drain(..).collect()
    }

    /// Stops the sound and says how it went.
    pub fn close_sound(&self) {
        let mut s = self.sound.lock().unwrap();
        if let Some(p) = s.player.as_mut() {
            p.close();
            let line = Self::report_line(&mut s, true);
            drop(s);
            if let Some(l) = line {
                self.say(l);
            }
        }
    }
}

/// numpy's median.
fn median(v: &VecDeque<f64>) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

pub(crate) const NV12_VIDEO: u32 = fourcc(b"420v");
pub(crate) const NV12_FULL: u32 = fourcc(b"420f");
pub(crate) const BGRA: u32 = fourcc(b"BGRA");

/// A picture from AVFoundation (a card's, or a file's standing in) as a frame. NV12 is copied tightly into the
/// converter's buffer and turned into BGRA8 by capture.py raw_to_bgr's maths, with bilinear chroma and the matrix and
/// range given, as the Python reads a card's raw frames. BGRA is taken as it is.
pub(crate) fn convert_pixels(conv: &mut Converter, img: &CVPixelBuffer, bt709: bool, full_range: bool) -> Result<Option<Frame>, String> {
    let pf = CVPixelBufferGetPixelFormatType(img);
    let (w, h) = (CVPixelBufferGetWidth(img), CVPixelBufferGetHeight(img));
    if w == 0 || h == 0 {
        return Ok(None);
    }
    if unsafe { CVPixelBufferLockBaseAddress(img, CVPixelBufferLockFlags::ReadOnly) } != 0 {
        return Err("its picture couldn't be read (CVPixelBufferLockBaseAddress)".into());
    }
    let unlock = || unsafe {
        CVPixelBufferUnlockBaseAddress(img, CVPixelBufferLockFlags::ReadOnly);
    };
    let out = match pf {
        NV12_VIDEO | NV12_FULL => {
            let (cw, ch) = (w / 2, h / 2);
            let n = w * h + cw * 2 * ch;
            let y = CVPixelBufferGetBaseAddressOfPlane(img, 0) as *const u8;
            let uv = CVPixelBufferGetBaseAddressOfPlane(img, 1) as *const u8;
            if y.is_null() || uv.is_null() {
                unlock();
                return Err("its picture had no planes".into());
            }
            let (ys, uvs) = (CVPixelBufferGetBytesPerRowOfPlane(img, 0), CVPixelBufferGetBytesPerRowOfPlane(img, 1));
            let dst = conv.staging(n);
            unsafe {
                for r in 0..h {
                    std::ptr::copy_nonoverlapping(y.add(r * ys), dst.as_mut_ptr().add(r * w), w);
                }
                for r in 0..ch {
                    std::ptr::copy_nonoverlapping(uv.add(r * uvs), dst.as_mut_ptr().add(w * h + r * cw * 2), cw * 2);
                }
            }
            unlock();
            conv.nv12_staged(w, h, bt709, full_range)
        }
        BGRA => {
            let p = CVPixelBufferGetBaseAddress(img) as *const u8;
            if p.is_null() {
                unlock();
                return Err("its picture had no pixels".into());
            }
            let stride = CVPixelBufferGetBytesPerRow(img);
            let dst = conv.staging(w * h * 4);
            unsafe {
                for r in 0..h {
                    std::ptr::copy_nonoverlapping(p.add(r * stride), dst.as_mut_ptr().add(r * w * 4), w * 4);
                }
            }
            unlock();
            conv.bgra_staged(w, h)
        }
        other => {
            unlock();
            return Err(format!("a picture in {} that SmoovPlay can't read", fourcc_name(other)));
        }
    };
    Ok(Some(out))
}

/// Raw frame bytes as a card sends them (NV12, YUY2, UYVY or BGRA, tightly packed) as a frame, or None for a short
/// one (capture.py _convert_raw). 4:2:2 is taken to NV12 first, averaging each pair of lines' chroma, because
/// sp-media has no 4:2:2 kernel. That gives the same picture for flat colour, and a touch softer at colour edges
/// than capture.py's 4:2:2 path.
pub(crate) fn convert_raw(conv: &mut Converter, data: &[u8], fmt: &str, w: usize, h: usize, bt709: bool, full_range: bool) -> Result<Option<Frame>, String> {
    let bpp = crate::devices::bytes_per_pixel(fmt).ok_or_else(|| format!("can't convert {fmt}"))?;
    let n = (w as f64 * h as f64 * bpp) as usize;
    if data.len() < n {
        return Ok(None); // a short packet from a format change mid-stream, so skip it
    }
    match fmt {
        "nv12" => {
            conv.staging(n).copy_from_slice(&data[..n]);
            Ok(Some(conv.nv12_staged(w, h, bt709, full_range)))
        }
        "bgr0" => {
            conv.staging(n).copy_from_slice(&data[..n]);
            Ok(Some(conv.bgra_staged(w, h)))
        }
        "yuyv422" | "uyvy422" => {
            let (iy0, iu, iy1, iv) = if fmt == "yuyv422" { (0, 1, 2, 3) } else { (1, 0, 3, 2) };
            let (cw, ch) = (w / 2, h / 2);
            let dst = conv.staging(w * h + cw * 2 * ch);
            for r in 0..h {
                let row = &data[r * w * 2..(r + 1) * w * 2];
                for x in 0..cw {
                    let p = &row[x * 4..x * 4 + 4];
                    dst[r * w + 2 * x] = p[iy0];
                    dst[r * w + 2 * x + 1] = p[iy1];
                }
            }
            for r in 0..ch {
                let (a, b) = (&data[2 * r * w * 2..], &data[(2 * r + 1) * w * 2..]);
                for x in 0..cw {
                    let i = w * h + r * cw * 2 + x * 2;
                    dst[i] = (a[x * 4 + iu] as u16 + b[x * 4 + iu] as u16).div_ceil(2) as u8;
                    dst[i + 1] = (a[x * 4 + iv] as u16 + b[x * 4 + iv] as u16).div_ceil(2) as u8;
                }
            }
            Ok(Some(conv.nv12_staged(w, h, bt709, full_range)))
        }
        other => Err(format!("can't convert {other}")),
    }
}
