//! The seams between the core's pieces, which mirror the Python pipeline's duck types. A source is an iterator of
//! ticks (framegen/io.py, chiaki_tap.py). An interpolator has RifeInterpolator's load, prepare and interpolate
//! (framegen/interpolator.py), and a presenter has GLWindowSink's write, wait and toast (framegen/io.py).

use crate::frame::Frame;
use crate::tick::{Event, SourceInfo, Sound, Waiting};

/// The stats overlay for a frame (framegen/io.py MiniText). It holds the plain text, which the Detailed style shows
/// and a text sink prints. For the Minimal style it also holds the coloured parts, how well the Mac keeps up, and
/// the test-mode line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overlay {
    pub text: String,
    /// (text, kind) with kind "main", "dim" or "warn".
    pub parts: Vec<(String, String)>,
    /// "good", "tight", "behind", "off" (frame generation off) or "" (not known yet).
    pub health: String,
    /// The test line, or "".
    pub second: String,
    /// Minimal (parts) or Detailed (text).
    pub minimal: bool,
}

/// Where frames come from: the frame tap, a file, a capture card, an app's window.
pub trait Source: Send {
    fn info(&self) -> &SourceInfo;
    /// The stream's size in pixels.
    fn size(&self) -> (usize, usize);
    /// The next tick, a Waiting while none comes (live sources yield one every [`crate::tick::WAIT_POLL_S`]), or
    /// None when the stream has ended.
    fn next_event(&mut self) -> Option<Event>;
    /// The real frame that test mode or a simulated PS5 left out at `content`, for scoring.
    fn ground_truth(&mut self, _content: i64) -> Option<Frame> {
        None
    }
    /// The game's sound since the last call (instant replay), if this source has it.
    fn read_sound(&mut self) -> Vec<Sound> {
        Vec::new()
    }
    /// chiaki-ng's requests the other way: a keyframe, a button press for the delay test, holding the sound back.
    fn request_keyframe(&mut self) {}
    fn set_audio_delay(&mut self, _ms: f64) {}
    /// Stop reading (and let the source's threads end).
    fn close(&mut self) {}
}

/// Draws in-between frames. Two ping-pong slots hold the last two real frames, prepared once each.
pub trait Interpolator: Send {
    /// "RIFE 4.22.lite", for the overlay and logs.
    fn label(&self) -> String;
    /// Stream size this interpolator was built for.
    fn size(&self) -> (usize, usize);
    /// Copy a real frame into the input. The GPU may read it later, so the frame must stay alive. It does, because
    /// frames are immutable and reference-counted.
    fn load(&mut self, frame: &Frame);
    /// Encode the loaded frame into slot 0 or 1.
    fn prepare(&mut self, slot: usize);
    /// The frame at fraction `t` from slot (1 - newest) to slot `newest`. The returned frame is the interpolator's
    /// output buffer. The next call overwrites it, so a caller that keeps it must copy it.
    fn interpolate(&mut self, newest: usize, t: f32) -> Frame;
    /// For Steady HUD, the largest real-frame difference (0..1) that counts as still, or 0 for off.
    fn set_steady(&mut self, _threshold: f32) {}
}

/// What a presenter tells the pipeline after a frame or a wait: carry on, or what the player asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SinkReply {
    Continue,
    Quit,
    Restart,
    SaveReplay,
    SaveCompare,
    DelayTest,
}

/// Shows frames. That's the video window, a file writer, or nothing for headless runs.
pub trait Presenter {
    /// Show `frame` with the stats overlay, or with none this frame if `overlay` is None. Returns once it's queued
    /// for the screen, which with vsync on is at the next refresh.
    fn write(&mut self, frame: &Frame, overlay: Option<&Overlay>) -> SinkReply;
    /// No frame is coming for a moment. Keep the window answering and say so on screen.
    fn wait(&mut self, waiting: &Waiting) -> SinkReply;
    /// A short message in the corner.
    fn toast(&mut self, _text: &str, _seconds: f64) {}
    fn close(&mut self) {}
}
