//! Capture cards for SmoovPlay's macOS core (Experimental), ported from framegen/capture.py and framegen/audio.py.
//!
//! - [`CardSource`]: a card as an sp_core::Source, read through AVFoundation (an AVCaptureSession whose video data
//!   output hands over IOSurface-backed CVPixelBuffers in the card's own format where it's NV12, timed by the
//!   session's clock), turned into BGRA8 frames by sp-media's [`sp_media::Converter`] with capture.py raw_to_bgr's
//!   maths. Test mode, Waiting while no frames come, dropped and damaged counts, and the card's sound from the same
//!   session, played through [`Player`] and kept for instant replay ([`sp_core::Source::read_sound`]).
//! - Stand-ins, through the same per-frame path ([`pipe`]): [`CardSource::video_file`] (any video file AVFoundation
//!   reads, at its own rate, its sound held until the picture gets to it), [`CardSource::raw_file`] and
//!   [`CardSource::raw_frames`] (raw frames as a card sends them).
//! - [`devices`]: the Mac's capture devices and their modes, [`pick_mode`] and [`pick_audio`], and the JSON of
//!   `python -m framegen.capture --list --json` / `--formats NAME --json` ([`cli`]).
//! - [`audio`]: the DelayLine that holds the sound back to meet the picture, and [`sound_report`].
//! - [`open_capture`]: run.py's `--capture` (a card by name, unique ID or number, or a file), with its sound
//!   device picked.
//! - examples: `capture` (`--list [--json]`, `--formats NAME [--json]`) and `card_check` (reads a card or a file
//!   for a while and says how it went: frame rate, drops, damaged frames, lateness, sound).
//!
//! Checked without a card (tests/): pick_mode and pick_audio against the Python on 339 and 32 cases, the DelayLine
//! byte for byte against the Python's on seven traces, the stand-ins' frames against the Python's (PyAV's decode
//! read with raw_to_bgr, within a level), the sound's timing against the picture (within a frame), the outputs'
//! delegates with a file's sample buffers through Objective-C, and the Player on the Mac's output (silence).
//!
//! Only a real card can show (live testing is for later; nothing here opens a camera or microphone): what its
//! formats look like to AVFoundation (whether 420v/420f/yuvs/dmb1 are listed per size and rate, and which pick_mode
//! takes), that the session keeps the format and frame rate asked for, the session clock's timestamps (frames'
//! lateness, sound against picture), whether its sound comes as its own device or on the video device, drops and
//! breaks macOS reports, MJPEG decoded by macOS, the camera and microphone prompts, and unplugging it while playing.

pub mod audio;
pub mod cli;
pub mod devices;
pub mod file;
pub(crate) mod pcm;
pub mod pipe;
pub mod player;
pub mod session;
pub mod source;

pub use audio::{sound_report, DelayLine, PlayerCore, SoundStats};
pub use devices::{list_devices, list_modes, pick_audio, pick_mode, Device, Mode};
pub use pipe::{SoundFactory, SoundOut, DAMAGED_GIVE_UP_S};
pub use player::{player_factory, Player};
pub use source::{open_capture, CaptureArgs, CardOptions, CardSource, Colour};

/// A four-character code ('420v') as an OSType.
pub const fn fourcc(s: &[u8; 4]) -> u32 {
    ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | s[3] as u32
}

/// An OSType as its four characters, for messages.
pub fn fourcc_name(code: u32) -> String {
    let s: String = code.to_be_bytes().iter().map(|&c| if c.is_ascii_graphic() || c == b' ' { c as char } else { '?' }).collect();
    format!("'{s}'")
}
