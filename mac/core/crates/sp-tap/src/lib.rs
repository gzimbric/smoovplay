//! The frame tap is the shared memory that chiaki-ng (PS5 Remote Play) and other apps write their video frames and
//! sound into, for SmoovPlay to read. It's byte-compatible with the C writer inside chiaki-ng (chiaki_tap/frametap.c)
//! and the Python reader and writers (framegen/chiaki_tap.py). docs/frame-tap.md and chiaki_tap/frametap.h describe
//! the layout.
//!
//! - [`FrameTap`]: the header and the frame slots, both sides. It also carries the reader's requests to the writer:
//!   a keyframe, a button press for the delay test, holding the sound back, and the heartbeat.
//! - [`AudioTap`]: the sound ring, both sides.
//! - [`nal`]: Annex-B access units, with their NAL units, keyframes and codec header (parameter sets).
//! - [`SizeRepeatDetector`]: new frame or repeat from compressed size alone, for the log, the stream check and test
//!   mode.
//! - [`saved`]: the codec header saved for a later run that joins the same stream (the same file Python uses).
//!
//! On macOS the tap is POSIX shared memory, named "/" followed by whatever comes after the name's last backslash
//! ([`posix_name`]). A reader polls the header every [`POLL_S`] instead of waiting on an event. Times are
//! nanoseconds of `CLOCK_UPTIME_RAW`. Nothing here touches the GPU.

pub mod audio;
pub mod clock;
pub mod frames;
pub mod layout;
pub mod nal;
pub mod saved;
pub mod shm;
pub mod size;

pub use audio::{AudioTap, SoundChunk};
pub use frames::{FrameTap, Header, Packet, Watch};
pub use layout::*;
pub use nal::Codec;
pub use shm::{check_name, posix_name, unlink};
pub use size::SizeRepeatDetector;

/// What can go wrong opening or starting a tap.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The name can't be used as shared memory here (empty, has a "/", or too long with the sound ring's suffix).
    #[error("{0}")]
    Name(String),
    /// A decoded frame too big for the tap's slots (raw_layout).
    #[error("{0}")]
    Layout(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
