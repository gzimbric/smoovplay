//! Video in for SmoovPlay's macOS core. VideoToolbox decodes the frame tap's compressed frames, the GPU turns the
//! pictures into Metal frames, and the sources yield those as ticks.
//!
//! - [`VtDecoder`] decodes H.264 or HEVC Annex-B access units to IOSurface-backed NV12 or P010 pictures, keeping and
//!   reusing the parameter sets.
//! - [`Converter`] makes those pictures into sp_core frames without copies, through a CVMetalTextureCache. It uses
//!   the Python path's colour maths, SDR as PyAV's `to_ndarray("bgr24")` and HDR as framegen/hdr.py ("sdr8", "pq16").
//! - The sources are sp_core::Source. [`ChiakiSource`] is PS5 Remote Play through chiaki-ng's tap. [`TapSource`]
//!   takes another app's decoded frames, and [`open_tap`] picks between the two. [`FileSource`] and [`Ps5SimSource`]
//!   play video files. [`TestMode`] is "Simulate a 30 fps game", and [`Clock60`] puts a variable-rate source on a
//!   60 Hz beat.

pub mod cf;
pub mod chiaki;
pub mod clock60;
pub mod colour;
pub mod convert;
pub mod file;
pub mod params;
pub mod pyrandom;
pub mod queue;
pub mod tap_source;
pub mod test_mode;
pub mod vt;

pub use chiaki::{ChiakiOptions, ChiakiSource, OpenError, STALE_S};
pub use clock60::{Clock60, QueuedSource};
pub use colour::{FrameInfo, Tags};
pub use convert::{Converter, HdrOut, HdrSettings};
pub use file::{FileSource, Ps5SimSource, SimOptions, VideoFile};
pub use tap_source::{open_tap, OpenedTap, TapSource};
pub use test_mode::{SharedTestMode, TestMode, WithTestMode};
pub use vt::{DecodeError, Decoded, VtDecoder};
