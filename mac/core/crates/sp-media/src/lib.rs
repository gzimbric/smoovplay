//! Video in for SmoovPlay's macOS core: the frame tap's compressed frames decoded by VideoToolbox, the pictures
//! turned into Metal frames on the GPU, and the sources that yield them as ticks.
//!
//! - [`VtDecoder`]: H.264 / HEVC Annex-B access units, parameter sets kept and reused, IOSurface-backed NV12 or P010
//!   pictures out.
//! - [`Converter`]: those pictures as sp_core frames without copies (CVMetalTextureCache), with the Python path's
//!   colour maths: SDR as PyAV's `to_ndarray("bgr24")`, HDR as framegen/hdr.py ("sdr8", "pq16").
//! - Sources (sp_core::Source): [`ChiakiSource`] (PS5 Remote Play through chiaki-ng's tap), [`TapSource`] (another
//!   app's decoded frames; [`open_tap`] picks), [`FileSource`] and [`Ps5SimSource`] (video files), with
//!   [`TestMode`] ("Simulate a 30 fps game") and [`Clock60`] (a variable-rate source on a 60 Hz beat).

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

pub use chiaki::{ChiakiOptions, ChiakiSource, OpenError};
pub use clock60::{Clock60, QueuedSource};
pub use colour::{FrameInfo, Tags};
pub use convert::{Converter, HdrOut, HdrSettings};
pub use file::{FileSource, Ps5SimSource, SimOptions, VideoFile};
pub use tap_source::{open_tap, OpenedTap, TapSource};
pub use test_mode::{SharedTestMode, TestMode, WithTestMode};
pub use vt::{DecodeError, Decoded, VtDecoder};
