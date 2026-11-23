//! Shared types for SmoovPlay's macOS core.
//!
//! Every other crate builds on these: one Metal device and queue ([`gpu`]), frames that live on the GPU
//! ([`Frame`]), what a source yields ([`Tick`], [`Waiting`], [`Event`]) and what it is ([`SourceInfo`]), the clock
//! the frame tap and Python's `time.perf_counter()` share ([`clock`]), the 16x16 block means the repeat and cut
//! detectors compare ([`stats`]), and the traits the pieces plug into ([`Source`], [`Interpolator`], [`Presenter`]).
//!
//! The Python pipeline (`framegen/`) is the reference: names and meanings follow it, so its docs and tests describe
//! these too.

pub mod clock;
pub mod frame;
pub mod gpu;
pub mod stats;
pub mod tick;
pub mod traits;

pub use frame::{Frame, PixelKind};
pub use gpu::Gpu;
pub use tick::{Event, SourceInfo, Sound, Tick, Waiting};
pub use traits::{Interpolator, Overlay, Presenter, SinkReply, Source};

/// What can go wrong in the core: a Metal call failed, a file or stream couldn't be read, and so on.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Metal: {0}")]
    Metal(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
