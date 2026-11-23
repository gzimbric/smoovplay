//! SmoovPlay's scheduling heart for the macOS core, ported from the Python reference (framegen/cadence.py,
//! pipeline.py, delaytest.py, streamcheck.py and run.py's results report).
//!
//! - [`cadence`]: repeat and cut detection, the cadence scheduler, the 30 fps lock, Dip smoothing's watch and the
//!   even-timing timeline.
//! - [`pipeline`]: the tick loop at 60, 90 and 120 Hz, generic over sp_core's Source, Interpolator and Presenter,
//!   with an injectable [`clock`] and the [`settings`] it follows live.
//! - [`stats`]: what a session measured, summary() for the log and results() for test mode's report.
//! - [`probe`] and [`score`]: the pixels the pipeline looks at on the GPU (block means, the delay test's blocks,
//!   --score's sums).
//! - [`delaytest`], [`streamcheck`], [`report`].
//!
//! tests/parity.rs checks parity with the Python against traces recorded by tools/sched_traces.py.

pub mod cadence;
pub mod clock;
pub mod delaytest;
pub mod np;
pub mod pipeline;
pub mod probe;
pub mod report;
pub mod score;
pub mod settings;
pub mod stats;
pub mod streamcheck;

pub use cadence::{
    Action, CadenceScheduler, CutDetector, CutFeatures, DipWatch, DuplicateDetector, FrameDiff, FrameTimeline, Place,
    Reference, RhythmLock, TimedFrame, make_detector,
};
pub use clock::{Clock, FakeClock, SystemClock};
pub use pipeline::{Pipeline, Remote, Replay, RunOptions, STEADY_HUD_THRESHOLD, Status, Tuning};
pub use probe::{CpuProbe, GpuProbe, Probe};
pub use settings::{Live, LiveSettings, OverlayStyle, SharedLive};
pub use stats::{Exit, GameRate, ShownRate, Stats, rate_advice};
