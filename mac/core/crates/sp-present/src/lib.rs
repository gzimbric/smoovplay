//! SmoovPlay's video window on macOS: framegen/io.py's GLWindowSink on Metal. It draws sp_core Frames as they are
//! (the decoder's and RIFE's textures, no CPU copies): FSR 1 (EASU + RCAS) or bilinear, the look, Fill the screen
//! (Fit, Zoom, Stretch, Panorama), the stats overlay, messages and the settings menu, into a CAMetalLayer in an
//! NSWindow, SDR (BGRA8, sRGB) or HDR10 (RGBA16F, PQ BT.2100 through EDR).
//!
//! # Opening it and driving it
//!
//! [`open`] makes the window (on the main thread: it takes a `MainThreadMarker`) and returns a [`VideoPresenter`],
//! which implements [`sp_core::Presenter`] and is `Send`. The window's AppKit side stays on the main thread; the
//! presenter draws from whichever thread calls it. Two ways to run it:
//!
//! - Pipeline on the main thread (as the Python window): call `write()` / `wait()` from the main thread. Each call
//!   handles the window's waiting events first (as glfw.poll_events), so keep calling `wait()` while no frames come
//!   (live sources yield a Waiting every 0.1 s), or use [`VideoPresenter::busy`] for long work.
//! - Pipeline on its own thread: move the presenter there, and run the main thread's event loop: [`run`] (until the
//!   presenter is closed), or the Swift app's own NSApplication run loop. Keys and the close button reach the
//!   presenter through a mutex; window changes (fullscreen, the layer's format and sync) go to the main thread
//!   through the main dispatch queue, which any of those run loops serves.
//!
//! The settings come in through [`MenuModel`] (pages and rows for the menu, get / set / step for hotkeys, and the
//! live [`DisplaySettings`]), which another crate adapts from the real settings; keys and values are
//! framegen/settings.py's. [`FixedSettings`] is a small stand-in.
//!
//! # Keys (io.py's)
//!
//! M or Tab: the settings menu (arrows and Enter in it, Esc or Backspace back a page). I: frame generation on/off
//! for this session. S: stats overlay off, minimal, detailed. F: fullscreen. U: next upscaler. Z: next fill.
//! K: look on/off. J: next look (Shift+J: the one before). L: delay test, R: save the instant replay, B: save a before
//! & after video (returned as [`sp_core::SinkReply`]). Q, Esc or Cmd+Q: quit. Moving the mouse shows the keys.
//!
//! # Presentation
//!
//! Fullscreen is a borderless window over the monitor the window is mostly on, with the menu bar and the Dock
//! hidden while SmoovPlay is in front. The first window is the frame's size, at most 1600x900 points and 90% of the
//! screen; drawables are at the screen's pixels (Retina). [`Pacing`] picks how frames are handed over; see its docs
//! and the `play` example for measurements.
//!
//! # Modules
//!
//! [`renderer`] draws a frame and the panels into any texture (the window's drawable, or an offscreen one for
//! checks and [`VideoPresenter::baked`]); [`looks`] and [`looks_gpu`] are the looks; [`fill`] the layout maths;
//! [`menu`], [`ui`] and [`text`] the menu and the overlays; [`window`] the AppKit side.

pub mod display;
pub mod fill;
pub mod gpu;
pub mod looks;
pub mod looks_gpu;
pub mod menu;
pub mod presenter;
pub mod renderer;
pub mod text;
pub mod ui;
pub mod window;

pub use display::{Corner, DisplaySettings, FixedSettings, MenuModel, MenuRow, OverlaySize, Pacing, Scaler};
pub use fill::Fill;
pub use presenter::{KEYS_HINT, PresentStats, Spread, Timing, VideoPresenter, WindowConfig, open, run};
pub use window::{Key, ScreenInfo, post_key, pump};
