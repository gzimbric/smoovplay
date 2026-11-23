//! SmoovPlay's video window on macOS. It's framegen/io.py's GLWindowSink, on Metal. It draws sp_core Frames as they
//! are, reading the decoder's and RIFE's textures with no CPU copies. It draws FSR 1 (EASU + RCAS) or bilinear, the
//! look, Fill the screen (Fit, Zoom, Stretch, Panorama), the stats overlay, messages and the settings menu into a
//! CAMetalLayer in an NSWindow. Output is SDR (BGRA8, sRGB) or HDR10 (RGBA16F, PQ BT.2100 through EDR).
//!
//! # Opening it and driving it
//!
//! [`open`] makes the window and returns a [`VideoPresenter`], which implements [`sp_core::Presenter`] and is `Send`.
//! Call it on the main thread, since it takes a `MainThreadMarker`. The window's AppKit side stays on the main
//! thread, and the presenter draws from whichever thread calls it. There are two ways to run it:
//!
//! - Pipeline on the main thread, as in the Python window. Call `write()` and `wait()` from the main thread. Each call
//!   first handles the window's waiting events, like glfw.poll_events, so keep calling `wait()` while no frames come.
//!   Live sources yield a Waiting every 0.1 s. For long work use [`VideoPresenter::busy`].
//! - Pipeline on its own thread. Move the presenter there and run the main thread's event loop, either with [`run`]
//!   (until the presenter is closed) or with the Swift app's own NSApplication run loop. Keys and the close button
//!   reach the presenter through a mutex. Window changes such as fullscreen and the layer's format and sync go to the
//!   main thread through the main dispatch queue, which either run loop serves.
//!
//! The settings come in through [`MenuModel`], which another crate adapts from the real settings. It gives the menu
//! its pages and rows, the hotkeys get, set and step, and the window the live [`DisplaySettings`]. Keys and values
//! are framegen/settings.py's. [`FixedSettings`] is a small stand-in.
//!
//! # Keys (io.py's)
//!
//! M or Tab opens the settings menu, where the arrows and Enter work and Esc or Backspace goes back a page. I turns
//! frame generation on or off for this session. S steps the stats overlay through off, minimal and detailed. F is
//! fullscreen, U the next upscaler and Z the next fill. K turns the look on or off, J picks the next look and Shift+J
//! the one before. L starts the delay test, R saves the instant replay and B saves a before & after video, all
//! returned as [`sp_core::SinkReply`]. Q, Esc or Cmd+Q quits. Moving the mouse shows the keys.
//!
//! # Presentation
//!
//! Fullscreen is a borderless window over the monitor the window is mostly on, with the menu bar and the Dock
//! hidden while SmoovPlay is in front. The first window is the frame's size, at most 1600x900 points and 90% of the
//! screen. Drawables are at the screen's pixels (Retina). [`Pacing`] picks how frames are handed over. Its docs and
//! the `play` example have the measurements.
//!
//! # Modules
//!
//! [`renderer`] draws a frame and the panels into any texture, either the window's drawable or an offscreen one for
//! checks and [`VideoPresenter::baked`]. [`looks`] and [`looks_gpu`] are the looks, and [`fill`] is the layout maths.
//! [`menu`], [`ui`] and [`text`] are the menu and the overlays. [`window`] is the AppKit side.

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
