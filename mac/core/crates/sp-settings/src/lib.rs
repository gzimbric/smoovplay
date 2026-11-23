//! SmoovPlay's settings, looks, launcher control protocol and Auto-detect, ported from the Python pipeline
//! (framegen/settings.py, remote.py, looks.py, fill.py, autodetect.py, paths.py), which stays the reference and the
//! Windows version. Files, protocol lines and share codes are byte for byte what the Python reads and writes, so the
//! Swift app talks to the Rust core exactly as the Windows launcher talks to `run.py --control`.
//!
//! - [`settings`]: the registry ([`settings::specs`]), [`settings::Settings`] (load/save/override/watch/step) and
//!   [`settings::specs_json`] (what `python -m framegen.settings` prints).
//! - [`remote`]: the `[settings-state]` protocol ([`remote::Remote`]).
//! - [`looks`]: the look format, normalize(), built-ins, the library and LUTs, share codes, the CLI subcommands.
//! - [`fill`]: Fill the screen's arithmetic.
//! - [`autodetect`]: recommend(), detect()'s result, the card-timing file.
//! - [`json`]: JSON with Python's types and formatting, which the rest is built on.

pub mod autodetect;
pub mod fill;
#[rustfmt::skip]
mod generated;
pub mod json;
pub mod looks;
pub mod paths;
pub mod py;
#[rustfmt::skip]
mod pyuni;
pub mod remote;
pub mod settings;
pub mod zlib;

pub use json::{Map, Value};
pub use settings::{LookChoices, Settings, SettingsError, Spec};
