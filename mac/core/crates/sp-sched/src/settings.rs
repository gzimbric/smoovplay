//! The settings the pipeline follows live (framegen/settings.py's keys that pipeline.py reads). Another crate maps
//! the real settings onto [`LiveSettings`]. [`SharedLive`] is a plain one for tests and simple hosts.

use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayStyle {
    /// A few numbers and a dot (MiniText).
    Minimal,
    /// The full text line.
    Detailed,
}

/// One reading of the settings, taken at the start of each tick. The Python reads them as it goes, but a change made
/// in the menu in the middle of a tick is picked up at the next one either way.
#[derive(Clone, Debug, PartialEq)]
pub struct Live {
    /// "interp": frame generation on.
    pub interp: bool,
    /// "hz": output frame rate, 60, 90 or 120 (60 without a presenter).
    pub hz: u32,
    /// "simulate_30": test mode, a 60 fps game played as 30 fps and scored.
    pub simulate_30: bool,
    /// "steady_hud".
    pub steady_hud: bool,
    /// "overlay": the stats overlay is drawn. When off, the presenter gets None.
    pub overlay: bool,
    /// "overlay_style".
    pub overlay_style: OverlayStyle,
    /// "even_out": Even out 40-60 fps.
    pub even_out: bool,
    /// "even_dips": Dip smoothing.
    pub even_dips: bool,
    /// "game_fps" == "30": every other frame is new (RhythmLock).
    pub game_fps_30: bool,
    /// "audio_sync": hold chiaki-ng's or the card's sound back to meet the picture.
    pub audio_sync: bool,
}

impl Default for Live {
    /// framegen/settings.py's defaults.
    fn default() -> Self {
        Live {
            interp: true,
            hz: 60,
            simulate_30: false,
            steady_hud: false,
            overlay: true,
            overlay_style: OverlayStyle::Minimal,
            even_out: true,
            even_dips: false,
            game_fps_30: false,
            audio_sync: true,
        }
    }
}

pub trait LiveSettings: Send + Sync {
    fn live(&self) -> Live;
    /// settings.override("simulate_30", on), for this run only. The pipeline turns test mode off on a game that's at
    /// 30 already. The source's test mode follows the setting.
    fn set_simulate_30(&self, on: bool);
}

/// Settings in a mutex, for tests and for hosts without a settings store of their own.
#[derive(Debug, Default)]
pub struct SharedLive(Mutex<Live>);

impl SharedLive {
    pub fn new(live: Live) -> SharedLive {
        SharedLive(Mutex::new(live))
    }
    pub fn update(&self, f: impl FnOnce(&mut Live)) {
        f(&mut self.0.lock().unwrap());
    }
}

impl LiveSettings for SharedLive {
    fn live(&self) -> Live {
        self.0.lock().unwrap().clone()
    }
    fn set_simulate_30(&self, on: bool) {
        self.0.lock().unwrap().simulate_30 = on;
    }
}
