//! The window's live display settings ([`DisplaySettings`]) and what the window needs from the settings to run its
//! menu and hotkeys ([`MenuModel`]). Another crate adapts the real settings (sp-settings) to these; keys and values
//! follow framegen/settings.py's SPECS, so a setting's key here is its key there.

use std::sync::Arc;

use serde_json::Value;

use crate::fill::Fill;
use crate::looks::Look;

/// Settings "scaler" (key U). RTX VSR is NVIDIA's: on a Mac the adapter offers FSR 1 and bilinear only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Scaler {
    /// AMD FSR 1 (EASU + RCAS).
    #[default]
    Fsr,
    Bilinear,
}

impl Scaler {
    pub fn name(self) -> &'static str {
        match self {
            Scaler::Fsr => "FSR 1",
            Scaler::Bilinear => "bilinear",
        }
    }
    pub fn from_key(key: &str) -> Scaler {
        if key == "bilinear" { Scaler::Bilinear } else { Scaler::Fsr }
    }
}

/// Settings "overlay_corner".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Corner {
    #[default]
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    pub fn from_key(key: &str) -> Corner {
        match key {
            "top_right" => Corner::TopRight,
            "bottom_left" => Corner::BottomLeft,
            "bottom_right" => Corner::BottomRight,
            _ => Corner::TopLeft,
        }
    }
    pub fn right(self) -> bool {
        matches!(self, Corner::TopRight | Corner::BottomRight)
    }
    pub fn bottom(self) -> bool {
        matches!(self, Corner::BottomLeft | Corner::BottomRight)
    }
}

/// Settings "overlay_size": times the screen's own scale (its height / 1080), io.py OVERLAY_SIZES.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OverlaySize {
    #[default]
    Small,
    Medium,
    Large,
}

impl OverlaySize {
    pub fn from_key(key: &str) -> OverlaySize {
        match key {
            "medium" => OverlaySize::Medium,
            "large" => OverlaySize::Large,
            _ => OverlaySize::Small,
        }
    }
    pub fn factor(self) -> f64 {
        match self {
            OverlaySize::Small => 0.8,
            OverlaySize::Medium => 1.0,
            OverlaySize::Large => 1.5,
        }
    }
}

/// How frames are handed to the display (new on the Mac).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Pacing {
    /// Steady with vsync on, Immediate with it off (measured: see the crate's docs).
    #[default]
    Auto,
    /// Each frame at the first refresh after it's drawn (vsync on), or at once (vsync off).
    Immediate,
    /// `presentDrawable:afterMinimumDuration:` 1/hz: on a variable-refresh display each frame stays up for one
    /// output frame, so the display runs at the output rate. On a fixed-refresh display it rounds up to whole
    /// refreshes (120 fps on 160 Hz comes out at 80).
    MinDuration,
    /// `presentDrawable:atTime:` on a steady grid of 1/hz, a little ahead of now (one refresh plus the GPU's time):
    /// each frame goes to the first refresh at or after its slot, so on a fixed-rate display the cadence is the
    /// best it can be (60 fps on 160 Hz: 3, 3, 2 refreshes) instead of moving with the GPU's and the pipeline's
    /// jitter. Late frames re-anchor the grid. On a variable-refresh display each frame lands on its slot.
    Steady,
}

/// What the window draws with, live (framegen/settings.py's display rows). The adapter builds it from the settings;
/// the window reads it again whenever [`MenuModel::revision`] changes.
#[derive(Clone, Debug)]
pub struct DisplaySettings {
    /// "scaler"
    pub scaler: Scaler,
    /// "fsr_sharpness": stops below the maximum (0 = max, each +1 halves it).
    pub fsr_sharpness: f32,
    /// "fill"
    pub fill: Fill,
    /// "overlay": the stats overlay on or off (its style comes with each frame's Overlay).
    pub overlay: bool,
    /// "overlay_corner"
    pub overlay_corner: Corner,
    /// "overlay_size"
    pub overlay_size: OverlaySize,
    /// "fullscreen": a borderless window over the monitor, with the menu bar and the Dock hidden while in front.
    pub fullscreen: bool,
    /// "vsync"
    pub vsync: bool,
    /// The look to draw ("look_on" + "look", or the launcher's preview), None: no look. Drawn again only when this
    /// changes (by pointer or contents).
    pub look: Option<Arc<Look>>,
    /// "hz": the output frame rate, for pacing.
    pub hz: f64,
    pub pacing: Pacing,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        DisplaySettings {
            scaler: Scaler::Fsr,
            fsr_sharpness: 0.5,
            fill: Fill::Fit,
            overlay: true,
            overlay_corner: Corner::TopLeft,
            overlay_size: OverlaySize::Small,
            fullscreen: false,
            vsync: false,
            look: None,
            hz: 60.0,
            pacing: Pacing::Auto,
        }
    }
}

/// A row of the settings menu: one setting, as the window shows it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MenuRow {
    /// settings.py's key ("scaler").
    pub key: String,
    pub label: String,
    /// The current value's name ("AMD FSR 1").
    pub value: String,
    /// Applies at once (false: on the next restart, tagged "restart").
    pub live: bool,
    pub experimental: bool,
    /// Why it can't be changed now (greyed out, and shown as its help), or None.
    pub disabled: Option<String>,
    pub help: String,
    /// A rough line on what it does to the picture, and to speed or delay ("" if none).
    pub impact_picture: String,
    pub impact_speed: String,
}

/// What the window needs from the settings: the menu's pages and rows, changing a value, and the live display
/// settings. Keys and values are settings.py's (values as JSON: bools, numbers, strings). Called on the thread that
/// calls the presenter's write() and wait().
pub trait MenuModel: Send {
    /// The sections that have a setting that applies right now, in order (settings.py's groups: "Smoothness",
    /// "Picture", "Looks", "Display", ...).
    fn sections(&self) -> Vec<String>;
    /// The settings rows of a page: None, the main page (the quick picks); Some(section), that section's.
    fn rows(&self, page: Option<&str>) -> Vec<MenuRow>;
    /// Labels of settings changed since start that need a restart (the menu then offers "Restart now").
    fn restart_reasons(&self) -> Vec<String> {
        Vec::new()
    }
    /// A setting's value.
    fn get(&self, key: &str) -> Option<Value>;
    /// Sets a value, saved (`save`) or for this session only (the I key). False if it's not one of its choices.
    fn set(&mut self, key: &str, value: Value, save: bool) -> bool;
    /// The next (+1) or previous (-1) choice, wrapping, skipping ones that can't be used now (settings.py step()).
    fn step(&mut self, key: &str, delta: i32) -> bool;
    /// The window's settings as they are now.
    fn display(&self) -> DisplaySettings;
    /// Goes up whenever any setting changes (menu, hotkeys, the launcher), so the window knows to read display()
    /// and redraw the menu.
    fn revision(&self) -> u64;
    /// A row by key, from whichever page has it (for the messages hotkeys show).
    fn row(&self, key: &str) -> Option<MenuRow> {
        std::iter::once(None)
            .chain(self.sections().into_iter().map(Some))
            .flat_map(|p| self.rows(p.as_deref()))
            .find(|r| r.key == key)
    }
}

/// A MenuModel with only the display settings and no pages, for examples, tests and a window without settings.
#[derive(Clone, Debug, Default)]
pub struct FixedSettings {
    pub display: DisplaySettings,
    pub rev: u64,
    pub interp: bool,
}

impl MenuModel for FixedSettings {
    fn sections(&self) -> Vec<String> {
        vec!["Display".into()]
    }
    fn rows(&self, page: Option<&str>) -> Vec<MenuRow> {
        let d = &self.display;
        let row = |key: &str, label: &str, value: String, help: &str| MenuRow {
            key: key.into(),
            label: label.into(),
            value,
            live: true,
            help: help.into(),
            ..Default::default()
        };
        let on = |b: bool| if b { "On".to_string() } else { "Off".to_string() };
        match page {
            None => vec![
                row("interp", "Frame generation", on(self.interp), "Draws new frames between the ones the game sends."),
                row("scaler", "Upscaler", if d.scaler == Scaler::Fsr { "AMD FSR 1".into() } else { "Bilinear".into() },
                    "Enlarges the picture from the game to fill your window or screen. Key: U."),
            ],
            Some("Display") => vec![
                row("fullscreen", "Fullscreen", on(d.fullscreen), "Fills the whole screen. Key: F."),
                row("fill", "Fill the screen", d.fill.name().into(), "For an ultrawide screen. Key: Z."),
                row("vsync", "VSync", on(d.vsync), "Off: the lowest delay. On: no tearing."),
                row("overlay", "Stats overlay", on(d.overlay), "Numbers in a corner of the video. Key: S."),
            ],
            _ => Vec::new(),
        }
    }
    fn get(&self, key: &str) -> Option<Value> {
        let d = &self.display;
        Some(match key {
            "interp" => Value::Bool(self.interp),
            "scaler" => Value::from(if d.scaler == Scaler::Fsr { "fsr" } else { "bilinear" }),
            "fullscreen" => Value::Bool(d.fullscreen),
            "vsync" => Value::Bool(d.vsync),
            "overlay" => Value::Bool(d.overlay),
            "overlay_style" => Value::from("minimal"),
            "fill" => Value::from(d.fill.key()),
            "look_on" => Value::Bool(d.look.is_some()),
            _ => return None,
        })
    }
    fn set(&mut self, key: &str, value: Value, _save: bool) -> bool {
        let d = &mut self.display;
        let b = value.as_bool();
        match (key, b) {
            ("interp", Some(b)) => self.interp = b,
            ("fullscreen", Some(b)) => d.fullscreen = b,
            ("vsync", Some(b)) => d.vsync = b,
            ("overlay", Some(b)) => d.overlay = b,
            ("scaler", _) => d.scaler = Scaler::from_key(value.as_str().unwrap_or("fsr")),
            ("fill", _) => match value.as_str().and_then(Fill::from_key) {
                Some(f) => d.fill = f,
                None => return false,
            },
            _ => return false,
        }
        self.rev += 1;
        true
    }
    fn step(&mut self, key: &str, delta: i32) -> bool {
        let d = &mut self.display;
        match key {
            "scaler" => d.scaler = if d.scaler == Scaler::Fsr { Scaler::Bilinear } else { Scaler::Fsr },
            "fill" => {
                let i = Fill::ALL.iter().position(|f| *f == d.fill).unwrap_or(0) as i32;
                d.fill = Fill::ALL[(i + delta).rem_euclid(4) as usize];
            }
            "interp" => self.interp = !self.interp,
            "fullscreen" => d.fullscreen = !d.fullscreen,
            "vsync" => d.vsync = !d.vsync,
            "overlay" => d.overlay = !d.overlay,
            _ => return false,
        }
        self.rev += 1;
        true
    }
    fn display(&self) -> DisplaySettings {
        self.display.clone()
    }
    fn revision(&self) -> u64 {
        self.rev
    }
}
