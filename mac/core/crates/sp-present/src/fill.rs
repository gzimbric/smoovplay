//! Filling a screen that isn't the picture's shape (framegen/fill.py): Fit, Zoom, Stretch and Panorama. Consoles send
//! 16:9 only, so on an ultrawide or a 16:10 screen these are ways of laying out the same 16:9 picture, never more of
//! the game.
//!
//! The arithmetic is framegen/fill.py's, GL style (y0 is the bottom edge), so tests can hold it to the same numbers;
//! [`Rect::from_gl`] turns a rectangle into Metal's top-left one. The Panorama pass itself is in the shaders
//! (`panorama` in shaders/present.metal).

use serde::{Deserialize, Serialize};

/// Settings "fill" (Fill the screen; key Z).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fill {
    /// All of it, true to shape, with black bars. The default.
    #[default]
    Fit,
    /// Fills the screen by cutting off the top and bottom (or the sides).
    Zoom,
    /// Fills the screen by making everything wider (or taller).
    Stretch,
    /// Screens wider than the picture: the middle close to true shape, the sides take the stretch.
    Panorama,
}

impl Fill {
    pub const ALL: [Fill; 4] = [Fill::Fit, Fill::Zoom, Fill::Stretch, Fill::Panorama];

    /// The setting's value ("fit", "zoom", ...).
    pub fn key(self) -> &'static str {
        match self {
            Fill::Fit => "fit",
            Fill::Zoom => "zoom",
            Fill::Stretch => "stretch",
            Fill::Panorama => "panorama",
        }
    }

    /// The name shown ("Fit", ...).
    pub fn name(self) -> &'static str {
        match self {
            Fill::Fit => "Fit",
            Fill::Zoom => "Zoom",
            Fill::Stretch => "Stretch",
            Fill::Panorama => "Panorama",
        }
    }

    pub fn from_key(key: &str) -> Option<Fill> {
        Fill::ALL.into_iter().find(|f| f.key() == key)
    }
}

/// Panorama: the middle is drawn this share of the extra width wider than true (fill.py PANORAMA_MIDDLE).
pub const PANORAMA_MIDDLE: f64 = 0.15;

/// Whether the screen is wider than the picture (by more than a pixel or so).
pub fn wider(fw: i64, fh: i64, w: i64, h: i64) -> bool {
    fw * h > (fh + 1) * w
}

/// Python's round(): halves go to the even neighbour.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The picture's rectangle (dw, dh, x0, y0) on an fw x fh screen, GL style (x0, y0 the bottom left). Zoom's goes past
/// the screen's edges. Panorama's is Fit's: that's drawn first, then bent across the screen.
pub fn layout(fill: Fill, fw: i64, fh: i64, w: i64, h: i64) -> (i64, i64, i64, i64) {
    if fill == Fill::Stretch {
        return (fw, fh, 0, 0);
    }
    let (sx, sy) = (fw as f64 / w as f64, fh as f64 / h as f64);
    let s = if fill == Fill::Zoom { sx.max(sy) } else { sx.min(sy) };
    let (mut dw, mut dh) = (py_round(w as f64 * s), py_round(h as f64 * s));
    if fill != Fill::Zoom {
        dw = dw.min(fw);
        dh = dh.min(fh);
    }
    (dw, dh, (fw - dw).div_euclid(2), (fh - dh).div_euclid(2))
}

/// The part of the picture's rectangle that's on the screen (x, y, w, h), GL style: where the menu and messages go.
pub fn visible(dw: i64, dh: i64, x0: i64, y0: i64, fw: i64, fh: i64) -> (i64, i64, i64, i64) {
    let (x, y) = (x0.max(0), y0.max(0));
    (x, y, fw.min(x0 + dw) - x, fh.min(y0 + dh) - y)
}

/// Panorama's curve: u = atan(t*x) / atan(t) maps the screen (x from -1 to 1) onto the picture (u from -1 to 1). t is
/// picked so the middle is 1 + PANORAMA_MIDDLE*(k - 1) times wider than true (k: how many times wider the screen is);
/// 0 (no bending) when the screen isn't wider.
pub fn panorama_t(fw: i64, fh: i64, w: i64, h: i64) -> f64 {
    let k = (fw as f64 / fh as f64) / (w as f64 / h as f64);
    if k <= 1.0 {
        return 0.0;
    }
    let goal = (1.0 + PANORAMA_MIDDLE * (k - 1.0)) / k;
    let (mut lo, mut hi) = (1e-6f64, 1e3f64);
    for _ in 0..80 {
        let mid = (lo + hi) / 2.0;
        if mid.atan() / mid > goal {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Where on the picture (-1 to 1) a point across the screen (-1 to 1) comes from.
pub fn panorama_u(x: f64, t: f64) -> f64 {
    if t <= 0.0 { x } else { (t * x).atan() / t.atan() }
}

/// A rectangle in pixels, Metal style: (x, y) is the top-left corner, y grows downwards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
}

impl Rect {
    pub fn new(x: i64, y: i64, w: i64, h: i64) -> Rect {
        Rect { x, y, w, h }
    }

    /// A GL-style rectangle (x0, y0 at its bottom left, on a screen fh tall) as a top-left one.
    pub fn from_gl(dw: i64, dh: i64, x0: i64, y0: i64, fh: i64) -> Rect {
        Rect { x: x0, y: fh - y0 - dh, w: dw, h: dh }
    }

    /// The part of this rectangle inside (0, 0, fw, fh), or None if nothing is.
    pub fn clip(&self, fw: i64, fh: i64) -> Option<Rect> {
        let (x, y) = (self.x.max(0), self.y.max(0));
        let (r, b) = ((self.x + self.w).min(fw), (self.y + self.h).min(fh));
        (r > x && b > y).then(|| Rect { x, y, w: r - x, h: b - y })
    }
}

#[cfg(test)]
mod tests {
    // tests/test_fill.py, the same numbers.
    use super::*;
    const W: i64 = 1920;
    const H: i64 = 1080;

    #[test]
    fn fit_keeps_all_of_the_picture_with_bars() {
        assert_eq!(layout(Fill::Fit, 3440, 1440, W, H), (2560, 1440, 440, 0));
        assert_eq!(layout(Fill::Fit, 1920, 1200, W, H), (1920, 1080, 0, 60));
        assert_eq!(layout(Fill::Fit, 2560, 1440, W, H), (2560, 1440, 0, 0));
        assert_eq!(layout(Fill::Panorama, 3440, 1440, W, H), layout(Fill::Fit, 3440, 1440, W, H));
    }

    #[test]
    fn zoom_fills_by_cropping_and_stretch_by_widening() {
        let (dw, dh, x0, y0) = layout(Fill::Zoom, 3440, 1440, W, H);
        assert!((dw, x0) == (3440, 0) && (dw as f64 / dh as f64 - W as f64 / H as f64).abs() < 0.002);
        assert!(y0 < 0 && y0 + dh > 1440);
        let cut = 1.0 - 1440.0 / dh as f64;
        assert!(0.24 < cut && cut < 0.27);
        let (_, dh, x0, y0) = layout(Fill::Zoom, 1920, 1200, W, H);
        assert!(dh == 1200 && y0 == 0 && x0 < 0);
        assert_eq!(layout(Fill::Stretch, 3440, 1440, W, H), (3440, 1440, 0, 0));
        for f in Fill::ALL {
            assert_eq!(layout(f, 2560, 1440, W, H), (2560, 1440, 0, 0), "{f:?}");
        }
    }

    #[test]
    fn messages_go_on_the_part_of_the_picture_on_screen() {
        let (dw, dh, x0, y0) = layout(Fill::Zoom, 3440, 1440, W, H);
        assert_eq!(visible(dw, dh, x0, y0, 3440, 1440), (0, 0, 3440, 1440));
        let (dw, dh, x0, y0) = layout(Fill::Fit, 3440, 1440, W, H);
        assert_eq!(visible(dw, dh, x0, y0, 3440, 1440), (440, 0, 2560, 1440));
    }

    #[test]
    fn panorama_keeps_the_middle_close_to_true_and_never_folds() {
        assert!(!wider(2560, 1440, W, H) && !wider(1920, 1200, W, H) && wider(3440, 1440, W, H));
        assert_eq!(panorama_t(2560, 1440, W, H), 0.0);
        assert_eq!(panorama_u(0.3, 0.0), 0.3);
        for (fw, fh, middle, edge) in [(3440, 1440, 1.06, 2.5), (2560, 1080, 1.06, 2.5), (5120, 1440, 1.16, 5.5)] {
            let t = panorama_t(fw, fh, W, H);
            let k = (fw as f64 / fh as f64) / (W as f64 / H as f64);
            let xs: Vec<f64> = (0..2001).map(|i| -1.0 + 2.0 * i as f64 / 2000.0).collect();
            let us: Vec<f64> = xs.iter().map(|&x| panorama_u(x, t)).collect();
            assert!((us[0] + 1.0).abs() < 1e-9 && (us[2000] - 1.0).abs() < 1e-9 && us[1000].abs() < 1e-9);
            let widen: Vec<f64> = us.windows(2).map(|p| k * (xs[1] - xs[0]) / (p[1] - p[0])).collect();
            assert!(widen.iter().all(|&v| v > 0.0));
            assert!(1.0 < widen[1000] && widen[1000] < middle && widen[0] < edge, "{fw}x{fh}");
            assert!(widen[1000] < widen[750] && widen[750] < widen[500] && widen[500] < widen[0]);
            assert!((widen[1000] - k * t.atan() / t).abs() < 0.01);
        }
    }

    #[test]
    fn rects() {
        assert_eq!(Rect::from_gl(1920, 1080, 0, 60, 1200), Rect::new(0, 60, 1920, 1080));
        assert_eq!(Rect::from_gl(100, 50, 10, 0, 100), Rect::new(10, 50, 100, 50));
        assert_eq!(Rect::new(-10, -20, 100, 100).clip(50, 50), Some(Rect::new(0, 0, 50, 50)));
        assert_eq!(Rect::new(60, 0, 10, 10).clip(50, 50), None);
    }
}
