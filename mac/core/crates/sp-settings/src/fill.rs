//! Filling a screen that isn't the picture's shape (framegen/fill.py; Settings: Fill the screen, key Z).
//!
//! Consoles only send 16:9, so on an ultrawide or a 16:10 screen there's no more of the game to show: these are
//! ways of laying out the same picture. fit: all of it, with bars. zoom: fills the screen by cutting off the top and
//! bottom (or the sides). stretch: fills it by making everything wider. panorama: the middle stays close to true
//! shape and the sides take the stretch (screens wider than the picture only), drawn as fit and then bent across the
//! screen by one more pass whose curve is [`panorama_u`].

pub const FILLS: [&str; 4] = ["fit", "zoom", "stretch", "panorama"];
/// Panorama: the middle is drawn this share of the extra width wider than true (5% on 21:9, 15% on 32:9).
pub const PANORAMA_MIDDLE: f64 = 0.15;

/// Whether the screen is wider than the picture (by more than a pixel or so).
pub fn wider(fw: i64, fh: i64, w: i64, h: i64) -> bool {
    fw * h > (fh + 1) * w
}

/// The picture's rectangle (dw, dh, x0, y0) on an fw x fh screen, GL style (x0, y0 the bottom left). Zoom's goes
/// past the screen's edges. Panorama's is fit's.
pub fn layout(fill: &str, fw: i64, fh: i64, w: i64, h: i64) -> (i64, i64, i64, i64) {
    if fill == "stretch" {
        return (fw, fh, 0, 0);
    }
    let (sx, sy) = (fw as f64 / w as f64, fh as f64 / h as f64);
    let s = if fill == "zoom" { sx.max(sy) } else { sx.min(sy) };
    let (mut dw, mut dh) = ((w as f64 * s).round_ties_even() as i64, (h as f64 * s).round_ties_even() as i64);
    if fill != "zoom" {
        dw = dw.min(fw);
        dh = dh.min(fh);
    }
    (dw, dh, (fw - dw).div_euclid(2), (fh - dh).div_euclid(2))
}

/// The part of the picture's rectangle that's on the screen (x, y, w, h): where the menu and messages go.
pub fn visible(dw: i64, dh: i64, x0: i64, y0: i64, fw: i64, fh: i64) -> (i64, i64, i64, i64) {
    let (x, y) = (x0.max(0), y0.max(0));
    (x, y, fw.min(x0 + dw) - x, fh.min(y0 + dh) - y)
}

/// Panorama's curve parameter: u = atan(t*x) / atan(t) maps the screen (x from -1 to 1) onto the picture (u from
/// -1 to 1), with t picked so the middle is 1 + PANORAMA_MIDDLE*(k - 1) times wider than true (k: how many times
/// wider the screen is). 0 (no bending) when the screen isn't wider.
pub fn panorama_t(fw: i64, fh: i64, w: i64, h: i64) -> f64 {
    let k = (fw as f64 / fh as f64) / (w as f64 / h as f64);
    if k <= 1.0 {
        return 0.0;
    }
    let goal = (1.0 + PANORAMA_MIDDLE * (k - 1.0)) / k;
    let (mut lo, mut hi): (f64, f64) = (1e-6, 1e3);
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
