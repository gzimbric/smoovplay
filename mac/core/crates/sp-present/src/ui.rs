//! The panels drawn over the video, as images. They are the stats overlay in both styles (io.py _mini and _detail),
//! messages (io.py _pill) and the settings menu (menu.py Menu.render). Layout and colours match the Python window's,
//! but the text uses Core Text's system font at the screen's own pixels instead of OpenCV's Hershey font.

use crate::display::{MenuModel, MenuRow};
use crate::menu::{Menu, Row};
use crate::text::{Canvas, Face, Font, Rgb, wrap};

/// A drawn panel, premultiplied BGRA with rows top first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub bgra: Vec<u8>,
}

impl Image {
    fn of(c: Canvas) -> Image {
        Image { w: c.w, h: c.h, bgra: c.pixels().to_vec() }
    }
}

// OpenCV's Hershey Simplex at scale 1 has digits about 22 px tall. SF's cap height is about 0.7 of its size, so a
// Python font scale `s` is SF at about 31 * s px.
fn hershey(scale: f64) -> f64 {
    31.0 * scale
}

// io.py's colours (BGR there)
pub const BOX: Rgb = Rgb::bgr(30, 27, 25);
pub const HEALTH_GOOD: Rgb = Rgb::bgr(110, 200, 80);
pub const HEALTH_TIGHT: Rgb = Rgb::bgr(40, 180, 250);
pub const HEALTH_BEHIND: Rgb = Rgb::bgr(70, 70, 235);
pub const HEALTH_OFF: Rgb = Rgb::bgr(150, 150, 150);
pub const HEALTH_UNKNOWN: Rgb = Rgb::bgr(110, 110, 110);
pub const MAIN: Rgb = Rgb::bgr(245, 245, 245);
pub const DIM_TEXT: Rgb = Rgb::bgr(185, 185, 185);
pub const WARN: Rgb = Rgb::bgr(40, 180, 250);
// menu.py's
const BG: Rgb = Rgb::bgr(34, 30, 28);
const ROW_SEL: Rgb = Rgb::bgr(70, 58, 48);
const ACCENT: Rgb = Rgb::bgr(40, 170, 255);
const TEXT: Rgb = Rgb::bgr(238, 238, 238);
const MUTED: Rgb = Rgb::bgr(150, 150, 150);
const IMPACT: Rgb = Rgb::bgr(200, 200, 200);
const DIM: Rgb = Rgb::bgr(95, 95, 95);
const RULE: Rgb = Rgb::bgr(70, 66, 62);

fn health_color(h: &str) -> Rgb {
    match h {
        "good" => HEALTH_GOOD,
        "tight" => HEALTH_TIGHT,
        "behind" => HEALTH_BEHIND,
        "off" => HEALTH_OFF,
        _ => HEALTH_UNKNOWN,
    }
}

fn kind_color(k: &str) -> Rgb {
    match k {
        "main" => MAIN,
        "warn" => WARN,
        _ => DIM_TEXT,
    }
}

fn i(v: f64) -> f64 {
    v.trunc()
}

/// The minimal overlay. It's a small dark box with a dot and a few numbers, plus the test line under them while a
/// test runs. The dot is green when keeping up, amber when tight and red when falling behind.
pub fn mini(parts: &[(String, String)], health: &str, second: &str, ui: f64) -> Image {
    let font = Font::new(Face::Digits, hershey(0.52 * ui));
    let small = Font::new(Face::Digits, hershey(0.42 * ui));
    let (pad, gap, r) = (i(6.0 * ui).max(3.0), i(10.0 * ui), i(4.5 * ui).max(3.0));
    let th_main = font.cap_height().ceil();
    let widths: Vec<f64> = parts.iter().map(|(t, _)| font.width(t).ceil()).collect();
    let line_w = 2.0 * r + gap + widths.iter().sum::<f64>() + gap * (widths.len().saturating_sub(1)) as f64;
    let (second_w, second_h) = if second.is_empty() { (0.0, 0.0) } else { (small.width(second).ceil(), small.cap_height().ceil()) };
    let w = 2.0 * pad + line_w.max(second_w);
    let h = 2.0 * pad + th_main + if second.is_empty() { 0.0 } else { second_h + i(8.0 * ui) };
    let mut c = Canvas::new(w as usize, h as usize);
    c.round_rect(0.0, 0.0, w, h, i(3.0 * ui), BOX, 1.0);
    let base = pad + th_main;
    c.circle(pad + r, base - (th_main / 2.0).floor(), r, health_color(health));
    let mut x = pad + 2.0 * r + gap;
    for ((t, k), tw) in parts.iter().zip(&widths) {
        c.text(x, base, t, &font, kind_color(k));
        x += tw + gap;
    }
    if !second.is_empty() {
        c.text(pad, base + i(8.0 * ui) + second_h, second, &small, DIM_TEXT);
    }
    Image::of(c)
}

/// The detailed overlay. Its line, and a second one with the advice or a test line, go on a box like the minimal
/// one's. With `min_w` the box only grows, so it doesn't jitter as the numbers change width.
pub fn detail(text: &str, ui: f64, min_w: usize) -> Image {
    let font = Font::new(Face::Digits, hershey(0.52 * ui));
    let (pad, lead) = (i(6.0 * ui).max(3.0), i(8.0 * ui));
    let lines: Vec<&str> = text.split('\n').take(2).collect();
    let line_h = font.cap_height().ceil();
    let w = (min_w as f64).max(2.0 * pad + lines.iter().map(|l| font.width(l).ceil()).fold(0.0, f64::max));
    let h = 2.0 * pad + lines.len() as f64 * line_h + (lines.len() as f64 - 1.0) * lead;
    let mut c = Canvas::new(w as usize, h as usize);
    c.round_rect(0.0, 0.0, w, h, i(3.0 * ui), BOX, 1.0);
    for (k, line) in lines.iter().enumerate() {
        c.text(pad, pad + line_h + k as f64 * (line_h + lead), line, &font, if k == 0 { MAIN } else { DIM_TEXT });
    }
    Image::of(c)
}

/// A short message (one line, or a few split by newlines) on a dark box with an accent bar.
pub fn pill(text: &str, ui: f64) -> Image {
    let font = Font::new(Face::Regular, hershey(0.7 * ui));
    let lines: Vec<&str> = text.split('\n').collect();
    let tw = lines.iter().map(|l| font.width(l).ceil()).fold(0.0, f64::max);
    let (asc, desc) = font.metrics();
    let tht = font.cap_height().ceil();
    let (padx, pady, gap) = (i(18.0 * ui), i(14.0 * ui), i(12.0 * ui));
    let h = lines.len() as f64 * (tht + gap) - gap + 2.0 * pady + i(6.0 * ui) + (desc - (asc - tht)).max(0.0).ceil();
    let w = tw + 2.0 * padx;
    let mut c = Canvas::new(w as usize, h as usize);
    c.round_rect(0.0, 0.0, w, h, i(6.0 * ui), BG, 1.0);
    c.rect(0.0, i(6.0 * ui) / 2.0, i(5.0 * ui).max(2.0), h - i(6.0 * ui), ACCENT);
    for (k, l) in lines.iter().enumerate() {
        c.text(padx, pady + tht + k as f64 * (tht + gap), l, &font, TEXT);
    }
    Image::of(c)
}

/// The keys at the bottom of the menu, on the main page and on a section's page (menu.py KEYS).
pub const MENU_KEYS: [&str; 2] = [
    "Up/Down select   Left/Right change   Enter open   M close   Q quit",
    "Up/Down select   Left/Right change   Esc back   M close   Q quit",
];

/// Draws the menu (menu.py render). Its size stays the same whatever page and row are shown, so the rows stay put
/// while the player goes through it.
#[derive(Default)]
pub struct MenuDrawer {
    size_key: Option<String>,
    size: (usize, f64),
}

impl MenuDrawer {
    pub fn render(&mut self, menu: &mut Menu, model: &dyn MenuModel, ui: f64) -> Image {
        let px = |v: f64| (v * ui).round().max(1.0);
        let (w, pad, mut row_h, head_h) = (px(780.0), px(26.0), px(46.0), px(78.0));
        let font = Font::new(Face::Regular, hershey(0.66 * ui));
        let small = Font::new(Face::Regular, hershey(0.5 * ui));
        let tag = Font::new(Face::Regular, hershey(0.45 * ui));
        let title = Font::new(Face::Bold, hershey(1.0 * ui));
        let rows = menu.rows(model);
        menu.follow(&rows);
        let rule = px(1.5).max(1.0);

        let foot = |row: &Row| -> (Vec<String>, Vec<String>) {
            let mut impact = Vec::new();
            let text = match row {
                Row::Restart => "Restarts the pipeline with the new settings. The stream keeps running; takes a few seconds.".to_string(),
                Row::Section(s) => menu.section_help(model, s),
                Row::Setting(r) => {
                    if r.disabled.is_none() {
                        for (name, v) in [("Picture", &r.impact_picture), ("Speed", &r.impact_speed)] {
                            if !v.is_empty() {
                                impact.extend(wrap(&format!("{name}: {v}"), w - 2.0 * pad, &small).into_iter().take(2));
                            }
                        }
                    }
                    r.disabled.clone().unwrap_or_else(|| r.help.clone())
                }
            };
            (wrap(&text, w - 2.0 * pad, &small).into_iter().take(4).collect(), impact)
        };
        let foot_size = |f: &(Vec<String>, Vec<String>)| (f.0.len() + f.1.len()) as f64 * px(26.0) + if f.1.is_empty() { 0.0 } else { px(8.0) };

        let (help_lines, impact_lines) = rows.get(menu.index).map(foot).unwrap_or_default();
        let mut pages: Vec<(Option<String>, Vec<Row>)> = vec![(None, menu.page_rows(model, None))];
        for s in model.sections() {
            let r = menu.page_rows(model, Some(&s));
            pages.push((Some(s), r));
        }
        let key = format!("{ui:.3}|{:?}", pages.iter().map(|(p, rs)| (p.clone(), rs.iter().map(|r| match r {
            Row::Setting(m) => format!("{}:{}:{}", m.key, m.disabled.is_some(), m.help.len()),
            Row::Section(s) => s.clone(),
            Row::Restart => "restart".into(),
        }).collect::<Vec<_>>())).collect::<Vec<_>>());
        if self.size_key.as_ref() != Some(&key) {
            let n = pages.iter().map(|(_, rs)| rs.len()).max().unwrap_or(1).max(1);
            let f = pages.iter().flat_map(|(_, rs)| rs.iter()).map(|r| foot_size(&foot(r))).fold(0.0, f64::max);
            self.size = (n, f);
            self.size_key = Some(key);
        }
        let (n, foot_body) = self.size;
        let foot_h = px(18.0) + foot_body + px(46.0);
        if head_h + n as f64 * row_h + foot_h > px(1040.0) {
            row_h = px(30.0).max(((px(1040.0) - head_h - foot_h) / n as f64).floor());
        }
        let h = head_h + n as f64 * row_h + foot_h;
        let mut c = Canvas::new(w as usize, h as usize);
        c.round_rect(0.0, 0.0, w, h, px(10.0), BG, 0.97);

        match &menu.page {
            None => {
                c.text(pad, px(50.0), "Settings", &title, TEXT);
                c.text_right(w - pad, px(48.0), "M / Esc to close", &small, MUTED);
            }
            Some(page) => {
                let x = pad + c.text(pad, px(50.0), "Settings  >  ", &title, MUTED);
                c.text(x, px(50.0), page, &title, TEXT);
                c.text_right(w - pad, px(48.0), "Esc: back", &small, MUTED);
            }
        }
        c.rect(pad, head_h - px(10.0), w - 2.0 * pad, rule, RULE);

        for (k, row) in rows.iter().enumerate() {
            let y0 = head_h + k as f64 * row_h;
            let base = y0 + (row_h * 0.66).floor();
            let selected = k == menu.index;
            if selected {
                c.rect(0.0, y0, w, row_h - px(2.0), ROW_SEL);
                c.rect(0.0, y0, px(6.0), row_h - px(2.0), ACCENT);
            }
            match row {
                Row::Restart => {
                    let reasons = model.restart_reasons().join(", ");
                    c.text(pad, base, &format!("Restart now  (applies: {reasons})"), &font, ACCENT);
                }
                Row::Section(s) => {
                    if k > 0 && matches!(rows[k - 1], Row::Setting(_)) {
                        c.rect(pad, y0, w - 2.0 * pad, rule, RULE);
                    }
                    c.text(pad, base, s, &font, TEXT);
                    c.text_right(w - pad, base, ">", &font, if selected { ACCENT } else { MUTED });
                }
                Row::Setting(r) => draw_setting(&mut c, r, selected, &font, &tag, pad, base, w, px(1.0)),
            }
        }

        let mut y = head_h + n as f64 * row_h;
        c.rect(pad, y + px(4.0), w - 2.0 * pad, rule, RULE);
        y += px(18.0);
        for line in &help_lines {
            y += px(26.0);
            c.text(pad, y, line, &small, MUTED);
        }
        if !impact_lines.is_empty() {
            y += px(8.0);
        }
        for line in &impact_lines {
            y += px(26.0);
            c.text(pad, y, line, &small, IMPACT);
        }
        c.text(pad, h - px(16.0), MENU_KEYS[menu.page.is_some() as usize], &tag, DIM);
        Image::of(c)
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_setting(c: &mut Canvas, r: &MenuRow, selected: bool, font: &Font, tag: &Font, pad: f64, base: f64, w: f64, one: f64) {
    let off = r.disabled.is_some();
    c.text(pad, base, &r.label, font, if off { DIM } else { TEXT });
    let mut tag_x = pad + font.width(&r.label) + 12.0 * one;
    if r.experimental {
        tag_x += c.text(tag_x, base, "experimental", tag, if off { DIM } else { ACCENT }) + 10.0 * one;
    }
    if !r.live {
        c.text(tag_x, base, "restart", tag, if off { DIM } else { MUTED });
    }
    let value = if off { "not available".to_string() } else { r.value.clone() };
    let right = w - pad;
    if selected && !off {
        c.text_right(right, base, ">", font, ACCENT);
        let vw = c.text_right(right - 30.0 * one, base, &value, font, TEXT);
        c.text_right(right - 30.0 * one - vw - 14.0 * one, base, "<", font, ACCENT);
    } else {
        c.text_right(right - if off { 0.0 } else { 30.0 * one }, base, &value, font, if off { DIM } else { MUTED });
    }
}
