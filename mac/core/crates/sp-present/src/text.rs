//! Drawing the panels' text and boxes on the CPU with Core Text and Core Graphics, into a premultiplied BGRA bitmap
//! at the screen's own pixels (so it's crisp on a Retina display), then uploaded once per change
//! ([`crate::renderer::Panel`]). Coordinates here are top-left based, in pixels, as the Python drawings' (OpenCV's).

use objc2_core_foundation::{CFAttributedString, CFDictionary, CFNumber, CFRetained, CFString, CFType, CGFloat, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColor, CGImageByteOrderInfo, CGColorSpace, CGContext, CGImageAlphaInfo, CGPath, kCGColorSpaceSRGB,
};
use objc2_core_text::{CTFont, CTFontUIFontType, CTLine, kCTFontAttributeName, kCTForegroundColorAttributeName};

/// An sRGB colour, 0..255 (the Python drawings' BGR tuples, turned around).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    /// From OpenCV's (B, G, R).
    pub const fn bgr(b: u8, g: u8, r: u8) -> Rgb {
        Rgb(r, g, b)
    }
    fn f(self) -> (CGFloat, CGFloat, CGFloat) {
        (self.0 as CGFloat / 255.0, self.1 as CGFloat / 255.0, self.2 as CGFloat / 255.0)
    }
}

/// A font: the system font (SF) at a size in pixels.
#[derive(Clone)]
pub struct Font {
    ct: CFRetained<CTFont>,
    pub size: f64,
}

/// Which system font.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    Regular,
    Bold,
    /// Digits all the same width, so numbers that change don't make the line jitter.
    Digits,
}

impl Font {
    pub fn new(face: Face, size: f64) -> Font {
        unsafe {
            let ui = if face == Face::Bold { CTFontUIFontType::EmphasizedSystem } else { CTFontUIFontType::System };
            let base = CTFont::new_ui_font_for_language(ui, size, None).expect("no system font");
            let ct = if face == Face::Digits {
                // kNumberSpacingType (6), kMonospacedNumbersSelector (0)
                let desc = base.font_descriptor().copy_with_feature(&CFNumber::new_i32(6), &CFNumber::new_i32(0));
                CTFont::with_font_descriptor(&desc, size, std::ptr::null())
            } else {
                base
            };
            Font { ct, size }
        }
    }

    /// Ascent and descent in pixels.
    pub fn metrics(&self) -> (f64, f64) {
        unsafe { (self.ct.ascent(), self.ct.descent()) }
    }

    /// The height of capital letters and digits, in pixels.
    pub fn cap_height(&self) -> f64 {
        unsafe { self.ct.cap_height() }
    }

    fn line(&self, text: &str, color: Rgb) -> CFRetained<CTLine> {
        unsafe {
            let (r, g, b) = color.f();
            let cg = CGColor::new_srgb(r, g, b, 1.0);
            let keys: [&CFString; 2] = [kCTFontAttributeName, kCTForegroundColorAttributeName];
            let font: &CFType = self.ct.as_ref();
            let col: &CFType = cg.as_ref();
            let attrs = CFDictionary::<CFString, CFType>::from_slices(&keys, &[font, col]);
            let s = CFString::from_str(text);
            let a = CFAttributedString::new(None, Some(&s), Some(attrs.as_opaque())).expect("attributed string");
            CTLine::with_attributed_string(&a)
        }
    }

    /// The width of `text` in pixels.
    pub fn width(&self, text: &str) -> f64 {
        if text.is_empty() {
            return 0.0;
        }
        unsafe { self.line(text, Rgb(0, 0, 0)).typographic_bounds(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) }
    }
}

/// A bitmap being drawn: premultiplied BGRA, rows top first.
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    ctx: CFRetained<CGContext>,
    data: Vec<u8>,
}

impl Canvas {
    /// A clear (transparent) canvas.
    pub fn new(w: usize, h: usize) -> Canvas {
        let (w, h) = (w.max(1), h.max(1));
        let mut data = vec![0u8; w * h * 4];
        let ctx = unsafe {
            let space = CGColorSpace::with_name(Some(kCGColorSpaceSRGB)).expect("sRGB");
            let info = CGImageAlphaInfo::PremultipliedFirst.0 | CGImageByteOrderInfo::Order32Little.0;
            CGBitmapContextCreate(data.as_mut_ptr() as *mut _, w, h, 8, w * 4, Some(&space), info).expect("bitmap context")
        };
        CGContext::set_should_antialias(Some(&ctx), true);
        Canvas { w, h, ctx, data }
    }

    fn y(&self, top: f64, height: f64) -> f64 {
        self.h as f64 - top - height
    }

    /// A filled rectangle (top-left x, y).
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, c: Rgb) {
        self.rect_alpha(x, y, w, h, c, 1.0);
    }

    pub fn rect_alpha(&mut self, x: f64, y: f64, w: f64, h: f64, c: Rgb, a: f64) {
        let (r, g, b) = c.f();
        CGContext::set_rgb_fill_color(Some(&self.ctx), r, g, b, a);
        CGContext::fill_rect(Some(&self.ctx), CGRect::new(CGPoint::new(x, self.y(y, h)), CGSize::new(w, h)));
    }

    /// A filled rectangle with rounded corners.
    pub fn round_rect(&mut self, x: f64, y: f64, w: f64, h: f64, radius: f64, c: Rgb, a: f64) {
        let (r, g, b) = c.f();
        let rad = radius.min(w / 2.0).min(h / 2.0).max(0.0);
        unsafe {
            let path = CGPath::with_rounded_rect(CGRect::new(CGPoint::new(x, self.y(y, h)), CGSize::new(w, h)), rad, rad,
                                                 std::ptr::null());
            CGContext::set_rgb_fill_color(Some(&self.ctx), r, g, b, a);
            CGContext::add_path(Some(&self.ctx), Some(&path));
            CGContext::fill_path(Some(&self.ctx));
        }
    }

    /// A filled circle around (cx, cy).
    pub fn circle(&mut self, cx: f64, cy: f64, radius: f64, c: Rgb) {
        self.round_rect(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius, radius, c, 1.0);
    }

    /// `text` with its baseline at `base`, starting at x; returns its width.
    pub fn text(&mut self, x: f64, base: f64, text: &str, font: &Font, c: Rgb) -> f64 {
        if text.is_empty() {
            return 0.0;
        }
        unsafe {
            let line = font.line(text, c);
            CGContext::set_text_position(Some(&self.ctx), x, self.h as f64 - base);
            line.draw(&self.ctx);
            line.typographic_bounds(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
        }
    }

    /// `text` ending at `right`; returns its width.
    pub fn text_right(&mut self, right: f64, base: f64, text: &str, font: &Font, c: Rgb) -> f64 {
        let w = font.width(text);
        self.text(right - w, base, text, font, c);
        w
    }

    /// The pixels: premultiplied BGRA, rows top first.
    pub fn pixels(&self) -> &[u8] {
        CGContext::flush(Some(&self.ctx));
        &self.data
    }
}

/// Words of `text` in lines no wider than `width` (menu.py _wrap).
pub fn wrap(text: &str, width: f64, font: &Font) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let trial = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
        if !line.is_empty() && font.width(&trial) > width {
            lines.push(std::mem::replace(&mut line, word.to_string()));
        } else {
            line = trial;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}
