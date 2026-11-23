//! The before & after video's labels (replay.py _label). Each is white text on a translucent dark box, sized for the
//! picture's height and stored as (Y, weight) per pixel. Core Text draws the text in the system font, where the
//! Python uses OpenCV's Hershey font. The box is laid out the same way, with 14 px of padding at 1080p and even sizes.

use objc2_core_foundation::{CFAttributedString, CFDictionary, CFRetained, CFString, CFType, CGFloat};
use objc2_core_graphics::{CGBitmapContextCreate, CGColor, CGColorSpace, CGContext, CGImageAlphaInfo, CGImageByteOrderInfo, kCGColorSpaceSRGB};
use objc2_core_text::{CTFont, CTFontUIFontType, CTLine, kCTFontAttributeName, kCTForegroundColorAttributeName};

/// A label of `w` x `h` (even) with (Y, weight) for each pixel. Y is what the box's colour would encode to in BT.709
/// limited range, and the weight is max(0.55, text coverage).
#[derive(Clone, Debug)]
pub struct Label {
    pub w: usize,
    pub h: usize,
    pub data: Vec<[f32; 2]>,
}

fn line(text: &str, font: &CTFont) -> CFRetained<CTLine> {
    unsafe {
        let white = CGColor::new_srgb(1.0, 1.0, 1.0, 1.0);
        let keys: [&CFString; 2] = [kCTFontAttributeName, kCTForegroundColorAttributeName];
        let f: &CFType = font.as_ref();
        let c: &CFType = white.as_ref();
        let attrs = CFDictionary::<CFString, CFType>::from_slices(&keys, &[f, c]);
        let s = CFString::from_str(text);
        let a = CFAttributedString::new(None, Some(&s), Some(attrs.as_opaque())).expect("attributed string");
        CTLine::with_attributed_string(&a)
    }
}

/// The label for `text` on a picture `h_frame` pixels high.
pub fn label(text: &str, h_frame: usize) -> Label {
    let s = (h_frame as f64 / 1080.0).max(0.5);
    let font = unsafe { CTFont::new_ui_font_for_language(CTFontUIFontType::EmphasizedSystem, 30.0 * s, None) }.expect("no system font");
    let (cap, descent) = unsafe { (font.cap_height(), font.descent()) };
    let l = line(text, &font);
    let tw = unsafe { l.typographic_bounds(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) };
    let pad = (14.0 * s).round();
    let (th, base) = (cap.ceil(), descent.ceil());
    let w = ((tw.ceil() + 2.0 * pad) as usize + 1) & !1;
    let h = ((th + base + 2.0 * pad) as usize + 1) & !1;
    let mut px = vec![0u8; w * h * 4];
    unsafe {
        let space = CGColorSpace::with_name(Some(kCGColorSpaceSRGB)).expect("sRGB");
        let info = CGImageAlphaInfo::PremultipliedFirst.0 | CGImageByteOrderInfo::Order32Little.0;
        let ctx = CGBitmapContextCreate(px.as_mut_ptr() as *mut _, w, h, 8, w * 4, Some(&space), info).expect("bitmap context");
        CGContext::set_rgb_fill_color(Some(&ctx), 0.0, 0.0, 0.0, 1.0);
        CGContext::fill_rect(
            Some(&ctx),
            objc2_core_foundation::CGRect::new(
                objc2_core_foundation::CGPoint::new(0.0, 0.0),
                objc2_core_foundation::CGSize::new(w as CGFloat, h as CGFloat),
            ),
        );
        CGContext::set_should_antialias(Some(&ctx), true);
        // Core Graphics counts up from the bottom, and the baseline is pad + th from the top
        CGContext::set_text_position(Some(&ctx), pad, h as CGFloat - (pad + th));
        l.draw(&ctx);
        CGContext::flush(Some(&ctx));
    }
    let data = px
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            let g = p[1] as f32; // (B, G, R, A), and the text is white, so any of them will do
            let y = (g * 219.0 / 255.0 + 16.0).round_ties_even();
            [y, (g / 255.0).max(0.55)]
        })
        .collect();
    Label { w, h, data }
}
