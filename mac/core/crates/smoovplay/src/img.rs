//! Picture files for --save-worst and the looks preview (framegen/imagefile.py read_image / write_image): any format
//! ImageIO reads, written by the extension (BMP here, PNG and JPEG through ImageIO; PNG without one). Pictures are
//! packed BGR, rows top first, as OpenCV's.

use std::ffi::{CString, c_char, c_void};
use std::path::Path;

type Ref = *const c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[link(name = "ImageIO", kind = "framework")]
unsafe extern "C" {
    fn CGImageSourceCreateWithURL(url: Ref, options: Ref) -> Ref;
    fn CGImageSourceCreateImageAtIndex(src: Ref, index: usize, options: Ref) -> Ref;
    fn CGImageDestinationCreateWithURL(url: Ref, kind: Ref, count: usize, options: Ref) -> Ref;
    fn CGImageDestinationAddImage(dest: Ref, image: Ref, properties: Ref);
    fn CGImageDestinationFinalize(dest: Ref) -> bool;
    static kCGImageDestinationLossyCompressionQuality: Ref;
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGImageGetWidth(img: Ref) -> usize;
    fn CGImageGetHeight(img: Ref) -> usize;
    fn CGColorSpaceCreateWithName(name: Ref) -> Ref;
    static kCGColorSpaceSRGB: Ref;
    fn CGBitmapContextCreate(data: *mut c_void, w: usize, h: usize, bpc: usize, bpr: usize, space: Ref, info: u32) -> Ref;
    fn CGBitmapContextCreateImage(ctx: Ref) -> Ref;
    fn CGContextDrawImage(ctx: Ref, rect: CGRect, img: Ref);
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFURLCreateFromFileSystemRepresentation(alloc: Ref, buf: *const u8, len: isize, is_dir: bool) -> Ref;
    fn CFStringCreateWithCString(alloc: Ref, s: *const c_char, encoding: u32) -> Ref;
    fn CFNumberCreate(alloc: Ref, kind: isize, value: *const c_void) -> Ref;
    fn CFDictionaryCreate(alloc: Ref, keys: *const Ref, values: *const Ref, n: isize, kcb: *const c_void, vcb: *const c_void) -> Ref;
    fn CFRelease(cf: Ref);
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

const ALPHA_SKIP_FIRST: u32 = 6; // kCGImageAlphaNoneSkipFirst
const ORDER_32_LITTLE: u32 = 2 << 12; // kCGBitmapByteOrder32Little: B, G, R, X in memory

/// A picture: packed BGR.
#[derive(Clone, Debug)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub bgr: Vec<u8>,
}

fn url(path: &Path) -> Ref {
    let b = path.as_os_str().as_encoded_bytes();
    unsafe { CFURLCreateFromFileSystemRepresentation(std::ptr::null(), b.as_ptr(), b.len() as isize, false) }
}

fn release(r: Ref) {
    if !r.is_null() {
        unsafe { CFRelease(r) }
    }
}

/// read_image(): None when the file is missing or isn't a picture.
pub fn read(path: &Path) -> Option<Image> {
    if !path.is_file() {
        return None;
    }
    unsafe {
        let u = url(path);
        let src = CGImageSourceCreateWithURL(u, std::ptr::null());
        release(u);
        if src.is_null() {
            return None;
        }
        let img = CGImageSourceCreateImageAtIndex(src, 0, std::ptr::null());
        release(src);
        if img.is_null() {
            return None;
        }
        let (w, h) = (CGImageGetWidth(img), CGImageGetHeight(img));
        let mut bgrx = vec![0u8; w * h * 4];
        let space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        let ctx = CGBitmapContextCreate(bgrx.as_mut_ptr() as *mut c_void, w, h, 8, w * 4, space, ALPHA_SKIP_FIRST | ORDER_32_LITTLE);
        if !ctx.is_null() {
            CGContextDrawImage(ctx, CGRect { x: 0.0, y: 0.0, w: w as f64, h: h as f64 }, img);
        }
        release(ctx);
        release(space);
        release(img);
        Some(Image { w, h, bgr: bgrx.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect() })
    }
}

fn bmp(img: &Image) -> Vec<u8> {
    let row = (img.w * 3).div_ceil(4) * 4;
    let size = 54 + row * img.h;
    let mut out = Vec::with_capacity(size);
    out.extend(b"BM");
    out.extend((size as u32).to_le_bytes());
    out.extend([0u8; 4]);
    out.extend(54u32.to_le_bytes());
    out.extend(40u32.to_le_bytes());
    out.extend((img.w as i32).to_le_bytes());
    out.extend((img.h as i32).to_le_bytes()); // bottom-up, as OpenCV writes it
    out.extend(1u16.to_le_bytes());
    out.extend(24u16.to_le_bytes());
    out.extend([0u8; 24]);
    for y in (0..img.h).rev() {
        let r = &img.bgr[y * img.w * 3..(y + 1) * img.w * 3];
        out.extend(r);
        out.extend(std::iter::repeat_n(0u8, row - r.len()));
    }
    out
}

/// write_image(): the format the extension names (PNG without one). False if it couldn't be written.
pub fn write(path: &Path, img: &Image) -> bool {
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if ext == "bmp" || ext == "dib" {
        return std::fs::write(path, bmp(img)).is_ok();
    }
    let uti = match ext.as_str() {
        "jpg" | "jpeg" | "jpe" => "public.jpeg",
        "tif" | "tiff" => "public.tiff",
        "" | "png" => "public.png",
        _ => return false,
    };
    let mut bgrx: Vec<u8> = img.bgr.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    unsafe {
        let space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        let ctx = CGBitmapContextCreate(bgrx.as_mut_ptr() as *mut c_void, img.w, img.h, 8, img.w * 4, space,
                                        ALPHA_SKIP_FIRST | ORDER_32_LITTLE);
        release(space);
        if ctx.is_null() {
            return false;
        }
        let cg = CGBitmapContextCreateImage(ctx);
        release(ctx);
        let c = CString::new(uti).unwrap();
        let kind = CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), 0x0800_0100);
        let u = url(path);
        let dest = CGImageDestinationCreateWithURL(u, kind, 1, std::ptr::null());
        release(u);
        release(kind);
        if dest.is_null() || cg.is_null() {
            release(dest);
            release(cg);
            return false;
        }
        let q = 0.95f64; // OpenCV's JPEG quality
        let num = CFNumberCreate(std::ptr::null(), 13, &q as *const f64 as *const c_void); // kCFNumberFloat64Type
        let keys = [kCGImageDestinationLossyCompressionQuality];
        let vals = [num];
        let props = CFDictionaryCreate(std::ptr::null(), keys.as_ptr(), vals.as_ptr(), 1,
                                       &kCFTypeDictionaryKeyCallBacks as *const c_void,
                                       &kCFTypeDictionaryValueCallBacks as *const c_void);
        CGImageDestinationAddImage(dest, cg, props);
        let ok = CGImageDestinationFinalize(dest);
        release(props);
        release(num);
        release(dest);
        release(cg);
        ok
    }
}

/// Each output pixel's (first source pixel, weights) along one axis: the source span it covers, as OpenCV's
/// INTER_AREA when shrinking (bilinear-like when enlarging).
fn area_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|i| {
            if scale <= 1.0 {
                // enlarging: linear between the two nearest
                let f = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
                let x0 = (f.floor() as usize).min(src - 1);
                let x1 = (x0 + 1).min(src - 1);
                let t = (f - x0 as f64) as f32;
                if x1 == x0 { (x0, vec![1.0]) } else { (x0, vec![1.0 - t, t]) }
            } else {
                let (a, b) = (i as f64 * scale, (i as f64 + 1.0) * scale);
                let first = a.floor() as usize;
                let last = (b.ceil() as usize).min(src);
                let w: Vec<f32> = (first..last)
                    .map(|x| ((b.min(x as f64 + 1.0) - a.max(x as f64)) / scale) as f32)
                    .collect();
                (first, w)
            }
        })
        .collect()
}

/// cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).
pub fn resize_area(img: &Image, w: usize, h: usize) -> Image {
    if (w, h) == (img.w, img.h) {
        return img.clone();
    }
    let (wx, wy) = (area_weights(img.w, w), area_weights(img.h, h));
    let mut tmp = vec![0f32; w * img.h * 3];
    for y in 0..img.h {
        let row = &img.bgr[y * img.w * 3..];
        for (x, (x0, ws)) in wx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, wt) in ws.iter().enumerate() {
                let p = &row[(x0 + k) * 3..];
                for c in 0..3 {
                    acc[c] += p[c] as f32 * wt;
                }
            }
            tmp[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&acc);
        }
    }
    let mut out = vec![0u8; w * h * 3];
    for (y, (y0, ws)) in wy.iter().enumerate() {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for (k, wt) in ws.iter().enumerate() {
                let p = &tmp[((y0 + k) * w + x) * 3..];
                for c in 0..3 {
                    acc[c] += p[c] * wt;
                }
            }
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = acc[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Image { w, h, bgr: out }
}

/// A label in white with a dark outline, its baseline at (x, base) (run.py's cv2.putText pair).
pub fn label(img: &mut Image, x: f64, base: f64, text: &str) {
    use sp_present::text::{Canvas, Face, Font, Rgb};
    let font = Font::new(Face::Bold, 34.0);
    let mut c = Canvas::new(img.w, img.h);
    for (dx, dy) in [(-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0), (-1.5, -1.5), (1.5, 1.5), (-1.5, 1.5), (1.5, -1.5)] {
        c.text(x + dx, base + dy, text, &font, Rgb(0, 0, 0));
    }
    c.text(x, base, text, &font, Rgb(255, 255, 255));
    let px = c.pixels();
    for (i, p) in img.bgr.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let q = &px[i * 4..i * 4 + 4];
        let a = q[3] as u32;
        if a == 0 {
            continue;
        }
        for k in 0..3 {
            // premultiplied over
            p[k] = (q[k] as u32 + p[k] as u32 * (255 - a) / 255).min(255) as u8;
        }
    }
}

/// Pictures side by side (cv2.hconcat), all the same height.
pub fn hconcat(a: &Image, b: &Image) -> Image {
    let w = a.w + b.w;
    let mut out = Vec::with_capacity(w * a.h * 3);
    for y in 0..a.h {
        out.extend(&a.bgr[y * a.w * 3..(y + 1) * a.w * 3]);
        out.extend(&b.bgr[y * b.w * 3..(y + 1) * b.w * 3]);
    }
    Image { w, h: a.h, bgr: out }
}

/// One above the other (cv2.vconcat), both the same width.
pub fn vconcat(a: &Image, b: &Image) -> Image {
    let mut bgr = a.bgr.clone();
    bgr.extend(&b.bgr);
    Image { w: a.w, h: a.h + b.h, bgr }
}
