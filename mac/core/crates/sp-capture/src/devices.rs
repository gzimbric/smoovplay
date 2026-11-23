//! What the Mac has (framegen/capture.py's devices and modes): cameras, capture cards and microphones as
//! AVFoundation lists them, the ways a card can send video, the one SmoovPlay asks for ([`pick_mode`]), and the
//! card's own sound device ([`pick_audio`]).
//!
//! AVFoundation's formats come as FourCCs. They're named here as FFmpeg names them, so pick_mode is Python's as it
//! stands and `--formats NAME --json` reads the same. '420v' and '420f' are "nv12" (limited and full range), 'yuvs'
//! is "yuyv422", '2vuy' "uyvy422", 'dmb1' "mjpeg" and 'x420' "p010", which is 10-bit and noted but not used, as on
//! Windows.
//!
//! Listing reads AVFoundation's device list and each device's format list. Neither opens a device, so nothing is
//! captured and macOS asks nothing.

use objc2::rc::Retained;
use objc2_av_foundation::{
    AVCaptureDevice, AVCaptureDeviceDiscoverySession, AVCaptureDeviceFormat, AVCaptureDevicePosition,
    AVCaptureDeviceType, AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeContinuityCamera,
    AVCaptureDeviceTypeDeskViewCamera, AVCaptureDeviceTypeExternal, AVCaptureDeviceTypeMicrophone, AVFrameRateRange,
    AVMediaTypeAudio, AVMediaTypeMuxed, AVMediaTypeVideo,
};
use objc2_core_foundation::{CFBoolean, CFRetained, CFString};
use objc2_core_media::{
    kCMFormatDescriptionExtension_FullRangeVideo, kCMFormatDescriptionExtension_YCbCrMatrix,
    CMVideoFormatDescriptionGetDimensions,
};
use objc2_foundation::NSArray;
use serde::{Deserialize, Serialize};

/// Raw formats, best first. NV12 is the smallest uncompressed format (4:2:0). YUY2 and UYVY are 4:2:2, which takes
/// more USB bandwidth for the same picture from 4:2:0 HDMI sources. Anything compressed comes after all of them.
pub const RAW_RANK: &[(&str, u32)] =
    &[("nv12", 0), ("yuyv422", 1), ("uyvy422", 2), ("yuv420p", 3), ("bgr24", 4), ("bgr0", 4), ("rgb24", 4)];
pub const COMPRESSED_RANK: &[(&str, u32)] = &[("mjpeg", 10)];
pub const BYTES_PER_PIXEL: &[(&str, f64)] = &[
    ("nv12", 1.5),
    ("yuv420p", 1.5),
    ("yuyv422", 2.0),
    ("uyvy422", 2.0),
    ("bgr24", 3.0),
    ("rgb24", 3.0),
    ("bgr0", 4.0),
];
/// A card's 10-bit formats (HDR), noted but not used (capture.py's _UNREADABLE).
pub const UNREADABLE: &[&str] = &["p010", "p016", "p210"];

fn rank(fmt: &str) -> Option<u32> {
    RAW_RANK.iter().chain(COMPRESSED_RANK).find(|(f, _)| *f == fmt).map(|(_, r)| *r)
}

pub fn bytes_per_pixel(fmt: &str) -> Option<f64> {
    BYTES_PER_PIXEL.iter().find(|(f, _)| *f == fmt).map(|(_, b)| *b)
}

/// A capture device, as `--list --json` gives it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Device {
    /// What macOS calls it, and what to open it by.
    pub name: String,
    /// "video", "audio" or "unknown".
    pub kind: String,
    /// AVFoundation's unique ID, for when two devices share a name (DirectShow's "alternative name" on Windows).
    pub alt: String,
}

impl Device {
    pub fn new(name: &str, kind: &str) -> Device {
        Device { name: name.into(), kind: kind.into(), alt: String::new() }
    }
}

/// One way a device can send video (capture.py Mode, the same fields in the same order).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mode {
    /// Pixel format for raw video, such as "nv12" or "yuyv422", or the codec ("mjpeg", "h264").
    pub fmt: String,
    pub raw: bool,
    pub w: u32,
    pub h: u32,
    pub fps_min: f64,
    pub fps_max: f64,
    /// "tv" (limited) or "pc" (full) when the card says, and "" when it doesn't.
    pub range: String,
    /// Colour matrix, such as "bt709" or "bt470bg", when the card says.
    pub space: String,
}

impl Mode {
    pub fn new(fmt: &str, raw: bool, w: u32, h: u32, fps_min: f64, fps_max: f64) -> Mode {
        Mode { fmt: fmt.into(), raw, w, h, fps_min, fps_max, range: String::new(), space: String::new() }
    }

    pub fn label(&self) -> String {
        let fps = if (self.fps_max - self.fps_min).abs() < 0.01 {
            g(self.fps_max)
        } else {
            format!("{}-{}", g(self.fps_min), g(self.fps_max))
        };
        let fmt = if self.raw { self.fmt.clone() } else { self.fmt.to_uppercase() };
        format!("{}x{} {} fps {}", self.w, self.h, fps, fmt)
    }
}

/// Python's `f"{x:g}"`, six significant digits with no trailing zeros.
pub fn g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() {
        return if x.is_nan() { "nan".into() } else if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let sci = format!("{x:.5e}"); // d.ddddde[-]x, rounded to six digits as %g rounds
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let strip = |s: String| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    };
    if !(-4..6).contains(&exp) {
        let m = strip(mant.to_string());
        return format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    strip(format!("{:.*}", (5 - exp).max(0) as usize, x))
}

/// The USB 2 note, in the Mac's words (Windows' says "blue, ideally on the back of the PC").
pub const USB2_NOTE: &str = "the card offers 1280x720 at 30 fps at most, as cards do on a USB 2 port: connect it to a \
                             USB 3 port (on the Mac itself, or a USB 3 hub)";

/// The mode to ask for, and a note for the log when it isn't what we'd want, or "" when it is (capture.py pick_mode).
///
/// The size asked for if the card has it, else the largest at or under it, else the smallest. The rate reaches `fps`
/// if possible, and the format is the best one, raw before compressed and NV12 first.
pub fn pick_mode(modes: &[Mode], w: u32, h: u32, fps: f64) -> (Option<Mode>, String) {
    if modes.is_empty() {
        return (None, "the card listed no video formats".into());
    }
    let usable: Vec<&Mode> = modes.iter().filter(|m| rank(&m.fmt).is_some()).collect();
    if usable.is_empty() {
        let mut fmts: Vec<&str> = modes.iter().map(|m| m.fmt.as_str()).collect();
        fmts.sort();
        fmts.dedup();
        return (None, format!("the card offers no format SmoovPlay can read ({})", fmts.join(", ")));
    }
    let mut sizes: Vec<(u32, u32)> = Vec::new();
    for m in &usable {
        if !sizes.contains(&(m.w, m.h)) {
            sizes.push((m.w, m.h));
        }
    }
    let area = |s: &(u32, u32)| s.0 as u64 * s.1 as u64;
    let size = if sizes.contains(&(w, h)) {
        (w, h)
    } else {
        let under: Vec<(u32, u32)> = sizes.iter().copied().filter(|s| s.0 <= w && s.1 <= h).collect();
        // the first of equal areas, in the card's order
        let best = |v: &[(u32, u32)], larger: bool| {
            let mut pick = v[0];
            for s in &v[1..] {
                if (larger && area(s) > area(&pick)) || (!larger && area(s) < area(&pick)) {
                    pick = *s;
                }
            }
            pick
        };
        if under.is_empty() { best(&sizes, false) } else { best(&under, true) }
    };
    let at: Vec<&Mode> = usable.iter().copied().filter(|m| (m.w, m.h) == size).collect();
    let fast: Vec<&Mode> = at.iter().copied().filter(|m| m.fps_max >= fps - 0.5).collect();
    let pool = if fast.is_empty() { &at } else { &fast };
    let key = |m: &Mode| (rank(&m.fmt).unwrap_or(99), -m.fps_max);
    let mut best = pool[0];
    for m in &pool[1..] {
        let (a, b) = (key(m), key(best));
        if a.0 < b.0 || (a.0 == b.0 && a.1 < b.1) {
            best = m;
        }
    }
    let mut notes: Vec<String> = Vec::new();
    if size != (w, h) {
        notes.push(format!("no {w}x{h} mode, using {}x{}", size.0, size.1));
    }
    if fast.is_empty() {
        notes.push(format!("no mode reaches {} fps at that size (best {})", g(fps), g(best.fps_max)));
    }
    if !best.raw {
        notes.push(format!(
            "only {} (compressed) at that size and rate: decoded by macOS before SmoovPlay gets it",
            best.fmt.to_uppercase()
        ));
    }
    if usable.iter().all(|m| m.w * m.h <= 1280 * 720 && m.fps_max <= 30.5) && (w * h > 1280 * 720 || fps > 30.5) {
        notes.push(USB2_NOTE.into());
    }
    if modes.iter().any(|m| UNREADABLE.contains(&m.fmt.as_str())) {
        notes.push(format!(
            "the card also offers 10-bit HDR, which SmoovPlay can't read yet, so it uses 8-bit {}: if the picture \
             looks washed out, turn HDR off on the console",
            best.fmt.to_uppercase()
        ));
    }
    (Some(best.clone()), notes.join("; "))
}

/// The rate to ask a mode for (capture.py open_options' framerate). That's `fps` if the mode reaches it, else its
/// fastest.
pub fn open_rate(mode: &Mode, fps: f64) -> f64 {
    if mode.fps_max >= fps - 0.5 { fps.min(mode.fps_max) } else { mode.fps_max }
}

// ----------------------------------------------------------------------------- the card's sound device, by name
/// Words in device names that say nothing about which card it is.
const GENERIC: &[&str] = &[
    "usb", "usb2", "usb3", "audio", "video", "capture", "device", "digital", "interface", "input", "hdmi", "line",
    "microphone", "mic", "camera", "webcam", "stream", "streaming", "game", "high", "definition", "the", "and", "for",
    "with", "pro", "plus",
];

fn words(name: &str) -> Vec<String> {
    let low = name.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    for w in low.split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit())) {
        if w.chars().count() >= 3 && !GENERIC.contains(&w) && !out.iter().any(|o| o == w) {
            out.push(w.to_string());
        }
    }
    out
}

/// AVerMedia's cards add a sound device named "Not To Use (<card>-Audio)" next to the real one.
fn not_to_use(name: &str) -> bool {
    let low = name.to_lowercase();
    let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    low.match_indices("not to use").any(|(i, m)| !word(low[..i].chars().last()) && !word(low[i + m.len()..].chars().next()))
}

/// A device's name without a "-Video" or "-Audio" ending, in lower case, so "Live Gamer Ultra 2.1-Video" becomes
/// "live gamer ultra 2.1". A name that would leave only generic words stays as it is, like "USB Video".
fn card_name(name: &str) -> String {
    let low = name.to_lowercase().trim().to_string();
    let core = ["video", "audio"]
        .iter()
        .find_map(|e| low.strip_suffix(e))
        .map(|c| c.trim_end_matches(|ch: char| ch.is_whitespace() || ch == '_' || ch == '-').to_string());
    match core {
        Some(c) if c != low && !words(&c).is_empty() => c,
        _ => low,
    }
}

/// The capture card's own sound device, which is one named like the video device, else None. For example "USB3.0
/// Capture" and "Digital Audio Interface (USB3.0 Capture)", "Cam Link 4K" for both, or "Live Gamer Ultra 2.1-Video"
/// and "HDMI/Line In (Live Gamer Ultra 2.1-Audio)". A device whose name says nothing (only generic words) matches
/// only on the whole name. AVerMedia's "Not To Use" sound device is never picked.
pub fn pick_audio(video: &str, devices: &[Device]) -> Option<String> {
    let audio: Vec<&str> = devices
        .iter()
        .filter(|d| d.kind == "audio" && !d.name.is_empty() && !not_to_use(&d.name))
        .map(|d| d.name.as_str())
        .collect();
    let v = card_name(video);
    let whole: Vec<&str> = audio
        .iter()
        .copied()
        .filter(|a| !v.is_empty() && (a.to_lowercase().contains(&v) || v.contains(&a.to_lowercase())))
        .collect();
    if let Some(first) = whole.first() {
        let mut best = *first;
        for a in &whole[1..] {
            if a.chars().count() < best.chars().count() {
                best = a;
            }
        }
        return Some(best.to_string());
    }
    let vw = words(video);
    let scored: Vec<(usize, &str)> =
        audio.iter().map(|a| (words(a).iter().filter(|w| vw.contains(w)).count(), *a)).collect();
    let top = scored.iter().map(|s| s.0).max().unwrap_or(0);
    let mut at_top = scored.iter().filter(|s| s.0 == top);
    match (at_top.next(), at_top.next()) {
        (Some(s), None) if top > 0 => Some(s.1.to_string()),
        _ => None,
    }
}

// ----------------------------------------------------------------------------- AVFoundation's formats, named
/// The FFmpeg name of an AVFoundation pixel format or codec, whether it's raw, and its range when the FourCC says.
pub fn format_name(code: u32) -> (String, bool, &'static str) {
    let (name, raw, range) = match &code.to_be_bytes() {
        b"420v" => ("nv12", true, "tv"),
        b"420f" => ("nv12", true, "pc"),
        b"yuvs" => ("yuyv422", true, ""),
        b"yuvf" => ("yuyv422", true, "pc"),
        b"2vuy" => ("uyvy422", true, ""),
        b"y420" => ("yuv420p", true, ""),
        b"f420" => ("yuv420p", true, "pc"),
        b"BGRA" => ("bgr0", true, ""),
        b"24BG" => ("bgr24", true, ""),
        b"24RG" => ("rgb24", true, ""),
        b"dmb1" | b"jpeg" | b"mjpa" | b"mjpb" => ("mjpeg", false, ""),
        b"avc1" => ("h264", false, ""),
        b"hvc1" | b"hev1" => ("hevc", false, ""),
        // 10-bit, noted but not used, with raw false as Windows' "unknown compression type" lists them
        b"x420" => ("p010", false, "tv"),
        b"xf20" => ("p010", false, "pc"),
        b"x422" => ("p210", false, "tv"),
        b"xf22" => ("p210", false, "pc"),
        other => {
            let s: String = other.iter().map(|&c| if c.is_ascii_graphic() { c as char } else { '?' }).collect();
            return (s.trim().to_lowercase(), false, "");
        }
    };
    (name.to_string(), raw, range)
}

/// FFmpeg's name for a CoreVideo YCbCr matrix.
fn space_name(matrix: &str) -> &'static str {
    match matrix {
        "ITU_R_709_2" => "bt709",
        "ITU_R_601_4" => "bt470bg",
        "SMPTE_240M_1995" => "smpte240m",
        "ITU_R_2020" => "bt2020nc",
        _ => "",
    }
}

/// A Mode from what AVFoundation says of a format and one of its frame rate ranges. Rates are rounded to 1/1000
/// (59.94005994 reads as 59.94). The device is opened with the range's own frame durations.
pub fn mode_of(code: u32, w: u32, h: u32, fps_min: f64, fps_max: f64, matrix: Option<&str>, full: Option<bool>) -> Mode {
    let (fmt, raw, range) = format_name(code);
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    let range = if !range.is_empty() {
        range.to_string()
    } else {
        match full {
            Some(true) => "pc".into(),
            Some(false) if raw => "tv".into(),
            _ => String::new(),
        }
    };
    Mode {
        fmt,
        raw,
        w,
        h,
        fps_min: r3(fps_min),
        fps_max: r3(fps_max),
        range,
        space: matrix.map(space_name).unwrap_or("").to_string(),
    }
}

/// One of a device's modes and what opens it.
#[derive(Clone)]
pub struct DeviceMode {
    pub mode: Mode,
    pub code: u32,
    pub format: Retained<AVCaptureDeviceFormat>,
    pub rates: Retained<AVFrameRateRange>,
}

// AVFoundation's format objects are immutable descriptions.
unsafe impl Send for DeviceMode {}

fn cfstring(v: Option<CFRetained<objc2_core_foundation::CFPropertyList>>) -> Option<String> {
    let s: CFRetained<CFString> = v?.downcast().ok()?;
    Some(s.to_string())
}

fn cfbool(v: Option<CFRetained<objc2_core_foundation::CFPropertyList>>) -> Option<bool> {
    let b: CFRetained<CFBoolean> = v?.downcast().ok()?;
    Some(b.as_bool())
}

/// A device's modes, one per format and frame rate range, in AVFoundation's order except that a limited-range
/// format comes before the same one in full range. HDMI video is limited range unless the card says otherwise, and
/// pick_mode takes the first of equals. Duplicates are dropped.
pub fn device_modes(device: &AVCaptureDevice) -> Vec<DeviceMode> {
    let mut out: Vec<DeviceMode> = Vec::new();
    unsafe {
        for f in device.formats().iter() {
            let desc = f.formatDescription();
            let code = desc.media_sub_type();
            let dims = CMVideoFormatDescriptionGetDimensions(&desc);
            let matrix = cfstring(desc.extension(kCMFormatDescriptionExtension_YCbCrMatrix));
            let full = cfbool(desc.extension(kCMFormatDescriptionExtension_FullRangeVideo));
            for r in f.videoSupportedFrameRateRanges().iter() {
                let mode = mode_of(
                    code,
                    dims.width.max(0) as u32,
                    dims.height.max(0) as u32,
                    r.minFrameRate(),
                    r.maxFrameRate(),
                    matrix.as_deref(),
                    full,
                );
                if !out.iter().any(|d| d.mode == mode) {
                    out.push(DeviceMode { mode, code, format: f.clone(), rates: r.clone() });
                }
            }
        }
    }
    // stable, so "tv" stays before "pc" for otherwise equal modes
    let same = |a: &Mode, b: &Mode| (a.fmt.as_str(), a.w, a.h, a.fps_max) == (b.fmt.as_str(), b.w, b.h, b.fps_max);
    let mut i = 1;
    while i < out.len() {
        let mut j = i;
        while j > 0 && same(&out[j - 1].mode, &out[j].mode) && out[j - 1].mode.range == "pc" && out[j].mode.range == "tv" {
            out.swap(j - 1, j);
            j -= 1;
        }
        i += 1;
    }
    out
}

fn discover(types: &[&AVCaptureDeviceType], media: Option<&objc2_av_foundation::AVMediaType>) -> Vec<Retained<AVCaptureDevice>> {
    unsafe {
        let types = NSArray::from_slice(types);
        let s = AVCaptureDeviceDiscoverySession::discoverySessionWithDeviceTypes_mediaType_position(
            &types,
            media,
            AVCaptureDevicePosition::Unspecified,
        );
        s.devices().iter().collect()
    }
}

/// The Mac's video capture devices (cards, cameras, virtual cameras), then its sound inputs, as AVFoundation lists
/// them. A device that carries both (a muxed one) counts as video.
pub fn av_devices() -> (Vec<Retained<AVCaptureDevice>>, Vec<Retained<AVCaptureDevice>>) {
    unsafe {
        let vtypes = [
            AVCaptureDeviceTypeExternal,
            AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVCaptureDeviceTypeContinuityCamera,
            AVCaptureDeviceTypeDeskViewCamera,
        ];
        let mut video = discover(&vtypes, AVMediaTypeVideo);
        for d in discover(&vtypes, AVMediaTypeMuxed) {
            if !video.iter().any(|v| v.uniqueID().to_string() == d.uniqueID().to_string()) {
                video.push(d);
            }
        }
        let audio: Vec<_> = discover(&[AVCaptureDeviceTypeMicrophone, AVCaptureDeviceTypeExternal], AVMediaTypeAudio)
            .into_iter()
            .filter(|a| !video.iter().any(|v| v.uniqueID().to_string() == a.uniqueID().to_string()))
            .collect();
        (video, audio)
    }
}

fn describe(d: &AVCaptureDevice, kind: &str) -> Device {
    unsafe { Device { name: d.localizedName().to_string(), kind: kind.into(), alt: d.uniqueID().to_string() } }
}

/// The capture devices the Mac has (capture.py list_devices), video ones first, then sound.
pub fn list_devices() -> Vec<Device> {
    let (video, audio) = av_devices();
    video.iter().map(|d| describe(d, "video")).chain(audio.iter().map(|d| describe(d, "audio"))).collect()
}

/// Whether a device carries sound itself, because its own input has an audio port. That's a muxed device, or one
/// with both.
pub fn has_sound(d: &AVCaptureDevice) -> bool {
    unsafe {
        AVMediaTypeAudio.is_some_and(|a| d.hasMediaType(a)) || AVMediaTypeMuxed.is_some_and(|m| d.hasMediaType(m))
    }
}

/// A video device by unique ID, name, or number in the list (`--capture N`).
pub fn find_video(spec: &str) -> Option<Retained<AVCaptureDevice>> {
    let (video, _) = av_devices();
    find_in(&video, spec)
}

/// A sound device by unique ID, name, or number among the sound devices. Naming a video device gives its own
/// sound.
pub fn find_audio(spec: &str) -> Option<Retained<AVCaptureDevice>> {
    let (video, audio) = av_devices();
    find_in(&audio, spec).or_else(|| find_in(&video, spec).filter(|d| has_sound(d)))
}

fn find_in(list: &[Retained<AVCaptureDevice>], spec: &str) -> Option<Retained<AVCaptureDevice>> {
    unsafe {
        list.iter()
            .find(|d| d.uniqueID().to_string() == spec)
            .or_else(|| list.iter().find(|d| d.localizedName().to_string() == spec))
            .or_else(|| spec.parse::<usize>().ok().and_then(|i| list.get(i)))
            .cloned()
    }
}

/// What a video device offers (capture.py list_modes), by unique ID, name or number.
pub fn list_modes(spec: &str) -> Result<Vec<Mode>, String> {
    let d = find_video(spec).ok_or_else(|| format!("macOS lists no video device {spec:?}"))?;
    Ok(device_modes(&d).into_iter().map(|m| m.mode).collect())
}

#[derive(Serialize)]
struct DevicesJson<'a> {
    video: Vec<&'a Device>,
    audio: Vec<&'a Device>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

/// `--list --json`: {"video": [...], "audio": [...]}, with devices of unknown kind among the video ones and keys in
/// the Python's order.
pub fn devices_json(devices: &[Device], error: Option<&str>) -> String {
    serde_json::to_string(&DevicesJson {
        video: devices.iter().filter(|d| d.kind == "video" || d.kind == "unknown").collect(),
        audio: devices.iter().filter(|d| d.kind == "audio").collect(),
        error,
    })
    .unwrap()
}

#[derive(Serialize)]
struct FormatsJson<'a> {
    modes: &'a [Mode],
    pick: Option<Mode>,
    note: String,
}

/// `--formats NAME --json`: {"modes": [...], "pick": mode or null, "note": "..."}.
pub fn formats_json(modes: &[Mode]) -> String {
    let (pick, note) = pick_mode(modes, 1920, 1080, 60.0);
    serde_json::to_string(&FormatsJson { modes, pick, note }).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_is_pythons() {
        for (x, s) in [(60.0, "60"), (59.94, "59.94"), (29.97, "29.97"), (7.5, "7.5"), (0.03, "0.03"), (120.0, "120"),
                       (59.9400599, "59.9401"), (1e-5, "1e-05"), (1234567.0, "1.23457e+06"), (100000.0, "100000")] {
            assert_eq!(g(x), s, "{x}");
        }
    }

    #[test]
    fn card_names() {
        assert_eq!(card_name("Live Gamer Ultra 2.1-Video"), "live gamer ultra 2.1");
        assert_eq!(card_name("USB Video"), "usb video");
        assert_eq!(card_name("Game Capture HD60 X Audio"), "game capture hd60 x");
        assert!(not_to_use("Not To Use (Live Gamer Ultra 2.1-Audio)") && !not_to_use("Notto use"));
    }
}
