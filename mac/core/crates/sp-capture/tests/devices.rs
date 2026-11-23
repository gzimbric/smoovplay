//! Devices and modes (tests/test_capture.py's pick_mode tests, tests/test_audio.py's pick_audio test), checked
//! against the Python's own answers on many more inputs (fixtures/capture/picks.json), and AVFoundation's formats
//! as FFmpeg names them.

mod common;

use sp_capture::devices::{devices_json, format_name, formats_json, g, mode_of, open_rate, USB2_NOTE};
use sp_capture::{fourcc, list_devices, pick_audio, pick_mode, Device, Mode};

/// tests/test_capture.py OPTIONS, as parse_modes reads it: an MS2130-style stick, YUY2 and MJPEG at 1080p, NV12 only
/// at 4K30, colour info on some lines.
fn options() -> Vec<Mode> {
    let m = |fmt: &str, raw: bool, w, h, fps: f64, range: &str, space: &str| Mode {
        range: range.into(),
        space: space.into(),
        ..Mode::new(fmt, raw, w, h, fps, fps)
    };
    vec![
        m("yuyv422", true, 1920, 1080, 60.0, "tv", "bt709"),
        m("yuyv422", true, 1920, 1080, 30.0, "", ""),
        m("yuyv422", true, 1280, 720, 60.0, "", ""),
        m("mjpeg", false, 1920, 1080, 60.0, "", ""),
        m("mjpeg", false, 3840, 2160, 30.0, "", ""),
        m("nv12", true, 3840, 2160, 30.0, "pc", ""),
    ]
}

#[test]
fn pick_prefers_raw_at_the_size_and_rate_asked_for() {
    let modes = options();
    let (best, note) = pick_mode(&modes, 1920, 1080, 60.0);
    let best = best.unwrap();
    assert_eq!((best.fmt.as_str(), best.w, best.fps_max), ("yuyv422", 1920, 60.0));
    assert_eq!(note, "");
    // NV12 beats YUY2 at the same size and rate
    let nv12 = Mode::new("nv12", true, 1920, 1080, 60.0, 60.0);
    let mut with = modes.clone();
    with.push(nv12.clone());
    assert_eq!(pick_mode(&with, 1920, 1080, 60.0).0, Some(nv12));
    // only MJPEG reaches 60 at 1080p: it's used, with a warning; a raw 30 fps mode isn't preferred over it
    let no_raw60: Vec<Mode> = modes.iter().filter(|m| !(m.raw && m.fps_max == 60.0 && m.w == 1920)).cloned().collect();
    let (best, note) = pick_mode(&no_raw60, 1920, 1080, 60.0);
    assert_eq!(best.unwrap().fmt, "mjpeg");
    assert!(note.contains("compressed"), "{note}");
    // no 1080p at all: the largest size under it
    let small: Vec<Mode> = modes.iter().filter(|m| m.w != 1920).cloned().collect();
    let (best, note) = pick_mode(&small, 1920, 1080, 60.0);
    let best = best.unwrap();
    assert_eq!((best.w, best.h), (1280, 720));
    assert!(note.contains("1280x720"));
    assert!(pick_mode(&[], 1920, 1080, 60.0).0.is_none());
    assert!(pick_mode(&[Mode::new("h264", false, 1920, 1080, 60.0, 60.0)], 1920, 1080, 60.0).0.is_none());
}

#[test]
fn a_cards_hdr_modes_are_noted_but_not_used() {
    let mut modes = options();
    modes.push(Mode::new("p010", false, 1920, 1080, 60.0, 60.0));
    let (best, note) = pick_mode(&modes, 1920, 1080, 60.0);
    assert_eq!(best.unwrap().fmt, "yuyv422");
    assert!(note.contains("10-bit HDR") && note.contains("turn HDR off"), "{note}");
    // as AVFoundation lists a 10-bit format
    assert_eq!(mode_of(fourcc(b"x420"), 1920, 1080, 60.0, 60.0, None, None), Mode { range: "tv".into(), ..Mode::new("p010", false, 1920, 1080, 60.0, 60.0) });
}

#[test]
fn a_card_on_usb_2_is_pointed_to_usb_3() {
    let slow = vec![Mode::new("nv12", true, 1280, 720, 30.0, 30.0), Mode::new("nv12", true, 640, 480, 30.0, 30.0)];
    assert!(pick_mode(&slow, 1920, 1080, 60.0).1.contains("USB 3"));
    let mut fast = slow.clone();
    fast.push(Mode::new("nv12", true, 1920, 1080, 60.0, 60.0));
    assert!(!pick_mode(&fast, 1920, 1080, 60.0).1.contains("USB 3"));
}

#[test]
fn open_rate_is_open_options_framerate() {
    // test_open_options: 59.94 asked as 59.94; a 30-60 mode at 60
    assert_eq!(g(open_rate(&Mode::new("nv12", true, 1920, 1080, 59.94, 59.94), 60.0)), "59.94");
    assert_eq!(open_rate(&Mode::new("mjpeg", false, 1920, 1080, 30.0, 60.0), 60.0), 60.0);
    assert_eq!(open_rate(&Mode::new("nv12", true, 1920, 1080, 30.0, 30.0), 60.0), 30.0);
    assert_eq!(open_rate(&Mode::new("nv12", true, 1920, 1080, 30.0, 120.0), 60.0), 60.0);
}

#[test]
fn modes_label_as_the_python_does() {
    assert_eq!(Mode::new("nv12", true, 1920, 1080, 60.0, 60.0).label(), "1920x1080 60 fps nv12");
    assert_eq!(Mode::new("mjpeg", false, 1920, 1080, 30.0, 60.0).label(), "1920x1080 30-60 fps MJPEG");
    assert_eq!(Mode::new("nv12", true, 1920, 1080, 59.94, 59.94).label(), "1920x1080 59.94 fps nv12");
}

#[test]
fn avfoundation_formats_are_named_as_ffmpeg_names_them() {
    let name = |s: &[u8; 4]| format_name(fourcc(s));
    assert_eq!(name(b"420v"), ("nv12".into(), true, "tv"));
    assert_eq!(name(b"420f"), ("nv12".into(), true, "pc"));
    assert_eq!(name(b"yuvs"), ("yuyv422".into(), true, ""));
    assert_eq!(name(b"2vuy"), ("uyvy422".into(), true, ""));
    assert_eq!(name(b"dmb1"), ("mjpeg".into(), false, ""));
    assert_eq!(name(b"avc1"), ("h264".into(), false, ""));
    assert_eq!(name(b"BGRA"), ("bgr0".into(), true, ""));
    assert_eq!(name(b"xf20"), ("p010".into(), false, "pc"));
    // a mode from a format: rates to 1/1000, the matrix and range from its tags
    let m = mode_of(fourcc(b"2vuy"), 1920, 1080, 60000.0 / 1001.0, 60000.0 / 1001.0, Some("ITU_R_709_2"), Some(false));
    assert_eq!(m, Mode { range: "tv".into(), space: "bt709".into(), ..Mode::new("uyvy422", true, 1920, 1080, 59.94, 59.94) });
    let m = mode_of(fourcc(b"dmb1"), 1280, 720, 5.0, 30.0, Some("ITU_R_601_4"), None);
    assert_eq!((m.range.as_str(), m.space.as_str(), m.fps_min), ("", "bt470bg", 5.0));
    // pick_mode on a Mac card's list: NV12 in limited range first of equals (the listing puts 420v before 420f)
    let modes = vec![
        mode_of(fourcc(b"420v"), 1920, 1080, 60.0, 60.0, None, None),
        mode_of(fourcc(b"420f"), 1920, 1080, 60.0, 60.0, None, None),
        mode_of(fourcc(b"yuvs"), 1920, 1080, 60.0, 60.0, None, None),
        mode_of(fourcc(b"dmb1"), 1920, 1080, 60.0, 60.0, None, None),
    ];
    let best = pick_mode(&modes, 1920, 1080, 60.0).0.unwrap();
    assert_eq!((best.fmt.as_str(), best.range.as_str()), ("nv12", "tv"));
}

#[test]
fn the_cards_sound_device_is_found_by_name() {
    let d = Device::new;
    let devs = vec![
        d("USB3.0 Capture", "video"),
        d("Digital Audio Interface (USB3.0 Capture)", "audio"),
        d("Microphone (Realtek Audio)", "audio"),
        d("Cam Link 4K", "video"),
        d("Cam Link 4K", "audio"),
        d("Game Capture HD60 X", "video"),
        d("Game Capture HD60 X Audio", "audio"),
    ];
    let pick = |v: &str, devs: &[Device]| pick_audio(v, devs);
    assert_eq!(pick("USB3.0 Capture", &devs).as_deref(), Some("Digital Audio Interface (USB3.0 Capture)"));
    assert_eq!(pick("Cam Link 4K", &devs).as_deref(), Some("Cam Link 4K"));
    assert_eq!(pick("Game Capture HD60 X", &devs).as_deref(), Some("Game Capture HD60 X Audio"));
    assert_eq!(pick("USB Video", &devs), None); // generic words only: no guess
    let elgato = vec![d("Elgato 4K X", "video"), d("Line (Elgato 4K X)", "audio"), d("Line (Elgato Wave)", "audio")];
    assert_eq!(pick("Elgato 4K X", &elgato).as_deref(), Some("Line (Elgato 4K X)"));
    // AVerMedia (issue 23): "-Video"/"-Audio" endings, and a "Not To Use" device named like the real one
    let avermedia = vec![
        d("Live Gamer Ultra 2.1-Video", "video"),
        d("HDMI/Line In (Live Gamer Ultra 2.1-Audio)", "audio"),
        d("Virtual Microphone (AVerMedia Streaming Center)", "audio"),
        d("Not To Use (Live Gamer Ultra 2.1-Audio)", "audio"),
    ];
    assert_eq!(pick("Live Gamer Ultra 2.1-Video", &avermedia).as_deref(), Some("HDMI/Line In (Live Gamer Ultra 2.1-Audio)"));
    let without: Vec<Device> = avermedia.iter().enumerate().filter(|(i, _)| *i != 1).map(|(_, d)| d.clone()).collect();
    assert_eq!(pick("Live Gamer Ultra 2.1-Video", &without), None); // never "Not To Use"
}

#[test]
fn the_sound_device_follows_capture_audio() {
    use sp_capture::source::pick_sound_device;
    let devs = vec![Device::new("Cam Link 4K", "video"), Device::new("Cam Link 4K", "audio"), Device::new("Mic", "audio")];
    let mut said = Vec::new();
    let mut say = |l: String| said.push(l);
    assert_eq!(pick_sound_device("Cam Link 4K", "auto", &devs, &mut say).as_deref(), Some("Cam Link 4K"));
    assert_eq!(pick_sound_device("Cam Link 4K", "off", &devs, &mut say), None);
    assert_eq!(pick_sound_device("Cam Link 4K", "1", &devs, &mut say).as_deref(), Some("Mic"));
    assert_eq!(pick_sound_device("Cam Link 4K", "Mic", &devs, &mut say).as_deref(), Some("Mic"));
    // a card with no sound device of its own (the AVerMedia GC573, issue 9): its own sound is tried
    let gc573 = vec![Device::new("AVerMedia HD Capture GC573 1", "video"), Device::new("Microphone (USB Audio)", "audio")];
    assert_eq!(pick_sound_device("AVerMedia HD Capture GC573 1", "auto", &gc573, &mut say).as_deref(), Some("AVerMedia HD Capture GC573 1"));
    assert!(said.iter().any(|l| l.contains("trying the card's own sound")));
}

#[test]
fn json_has_the_pythons_shape() {
    let devs = vec![Device { name: "Cam Link 4K".into(), kind: "video".into(), alt: "0x1234".into() }, Device::new("Cam Link 4K", "audio")];
    assert_eq!(
        devices_json(&devs, None),
        r#"{"video":[{"name":"Cam Link 4K","kind":"video","alt":"0x1234"}],"audio":[{"name":"Cam Link 4K","kind":"audio","alt":""}]}"#
    );
    assert!(devices_json(&[], Some("boom")).ends_with(r#""error":"boom"}"#));
    let f: serde_json::Value = serde_json::from_str(&formats_json(&options())).unwrap();
    assert_eq!(f["modes"].as_array().unwrap().len(), 6);
    assert_eq!(f["modes"][0], serde_json::json!({"fmt": "yuyv422", "raw": true, "w": 1920, "h": 1080, "fps_min": 60.0, "fps_max": 60.0, "range": "tv", "space": "bt709"}));
    assert_eq!(f["pick"]["fmt"], "yuyv422");
    assert_eq!(f["note"], "");
}

/// The Python's pick_mode, pick_audio and sound_report on the same inputs (picks.json): the same answers, the
/// Windows-only words aside.
#[test]
fn picks_match_the_python() {
    let Some(p) = common::fixture("picks.json") else { return };
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let py_usb = "the card offers 1280x720 at 30 fps at most, as cards do on a USB 2 port: connect it to a USB 3 port \
                  (blue, ideally on the back of the PC)";
    let mac = |note: &str| note.replace(py_usb, USB2_NOTE).replace("decoded on the GPU where it can be", "decoded by macOS before SmoovPlay gets it");
    let mut n = 0;
    for c in v["modes"].as_array().unwrap() {
        let modes: Vec<Mode> = serde_json::from_value(c["modes"].clone()).unwrap();
        let (w, h, fps) = (c["w"].as_u64().unwrap() as u32, c["h"].as_u64().unwrap() as u32, c["fps"].as_f64().unwrap());
        let (best, note) = pick_mode(&modes, w, h, fps);
        let want: Option<Mode> = serde_json::from_value(c["pick"].clone()).unwrap();
        assert_eq!(best, want, "{modes:?} {w}x{h}@{fps}");
        assert_eq!(note, mac(c["note"].as_str().unwrap()), "{modes:?} {w}x{h}@{fps}");
        if let Some(l) = c["label"].as_str() {
            assert_eq!(best.unwrap().label(), l);
        }
        n += 1;
    }
    for c in v["audio"].as_array().unwrap() {
        let devs: Vec<Device> = serde_json::from_value(c["devices"].clone()).unwrap();
        let video = c["video"].as_str().unwrap();
        assert_eq!(pick_audio(video, &devs).as_deref(), c["pick"].as_str(), "{video} in {devs:?}");
    }
    for r in v["reports"].as_array().unwrap() {
        let stats = |v: &serde_json::Value| sp_capture::SoundStats {
            ran_dry: v["ran_dry"].as_u64().unwrap_or(0),
            output_short: v["output_short"].as_u64().unwrap_or(0),
            skips: v["skips"].as_u64().unwrap_or(0),
            refills: v["refills"].as_u64().unwrap_or(0),
            stops: v["stops"].as_u64().unwrap_or(0),
            longest_wait_ms: v["longest_wait_ms"].as_i64().unwrap_or(0),
            kept_ms: v["kept_ms"].as_i64().unwrap_or(0),
            longest_call_ms: v["longest_call_ms"].as_i64(),
            output_ms: v["output_ms"].as_i64(),
        };
        let since = (!r["since"].is_null()).then(|| stats(&r["since"]));
        let got = sp_capture::sound_report(&stats(&r["stats"]), since.as_ref());
        let want = r["line"].as_str().map(|l| {
            l.replace("Windows' output ran short", "the Mac's output ran short")
                .replace("for Windows' calls", "for the Mac's calls")
                .replace("(and Windows' buffer", "(and the Mac's output")
        });
        assert_eq!(got, want);
    }
    assert!(n > 300, "{n} cases");
}

/// AVFoundation's list (names and unique IDs only: nothing is opened).
#[test]
fn the_macs_devices_are_listed() {
    let devs = list_devices();
    for d in &devs {
        assert!(!d.name.is_empty() && !d.alt.is_empty() && (d.kind == "video" || d.kind == "audio"), "{d:?}");
    }
    let ids: std::collections::HashSet<&str> = devs.iter().map(|d| d.alt.as_str()).collect();
    assert_eq!(ids.len(), devs.len(), "a device listed twice");
    println!("{}", devices_json(&devs, None));
}
