//! `python -m framegen.capture`'s command line, for the Mac (examples/capture.rs runs it):
//!
//! ```text
//! capture --list [--json] [--raw]   # the video and audio capture devices macOS has
//! capture --formats NAME [--json]   # what a device offers, and the format SmoovPlay would pick
//! ```
//!
//! --list --json gives {"video": [{"name", "kind", "alt"}...], "audio": [...]}, where alt is AVFoundation's unique
//! ID. --formats NAME --json gives {"modes": [Mode...], "pick": Mode or null, "note": "..."}, as the Python does.
//! --raw lists what AVFoundation says of each device (its unique ID, model, maker, and whether it carries sound)
//! instead of FFmpeg's log. Neither opens a device.

use crate::devices::{self, av_devices, formats_json, has_sound, pick_mode};

pub fn main(args: &[String]) -> i32 {
    let has = |f: &str| args.iter().any(|a| a == f);
    let json = has("--json");
    if let Some(i) = args.iter().position(|a| a == "--formats") {
        let Some(name) = args.get(i + 1) else {
            eprintln!("--formats needs a device name, unique ID or number (see --list)");
            return 2;
        };
        let modes = match devices::list_modes(name) {
            Ok(m) => m,
            Err(e) => {
                if json {
                    println!("{}", serde_json::json!({ "modes": [], "pick": null, "note": e }));
                } else {
                    println!("{e}");
                }
                return 0;
            }
        };
        let (best, note) = pick_mode(&modes, 1920, 1080, 60.0);
        if json {
            println!("{}", formats_json(&modes));
        } else {
            for m in &modes {
                println!("{}{}", if Some(m) == best.as_ref() { "* " } else { "  " }, m.label());
            }
            if !note.is_empty() {
                println!("{note}");
            } else {
                println!("{}", best.map(|b| format!("would use {}", b.label())).unwrap_or("nothing usable".into()));
            }
        }
        return 0;
    }
    if has("--list") && has("--raw") {
        let (video, audio) = av_devices();
        for (kind, list) in [("video", &video), ("audio", &audio)] {
            for d in list.iter() {
                unsafe {
                    println!(
                        "{kind:7}  {}  [{}] model {:?}, by {:?}, transport {}{}",
                        d.localizedName(),
                        d.uniqueID(),
                        d.modelID().to_string(),
                        d.manufacturer().to_string(),
                        crate::fourcc_name(d.transportType() as u32),
                        if kind == "video" && has_sound(d) { ", carries sound" } else { "" }
                    );
                }
            }
        }
        if video.is_empty() && audio.is_empty() {
            println!("AVFoundation listed nothing");
        }
        return 0;
    }
    if has("--list") || args.is_empty() {
        let devices = devices::list_devices();
        if json {
            println!("{}", devices::devices_json(&devices, None));
        } else {
            for d in &devices {
                println!("{:7}  {}", d.kind, d.name);
            }
            if devices.is_empty() {
                println!("no capture devices found");
            }
        }
        return 0;
    }
    eprintln!("usage: capture --list [--json] [--raw] | --formats NAME [--json]");
    2
}
