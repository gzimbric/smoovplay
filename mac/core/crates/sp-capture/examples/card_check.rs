//! Reads a capture card (or a file standing in for one) for a while the way the pipeline does, and says how it went:
//! the frame rate it delivered, frames dropped (the queue, and macOS before handing them over), damaged frames, how
//! late frames reach SmoovPlay after the card's own timestamps, how much later the sound comes than the picture, and
//! how the sound played. Plays the card's sound on the Mac's output unless --audio off.
//!
//! ```text
//! cargo run --release -p sp-capture --example card_check -- --capture "Cam Link 4K" [--seconds 20]
//!     [--audio auto|off|NAME|N] [--format auto|nv12|...] [--size 1920x1080] [--fps 60]
//!     [--colour auto|709-limited|...] [--delay-ms 60] [--save first.ppm] [--test-mode]
//! ```
//!
//! --capture takes a name, unique ID or number from `--example capture -- --list`, or a video file.

use std::io::Write;

use sp_capture::{open_capture, player_factory, CaptureArgs};
use sp_core::{clock, Event, Source};
use sp_media::TestMode;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(capture) = arg(&args, "--capture") else {
        eprintln!("usage: card_check --capture NAME|N|FILE [--seconds S] [--audio auto|off|NAME] [--format F] [--size WxH] [--fps F] [--colour C] [--delay-ms MS] [--save FILE.ppm] [--test-mode]");
        std::process::exit(2);
    };
    let (w, h) = arg(&args, "--size")
        .and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap_or(1920), b.parse().unwrap_or(1080))))
        .unwrap_or((1920, 1080));
    let a = CaptureArgs {
        capture,
        w,
        h,
        fps: arg(&args, "--fps").and_then(|s| s.parse().ok()).unwrap_or(60.0),
        format: arg(&args, "--format").unwrap_or("auto".into()),
        colour: arg(&args, "--colour").unwrap_or("auto".into()),
        audio: arg(&args, "--audio").unwrap_or("auto".into()),
        test: args.iter().any(|a| a == "--test-mode").then(|| TestMode::shared(true)),
    };
    let seconds: f64 = arg(&args, "--seconds").and_then(|s| s.parse().ok()).unwrap_or(20.0);
    let delay_ms: f64 = arg(&args, "--delay-ms").and_then(|s| s.parse().ok()).unwrap_or(60.0);
    let save = arg(&args, "--save");
    let mut src = match open_capture(&a, Some(player_factory())) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[check] couldn't open it: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "[check] {}: {}x{} {} at {} fps, raw {}, {} {}; sound {}",
        src.name(),
        src.width(),
        src.height(),
        src.fmt(),
        src.fps(),
        src.raw(),
        if src.bt709() { "BT.709" } else { "BT.601" },
        if src.full_range() { "full range" } else { "limited range" },
        if src.can_delay_audio() { "on" } else { "off" }
    );
    let t0 = clock::now();
    let (mut ticks, mut waits, mut sec_ticks, mut next_report, mut next_delay) = (0u64, 0u64, 0u64, t0 + 1.0, t0);
    let mut lateness: Vec<f64> = Vec::new();
    let mut saved = false;
    while clock::now() - t0 < seconds {
        let t = clock::now();
        if t >= next_delay {
            src.set_audio_delay(delay_ms); // as the pipeline does every couple of seconds (Sound in sync)
            next_delay = t + 2.0;
        }
        match src.next_event() {
            None => {
                println!("[check] the stream ended");
                break;
            }
            Some(Event::Waiting(w)) => {
                waits += 1;
                if waits % 10 == 1 {
                    println!("[check] waiting {:.1} s for {}{}", w.since, w.what, if w.why.is_empty() { String::new() } else { format!(": {}", w.why) });
                }
            }
            Some(Event::Tick(tick)) => {
                ticks += 1;
                sec_ticks += 1;
                if let Some(ts) = tick.t_source {
                    lateness.push(tick.t_arrival - ts);
                }
                if let (false, Some(path)) = (saved, save.as_ref()) {
                    let bgr = tick.frame.to_bgr();
                    let mut f = std::fs::File::create(path).expect("can't write the picture");
                    write!(f, "P6\n{} {}\n255\n", tick.frame.width, tick.frame.height).unwrap();
                    f.write_all(&bgr.as_chunks::<3>().0.iter().flat_map(|p| [p[2], p[1], p[0]]).collect::<Vec<u8>>()).unwrap();
                    println!("[check] the first frame -> {path}");
                    saved = true;
                }
            }
        }
        if clock::now() >= next_report {
            println!(
                "[check] {sec_ticks} frames in the last second; dropped {}, macOS dropped {}, damaged {}; sound {:+.1} ms later than the picture",
                src.dropped(),
                src.os_dropped(),
                src.damaged(),
                src.sound_late_s() * 1000.0
            );
            sec_ticks = 0;
            next_report += 1.0;
        }
    }
    let elapsed = clock::now() - t0;
    lateness.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = lateness.get(lateness.len() / 2).copied().unwrap_or(f64::NAN);
    println!("[check] {ticks} frames in {elapsed:.1} s ({:.2} fps), {waits} waits", ticks as f64 / elapsed);
    if src.is_card() {
        println!(
            "[check] frames reach SmoovPlay {:.1} ms after the card's timestamps (median); the sound comes {:+.1} ms later than the picture",
            med * 1000.0,
            src.sound_late_s() * 1000.0
        );
    }
    let replay: usize = src.read_sound().iter().map(|s| s.pcm.len() / s.channels.max(1) as usize).sum();
    if replay > 0 {
        println!("[check] {replay} sample frames of sound kept for instant replay since the start");
    }
    src.close();
}
