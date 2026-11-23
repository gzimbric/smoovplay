//! Opens the video window with a synthetic moving picture (no decoding), paced the way the pipeline paces it, and
//! prints how frames reached the screen so pacing can be judged.
//!
//!     cargo run --release -p sp-present --example play -- --hz 120 --seconds 10 [--vsync off] [--pacing min]
//!         [--fullscreen] [--drawables 2] [--hdr] [--look cinematic] [--fill panorama] [--scaler bilinear]
//!         [--thread] [--press m@2,down@2.5,f@4] [--shot /path/shot.png@3] [--size 1920x1080] [--detailed]
//!         [--stall 3.5@2] (no frames for 3.5 s from 2 s, for the no-picture screen) [--busy 2@3] (busy() for 2 s)
//!         [--real-keys] (sends --press as real key events through AppKit)
//!
//! --thread runs the stand-in pipeline on its own thread with the main thread in NSApp.run, as the Swift app does.
//! Without it the pipeline runs on the main thread and write() pumps the window's events, as io.py does.
//! The picture has a scrolling checkerboard, a white bar sweeping across every 2 s, a circle, and the frame number in
//! binary along the bottom. --hdr sends HDR10 frames (PQ, BT.2020) with patches at 100, 203, 400, 1000 and 4000 nits.

use std::sync::Arc;
use std::time::Duration;

use objc2::rc::Retained;
use objc2_foundation::MainThreadMarker;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use sp_core::{Frame, Gpu, Overlay, PixelKind, Presenter, SinkReply, clock};
use sp_present::display::{DisplaySettings, FixedSettings, Pacing, Scaler};
use sp_present::gpu::{self, Blend, Load, Pipelines, bytes_of};
use sp_present::looks;
use sp_present::{Fill, Key, VideoPresenter, WindowConfig};

const PATTERN: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; };
struct PatP { float t; uint frame; uint hdr; float w; float h; };

static float3 pq(float nits) {
    float y = clamp(nits / 10000.0, 0.0, 1.0);
    float p = pow(y, 2610.0 / 16384.0);
    return float3(pow((3424.0 / 4096.0 + 2413.0 / 128.0 * p) / (1.0 + 2392.0 / 128.0 * p), 2523.0 / 32.0));
}

fragment float4 pattern(VOut in [[stage_in]], constant PatP& P [[buffer(0)]]) {
    float2 q = in.pos.xy, uv = q / float2(P.w, P.h);
    float3 c = float3(0.15 + uv.x * 0.5, 0.15 + uv.y * 0.4, 0.4);
    float2 cell = floor((q + float2(P.t * 240.0, 0.0)) / 60.0);
    if (fmod(cell.x + cell.y, 2.0) == 0.0) c *= 0.7;
    float bx = fract(P.t * 0.5) * P.w;
    if (abs(q.x - bx) < P.w / 80.0) c = float3(1.0);
    float2 cc = float2(P.w * (0.5 + 0.35 * sin(P.t * 1.3)), P.h * (0.5 + 0.35 * sin(P.t * 2.1)));
    if (length(q - cc) < P.h * 0.08) c = float3(1.0, 0.45, 0.1);
    float bw = P.w / 48.0;
    if (q.y > P.h - bw && q.x < 16.0 * bw) {
        uint bit = uint(q.x / bw);
        c = ((P.frame >> (15u - bit)) & 1u) != 0u ? float3(1.0) : float3(0.08);
    }
    if (P.hdr == 0u) return float4(c, 1.0);
    // HDR10 puts white at 203 nits (BT.2408), with patches along the top at fixed brightness
    float3 l709 = pow(c, 2.2) * 203.0;
    float3 lin = float3(dot(float3(0.627404, 0.329283, 0.043313), l709), dot(float3(0.069097, 0.919540, 0.011362), l709),
                        dot(float3(0.016391, 0.088013, 0.895595), l709));
    float3 o = float3(pq(lin.r).r, pq(lin.g).r, pq(lin.b).r);
    const float nits[5] = {100.0, 203.0, 400.0, 1000.0, 4000.0};
    if (q.y < P.h * 0.12) {
        int k = int(q.x / (P.w / 5.0));
        if (fmod(q.x, P.w / 5.0) > P.w / 50.0) o = pq(nits[clamp(k, 0, 4)]);
    }
    return float4(o, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct PatP {
    t: f32,
    frame: u32,
    hdr: u32,
    w: f32,
    h: f32,
}

struct Args {
    hz: f64,
    seconds: f64,
    vsync: bool,
    pacing: Pacing,
    fullscreen: bool,
    drawables: usize,
    hdr: bool,
    look: Option<String>,
    fill: Fill,
    scaler: Scaler,
    thread: bool,
    press: Vec<(f64, Key, bool)>,
    shots: Vec<(f64, String)>,
    size: (usize, usize),
    detailed: bool,
    /// (from, seconds) with no frames, like a live source that stopped (wait() every 0.1 s)
    stall: Option<(f64, f64)>,
    /// --press goes through real key events (NSEvents in the app's queue, main thread only)
    real_keys: bool,
    /// (at, seconds) in busy() for that long, as a restart does
    busy: Option<(f64, f64)>,
}

fn key_of(name: &str) -> Option<(Key, bool)> {
    Some(match name {
        "up" => (Key::Up, false),
        "down" => (Key::Down, false),
        "left" => (Key::Left, false),
        "right" => (Key::Right, false),
        "enter" => (Key::Enter, false),
        "esc" => (Key::Esc, false),
        "tab" => (Key::Tab, false),
        s if s.len() == 1 => (Key::Char(s.chars().next()?.to_ascii_lowercase()), s.chars().next()?.is_uppercase()),
        _ => return None,
    })
}

fn parse() -> Args {
    let mut a = Args { hz: 60.0, seconds: 8.0, vsync: true, pacing: Pacing::Auto, fullscreen: false, drawables: 3,
                       hdr: false, look: None, fill: Fill::Fit, scaler: Scaler::Fsr, thread: false, press: Vec::new(),
                       shots: Vec::new(), size: (1920, 1080), detailed: false, stall: None,
                       real_keys: false, busy: None };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |i: &mut usize| -> String {
        *i += 1;
        argv.get(*i).cloned().unwrap_or_default()
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--hz" => a.hz = next(&mut i).parse().unwrap(),
            "--seconds" => a.seconds = next(&mut i).parse().unwrap(),
            "--vsync" => a.vsync = next(&mut i) != "off",
            "--pacing" => a.pacing = match next(&mut i).as_str() {
                "min" => Pacing::MinDuration,
                "immediate" => Pacing::Immediate,
                "steady" => Pacing::Steady,
                _ => Pacing::Auto,
            },
            "--fullscreen" => a.fullscreen = true,
            "--drawables" => a.drawables = next(&mut i).parse().unwrap(),
            "--hdr" => a.hdr = true,
            "--look" => a.look = Some(next(&mut i)),
            "--fill" => a.fill = Fill::from_key(&next(&mut i)).expect("fit, zoom, stretch or panorama"),
            "--scaler" => a.scaler = Scaler::from_key(&next(&mut i)),
            "--thread" => a.thread = true,
            "--detailed" => a.detailed = true,
            "--real-keys" => a.real_keys = true,
            "--busy" => {
                let s = next(&mut i);
                let (d, t) = s.split_once('@').unwrap();
                a.busy = Some((t.parse().unwrap(), d.parse().unwrap()));
            }
            "--stall" => {
                let s = next(&mut i);
                let (t, d) = s.split_once('@').map_or((s.as_str(), "3.5"), |(d, t)| (t, d));
                a.stall = Some((t.parse().unwrap(), d.parse().unwrap()));
            }
            "--size" => {
                let s = next(&mut i);
                let (w, h) = s.split_once('x').unwrap();
                a.size = (w.parse().unwrap(), h.parse().unwrap());
            }
            "--press" => {
                for item in next(&mut i).split(',') {
                    let (k, t) = item.split_once('@').unwrap();
                    let (key, shift) = key_of(k).expect("a key: a letter, up, down, left, right, enter, esc, tab");
                    a.press.push((t.parse().unwrap(), key, shift));
                }
            }
            "--shot" => {
                let s = next(&mut i);
                let (p, t) = s.rsplit_once('@').unwrap_or((&s, "3"));
                a.shots.push((t.parse().unwrap(), p.to_string()));
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    a
}

struct Pattern {
    pipes: Pipelines,
    lib: Retained<sp_core::gpu::Library>,
}

impl Pattern {
    fn draw(&mut self, f: &Frame, frame: u64, t: f64, hdr: bool) {
        let fmt = f.kind.format();
        let ps = self.pipes.get_in(&self.lib, 1, "pattern", fmt, Blend::Replace).unwrap();
        let p = PatP { t: t as f32, frame: frame as u32, hdr: hdr as u32, w: f.width as f32, h: f.height as f32 };
        let cmd = Gpu::shared().queue.commandBuffer().unwrap();
        gpu::pass(&cmd, &f.texture, Load::DontCare, &ps, gpu::whole(&f.texture), &[], &[(bytes_of(&p), 0)]);
        cmd.commit();
    }
}

fn sleep_until(t: f64) {
    loop {
        let left = t - clock::now();
        if left <= 0.0 {
            return;
        }
        if left > 0.002 {
            std::thread::sleep(Duration::from_secs_f64(left - 0.0015));
        } else {
            std::hint::spin_loop();
        }
    }
}

/// The stand-in pipeline. It writes a frame every 1/hz on the clock, as the scheduler hands them over.
fn play(p: &mut VideoPresenter, a: &Args) {
    let kind = if a.hdr { PixelKind::Rgba16f } else { PixelKind::Bgra8 };
    let frames: Vec<Frame> = (0..4).map(|_| Frame::new(a.size.0, a.size.1, kind)).collect();
    let lib = Gpu::shared().library(PATTERN).unwrap();
    let mut pat = Pattern { pipes: Pipelines::new().unwrap(), lib };
    let period = 1.0 / a.hz;
    let t0 = clock::now() + 0.3;
    let warm = 1.0; // seconds left out of the numbers while the window opens
    let mut from = None;
    let mut presses = a.press.clone();
    let mut shots = a.shots.clone();
    let timing = p.timing();
    let mut k: u64 = 0;
    loop {
        let target = t0 + k as f64 * period;
        sleep_until(target);
        let el = clock::now() - t0;
        if el >= warm && from.is_none() {
            from = Some(timing.frames());
        }
        let real = a.real_keys;
        presses.retain(|(t, key, shift)| {
            if el >= *t {
                match (real, MainThreadMarker::new()) {
                    (true, Some(mtm)) => sp_present::post_key(mtm, *key, *shift),
                    _ => p.press(*key, *shift),
                }
                false
            } else {
                true
            }
        });
        shots.retain(|(t, path)| {
            // started a second early with a 1 s delay, so a shot can land while write() isn't being called (busy())
            if el >= *t - 1.0 {
                let n = p.window_number();
                let _ = std::process::Command::new("screencapture").args(["-x", "-o", "-T", "1", &format!("-l{n}"), path]).spawn();
                println!("[play] screenshot of window {n} -> {path}");
                false
            } else {
                true
            }
        });
        if let Some((at, len)) = a.busy
            && el >= at && el < at + 0.1 {
                let t = clock::now();
                let got = p.busy("Restarting", move || {
                    std::thread::sleep(Duration::from_secs_f64(len));
                    42
                });
                println!("[play] busy() returned {got} after {:.2} s", clock::now() - t);
                k = ((clock::now() - t0) / period).ceil() as u64;
                continue;
            }
        if let Some((from, len)) = a.stall
            && el >= from && el < from + len {
                let w = sp_core::Waiting { since: el - from, what: "the PS5".into(), why: "chiaki-ng says: reconnecting".into() };
                if p.wait(&w) == SinkReply::Quit {
                    break;
                }
                sleep_until(clock::now() + 0.1);
                k = ((clock::now() - t0) / period).ceil() as u64;
                continue;
            }
        let f = &frames[(k % 4) as usize];
        pat.draw(f, k, k as f64 * period, a.hdr);
        let shown = timing.presents(timing.frames().saturating_sub(a.hz as usize));
        let rate = if shown.shown.mean > 0.0 { 1000.0 / shown.shown.mean } else { 0.0 };
        let overlay = if a.detailed {
            Overlay { text: format!("{:.0} fps target  on screen {rate:.1} fps  p95 {:.1} ms\nframe {k}", a.hz, shown.shown.p95),
                      minimal: false, ..Default::default() }
        } else {
            Overlay {
                parts: vec![(format!("{rate:.0} fps"), "main".into()), (format!("target {:.0}", a.hz), "dim".into()),
                            (format!("p95 {:.1} ms", shown.shown.p95), if shown.shown.p95 > period * 1500.0 { "warn" } else { "dim" }.into())],
                health: if shown.dropped > 0 { "tight".into() } else { "good".into() },
                minimal: true,
                ..Default::default()
            }
        };
        match p.write(f, Some(&overlay)) {
            SinkReply::Quit => break,
            SinkReply::Continue => {}
            other => println!("[play] {other:?}"),
        }
        k += 1;
        if el > a.seconds && presses.is_empty() && shots.is_empty() {
            break;
        }
    }
    // let the last frames reach the screen before reading their times
    let end = clock::now() + 0.2;
    while clock::now() < end {
        let _ = p.wait(&sp_core::Waiting { since: 0.0, what: String::new(), why: String::new() });
        std::thread::sleep(Duration::from_millis(5));
    }
    report(p, a, from.unwrap_or(0));
}

fn report(p: &VideoPresenter, a: &Args, from: usize) {
    let s = p.screen().clone();
    let st = p.timing().presents(from);
    let refresh = s.refresh_min * 1e3;
    println!("[play] {:.0} fps, vsync {}, pacing {:?}{}, {} drawables, {}, {}{} on {} ({})",
        a.hz, if a.vsync { "on" } else { "off" }, p.pacing(), if a.pacing == Pacing::Auto { " (auto)" } else { "" },
        a.drawables, if a.thread { "pipeline thread" } else { "main thread" }, if a.fullscreen { "fullscreen " } else { "window " },
        p.label(), s.name,
        if s.variable_refresh() { format!("variable {:.0}-{:.0} Hz", 1.0 / s.refresh_max, 1.0 / s.refresh_min) } else { format!("{:.0} Hz fixed", 1.0 / s.refresh_min) });
    println!("  write() intervals:   {}", st.written);
    println!("  on-screen intervals: {}", st.shown);
    println!("  write -> on screen:  {}", st.latency);
    let mut hist = std::collections::BTreeMap::new();
    for v in &st.intervals {
        *hist.entry((v / refresh).round() as i64).or_insert(0usize) += 1;
    }
    let total = st.intervals.len().max(1) as f64;
    let h: Vec<String> = hist.iter().map(|(k, n)| format!("{k}x: {:.1}%", 100.0 * *n as f64 / total)).collect();
    println!("  intervals in refreshes of {refresh:.2} ms: {}  (ideal {:.2} refreshes each)", h.join("  "), 1000.0 / a.hz / refresh);
    let off: usize = st.intervals.iter().filter(|v| (**v - 1000.0 / a.hz).abs() > refresh * 0.5).count();
    println!("  frames shown {} of {}, not shown {}, intervals more than half a refresh off 1/hz: {:.1}%",
        st.frames - st.dropped, st.frames, st.dropped, 100.0 * off as f64 / total);
    for (k, v) in p.timing().gpu() {
        println!("  GPU {k}: {v}");
    }
}

fn main() {
    let a = parse();
    let mtm = MainThreadMarker::new().expect("run on the main thread");
    let look = a.look.as_ref().map(|id| Arc::new(looks::builtin().iter().find(|(k, _)| k == id).expect("a built-in look id").1.clone()));
    let display = DisplaySettings { hz: a.hz, vsync: a.vsync, pacing: a.pacing, fullscreen: a.fullscreen, look, fill: a.fill,
                                    scaler: a.scaler, ..Default::default() };
    let cfg = WindowConfig { drawables: a.drawables, ..Default::default() };
    let settings = FixedSettings { display, rev: 0, interp: true };
    let mut p = sp_present::open(mtm, a.size.0, a.size.1, cfg, Box::new(settings)).expect("window");
    if a.thread {
        let worker = std::thread::spawn(move || {
            play(&mut p, &a);
            p.close();
        });
        sp_present::run(mtm);
        worker.join().unwrap();
    } else {
        play(&mut p, &a);
        p.close();
    }
}
