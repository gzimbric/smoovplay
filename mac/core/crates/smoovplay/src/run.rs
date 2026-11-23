//! `smoovplay [run.py's flags]` is run.py main() on the macOS core. The source is opened and RIFE built on the main
//! thread. With --display the window opens there too, the pipeline moves to a thread of its own (QoS
//! user-interactive), and the main thread runs AppKit's event loop until the window closes. That's sp-present's
//! layout for a pipeline off the main thread. Headless runs stay on the main thread.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use objc2_foundation::MainThreadMarker;
use sp_core::{Frame, Gpu, Presenter, SourceInfo, Tick};
use sp_media::{HdrOut, HdrSettings, SharedTestMode, TestMode};
use sp_present::WindowConfig;
use sp_rife::{Options as RifeOptions, RifeInterpolator};
use sp_sched::pipeline::Status;
use sp_sched::report::{self, ReportHeader, SessionInfo};
use sp_sched::streamcheck::StreamCheck;
use sp_sched::{Action, Exit, FrameDiff, LiveSettings, Pipeline, RunOptions, Stats, make_detector};
use sp_settings::autodetect::{SHARE, LAPTOP_SHARE, auto_scale, saved_full_ms};
use sp_settings::looks::{Library, LookState};
use sp_settings::remote::Remote;
use sp_settings::settings::{LookChoices, spec, specs};
use sp_settings::{Map, Settings, Value};

use crate::adapt::{
    Ctx, Cfg, LiveAdapter, Looks, MenuAdapter, Rows, RunRules, SharedLooks, SharedSettings, lock, lock_any, py_float,
};
use crate::args::{self, Args};
use crate::img;
use crate::sources::{self, Limit, Opened};
use crate::system;
use crate::win::{self, Control, Window};

/// What showing each frame costs on top of RIFE's time, in ms, for Auto motion detail and Auto-detect. The Python
/// core's MAC_PRESENT_MS is 3.0, for its copy through the CPU to OpenGL. Measured through the real window on an M5
/// Max with a 6016x3384 display, write() takes 0.08 ms of CPU (p50). FSR 1 takes 0.45 ms of GPU into the 3200x1800
/// window and 1.17 ms fullscreen. recommend() already counts about 0.24 ms of FSR there, so 1.0 ms covers fullscreen.
pub const PRESENT_MS: f64 = 1.0;

/// When the video loop is stuck this long while playing, the log gets where each thread is.
const WATCHDOG_S: f64 = 4.0;
/// The same when no picture has come from the source for this long, once per stall.
const STARVED_S: f64 = 8.0;

/// The finished sessions and the one playing, for the reports written while it plays (one lock for both, so a
/// session is never counted twice).
#[derive(Default)]
struct Reports {
    done: Vec<(SessionInfo, Stats)>,
    /// The playing pipeline's status, what it plays with, and when it started.
    live: Option<(Arc<Status>, Cfg, f64)>,
}

fn session_info(cfg: &Cfg) -> SessionInfo {
    SessionInfo { engine: cfg.engine.clone(), model: cfg.model.clone(), motion: cfg.motion, vfg_quality: String::new() }
}

fn upscaler_name(settings: &Settings) -> String {
    let sp = spec("scaler").unwrap();
    settings.get("scaler").map_or(String::new(), |v| sp.name(v, settings.look_choices()))
}

/// Test mode's report from the finished sessions and the one playing, as run.py results_now() writes it.
fn results_now(path: &Path, reports: &Mutex<Reports>, settings: &SharedSettings, size: &str, gpu: &str) -> bool {
    let sessions: Vec<(SessionInfo, Stats)> = {
        let r = lock_any(reports);
        let mut v = r.done.clone();
        if let Some((st, cfg, _)) = &r.live {
            v.push((session_info(cfg), st.stats()));
        }
        v
    };
    let upscaler = upscaler_name(&lock(settings)); // after the stats, as the pipeline takes the settings under them
    let header = ReportHeader { stamp: system::strftime("%Y-%m-%d %H:%M"), size: size.into(), gpu: gpu.into(), upscaler };
    let refs: Vec<(SessionInfo, &Stats)> = sessions.iter().map(|(i, s)| (i.clone(), s)).collect();
    report::write_results(path, &header, &refs)
}

/// The watchdog's "where each thread is". It runs macOS's sample tool on this process and trims the call graph.
fn where_threads(headline: &str) {
    let mut lines = vec![format!("{headline} Where each thread is:")];
    let out = std::process::Command::new("/usr/bin/sample")
        .args([&std::process::id().to_string(), "1", "-mayDie"])
        .output();
    match out {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            let mut on = false;
            let mut per_thread = 0;
            for l in text.lines() {
                if l.starts_with("Call graph:") {
                    on = true;
                    continue;
                }
                if on && (l.starts_with("Total number in stack") || l.trim().is_empty()) {
                    break;
                }
                if !on {
                    continue;
                }
                let t = l.trim_start();
                let depth = l.len() - t.len();
                if depth <= 4 {
                    per_thread = 0; // a thread's line
                }
                per_thread += 1;
                if per_thread <= 14 {
                    lines.push(format!("  {}", l.trim_end()));
                }
                if lines.len() > 160 {
                    break;
                }
            }
        }
        Err(e) => lines.push(format!("  (sample: {e})")),
    }
    println!("{}", system::mask_home(&lines.join("\n")));
}

fn watchdog(reports: Arc<Mutex<Reports>>, stop: Arc<AtomicBool>) {
    let (mut stuck, mut starved) = (false, false);
    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_secs(1));
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let live = lock_any(&reports).live.as_ref().map(|(st, _, t0)| (st.clone(), *t0));
        let now = sp_core::clock::now();
        let age = live.as_ref().map_or(0.0, |(st, t0)| now - st.beat().max(*t0));
        let w = live.as_ref().and_then(|(st, _)| st.waiting());
        if age >= WATCHDOG_S && !stuck {
            stuck = true;
            where_threads(&format!("[watchdog] the video loop hasn't moved for {age:.0} s."));
        } else if age < WATCHDOG_S && stuck {
            stuck = false;
            println!("[watchdog] the video loop is going again");
        }
        match &w {
            Some(w) if w.since >= STARVED_S && !starved => {
                starved = true;
                let why = w.why.replace('\n', " ");
                where_threads(&format!(
                    "[watchdog] no picture from {} for {:.0} s{}",
                    w.what,
                    w.since,
                    if why.is_empty() { ".".to_string() } else { format!(" ({why}).") }
                ));
            }
            None if starved => {
                starved = false;
                println!("[watchdog] pictures again");
            }
            _ => {}
        }
    }
}

/// How well the size check read this game, written to the log and beside the results (run.py write_stream_check()).
fn write_stream_check(check: &StreamCheck, results: Option<&str>) {
    let lines = check.report();
    if lines.is_empty() {
        return;
    }
    println!("{}", lines.join("\n"));
    let Some(results) = results else { return };
    let out = PathBuf::from(results);
    let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let file = if name.contains("results-") { name.replacen("results-", "stream-check-", 1) } else { "stream-check.txt".into() };
    let path = out.with_file_name(file);
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let text = format!("SmoovPlay stream check, {}\n{}\n", system::strftime("%Y-%m-%d %H:%M"), lines.join("\n"));
    match std::fs::write(&path, text) {
        Ok(()) => println!("[results] stream check -> {}", path.display()),
        Err(e) => println!("[results] couldn't write {}: {e}", path.display()),
    }
}

/// --save-worst. The previous real frame and RIFE's go on top, the actual in-between and the next real frame below,
/// each at half size and labelled.
fn save_worst(path: &str, st: &Stats) {
    let Some(w) = &st.worst else { return };
    if w.images.len() != 4 {
        println!("[run] the worst frame isn't SDR: not saved");
        return;
    }
    let names = ["previous real".to_string(), format!("RIFE ({:.1} dB)", w.psnr), "actual in-between".into(), "next real".into()];
    let tiles: Vec<img::Image> = w
        .images
        .iter()
        .zip(&names)
        .map(|(bgr, name)| {
            let full = img::Image { w: w.width, h: w.height, bgr: bgr.clone() };
            let mut t = img::resize_area(&full, w.width / 2, w.height / 2);
            img::label(&mut t, 15.0, 40.0, name);
            t
        })
        .collect();
    let out = img::vconcat(&img::hconcat(&tiles[0], &tiles[1]), &img::hconcat(&tiles[2], &tiles[3]));
    if img::write(Path::new(path), &out) {
        println!("[run] worst frame (tick {}) -> {path}", w.tick);
    } else {
        println!("[run] couldn't write {path}");
    }
}

fn build(w: usize, h: usize, cfg: &Cfg, hdr: bool) -> Result<RifeInterpolator, String> {
    RifeInterpolator::new(w, h, RifeOptions { version: cfg.model.clone(), scale: cfg.motion as f32, fp16: true, hdr, model_dir: None })
        .map_err(|e| e.to_string())
}

/// How far the main display can go above SDR white right now (1 for not at all).
fn edr_potential(mtm: MainThreadMarker) -> f64 {
    objc2_app_kit::NSScreen::mainScreen(mtm).map_or(1.0, |s| s.maximumPotentialExtendedDynamicRangeColorComponentValue())
}

/// Everything a run plays with once it's set up. With --display it moves to the pipeline's thread.
struct Play {
    a: Args,
    opened: Opened,
    settings: SharedSettings,
    ctx: Arc<Ctx>,
    rules: Arc<RunRules>,
    test: Option<SharedTestMode>,
    cfg: Cfg,
    interp: RifeInterpolator,
    window: Option<Window>,
    hdr_out: Option<&'static str>,
    gpu: String,
    replay: Option<Arc<sp_replay::Recorder>>,
}

/// What instant replay's "on" message says about keys. The Mac app holds Control-Option-R and B from anywhere.
fn play_hotkeys(a: &Args) -> Option<String> {
    a.control.then(|| "Control-Option-R and Control-Option-B from anywhere".to_string())
}

/// "Looks in clips". The look the window shows is drawn again at the stream's size on the clip's frames (io.py
/// baked()), by a renderer of its own on the pipeline's thread, because the window's is busy drawing the screen.
struct ClipLook {
    r: sp_present::renderer::Renderer,
    luts: Arc<sp_present::looks::LutStore>,
    look: Arc<Mutex<Option<Arc<sp_present::looks::Look>>>>,
    n: u64,
}

// Metal objects are thread-safe, and it's used from one thread at a time (the pipeline's, through the recorder).
unsafe impl Send for ClipLook {}

fn clip_look(look: Arc<Mutex<Option<Arc<sp_present::looks::Look>>>>, _w: usize, _h: usize) -> Option<sp_replay::recorder::LookFn> {
    let folder = sp_settings::looks::looks_dir().join("luts");
    let luts = Arc::new(sp_present::looks::LutStore::new(Some(folder)));
    let r = match sp_present::renderer::Renderer::new(luts.clone()) {
        Ok(r) => r,
        Err(e) => {
            println!("[looks] Looks in clips unavailable ({e})");
            return None;
        }
    };
    let mut c = ClipLook { r, luts, look, n: 0 };
    Some(Box::new(move |f: &Frame| -> Result<Frame, String> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
        let c = &mut c; // the whole ClipLook, which is Send, not its fields one by one
        let Some(look) = lock_any(&c.look).clone() else { return Ok(f.clone()) };
        if f.kind != sp_core::PixelKind::Bgra8 {
            return Ok(f.clone());
        }
        let Some(plan) = c.r.looks.plan(&look, &c.luts) else {
            return match &c.r.looks.broken {
                Some(why) => Err(why.clone()),
                None => Ok(f.clone()),
            };
        };
        c.n += 1;
        let out = Frame::new(f.width, f.height, sp_core::PixelKind::Bgra8);
        let cmd = Gpu::shared().queue.commandBuffer().ok_or("no Metal command buffer")?;
        c.r.bake(&cmd, &plan, &f.texture, &out.texture, c.n).map_err(|e| e.to_string())?;
        cmd.commit();
        Ok(out)
    }))
}

pub fn main(argv: &[String]) -> i32 {
    let mut a = args::parse(argv);
    if a.list_sources {
        return crate::tools::sources_main(a.json, false);
    }
    if a.list_devices {
        let mut args = vec!["--list".to_string()];
        if a.json {
            args.push("--json".into());
        }
        return crate::tools::capture_main(&args);
    }
    let gpu = Gpu::shared().name();
    println!("[run] RIFE runs on Apple's GPU (Metal and MPSGraph), {gpu}");
    a.chiaki_decoder = "videotoolbox".into(); // cuda, d3d11va and sw all mean VideoToolbox here
    if a.out.is_some() {
        args::error("--out (writing the result to a video file) isn't in the Mac core yet");
    }
    if a.fp32 {
        args::error("--fp32: the Mac's RIFE engine runs in half precision only");
    }

    let kind: Option<String> = a.source.clone().or_else(|| {
        if a.chiaki {
            Some("chiaki".into())
        } else if a.capture.is_some() {
            Some("capture".into())
        } else if a.video.is_some() {
            Some("file".into())
        } else {
            None
        }
    });
    let picked = kind.clone().unwrap_or_default();
    let kind = kind.unwrap_or_default();
    let mut test: Option<SharedTestMode> =
        (matches!(kind.as_str(), "chiaki" | "capture" | "tap") || (kind == "file" && a.raw)).then(|| TestMode::shared(a.test_mode));
    if a.test_mode && test.is_none() {
        args::error("--test-mode needs Remote Play, a capture card, the frame tap, or a video file with --raw");
    }
    let opened = match kind.as_str() {
        "tap" | "chiaki" => sources::open(&a, &kind, &test),
        "file" if a.video.is_some() => sources::open(&a, "file", &test),
        "capture" => {
            if a.capture.is_none() {
                args::error("--source capture needs --capture N (a device) or --capture FILE");
            }
            sources::open(&a, "capture", &test)
        }
        "window" => Err((
            "[window] on a Mac the SmoovPlay app copies the window (ScreenCaptureKit) and sends it through the frame tap: \
             --source tap --tap smoovplay-window"
                .to_string(),
            1,
        )),
        _ => args::error("give a video file, --capture N or --chiaki"),
    };
    let mut opened = match opened {
        Ok(o) => o,
        Err((msg, rc)) => {
            eprintln!("{msg}");
            return rc;
        }
    };
    if picked == "tap" && !opened.info.test_mode {
        test = None; // no test mode for an app that sends only new pictures
    }
    let kind = opened.kind.clone();
    if let Some(m) = a.max_ticks
        && kind != "file"
    {
        let inner = std::mem::replace(&mut opened.src, Box::new(EmptySource(opened.info.clone())));
        opened.src = Box::new(Limit { inner, left: m });
    }
    println!("[run] source: {}", opened.info.name);

    // The looks library gives the "Which look" setting's choices (the user's looks too) and the window's looks and
    // LUTs.
    let library = if a.display { Some(Library::open_default()) } else { None };
    let choices = library.as_ref().map_or_else(LookChoices::builtin, |l| l.choices());
    let settings_path = a.settings_path();
    let mut settings = match &settings_path {
        None => Settings::new(&Map::new(), None, choices),
        Some(p) => Settings::load(p, choices),
    };
    let overrides: [(&str, Option<Value>); 15] = [
        ("hz", a.hz.map(Value::Int)),
        ("scaler", a.scaler.clone().map(Value::Str)),
        ("vsr_quality", a.vsr_quality.map(Value::Int)),
        ("fsr_sharpness", a.fsr_sharpness.map(Value::Float)),
        ("engine", if a.trt { Some(Value::from("trt")) } else { a.engine.clone().map(Value::Str) }),
        ("interp", a.off.then_some(Value::Bool(false))),
        ("vsync", a.vsync.map(Value::Bool)),
        ("fullscreen", a.fullscreen.map(Value::Bool)),
        ("minimize_chiaki", a.minimize_chiaki.map(Value::Bool)),
        ("simulate_30", a.test_mode.then_some(Value::Bool(true))),
        ("steady_hud", a.steady_hud.map(Value::Bool)),
        ("vfg_quality", a.vfg_quality.clone().map(Value::Str)),
        ("present", a.present.clone().map(Value::Str)),
        ("hdr", a.hdr.clone().map(Value::Str)),
        ("fill", a.fill.clone().map(Value::Str)),
    ];
    for (k, v) in overrides {
        if let Some(v) = v
            && let Err(e) = settings.override_value(k, v)
        {
            args::error(&e.to_string());
        }
    }
    // --model is one of the settings' choices, or another RIFE version for testing
    let model_choices = spec("model").unwrap().choices(settings.look_choices());
    let test_model = a.model.clone().filter(|m| !Value::from(m.as_str()).py_in(&model_choices));
    if let Some(m) = &a.model
        && test_model.is_none()
    {
        let _ = settings.override_value("model", m.as_str());
    }
    if let Some(t) = &test {
        let t = t.clone();
        settings.watch(move |s, key| {
            if key == "simulate_30" {
                lock_any(&t).set_on(s.get_bool("simulate_30"));
            }
        });
    }
    if settings.get_i64("hz") > 60 && !a.display {
        if a.hz.is_some() {
            args::error("--hz 90/120 needs --display, and doesn't write --out yet");
        }
        println!("[run] output {} Hz needs --display without --out; 60 Hz this run", settings.get_i64("hz"));
        let _ = settings.override_value("hz", 60i64);
    }

    // HDR (the "hdr" setting) is only for a PS5 Remote Play stream in HDR, with the setting on. "sdr8" is tone-mapped
    // to SDR. "pq16" keeps HDR10 for the window (EDR), when the display can show it.
    let mtm = MainThreadMarker::new();
    let mut hdr_out: Option<&'static str> = None;
    if kind == "chiaki" && settings.get_str("hdr") != "off" && opened.hdr_stream {
        hdr_out = Some("sdr8");
        if settings.get_str("hdr") == "hdr10" {
            let edr = mtm.map_or(1.0, edr_potential);
            if !a.display {
                println!("[run] HDR output needs the video window; showing HDR in SDR");
            } else if edr > 1.0 {
                hdr_out = Some("pq16");
                println!("[run] HDR output: HDR10 through EDR, to a display that goes {edr:.1}x past SDR white");
            } else {
                println!("[run] HDR output: this display can't go past SDR white (no EDR headroom), so HDR is shown in SDR");
            }
        }
    }
    let hdr10 = hdr_out == Some("pq16");

    if settings.get_str("engine") == "trt" {
        println!("[run] TensorRT is NVIDIA's; RIFE runs on Apple's GPU (Metal)");
    }
    let vfg_why = "No NVIDIA graphics card was found.".to_string(); // framegen/vfg.py vfg_unavailable() on a Mac
    let rife = crate::adapt::default_model();
    if settings.get_str("model") == "vfg" {
        let name = spec("model").unwrap().name(&Value::from(rife.as_str()), settings.look_choices());
        println!("[run] {vfg_why} Using {name}.");
        let _ = settings.override_value("model", rife.as_str()); // this run only
    }

    // Auto motion detail also goes by Auto-detect's last time at full detail on this GPU, plus the window's cost
    let card_full_ms = settings_path.as_ref().and_then(|p| saved_full_ms(p, &gpu)).map(|ms| ms + PRESENT_MS);
    let card_share = if gpu.to_lowercase().contains("laptop") { LAPTOP_SHARE } else { SHARE };
    let hz = settings.get_i64("hz");
    let (w, h) = (opened.w, opened.h);
    if a.scale.is_none()
        && settings.get_str("motion") == "auto"
        && let Some(ms) = card_full_ms
        && auto_scale(w as i64, h as i64, hz, Some(ms), card_share) < auto_scale(w as i64, h as i64, hz, None, card_share)
    {
        println!("[run] motion detail Auto: half, as Auto-detect timed full detail at {ms:.1} ms on this graphics card, more than {hz} fps leaves");
    }
    let ctx = Arc::new(Ctx { w, h, scale: a.scale, test_model: test_model.clone(), card_full_ms, card_share, vfg_why: vfg_why.clone(), hdr10 });
    let cfg = ctx.session_config(&settings);
    println!("[run] {w}x{h}  {}  output {hz} Hz", Ctx::describe(&cfg));
    let interp = match build(w, h, &cfg, hdr10) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("[run] couldn't start RIFE: {e}");
            return 1;
        }
    };

    // The rows the window and the launcher grey out, and the choices they limit (run.py on a Mac).
    let mut rows = Rows::default();
    for s in specs() {
        if let Some(why) = s.unused_with(&picked) {
            rows.unavailable.insert(s.key.into(), why);
        }
    }
    if test.is_none() {
        let why = if picked == "window" || picked == "tap" {
            "Not with window capture yet."
        } else {
            "Video files already play as a 30 fps game."
        };
        rows.unavailable.insert("simulate_30".into(), why.into());
    }
    if kind != "chiaki" {
        rows.unavailable.insert("hdr".into(), "Only used with PS5 Remote Play.".into());
    }
    rows.unavailable.insert("engine".into(), "On a Mac, RIFE runs on Apple's GPU (Metal).".into());
    rows.unavailable.insert("model".into(), "NVIDIA VFG needs an NVIDIA RTX card.".into());
    rows.unavailable.insert("present".into(), "On a Mac, the window always presents through Metal.".into());
    if hdr10 {
        for k in ["look_on", "look", "look_clips"] {
            rows.unavailable.insert(k.into(), "Not with HDR output yet.".into());
        }
        rows.unavailable.insert("present".into(), "HDR output always uses the low-delay presentation.".into());
    }
    if let Some(sc) = a.scale {
        rows.unavailable.insert("motion".into(), format!("Set to {} by --scale on the command line.", py_float(sc)));
    }
    if let Some(m) = &test_model {
        rows.unavailable.insert("model".into(), format!("RIFE {m}, set by --model on the command line, for testing."));
    }
    rows.choices.insert("engine".into(), vec![Value::from("auto"), Value::from("torch")]);
    rows.choices.insert(
        "model".into(),
        model_choices.iter().filter(|m| m.as_str() != Some("vfg") || vfg_why.is_empty()).cloned().collect(),
    );
    let menu_open = Arc::new(AtomicBool::new(false));
    let rules = Arc::new(RunRules { ctx: ctx.clone(), rows: Arc::new(Mutex::new(rows)), menu_open: menu_open.clone() });

    if let Some(c) = &opened.chiaki {
        let s = hdr_out.map(|o| HdrSettings::new(if o == "pq16" { HdrOut::Pq16 } else { HdrOut::Sdr8 }, None));
        lock_any(c).set_hdr(s, "");
    }

    // Every change bumps the window's revision, so its menu and display settings are read again.
    let rev = Arc::new(AtomicU64::new(0));
    {
        let r = rev.clone();
        settings.watch(move |_, _| {
            r.fetch_add(1, Ordering::SeqCst);
        });
    }
    let shared: SharedSettings;
    let mut window: Option<Window> = None;
    if a.display {
        let Some(mtm) = mtm else {
            eprintln!("[display] the video window has to be opened from the main thread");
            return 1;
        };
        let library = library.unwrap();
        let folder = library.folder.clone();
        let state = LookState::new(&mut settings);
        let looks: SharedLooks = Arc::new(Mutex::new(Looks { state, library }));
        let control = a.control.then(|| {
            let remote = Remote::new(&mut settings, ArcRules(rules.clone()), std::io::stdout());
            Control::new(remote, looks.clone())
        });
        shared = Arc::new(Mutex::new(settings));
        let model = MenuAdapter::new(shared.clone(), rules.clone(), looks, rev);
        let shown_look = model.shown.clone();
        let cfgw = WindowConfig { looks_folder: Some(folder), ..Default::default() };
        let p = match sp_present::open(mtm, w, h, cfgw, Box::new(model)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[display] couldn't open the video window: {e}");
                return 1;
            }
        };
        if let Some(c) = &control {
            win::read_commands(c.remote.sender(), std::io::stdin());
        }
        window = Some(Window { p, settings: shared.clone(), control, menu_open, toasts: Arc::new(Mutex::new(Vec::new())), shown_look });
    } else {
        shared = Arc::new(Mutex::new(settings));
    }

    // Instant replay (sp-replay, framegen/replay.py) runs only while playing on screen. It lasts across restarts, so
    // a restart for a setting doesn't lose what's held.
    let replay = window.as_ref().map(|wd| {
        let st = lock(&shared);
        let toasts = wd.toasts.clone();
        let notify: sp_replay::recorder::Notify = Arc::new(move |text: &str, secs: f64| lock_any(&toasts).push((text.to_string(), secs)));
        let opts = sp_replay::Options {
            length_s: st.get_i64("replay").max(0) as u32,
            clip: sp_replay::Kind::parse(st.get_str("replay_save")).unwrap_or(sp_replay::Kind::Smoovplay),
            layout: sp_replay::Layout::parse(st.get_str("replay_layout")).unwrap_or(sp_replay::Layout::Split),
            notify: Some(notify),
            size: Some((w, h)),
            mark_dir: Some(sp_replay::saving_dir()),
            hotkeys: play_hotkeys(&a),
            ..Default::default()
        };
        drop(st);
        let r = Arc::new(sp_replay::Recorder::new(opts));
        r.warm_up(); // its encoders open now, not during play
        if let Some(sound) = opened.sound.take() {
            r.attach_sound(Some(sound));
        }
        let (r2, shown, (sw, sh)) = (r.clone(), wd.shown_look.clone(), (w, h));
        let mut st = lock(&shared);
        let apply = move |s: &Settings, key: &str| match key {
            "replay" => r2.set_length(s.get_i64("replay").max(0) as u32),
            "replay_save" => r2.set_clip(sp_replay::Kind::parse(s.get_str("replay_save")).unwrap_or(sp_replay::Kind::Smoovplay)),
            "replay_layout" => r2.set_layout(sp_replay::Layout::parse(s.get_str("replay_layout")).unwrap_or(sp_replay::Layout::Split)),
            "look_clips" => r2.set_look(if s.get_bool("look_clips") { clip_look(shown.clone(), sw, sh) } else { None }),
            _ => {}
        };
        apply(&st, "look_clips");
        st.watch(apply);
        r
    });

    let play = Play { a, opened, settings: shared, ctx, rules, test, cfg, interp, window, hdr_out, gpu, replay };
    match (play.window.is_some(), mtm) {
        (true, Some(mtm)) => {
            let worker = std::thread::Builder::new().name("pipeline".into()).spawn(move || run_sessions(play));
            let worker = match worker {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("[run] couldn't start the pipeline's thread: {e}");
                    return 1;
                }
            };
            sp_present::run(mtm); // until the window closes
            worker.join().unwrap_or(1)
        }
        _ => run_sessions(play),
    }
}

/// The source that's swapped out while --max-ticks wraps the real one.
struct EmptySource(SourceInfo);

impl sp_core::Source for EmptySource {
    fn info(&self) -> &SourceInfo {
        &self.0
    }
    fn size(&self) -> (usize, usize) {
        (0, 0)
    }
    fn next_event(&mut self) -> Option<sp_core::Event> {
        None
    }
}

/// The Remote's rules are the window's, shared.
struct ArcRules(Arc<RunRules>);

impl sp_settings::remote::Rules for ArcRules {
    fn allowed(&self, s: &Settings, key: &str, value: &Value) -> bool {
        self.0.allowed(s, key, value)
    }
    fn disabled(&self, s: &Settings, key: &str) -> Option<String> {
        self.0.disabled(s, key)
    }
    fn restart_reasons(&self, s: &Settings) -> Vec<String> {
        self.0.restart_reasons(s)
    }
    fn extra(&self, s: &Settings) -> Vec<(String, Value)> {
        sp_settings::remote::Rules::extra(&*self.0, s)
    }
}

/// The session loop (run.py's `while True`) and what follows it.
fn run_sessions(play: Play) -> i32 {
    let Play { a, mut opened, settings, ctx, rules, test, mut cfg, interp, mut window, hdr_out, gpu, replay } = play;
    let (w, h) = (opened.w, opened.h);
    let size = format!("{w}x{h}");
    let live_settings = LiveAdapter(settings.clone());
    let reports = Arc::new(Mutex::new(Reports::default()));
    let stop = Arc::new(AtomicBool::new(false));

    let results_thread = a.results.clone().map(|path| {
        let (r, s, st, size, gpu) = (reports.clone(), settings.clone(), stop.clone(), size.clone(), gpu.clone());
        std::thread::Builder::new()
            .name("results".into())
            .spawn(move || {
                let path = PathBuf::from(path);
                let mut next = sp_core::clock::now() + 10.0;
                while !st.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    if sp_core::clock::now() >= next {
                        next += 10.0;
                        results_now(&path, &r, &s, &size, &gpu);
                    }
                }
            })
            .ok()
    });
    let mut awake = None;
    if window.is_some() {
        let (r, s) = (reports.clone(), stop.clone());
        let _ = std::thread::Builder::new().name("watchdog".into()).spawn(move || watchdog(r, s));
        let qos = system::boost_thread();
        awake = system::Awake::hold();
        println!(
            "[run] scheduling: QoS user-interactive {}, GPU priority normal, screen kept on {}",
            if qos { "on" } else { "off" },
            if awake.is_some() { "yes" } else { "no" }
        );
    }

    let mut check = StreamCheck::default();
    let mut interp = Some(interp);
    let mut stats;
    let mut rc = 0;
    loop {
        lock_any(&rules.rows).started = Some(cfg.clone());
        println!("[run] playing"); // the launcher's cue that the video is up again after a restart
        if let Some(wd) = &window {
            wd.publish(); // the restart list starts empty again
        }
        let label = ctx.label_of(&cfg, hdr_out);
        let mut pipe = Pipeline::new(interp.take().unwrap(), Some(make_detector(&opened.info, a.dup_threshold)));
        lock_any(&reports).live = Some((pipe.status(), cfg.clone(), sp_core::clock::now()));
        let chiaki = opened.chiaki.clone();
        let mut on_tick = |t: &Tick, d: &FrameDiff, act: Action, proc_ms: f64, t0: f64| {
            if let Some(c) = &chiaki {
                check.add_tick(t, d);
                lock_any(c).log_tick(t, d.is_new, d.max_diff, d.mean_diff, act.as_str(), proc_ms, t0);
            }
        };
        let tm = test.clone();
        let mut gt = move |c: i64| -> Option<Frame> { tm.as_ref().and_then(|t| lock_any(t).gt(c)) };
        let mut rec: Option<&sp_replay::Recorder> = replay.as_deref();
        let opts = RunOptions {
            enabled: true,
            score: a.score,
            label,
            hz: 60,
            settings: Some(&live_settings as &dyn LiveSettings),
            remote: opened.remote.as_deref_mut().map(|r| r as &mut dyn sp_sched::Remote),
            replay: rec.as_mut().map(|r| r as &mut dyn sp_sched::Replay),
            on_tick: opened.chiaki.is_some().then_some(&mut on_tick as &mut dyn FnMut(&Tick, &FrameDiff, Action, f64, f64)),
            gt: test.is_some().then_some(&mut gt as &mut dyn FnMut(i64) -> Option<Frame>),
        };
        stats = pipe.run(&mut *opened.src, window.as_mut().map(|wd| wd as &mut dyn Presenter), opts);
        {
            let mut r = lock_any(&reports);
            r.live = None; // first, so a report written meanwhile never counts this session twice
            if a.results.is_some() {
                r.done.push((session_info(&cfg), stats.clone()));
            }
        }
        if stats.exit != Exit::Restart {
            break;
        }
        println!("{}", stats.summary());
        let reasons = {
            let s = lock(&settings);
            let r = rules.restart_reasons(&s).join(", ");
            if r.is_empty() { "settings".to_string() } else { r }
        };
        if let Some(hold) = opened.hold.as_mut() {
            hold(true); // no decoding on the GPU while the pipeline is rebuilt
        }
        drop(pipe); // RIFE's GPU memory goes before the new one is made
        cfg = ctx.session_config(&lock(&settings));
        println!("[run] restarting for {reasons}: {}", Ctx::describe(&cfg));
        let (c2, hdr) = (cfg.clone(), ctx.hdr10);
        let built = match window.as_mut() {
            Some(wd) => wd.busy(&format!("Restarting ({reasons})"), move || build(w, h, &c2, hdr)),
            None => build(w, h, &c2, hdr),
        };
        match built {
            Ok(i) => interp = Some(i),
            Err(e) => {
                eprintln!("[run] couldn't restart RIFE: {e}");
                rc = 1;
                break;
            }
        }
        if let Some(hold) = opened.hold.as_mut() {
            hold(false);
        }
    }

    // The video window goes at once, then the rest of stopping.
    if let Some(wd) = window.as_mut() {
        wd.p.hide();
    }
    drop(awake); // the screen can turn off again
    if let Some(c) = &opened.chiaki {
        sp_core::Source::set_audio_delay(&mut *lock_any(c), 0.0); // chiaki-ng's sound back on time
    }
    if let Some(r) = &replay {
        r.close(30.0); // a clip being saved is finished first. A before & after carries on alone.
    }
    if let Some(mut wd) = window.take() {
        wd.close();
    }
    opened.src.close();
    stop.store(true, Ordering::SeqCst);
    if let Some(Some(t)) = results_thread {
        let _ = t.join(); // so the last write below isn't racing one already under way
    }
    println!("{}", stats.summary());
    if let Some(path) = &a.results
        && results_now(Path::new(path), &reports, &settings, &size, &gpu)
    {
        println!("[results] test results -> {path}");
    }
    if opened.chiaki.is_some() {
        write_stream_check(&check, a.results.as_deref());
    }
    if let Some(path) = &a.save_worst {
        save_worst(path, &stats);
    }
    rc
}
