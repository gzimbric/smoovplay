//! The Python helpers the launcher runs, as subcommands with the same arguments and output:
//! `smoovplay autodetect` (python -m framegen.autodetect), `looks` (framegen.looks), `looks-preview`
//! (framegen.looks_preview), `prepare` (framegen.prepare), `settings` (framegen.settings), `sources` (framegen.sources,
//! also run.py --list-sources), and the hooks for `capture` and `replay-compose`.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::os::fd::FromRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLPixelFormat, MTLStorageMode, MTLTexture};
use sp_core::{Frame, Gpu, Interpolator};
use sp_rife::{Options as RifeOptions, RifeInterpolator};
use sp_settings::autodetect::{self, Machine, Timing, VfgTiming};
use sp_settings::json::{self, Map, Value};
use sp_settings::looks::{self, Library};
use sp_settings::settings::{LookChoices, default_path, specs_json};
use sp_settings::Settings;

use crate::img;


fn arg_value(args: &[String], i: &mut usize, name: &str) -> String {
    if let Some(v) = args[*i].strip_prefix(&format!("{name}=")) {
        return v.to_string();
    }
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("smoovplay: error: argument {name}: expected one argument");
        std::process::exit(2)
    })
}

// ------------------------------------------------------------------------------------------- sources
const GUIDES: [(&str, &str); 5] = [
    ("chiaki", "Connect your PS5 with chiaki-ng (SmoovPlay sets it up), then press Play."),
    ("capture", "Connect the console's HDMI to the capture card and the card to this PC. On a PS5, turn off HDCP while you use it (Settings > System > HDMI). Set the console to 1080p, 60 Hz, HDR off (or the Capture card resolution picked in Settings)."),
    ("window", "Open the app you stream with (PS Remote Play, the Xbox app, Moonlight, or a browser) and start the game, pick its window, then press Play. Keep the app's window on screen: SmoovPlay shows the smooth picture on top of it, and the app keeps the controller and the keyboard. SmoovPlay's own keys use Ctrl+Alt (Ctrl+Alt+M menu, Ctrl+Alt+F fullscreen, Ctrl+Alt+Esc quit), and can be changed on the Play page."),
    ("tap", "For app developers: an app built to send its frames straight to SmoovPlay (docs/frame-tap.md in SmoovPlay's repo). For the Xbox app, PS Remote Play, Moonlight or a browser, pick App window instead. Press Play, then start the app: its smooth video opens in its own window, and the app keeps the controller and its sound."),
    ("file", "For testing: a video file played as a source."),
];

/// framegen/sources.py KINDS[kind] as listed (the Python's words, "this PC" included, so the list is the same).
fn kind_json(kind: &str, show_all: bool) -> Value {
    let i = sp_core::SourceInfo::of(kind);
    let controls = match kind {
        "chiaki" => "Your controller connects to this PC; chiaki-ng sends it to the PS5".to_string(),
        "tap" => "Your controller connects to the app as usual".to_string(), // sp-core's SourceInfo::of leaves it out
        _ => i.controls.clone(),
    };
    let mut m = Map::new();
    m.insert("kind", i.kind.as_str());
    m.insert("name", i.name.as_str());
    m.insert("live", i.live);
    m.insert("rate", i.rate.as_str());
    m.insert("hz", Value::Float(i.hz));
    m.insert("repeats", i.repeats.as_str());
    m.insert("test_mode", i.test_mode);
    m.insert("full_range", i.full_range);
    m.insert("audio", i.audio.as_str());
    m.insert("controls", controls);
    m.insert("hidden", kind == "file" && !show_all);
    m.insert("experimental", matches!(kind, "capture" | "window" | "tap"));
    m.insert("guide", GUIDES.iter().find(|(k, _)| *k == kind).map_or("", |(_, g)| g));
    Value::Dict(m)
}

/// `smoovplay sources [--json] [--all]` and run.py --list-sources.
pub fn sources_main(as_json: bool, all: bool) -> i32 {
    let show_all = all || std::env::var("SMOOVPLAY_SOURCES").is_ok_and(|v| v.eq_ignore_ascii_case("all"));
    let list: Vec<Value> = ["chiaki", "capture", "window", "tap", "file"].iter().map(|k| kind_json(k, show_all)).collect();
    if as_json {
        println!("{}", json::dumps(&json::obj([("sources", Value::List(list))])));
    } else {
        for s in &list {
            let g = |k: &str| s.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            let hidden = s.get("hidden").is_some_and(Value::truthy);
            println!("{:8} {}{}\n         {}", g("kind"), g("name"), if hidden { "  (hidden)" } else { "" }, g("guide"));
        }
    }
    0
}

// ------------------------------------------------------------------------------------------- capture and replay
/// `smoovplay capture --list [--json]` or `--formats NAME [--json]` (python -m framegen.capture), from sp-capture.
pub fn capture_main(args: &[String]) -> i32 {
    sp_capture::cli::main(args)
}

/// `smoovplay replay-compose ...` (python -m framegen.replay compose) runs sp-replay's before & after maker, which
/// the recorder starts as a process of its own.
pub fn replay_compose_main(args: &[String]) -> i32 {
    let args: Vec<String> = args.iter().skip_while(|a| *a == "compose").cloned().collect();
    sp_replay::compose_main(&args)
}

// ------------------------------------------------------------------------------------------- settings
/// `smoovplay settings` (python -m framegen.settings) prints the registry for the launcher's Settings page.
pub fn settings_main() -> i32 {
    println!("{}", json::dumps(&specs_json(&default_path(), &LookChoices::builtin())));
    0
}

// ------------------------------------------------------------------------------------------- autodetect
fn noise_frame(w: usize, h: usize) -> Frame {
    let mut x: u32 = 0x1234_5678;
    let bgra: Vec<u8> = (0..w * h * 4)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 24) as u8
        })
        .collect();
    Frame::from_bgra(&bgra, w, h)
}

/// autodetect.py time_rife() on Metal. Returns the ms for one new real frame plus one in-between frame at 1080p, at
/// full and half motion detail. Each is the median of five rounds of 12 ticks, after two rounds that let the GPU's
/// clocks come up, and each tick is waited for as the pipeline waits for it.
fn time_rife(model: &str, log: &dyn Fn(&str)) -> Result<Timing, String> {
    let (w, h) = (autodetect::STREAM.0 as usize, autodetect::STREAM.1 as usize);
    let frame = noise_frame(w, h);
    let mut ms = [0.0f64; 2];
    for (i, (motion, scale)) in [("full", 1.0f32), ("half", 0.5)].iter().enumerate() {
        log(&format!("timing the AI model at {motion} motion detail (Metal)"));
        let opts = RifeOptions { version: model.into(), scale: *scale, fp16: true, hdr: false, model_dir: None };
        let mut it = RifeInterpolator::new(w, h, opts).map_err(|e| e.to_string())?;
        it.load(&frame);
        it.prepare(0);
        it.wait();
        let mut rounds = Vec::new();
        for r in 0..7 {
            let t0 = sp_core::clock::now();
            for _ in 0..12 {
                it.prepare(1);
                it.interpolate(1, 0.5);
            }
            it.wait();
            if r >= 2 {
                rounds.push((sp_core::clock::now() - t0) * 1000.0 / 12.0);
            }
        }
        rounds.sort_by(|a, b| a.total_cmp(b));
        ms[i] = rounds[rounds.len() / 2];
    }
    Ok(Timing::measured(ms[0], ms[1], "metal"))
}

/// `smoovplay autodetect [--apply] [--if-new] [--settings P] [--timing F,H] [--vfg-timing L,M,H] [--present-ms MS]`
pub fn autodetect_main(args: &[String]) -> i32 {
    let (mut apply, mut if_new) = (false, false);
    let mut path = default_path();
    let (mut timing_arg, mut vfg_arg, mut present) = (None::<String>, None::<String>, crate::run::PRESENT_MS);
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a.split('=').next().unwrap() {
            "--apply" => apply = true,
            "--if-new" => if_new = true,
            "--settings" => path = PathBuf::from(arg_value(args, &mut i, "--settings")),
            "--timing" => timing_arg = Some(arg_value(args, &mut i, "--timing")),
            "--vfg-timing" => vfg_arg = Some(arg_value(args, &mut i, "--vfg-timing")),
            "--present-ms" => present = arg_value(args, &mut i, "--present-ms").parse().unwrap_or(present),
            "-h" | "--help" => {
                println!(
                    "usage: smoovplay autodetect [-h] [--apply] [--if-new] [--settings SETTINGS] [--timing FULL,HALF] \
                     [--vfg-timing LOW,MEDIUM,HIGH] [--present-ms MS]\n\nAuto-detect: suggests settings for this Mac from \
                     a short test of RIFE on its GPU and the display (python -m framegen.autodetect)."
                );
                return 0;
            }
            _ => {
                eprintln!("smoovplay autodetect: error: unrecognized arguments: {a}");
                return 2;
            }
        }
        i += 1;
    }
    let say = |m: &str| {
        println!("[detect] {m}");
        let _ = std::io::stdout().flush();
    };
    if if_new && path.exists() {
        println!("{}", autodetect::ALREADY_SAVED);
        return 0;
    }
    let mut settings = Settings::load(&path, Library::open_default().choices());
    let current = settings.values();
    say("reading the display");
    let display = autodetect::primary_display();
    say("starting the graphics card");
    let gpu = Gpu::shared().name();
    let machine = Machine::mac(&gpu, autodetect::mac_memory_gb(), display);
    let parse = |s: &str| -> Option<Vec<f64>> { s.split(',').map(|x| x.trim().parse().ok()).collect() };
    let timing = match &timing_arg {
        Some(t) => match parse(t).as_deref() {
            Some([f, h]) => Timing::for_testing(*f, *h, false),
            _ => {
                eprintln!("smoovplay autodetect: error: --timing FULL,HALF");
                return 2;
            }
        },
        None => {
            let model = current.get("model").and_then(Value::as_str).unwrap_or("4.22.lite");
            let model = if model == "vfg" { crate::adapt::default_model() } else { model.to_string() };
            match time_rife(&model, &say) {
                Ok(t) => t,
                Err(e) => {
                    println!("[detect] ERROR: {e}");
                    return 2;
                }
            }
        }
    };
    let vfg = vfg_arg.as_deref().and_then(parse).and_then(|v| match v[..] {
        [low, medium, high] => Some(VfgTiming { low, medium, high }),
        _ => None,
    });
    say("choosing settings");
    let result = autodetect::detect_with(&current, &machine, &timing, vfg.as_ref(), Some(present));
    println!("{}{}", autodetect::PREFIX, json::dumps(&result));
    for line in autodetect::finish(&mut settings, &path, &result, timing_arg.is_none(), apply, if_new) {
        println!("{line}");
    }
    0
}

// ------------------------------------------------------------------------------------------- prepare
/// `smoovplay prepare [--model M] [--no-detect]` (python -m framegen.prepare on a Mac).
pub fn prepare_main(args: &[String]) -> i32 {
    let mut model = "4.22.lite".to_string();
    let mut detect = true;
    let mut i = 0;
    while i < args.len() {
        match args[i].split('=').next().unwrap() {
            "--model" => model = arg_value(args, &mut i, "--model"),
            "--no-detect" => detect = false,
            "--width" | "--height" | "--scales" => {
                let name = args[i].split('=').next().unwrap().to_string();
                let _ = arg_value(args, &mut i, &name); // TensorRT's, and there's nothing to build ahead on a Mac
            }
            other => {
                eprintln!("smoovplay prepare: error: unrecognized arguments: {other}");
                return 2;
            }
        }
        i += 1;
    }
    println!("[prepare] checking the graphics card");
    println!("[prepare] {}: RIFE runs on Apple's GPU (Metal and MPSGraph)", Gpu::shared().name());
    if let Err(e) = sp_rife::weights::ensure_weights(&model, &sp_rife::weights::default_model_dir()) {
        println!("[prepare] ERROR: couldn't get RIFE {model}'s weights: {e}");
        return 2;
    }
    println!("[prepare] RIFE {model} weights ready");
    if detect {
        println!("[prepare] picking settings for this PC");
        let rc = autodetect_main(&["--apply".into(), "--if-new".into()]);
        if rc != 0 {
            println!("[prepare] couldn't pick settings (Auto-detect ended with {rc}); the defaults are used");
        }
    }
    println!("[prepare] done");
    0
}

// ------------------------------------------------------------------------------------------- looks
/// framegen/samples, as shipped with the repo ($SMOOVPLAY_SAMPLES for another).
pub fn samples_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("SMOOVPLAY_SAMPLES") {
        return PathBuf::from(d);
    }
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../framegen/samples");
    std::fs::canonicalize(&d).unwrap_or(d)
}

/// Looks drawn by the video window's own renderer (sp-present), offscreen.
struct LookDrawer {
    r: sp_present::renderer::Renderer,
    luts: Arc<sp_present::looks::LutStore>,
}

impl LookDrawer {
    fn new(lut_folder: PathBuf) -> Result<LookDrawer, String> {
        let luts = Arc::new(sp_present::looks::LutStore::new(Some(lut_folder)));
        let r = sp_present::renderer::Renderer::new(luts.clone()).map_err(|e| e.to_string())?;
        Ok(LookDrawer { r, luts })
    }

    /// The look on a picture at its own size (as the window's "bake"), rows top first in and out.
    fn draw(&mut self, look: &looks::Look, pic: &img::Image, frame: i64) -> Result<img::Image, String> {
        let Some(look) = crate::adapt::present_look(look) else { return Err("the look couldn't be drawn".into()) };
        let Some(plan) = self.r.looks.plan(&look, &self.luts) else {
            if let Some(why) = &self.r.looks.broken {
                return Err(format!("the look shaders didn't build: {why}"));
            }
            return Ok(pic.clone()); // nothing to draw
        };
        let src = Frame::from_bgr(&pic.bgr, pic.w, pic.h);
        let out = sp_present::gpu::texture_with(pic.w, pic.h, MTLPixelFormat::BGRA8Unorm, MTLStorageMode::Shared);
        let cmd = Gpu::shared().queue.commandBuffer().ok_or("no Metal command buffer")?;
        self.r.bake(&cmd, &plan, &src.texture, &out, frame.max(0) as u64).map_err(|e| e.to_string())?;
        cmd.commit();
        cmd.waitUntilCompleted();
        let mut raw = vec![0u8; pic.w * pic.h * 4];
        unsafe {
            out.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(raw.as_mut_ptr() as *mut _).unwrap(),
                pic.w * 4,
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize { width: pic.w, height: pic.h, depth: 1 },
                },
                0,
            );
        }
        Ok(img::Image { w: pic.w, h: pic.h, bgr: raw.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect() })
    }
}

/// `smoovplay looks ...` (python -m framegen.looks). It's sp-settings' CLI, with `preview` drawn here.
pub fn looks_main(args: &[String]) -> i32 {
    let lib_dir = looks::looks_dir();
    let samples = samples_dir();
    let cmd = match looks::cli::parse(args) {
        Err(usage) => {
            eprintln!("{usage}");
            return 2;
        }
        Ok(c) => c,
    };
    if let looks::cli::Command::Preview { look, src, out, frame } = &cmd {
        let say = |v: Value| -> i32 {
            let bad = v.get("error").is_some();
            println!("{}", json::dumps(&v));
            bad as i32
        };
        let lib = Library::new(&lib_dir);
        let l = match lib.get(look) {
            Some(l) => l.clone(),
            None => match looks::decode(look) {
                Ok(l) => l,
                Err(e) => return say(json::obj([("error", Value::from(e.0))])),
            },
        };
        let Some(pic) = img::read(src) else {
            return say(json::obj([("error", Value::from(format!("couldn't read {}", src.display())))]));
        };
        let mut d = match LookDrawer::new(lib.lut_folder()) {
            Ok(d) => d,
            Err(e) => return say(json::obj([("error", Value::from(e))])),
        };
        return match d.draw(&l, &pic, *frame) {
            Err(e) => say(json::obj([("error", Value::from(e))])),
            Ok(o) if !img::write(out, &o) => say(json::obj([("error", Value::from(format!("couldn't write {}", out.display())))])),
            Ok(_) => say(json::obj([("out", Value::from(out.to_string_lossy().as_ref()))])),
        };
    }
    let (line, rc) = looks::cli::main(args, &lib_dir, &samples, &mut std::io::stdin());
    if rc == 2 {
        eprintln!("{line}");
    } else {
        println!("{line}");
    }
    rc
}

/// Python's str(KeyError(k)).
fn key_error(k: &str) -> String {
    format!("'{k}'")
}

/// `smoovplay looks-preview` (python -m framegen.looks_preview) reads one JSON request per line on stdin and writes
/// one JSON answer per line on stdout. Anything else printed goes to stderr.
pub fn looks_preview_main() -> i32 {
    // stdout carries the answers, so everything else this process prints goes to stderr
    let out_fd = unsafe { libc::dup(1) };
    unsafe { libc::dup2(2, 1) };
    let mut out = unsafe { std::fs::File::from_raw_fd(out_fd) };
    let mut say = |v: Value| {
        let _ = out.write_all(format!("{}\n", json::dumps(&v)).as_bytes());
        let _ = out.flush();
    };
    let lib = Library::open_default();
    let listed = looks::samples(&samples_dir());
    let by_id: HashMap<String, Value> = listed
        .iter()
        .filter_map(|s| Some((s.get("id")?.as_str()?.to_string(), s.clone())))
        .collect();
    let mut drawer = match LookDrawer::new(lib.lut_folder()) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[looks] preview without the GPU ({e})");
            None
        }
    };
    say(json::obj([("ready", true.into()), ("gl", drawer.is_some().into()), ("samples", Value::List(listed.clone()))]));
    let mut pictures: HashMap<(String, usize, usize), img::Image> = HashMap::new();
    let mut originals: HashMap<String, img::Image> = HashMap::new();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut rid = Value::Null;
        let t0 = sp_core::clock::now();
        let r = (|| -> Result<Value, String> {
            let req = json::loads(line).map_err(|e| e.to_string())?;
            if req.as_dict().is_none() {
                return Err("'list' object has no attribute 'get'".into());
            }
            rid = req.get("id").cloned().unwrap_or(Value::Null);
            let look_v = req.get("look").ok_or_else(|| key_error("look"))?;
            let sample = req.get("sample").ok_or_else(|| key_error("sample"))?.as_str().ok_or("the sample isn't a name")?.to_string();
            let num = |k: &str, d: i64| -> Result<i64, String> {
                match req.get(k) {
                    None => Ok(d),
                    Some(v) => v.as_f64().map(|f| f as i64).ok_or_else(|| format!("{k} isn't a number")),
                }
            };
            let (w, h) = (num("width", 640)?.clamp(16, 1920) as usize, num("height", 360)?.clamp(16, 1080) as usize);
            let frame = num("frame", 0)?;
            let out_path = req.get("out").ok_or_else(|| key_error("out"))?.as_str().ok_or("out isn't a path")?.to_string();
            let key = (sample.clone(), w, h);
            if !pictures.contains_key(&key) {
                let s = by_id.get(&sample).ok_or_else(|| format!("there's no sample picture called {}", sp_settings::py::str_repr(&sample)))?;
                if !originals.contains_key(&sample) {
                    let p = s.get("path").and_then(Value::as_str).unwrap_or("");
                    let o = img::read(Path::new(p))
                        .ok_or_else(|| format!("couldn't read the sample picture {}", sp_settings::py::str_repr(&sample)))?;
                    originals.insert(sample.clone(), o);
                }
                if pictures.len() > 16 {
                    pictures.clear();
                }
                pictures.insert(key.clone(), img::resize_area(&originals[&sample], w, h));
            }
            let pic = &pictures[&key];
            let look = looks::normalize(look_v).map_err(|e| e.to_string())?;
            let drawn = match drawer.as_mut() {
                Some(d) => d.draw(&look, pic, frame)?,
                None => return Err("no GPU to draw the look with".into()),
            };
            if !img::write(Path::new(&out_path), &drawn) {
                return Err(format!("couldn't write {out_path}"));
            }
            let ms = ((sp_core::clock::now() - t0) * 10000.0).round() / 10.0;
            Ok(json::obj([("id", rid.clone()), ("file", out_path.as_str().into()), ("sample", sample.as_str().into()), ("ms", Value::Float(ms))]))
        })();
        match r {
            Ok(v) => say(v),
            Err(e) => say(json::obj([("id", rid), ("error", Value::from(e.chars().take(300).collect::<String>()))])),
        }
    }
    0
}
