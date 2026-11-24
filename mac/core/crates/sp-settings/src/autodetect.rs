//! Auto-detect (framegen/autodetect.py) suggests settings for this machine from a short test of the GPU, the display
//! and what's installed. The launcher's Settings page runs it ("Auto-detect") and reads one `[detect-result] {json}`
//! line, [`detect`]'s result.
//!
//! The binary times RIFE (one new real frame plus one in-between frame at 1080p, at full and at half motion detail)
//! and hands the times in as a [`Timing`]. [`recommend`] turns them and the display into settings. It's a plain
//! function, so the tests can try it on any machine.

use std::path::{Path, PathBuf};

use crate::json::{self, Map, Value};
use crate::py::fmt_fixed;
use crate::settings::{LookChoices, Settings, spec};

pub const PREFIX: &str = "[detect-result] ";
pub const STREAM: (i64, i64) = (1920, 1080);
/// An in-between frame has to be ready within this share of a display tick (calibrated on an RTX 4070 Ti SUPER).
pub const SHARE: f64 = 0.90;
/// Laptop graphics slow down as they warm up.
pub const LAPTOP_SHARE: f64 = 0.80;
/// Upscaler times per frame at 1440p on that card (scaled by the card's speed and the display's size).
pub const FSR_MS: f64 = 0.12;
pub const VSR_MS: [(i64, f64); 4] = [(1, 1.3), (2, 1.6), (3, 3.6), (4, 4.9)];
/// The test's half-motion time on that card, the yardstick for the card's speed.
pub const REF_HALF_MS: f64 = 6.8;
pub const REF_PIXELS: i64 = 2560 * 1440;
/// TensorRT's time at full motion detail vs PyTorch's, when its engine isn't built yet to time.
pub const TRT_SPEEDUP: f64 = 0.88;
/// On a Mac each frame shown also goes from the GPU to memory to OpenGL and is swapped. The Python pipeline measured
/// that at about 3 ms on an M5 Max at 1080p, beyond the test's time. It's counted for any GPU whose name starts with
/// "Apple".
pub const MAC_PRESENT_MS: f64 = 3.0;
/// Frame rate and motion detail, best first. 90 fps at full detail beats 120 at half.
pub const OPTIONS: [(i64, &str); 6] =
    [(120, "full"), (90, "full"), (120, "half"), (90, "half"), (60, "full"), (60, "half")];
/// NVIDIA VFG's modes, most careful first.
pub const VFG_QUALITIES: [&str; 3] = ["high", "medium", "low"];
/// The last test's times on this GPU, kept next to the settings file. Auto motion detail goes by them.
pub const TIMING_FILE: &str = "card-timing.json";
/// NVIDIA VFG's oldest driver.
pub const VFG_MIN_DRIVER: (i128, i128) = (570, 65);
/// framegen/vfg.py NOT_SET_UP: what vfg_unavailable() says on a card that can run it.
pub const VFG_NOT_SET_UP: &str =
    "NVIDIA VFG isn't set up on this PC. SmoovPlay's Settings page downloads it from NVIDIA.";

/// The main display: its size in pixels, the refresh rate it runs at, and the fastest it can run at that size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    pub width: i64,
    pub height: i64,
    pub hz: i64,
    pub max_hz: i64,
}

/// One motion detail's time: ms for a new real frame plus an in-between frame at 1080p, the engine that ran it, and
/// whether the time is estimated (TensorRT's, from PyTorch's).
#[derive(Clone, Debug, PartialEq)]
pub struct MotionTime {
    pub ms: f64,
    pub engine: String,
    pub estimated: bool,
}

/// time_rife()'s result.
#[derive(Clone, Debug, PartialEq)]
pub struct Timing {
    pub full: MotionTime,
    pub half: MotionTime,
}

impl Timing {
    /// From measured times (rounded to 0.01 ms, as time_rife() keeps them).
    pub fn measured(full_ms: f64, half_ms: f64, engine: &str) -> Timing {
        let t = |ms: f64| MotionTime { ms: crate::py::round_to(ms, 2), engine: engine.into(), estimated: false };
        Timing { full: t(full_ms), half: t(half_ms) }
    }

    /// --timing FULL,HALF, for testing the launcher without loading the GPU. Full is timed with the engine Auto runs
    /// at full detail ("trt" with TensorRT, else "torch"), and half with "torch".
    pub fn for_testing(full_ms: f64, half_ms: f64, trt: bool) -> Timing {
        Timing {
            full: MotionTime { ms: full_ms, engine: if trt { "trt" } else { "torch" }.into(), estimated: false },
            half: MotionTime { ms: half_ms, engine: "torch".into(), estimated: false },
        }
    }

    pub fn get(&self, motion: &str) -> &MotionTime {
        if motion == "half" { &self.half } else { &self.full }
    }

    pub fn to_value(&self) -> Value {
        let m = |t: &MotionTime| {
            json::obj([("ms", t.ms.into()), ("engine", t.engine.as_str().into()), ("estimated", t.estimated.into())])
        };
        json::obj([("full", m(&self.full)), ("half", m(&self.half))])
    }
}

/// time_vfg()'s result: NVIDIA VFG's ms at each quality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VfgTiming {
    pub low: f64,
    pub medium: f64,
    pub high: f64,
}

impl VfgTiming {
    pub fn get(&self, q: &str) -> f64 {
        match q {
            "low" => self.low,
            "medium" => self.medium,
            _ => self.high,
        }
    }
    pub fn to_value(&self) -> Value {
        json::obj([
            ("low", json::obj([("ms", self.low.into())])),
            ("medium", json::obj([("ms", self.medium.into())])),
            ("high", json::obj([("ms", self.high.into())])),
        ])
    }
}

/// What recommend() knows besides the times and the display.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    /// The GPU's name ("Apple M5 Max", "NVIDIA GeForce RTX 4080 Laptop GPU").
    pub gpu: String,
    /// TensorRT is installed.
    pub trt: bool,
    /// Why RTX VSR can't be used, or None if it can.
    pub vsr: Option<String>,
    /// How busy the GPU was before the test (%).
    pub busy: Option<i64>,
    /// NVIDIA VFG's times if it's set up.
    pub vfg: Option<VfgTiming>,
    /// Why NVIDIA VFG isn't set up ("" if it is).
    pub vfg_why: String,
}

/// recommend()'s result.
#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    pub settings: Map,
    pub reasons: Map,
    pub notes: Vec<String>,
}

impl Recommendation {
    pub fn to_value(&self) -> Value {
        json::obj([
            ("settings", Value::Dict(self.settings.clone())),
            ("reasons", Value::Dict(self.reasons.clone())),
            ("notes", Value::List(self.notes.iter().map(|n| Value::from(n.as_str())).collect())),
        ])
    }
}

fn fps(hz: i64) -> String {
    spec("hz").expect("hz").name(&Value::Int(hz), &LookChoices::builtin())
}

fn vsr_quality_name(q: i64) -> String {
    spec("vsr_quality").expect("vsr_quality").name(&Value::Int(q), &LookChoices::builtin())
}

fn vfg_quality_name(q: &str) -> String {
    spec("vfg_quality").expect("vfg_quality").name(&Value::from(q), &LookChoices::builtin())
}

struct Ctx<'a> {
    current: &'a Map,
    share: f64,
    fsr_ms: f64,
    speed: f64,
    pixels: i64,
    enlarge: bool,
    rate: i64,
    top: i64,
    display: Option<&'a Display>,
    opts: &'a Options,
}

fn is(current: &Map, key: &str, value: &str) -> bool {
    current.get(key).is_some_and(|v| v.py_eq(&Value::from(value)))
}

/// Settings for this machine. Only frame generation and picture settings are suggested, and the rest are left as
/// they are. NVIDIA VFG is suggested when it's set up and keeps up at least as high a frame rate as RIFE.
pub fn recommend(current: &Map, timing: &Timing, display: Option<&Display>, opts: &Options) -> Recommendation {
    recommend_with(current, timing, display, opts, None)
}

/// recommend() with the cost of showing each frame given, in ms per frame on top of the test's time. The Rust core's
/// Metal window costs less than the Python's MAC_PRESENT_MS, because frames reach it without a CPU copy. None keeps
/// the Python's rule, MAC_PRESENT_MS for a GPU whose name starts with "Apple" and nothing otherwise.
pub fn recommend_with(current: &Map, timing: &Timing, display: Option<&Display>, opts: &Options,
                      present_ms: Option<f64>) -> Recommendation {
    let share = if opts.gpu.to_lowercase().contains("laptop") { LAPTOP_SHARE } else { SHARE };
    let mac = opts.gpu.starts_with("Apple");
    let present_ms = present_ms.unwrap_or(if mac { MAC_PRESENT_MS } else { 0.0 });
    let speed = timing.half.ms / REF_HALF_MS; // above 1 means slower than the card the numbers come from
    let pixels = display.map_or(REF_PIXELS, |d| d.width * d.height);
    let enlarge = pixels as f64 > (STREAM.0 * STREAM.1) as f64 * 1.05; // a 1080p display shows the stream as it is
    let fsr_ms = if enlarge { FSR_MS * speed * pixels as f64 / REF_PIXELS as f64 } else { 0.0 };
    let rate = display.map_or(60, |d| d.hz);
    let top = if rate >= 119 {
        120
    } else if rate >= 89 {
        90
    } else {
        60
    }; // 119 = 119.88 Hz
    let cx = Ctx { current, share, fsr_ms, speed, pixels, enlarge, rate, top, display, opts };

    let cost = |motion: &str| timing.get(motion).ms + fsr_ms + present_ms;
    let fits = |hz: i64, motion: &str| cost(motion) <= share * 1000.0 / hz as f64;
    let (hz, motion) = OPTIONS.iter().copied().find(|&(h, m)| h <= top && fits(h, m)).unwrap_or((60, "half"));
    let mut settings = Map::new();
    let mut reasons = Map::new();
    let mut notes = Vec::new();

    // ---- model. NVIDIA VFG at the highest frame rate it keeps up with, then its most careful quality that fits.
    let rife = spec("model").expect("model").default_value();
    let vfg_pick = opts.vfg.and_then(|vfg| {
        [120, 90, 60].into_iter().filter(|&h| h <= top).find_map(|h| {
            VFG_QUALITIES.into_iter().find(|q| vfg.get(q) + fsr_ms <= share * 1000.0 / h as f64).map(|q| (h, q))
        })
    });
    if let (Some(pick), Some(vfg)) = (vfg_pick, opts.vfg)
        && pick.0 >= hz
    {
        return recommend_vfg(&cx, pick, &vfg, hz);
    }
    if opts.vfg.is_some() {
        settings.insert("model", rife.clone());
        reasons.insert(
            "model",
            match vfg_pick {
                Some((h, _)) => format!(
                    "NVIDIA VFG is set up, but it only keeps up at {} on this graphics card, and RIFE at {}.",
                    fps(h),
                    fps(hz)
                ),
                None => format!("NVIDIA VFG is set up, but it can't keep up at {} on this graphics card.", fps(60)),
            },
        );
    } else if is(current, "model", "vfg") {
        settings.insert("model", rife);
        reasons.insert(
            "model",
            if opts.vfg_why.is_empty() {
                "NVIDIA VFG isn't set up on this PC.".to_string()
            } else {
                opts.vfg_why.clone()
            },
        );
    }
    let tick = 1000.0 / hz as f64;

    // ---- frame rate
    settings.insert("hz", hz);
    let t = cost(motion);
    let reason = if !fits(hz, motion) {
        format!(
            "Even 60 fps is tight on this graphics card: an in-between frame takes {} ms, and 60 fps leaves {} ms. \
             Expect some frames to be shown twice.",
            fmt_fixed(t, 1),
            fmt_fixed(tick, 1)
        )
    } else if top == 60 {
        format!("Your display runs at {rate} Hz, so 60 fps is the most it can show.")
    } else if hz == top {
        format!(
            "Your display runs at {rate} Hz, and your graphics card draws an in-between frame in {} ms, inside the {} \
             ms each frame has at {}.",
            fmt_fixed(t, 1),
            fmt_fixed(tick, 1),
            fps(hz)
        )
    } else if hz == 90 && motion == "full" && fits(120, "half") {
        "120 fps would only fit with half motion detail on this graphics card. 90 fps keeps full detail, and looked \
         better in testing."
            .to_string()
    } else {
        let (cf, ch) = (cost("full"), cost("half"));
        let need = if ch < cf { ch } else { cf }; // the first of equals, as min() picks
        format!(
            "At {} each frame has {} ms, but your graphics card needs {} ms for an in-between frame. {} leaves it \
             enough time.",
            fps(top),
            fmt_fixed(1000.0 / top as f64, 1),
            fmt_fixed(need, 1),
            fps(hz)
        )
    };
    reasons.insert("hz", reason);

    // ---- motion detail. Auto is half at 120 fps and full otherwise. Half for this card's speed is saved as Half.
    let auto_motion = if hz == 120 { "half" } else { "full" };
    settings.insert("motion", if motion == auto_motion { "auto" } else { motion });
    let reason = if motion == "half" && hz < 120 {
        format!(
            "Half detail takes {} ms instead of {}, which is what lets {} keep up on this graphics card.",
            fmt_fixed(timing.half.ms, 1),
            fmt_fixed(timing.full.ms, 1),
            fps(hz)
        )
    } else if motion == "full" && hz == 120 {
        format!(
            "This graphics card is fast enough for full detail even at 120 fps ({} ms of {}).",
            fmt_fixed(timing.full.ms, 1),
            fmt_fixed(tick, 1)
        )
    } else if hz == 120 {
        "Auto uses half detail at 120 fps, which this graphics card needs.".to_string()
    } else {
        "Auto uses full detail here: the sharpest on fast motion.".to_string()
    };
    reasons.insert("motion", reason);

    // ---- engine
    settings.insert("engine", "auto");
    let reason = if timing.full.engine == "dml" {
        "Without an NVIDIA card, RIFE runs through DirectML (Experimental). It's slower than on an NVIDIA card of the \
         same speed, and hasn't been tried with a game on AMD or Intel graphics yet: Send feedback on the Help page \
         says how it went."
    } else if opts.gpu.starts_with("Apple") {
        "On a Mac, RIFE runs through PyTorch on Apple's GPU (Metal). TensorRT is for NVIDIA cards only."
    } else if opts.trt {
        "TensorRT is installed. Auto uses it at full motion detail and PyTorch at half: the faster of the two for each."
    } else {
        "TensorRT isn't installed, so Auto uses PyTorch, about 12% slower at full motion detail. Repair SmoovPlay, on \
         the Help page, puts it back."
    };
    reasons.insert("engine", reason);
    if timing.full.estimated {
        notes.push(
            "TensorRT hasn't been set up for this graphics card yet, so its speed is estimated. It does that the first \
             time you play, in a minute or two."
                .to_string(),
        );
    }
    upscaler(&cx, &mut settings, &mut reasons, hz, share * tick - timing.get(motion).ms);
    add_notes(&cx, &mut notes, &settings, hz, mac);
    Recommendation { settings, reasons, notes }
}

fn recommend_vfg(cx: &Ctx, pick: (i64, &str), vfg: &VfgTiming, rife_hz: i64) -> Recommendation {
    let (hz, q) = pick;
    let mut settings = Map::new();
    let mut reasons = Map::new();
    let mut notes = Vec::new();
    let (tick, name, t) = (1000.0 / hz as f64, vfg_quality_name(q), vfg.get(q) + cx.fsr_ms);
    settings.insert("model", "vfg");
    settings.insert("vfg_quality", q);
    settings.insert("hz", hz);
    reasons.insert(
        "model",
        format!(
            "NVIDIA VFG is set up on this PC and keeps up at {}{}",
            fps(hz),
            if rife_hz == hz {
                ", like RIFE.".to_string()
            } else {
                format!(", where RIFE keeps up at {}.", fps(rife_hz))
            }
        ),
    );
    let reason = if cx.top == 60 {
        format!("Your display runs at {} Hz, so 60 fps is the most it can show.", cx.rate)
    } else if hz == cx.top {
        format!(
            "Your display runs at {} Hz, and NVIDIA VFG draws an in-between frame in {} ms at {name} quality, inside \
             the {} ms each frame has at {}.",
            cx.rate,
            fmt_fixed(t, 1),
            fmt_fixed(tick, 1),
            fps(hz)
        )
    } else {
        let need = vfg.low + cx.fsr_ms;
        format!(
            "At {} each frame has {} ms, but NVIDIA VFG needs {} ms for an in-between frame even at Low quality. {} \
             leaves it enough time.",
            fps(cx.top),
            fmt_fixed(1000.0 / cx.top as f64, 1),
            fmt_fixed(need, 1),
            fps(hz)
        )
    };
    reasons.insert("hz", reason);
    let at = VFG_QUALITIES.iter().position(|x| *x == q).unwrap_or(0);
    let tail = match VFG_QUALITIES[..at].last() {
        Some(b) => format!(": {} takes {} ms.", vfg_quality_name(b), fmt_fixed(vfg.get(b), 1)),
        None => ".".to_string(),
    };
    reasons
        .insert("vfg_quality", format!("{name} is the most careful quality with time to spare at {}{tail}", fps(hz)));
    upscaler(cx, &mut settings, &mut reasons, hz, cx.share * tick - vfg.get(q));
    add_notes(cx, &mut notes, &settings, hz, false);
    Recommendation { settings, reasons, notes }
}

/// AMD FSR 1, unless RTX VSR is picked and still fits in what frame generation leaves (`left` ms).
fn upscaler(cx: &Ctx, settings: &mut Map, reasons: &mut Map, hz: i64, left: f64) {
    let vsr_ms = |q: i64| {
        let base = VSR_MS.iter().find(|(k, _)| *k == q).map_or(0.0, |(_, v)| *v);
        base * cx.speed * cx.pixels as f64 / REF_PIXELS as f64
    };
    let size = cx.display.map_or("your display".to_string(), |d| format!("{}\u{d7}{}", d.width, d.height));
    let vsr = cx.opts.vsr.as_deref();
    if is(cx.current, "scaler", "vsr") && vsr.is_none() && cx.enlarge {
        match [4, 3, 2, 1].into_iter().find(|&q| vsr_ms(q) <= left) {
            Some(q) => {
                settings.insert("scaler", "vsr");
                settings.insert("vsr_quality", q);
                reasons.insert(
                    "scaler",
                    format!(
                        "RTX VSR, as you picked: it fits alongside {} (about {} ms a frame at {size}).",
                        fps(hz),
                        fmt_fixed(vsr_ms(q), 1)
                    ),
                );
                reasons.insert(
                    "vsr_quality",
                    format!("{} is the highest quality with time to spare at {}.", vsr_quality_name(q), fps(hz)),
                );
            }
            None => {
                settings.insert("scaler", "fsr");
                let room = if left >= 0.1 {
                    format!("more than {} leaves ({} ms)", fps(hz), fmt_fixed(left, 1))
                } else {
                    format!("and {} leaves no time for it", fps(hz))
                };
                reasons.insert(
                    "scaler",
                    format!(
                        "RTX VSR needs about {} ms a frame at {size}, {room}. AMD FSR 1 looked about the same in \
                         testing and takes about 0.1 ms.",
                        fmt_fixed(vsr_ms(1), 1)
                    ),
                );
            }
        }
    } else {
        settings.insert("scaler", "fsr");
        let reason = if is(cx.current, "scaler", "vsr") {
            match vsr.filter(|v| !v.is_empty()) {
                Some(v) => format!("RTX VSR isn't set up on this PC ({v})."),
                None => format!(
                    "Your display is {size}, the stream's own size, so there's nothing for RTX VSR to enlarge. AMD FSR 1 \
                     is the lightest."
                ),
            }
        } else if !cx.enlarge {
            format!(
                "Your display is {size}, the stream's own size, so the picture is hardly enlarged. AMD FSR 1 is the \
                 lightest and plenty."
            )
        } else {
            let ms = if 0.05 > cx.fsr_ms { 0.05 } else { cx.fsr_ms };
            format!(
                "Sharp and almost free: about {} ms a frame at {size}. RTX VSR looked about the same in testing and \
                 takes far longer.",
                fmt_fixed(ms, 1)
            )
        };
        reasons.insert("scaler", reason);
    }
}

fn add_notes(cx: &Ctx, notes: &mut Vec<String>, settings: &Map, hz: i64, mac: bool) {
    match cx.display {
        None => notes.push("Couldn't read the display's refresh rate, so this assumes 60 Hz.".into()),
        Some(d) if d.max_hz > d.hz + 5 => notes.push(format!(
            "Your display can run at {} Hz, but Windows has it at {} Hz. Raise it (Windows Settings > System > \
             Display > Advanced display), then run Auto-detect again for a smoother frame rate.",
            d.max_hz, d.hz
        )),
        _ => {}
    }
    let rate = cx.rate;
    if cx.display.is_some() && hz > 60 && rate.rem_euclid(hz) > 2 && hz - rate.rem_euclid(hz) > 2 {
        notes.push(if !mac {
            format!(
                "{} doesn't divide evenly into your display's {rate} Hz, so keep G-Sync or FreeSync on (most fast \
                 displays have one) for even motion.",
                fps(hz)
            )
        } else {
            format!(
                "{} doesn't divide evenly into your display's {rate} Hz, so for even motion set its refresh rate to \
                 Variable in System Settings > Displays, if it offers that (most fast displays do).",
                fps(hz)
            )
        });
    }
    if cx.share == LAPTOP_SHARE {
        notes.push("Laptop graphics slow down as they warm up, so this leaves extra room.".into());
    }
    if let Some(busy) = cx.opts.busy.filter(|b| *b >= 30) {
        notes.push(format!(
            "Your graphics card was already {busy}% busy, so the test may have run slow. Close games and video apps, \
             then run it again."
        ));
    }
    debug_assert!(settings.iter().all(|(k, v)| { spec(k).is_some_and(|s| s.is_choice(v, &LookChoices::builtin())) }));
}

/// Auto motion detail for a w x h stream at hz. It's a quarter for 4K. It's half at 120 fps, for a capture bigger
/// than 1080p at 90, and where the test's time at full detail (full_ms, at 1080p) doesn't fit a frame at hz on this
/// GPU by recommend()'s own rule. Otherwise it's full.
pub fn auto_scale(w: i64, h: i64, hz: i64, full_ms: Option<f64>, share: f64) -> f64 {
    let _ = w;
    if h > 1440 {
        return 0.25;
    }
    if hz == 120 || (h > 1080 && hz >= 90) {
        return 0.5;
    }
    if h <= 1080 && full_ms.is_some_and(|ms| ms > share * 1000.0 / hz as f64) {
        return 0.5;
    }
    1.0
}

/// card-timing.json, next to the settings file.
pub fn timing_path(settings_path: &Path) -> PathBuf {
    settings_path.with_file_name(TIMING_FILE)
}

/// Keeps the test's times for Auto motion detail. This is best effort. A folder that can't be written to only means
/// Auto goes by its plain rule.
pub fn save_timing(settings_path: &Path, gpu: &str, timing: &Value) {
    let path = timing_path(settings_path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, json::dumps(&json::obj([("gpu", gpu.into()), ("timing", timing.clone())])));
}

/// The last test's time at full motion detail on this GPU, in ms. None if there's none, it was another GPU's, or,
/// with `engine`, it was timed with another engine: on a Mac the Python core writes the same file with "torch"
/// times under the same GPU name, and those don't apply to this core's "metal" ones (nor the other way round).
pub fn saved_full_ms(settings_path: &Path, gpu: &str, engine: Option<&str>) -> Option<f64> {
    let text = crate::py::read_text(&timing_path(settings_path)).ok()?;
    let saved = json::loads(&text).ok()?;
    if !saved.get("gpu").is_some_and(|g| g.py_eq(&Value::from(gpu))) {
        return None;
    }
    let full = saved.get("timing")?.get("full")?;
    if let Some(e) = engine
        && !full.get("engine").is_some_and(|g| g.py_eq(&Value::from(e)))
    {
        return None;
    }
    match full.get("ms")? {
        Value::Str(s) => crate::py::float_from_str(s),
        v @ (Value::Bool(_) | Value::Int(_) | Value::Float(_)) => v.as_f64(),
        _ => None,
    }
}

/// Whether nvidia-smi's driver version (like "566.36") is older than NVIDIA VFG's oldest. False if it's unknown or
/// in another form.
pub fn vfg_driver_too_old(driver: Option<&str>) -> bool {
    let Some(d) = driver.filter(|d| !d.is_empty()) else { return false };
    let parts: Option<Vec<i128>> = d.split('.').take(2).map(crate::py::int_from_str).collect();
    match parts {
        Some(p) => p.as_slice() < [VFG_MIN_DRIVER.0, VFG_MIN_DRIVER.1].as_slice(),
        None => false,
    }
}

/// What Auto-detect says about setting up NVIDIA VFG, or "" for nothing. It's said when the card can run it
/// (not_set_up) and it isn't set up. It makes no speed or quality claims, because NVIDIA's licence forbids publishing
/// VFG's results.
pub fn vfg_offer(not_set_up: bool, gpu: &str, driver: Option<&str>) -> String {
    if !not_set_up {
        return String::new();
    }
    let name = gpu.replace("NVIDIA ", "").replace("GeForce ", "");
    let name = crate::py::strip(&name);
    let name = if name.is_empty() { "graphics card" } else { name };
    let mut text = format!(
        "Your {name} can also run NVIDIA VFG, NVIDIA's own frame generation, in place of RIFE. It's a separate download \
         from NVIDIA, under NVIDIA's licence. Set it up, and Auto-detect times it and suggests it if it keeps up on this \
         PC."
    );
    if vfg_driver_too_old(driver) {
        text.push_str(&format!(
            " It needs NVIDIA driver {}.{} or newer, and this PC has {}: update it first (see the PC check).",
            VFG_MIN_DRIVER.0,
            VFG_MIN_DRIVER.1,
            driver.unwrap_or("")
        ));
    }
    text
}

/// What the test found about the machine besides the times (detect() reads these from the system).
#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    pub display: Option<Display>,
    /// nvidia-smi's driver version.
    pub driver: Option<String>,
    /// How busy the GPU was (%), from nvidia-smi.
    pub busy: Option<i64>,
    pub gpu: String,
    pub vram_gb: f64,
    /// Apple's GPU (the memory is shared).
    pub mac_mode: bool,
    /// An AMD or Intel card through DirectML.
    pub dml_mode: bool,
    /// TensorRT is installed.
    pub trt: bool,
    /// Why RTX VSR can't be used, or None.
    pub vsr: Option<String>,
    /// Why NVIDIA VFG isn't available or set up, or "" if it is. That's vfg_unavailable(), or why it couldn't start.
    pub vfg_why: String,
}

impl Machine {
    /// A Mac, so no NVIDIA anything.
    pub fn mac(gpu: &str, memory_gb: f64, display: Option<Display>) -> Machine {
        Machine {
            display,
            driver: None,
            busy: None,
            gpu: gpu.into(),
            vram_gb: memory_gb,
            mac_mode: true,
            dml_mode: false,
            trt: false,
            vsr: Some("it needs an NVIDIA RTX card".into()),
            vfg_why: "NVIDIA VFG needs an NVIDIA RTX card".into(),
        }
    }
}

fn engine_name(engine: &str, mac: bool) -> String {
    match engine {
        "trt" => "TensorRT".into(),
        "torch" => if mac { "PyTorch on Metal" } else { "PyTorch" }.into(),
        "dml" => "DirectML (Experimental)".into(),
        "metal" => "Metal".into(), // the Rust core's own, which the Python never reports
        other => other.into(),
    }
}

/// The whole result the launcher reads after "[detect-result] ". That's recommend()'s, plus what was found ("found",
/// rows of label, value and detail), the times, how busy the GPU was and its name. vfg is NVIDIA VFG's times, if it
/// was timed.
pub fn detect(current: &Map, machine: &Machine, timing: &Timing, vfg: Option<&VfgTiming>) -> Value {
    detect_with(current, machine, timing, vfg, None)
}

/// detect() with the cost of showing each frame given (see recommend_with).
pub fn detect_with(current: &Map, machine: &Machine, timing: &Timing, vfg: Option<&VfgTiming>,
                   present_ms: Option<f64>) -> Value {
    let (mac_mode, dml_mode) = (machine.mac_mode, machine.dml_mode);
    let (trt, vsr) = if dml_mode || mac_mode {
        (false, Some("it needs an NVIDIA RTX card".to_string()))
    } else {
        (machine.trt, machine.vsr.clone())
    };
    let vfg_why = if vfg.is_some() { String::new() } else { machine.vfg_why.clone() };
    let opts = Options {
        gpu: machine.gpu.clone(),
        trt,
        vsr: vsr.clone(),
        busy: machine.busy,
        vfg: vfg.copied(),
        vfg_why: vfg_why.clone(),
    };
    let rec = recommend_with(current, timing, machine.display.as_ref(), &opts, present_ms);
    let Value::Dict(mut result) = rec.to_value() else { unreachable!() };
    let name = machine.gpu.replace("NVIDIA ", "").replace("GeForce ", "");
    let nvidia_only = dml_mode || mac_mode;
    let row = |label: &str, value: String, detail: String| {
        json::obj([("label", label.into()), ("value", value.into()), ("detail", detail.into())])
    };
    let (full, half) = (&timing.full, &timing.half);
    let d = machine.display.as_ref();
    let mut found = vec![
        row(
            "Graphics card",
            format!("{name}, {} GB{}", fmt_fixed(machine.vram_gb, 0), if mac_mode { " shared" } else { "" }),
            machine.driver.as_deref().filter(|d| !d.is_empty()).map(|d| format!("Driver {d}")).unwrap_or_default(),
        ),
        row(
            "AI speed",
            format!("{} ms per in-between frame", fmt_fixed(full.ms, 1)),
            format!(
                "Full motion detail, {}{}. Half: {} ms, {}.",
                engine_name(&full.engine, mac_mode),
                if full.estimated { " (estimated)" } else { "" },
                fmt_fixed(half.ms, 1),
                engine_name(&half.engine, mac_mode)
            ),
        ),
        row(
            "Display",
            d.map_or("Unknown".into(), |d| format!("{}\u{d7}{} at {} Hz", d.width, d.height, d.hz)),
            d.filter(|d| d.max_hz > d.hz + 5).map(|d| format!("Can run at {} Hz", d.max_hz)).unwrap_or_default(),
        ),
        row(
            "TensorRT",
            (if nvidia_only {
                "NVIDIA cards only"
            } else if trt {
                "Installed"
            } else {
                "Not installed"
            })
            .into(),
            String::new(),
        ),
        row(
            "RTX VSR",
            (if nvidia_only {
                "NVIDIA cards only"
            } else if vsr.is_none() {
                "Ready"
            } else {
                "Not set up"
            })
            .into(),
            String::new(),
        ),
    ];
    match vfg {
        Some(v) => found.push(row(
            "NVIDIA VFG",
            format!("{} ms per in-between frame", fmt_fixed(v.medium, 1)),
            format!("Medium quality. Low: {} ms, High: {} ms.", fmt_fixed(v.low, 1), fmt_fixed(v.high, 1)),
        )),
        None => {
            let not_set_up = vfg_why == VFG_NOT_SET_UP;
            found.push(row(
                "NVIDIA VFG",
                (if vfg_why.is_empty() {
                    "Set up"
                } else if not_set_up {
                    "Not set up"
                } else {
                    "Not available"
                })
                .into(),
                if vfg_why.is_empty() {
                    String::new()
                } else if not_set_up {
                    "Optional: see below.".into()
                } else {
                    vfg_why.clone()
                },
            ));
        }
    }
    result.insert("found", Value::List(found));
    if vfg.is_none() {
        let offer = vfg_offer(vfg_why == VFG_NOT_SET_UP, &machine.gpu, machine.driver.as_deref());
        if !offer.is_empty() {
            result.insert("vfg_offer", offer);
            result.insert("vfg_offer_ready", !vfg_driver_too_old(machine.driver.as_deref()));
        }
    }
    result.insert("timing", timing.to_value());
    result.insert("vfg_timing", vfg.map_or(Value::Null, VfgTiming::to_value));
    result.insert("busy", machine.busy.map_or(Value::Null, Value::Int));
    result.insert("gpu", machine.gpu.as_str());
    Value::Dict(result)
}

/// Runs after the test, as `python -m framegen.autodetect` finishes. It keeps the times when they were measured and
/// the GPU wasn't busy. With `apply` it saves the suggestions, except with `if_new` on a busy GPU, where setup's
/// suggestions would be too low. Returns the `[detect]` lines it would print.
pub fn finish(
    settings: &mut Settings,
    settings_path: &Path,
    result: &Value,
    measured: bool,
    apply: bool,
    if_new: bool,
) -> Vec<String> {
    let mut lines = Vec::new();
    let busy = result.get("busy").and_then(Value::as_i64).unwrap_or(0);
    if measured && busy < 30 {
        let gpu = result.get("gpu").and_then(Value::as_str).unwrap_or("");
        save_timing(settings_path, gpu, result.get("timing").unwrap_or(&Value::Null));
    }
    if apply && if_new && busy >= 30 {
        lines.push(format!(
            "[detect] the graphics card was {}% busy, so nothing was saved",
            result.get("busy").map_or("None".into(), Value::py_str)
        ));
    } else if apply {
        if let Some(Value::Dict(s)) = result.get("settings") {
            for (k, v) in s.iter() {
                let _ = settings.set(k, v.clone());
            }
        }
        lines.push(format!("[detect] saved to {}", settings_path.display()));
    }
    lines
}

/// The line printed when --if-new finds a settings file already there.
pub const ALREADY_SAVED: &str = "[detect] settings already saved, so they're left as they are";

// ------------------------------------------------------------------------------------------------ this Mac
#[cfg(target_os = "macos")]
mod mac {
    use std::ffi::{CString, c_char, c_int, c_void};

    use super::Display;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGMainDisplayID() -> u32;
        fn CGDisplayCopyDisplayMode(display: u32) -> *mut c_void;
        fn CGDisplayCopyAllDisplayModes(display: u32, options: *const c_void) -> *const c_void;
        fn CGDisplayModeGetWidth(mode: *const c_void) -> usize;
        fn CGDisplayModeGetHeight(mode: *const c_void) -> usize;
        fn CGDisplayModeGetPixelWidth(mode: *const c_void) -> usize;
        fn CGDisplayModeGetPixelHeight(mode: *const c_void) -> usize;
        fn CGDisplayModeGetRefreshRate(mode: *const c_void) -> f64;
        fn CGDisplayModeRelease(mode: *mut c_void);
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFArrayGetCount(a: *const c_void) -> isize;
        fn CFArrayGetValueAtIndex(a: *const c_void, i: isize) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    pub fn display() -> Option<Display> {
        // SAFETY: CoreGraphics calls on the main display. Every copied object is released and nulls are checked.
        unsafe {
            let id = CGMainDisplayID();
            let mode = CGDisplayCopyDisplayMode(id);
            if mode.is_null() {
                return None;
            }
            let (w, h) = (CGDisplayModeGetPixelWidth(mode), CGDisplayModeGetPixelHeight(mode));
            let size = (CGDisplayModeGetWidth(mode), CGDisplayModeGetHeight(mode));
            let r = CGDisplayModeGetRefreshRate(mode).round_ties_even() as i64;
            CGDisplayModeRelease(mode);
            let hz = if r == 0 { 60 } else { r };
            let mut max_hz = hz;
            let all = CGDisplayCopyAllDisplayModes(id, std::ptr::null());
            if !all.is_null() {
                for i in 0..CFArrayGetCount(all) {
                    let m = CFArrayGetValueAtIndex(all, i);
                    if !m.is_null() && (CGDisplayModeGetWidth(m), CGDisplayModeGetHeight(m)) == size {
                        max_hz = max_hz.max(CGDisplayModeGetRefreshRate(m).round_ties_even() as i64);
                    }
                }
                CFRelease(all);
            }
            Some(Display { width: w as i64, height: h as i64, hz, max_hz })
        }
    }

    fn sysctl(name: &str) -> Option<Vec<u8>> {
        let n = CString::new(name).ok()?;
        let mut len = 0usize;
        // SAFETY: the size query, then a buffer of that size.
        unsafe {
            if sysctlbyname(n.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) != 0 || len == 0 {
                return None;
            }
            let mut buf = vec![0u8; len];
            if sysctlbyname(n.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) != 0 {
                return None;
            }
            buf.truncate(len);
            Some(buf)
        }
    }

    pub fn memory_gb() -> f64 {
        sysctl("hw.memsize")
            .filter(|b| b.len() == 8)
            .map_or(0.0, |b| u64::from_ne_bytes(b.try_into().expect("8 bytes")) as f64 / (1u64 << 30) as f64)
    }

    pub fn chip_name() -> String {
        sysctl("machdep.cpu.brand_string")
            .map(|b| String::from_utf8_lossy(&b).trim_end_matches('\0').trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into())
    }
}

/// The Mac's main display: its size in pixels and refresh rate (a ProMotion display reports 120). macOS runs a display
/// at its top rate for that size, so hz and max_hz are usually the same. None where it can't be read.
pub fn primary_display() -> Option<Display> {
    #[cfg(target_os = "macos")]
    {
        mac::display()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// A Mac's memory, which its GPU shares, in GB (0 if it can't be read).
pub fn mac_memory_gb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        mac::memory_gb()
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}

/// The chip's name, as framegen/gpu.py name() gives it on a Mac ("Apple M5 Max"), or "unknown" if it can't be told.
pub fn mac_gpu_name() -> String {
    #[cfg(target_os = "macos")]
    {
        mac::chip_name()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unknown".into()
    }
}
