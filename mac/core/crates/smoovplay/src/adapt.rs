//! The real settings (sp-settings) as the other crates want them: sp-sched's LiveSettings for the pipeline,
//! sp-present's MenuModel and DisplaySettings for the window's menu and hotkeys, and the Remote's Rules for the
//! launcher. The rules are run.py's on a Mac: the window's allowed() and disabled() (io.py GLWindowSink), run.py's
//! unavailable rows and choices, model_rows() and restart_reasons().

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use sp_present::display::{Corner, DisplaySettings, MenuModel, MenuRow, OverlaySize, Pacing, Scaler};
use sp_present::fill::Fill;
use sp_sched::{Live, LiveSettings, OverlayStyle};
use sp_settings::autodetect::auto_scale;
use sp_settings::looks::{Library, LookState};
use sp_settings::remote::Rules;
use sp_settings::settings::{spec, specs};
use sp_settings::{Settings, Value};

pub type SharedSettings = Arc<Mutex<Settings>>;

/// The settings, whatever a panicked holder left them as.
pub fn lock(s: &Mutex<Settings>) -> MutexGuard<'_, Settings> {
    s.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn lock_any<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// sp-settings' JSON as serde_json's (through the text, which keeps 1.0 a float).
pub fn to_serde(v: &Value) -> serde_json::Value {
    serde_json::from_str(&sp_settings::json::dumps(v)).unwrap_or(serde_json::Value::Null)
}

pub fn from_serde(v: &serde_json::Value) -> Value {
    sp_settings::json::loads(&v.to_string()).unwrap_or(Value::Null)
}

/// What a session is built from (run.py session_config). A restart is needed when this changes.
#[derive(Clone, Debug, PartialEq)]
pub struct Cfg {
    /// Always "metal" here (the Python's Mac engine is "torch").
    pub engine: String,
    pub motion: f64,
    pub model: String,
}

/// What the run's rules go by, which doesn't change while it plays.
#[derive(Clone, Debug)]
pub struct Ctx {
    /// The stream's size.
    pub w: usize,
    pub h: usize,
    /// --scale.
    pub scale: Option<f64>,
    /// --model when it isn't one of the setting's choices (another RIFE version, for testing).
    pub test_model: Option<String>,
    /// Auto-detect's time at full motion detail on this GPU plus the window's cost (ms), and the share of a tick it
    /// may take.
    pub card_full_ms: Option<f64>,
    pub card_share: f64,
    /// Why NVIDIA VFG can't be used, or "" if it can (never on a Mac).
    pub vfg_why: String,
    /// The pipeline carries HDR10 frames (RGBA16F PQ) to the window.
    pub hdr10: bool,
}

impl Ctx {
    pub fn session_config(&self, s: &Settings) -> Cfg {
        let scale = match self.scale {
            Some(v) => v,
            None => match s.get_str("motion") {
                "full" => 1.0,
                "half" => 0.5,
                _ => auto_scale(self.w as i64, self.h as i64, s.get_i64("hz"), self.card_full_ms, self.card_share),
            },
        };
        let model = self.test_model.clone().unwrap_or_else(|| s.get_str("model").to_string());
        let model = if model == "vfg" { default_model() } else { model };
        Cfg { engine: "metal".into(), motion: scale, model }
    }

    /// run.py describe().
    pub fn describe(cfg: &Cfg) -> String {
        format!("RIFE {} scale={} fp16  Metal (MPSGraph)", cfg.model, py_float(cfg.motion))
    }

    /// The overlay's first words (run.py label_of).
    pub fn label_of(&self, cfg: &Cfg, hdr_out: Option<&str>) -> String {
        let hdr_tag = match hdr_out {
            Some("sdr8") => "  HDR in SDR",
            Some("pq16") => "  HDR10",
            _ => "",
        };
        format!("RIFE {} Metal{}{hdr_tag}", cfg.model, if cfg.motion < 1.0 { " half" } else { "" })
    }
}

pub fn default_model() -> String {
    spec("model").and_then(|s| s.default_value().as_str().map(str::to_string)).unwrap_or_else(|| "4.22.lite".into())
}

/// Python's str(float): 1.0, 0.5, 0.25.
pub fn py_float(x: f64) -> String {
    Value::Float(x).py_str()
}

/// The rows and choices the window and the launcher grey out or limit, and the session that's playing.
#[derive(Debug, Default)]
pub struct Rows {
    /// run.py's `unavailable`, why each key is greyed out.
    pub unavailable: BTreeMap<String, String>,
    /// run.py's `choices`, the values that can be picked for each key.
    pub choices: BTreeMap<String, Vec<Value>>,
    /// The playing session's config (run.py `started`).
    pub started: Option<Cfg>,
}

/// The rules, shared by the window's menu and the launcher's link.
pub struct RunRules {
    pub ctx: Arc<Ctx>,
    pub rows: Arc<Mutex<Rows>>,
    pub menu_open: Arc<AtomicBool>,
}

impl RunRules {
    fn model_is_vfg(&self, s: &Settings) -> bool {
        self.ctx.test_model.as_deref().unwrap_or_else(|| s.get_str("model")) == "vfg"
    }

    /// The window's allowed(key, value).
    pub fn allowed(&self, _s: &Settings, key: &str, value: &Value) -> bool {
        if key == "scaler" {
            return value.as_str().is_some_and(|v| v == "fsr" || v == "bilinear"); // RTX VSR is NVIDIA's
        }
        match lock_any(&self.rows).choices.get(key) {
            Some(vs) => value.py_in(vs),
            None => true,
        }
    }

    /// The window's disabled(key), with run.py's model_rows() applied.
    pub fn disabled(&self, s: &Settings, key: &str) -> Option<String> {
        if key == "vsr_quality" {
            return Some("RTX VSR isn't available: it needs an NVIDIA RTX card.".into());
        }
        if let Some(why) = lock_any(&self.rows).unavailable.get(key) {
            return Some(why.clone());
        }
        let vfg = self.model_is_vfg(s);
        match key {
            "engine" | "motion" if vfg => Some("Not used with NVIDIA VFG.".into()),
            "vfg_quality" if !vfg => {
                Some(if self.ctx.vfg_why.is_empty() { "Only used with NVIDIA VFG.".into() } else { self.ctx.vfg_why.clone() })
            }
            _ => None,
        }
    }

    /// run.py restart_reasons().
    pub fn restart_reasons(&self, s: &Settings) -> Vec<String> {
        let Some(started) = lock_any(&self.rows).started.clone() else { return Vec::new() };
        let cfg = self.ctx.session_config(s);
        let label = |k: &str| spec(k).map_or(k.to_string(), |s| s.label.to_string());
        if cfg.model != started.model {
            return vec![label("model")];
        }
        let mut out = Vec::new();
        if cfg.engine != started.engine {
            out.push(label("engine"));
        }
        if cfg.motion != started.motion {
            out.push(label("motion"));
        }
        out
    }
}

impl Rules for RunRules {
    fn allowed(&self, s: &Settings, key: &str, value: &Value) -> bool {
        RunRules::allowed(self, s, key, value)
    }
    fn disabled(&self, s: &Settings, key: &str) -> Option<String> {
        RunRules::disabled(self, s, key)
    }
    fn restart_reasons(&self, s: &Settings) -> Vec<String> {
        RunRules::restart_reasons(self, s)
    }
    fn extra(&self, _s: &Settings) -> Vec<(String, Value)> {
        vec![("menu_open".into(), Value::Bool(self.menu_open.load(Ordering::SeqCst)))]
    }
}

/// The settings the pipeline follows live.
pub struct LiveAdapter(pub SharedSettings);

impl LiveSettings for LiveAdapter {
    fn live(&self) -> Live {
        let s = lock(&self.0);
        Live {
            interp: s.get_bool("interp"),
            hz: s.get_i64("hz").clamp(60, 120) as u32,
            simulate_30: s.get_bool("simulate_30"),
            steady_hud: s.get_bool("steady_hud"),
            overlay: s.get_bool("overlay"),
            overlay_style: if s.get_str("overlay_style") == "detailed" { OverlayStyle::Detailed } else { OverlayStyle::Minimal },
            even_out: s.get_bool("even_out"),
            even_dips: s.get_bool("even_dips"),
            game_fps_30: s.get_str("game_fps") == "30",
            audio_sync: s.get_bool("audio_sync"),
        }
    }
    fn set_simulate_30(&self, on: bool) {
        let _ = lock(&self.0).override_value("simulate_30", on);
    }
}

/// The looks. It holds which one the video shows (the settings', or the launcher's preview) and the library.
pub struct Looks {
    pub state: LookState,
    pub library: Library,
}

pub type SharedLooks = Arc<Mutex<Looks>>;

/// The window's menu, hotkeys and display settings, on the real settings.
pub struct MenuAdapter {
    pub settings: SharedSettings,
    pub rules: Arc<RunRules>,
    pub looks: SharedLooks,
    /// Bumped by a watcher on every settings change.
    pub rev: Arc<AtomicU64>,
    /// The look last converted for the renderer, with the LookState rev it was converted at.
    cache: RefCell<Option<(u64, Option<Arc<sp_present::looks::Look>>)>>,
    /// The look the window shows, for instant replay's "Looks in clips".
    pub shown: Arc<Mutex<Option<Arc<sp_present::looks::Look>>>>,
}

// RefCell is enough because the window calls its model from one thread at a time. MenuModel needs Send only.
impl MenuAdapter {
    pub fn new(settings: SharedSettings, rules: Arc<RunRules>, looks: SharedLooks, rev: Arc<AtomicU64>) -> MenuAdapter {
        MenuAdapter { settings, rules, looks, rev, cache: RefCell::new(None), shown: Arc::new(Mutex::new(None)) }
    }

    fn row_of(&self, s: &Settings, sp: &sp_settings::Spec) -> MenuRow {
        let impact = |k: &str| sp.impact.iter().find(|(n, _)| *n == k).map_or(String::new(), |(_, t)| t.to_string());
        let value = s.get(sp.key).map_or(String::new(), |v| sp.name(v, s.look_choices()));
        MenuRow {
            key: sp.key.into(),
            label: sp.label.into(),
            value,
            live: sp.live,
            experimental: sp.experimental,
            disabled: self.rules.disabled(s, sp.key).filter(|w| !w.is_empty()),
            help: sp.help.into(),
            impact_picture: impact("picture"),
            impact_speed: impact("speed"),
        }
    }
}

/// A look as sp-present's renderer takes it. sp-settings' normalized look goes through its JSON, as both normalize
/// the same format.
pub fn present_look(look: &sp_settings::looks::Look) -> Option<Arc<sp_present::looks::Look>> {
    let v: serde_json::Value = serde_json::from_str(&sp_settings::json::dumps(&look.to_value())).ok()?;
    match sp_present::looks::normalize(&v) {
        Ok(l) => Some(Arc::new(l)),
        Err(e) => {
            println!("[looks] can't draw {:?}: {e}", look.name);
            None
        }
    }
}

impl MenuModel for MenuAdapter {
    fn sections(&self) -> Vec<String> {
        let s = lock(&self.settings);
        let values = s.values();
        let mut out: Vec<String> = Vec::new();
        for sp in specs() {
            if !sp.group.is_empty() && !out.iter().any(|g| g == sp.group) && sp.applies_to(&values) {
                out.push(sp.group.into());
            }
        }
        out
    }

    fn rows(&self, page: Option<&str>) -> Vec<MenuRow> {
        let s = lock(&self.settings);
        let values = s.values();
        specs()
            .iter()
            .filter(|sp| match page {
                None => (sp.quick && sp.live || sp.group.is_empty()) && sp.applies_to(&values),
                Some(p) => sp.group == p && sp.applies_to(&values),
            })
            .map(|sp| self.row_of(&s, sp))
            .collect()
    }

    fn restart_reasons(&self) -> Vec<String> {
        let s = lock(&self.settings);
        self.rules.restart_reasons(&s)
    }

    fn get(&self, key: &str) -> Option<serde_json::Value> {
        lock(&self.settings).get(key).map(to_serde)
    }

    fn set(&mut self, key: &str, value: serde_json::Value, save: bool) -> bool {
        let v = from_serde(&value);
        let mut s = lock(&self.settings);
        let Some(sp) = spec(key) else { return false };
        if !sp.is_choice(&v, s.look_choices()) {
            return false;
        }
        if key == "look_on" {
            lock_any(&self.looks).state.preview = None; // K goes back to the Look settings' look (LookState.toggle)
        }
        let v = sp.canonical(&v, s.look_choices()).unwrap_or(v);
        if save { s.set(key, v).is_ok() } else { s.override_value(key, v).is_ok() }
    }

    fn step(&mut self, key: &str, delta: i32) -> bool {
        let mut s = lock(&self.settings);
        if key == "look" {
            lock_any(&self.looks).state.preview = None; // J (LookState.next)
        }
        // the window's allowed() (it doesn't read the settings, which step() holds)
        let rules = self.rules.clone();
        let allowed = move |k: &str, v: &Value| -> bool {
            if k == "scaler" {
                return v.as_str().is_some_and(|x| x == "fsr" || x == "bilinear");
            }
            match lock_any(&rules.rows).choices.get(k) {
                Some(vs) => v.py_in(vs),
                None => true,
            }
        };
        s.step(key, delta as i64, Some(&allowed)).is_ok()
    }

    fn display(&self) -> DisplaySettings {
        let s = lock(&self.settings);
        let looks = lock_any(&self.looks);
        let rev = looks.state.rev();
        let look = {
            let mut cache = self.cache.borrow_mut();
            match cache.as_ref() {
                Some((r, l)) if *r == rev => l.clone(),
                _ => {
                    let l = if self.rules.ctx.hdr10 {
                        None // looks are made for SDR pictures
                    } else {
                        looks.state.current(&s, &looks.library).and_then(present_look)
                    };
                    *cache = Some((rev, l.clone()));
                    *lock_any(&self.shown) = l.clone();
                    l
                }
            }
        };
        let pacing = match std::env::var("SMOOVPLAY_PACING").ok().as_deref() {
            Some("immediate") => Pacing::Immediate,
            Some("min") => Pacing::MinDuration,
            Some("steady") => Pacing::Steady,
            _ => Pacing::Auto,
        };
        DisplaySettings {
            scaler: Scaler::from_key(s.get_str("scaler")),
            fsr_sharpness: s.get_f64("fsr_sharpness") as f32,
            fill: Fill::from_key(s.get_str("fill")).unwrap_or(Fill::Fit),
            overlay: s.get_bool("overlay"),
            overlay_corner: Corner::from_key(s.get_str("overlay_corner")),
            overlay_size: OverlaySize::from_key(s.get_str("overlay_size")),
            fullscreen: s.get_bool("fullscreen"),
            vsync: s.get_bool("vsync"),
            look,
            hz: s.get_i64("hz") as f64,
            pacing,
        }
    }

    fn revision(&self) -> u64 {
        self.rev.load(Ordering::SeqCst) + lock_any(&self.looks).state.rev()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use sp_settings::remote::{OpenRules, Remote};

    #[derive(Clone, Default)]
    struct Out(Arc<Mutex<Vec<u8>>>);
    impl Write for Out {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn rules() -> RunRules {
        let ctx = Ctx { w: 1920, h: 1080, scale: None, test_model: None, card_full_ms: None, card_share: 0.9,
                        vfg_why: "No NVIDIA graphics card was found.".into(), hdr10: false };
        RunRules { ctx: Arc::new(ctx), rows: Arc::new(Mutex::new(Rows::default())), menu_open: Arc::new(AtomicBool::new(false)) }
    }

    #[test]
    fn menu_open_comes_after_restart_and_only_with_these_rules() {
        let mut s = Settings::defaults();
        let out = Out::default();
        let r = rules();
        let open = r.menu_open.clone();
        let remote = Remote::new(&mut s, r, out.clone());
        remote.publish(&s);
        open.store(true, Ordering::SeqCst);
        remote.publish(&s);
        let text = String::from_utf8(out.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].ends_with("\"restart\": [], \"menu_open\": false}"), "{}", lines[0]);
        assert!(lines[1].ends_with("\"restart\": [], \"menu_open\": true}"));

        let mut s2 = Settings::defaults();
        let out2 = Out::default();
        Remote::new(&mut s2, OpenRules, out2.clone()).publish(&s2);
        let t2 = String::from_utf8(out2.0.lock().unwrap().clone()).unwrap();
        assert!(t2.trim_end().ends_with("\"restart\": []}") && !t2.contains("menu_open"));
    }

    #[test]
    fn restart_reasons_follow_the_session_config() {
        let r = rules();
        let mut s = Settings::defaults();
        r.rows.lock().unwrap().started = Some(r.ctx.session_config(&s));
        assert!(r.restart_reasons(&s).is_empty());
        s.set("motion", "half").unwrap();
        assert_eq!(r.restart_reasons(&s), vec!["Motion detail".to_string()]);
        s.set("motion", "auto").unwrap();
        s.set("hz", 120i64).unwrap(); // Auto at 120 fps is half
        assert_eq!(r.restart_reasons(&s), vec!["Motion detail".to_string()]);
        assert_eq!(r.disabled(&s, "vfg_quality").as_deref(), Some("No NVIDIA graphics card was found."));
        assert!(!r.allowed(&s, "scaler", &Value::from("vsr")));
    }
}
