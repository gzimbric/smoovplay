//! User settings (framegen/settings.py): one registry, [`SPECS`], that the in-video menu, the launcher's Settings
//! page, the command line and the saved file all share, and [`Settings`], the current values, saved to the same JSON
//! file the Python writes, in the same layout.
//!
//! Values are kept as the JSON they came as ([`Value`]): the Python keeps a file's `"hz": 90.0` as a float, and
//! writes it back and reports it that way, so this does too. A value is accepted when it equals one of the
//! setting's choices by Python's == (so `1` is a choice of On/Off, and `0` of FSR sharpening's `0.0`).
//!
//! `live: false` settings are saved at once but take effect at the next in-app restart. `saved: false` settings
//! (test modes) are never written to the file.

use std::path::{Path, PathBuf};

use crate::generated;
use crate::json::{self, Map, Value};
use crate::paths;

/// A value in the registry: a choice, a default, a key of names or hints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum C {
    B(bool),
    I(i64),
    F(f64),
    S(&'static str),
}

impl C {
    pub fn value(self) -> Value {
        match self {
            C::B(b) => Value::Bool(b),
            C::I(n) => Value::Int(n),
            C::F(f) => Value::Float(f),
            C::S(s) => Value::Str(s.to_string()),
        }
    }
    fn eq_value(self, v: &Value) -> bool {
        self.value().py_eq(v)
    }
}

/// One setting. Mirrors framegen/settings.py's Spec field for field.
#[derive(Debug)]
pub struct Spec {
    pub key: &'static str,
    pub label: &'static str,
    pub(crate) choices: &'static [C],
    /// The Python keeps this setting's choices in a list, not a tuple (only "look"): it shows in error messages.
    pub(crate) choices_are_list: bool,
    pub(crate) default: C,
    pub(crate) names: &'static [(C, &'static str)],
    /// Applies immediately; false: on the next restart.
    pub live: bool,
    pub help: &'static str,
    /// False: this session only, never written to the settings file (test modes).
    pub saved: bool,
    /// The section on the launcher's Settings page.
    pub group: &'static str,
    /// One of the few big choices at the top of the launcher's Settings page.
    pub quick: bool,
    /// (key, values): only used, and only shown, while setting `key` is one of `values`.
    pub(crate) applies: Option<(&'static str, &'static [C])>,
    /// A note at the end of the launcher's dropdown, about choices that aren't offered yet.
    pub later: &'static str,
    pub(crate) hints: &'static [(C, &'static str)],
    /// "picture" and/or "speed": a rough line on what it does, in this order.
    pub impact: &'static [(&'static str, &'static str)],
    pub experimental: bool,
    /// Only used with these sources (framegen/sources.py kinds); empty: all.
    pub sources: &'static [&'static str],
    /// Source kind -> why it isn't used there ("": any other source).
    pub elsewhere: &'static [(&'static str, &'static str)],
    /// The "Which look" setting: its choices and names are the looks library's ([`LookChoices`]).
    pub(crate) looks: bool,
}

/// The "Which look" setting's choices and names: the built-in looks, then the user's (looks::Library::choices()).
/// The Python keeps them in module globals that the library updates in place; here each Settings holds its own.
#[derive(Clone, Debug, PartialEq)]
pub struct LookChoices {
    pub ids: Vec<String>,
    pub names: Vec<String>,
}

impl LookChoices {
    /// The built-in looks only.
    pub fn builtin() -> Self {
        LookChoices {
            ids: generated::BUILTIN.iter().map(|(id, _, _)| id.to_string()).collect(),
            names: generated::BUILTIN.iter().map(|(_, name, _)| name.to_string()).collect(),
        }
    }
    pub fn contains(&self, id: &str) -> bool {
        self.ids.iter().any(|i| i == id)
    }
    pub fn name(&self, id: &str) -> Option<&str> {
        self.ids.iter().position(|i| i == id).map(|i| self.names[i].as_str())
    }
}

impl Default for LookChoices {
    fn default() -> Self {
        Self::builtin()
    }
}

impl Spec {
    /// The choices in display order, as JSON values of the spec's own types.
    pub fn choices(&self, looks: &LookChoices) -> Vec<Value> {
        if self.looks {
            looks.ids.iter().map(|s| Value::Str(s.clone())).collect()
        } else {
            self.choices.iter().map(|c| c.value()).collect()
        }
    }

    pub fn default_value(&self) -> Value {
        self.default.value()
    }

    /// Whether `v` is one of the choices (Python's `v in spec.choices`).
    pub fn is_choice(&self, v: &Value, looks: &LookChoices) -> bool {
        if self.looks {
            return v.as_str().is_some_and(|s| looks.contains(s));
        }
        self.choices.iter().any(|c| c.eq_value(v))
    }

    /// The choice equal to `v`, in the spec's own type (JSON 1 -> 1.0 for FSR sharpening), if there is one.
    pub fn canonical(&self, v: &Value, looks: &LookChoices) -> Option<Value> {
        self.choices(looks).into_iter().find(|c| c.py_eq(v))
    }

    /// The text shown for a value; values without a name are shown as str(value).
    pub fn name(&self, v: &Value, looks: &LookChoices) -> String {
        if self.looks {
            if let Some(n) = v.as_str().and_then(|s| looks.name(s)) {
                return n.to_string();
            }
        } else if let Some((_, n)) = self.names.iter().find(|(c, _)| c.eq_value(v)) {
            return n.to_string();
        }
        v.py_str()
    }

    /// The line under a choice in the launcher's dropdown ("" for none).
    pub fn hint(&self, v: &Value) -> &'static str {
        self.hints.iter().find(|(c, _)| c.eq_value(v)).map_or("", |(_, h)| h)
    }

    /// (key, values) this setting depends on, if any.
    pub fn applies(&self) -> Option<(&'static str, Vec<Value>)> {
        self.applies.map(|(k, vs)| (k, vs.iter().map(|c| c.value()).collect()))
    }

    /// Whether this setting is used with these values (e.g. VFG quality only while NVIDIA VFG is picked).
    pub fn applies_to(&self, values: &Map) -> bool {
        match self.applies {
            None => true,
            Some((k, vs)) => values.get(k).is_some_and(|v| vs.iter().any(|c| c.eq_value(v))),
        }
    }

    /// Why this setting isn't used with frames from `source` (a framegen/sources.py kind), or None if it is.
    pub fn unused_with(&self, source: &str) -> Option<String> {
        if self.sources.is_empty() || self.sources.contains(&source) {
            return None;
        }
        let get = |k: &str| self.elsewhere.iter().find(|(s, _)| *s == k).map(|(_, t)| *t).filter(|t| !t.is_empty());
        Some(get(source).or_else(|| get("")).unwrap_or("Not used with this source.").to_string())
    }

    /// repr() of the choices, as the Python's error message shows them: a tuple, or for "look" a list.
    fn choices_repr(&self, looks: &LookChoices) -> String {
        let items: Vec<String> = self.choices(looks).iter().map(Value::py_repr).collect();
        if self.choices_are_list || self.looks {
            format!("[{}]", items.join(", "))
        } else if items.len() == 1 {
            format!("({},)", items[0])
        } else {
            format!("({})", items.join(", "))
        }
    }
}

/// Every setting, in the order the menus and the file list them.
pub fn specs() -> &'static [Spec] {
    &generated::SPECS
}

/// BY_KEY[key].
pub fn spec(key: &str) -> Option<&'static Spec> {
    generated::SPECS.iter().find(|s| s.key == key)
}

fn index(key: &str) -> Option<usize> {
    generated::SPECS.iter().position(|s| s.key == key)
}

/// Capture card resolution: name -> (width, height).
pub fn capture_size(name: &str) -> Option<(u32, u32)> {
    generated::CAPTURE_SIZES.iter().find(|(k, _)| *k == name).map(|(_, wh)| *wh)
}

/// Where the numbers in `impact` and the hints were measured.
pub const IMPACT_PC: &str = generated::IMPACT_PC;

/// Where the settings file goes: data_dir()/settings.json.
pub fn default_path() -> PathBuf {
    paths::data_dir().join("settings.json")
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SettingsError {
    /// KeyError in the Python.
    #[error("{0:?} is not a setting")]
    UnknownKey(String),
    /// ValueError in the Python, with its text: "hz: 75 is not one of (60, 90, 120)".
    #[error("{0}")]
    NotAChoice(String),
}

/// allowed(key, value) for [`Settings::step`]: whether a choice can be picked right now.
pub type Allowed<'a> = &'a dyn Fn(&str, &Value) -> bool;

/// Called with the settings and the key after any value changes.
pub type Listener = Box<dyn FnMut(&Settings, &str) + Send>;

/// Stops a listener: [`Settings::unwatch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WatchId(u64);

/// Current values, saved to JSON on every change. Values given with [`Settings::override_value`] apply to this run
/// only and aren't saved unless changed with [`Settings::set`].
pub struct Settings {
    pub path: Option<PathBuf>,
    v: Vec<Value>,
    saved: Vec<Value>,
    looks: LookChoices,
    listeners: Vec<(u64, Listener)>,
    next_id: u64,
}

impl std::fmt::Debug for Settings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Settings").field("path", &self.path).field("values", &self.values()).finish()
    }
}

impl Settings {
    /// Settings(values, path): known, saved settings whose value is one of their choices are taken; the rest are
    /// defaults.
    pub fn new(values: &Map, path: Option<PathBuf>, looks: LookChoices) -> Self {
        let specs = specs();
        let mut v: Vec<Value> = specs.iter().map(Spec::default_value).collect();
        for (k, val) in values.iter() {
            if let Some(i) = index(k) {
                let s = &specs[i];
                if s.saved && s.is_choice(val, &looks) {
                    v[i] = val.clone();
                }
            }
        }
        Settings { path, saved: v.clone(), v, looks, listeners: Vec::new(), next_id: 0 }
    }

    /// The defaults, not saved anywhere.
    pub fn defaults() -> Self {
        Self::new(&Map::new(), None, LookChoices::builtin())
    }

    /// Settings.load(path): a missing, unreadable or broken file (or one that isn't a JSON object) gives the
    /// defaults.
    pub fn load(path: impl AsRef<Path>, looks: LookChoices) -> Self {
        let path = path.as_ref();
        let values = match crate::py::read_text(path) {
            Ok(text) => match json::loads(&text) {
                Ok(Value::Dict(m)) => m,
                _ => Map::new(),
            },
            Err(_) => Map::new(),
        };
        Self::new(&values, Some(path.to_path_buf()), looks)
    }

    /// A setting's value (`settings.<key>` in the Python).
    pub fn get(&self, key: &str) -> Option<&Value> {
        index(key).map(|i| &self.v[i])
    }

    /// bool(value): On/Off settings (a file's 1 reads as On, as in the Python).
    pub fn get_bool(&self, key: &str) -> bool {
        self.get(key).is_some_and(Value::truthy)
    }

    /// A number setting as a float (0.0 if it isn't one).
    pub fn get_f64(&self, key: &str) -> f64 {
        self.get(key).and_then(Value::as_f64).unwrap_or(0.0)
    }

    /// A whole-number setting (hz, replay, vsr_quality); 0 if it isn't one.
    pub fn get_i64(&self, key: &str) -> i64 {
        self.get(key).and_then(Value::as_i64).unwrap_or(0)
    }

    /// A text setting ("" if it isn't one).
    pub fn get_str(&self, key: &str) -> &str {
        self.get(key).and_then(Value::as_str).unwrap_or("")
    }

    /// Every value, in SPECS order (Settings.values()).
    pub fn values(&self) -> Map {
        specs().iter().zip(&self.v).map(|(s, v)| (s.key, v.clone())).collect()
    }

    /// What the file holds (or would).
    pub fn saved_values(&self) -> Map {
        specs().iter().zip(&self.saved).map(|(s, v)| (s.key, v.clone())).collect()
    }

    pub fn look_choices(&self) -> &LookChoices {
        &self.looks
    }

    /// The looks library changed (Library::reload()): the "Which look" setting's choices are now these. The value
    /// isn't touched; LookState::reload puts it back to the default if its look is gone, as the Python does.
    pub fn set_look_choices(&mut self, looks: LookChoices) {
        self.looks = looks;
    }

    /// A setting's choices right now.
    pub fn choices(&self, key: &str) -> Vec<Value> {
        spec(key).map(|s| s.choices(&self.looks)).unwrap_or_default()
    }

    /// fn(settings, key) is called after any value changes.
    pub fn watch(&mut self, f: impl FnMut(&Settings, &str) + Send + 'static) -> WatchId {
        self.next_id += 1;
        self.listeners.push((self.next_id, Box::new(f)));
        WatchId(self.next_id)
    }

    pub fn unwatch(&mut self, id: WatchId) {
        self.listeners.retain(|(i, _)| *i != id.0);
    }

    /// This run only (command-line flags, the I key).
    pub fn override_value(&mut self, key: &str, value: impl Into<Value>) -> Result<(), SettingsError> {
        self.put(key, value.into(), false)
    }

    /// Changes a setting and saves the file.
    pub fn set(&mut self, key: &str, value: impl Into<Value>) -> Result<(), SettingsError> {
        self.put(key, value.into(), true)
    }

    fn put(&mut self, key: &str, value: Value, save: bool) -> Result<(), SettingsError> {
        let i = index(key).ok_or_else(|| SettingsError::UnknownKey(key.to_string()))?;
        let s = &specs()[i];
        if !s.is_choice(&value, &self.looks) {
            return Err(SettingsError::NotAChoice(format!(
                "{key}: {} is not one of {}",
                value.py_repr(),
                s.choices_repr(&self.looks)
            )));
        }
        let changed = !self.v[i].py_eq(&value);
        self.v[i] = value.clone();
        if save && s.saved {
            self.saved[i] = value;
            self.save_logged();
        }
        if changed {
            self.notify(key);
        }
        Ok(())
    }

    fn notify(&mut self, key: &str) {
        let mut ls = std::mem::take(&mut self.listeners);
        for (_, f) in ls.iter_mut() {
            f(self, key);
        }
        // (listeners added while these ran go after them)
        ls.append(&mut self.listeners);
        self.listeners = ls;
    }

    /// The next/previous choice (wrapping), skipping values `allowed(key, value)` rejects. Returns the value now.
    pub fn step(&mut self, key: &str, delta: i64, allowed: Option<Allowed>) -> Result<Value, SettingsError> {
        let i = index(key).ok_or_else(|| SettingsError::UnknownKey(key.to_string()))?;
        let choices = specs()[i].choices(&self.looks);
        let cur = &self.v[i];
        let Some(mut at) = choices.iter().position(|c| c.py_eq(cur)) else {
            return Err(SettingsError::NotAChoice(format!("{} is not in list", cur.py_repr())));
        };
        let n = choices.len() as i64;
        for _ in 0..n {
            at = (at as i64 + delta).rem_euclid(n) as usize;
            if allowed.is_none_or(|f| f(key, &choices[at])) {
                self.set(key, choices[at].clone())?;
                break;
            }
        }
        Ok(self.v[i].clone())
    }

    /// The file's text: json.dumps(saved, indent=2), every setting in SPECS order.
    pub fn file_text(&self) -> String {
        json::dumps_indent(&Value::Dict(self.saved_values()), 2)
    }

    /// Writes the file (nothing without a path).
    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, self.file_text())
    }

    fn save_logged(&self) {
        if let Err(e) = self.save() {
            let path = self.path.as_deref().unwrap_or(Path::new(""));
            println!("[settings] couldn't save {}: {}", path.display(), crate::py::oserror_text(&e, path));
        }
    }
}

/// The registry for the launcher's Settings page (`python -m framegen.settings`): every setting with its choices in
/// display order, and where the saved file is.
pub fn specs_json(path: &Path, looks: &LookChoices) -> Value {
    let rows: Vec<Value> = specs()
        .iter()
        .map(|s| {
            let pairs =
                |ps: &[(&str, &str)]| -> Value { Value::Dict(ps.iter().map(|(k, v)| (*k, Value::from(*v))).collect()) };
            json::obj([
                ("key", s.key.into()),
                ("label", s.label.into()),
                ("group", s.group.into()),
                ("help", s.help.into()),
                ("default", s.default_value()),
                ("live", s.live.into()),
                ("saved", s.saved.into()),
                ("later", s.later.into()),
                ("quick", s.quick.into()),
                ("impact", pairs(s.impact)),
                ("experimental", s.experimental.into()),
                ("sources", if s.sources.is_empty() { Value::Null } else { s.sources.to_vec().into() }),
                ("elsewhere", if s.elsewhere.is_empty() { Value::Null } else { pairs(s.elsewhere) }),
                (
                    "applies",
                    match s.applies() {
                        None => Value::Null,
                        Some((k, vs)) => json::obj([("key", k.into()), ("values", Value::List(vs))]),
                    },
                ),
                (
                    "choices",
                    Value::List(
                        s.choices(looks)
                            .into_iter()
                            .map(|v| {
                                let (name, hint) = (s.name(&v, looks), s.hint(&v));
                                json::obj([("value", v), ("name", name.into()), ("hint", hint.into())])
                            })
                            .collect(),
                    ),
                ),
            ])
        })
        .collect();
    json::obj([
        ("path", path.to_string_lossy().as_ref().into()),
        ("impact_pc", IMPACT_PC.into()),
        ("specs", Value::List(rows)),
    ])
}
