//! The looks library (the built-in looks and the user's, in a folder of .json files, ids "my:<file name>"), their
//! .cube LUTs (its "luts" folder), the preview's sample pictures, and LookState, which look the video shows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use super::{DEFAULT_LOOK, Look, LookError, builtin_look, lut_name, normalize};
use crate::json::{self, Map, Value};
use crate::py;
use crate::settings::{LookChoices, Settings, WatchId};

/// The user's looks folder: $SMOOVPLAY_LOOKS_DIR, else $APPDATA/framegen/looks, else ~/framegen/looks. (The Python
/// doesn't use data_dir() here, so on a Mac the looks are in ~/framegen/looks, not next to the settings file.)
pub fn looks_dir() -> PathBuf {
    if let Some(env) = std::env::var_os("SMOOVPLAY_LOOKS_DIR").filter(|e| !e.is_empty()) {
        return clean_path(PathBuf::from(env));
    }
    // Beside the settings, as framegen/looks.py looks_dir: %APPDATA%\framegen\looks on Windows,
    // ~/Library/Application Support/framegen/looks on a Mac.
    clean_path(crate::paths::data_dir().join("looks"))
}

/// A path as pathlib writes it: no "." parts, no doubled or trailing slashes.
pub(crate) fn clean_path(p: PathBuf) -> PathBuf {
    use std::path::Component;
    let parts: Vec<Component> = p.components().filter(|c| !matches!(c, Component::CurDir)).collect();
    if parts.is_empty() {
        return PathBuf::from(if p.as_os_str().is_empty() { "" } else { "." });
    }
    parts.iter().collect()
}

/// PurePath.suffix / stem, as Python 3.12 splits a name.
fn suffix_at(name: &str) -> Option<usize> {
    let i = name.rfind('.')?;
    (i > 0 && i < name.len() - 1).then_some(i)
}

pub(crate) fn py_stem(name: &str) -> &str {
    suffix_at(name).map_or(name, |i| &name[..i])
}

fn py_suffix(name: &str) -> &str {
    suffix_at(name).map_or("", |i| &name[i..])
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The names in `dir` that end in `ext` (case-sensitive, as pathlib's glob is on a Mac), hidden ones and folders
/// included; sorted.
fn glob_names(dir: &Path, ext: &str) -> std::io::Result<Vec<String>> {
    let mut out: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| n.ends_with(ext))
        .collect();
    out.sort();
    Ok(out)
}

/// A 3D LUT: table is size^3 RGB triples, indexed [b][g][r] (r fastest), as the .cube file lists them.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut {
    pub size: usize,
    pub table: Vec<f32>,
    pub dmin: Vec<f64>,
    pub dmax: Vec<f64>,
    pub title: String,
}

impl Lut {
    /// The color at (r, g, b) table units.
    pub fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        let i = ((b * self.size + g) * self.size + r) * 3;
        [self.table[i], self.table[i + 1], self.table[i + 2]]
    }
}

/// str.splitlines(), on text already read with universal newlines.
fn split_lines(s: &str) -> Vec<&str> {
    let mut out: Vec<&str> =
        s.split(['\n', '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u{85}', '\u{2028}', '\u{2029}']).collect();
    if out.last() == Some(&"") {
        out.pop();
    }
    out
}

/// A 3D .cube LUT (Adobe/Resolve format). Err: a message for the user (the Python's ValueError text).
pub fn load_cube(path: &Path) -> Result<Lut, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("couldn't read it ({})", py::strerror(&e)))?;
    let text = py::universal_newlines(&String::from_utf8_lossy(&bytes));
    let (mut size, mut dmin, mut dmax, mut title) = (0i128, vec![0.0; 3], vec![1.0; 3], String::new());
    let mut rows: Vec<Vec<String>> = Vec::new();
    let float = |t: &str| {
        py::float_from_str(t).ok_or_else(|| format!("could not convert string to float: {}", py::str_repr(t)))
    };
    for line in split_lines(&text) {
        let line = py::strip(line.split('#').next().unwrap_or(""));
        if line.is_empty() {
            continue;
        }
        let words = py::split_ws(line);
        let word = words[0].to_uppercase();
        match word.as_str() {
            "TITLE" => {
                title = if line.contains(' ') {
                    py::split_ws_once(line)
                        .and_then(|(_, rest)| rest)
                        .map(|r| py::strip(r).trim_matches('"').to_string())
                        .unwrap_or_default()
                } else {
                    String::new()
                };
            }
            "LUT_3D_SIZE" => {
                let t = words.get(1).ok_or("list index out of range")?;
                size = py::int_from_str(t)
                    .ok_or_else(|| format!("invalid literal for int() with base 10: {}", py::str_repr(t)))?;
            }
            "LUT_1D_SIZE" => return Err("it's a 1D LUT; SmoovPlay uses 3D LUTs (LUT_3D_SIZE)".into()),
            "DOMAIN_MIN" => dmin = words.iter().skip(1).take(3).map(|t| float(t)).collect::<Result<_, _>>()?,
            "DOMAIN_MAX" => dmax = words.iter().skip(1).take(3).map(|t| float(t)).collect::<Result<_, _>>()?,
            _ => {
                let c = word.chars().next().unwrap_or(' ');
                if py::is_digit(c) || "-+.".contains(c) {
                    rows.push(words.iter().take(3).map(|s| s.to_string()).collect());
                }
            }
        }
    }
    if !(2..=129).contains(&size) {
        return Err("it has no LUT_3D_SIZE between 2 and 129".into());
    }
    let size = size as usize;
    let want = size * size * size;
    if rows.len() != want {
        return Err(format!("it should have {want} rows of colors for size {size}, but has {}", rows.len()));
    }
    let mut table = Vec::with_capacity(want * 3);
    for r in &rows {
        if r.len() != 3 {
            return Err("some of its rows aren't three numbers".into());
        }
        for t in r {
            let v = py::float_from_str(t).ok_or("some of its rows aren't three numbers")?;
            table.push(v as f32);
        }
    }
    if !table.iter().all(|v| v.is_finite()) {
        return Err("it has colors that aren't numbers".into());
    }
    Ok(Lut { size, table, dmin, dmax, title })
}

/// Why a library change failed: the look (ValueError) or the disk (OSError).
#[derive(Debug, thiserror::Error)]
pub enum LibraryError {
    #[error("{0}")]
    Look(#[from] LookError),
    /// A message for the user (ValueError text): a LUT SmoovPlay can't use.
    #[error("{0}")]
    Value(String),
    /// An OSError, and the path(s) it was about, as Python's message shows them ("'/a'" or "'/a' -> '/b'").
    #[error("{}", py::strerror(.0))]
    Io(std::io::Error, String),
}

impl LibraryError {
    /// str(e), as the Python's exception says it: "[Errno 1] Operation not permitted: '/x'" for an OSError. (The
    /// command line, like the Python's, says only the strerror part: Display.)
    pub fn py_message(&self) -> String {
        match self {
            LibraryError::Io(e, at) => match e.raw_os_error() {
                Some(n) if !at.is_empty() => format!("[Errno {n}] {}: {at}", py::strerror(e)),
                _ => py::strerror(e),
            },
            other => other.to_string(),
        }
    }
}

fn io_at(path: &Path) -> impl FnOnce(std::io::Error) -> LibraryError + '_ {
    move |e| LibraryError::Io(e, py::str_repr(&path.to_string_lossy()))
}

fn io_between<'a>(a: &'a Path, b: &'a Path) -> impl FnOnce(std::io::Error) -> LibraryError + 'a {
    move |e| {
        LibraryError::Io(e, format!("{} -> {}", py::str_repr(&a.to_string_lossy()), py::str_repr(&b.to_string_lossy())))
    }
}

type LutCache = HashMap<String, (Option<SystemTime>, Option<Arc<Lut>>)>;

/// The built-in looks and the user's (a folder of .json files). Ids: the built-in keys, and "my:<file name without
/// .json>" for the user's.
pub struct Library {
    pub folder: PathBuf,
    /// What the last reload() said about files it skipped ("[looks] skipped x.json: why"), also printed.
    pub skipped: Vec<String>,
    user: Vec<(String, Look)>,
    choices: LookChoices,
    luts: Mutex<LutCache>,
}

impl Library {
    /// The library in `folder` (read at once).
    pub fn new(folder: impl Into<PathBuf>) -> Self {
        let mut lib = Library {
            folder: clean_path(folder.into()),
            skipped: Vec::new(),
            user: Vec::new(),
            choices: LookChoices::builtin(),
            luts: Mutex::new(HashMap::new()),
        };
        lib.reload();
        lib
    }

    /// The library in looks_dir().
    pub fn open_default() -> Self {
        Self::new(looks_dir())
    }

    pub fn lut_folder(&self) -> PathBuf {
        self.folder.join("luts")
    }

    /// Reads the user's looks again (the launcher saved or deleted one). A file that can't be read is skipped, and
    /// said so on stdout, as the Python does. Settings::set_look_choices(lib.choices()) then updates the setting.
    pub fn reload(&mut self) {
        let mut user: Vec<(String, Look)> = Vec::new();
        self.skipped.clear();
        for name in glob_names(&self.folder, ".json").unwrap_or_default() {
            let path = self.folder.join(&name);
            let got = match py::read_text(&path) {
                Err(py::ReadError::Os(e)) => Err(py::oserror_text(&e, &path)),
                Err(py::ReadError::Decode(t)) => Err(t),
                Ok(text) => match json::loads(&text) {
                    Err(e) => Err(e.to_string()),
                    Ok(v) => normalize(&v).map_err(|e| e.to_string()),
                },
            };
            match got {
                Ok(look) => {
                    let id = format!("my:{}", py_stem(&name));
                    match user.iter_mut().find(|(i, _)| *i == id) {
                        Some(slot) => slot.1 = look,
                        None => user.push((id, look)),
                    }
                }
                Err(e) => {
                    let line = format!("[looks] skipped {name}: {e}");
                    println!("{line}");
                    self.skipped.push(line);
                }
            }
        }
        let mut order: Vec<usize> = (0..user.len()).collect();
        order.sort_by(|&a, &b| {
            (user[a].1.name.to_lowercase(), &user[a].0).cmp(&(user[b].1.name.to_lowercase(), &user[b].0))
        });
        let mut choices = LookChoices::builtin();
        for i in order {
            choices.ids.push(user[i].0.clone());
            choices.names.push(user[i].1.name.clone());
        }
        self.user = user;
        self.choices = choices;
    }

    /// The "Which look" setting's choices and names: built-in looks, then the user's by name.
    pub fn choices(&self) -> LookChoices {
        self.choices.clone()
    }

    pub fn ids(&self) -> Vec<String> {
        self.choices.ids.clone()
    }

    /// The user's looks, in file order.
    pub fn user(&self) -> &[(String, Look)] {
        &self.user
    }

    pub fn user_look(&self, look_id: &str) -> Option<&Look> {
        self.user.iter().find(|(i, _)| i == look_id).map(|(_, l)| l)
    }

    pub fn get(&self, look_id: &str) -> Option<&Look> {
        builtin_look(look_id).or_else(|| self.user_look(look_id))
    }

    /// A user look's file; None for a built-in id or one with folders in it.
    pub fn path(&self, look_id: &str) -> Option<PathBuf> {
        let rest = look_id.strip_prefix("my:")?;
        if look_id.contains('/') || look_id.contains('\\') {
            return None;
        }
        Some(self.folder.join(format!("{rest}.json")))
    }

    /// Writes a user look (a new file unless look_id is one of the user's); returns its id.
    pub fn save(&mut self, look: &Value, look_id: Option<&str>) -> Result<String, LibraryError> {
        let look = normalize(look)?;
        let mut path = look_id.filter(|i| !i.is_empty()).and_then(|i| self.path(i));
        if path.is_none() {
            let base = slug(&look.name);
            let (mut stem, mut i) = (base.clone(), 1);
            while self.folder.join(format!("{stem}.json")).exists() {
                i += 1;
                stem = format!("{base}-{i}");
            }
            path = Some(self.folder.join(format!("{stem}.json")));
        }
        let path = path.expect("a path");
        std::fs::create_dir_all(&self.folder).map_err(io_at(&self.folder))?;
        let name = file_name(&path);
        let old = py_suffix(&name);
        let tmp_name = format!("{}.json.part", &name[..name.len() - old.len()]);
        let tmp = path.with_file_name(tmp_name);
        let mut data = Map::new();
        data.insert("version", 1);
        if let Value::Dict(m) = look.to_value() {
            for (k, v) in m.iter() {
                data.insert(k, v.clone());
            }
        }
        std::fs::write(&tmp, json::dumps_indent(&Value::Dict(data), 1)).map_err(io_at(&tmp))?;
        std::fs::rename(&tmp, &path).map_err(io_between(&tmp, &path))?;
        self.reload();
        Ok(format!("my:{}", py_stem(&name)))
    }

    /// Deletes a user look's file; false if there's none.
    pub fn delete(&mut self, look_id: &str) -> Result<bool, LibraryError> {
        let Some(path) = self.path(look_id) else { return Ok(false) };
        if !path.exists() {
            return Ok(false);
        }
        std::fs::remove_file(&path).map_err(io_at(&path))?;
        self.reload();
        Ok(true)
    }

    /// The .cube files in the luts folder, sorted.
    pub fn luts(&self) -> Vec<String> {
        glob_names(&self.lut_folder(), ".cube").unwrap_or_default()
    }

    /// A LUT from the luts folder, read once (and again if the file changes); None if it's missing or broken.
    pub fn lut(&self, name: &str) -> Option<Arc<Lut>> {
        let clean = lut_name(&Value::Str(name.to_string()));
        if clean.is_empty() {
            return None;
        }
        let path = self.lut_folder().join(&clean);
        let meta = std::fs::metadata(&path).ok()?;
        let mtime = meta.modified().ok();
        let mut cache = self.luts.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((t, lut)) = cache.get(name)
            && *t == mtime
        {
            return lut.clone();
        }
        let lut = match load_cube(&path) {
            Ok(l) => Some(Arc::new(l)),
            Err(e) => {
                println!("[looks] LUT {name}: {e}");
                None
            }
        };
        cache.insert(name.to_string(), (mtime, lut.clone()));
        lut
    }

    /// Copies a .cube file into the luts folder (after checking it); returns its name.
    pub fn add_lut(&mut self, src: &Path) -> Result<String, LibraryError> {
        let src_name = file_name(src);
        if py_suffix(&src_name).to_lowercase() != ".cube" {
            return Err(LibraryError::Value("SmoovPlay reads LUTs in the .cube format".into()));
        }
        load_cube(src).map_err(LibraryError::Value)?;
        let dir = self.lut_folder();
        std::fs::create_dir_all(&dir).map_err(io_at(&dir))?;
        let name = lut_name(&Value::Str(src_name));
        let dst = dir.join(&name);
        let resolve = |p: &Path| -> PathBuf {
            std::fs::canonicalize(p).unwrap_or_else(|_| match (p.parent(), p.file_name()) {
                (Some(d), Some(n)) => std::fs::canonicalize(d).map(|d| d.join(n)).unwrap_or_else(|_| p.to_path_buf()),
                _ => p.to_path_buf(),
            })
        };
        if resolve(src) != resolve(&dst) {
            let bytes = std::fs::read(src).map_err(io_at(src))?;
            std::fs::write(&dst, bytes).map_err(io_at(&dst))?;
        }
        self.luts.lock().unwrap_or_else(|e| e.into_inner()).remove(&name);
        Ok(name)
    }
}

/// A file name from a look's name: lower case, letters and digits, "-" between (at most 40 characters).
pub fn slug(name: &str) -> String {
    let lower = name.to_lowercase();
    let mut s = String::new();
    let mut dash = false;
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            s.push(c);
            dash = false;
        } else if !dash {
            s.push('-');
            dash = true;
        }
    }
    let s: String = s.trim_matches('-').chars().take(40).collect();
    if s.is_empty() { "look".into() } else { s }
}

/// The preview's sample pictures (framegen/looks_preview.py samples()): id, game, scene, credit and path, from
/// `folder`/samples.json. Missing files are left out.
pub fn samples(folder: &Path) -> Vec<Value> {
    let Ok(text) = py::read_text(&folder.join("samples.json")) else { return Vec::new() };
    let Some(listed) = json::loads(&text).ok().and_then(|v| v.get("samples").cloned()) else { return Vec::new() };
    let Value::List(listed) = listed else { return Vec::new() };
    let mut out = Vec::new();
    for s in &listed {
        let (Some(file), Some(id), Some(game), Some(scene), Some(credit)) =
            (s.get("file").and_then(Value::as_str), s.get("id"), s.get("game"), s.get("scene"), s.get("credit"))
        else {
            continue;
        };
        let path = clean_path(folder.join(file));
        if path.is_file() {
            out.push(json::obj([
                ("id", id.clone()),
                ("game", game.clone()),
                ("scene", scene.clone()),
                ("credit", credit.clone()),
                ("path", path.to_string_lossy().as_ref().into()),
            ]));
        }
    }
    out
}

/// Which look the video shows: the "Look" settings (look_on, look), or, while the launcher's Looks page edits one,
/// that one (the preview). rev() changes whenever the answer may have.
pub struct LookState {
    pub preview: Option<Look>,
    rev: Arc<AtomicU64>,
    watch: WatchId,
}

impl LookState {
    pub fn new(settings: &mut Settings) -> Self {
        let rev = Arc::new(AtomicU64::new(0));
        let r2 = Arc::clone(&rev);
        let watch = settings.watch(move |_, key| {
            if key == "look" || key == "look_on" {
                r2.fetch_add(1, Ordering::SeqCst);
            }
        });
        LookState { preview: None, rev, watch }
    }

    pub fn rev(&self) -> u64 {
        self.rev.load(Ordering::SeqCst)
    }

    fn bump(&self) {
        self.rev.fetch_add(1, Ordering::SeqCst);
    }

    /// The look to draw, or None.
    pub fn current<'a>(&'a self, settings: &Settings, library: &'a Library) -> Option<&'a Look> {
        if let Some(p) = &self.preview {
            return Some(p);
        }
        if !settings.get_bool("look_on") {
            return None;
        }
        library.get(settings.get_str("look")).or_else(|| builtin_look(DEFAULT_LOOK))
    }

    pub fn name(&self, settings: &Settings, library: &Library) -> String {
        self.current(settings, library).map(|l| l.name.clone()).unwrap_or_default()
    }

    /// The launcher's live preview (None: back to the settings' look).
    pub fn show(&mut self, look: Option<&Value>) -> Result<(), LookError> {
        self.preview = match look {
            None => None,
            Some(v) => Some(normalize(v)?),
        };
        self.bump();
        Ok(())
    }

    /// The user's looks changed on disk: read them again, and back to the default look if the one in use is gone.
    pub fn reload(&mut self, settings: &mut Settings, library: &mut Library) {
        library.reload();
        settings.set_look_choices(library.choices());
        if !library.choices().contains(settings.get_str("look"))
            || settings.get("look").and_then(Value::as_str).is_none()
        {
            let _ = settings.set("look", DEFAULT_LOOK);
        }
        self.bump();
    }

    /// K: the look on or off.
    pub fn toggle(&mut self, settings: &mut Settings) {
        self.preview = None;
        let on = settings.get_bool("look_on");
        let _ = settings.set("look_on", !on);
    }

    /// J: the next look (and on).
    pub fn next(&mut self, settings: &mut Settings, delta: i64) {
        self.preview = None;
        if !settings.get_bool("look_on") {
            let _ = settings.set("look_on", true);
            return;
        }
        let _ = settings.step("look", delta, None);
    }

    pub fn close(self, settings: &mut Settings) {
        settings.unwatch(self.watch);
    }
}
